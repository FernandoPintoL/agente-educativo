"""
Agent Service - LLM-Based Synthesis and Reasoning
Port: 8080

Synthesizes discoveries from unsupervised and supervised ML pipelines
using LLM (Groq) to generate intelligent insights and personalized
intervention strategies.

Architecture:
- LLMSynthesizer: Groq integration with graceful fallback
- AgentOrchestrator: Orchestrates synthesis, reasoning, and strategies
- FastAPI endpoints for ML pipeline integration
"""

from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import List, Dict, Any, Optional
import logging
import os
from datetime import datetime
import json
from dotenv import load_dotenv

# Importar el nuevo agente de mejora de tareas
try:
    from .task_enhancement_agent import (
        TaskEnhancementAgent,
        ContentAnalysisRequest,
        ContentAnalysisResponse,
        DifficultyRecommendationRequest,
        DifficultyRecommendation,
        StudentSolutionAnalysisRequest,
        StudentSolutionAnalysisResponse,
        StudentSolutionFeedback,
        TitleAnalysisRequest,
        TitleAnalysisResponse
    )
except ImportError:
    # Fallback para ejecución directa
    from task_enhancement_agent import (
        TaskEnhancementAgent,
        ContentAnalysisRequest,
        ContentAnalysisResponse,
        DifficultyRecommendationRequest,
        DifficultyRecommendation,
        StudentSolutionAnalysisRequest,
        StudentSolutionAnalysisResponse,
        StudentSolutionFeedback,
        TitleAnalysisRequest,
        TitleAnalysisResponse
    )

# Importar servicios de rate limiting
try:
    from .services.rate_limiting import (
        check_analysis_quota,
        check_cooldown,
        record_analysis,
        detect_abuse_pattern,
        get_usage_stats
    )
except ImportError:
    # Fallback: funciones stub si los servicios no están disponibles
    def check_analysis_quota(*args, **kwargs): return True
    def check_cooldown(*args, **kwargs): return True
    def record_analysis(*args, **kwargs): pass
    def detect_abuse_pattern(*args, **kwargs): return False
    def get_usage_stats(*args, **kwargs): return {}

# Importar servicios de auditoría
try:
    from .services.audit_service import (
        log_analysis,
        create_abuse_alert,
        get_student_history,
        get_abuse_alerts,
        get_professor_dashboard,
        get_task_usage_stats as get_audit_task_stats
    )
except ImportError:
    # Fallback: funciones stub si los servicios no están disponibles
    def log_analysis(*args, **kwargs): pass
    def create_abuse_alert(*args, **kwargs): pass
    def get_student_history(*args, **kwargs): return []
    def get_abuse_alerts(*args, **kwargs): return []
    def get_professor_dashboard(*args, **kwargs): return {}
    def get_audit_task_stats(*args, **kwargs): return {}

# Importar ML integrator
try:
    from .services.ml_integrator import get_integrator as get_ml_integrator
except ImportError:
    # Fallback para cuando se ejecuta desde start_server
    from services.ml_integrator import get_integrator as get_ml_integrator

# Importar modelos de base de datos
try:
    from .models.student_solution_analysis import StudentSolutionAnalysis
    from .models.audit_models import AnalysisLog, AbuseAlert
except ImportError:
    # Fallback: modelos stub si no están disponibles
    class StudentSolutionAnalysis: pass
    class AnalysisLog: pass
    class AbuseAlert: pass

# Importar conexión a base de datos
try:
    from .shared.database.connection import get_db, engine as db_engine
except ImportError:
    try:
        from shared.database.connection import get_db, engine as db_engine
    except ImportError:
        # Fallback si shared no está disponible
        db_engine = None
        def get_db():
            return None

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title="Agent Synthesis Service",
    description="LLM-based synthesis of ML discoveries",
    version="1.0.0"
)

# CORS middleware - Restrict in production
cors_origins = os.getenv('CORS_ORIGINS', '*').split(',')
app.add_middleware(
    CORSMiddleware,
    allow_origins=cors_origins,
    allow_credentials=True,
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)

# ========================
# Initialize Enhancement Agent
# ========================
try:
    enhancement_agent = TaskEnhancementAgent()
    logger.info("✅ Task Enhancement Agent inicializado")
except Exception as e:
    logger.error(f"❌ Error inicializando Enhancement Agent: {str(e)}")
    enhancement_agent = None

# ========================
# Pydantic Models
# ========================

class DiscoveryData(BaseModel):
    """ML discovery data from unsupervised pipeline"""
    cluster_analysis: Optional[Dict[str, Any]] = None
    concept_topics: Optional[Dict[str, Any]] = None
    anomalies: Optional[Dict[str, Any]] = None
    correlations: Optional[Dict[str, Any]] = None


class PredictionData(BaseModel):
    """ML prediction data from supervised pipeline"""
    predictions: Optional[Dict[str, Any]] = None
    confidence_scores: Optional[List[float]] = None
    model_type: Optional[str] = None


class SynthesisRequest(BaseModel):
    """Request for synthesis endpoint"""
    student_id: int
    discoveries: Dict[str, Any] = Field(default_factory=dict)
    predictions: Dict[str, Any] = Field(default_factory=dict)
    context: str = "unified_learning_pipeline"


class ReasoningRequest(BaseModel):
    """Request for reasoning endpoint"""
    student_id: int
    discoveries: Dict[str, Any] = Field(default_factory=dict)
    predictions: Dict[str, Any] = Field(default_factory=dict)


class InterventionRequest(BaseModel):
    """Request for intervention strategy endpoint"""
    student_id: int
    discoveries: Dict[str, Any] = Field(default_factory=dict)
    predictions: Dict[str, Any] = Field(default_factory=dict)


class SynthesisResponse(BaseModel):
    """Response from synthesis endpoint"""
    success: bool
    student_id: int
    synthesis: Dict[str, Any]
    reasoning: List[str]
    confidence: float = Field(ge=0, le=1)
    timestamp: str
    method: str  # "agent_llm" or "local_synthesis"


class ReasoningResponse(BaseModel):
    """Response from reasoning endpoint"""
    success: bool
    reasoning_steps: List[str]
    key_insights: List[str]
    recommendations: List[str]
    confidence: float = Field(ge=0, le=1)
    timestamp: str


class InterventionResponse(BaseModel):
    """Response from intervention strategy endpoint"""
    success: bool
    student_id: int
    strategy: Dict[str, Any]
    actions: List[str]
    resources: List[Dict[str, Any]]
    confidence: float = Field(ge=0, le=1)
    timestamp: str


class HealthResponse(BaseModel):
    """Health check response"""
    status: str
    llm_available: bool
    service_version: str = "1.0.0"
    timestamp: str


# ========================
# LLM Synthesizer
# ========================

class LLMSynthesizer:
    """
    Handles LLM integration with Groq API.
    Provides graceful fallback to local synthesis if LLM unavailable.
    """

    def __init__(self):
        self.groq_api_key = os.getenv('GROQ_API_KEY')
        self.llm = None
        self.llm_available = False
        self._initialize_llm()

    def _initialize_llm(self):
        """Initialize LLM with Groq API"""
        try:
            if not self.groq_api_key:
                logger.warning("GROQ_API_KEY not set in environment")
                return

            from langchain_groq import ChatGroq

            # Lee modelo del .env, con fallback a llama
            model_name = os.getenv('GROQ_MODEL', 'llama-3.3-70b-versatile')
            temperature = float(os.getenv('GROQ_TEMPERATURE', 0.3))

            self.llm = ChatGroq(
                temperature=temperature,
                model_name=model_name,
                groq_api_key=self.groq_api_key
            )
            self.llm_available = True
            logger.info(f"LLM initialized successfully with Groq - Model: {model_name}")
        except ImportError:
            logger.warning("langchain_groq not installed. Install with: pip install langchain-groq")
        except Exception as e:
            logger.warning(f"Error initializing LLM: {str(e)}")

    def synthesize(
        self,
        student_id: int,
        discoveries: Dict[str, Any],
        predictions: Dict[str, Any],
        context: str = "unified_learning_pipeline"
    ) -> Dict[str, Any]:
        """
        Synthesize discoveries using LLM
        Falls back to local synthesis if LLM unavailable
        """
        if not self.llm_available:
            logger.info("LLM unavailable, using local synthesis")
            return self._local_synthesis(student_id, discoveries, predictions)

        try:
            prompt = self._build_synthesis_prompt(
                student_id, discoveries, predictions, context
            )

            response = self.llm.invoke(prompt)
            synthesis_text = response.content

            return {
                'success': True,
                'student_id': student_id,
                'synthesis': {
                    'method': 'llm_synthesis',
                    'text': synthesis_text,
                    'insights': self._extract_insights(synthesis_text),
                    'recommendations': self._extract_recommendations(synthesis_text),
                },
                'reasoning': self._extract_reasoning_steps(synthesis_text),
                'confidence': 0.85,
                'timestamp': datetime.utcnow().isoformat() + 'Z',
                'method': 'agent_llm',
            }
        except Exception as e:
            logger.warning(f"LLM synthesis failed: {str(e)}, falling back to local")
            return self._local_synthesis(student_id, discoveries, predictions)

    def explain_reasoning(
        self,
        student_id: int,
        discoveries: Dict[str, Any],
        predictions: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Generate detailed reasoning explanation
        """
        if not self.llm_available:
            return self._local_reasoning(discoveries, predictions)

        try:
            prompt = self._build_reasoning_prompt(
                student_id, discoveries, predictions
            )

            response = self.llm.invoke(prompt)
            reasoning_text = response.content

            return {
                'success': True,
                'reasoning_steps': self._extract_reasoning_steps(reasoning_text),
                'key_insights': self._extract_insights(reasoning_text),
                'recommendations': self._extract_recommendations(reasoning_text),
                'confidence': 0.8,
                'timestamp': datetime.utcnow().isoformat() + 'Z',
            }
        except Exception as e:
            logger.warning(f"LLM reasoning failed: {str(e)}")
            return self._local_reasoning(discoveries, predictions)

    def _build_synthesis_prompt(self, student_id: int, discoveries: Dict[str, Any], predictions: Dict[str, Any], context: str) -> str:
        """Build prompt for LLM synthesis"""
        prompt = f"""
Analyze this student's learning data and provide synthesis:

Student ID: {student_id}
Context: {context}

UNSUPERVISED ML DISCOVERIES:
{json.dumps(discoveries, indent=2)}

SUPERVISED ML PREDICTIONS:
{json.dumps(predictions, indent=2)}

Provide:
1. Key insights from the data
2. Main recommendations for intervention
3. Student's learning profile summary
"""
        return prompt

    def _build_reasoning_prompt(self, student_id: int, discoveries: Dict[str, Any], predictions: Dict[str, Any]) -> str:
        """Build prompt for detailed reasoning"""
        return f"""
Provide step-by-step reasoning for these discoveries:

Student ID: {student_id}
Discoveries: {json.dumps(discoveries, indent=2)}
Predictions: {json.dumps(predictions, indent=2)}

Explain how these lead to recommendations.
"""

    def _extract_insights(self, text: str) -> List[str]:
        """Extract key insights from LLM response"""
        lines = text.split('\n')
        insights = [line.strip() for line in lines if line.strip() and len(line.strip()) > 10]
        return insights[:5]

    def _extract_recommendations(self, text: str) -> List[str]:
        """Extract recommendations from LLM response"""
        lines = text.split('\n')
        recommendations = []
        for line in lines:
            if any(word in line.lower() for word in ['recommend', 'suggest', 'should']):
                if line.strip():
                    recommendations.append(line.strip())
                    if len(recommendations) >= 5:
                        break
        return recommendations or ['Review student progress regularly']

    def _extract_reasoning_steps(self, text: str) -> List[str]:
        """Extract reasoning steps from text"""
        lines = text.split('\n')
        steps = [line.strip() for line in lines if line.strip() and len(line.strip()) > 10]
        return steps[:10]

    def _local_synthesis(self, student_id: int, discoveries: Dict[str, Any], predictions: Dict[str, Any]) -> Dict[str, Any]:
        """Local synthesis fallback"""
        insights = ["Student analysis complete", "Multiple patterns identified"]
        recommendations = ["Monitor progress", "Provide targeted support"]

        if 'cluster_analysis' in discoveries:
            insights.append("Student assigned to learning segment")
            recommendations.append("Apply segment-specific strategies")

        if 'anomalies' in discoveries:
            anomalies = discoveries['anomalies'].get('data', {}).get('detected_patterns', [])
            if anomalies:
                insights.append(f"{len(anomalies)} unusual patterns detected")
                recommendations.append("Provide focused intervention")

        return {
            'success': True,
            'student_id': student_id,
            'synthesis': {
                'method': 'local_synthesis',
                'key_insights': insights,
                'recommendations': recommendations,
            },
            'reasoning': ['Analyzing clustering', 'Evaluating anomalies', 'Assessing topics', 'Synthesizing results'],
            'confidence': 0.7,
            'timestamp': datetime.utcnow().isoformat() + 'Z',
            'method': 'local_synthesis',
        }

    def _local_reasoning(self, discoveries: Dict[str, Any], predictions: Dict[str, Any]) -> Dict[str, Any]:
        """Local reasoning fallback"""
        return {
            'success': True,
            'reasoning_steps': ['Analyze cluster assignments', 'Evaluate anomalies', 'Review topic mastery', 'Validate predictions', 'Synthesize findings'],
            'key_insights': ['Patterns identified', 'Academic profile clear', 'Anomalies noted'],
            'recommendations': ['Implement interventions', 'Monitor anomalies', 'Reinforce strengths'],
            'confidence': 0.7,
            'timestamp': datetime.utcnow().isoformat() + 'Z',
        }


# ========================
# Agent Orchestrator
# ========================

class AgentOrchestrator:
    """
    Orchestrates agent synthesis, reasoning, and intervention strategy generation
    """

    def __init__(self):
        self.synthesizer = LLMSynthesizer()

    def synthesize_discoveries(
        self,
        student_id: int,
        discoveries: Dict[str, Any],
        predictions: Dict[str, Any],
        context: str = "unified_learning_pipeline"
    ) -> Dict[str, Any]:
        """Synthesize discoveries"""
        result = self.synthesizer.synthesize(
            student_id, discoveries, predictions, context
        )
        return result

    def explain_reasoning(
        self,
        student_id: int,
        discoveries: Dict[str, Any],
        predictions: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Explain reasoning process"""
        return self.synthesizer.explain_reasoning(
            student_id, discoveries, predictions
        )

    def generate_intervention_strategy(
        self,
        student_id: int,
        discoveries: Dict[str, Any],
        predictions: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Generate personalized intervention strategy"""
        synthesis = self.synthesizer.synthesize(
            student_id, discoveries, predictions
        )

        # Extract recommendations and insights
        insights = []
        if 'synthesis' in synthesis:
            insights = synthesis['synthesis'].get('insights', []) or \
                      synthesis['synthesis'].get('key_insights', [])

        strategy = {
            'type': 'personalized',
            'frequency': 'weekly',
            'focus_areas': insights[:3] if insights else ['Core concepts', 'Engagement'],
            'success_criteria': [
                '10% improvement in metrics',
                'Reduce anomalies',
                'Consistent participation',
                'Better performance',
            ],
        }

        return {
            'success': True,
            'student_id': student_id,
            'strategy': strategy,
            'actions': [
                'Review and reinforce weak areas',
                'Implement personalized interventions',
                'Monitor detected anomalies',
                'Evaluate progress regularly',
                'Adjust strategy based on results',
            ],
            'resources': [
                {'type': 'tutorial', 'priority': 'high', 'topic': 'Core concepts'},
                {'type': 'practice_problem', 'priority': 'high', 'topic': 'Skill reinforcement'},
                {'type': 'peer_group', 'priority': 'medium', 'topic': 'Collaborative learning'},
                {'type': 'mentor_session', 'priority': 'medium', 'topic': 'Personal guidance'},
                {'type': 'assessment', 'priority': 'low', 'topic': 'Progress evaluation'},
            ],
            'confidence': 0.75,
            'timestamp': datetime.utcnow().isoformat() + 'Z',
        }


# Initialize Orchestrator
orchestrator = AgentOrchestrator()


# ========================
# API Endpoints
# ========================

@app.post("/synthesize", response_model=SynthesisResponse)
async def synthesize_discoveries(request: SynthesisRequest) -> SynthesisResponse:
    """
    Synthesize ML discoveries using LLM

    Integrates discoveries from unsupervised ML (clustering, topics, anomalies)
    and supervised ML predictions to generate intelligent insights.
    """
    try:
        logger.info(f"Synthesizing discoveries for student {request.student_id}")

        result = orchestrator.synthesize_discoveries(
            student_id=request.student_id,
            discoveries=request.discoveries,
            predictions=request.predictions,
            context=request.context
        )

        return SynthesisResponse(**result)

    except Exception as e:
        logger.error(f"Error in synthesis: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/reasoning", response_model=ReasoningResponse)
async def explain_reasoning(request: ReasoningRequest) -> ReasoningResponse:
    """
    Provide detailed reasoning for synthesis

    Explains the thinking process behind recommendations
    and provides transparent decision-making steps.
    """
    try:
        logger.info(f"Explaining reasoning for student {request.student_id}")

        result = orchestrator.explain_reasoning(
            student_id=request.student_id,
            discoveries=request.discoveries,
            predictions=request.predictions
        )

        return ReasoningResponse(**result)

    except Exception as e:
        logger.error(f"Error in reasoning: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/intervention-strategy", response_model=InterventionResponse)
async def generate_intervention(request: InterventionRequest) -> InterventionResponse:
    """
    Generate personalized intervention strategy

    Creates a targeted intervention plan with specific actions,
    resource recommendations, and success criteria.
    """
    try:
        logger.info(f"Generating intervention for student {request.student_id}")

        result = orchestrator.generate_intervention_strategy(
            student_id=request.student_id,
            discoveries=request.discoveries,
            predictions=request.predictions
        )

        return InterventionResponse(**result)

    except Exception as e:
        logger.error(f"Error in intervention generation: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


# ========================
# INTEGRATED ML ANALYSIS ENDPOINT (NEW)
# ========================

class IntegratedAnalysisRequest(BaseModel):
    """Request para análisis integrado"""
    student_id: int


class IntegratedAnalysisResponse(BaseModel):
    """Response para análisis integrado"""
    success: bool
    student_id: int
    ml_data: Dict[str, Any]  # Datos de supervisada y no_supervisado
    synthesis: Dict[str, Any]  # Síntesis LLM
    intervention_strategy: Dict[str, Any]  # Estrategia de intervención
    timestamp: str


@app.post("/api/student/{student_id}/analysis", response_model=IntegratedAnalysisResponse)
async def analyze_student_integrated(student_id: int) -> IntegratedAnalysisResponse:
    """
    Análisis Integrado Completo del Estudiante

    Este es el endpoint principal que coordina todo:
    1. Obtiene predicciones de supervisada (8001)
    2. Obtiene clustering de no_supervisado (8002)
    3. Sintetiza con LLM (Groq)
    4. Genera estrategia de intervención
    5. Retorna análisis completo

    Este endpoint es la "puerta de entrada" para que Laravel obtenga
    todo el análisis de ML en una sola llamada.

    Args:
        student_id: ID del estudiante a analizar

    Returns:
        IntegratedAnalysisResponse con análisis completo
    """
    try:
        logger.info(f"[INTEGRATED] Iniciando análisis completo para estudiante {student_id}")
        logger.info(f"[INTEGRATED] " + "="*60)

        # ====================================================================
        # 1. OBTENER DATOS DE ML (Supervisada + No Supervisada)
        # ====================================================================

        logger.info(f"[INTEGRATED] PASO 1: Obteniendo datos de ML...")
        ml_integrator = get_ml_integrator()
        ml_data = ml_integrator.get_student_ml_analysis(student_id)

        if not ml_data['success']:
            logger.warning(f"[INTEGRATED] Advertencia: ML data obtenida con errores")
            logger.warning(f"[INTEGRATED] Errores: {ml_data['errors']}")

        logger.info(f"[INTEGRATED] Datos ML obtenidos exitosamente")
        logger.info(f"[INTEGRATED]   - Predicciones: {bool(ml_data['predictions'])}")
        logger.info(f"[INTEGRATED]   - Descubrimientos: {bool(ml_data['discoveries'])}")

        # ====================================================================
        # 2. SINTETIZAR CON LLM
        # ====================================================================

        logger.info(f"[INTEGRATED] PASO 2: Sintetizando con LLM...")
        synthesis = orchestrator.synthesize_discoveries(
            student_id=student_id,
            discoveries=ml_data['discoveries'],
            predictions=ml_data['predictions']
        )
        logger.info(f"[INTEGRATED] Síntesis completada")

        # ====================================================================
        # 3. GENERAR ESTRATEGIA DE INTERVENCIÓN
        # ====================================================================

        logger.info(f"[INTEGRATED] PASO 3: Generando estrategia de intervención...")
        intervention_strategy = orchestrator.generate_intervention_strategy(
            student_id=student_id,
            discoveries=ml_data['discoveries'],
            predictions=ml_data['predictions']
        )
        logger.info(f"[INTEGRATED] Estrategia generada")

        # ====================================================================
        # 4. COMPILAR RESPUESTA
        # ====================================================================

        logger.info(f"[INTEGRATED] PASO 4: Compilando respuesta final...")

        response = IntegratedAnalysisResponse(
            success=True,
            student_id=student_id,
            ml_data=ml_data,
            synthesis=synthesis,
            intervention_strategy=intervention_strategy,
            timestamp=datetime.utcnow().isoformat() + 'Z'
        )

        logger.info(f"[INTEGRATED] " + "="*60)
        logger.info(f"[INTEGRATED] Análisis completo exitoso para estudiante {student_id}")

        return response

    except Exception as e:
        logger.error(f"[INTEGRATED] Error en análisis integrado: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail=f"Error en análisis integrado: {str(e)}"
        )


# ========================
# Task Enhancement Endpoints (Fase 1)
# ========================

@app.post("/api/analysis/content-check", response_model=ContentAnalysisResponse)
async def analyze_content(request: ContentAnalysisRequest) -> ContentAnalysisResponse:
    """
    Analiza claridad y calidad pedagógica del contenido

    Utiliza LLM para evaluar:
    - Claridad y comprensibilidad
    - Problemas: ambigüedad, gramática, vaguedad
    - Nivel Bloom (cognitivo)
    - Dificultad estimada
    - Tiempo estimado
    - Prerequisitos necesarios
    - Recomendaciones de mejora

    Args:
        request: ContentAnalysisRequest con el contenido a analizar

    Returns:
        ContentAnalysisResponse con análisis detallado
    """
    if not enhancement_agent:
        raise HTTPException(
            status_code=503,
            detail="Task Enhancement Agent no está disponible"
        )

    try:
        logger.info(f"Analizando contenido tipo {request.task_type}...")
        analysis = enhancement_agent.analyze_content_clarity(request)
        logger.info(f"✅ Análisis completado - Claridad: {analysis.clarity_score:.0%}")
        return analysis

    except Exception as e:
        logger.error(f"❌ Error en análisis de contenido: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Error analizando contenido: {str(e)}"
        )


@app.post("/api/recommendation/difficulty")
async def recommend_difficulty(request: Dict[str, Any]):
    """
    Recomienda dificultad y puntuación basada en contexto y datos históricos

    Análisis basado en:
    - Promedio histórico del curso
    - Desviación estándar de notas
    - Complejidad estimada de la tarea
    - Tipo de tarea (tarea, evaluación, trabajo)

    Args:
        request: {
            "course_id": int,
            "task_context": {
                "type": str,
                "complexity": float,
                "title": str
            },
            "student_stats": {
                "avg_score": float,
                "std_dev": float,
                "count": int
            }
        }

    Returns:
        DifficultyRecommendation con:
        - Puntos recomendados
        - Rango seguro
        - Tasa de éxito esperada
        - Tiempo estimado
    """
    if not enhancement_agent:
        raise HTTPException(
            status_code=503,
            detail="Task Enhancement Agent no está disponible"
        )

    try:
        course_id = request.get('course_id')
        task_context = request.get('task_context', {})
        student_stats = request.get('student_stats')

        logger.info(f"Recomendando dificultad para curso {course_id}...")

        recommendation = enhancement_agent.recommend_difficulty(
            course_id=course_id,
            task_context=task_context,
            student_stats=student_stats
        )

        logger.info(f"✅ Recomendación completada - Puntos: {recommendation.recommended_points}")

        return {
            'success': True,
            'data': recommendation.dict()
        }

    except Exception as e:
        logger.error(f"❌ Error en recomendación de dificultad: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Error en recomendación: {str(e)}"
        )


# ========================
# TITLE ANALYSIS ENDPOINTS (for Content Assistant)
# ========================

@app.post("/api/analysis/task-title", response_model=TitleAnalysisResponse)
async def analyze_task_title(request: TitleAnalysisRequest) -> TitleAnalysisResponse:
    """
    Analiza un título de tarea y genera sugerencias completas

    Genera:
    - Descripción pedagógica
    - Plantilla de instrucciones
    - Tiempo estimado
    - Puntuación sugerida
    - Nivel Bloom estimado
    - Observaciones pedagógicas
    - Conceptos clave

    Args:
        request: TitleAnalysisRequest con título y contexto

    Returns:
        TitleAnalysisResponse con análisis y sugerencias
    """
    if not enhancement_agent:
        raise HTTPException(
            status_code=503,
            detail="Task Enhancement Agent no está disponible"
        )

    try:
        logger.info(f"Analizando título de tarea: {request.titulo[:50]}...")
        analysis = enhancement_agent.analyze_task_title(request)
        logger.info(f"✅ Análisis de tarea completado - Confianza: {analysis.confidence:.0%}")
        return analysis

    except Exception as e:
        logger.error(f"❌ Error en análisis de título de tarea: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Error analizando título de tarea: {str(e)}"
        )


@app.post("/api/analysis/evaluation-title", response_model=TitleAnalysisResponse)
async def analyze_evaluation_title(request: TitleAnalysisRequest) -> TitleAnalysisResponse:
    """
    Analiza un título de evaluación y genera sugerencias completas

    Genera:
    - Descripción pedagógica
    - Plantilla de instrucciones
    - Tiempo estimado
    - Puntuación sugerida
    - Nivel Bloom estimado
    - Observaciones pedagógicas
    - Conceptos clave

    Args:
        request: TitleAnalysisRequest con título y contexto

    Returns:
        TitleAnalysisResponse con análisis y sugerencias
    """
    if not enhancement_agent:
        raise HTTPException(
            status_code=503,
            detail="Task Enhancement Agent no está disponible"
        )

    try:
        logger.info(f"Analizando título de evaluación: {request.titulo[:50]}...")
        analysis = enhancement_agent.analyze_evaluation_title(request)
        logger.info(f"✅ Análisis de evaluación completado - Confianza: {analysis.confidence:.0%}")
        return analysis

    except Exception as e:
        logger.error(f"❌ Error en análisis de título de evaluación: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Error analizando título de evaluación: {str(e)}"
        )


@app.post("/api/analysis/resource-title", response_model=TitleAnalysisResponse)
async def analyze_resource_title(request: TitleAnalysisRequest) -> TitleAnalysisResponse:
    """
    Analiza un título de recurso y genera sugerencias completas

    Genera:
    - Descripción pedagógica
    - Plantilla de instrucciones
    - Tiempo estimado
    - Puntuación sugerida
    - Nivel Bloom estimado
    - Observaciones pedagógicas
    - Conceptos clave

    Args:
        request: TitleAnalysisRequest con título y contexto

    Returns:
        TitleAnalysisResponse con análisis y sugerencias
    """
    if not enhancement_agent:
        raise HTTPException(
            status_code=503,
            detail="Task Enhancement Agent no está disponible"
        )

    try:
        logger.info(f"Analizando título de recurso: {request.titulo[:50]}...")
        analysis = enhancement_agent.analyze_resource_title(request)
        logger.info(f"✅ Análisis de recurso completado - Confianza: {analysis.confidence:.0%}")
        return analysis

    except Exception as e:
        logger.error(f"❌ Error en análisis de título de recurso: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Error analizando título de recurso: {str(e)}"
        )


@app.post("/api/analysis/student-solution")
async def analyze_student_solution(request: Dict[str, Any], db = None):
    """
    Analiza la solución de un estudiante usando método socrático.

    SEGURIDAD (4 capas):
    1. Frontend: Validación local + UI rate limiting
    2. Endpoint: Validación de entrada + task type
    3. Rate Limiting: Quota (5 max) + Cooldown (5 min)
    4. Abuse Detection: Detección de patrones sospechosos

    Args:
        request: {
            "task_id": int,
            "student_id": int,
            "solution_code": str,
            "task_type": "tarea" | "evaluacion" | "examen",
            "language": "python" | "js" | "java" | "auto"
        }

    Returns:
        {
            "success": true,
            "data": {
                "id": "uuid",
                "analysis_count": 1,
                "max_analyses": 5,
                "feedback": {...},
                "metadata": {...}
            }
        }
    """
    if not enhancement_agent:
        raise HTTPException(
            status_code=503,
            detail="Task Enhancement Agent no está disponible"
        )

    # Obtener conexión a BD si no se pasó
    if db is None:
        from shared.database.connection import get_db_session
        db = get_db_session()
        should_close_db = True
    else:
        should_close_db = False

    try:
        # ====================================================================
        # CAPA 1: VALIDACIONES BÁSICAS
        # ====================================================================

        # Extraer datos del request
        task_id = request.get('task_id')
        student_id = request.get('student_id')
        solution_code = request.get('solution_code', '')
        task_type = request.get('task_type', 'tarea')
        language = request.get('language', 'auto')

        # Validar IDs
        if not task_id or not student_id:
            logger.warning(f"Missing IDs - task_id: {task_id}, student_id: {student_id}")
            raise HTTPException(
                status_code=400,
                detail="task_id y student_id son requeridos"
            )

        # Validar tipo de tarea (Examen bloqueado)
        if task_type == 'examen':
            logger.info(f"❌ Exam blocked - Student: {student_id}, Task: {task_id}")
            raise HTTPException(
                status_code=403,
                detail="El análisis no está disponible para exámenes"
            )

        # Validar código
        if not solution_code or len(solution_code.strip()) < 10:
            logger.warning(
                f"Invalid code length - Student: {student_id}, "
                f"Length: {len(solution_code.strip())}"
            )
            raise HTTPException(
                status_code=400,
                detail="El código debe tener al menos 10 caracteres"
            )

        # ====================================================================
        # CAPA 2: RATE LIMITING - QUOTA CHECK
        # ====================================================================

        quota_allowed, quota_message, analyses_count = check_analysis_quota(
            db, student_id, task_id
        )

        if not quota_allowed:
            logger.warning(
                f"⚠️ Quota exceeded - Student: {student_id}, Task: {task_id}, "
                f"Count: {analyses_count}"
            )
            raise HTTPException(
                status_code=429,
                detail=quota_message
            )

        # ====================================================================
        # CAPA 3: RATE LIMITING - COOLDOWN CHECK
        # ====================================================================

        cooldown_allowed, seconds_remaining = check_cooldown(db, student_id, task_id)

        if not cooldown_allowed:
            logger.warning(
                f"⚠️ Cooldown active - Student: {student_id}, Task: {task_id}, "
                f"Remaining: {seconds_remaining}s"
            )
            raise HTTPException(
                status_code=429,
                detail=f"Espera {seconds_remaining} segundos antes del próximo análisis"
            )

        # ====================================================================
        # CAPA 4: ABUSE DETECTION
        # ====================================================================

        abuse_pattern = detect_abuse_pattern(db, student_id)

        if abuse_pattern['is_suspicious']:
            logger.warning(
                f"🚨 Suspicious pattern detected - Student: {student_id}, "
                f"Reason: {abuse_pattern['reason']}, "
                f"Severity: {abuse_pattern['severity']}"
            )
            # Para ahora: log pero no bloquear
            # Futuro: bloquear o requerir verificación

        # ====================================================================
        # CREAR REQUEST Y ANALIZAR
        # ====================================================================

        student_solution_request = StudentSolutionAnalysisRequest(
            task_id=task_id,
            student_id=student_id,
            solution_code=solution_code,
            task_type=task_type,
            language=language
        )

        logger.info(
            f"📊 Analyzing solution - Student: {student_id}, Task: {task_id}, "
            f"Language: {language}, Type: {task_type}"
        )

        # Analizar con el agente
        analysis_response = enhancement_agent.analyze_student_solution(
            student_solution_request
        )

        # ====================================================================
        # REGISTRAR EN BASE DE DATOS (Rate Limiting)
        # ====================================================================

        if db:
            db_record = record_analysis(
                db=db,
                student_id=student_id,
                task_id=task_id,
                analysis_id=str(analysis_response.id),
                language=language,
                task_type=task_type,
                response_json=analysis_response.dict(),
                analysis_duration_ms=analysis_response.metadata.get('analysis_duration_ms', 0),
                ip_address=None,  # Obtener de request si está disponible
                user_agent=None   # Obtener de request si está disponible
            )

            if db_record:
                logger.info(
                    f"✅ Analysis recorded in DB - ID: {db_record.id}, "
                    f"Analysis_ID: {analysis_response.id}"
                )

        # ====================================================================
        # REGISTRAR EN AUDITORÍA (Logging & Auditoría)
        # ====================================================================

        if db and analysis_response:
            try:
                # Extraer datos de feedback para auditoría
                concepts_count = len(analysis_response.feedback.conceptos_correctos) if analysis_response.feedback.conceptos_correctos else 0
                errors_count = len(analysis_response.feedback.errores_encontrados) if analysis_response.feedback.errores_encontrados else 0

                # Registrar en tabla de auditoría
                audit_log = log_analysis(
                    db=db,
                    student_id=student_id,
                    task_id=task_id,
                    course_id=1,  # Obtener del request o contexto
                    analysis_id=str(analysis_response.id),
                    language=language,
                    task_type=task_type,
                    duration_ms=analysis_response.metadata.get('analysis_duration_ms', 0),
                    success=True,
                    concepts_found=concepts_count,
                    errors_found=errors_count,
                    code_length=len(solution_code),
                    feedback_json=analysis_response.feedback.dict() if analysis_response.feedback else None,
                    ip_address=None,
                    user_agent=None,
                    error_message=None
                )

                if audit_log:
                    logger.info(
                        f"✅ Audit log recorded - Student: {student_id}, "
                        f"Analysis_ID: {analysis_response.id}, "
                        f"Concepts: {concepts_count}, Errors: {errors_count}"
                    )
            except Exception as e:
                logger.error(f"❌ Error recording audit log: {str(e)}")
                # No bloquear el análisis si el audit falla

        # ====================================================================
        # RETORNAR RESPUESTA
        # ====================================================================

        logger.info(
            f"✅ Solution analysis completed - Student: {student_id}, "
            f"Concepts: {len(analysis_response.feedback.conceptos_correctos)}, "
            f"Duration: {analysis_response.metadata.get('analysis_duration_ms', 0)}ms"
        )

        return {
            'success': True,
            'data': analysis_response.dict(),
            'rate_limit': {
                'remaining_analyses': 5 - (analyses_count + 1),
                'total_analyses': analyses_count + 1,
                'quota_message': quota_message
            }
        }

    except HTTPException:
        raise

    except Exception as e:
        logger.error(f"❌ Error analyzing solution: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail=f"Error analizando solución: {str(e)}"
        )

    finally:
        # Cerrar conexión a BD si la abrimos
        if should_close_db and db:
            try:
                db.close()
            except:
                pass


# ============================================================================
# AUDITORÍA - NUEVOS ENDPOINTS (Punto 5)
# ============================================================================


@app.get("/api/audit/student/{student_id}/history")
async def get_student_analysis_history(student_id: int, course_id: Optional[int] = None, limit: int = 50):
    """
    Obtiene historial de análisis de un estudiante.

    Args:
        student_id: ID del estudiante
        course_id: ID del curso (opcional)
        limit: Máximo de registros (default: 50)

    Returns:
        Lista de análisis ordenados por fecha DESC
    """
    db = None
    try:
        from shared.database.connection import get_db_session
        db = get_db_session()

        history = get_student_history(
            db=db,
            student_id=student_id,
            course_id=course_id,
            limit=limit
        )

        logger.info(
            f"✅ Retrieved student history - Student: {student_id}, "
            f"Records: {len(history) if history else 0}"
        )

        return {
            'success': True,
            'student_id': student_id,
            'total_records': len(history) if history else 0,
            'data': [record.to_dict() for record in history] if history else []
        }

    except Exception as e:
        logger.error(f"❌ Error retrieving student history: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        if db:
            try:
                db.close()
            except:
                pass


@app.get("/api/audit/task/{task_id}/usage")
async def get_task_analysis_usage(task_id: int, course_id: Optional[int] = None):
    """
    Obtiene estadísticas de uso de una tarea.

    Args:
        task_id: ID de la tarea
        course_id: ID del curso (opcional)

    Returns:
        Estadísticas de uso (total analyses, estudiantes, tasa de éxito, etc)
    """
    db = None
    try:
        from shared.database.connection import get_db_session
        db = get_db_session()

        stats = get_audit_task_stats(
            db=db,
            task_id=task_id,
            course_id=course_id
        )

        logger.info(
            f"✅ Retrieved task usage stats - Task: {task_id}, "
            f"Total analyses: {stats.get('total_analyses', 0)}"
        )

        return {
            'success': True,
            'task_id': task_id,
            'data': stats
        }

    except Exception as e:
        logger.error(f"❌ Error retrieving task usage stats: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        if db:
            try:
                db.close()
            except:
                pass


@app.get("/api/audit/abuse-alerts")
async def get_active_abuse_alerts(course_id: int, severity: Optional[str] = None, limit: int = 50):
    """
    Obtiene alertas de abuso activas para un curso.

    Args:
        course_id: ID del curso
        severity: Filtrar por severidad (low, medium, high) - opcional
        limit: Máximo de registros (default: 50)

    Returns:
        Lista de alertas activas ordenadas por fecha DESC
    """
    db = None
    try:
        from shared.database.connection import get_db_session
        db = get_db_session()

        alerts = get_abuse_alerts(
            db=db,
            course_id=course_id,
            severity=severity,
            resolved=False,
            limit=limit
        )

        logger.info(
            f"✅ Retrieved abuse alerts - Course: {course_id}, "
            f"Alerts: {len(alerts) if alerts else 0}"
        )

        return {
            'success': True,
            'course_id': course_id,
            'severity_filter': severity,
            'total_alerts': len(alerts) if alerts else 0,
            'data': [alert.to_dict() for alert in alerts] if alerts else []
        }

    except Exception as e:
        logger.error(f"❌ Error retrieving abuse alerts: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        if db:
            try:
                db.close()
            except:
                pass


@app.get("/api/audit/professor/dashboard/{course_id}")
async def get_professor_dashboard_data(course_id: int):
    """
    Obtiene datos para el dashboard del profesor.

    Incluye:
    - Estadísticas generales del curso
    - Alertas activas
    - Estudiantes problemáticos
    - Estadísticas por tarea

    Args:
        course_id: ID del curso

    Returns:
        Dict con datos completos del dashboard
    """
    db = None
    try:
        from shared.database.connection import get_db_session
        db = get_db_session()

        dashboard_data = get_professor_dashboard(
            db=db,
            course_id=course_id
        )

        logger.info(
            f"✅ Generated professor dashboard - Course: {course_id}, "
            f"Alerts: {len(dashboard_data.get('active_alerts', []))}"
        )

        return {
            'success': True,
            'course_id': course_id,
            'data': dashboard_data,
            'generated_at': datetime.utcnow().isoformat() + 'Z'
        }

    except Exception as e:
        logger.error(f"❌ Error generating professor dashboard: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        if db:
            try:
                db.close()
            except:
                pass


@app.post("/api/audit/student/{student_id}/notify")
async def notify_student_abuse(student_id: int, action_request: Dict[str, Any]):
    """
    Registra una acción manual sobre un estudiante (por profesor).

    Args:
        student_id: ID del estudiante
        action_request: {
            "alert_id": int,
            "action": "reviewed" | "blocked" | "notified" | "resolved",
            "resolution_notes": str (opcional)
        }

    Returns:
        Confirmación de acción registrada
    """
    db = None
    try:
        from shared.database.connection import get_db_session
        db = get_db_session()

        alert_id = action_request.get('alert_id')
        action = action_request.get('action', 'reviewed')
        resolution_notes = action_request.get('resolution_notes')

        if not alert_id:
            raise HTTPException(status_code=400, detail="alert_id es requerido")

        # Obtener alerta y actualizar
        alert = db.query(AbuseAlert).filter(AbuseAlert.id == alert_id).first()

        if not alert:
            raise HTTPException(status_code=404, detail=f"Alerta {alert_id} no encontrada")

        # Actualizar alerta
        alert.action_taken = action
        alert.action_timestamp = datetime.utcnow()
        alert.action_by = f"professor_{student_id}"  # Placeholder

        if action == 'resolved':
            alert.resolved = True
            alert.resolved_at = datetime.utcnow()
            alert.resolution_notes = resolution_notes

        db.commit()

        logger.info(
            f"✅ Alert action recorded - Alert: {alert_id}, "
            f"Student: {student_id}, Action: {action}"
        )

        return {
            'success': True,
            'alert_id': alert_id,
            'student_id': student_id,
            'action': action,
            'message': f"Acción '{action}' registrada exitosamente"
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"❌ Error registering alert action: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        if db:
            try:
                db.close()
            except:
                pass


@app.get("/health", response_model=HealthResponse)
async def health_check() -> HealthResponse:
    """
    Check Agent Service health

    Verifies that the service is running and LLM is available.
    """
    return HealthResponse(
        status="healthy",
        llm_available=orchestrator.synthesizer.llm_available,
        timestamp=datetime.utcnow().isoformat() + 'Z'
    )


@app.get("/")
async def service_info() -> Dict[str, Any]:
    """
    Get Agent Service information

    Returns service metadata and capabilities.
    """
    return {
        'name': 'Agent Synthesis Service',
        'version': '2.0.0',
        'description': 'LLM-based synthesis of ML discoveries with audit logging',
        'endpoints': {
            # Core endpoints
            'synthesize': 'POST /synthesize',
            'reasoning': 'POST /reasoning',
            'intervention_strategy': 'POST /intervention-strategy',
            # Analysis endpoints (Punto 3)
            'content_analysis': 'POST /api/analysis/content-check',
            'difficulty_recommendation': 'POST /api/recommendation/difficulty',
            'student_solution_analysis': 'POST /api/analysis/student-solution',
            # Audit endpoints (Punto 5)
            'student_history': 'GET /api/audit/student/{student_id}/history',
            'task_usage': 'GET /api/audit/task/{task_id}/usage',
            'abuse_alerts': 'GET /api/audit/abuse-alerts',
            'professor_dashboard': 'GET /api/audit/professor/dashboard/{course_id}',
            'alert_action': 'POST /api/audit/student/{student_id}/notify',
            # System endpoints
            'health': 'GET /health',
            'info': 'GET /',
        },
        'components': {
            'rate_limiting': 'Enabled (max 5 analyses per task, 5 min cooldown)',
            'audit_logging': 'Enabled (complete audit trail)',
            'abuse_detection': 'Enabled (real-time pattern detection)',
        },
        'llm_available': orchestrator.synthesizer.llm_available,
        'llm_model': 'llama-3.3-70b-versatile' if orchestrator.synthesizer.llm_available else 'local_fallback',
        'timestamp': datetime.utcnow().isoformat() + 'Z',
    }


# ========================
# Error Handlers
# ========================

@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    """Global exception handler"""
    logger.error(f"Unhandled exception: {str(exc)}")
    return {
        'success': False,
        'error': 'Internal server error',
        'timestamp': datetime.utcnow().isoformat() + 'Z',
    }


# ========================
# Startup/Shutdown
# ========================

@app.on_event("startup")
async def startup_event():
    """Initialize on startup"""
    logger.info("Agent Service starting...")
    logger.info(f"LLM Available: {orchestrator.synthesizer.llm_available}")
    if not orchestrator.synthesizer.llm_available:
        logger.warning("LLM not available - using local synthesis fallback")


@app.on_event("shutdown")
async def shutdown_event():
    """Cleanup on shutdown"""
    logger.info("Agent Service shutting down...")


if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv('PORT', 8080))
    log_level = os.getenv('LOG_LEVEL', 'info').lower()
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=port,
        log_level=log_level
    )
