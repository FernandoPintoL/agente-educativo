"""
FastAPI Agent Server - LLM-Based Synthesis and Reasoning (v2.0)
Puerto LOCAL: 8003 | Puerto RAILWAY: 8080

Sintetiza descubrimientos de pipelines ML (supervisado y no supervisado)
usando LLM (Groq) para generar insights inteligentes y estrategias
de intervención educativa personalizadas.

Arquitectura:
- LLMSynthesizer: Integración Groq con fallback elegante
- AgentOrchestrator: Orquesta síntesis, razonamiento y estrategias
- Endpoints FastAPI para integración con pipelines ML

Uso:
    python api_server.py          (Local: puerto 8003)
    uvicorn api_server:app --port 8080  (Railway: puerto 8080)
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
from urllib.parse import quote

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

# Import configuration
try:
    from config import PORT, HOST, ENVIRONMENT, LOG_LEVEL, DEBUG
except ImportError:
    try:
        from .config import PORT, HOST, ENVIRONMENT, LOG_LEVEL, DEBUG
    except ImportError:
        # Fallback: use environment variables directly
        PORT = int(os.getenv('PORT', 8003))
        HOST = os.getenv('HOST', '0.0.0.0')
        ENVIRONMENT = os.getenv('ENVIRONMENT', 'development')
        LOG_LEVEL = os.getenv('LOG_LEVEL', 'INFO')
        DEBUG = os.getenv('DEBUG', 'true').lower() == 'true'

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
    if ENVIRONMENT == 'production':
        logger.error(f"❌ Error inicializando Enhancement Agent: {str(e)}")
    else:
        logger.warning(f"⚠️  Enhancement Agent fallback en LOCAL: {str(e)}")
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


class GenerateQuestionsRequest(BaseModel):
    """Request to generate questions for evaluation"""
    titulo: str = Field(..., min_length=5, max_length=255)
    tipo_evaluacion: str
    curso_id: int
    cantidad_preguntas: int = Field(default=5, ge=1, le=50)
    dificultad_deseada: str = Field(default="intermedia")
    contexto: Optional[str] = None


class PreguntaGenerada(BaseModel):
    """Generated question structure - enriched with pedagogical metadata"""
    enunciado: str
    tipo: str
    opciones: List[str]
    respuesta_correcta: str
    puntos: int
    bloom_level: Optional[str] = None
    explicacion_detallada: Optional[str] = None
    dificultad_estimada: Optional[float] = None
    conceptos_clave: Optional[List[str]] = None
    notas_profesor: Optional[str] = None
    distractores_explicacion: Optional[List[str]] = None


class GenerateQuestionsResponse(BaseModel):
    """Response with generated questions"""
    success: bool
    preguntas: List[PreguntaGenerada]
    metadata: Dict[str, Any]
    timestamp: str


class DistractorRequest(BaseModel):
    """Request to generate intelligent distractors for a question"""
    enunciado: str = Field(..., min_length=10)
    respuesta_correcta: str = Field(..., min_length=1)
    tipo: str = Field(default="opcion_multiple")
    opciones_actuales: Optional[List[str]] = None
    nivel_bloom: Optional[str] = None
    errores_comunes: Optional[List[str]] = None


class DistractorResponse(BaseModel):
    """Response with intelligent distractors"""
    success: bool
    distractores: List[Dict[str, str]]  # [{"opcion": "...", "razon": "...", "error_conceptual": "..."}]
    metadata: Dict[str, Any]
    timestamp: str


# ========================
# VOCATIONAL SYNTHESIS MODELS
# ========================

class VocationalSynthesisRequest(BaseModel):
    """Request para síntesis vocacional personalizada"""
    student_id: int
    nombre_estudiante: str
    promedio_academico: float  # 0-100
    carrera_predicha: str  # Carrera recomendada por ML
    confianza: float  # 0-1 (confianza de la predicción)
    cluster_aptitud: int  # ID del cluster (0, 1, 2)
    cluster_nombre: str  # Nombre del cluster (Bajo/Medio/Alto desempeño)
    areas_interes: Dict[str, float]  # {"tecnologia": 95, "ciencias": 80, ...}


class VocationalSynthesisResponse(BaseModel):
    """Response con síntesis vocacional personalizada"""
    student_id: int
    narrativa: str  # Narrativa personalizada sobre el perfil vocacional
    recomendaciones: List[str]  # Recomendaciones actionables
    pasos_siguientes: List[str]  # Plan de acción a corto/mediano plazo
    sintesis_tipo: str  # "groq" si usó LLM, "fallback" si usó generación local
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
                if ENVIRONMENT == 'production':
                    logger.error("GROQ_API_KEY requerida en PRODUCTION")
                else:
                    logger.info("GROQ_API_KEY no configurada en LOCAL - using local synthesis fallback")
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

    def generate(self, prompt: str) -> str:
        """
        Generate text response using LLM for a given prompt
        Returns plain text response from Groq API
        Falls back to local generation if LLM unavailable
        """
        if not self.llm_available:
            logger.info("LLM unavailable, using local fallback")
            return self._local_generate(prompt)

        try:
            response = self.llm.invoke(prompt)
            generated_text = response.content
            logger.info("✅ Generated response from LLM")
            return generated_text
        except Exception as e:
            logger.warning(f"LLM generation failed: {str(e)}, falling back to local")
            return self._local_generate(prompt)

    def _local_generate(self, prompt: str) -> str:
        """
        Local fallback for generating text when LLM is unavailable
        Returns a basic template response
        """
        logger.info("Using local generation fallback")
        return "No se pudo generar contenido - servicio de IA no disponible. Por favor intenta de nuevo más tarde."

    def _build_synthesis_prompt(self, student_id: int, discoveries: Dict[str, Any], predictions: Dict[str, Any], context: str) -> str:
        """Build prompt for LLM synthesis - generates response in Spanish"""
        prompt = f"""
Analiza los datos de aprendizaje de este estudiante y proporciona un análisis detallado EN ESPAÑOL:

ID del Estudiante: {student_id}
Contexto: {context}

DESCUBRIMIENTOS ML NO SUPERVISADO:
{json.dumps(discoveries, indent=2)}

PREDICCIONES ML SUPERVISADO:
{json.dumps(predictions, indent=2)}

Por favor, proporciona:
1. Insights principales de los datos (análisis profundo de cada predicción)
2. Recomendaciones principales para intervención pedagógica
3. Resumen del perfil de aprendizaje del estudiante

IMPORTANTE: Responde COMPLETAMENTE en ESPAÑOL. Estructura tu respuesta de forma clara y profesional.
"""
        return prompt

    def _build_reasoning_prompt(self, student_id: int, discoveries: Dict[str, Any], predictions: Dict[str, Any]) -> str:
        """Build prompt for detailed reasoning - generates response in Spanish"""
        return f"""
Proporciona un razonamiento paso a paso para estos descubrimientos EN ESPAÑOL:

ID del Estudiante: {student_id}
Descubrimientos: {json.dumps(discoveries, indent=2)}
Predicciones: {json.dumps(predictions, indent=2)}

Explica cómo estos datos conducen a recomendaciones educativas específicas y medidas de intervención.
Responde COMPLETAMENTE en ESPAÑOL.
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
        """Local synthesis fallback - Spanish version"""
        insights = ["Análisis del estudiante completado", "Se identificaron múltiples patrones de aprendizaje"]
        recommendations = ["Monitorear el progreso académico", "Proporcionar apoyo académico dirigido"]

        if 'cluster_analysis' in discoveries:
            insights.append("Estudiante asignado a un segmento de aprendizaje específico")
            recommendations.append("Aplicar estrategias específicas para este segmento")

        if 'anomalies' in discoveries:
            anomalies = discoveries['anomalies'].get('data', {}).get('detected_patterns', [])
            if anomalies:
                insights.append(f"Se detectaron {len(anomalies)} patrones inusuales en el comportamiento académico")
                recommendations.append("Proporcionar intervención educativa enfocada")

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
        """Local reasoning fallback - Spanish version"""
        return {
            'success': True,
            'reasoning_steps': ['Analizar asignaciones de segmentos', 'Evaluar anomalías detectadas', 'Revisar dominio de temas', 'Validar predicciones ML', 'Sintetizar hallazgos'],
            'key_insights': ['Patrones identificados en el aprendizaje', 'Perfil académico definido', 'Anomalías registradas'],
            'recommendations': ['Implementar intervenciones pedagógicas', 'Monitorear anomalías', 'Reforzar fortalezas académicas'],
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


@app.post("/synthesis/vocational", response_model=VocationalSynthesisResponse)
async def synthesis_vocational(request: VocationalSynthesisRequest) -> VocationalSynthesisResponse:
    """
    Síntesis personalizada de perfil vocacional

    Genera una narrativa personalizada basada en:
    - Predicción de carrera (supervisada)
    - Clustering de aptitudes (no supervisada)
    - Áreas de interés del estudiante
    - Perfil académico

    Retorna:
    - Narrativa personalizada del perfil vocacional
    - Recomendaciones de carreras y especialidades
    - Plan de acción con pasos concretos
    """
    try:
        logger.info(f"Síntesis vocacional solicitada para estudiante {request.student_id}")

        # Construir prompt para Groq
        prompt = f"""
Eres un consejero educativo y orientador vocacional experto. Tu tarea es generar una síntesis personalizada del perfil vocacional de un estudiante basada en datos de predicción ML.

=== DATOS DEL ESTUDIANTE ===
- ID: {request.student_id}
- Nombre: {request.nombre_estudiante}
- Promedio Académico: {request.promedio_academico:.1f}/100
- Carrera Predicha: {request.carrera_predicha}
- Confianza de Predicción: {request.confianza:.0%}
- Cluster de Aptitud: {request.cluster_nombre} (ID: {request.cluster_aptitud})

=== ÁREAS DE INTERÉS ===
{chr(10).join([f"- {area}: {score:.0f}/100" for area, score in sorted(request.areas_interes.items(), key=lambda x: x[1], reverse=True)])}

=== REQUISITOS DE LA SÍNTESIS ===

1. **Narrativa Personalizada** (150-200 palabras):
   - Analiza el perfil del estudiante de forma holística
   - Conecta áreas de interés con la carrera predicha
   - Considerer el cluster de aptitud para contexto
   - Usa un tono motivador y constructivo
   - Incluye fortalezas identificadas

2. **Recomendaciones** (5-7 puntos concretos):
   - Acciones específicas para explorar la carrera
   - Habilidades a desarrollar
   - Experiencias prácticas sugeridas
   - Conexiones profesionales a buscar
   - Recursos educativos relevantes

3. **Pasos Siguientes** (4-5 etapas con timeline):
   - Semana 1-2: Acciones inmediatas
   - Mes 1-2: Exploración más profunda
   - Mes 2-3: Experiencias prácticas
   - Mes 3+: Consolidación y decisiones

=== RESTRICCIONES IMPORTANTES ===
- Responde COMPLETAMENTE en ESPAÑOL
- Sé motivador pero realista
- Menciona desafíos potenciales con soluciones
- Personaliza cada recomendación al perfil específico
- Nunca desanimes al estudiante
- Estructura claramente cada sección

=== FORMATO JSON REQUERIDO ===

Retorna SOLO JSON válido, sin explicaciones adicionales:

{{
    "narrativa": "Tu análisis personalizado aquí...",
    "recomendaciones": [
        "Recomendación 1",
        "Recomendación 2",
        ...
    ],
    "pasos_siguientes": [
        "Semana 1-2: ...",
        "Mes 1-2: ...",
        ...
    ]
}}

Genera la síntesis personalizada ahora:
"""

        # Intentar usar Groq si está disponible
        if orchestrator.synthesizer.llm_available:
            try:
                logger.info("Usando Groq para síntesis vocacional")
                response = orchestrator.synthesizer.llm.invoke(prompt)
                response_text = response.content

                # Parsear JSON
                import json
                try:
                    # Limpiar markdown si está presente
                    if "```json" in response_text:
                        response_text = response_text.split("```json")[1].split("```")[0]
                    elif "```" in response_text:
                        response_text = response_text.split("```")[1].split("```")[0]

                    synthesis_data = json.loads(response_text)

                    return VocationalSynthesisResponse(
                        student_id=request.student_id,
                        narrativa=synthesis_data.get("narrativa", ""),
                        recomendaciones=synthesis_data.get("recomendaciones", []),
                        pasos_siguientes=synthesis_data.get("pasos_siguientes", []),
                        sintesis_tipo="groq",
                        timestamp=datetime.utcnow().isoformat() + 'Z'
                    )

                except json.JSONDecodeError:
                    logger.warning("No se pudo parsear respuesta JSON de Groq, usando fallback")
                    synthesis_data = _generate_fallback_vocational_synthesis(request)

                    return VocationalSynthesisResponse(
                        student_id=request.student_id,
                        narrativa=synthesis_data["narrativa"],
                        recomendaciones=synthesis_data["recomendaciones"],
                        pasos_siguientes=synthesis_data["pasos_siguientes"],
                        sintesis_tipo="fallback",
                        timestamp=datetime.utcnow().isoformat() + 'Z'
                    )

            except Exception as e:
                logger.warning(f"Error usando Groq para síntesis: {str(e)}, usando fallback")
                synthesis_data = _generate_fallback_vocational_synthesis(request)

                return VocationalSynthesisResponse(
                    student_id=request.student_id,
                    narrativa=synthesis_data["narrativa"],
                    recomendaciones=synthesis_data["recomendaciones"],
                    pasos_siguientes=synthesis_data["pasos_siguientes"],
                    sintesis_tipo="fallback",
                    timestamp=datetime.utcnow().isoformat() + 'Z'
                )
        else:
            logger.info("Groq no disponible, usando síntesis local")
            synthesis_data = _generate_fallback_vocational_synthesis(request)

            return VocationalSynthesisResponse(
                student_id=request.student_id,
                narrativa=synthesis_data["narrativa"],
                recomendaciones=synthesis_data["recomendaciones"],
                pasos_siguientes=synthesis_data["pasos_siguientes"],
                sintesis_tipo="fallback",
                timestamp=datetime.utcnow().isoformat() + 'Z'
            )

    except Exception as e:
        logger.error(f"Error en synthesis_vocational: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


def _generate_fallback_vocational_synthesis(request: VocationalSynthesisRequest) -> Dict[str, Any]:
    """Generar síntesis vocacional sin LLM"""
    top_interests = sorted(request.areas_interes.items(), key=lambda x: x[1], reverse=True)[:2]
    top_areas = ", ".join([area.capitalize() for area, _ in top_interests])

    narrativa = f"""Tu perfil vocacional muestra una sólida afinidad por {top_areas} y el análisis predictivo sugiere que {request.carrera_predicha} sería una excelente opción para ti. Tu promedio académico de {request.promedio_academico:.1f}/100 demuestra tu capacidad de aprendizaje, y tu cluster de {request.cluster_nombre} indica que tienes el potencial necesario para prosperar en esta carrera. La confianza en esta predicción es del {request.confianza:.0%}, lo que refleja la solidez del análisis. Te recomendamos que explores esta carrera de forma sistemática mientras desarrollas las habilidades complementarias que hoy en día son cada vez más valoradas en el mercado laboral."""

    recomendaciones = [
        f"Investiga programas académicos de {request.carrera_predicha} en universidades de renombre",
        "Conecta con profesionales en el área mediante redes profesionales y eventos de networking",
        f"Desarrolla habilidades técnicas complementarias a {top_areas.lower()}",
        "Busca experiencias prácticas como internados o voluntariado en el área",
        f"Participa en proyectos o clubes relacionados con tus áreas de interés",
        "Mantén tu promedio académico alto para futuras oportunidades",
    ]

    pasos_siguientes = [
        f"Semana 1-2: Investigar al menos 3 universidades que ofrecen {request.carrera_predicha}",
        f"Mes 1-2: Conectar con al menos 2 profesionales trabajando en {request.carrera_predicha}",
        f"Mes 2-3: Completar un curso online introductorio en el área de {top_areas.lower()}",
        f"Mes 3+: Buscar y aplicar para oportunidades de experiencia práctica o voluntariado",
    ]

    return {
        "narrativa": narrativa,
        "recomendaciones": recomendaciones,
        "pasos_siguientes": pasos_siguientes,
    }


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


@app.post("/api/generation/questions", response_model=GenerateQuestionsResponse)
async def generate_questions(request: GenerateQuestionsRequest) -> GenerateQuestionsResponse:
    """
    Generate multiple questions using Groq LLM

    Generates educational questions based on evaluation title, type, and context.
    Uses Groq LLM to create varied, pedagogically sound questions.

    Args:
        request: GenerateQuestionsRequest with evaluation details

    Returns:
        GenerateQuestionsResponse with generated questions
    """
    if not enhancement_agent:
        raise HTTPException(
            status_code=503,
            detail="Task Enhancement Agent no está disponible"
        )

    try:
        logger.info(f"Generando {request.cantidad_preguntas} preguntas para: {request.titulo[:50]}...")

        # Use LLM to generate questions
        if not orchestrator.synthesizer.llm_available:
            logger.warning("LLM no disponible, generando preguntas locales")
            preguntas = _generate_local_questions(request)
        else:
            preguntas = await _generate_questions_with_llm(request)

        logger.info(f"✅ Generación completada - {len(preguntas)} preguntas generadas")

        return GenerateQuestionsResponse(
            success=True,
            preguntas=preguntas,
            metadata={
                'cantidad': len(preguntas),
                'dificultad': request.dificultad_deseada,
                'tipo': request.tipo_evaluacion,
                'fuente': 'llm' if orchestrator.synthesizer.llm_available else 'local',
                'confidence': 0.8 if orchestrator.synthesizer.llm_available else 0.5,
            },
            timestamp=datetime.utcnow().isoformat() + 'Z'
        )

    except Exception as e:
        logger.error(f"❌ Error generando preguntas: {str(e)}")
        # Fallback to local generation
        try:
            preguntas = _generate_local_questions(request)
            return GenerateQuestionsResponse(
                success=True,
                preguntas=preguntas,
                metadata={
                    'cantidad': len(preguntas),
                    'error': str(e),
                    'fuente': 'local_fallback',
                    'confidence': 0.5,
                },
                timestamp=datetime.utcnow().isoformat() + 'Z'
            )
        except Exception as fallback_error:
            logger.error(f"❌ Fallback error: {str(fallback_error)}")
            raise HTTPException(
                status_code=500,
                detail=f"Error generando preguntas: {str(e)}"
            )


def _generate_local_questions(request: GenerateQuestionsRequest) -> List[PreguntaGenerada]:
    """Generate basic questions locally without LLM"""
    preguntas = []

    # Crear templates básicos que el usuario puede editar
    tipos = ['opcion_multiple', 'verdadero_falso', 'respuesta_corta']

    for i in range(request.cantidad_preguntas):
        tipo = tipos[i % len(tipos)]

        if tipo == 'opcion_multiple':
            opciones = [f'Opción A', 'Opción B', 'Opción C', 'Opción D']
            respuesta = 'Opción A'
        elif tipo == 'verdadero_falso':
            opciones = ['Verdadero', 'Falso']
            respuesta = 'Verdadero'
        else:
            opciones = []
            respuesta = '[Respuesta esperada]'

        preguntas.append(PreguntaGenerada(
            enunciado=f"Pregunta {i+1}: [Edita este enunciado - Tema: {request.titulo}]",
            tipo=tipo,
            opciones=opciones,
            respuesta_correcta=respuesta,
            puntos=10,
            bloom_level='remember'
        ))

    return preguntas


async def _generate_questions_with_llm(request: GenerateQuestionsRequest) -> List[PreguntaGenerada]:
    """Generate questions using Groq LLM with enriched pedagogical context"""
    try:
        # Prepare enriched prompt for LLM
        # Map dificultad_deseada to numerical range for Bloom distribution
        dificultad_map = {
            'facil': {'min': 0.2, 'max': 0.4},
            'intermedia': {'min': 0.4, 'max': 0.6},
            'dificil': {'min': 0.6, 'max': 0.8},
            'muy_dificil': {'min': 0.8, 'max': 1.0},
        }

        dif_range = dificultad_map.get(request.dificultad_deseada, dificultad_map['intermedia'])

        # Define Bloom distribution based on difficulty
        if request.dificultad_deseada == 'facil':
            bloom_dist = "40% remember, 40% understand, 20% apply"
        elif request.dificultad_deseada == 'dificil':
            bloom_dist = "10% remember, 20% understand, 30% apply, 25% analyze, 15% evaluate/create"
        elif request.dificultad_deseada == 'muy_dificil':
            bloom_dist = "5% remember, 10% understand, 20% apply, 35% analyze, 30% evaluate/create"
        else:  # intermedia
            bloom_dist = "20% remember, 30% understand, 25% apply, 15% analyze, 10% evaluate/create"

        prompt = f"""
Eres un experto en pedagogía y diseño educativo. Tu tarea es generar exactamente {request.cantidad_preguntas} preguntas de evaluación de alta calidad en formato JSON.

=== CONTEXTO PEDAGÓGICO ===
- Título de evaluación: {request.titulo}
- Tipo: {request.tipo_evaluacion}
- Nivel de dificultad deseado: {request.dificultad_deseada.upper()} (escala 0.0-1.0: {dif_range['min']}-{dif_range['max']})
- Contexto específico: {request.contexto or 'General'}
- Distribución de niveles Bloom objetivo: {bloom_dist}

=== REQUISITOS DE CALIDAD ===

1. **Variedad de tipos**:
   - Opción múltiple (60%): 4 opciones, una correcta
   - Verdadero/Falso (20%): solo 2 opciones
   - Respuesta corta (20%): respuesta concisa, clara

2. **Cada pregunta DEBE incluir**:
   - enunciado: claro, conciso, sin ambigüedad (50-200 caracteres)
   - tipo: opcion_multiple, verdadero_falso, respuesta_corta
   - opciones: array con las opciones (varía según tipo)
   - respuesta_correcta: texto exacto de la respuesta correcta
   - explicacion_detallada: por qué es correcta (50-150 caracteres)
   - nivel_bloom: remember|understand|apply|analyze|evaluate|create
   - dificultad_estimada: número 0.0-1.0 (0=muy fácil, 1=muy difícil)
   - conceptos_clave: array de 2-3 conceptos evaluados
   - notas_profesor: sugerencia pedagógica breve para el profesor
   - distractores_explicacion: array explicando por qué cada opción incorrecta es un error común

3. **Distractores inteligentes**:
   - Deben representar errores conceptuales comunes
   - No deben ser obviamente incorrectos
   - Para múltiple, incluir una opción "parcialmente correcta"

4. **Distribución de dificultad**:
   - Respetar el rango especificado ({dif_range['min']}-{dif_range['max']})
   - Variar dentro del rango para mantener engagement

5. **Estándares pedagógicos**:
   - Alineado con nivel de estudiante del curso
   - Cada pregunta evalúa UN concepto principal
   - Enunciados sin jerga innecesaria
   - Opciones de similar longitud (evitar pistas)

=== FORMATO JSON REQUERIDO ===

RETORNA SOLO UN JSON VÁLIDO, sin explicaciones adicionales:

{{
    "preguntas": [
        {{
            "enunciado": "¿Cuál es la capital de Francia?",
            "tipo": "opcion_multiple",
            "opciones": ["París", "Londres", "Berlín", "Madrid"],
            "respuesta_correcta": "París",
            "explicacion_detallada": "París es la capital de Francia desde el siglo XII",
            "nivel_bloom": "remember",
            "dificultad_estimada": 0.1,
            "conceptos_clave": ["geografía", "capitales", "Francia"],
            "notas_profesor": "Pregunta básica para evaluar conocimiento factual",
            "distractores_explicacion": [
                "Londres: error común de confundir capitales europeas",
                "Berlín: otro error de confundir capitales",
                "Madrid: otro error de confundir capitales"
            ],
            "puntos": 10
        }}
    ]
}}

=== INSTRUCCIONES FINALES ===
- Genera EXACTAMENTE {request.cantidad_preguntas} preguntas
- Respeta la distribución Bloom: {bloom_dist}
- Mantén dificultades en rango: {dif_range['min']}-{dif_range['max']}
- JSON válido, parseable, sin markdown
- Valida que cada pregunta tenga TODOS los campos requeridos
"""

        # Call Groq LLM
        llm_response = orchestrator.synthesizer.generate(prompt)

        # Parse JSON response with better error handling
        import json
        try:
            # Remove potential markdown code blocks
            if "```json" in llm_response:
                llm_response = llm_response.split("```json")[1].split("```")[0]
            elif "```" in llm_response:
                llm_response = llm_response.split("```")[1].split("```")[0]

            data = json.loads(llm_response)
            preguntas_data = data.get('preguntas', [])
        except json.JSONDecodeError as e:
            logger.warning(f"No se pudo parsear respuesta JSON: {str(e)}, usando fallback")
            return _generate_local_questions(request)

        # Convert to PreguntaGenerada objects
        preguntas = []
        for q in preguntas_data[:request.cantidad_preguntas]:
            try:
                # Validate required fields
                if not q.get('enunciado') or not q.get('respuesta_correcta'):
                    logger.warning(f"Pregunta incompleta, saltando: {q}")
                    continue

                preguntas.append(PreguntaGenerada(
                    enunciado=q.get('enunciado', ''),
                    tipo=q.get('tipo', 'opcion_multiple'),
                    opciones=q.get('opciones', []),
                    respuesta_correcta=q.get('respuesta_correcta', ''),
                    puntos=int(q.get('puntos', 10)),
                    bloom_level=q.get('nivel_bloom', 'remember'),
                    explicacion_detallada=q.get('explicacion_detallada'),
                    dificultad_estimada=float(q.get('dificultad_estimada', 0.5)) if q.get('dificultad_estimada') else None,
                    conceptos_clave=q.get('conceptos_clave', []),
                    notas_profesor=q.get('notas_profesor'),
                    distractores_explicacion=q.get('distractores_explicacion', [])
                ))
            except Exception as e:
                logger.warning(f"Error procesando pregunta: {str(e)}, saltando")
                continue

        return preguntas if preguntas else _generate_local_questions(request)

    except Exception as e:
        logger.error(f"Error en LLM generation: {str(e)}")
        return _generate_local_questions(request)


@app.post("/api/generation/distractors", response_model=DistractorResponse)
async def generate_distractors(request: DistractorRequest) -> DistractorResponse:
    """
    Generate intelligent distractors for a question

    Creates pedagogically sound distractors that represent common student errors
    and misconceptions based on the question context.

    Args:
        request: DistractorRequest with question details

    Returns:
        DistractorResponse with generated distractors
    """
    try:
        logger.info(f"Generando distractores para pregunta: {request.enunciado[:50]}...")

        if not orchestrator.synthesizer.llm_available:
            logger.warning("LLM no disponible, usando distractores básicos")
            distractores = _generate_local_distractors(request)
        else:
            distractores = await _generate_distractors_with_llm(request)

        logger.info(f"✅ Generación de distractores completada - {len(distractores)} distractores generados")

        return DistractorResponse(
            success=True,
            distractores=distractores,
            metadata={
                'cantidad': len(distractores),
                'tipo_pregunta': request.tipo,
                'fuente': 'llm' if orchestrator.synthesizer.llm_available else 'local',
                'confidence': 0.8 if orchestrator.synthesizer.llm_available else 0.5,
            },
            timestamp=datetime.utcnow().isoformat() + 'Z'
        )

    except Exception as e:
        logger.error(f"❌ Error generando distractores: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Error generando distractores: {str(e)}"
        )


async def _generate_distractors_with_llm(request: DistractorRequest) -> List[Dict[str, str]]:
    """Generate intelligent distractors using Groq LLM"""
    try:
        prompt = f"""
Eres un experto en educación. Tu tarea es generar distractores inteligentes para una pregunta de evaluación.

=== CONTEXTO ===
- Enunciado: {request.enunciado}
- Respuesta correcta: {request.respuesta_correcta}
- Tipo de pregunta: {request.tipo}
- Nivel Bloom: {request.nivel_bloom or 'No especificado'}
- Errores comunes conocidos: {', '.join(request.errores_comunes) if request.errores_comunes else 'Ninguno'}

=== REQUISITOS ===
1. Genera 3-4 distractores (opciones incorrectas)
2. Cada distractor DEBE:
   - Representar un error conceptual común o malinterpretación frecuente
   - Ser plausible (no obvio que sea incorrecto)
   - Tener similar extensión que la respuesta correcta
   - No ser exactamente lo opuesto de la respuesta correcta

3. Para cada distractor, proporciona:
   - opcion: el texto del distractor
   - razon: por qué es incorrecto
   - error_conceptual: qué concepto erróneo representa

=== FORMATO JSON REQUERIDO ===

Retorna SOLO JSON válido, sin explicaciones:

{{
    "distractores": [
        {{
            "opcion": "Texto del distractor incorrecto",
            "razon": "Explicación técnica de por qué es incorrecto",
            "error_conceptual": "Describe el error conceptual que representa"
        }}
    ]
}}

Genera EXACTAMENTE entre 3 y 4 distractores.
"""

        # Call Groq LLM
        llm_response = orchestrator.synthesizer.generate(prompt)

        # Parse JSON response
        import json
        try:
            # Remove potential markdown code blocks
            if "```json" in llm_response:
                llm_response = llm_response.split("```json")[1].split("```")[0]
            elif "```" in llm_response:
                llm_response = llm_response.split("```")[1].split("```")[0]

            data = json.loads(llm_response)
            distractores_data = data.get('distractores', [])
        except json.JSONDecodeError:
            logger.warning("No se pudo parsear respuesta JSON, usando fallback")
            return _generate_local_distractors(request)

        return distractores_data[:4]  # Máximo 4 distractores

    except Exception as e:
        logger.error(f"Error en distractor LLM generation: {str(e)}")
        return _generate_local_distractors(request)


def _generate_local_distractors(request: DistractorRequest) -> List[Dict[str, str]]:
    """Generate basic distractors locally without LLM"""
    distractores = []

    # Usar errores comunes conocidos si existen
    if request.errores_comunes:
        for error in request.errores_comunes[:3]:
            distractores.append({
                'opcion': error,
                'razon': 'Esta es una interpretación incorrecta del concepto',
                'error_conceptual': 'Confusión conceptual común en este tema'
            })
    else:
        # Generar distractores básicos
        distractores = [
            {
                'opcion': f"Una variación de: {request.respuesta_correcta}",
                'razon': 'Esta es una variación incorrecta de la respuesta correcta',
                'error_conceptual': 'Interpretación parcial del concepto'
            },
            {
                'opcion': f"Lo opuesto a: {request.respuesta_correcta}",
                'razon': 'Esta es la opción opuesta',
                'error_conceptual': 'Confusión de polaridad o inversión del concepto'
            },
            {
                'opcion': 'Una respuesta plausible pero incorrecta',
                'razon': 'Aunque suena correcta, no responde adecuadamente a la pregunta',
                'error_conceptual': 'Malinterpretación del enunciado de la pregunta'
            }
        ]

    return distractores


@app.post("/api/resources")
async def get_study_resources(request: Request) -> Dict[str, Any]:
    """
    Obtener recursos educativos REALES de YouTube y plataformas educativas

    Este endpoint busca y retorna recursos educativos reales (videos, playlists, etc.)
    basados en:
    - El tema/materia
    - El nivel de riesgo (para ajustar dificultad)
    - El idioma preferido

    Args:
        subject: Tema/Materia a estudiar
        risk_level: Nivel de riesgo (LOW, MEDIUM, HIGH) para ajustar dificultad
        current_grade: Calificación actual (usado para contexto)
        student_name: Nombre del estudiante (opcional)
        need: Tipo de necesidad (opcional)

    Returns:
        Dict con recursos REALES con URLs verificadas
    """
    try:
        # Leer JSON del request
        body = await request.json()

        subject = body.get('subject', 'General')
        risk_level = body.get('risk_level', 'MEDIUM')
        current_grade = float(body.get('current_grade', 0))
        student_name = body.get('student_name', 'Estudiante')
        language = body.get('language', 'es')

        logger.info(f"[RESOURCES] 🔍 Buscando recursos REALES para: {subject}")
        logger.info(f"[RESOURCES] Nivel: {risk_level}, Grado: {current_grade}")

        # Obtener YouTube Finder
        from youtube_resources import get_youtube_finder
        finder = get_youtube_finder()

        # Buscar MÚLTIPLES FORMATOS de recursos educativos
        resources_by_format = finder.search_educational_resources_multiformat(
            subject=subject,
            risk_level=risk_level,
            language=language
        )

        # Contar total de recursos
        total_resources = sum(len(v) for v in resources_by_format.values())
        logger.info(f"[RESOURCES] ✓ {total_resources} recursos en múltiples formatos encontrados")

        # Mapeo de emojis por tipo de recurso
        emoji_map = {
            'videos': '📺',
            'articles': '📄',
            'exercises': '🎯',
            'interactive': '📱',
            'documentation': '📖',
            'communities': '👥',
        }

        # Formatear recursos por categoría para el frontend
        formatted_by_category = {}
        for category, resources_list in resources_by_format.items():
            emoji = emoji_map.get(category, '📌')
            formatted_by_category[category] = []

            for resource in resources_list:
                description = resource.get('description', '')
                title = resource.get('title', '')
                url = resource.get('url', '')

                formatted_item = {
                    'title': title,
                    'url': url,
                    'source': resource.get('source', 'Unknown'),
                    'description': description,
                    'type': resource.get('type', 'resource'),
                    'emoji': emoji,
                    'display': f"{emoji} {title}\n   {description}\n   {resource.get('source', '')}"
                }
                formatted_by_category[category].append(formatted_item)

        return {
            'success': True,
            'subject': subject,
            'student_name': student_name,
            'resources_by_format': formatted_by_category,  # Agrupado por tipo
            'raw_resources': resources_by_format,  # Datos crudos para procesamiento
            'total_count': total_resources,
            'breakdown': {category: len(resources) for category, resources in resources_by_format.items()},
            'note': f'{total_resources} recursos en {len(resources_by_format)} categorías diferentes para {subject}',
            'generated_at': datetime.utcnow().isoformat() + 'Z',
        }

    except Exception as e:
        logger.error(f"[RESOURCES] ❌ Error buscando recursos: {str(e)}", exc_info=True)

        # Retornar recursos fallback si hay error
        subject = body.get('subject', 'General')
        return {
            'success': False,
            'subject': subject,
            'resources': [
                f"📺 Buscar en Khan Academy: https://www.khanacademy.org/search?page_search_query={quote(subject)}",
                f"📺 Buscar en YouTube Educativo: https://www.youtube.com/results?search_query={quote(subject)}+educativo+tutorial",
                f"📚 MIT OpenCourseWare: https://ocw.mit.edu/search/?q={quote(subject)}",
                f"🎯 Coursera: https://www.coursera.org/search?query={quote(subject)}",
            ],
            'raw_resources': [],
            'note': 'Recursos fallback - Enlaces de búsqueda para encontrar contenido',
            'error': str(e),
            'generated_at': datetime.utcnow().isoformat() + 'Z',
        }


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
            'synthesis_vocational': 'POST /synthesis/vocational',
            # Analysis endpoints (Punto 3)
            'content_analysis': 'POST /api/analysis/content-check',
            'difficulty_recommendation': 'POST /api/recommendation/difficulty',
            'student_solution_analysis': 'POST /api/analysis/student-solution',
            # Generation endpoints (Punto 6)
            'generate_questions': 'POST /api/generation/questions',
            # Resources endpoint (Punto 7 - NEW)
            'get_resources': 'POST /api/resources',
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

    logger.info("\n" + "=" * 70)
    logger.info("Iniciando servidor Agent (LLM Synthesis) con:")
    logger.info(f"  Host: {HOST}")
    logger.info(f"  Puerto: {PORT}")
    logger.info(f"  Ambiente: {ENVIRONMENT}")
    logger.info("=" * 70 + "\n")

    uvicorn.run(
        "api_server:app",
        host=HOST,
        port=PORT,
        reload=DEBUG,
        log_level=LOG_LEVEL.lower()
    )
