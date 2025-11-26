"""
Task Enhancement Agent
======================

Análisis y recomendaciones inteligentes para mejora de tareas/evaluaciones
Utilizando LLM Groq para análisis de contenido y recomendaciones basadas en datos.

Endpoints:
- POST /api/analysis/content-check - Analiza claridad y calidad pedagógica
- POST /api/recommendation/difficulty - Recomienda dificultad basada en datos
- GET /health - Health check
"""

import os
import json
import logging
from typing import Optional, List, Dict, Any
from datetime import datetime

from fastapi import HTTPException
from pydantic import BaseModel, Field
from langchain_groq import ChatGroq
from dotenv import load_dotenv

# Cargar variables de entorno
load_dotenv()

# Configurar logging
logger = logging.getLogger(__name__)

# ============================================================
# MODELOS PYDANTIC
# ============================================================


class Issue(BaseModel):
    """Problema identificado en el contenido"""
    type: str = Field(..., description="Tipo de problema: ambiguity, clarity, grammar, bias, etc")
    severity: str = Field(..., description="Severidad: critical, warning, info")
    text: str = Field(..., description="Texto específico problemático")
    explanation: str = Field(..., description="Explicación de por qué es problema")
    suggestion: str = Field(..., description="Sugerencia para arreglarlo")


class ContentAnalysisRequest(BaseModel):
    """Solicitud de análisis de contenido"""
    content: str = Field(..., description="Instrucción o pregunta a analizar")
    task_type: str = Field(..., description="Tipo: 'tarea', 'evaluacion', 'pregunta'")
    course_context: Dict[str, str] = Field(
        ...,
        description="Contexto del curso: nombre, nivel, tema"
    )
    student_data: Optional[Dict[str, Any]] = Field(
        None,
        description="Datos de estudiantes: count, avg_score, std_dev"
    )


class ContentAnalysisResponse(BaseModel):
    """Respuesta del análisis de contenido"""
    clarity_score: float = Field(..., description="0-1: qué tan claro")
    is_clear: bool = Field(..., description="¿Es claramente comprensible?")
    issues: List[Issue] = Field(..., description="Problemas identificados")
    bloom_level: str = Field(..., description="Nivel Bloom: remember-understand-apply-analyze-evaluate-create")
    estimated_difficulty: float = Field(..., description="0-1: dificultad estimada")
    estimated_time_minutes: int = Field(..., description="Minutos estimados para completar")
    prerequisites: List[str] = Field(..., description="Conceptos prerequisitos necesarios")
    strengths: List[str] = Field(..., description="Puntos fuertes de la tarea")
    recommendations: List[str] = Field(..., description="Recomendaciones de mejora")


class DifficultyRecommendationRequest(BaseModel):
    """Solicitud de recomendación de dificultad"""
    course_id: int
    task_context: Dict[str, Any] = Field(..., description="Contexto de la tarea")
    student_stats: Optional[Dict[str, float]] = Field(
        None,
        description="Estadísticas de estudiantes: avg_score, std_dev"
    )


class DifficultyRecommendation(BaseModel):
    """Recomendación de dificultad"""
    recommended_points: int = Field(..., description="Puntos recomendados (0-100)")
    points_range: tuple = Field(..., description="Rango recomendado (min, max)")
    estimated_time_minutes: int = Field(..., description="Tiempo estimado para completar")
    expected_pass_rate: float = Field(..., description="0-1: porcentaje esperado de aprobados")
    difficulty_level: str = Field(..., description="easy, medium, hard")
    reasoning: str = Field(..., description="Explicación del razonamiento")


# ============================================================
# MODELOS PARA ANÁLISIS DE SOLUCIONES DE ESTUDIANTES
# ============================================================

class StudentSolutionAnalysisRequest(BaseModel):
    """Solicitud de análisis de solución de estudiante"""
    task_id: int = Field(..., description="ID de la tarea")
    student_id: int = Field(..., description="ID del estudiante")
    solution_code: str = Field(..., description="Código de la solución a analizar")
    task_type: str = Field(..., description="Tipo: 'tarea', 'evaluacion'")
    language: str = Field(default="auto", description="Lenguaje: python, js, java, etc")


class ConceptoCorreto(BaseModel):
    """Concepto que el estudiante entendió correctamente"""
    concepto: str
    evidencia: str
    nivel: Optional[str] = "intermedio"


class ConceptoIncompleto(BaseModel):
    """Concepto que está incompleto"""
    concepto: str
    que_falta: str
    pregunta_reflexiva: str


class ErrorDetectado(BaseModel):
    """Error o problema en el código"""
    tipo: str = Field(..., description="lógica|rendimiento|estilo|seguridad")
    donde: str = Field(..., description="Ubicación aproximada")
    problema_descripcion: str
    pregunta_guia: str
    pista: str


class AspectoPorValidar(BaseModel):
    """Aspecto que el estudiante debería validar"""
    aspecto: str
    pregunta: str
    caso_especial: Optional[str] = None


class StudentSolutionFeedback(BaseModel):
    """Feedback completo para la solución"""
    conceptos_correctos: List[ConceptoCorreto]
    conceptos_incompletos: List[ConceptoIncompleto]
    errores_detectados: List[ErrorDetectado]
    aspectos_a_validar: List[AspectoPorValidar]
    pistas_progresivas: List[str]
    cosas_bien_hechas: List[str]
    siguiente_paso: str


class StudentSolutionAnalysisResponse(BaseModel):
    """Respuesta del análisis de solución"""
    id: str = Field(default_factory=lambda: str(__import__('uuid').uuid4()))
    analysis_count: int = Field(..., description="Número de análisis para esta tarea")
    max_analyses: int = Field(default=5)
    feedback: StudentSolutionFeedback
    metadata: Dict[str, Any] = Field(
        default_factory=lambda: {
            "analysis_duration_ms": 0,
            "timestamp": datetime.utcnow().isoformat() + 'Z'
        }
    )


# ============================================================
# MODELOS PARA ANÁLISIS DE TÍTULOS
# ============================================================

class TitleAnalysisRequest(BaseModel):
    """Solicitud de análisis de título para tareas/evaluaciones/recursos"""
    titulo: str = Field(..., description="Título a analizar")
    content_type: str = Field(..., description="Tipo: 'tarea', 'evaluacion', 'recurso'")
    course_context: Dict[str, Any] = Field(..., description="Contexto del curso")


class TitleAnalysisResponse(BaseModel):
    """Respuesta del análisis de título con sugerencias"""
    success: bool = True
    titulo_original: str
    content_type: str
    descripcion: str = Field(..., description="Descripción sugerida de la tarea/evaluación")
    instrucciones_plantilla: Optional[str] = Field(None, description="Plantilla de instrucciones")
    tiempo_limite: Optional[int] = Field(None, description="Tiempo límite sugerido en minutos")
    puntuacion_sugerida: Optional[int] = Field(None, description="Puntuación sugerida")
    dificultad: str = Field(..., description="easy|medium|hard")
    nivel_bloom: str = Field(..., description="Nivel cognitivo: remember-understand-apply-analyze-evaluate-create")
    observaciones_pedagogicas: List[str] = Field(..., description="Observaciones sobre la pedagogía")
    conceptos: List[str] = Field(..., description="Conceptos clave a enseñar/evaluar")
    confidence: float = Field(..., description="0-1: confianza del análisis")
    timestamp: str = Field(default_factory=lambda: datetime.utcnow().isoformat() + 'Z')


# ============================================================
# TASK ENHANCEMENT AGENT
# ============================================================


class TaskEnhancementAgent:
    """Agente inteligente para mejora de tareas usando LLM"""

    def __init__(self):
        """Inicializar agente con LLM Groq"""
        api_key = os.getenv('GROQ_API_KEY')
        if not api_key:
            logger.error("GROQ_API_KEY no está configurada")
            raise ValueError("GROQ_API_KEY no está configurada en .env")

        self.llm = ChatGroq(
            groq_api_key=api_key,
            model_name=os.getenv('GROQ_MODEL', 'llama-3.3-70b-versatile'),
            temperature=float(os.getenv('GROQ_TEMPERATURE', '0.3')),
            max_tokens=2048,
            timeout=30
        )

        logger.info(f"✅ TaskEnhancementAgent inicializado con modelo {self.llm.model_name}")

    def analyze_content_clarity(
        self,
        request: ContentAnalysisRequest
    ) -> ContentAnalysisResponse:
        """
        Analiza claridad y calidad pedagógica del contenido

        Args:
            request: ContentAnalysisRequest con el contenido a analizar

        Returns:
            ContentAnalysisResponse con análisis detallado
        """

        logger.info(f"Analizando contenido tipo {request.task_type}...")

        # Construir prompt detallado
        prompt = f"""Eres un experto educativo analizando la claridad y calidad pedagógica de una tarea.

CONTEXTO DEL CURSO:
- Nombre: {request.course_context.get('nombre', 'Desconocido')}
- Nivel: {request.course_context.get('nivel', 'basico')}
- Tema: {request.course_context.get('tema', 'General')}
{f"- Total estudiantes: {request.student_data.get('count', '?')}" if request.student_data else ""}
{f"- Promedio histórico: {request.student_data.get('avg_score', '?')}" if request.student_data else ""}

TIPO DE CONTENIDO: {request.task_type}

CONTENIDO A ANALIZAR:
"{request.content}"

ANÁLISIS REQUERIDO:

1. CLARIDAD (0-100): ¿Qué tan claramente se entiende el contenido?
2. PROBLEMAS: Identifica:
   - Ambigüedad: ¿Palabras que pueden tener múltiples interpretaciones?
   - Claridad: ¿Frases confusas o mal estructuradas?
   - Gramática: ¿Errores gramaticales u ortográficos?
   - Prejuicio: ¿Contenido discriminatorio o sesgado?
   - Vaguedad: ¿Términos no definidos claramente?

3. NIVEL BLOOM: ¿Qué nivel cognitivo requiere?
   - remember: Recordar hechos básicos
   - understand: Comprender conceptos
   - apply: Aplicar a nuevas situaciones
   - analyze: Desglosar en partes
   - evaluate: Juzgar crítica mente
   - create: Crear algo nuevo

4. DIFICULTAD (0-1): ¿Cuán difícil es relativamente?
5. TIEMPO: ¿Minutos estimados para completar?
6. PREREQUISITOS: ¿Qué necesitan saber antes?
7. FORTALEZAS: ¿Qué está bien hecho?
8. RECOMENDACIONES: ¿Cómo mejorar?

RESPONDE EXACTAMENTE CON ESTE JSON (sin markdown):
{{
    "clarity_score": 85,
    "issues": [
        {{
            "type": "ambiguity|clarity|grammar|bias|vagueness",
            "severity": "critical|warning|info",
            "text": "texto específico problemático",
            "explanation": "por qué es problema",
            "suggestion": "cómo arreglarlo"
        }}
    ],
    "bloom_level": "remember|understand|apply|analyze|evaluate|create",
    "estimated_difficulty": 0.65,
    "estimated_time_minutes": 45,
    "prerequisites": ["concepto1", "concepto2"],
    "strengths": ["fortaleza1", "fortaleza2"],
    "recommendations": ["recomendación1", "recomendación2"]
}}"""

        try:
            # Llamar a LLM
            response = self.llm.invoke(prompt)
            content = response.content.strip()

            # Limpiar markdown si existe
            if content.startswith('```json'):
                content = content[7:]
            if content.endswith('```'):
                content = content[:-3]
            content = content.strip()

            logger.debug(f"Respuesta LLM: {content[:200]}...")

            # Parse JSON
            analysis_data = json.loads(content)

            # Convertir a respuesta tipada
            return ContentAnalysisResponse(
                clarity_score=min(analysis_data.get('clarity_score', 70) / 100, 1.0),
                is_clear=analysis_data.get('clarity_score', 70) >= 75,
                issues=[
                    Issue(**issue)
                    for issue in analysis_data.get('issues', [])
                ],
                bloom_level=analysis_data.get('bloom_level', 'understand'),
                estimated_difficulty=min(analysis_data.get('estimated_difficulty', 0.5), 1.0),
                estimated_time_minutes=max(analysis_data.get('estimated_time_minutes', 30), 5),
                prerequisites=analysis_data.get('prerequisites', []),
                strengths=analysis_data.get('strengths', []),
                recommendations=analysis_data.get('recommendations', [])
            )

        except json.JSONDecodeError as e:
            logger.error(f"Error parseando JSON de LLM: {str(e)}")
            # Retornar análisis por defecto
            return ContentAnalysisResponse(
                clarity_score=0.7,
                is_clear=True,
                issues=[],
                bloom_level="understand",
                estimated_difficulty=0.5,
                estimated_time_minutes=45,
                prerequisites=[],
                strengths=["Contenido bien estructurado"],
                recommendations=["Revisar con expertos del dominio"]
            )

    def recommend_difficulty(
        self,
        course_id: int,
        task_context: Dict[str, Any],
        student_stats: Optional[Dict[str, float]] = None
    ) -> DifficultyRecommendation:
        """
        Recomienda dificultad y puntuación basada en contexto y datos históricos

        Args:
            course_id: ID del curso
            task_context: Contexto de la tarea (tipo, complejidad, etc)
            student_stats: Estadísticas de estudiantes (promedio, desv. est.)

        Returns:
            DifficultyRecommendation con sugerencias
        """

        logger.info(f"Recomendando dificultad para curso {course_id}...")

        # Valores por defecto si no hay datos
        if not student_stats:
            student_stats = {
                'avg_score': 75,
                'std_dev': 15,
                'count': 0
            }

        task_complexity = task_context.get('complexity', 0.5)
        task_type = task_context.get('type', 'tarea')

        # Construcción del prompt
        prompt = f"""Eres un experto en diseño curricular y evaluación educativa.
Tu tarea es recomendar la dificultad y puntuación para una nueva tarea.

CONTEXTO DE LA TAREA:
- Tipo: {task_type}
- Complejidad estimada: {task_complexity}/1.0 (0=trivial, 1=muy difícil)
- Título: {task_context.get('title', 'Sin título')}

DATOS HISTÓRICOS DEL CURSO:
- Promedio estudiantes: {student_stats.get('avg_score', 75):.1f} puntos
- Desv. estándar: {student_stats.get('std_dev', 15):.1f}
- Total estudiantes: {student_stats.get('count', 0)}

OBJETIVO PEDAGÓGICO:
- 60-70% de estudiantes deberían PASAR (nota ≥ 60)
- 20-30% deberían obtener BUENA nota (≥ 80)
- 10% EXCELENTE (≥ 95)

Esto significa una distribución aproximadamente normal con media ≈ 75.

CONSIDERACIONES:
- Si complejidad es baja: más puntos base, tarea más fácil
- Si complejidad es alta: menos puntos base, pero más desafiante
- Rango recomendado para puntuación: 50-100

DEVUELVE EXACTAMENTE ESTE JSON (sin markdown):
{{
    "recommended_points": 85,
    "points_range": [70, 95],
    "estimated_time_minutes": 60,
    "expected_pass_rate": 0.67,
    "difficulty_level": "medium",
    "reasoning": "Basado en el promedio histórico del curso ({student_stats.get('avg_score', 75):.0f}), recomiendo..."
}}"""

        try:
            response = self.llm.invoke(prompt)
            content = response.content.strip()

            # Limpiar markdown si existe
            if content.startswith('```json'):
                content = content[7:]
            if content.endswith('```'):
                content = content[:-3]
            content = content.strip()

            logger.debug(f"Respuesta LLM dificultad: {content[:200]}...")

            # Parse JSON
            data = json.loads(content)

            points_range = tuple(data.get('points_range', [70, 90]))

            return DifficultyRecommendation(
                recommended_points=int(data.get('recommended_points', 85)),
                points_range=points_range,
                estimated_time_minutes=int(data.get('estimated_time_minutes', 60)),
                expected_pass_rate=float(data.get('expected_pass_rate', 0.65)),
                difficulty_level=data.get('difficulty_level', 'medium'),
                reasoning=data.get('reasoning', 'Recomendación basada en contexto del curso')
            )

        except (json.JSONDecodeError, KeyError) as e:
            logger.error(f"Error en recomendación de dificultad: {str(e)}")
            # Retornar recomendación por defecto
            avg_score = student_stats.get('avg_score', 75)
            return DifficultyRecommendation(
                recommended_points=int(avg_score * 0.9),
                points_range=(int(avg_score * 0.75), int(avg_score * 1.1)),
                estimated_time_minutes=60,
                expected_pass_rate=0.65,
                difficulty_level='medium',
                reasoning=f"Basado en promedio histórico de {avg_score:.0f}"
            )

    def analyze_student_solution(
        self,
        request: StudentSolutionAnalysisRequest
    ) -> StudentSolutionAnalysisResponse:
        """
        Analiza la solución de un estudiante usando método socrático.

        IMPORTANTE: NUNCA da la solución directa, solo guía con preguntas.

        Args:
            request: Datos de la solución a analizar

        Returns:
            Feedback sin respuestas directas
        """
        import time
        start_time = time.time()

        try:
            # Validar entrada
            if not request.solution_code or len(request.solution_code.strip()) < 10:
                raise ValueError("El código debe tener al menos 10 caracteres")

            # Construir el prompt (MÉTODO SOCRÁTICO)
            prompt = self._build_student_feedback_prompt(request)

            logger.info(f"Analizando solución de estudiante {request.student_id}...")

            # Llamar LLM
            response = self.llm.invoke(prompt)

            # Procesar respuesta
            content = response.content.strip()

            # Limpiar markdown si está presente
            if content.startswith('```json'):
                content = content[7:]
            if content.endswith('```'):
                content = content[:-3]
            content = content.strip()

            # Parse JSON
            feedback_data = json.loads(content)

            # Construir respuesta tipada
            duration_ms = int((time.time() - start_time) * 1000)

            return StudentSolutionAnalysisResponse(
                analysis_count=1,  # Este número lo maneja el backend
                max_analyses=5,
                feedback=StudentSolutionFeedback(
                    conceptos_correctos=[
                        ConceptoCorreto(**c) for c in feedback_data.get('conceptos_correctos', [])
                    ],
                    conceptos_incompletos=[
                        ConceptoIncompleto(**c) for c in feedback_data.get('conceptos_incompletos', [])
                    ],
                    errores_detectados=[
                        ErrorDetectado(**e) for e in feedback_data.get('errores_detectados', [])
                    ],
                    aspectos_a_validar=[
                        AspectoPorValidar(**a) for a in feedback_data.get('aspectos_a_validar', [])
                    ],
                    pistas_progresivas=feedback_data.get('pistas_progresivas', []),
                    cosas_bien_hechas=feedback_data.get('cosas_bien_hechas', []),
                    siguiente_paso=feedback_data.get('siguiente_paso', 'Continúa iterando')
                ),
                metadata={
                    'analysis_duration_ms': duration_ms,
                    'timestamp': datetime.utcnow().isoformat() + 'Z',
                    'llm_model_used': 'llama-3.3-70b-versatile',
                    'task_type': request.task_type,
                    'language': request.language
                }
            )

        except json.JSONDecodeError as e:
            logger.error(f"Error parseando respuesta LLM: {str(e)}")
            # Retornar feedback genérico
            return StudentSolutionAnalysisResponse(
                analysis_count=1,
                feedback=StudentSolutionFeedback(
                    conceptos_correctos=[ConceptoCorreto(
                        concepto="Estructura básica",
                        evidencia="Detectamos código válido",
                        nivel="básico"
                    )],
                    conceptos_incompletos=[],
                    errores_detectados=[],
                    aspectos_a_validar=[AspectoPorValidar(
                        aspecto="Validación",
                        pregunta="¿Validaste todos los casos especiales?"
                    )],
                    pistas_progresivas=[
                        "💡 Piensa en casos especiales",
                        "💡 ¿Qué pasa con entrada negativa?",
                        "💡 ¿Y con entrada vacía?",
                    ],
                    cosas_bien_hechas=["Código bien formateado"],
                    siguiente_paso="Revisa con un compañero"
                )
            )

        except Exception as e:
            logger.error(f"Error analizando solución: {str(e)}")
            raise

    def analyze_task_title(
        self,
        request: TitleAnalysisRequest
    ) -> TitleAnalysisResponse:
        """
        Analiza un título de tarea y genera sugerencias para completar la tarea

        Args:
            request: TitleAnalysisRequest con el título y contexto

        Returns:
            TitleAnalysisResponse con sugerencias
        """
        logger.info(f"Analizando título de tarea: {request.titulo[:50]}...")

        return self._analyze_title(request, 'tarea')

    def analyze_evaluation_title(
        self,
        request: TitleAnalysisRequest
    ) -> TitleAnalysisResponse:
        """
        Analiza un título de evaluación y genera sugerencias

        Args:
            request: TitleAnalysisRequest con el título y contexto

        Returns:
            TitleAnalysisResponse con sugerencias
        """
        logger.info(f"Analizando título de evaluación: {request.titulo[:50]}...")

        return self._analyze_title(request, 'evaluacion')

    def analyze_resource_title(
        self,
        request: TitleAnalysisRequest
    ) -> TitleAnalysisResponse:
        """
        Analiza un título de recurso y genera sugerencias

        Args:
            request: TitleAnalysisRequest con el título y contexto

        Returns:
            TitleAnalysisResponse con sugerencias
        """
        logger.info(f"Analizando título de recurso: {request.titulo[:50]}...")

        return self._analyze_title(request, 'recurso')

    def _analyze_title(
        self,
        request: TitleAnalysisRequest,
        content_type: str
    ) -> TitleAnalysisResponse:
        """
        Análisis genérico de título que se reutiliza para diferentes tipos de contenido

        Args:
            request: TitleAnalysisRequest con el título y contexto
            content_type: 'tarea', 'evaluacion', o 'recurso'

        Returns:
            TitleAnalysisResponse con análisis y sugerencias
        """
        try:
            # Construir prompt específico para el tipo de contenido
            prompt = self._build_title_analysis_prompt(request, content_type)

            # Llamar a LLM
            response = self.llm.invoke(prompt)
            content = response.content.strip()

            # Limpiar markdown si existe
            if content.startswith('```json'):
                content = content[7:]
            if content.endswith('```'):
                content = content[:-3]
            content = content.strip()

            logger.debug(f"Respuesta LLM análisis de título: {content[:200]}...")

            # Parse JSON
            analysis_data = json.loads(content)

            # Convertir a respuesta tipada
            return TitleAnalysisResponse(
                success=True,
                titulo_original=request.titulo,
                content_type=content_type,
                descripcion=analysis_data.get('descripcion', ''),
                instrucciones_plantilla=analysis_data.get('instrucciones_plantilla'),
                tiempo_limite=analysis_data.get('tiempo_limite'),
                puntuacion_sugerida=analysis_data.get('puntuacion_sugerida'),
                dificultad=analysis_data.get('dificultad', 'medium'),
                nivel_bloom=analysis_data.get('nivel_bloom', 'understand'),
                observaciones_pedagogicas=analysis_data.get('observaciones_pedagogicas', []),
                conceptos=analysis_data.get('conceptos', []),
                confidence=analysis_data.get('confidence', 0.75)
            )

        except json.JSONDecodeError as e:
            logger.error(f"Error parseando JSON de análisis de título: {str(e)}")
            # Retornar respuesta por defecto
            return TitleAnalysisResponse(
                success=True,
                titulo_original=request.titulo,
                content_type=content_type,
                descripcion=f"Tarea basada en: {request.titulo}",
                dificultad='medium',
                nivel_bloom='understand',
                observaciones_pedagogicas=['Revisar con expertos del dominio'],
                conceptos=['Conceptos principales del título'],
                confidence=0.5
            )

        except Exception as e:
            logger.error(f"Error analizando título: {str(e)}")
            raise

    def _build_title_analysis_prompt(
        self,
        request: TitleAnalysisRequest,
        content_type: str
    ) -> str:
        """
        Construye el prompt para análisis de título

        Args:
            request: TitleAnalysisRequest
            content_type: 'tarea', 'evaluacion', o 'recurso'

        Returns:
            Prompt para LLM
        """

        tipo_content = {
            'tarea': 'tarea educativa',
            'evaluacion': 'evaluación o examen',
            'recurso': 'recurso de aprendizaje'
        }.get(content_type, 'contenido educativo')

        curso_nombre = request.course_context.get('nombre', 'Sin especificar')
        curso_nivel = request.course_context.get('nivel', 'intermedio')

        return f"""Eres un experto en diseño pedagógico y educación.
Tu tarea es analizar el siguiente título de {tipo_content} y generar sugerencias completas para crear contenido educativo de calidad.

CONTEXTO DEL CURSO:
- Nombre: {curso_nombre}
- Nivel: {curso_nivel}

TÍTULO A ANALIZAR:
"{request.titulo}"

TIPO DE CONTENIDO: {content_type}

Tu tarea es generar:

1. **DESCRIPCIÓN**: Una descripción clara y pedagógica del {tipo_content}
2. **INSTRUCCIONES (si aplica)**: Una plantilla de instrucciones detalladas
3. **TIEMPO LÍMITE**: Tiempo estimado en minutos para completar
4. **PUNTUACIÓN**: Puntuación sugerida (100 máximo)
5. **DIFICULTAD**: easy, medium, o hard
6. **NIVEL BLOOM**: Nivel cognitivo de Bloom (remember, understand, apply, analyze, evaluate, create)
7. **OBSERVACIONES PEDAGÓGICAS**: 3-5 observaciones sobre cómo enseñar/evaluar esto
8. **CONCEPTOS CLAVE**: 3-5 conceptos principales a enseñar/evaluar
9. **CONFIANZA**: 0-1, qué tan confiable es este análisis (0.7-0.9 típicamente)

RESPONDE EXACTAMENTE CON ESTE JSON (sin markdown):
{{
    "descripcion": "Descripción clara y detallada de la {tipo_content}",
    "instrucciones_plantilla": "Plantilla de instrucciones paso a paso (si aplica)",
    "tiempo_limite": 45,
    "puntuacion_sugerida": 100,
    "dificultad": "medium",
    "nivel_bloom": "apply",
    "observaciones_pedagogicas": [
        "Observación pedagógica 1",
        "Observación pedagógica 2",
        "Observación pedagógica 3"
    ],
    "conceptos": [
        "Concepto clave 1",
        "Concepto clave 2",
        "Concepto clave 3"
    ],
    "confidence": 0.85
}}

IMPORTANTE:
- Sé específico y detallado
- Crea contenido pedagógicamente sólido
- Usa lenguaje educativo
- Sugiere actividades que promuevan el aprendizaje activo
- La descripción debe ser clara para estudiantes de nivel {curso_nivel}
"""

    def _build_student_feedback_prompt(
        self,
        request: StudentSolutionAnalysisRequest
    ) -> str:
        """
        Construye el prompt para análisis de solución usando MÉTODO SOCRÁTICO.

        CRÍTICO: El prompt DEBE:
        - NO resolver el problema
        - NO dar código compilable
        - SÍ hacer preguntas reflexivas
        - SÍ señalar problemas
        - SÍ guiar sin responder
        """

        return f"""
Eres un profesor de programación usando el Método Socrático.
Tu objetivo: hacer que el estudiante PIENSE, no darle respuestas directas.

CONTEXTO:
- Tipo de tarea: {request.task_type}
- Lenguaje: {request.language}
- Estudiante ID: {request.student_id}

CÓDIGO DEL ESTUDIANTE:
```{request.language if request.language != 'auto' else 'code'}
{request.solution_code}
```

ANÁLISIS REQUERIDO (formato JSON):

Tu tarea es analizar este código y proporcionar feedback usando el Método Socrático.

REGLAS ESTRICTAS:
1. NUNCA escribas código compilable que resuelva el problema
2. NUNCA des instrucciones paso a paso
3. NUNCA especifiques "cambia X por Y"
4. NUNCA hagas la tarea por el estudiante
5. SIEMPRE haz preguntas reflexivas
6. SIEMPRE sé constructivo y alentador

ESTRUCTURA DE RESPUESTA (JSON válido):

{{
  "conceptos_correctos": [
    {{
      "concepto": "nombre del concepto bien entendido",
      "evidencia": "prueba de dónde en el código lo vemos",
      "nivel": "básico|intermedio|avanzado"
    }}
  ],

  "conceptos_incompletos": [
    {{
      "concepto": "nombre del concepto",
      "que_falta": "qué parte falta o está incompleta",
      "pregunta_reflexiva": "¿Pregunta que lo haga pensar?"
    }}
  ],

  "errores_detectados": [
    {{
      "tipo": "lógica|rendimiento|estilo|seguridad",
      "donde": "ubicación aproximada (línea X-Y o sección)",
      "problema_descripcion": "descripción del problema SIN cómo resolverlo",
      "pregunta_guia": "pregunta reflexiva sobre el problema",
      "pista": "💡 Una pista sin resolver"
    }}
  ],

  "aspectos_a_validar": [
    {{
      "aspecto": "validación|documentación|robustez|etc",
      "pregunta": "¿Pregunta reflexiva sobre este aspecto?",
      "caso_especial": "Ejemplo: ¿y si el input está vacío?"
    }}
  ],

  "pistas_progresivas": [
    "💡 Pista 1 (general, abierta)",
    "💡 Pista 2 (más específica)",
    "💡 Pista 3 (aún más específica)",
    "💡 Pista 4 (casi un hint, pero no la solución)",
    "💡 Pista 5 (reflexión final)"
  ],

  "cosas_bien_hechas": [
    "Lo que el estudiante hizo bien",
    "Otro punto positivo",
    "Etc"
  ],

  "siguiente_paso": "Una sola acción enfocada para el estudiante"
}}

EJEMPLOS DE QUÉ HACER / QUÉ NO HACER:

❌ MALO:
"Tu función está mal. Usa un bucle for como esto: for i in range(n):"

✅ BUENO:
"Tu función no maneja casos especiales. ¿Qué sucede si alguien pasa un número negativo?"

❌ MALO:
"Deberías agregar validación así: if n < 0: raise ValueError(...)"

✅ BUENO:
"¿Cómo maneja tu código entrada inválida? ¿Qué debería pasar?"

TONO Y ESTILO:
- Profesional pero amable
- Desafiante pero no frustrante
- Educativo, nunca condescendiente
- Usa ejemplos del código EXISTENTE del estudiante
- Guía, nunca resuelves

Ahora analiza el código proporcionado y responde SOLO en formato JSON válido.
No incluyas texto adicional, solo el JSON.
"""


# ============================================================
# INSTANCIA GLOBAL
# ============================================================

try:
    agent = TaskEnhancementAgent()
except Exception as e:
    logger.error(f"Error inicializando TaskEnhancementAgent: {str(e)}")
    agent = None
