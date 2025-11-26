"""
Prompts optimizados para Groq/LLaMA
Estos prompts están diseñados para generar recomendaciones educativas
"""

# ============================================================
# PROMPT PRINCIPAL - GENERAR RECOMENDACIONES
# ============================================================

RECOMMENDATION_PROMPT = """Eres un experto en educación personalizada y pedagogía.

Tu tarea: Analizar la situación académica de un estudiante y generar recomendaciones educativas específicas.

=== DATOS DEL ESTUDIANTE ===
Nombre: {student_name}
Materia: {subject}
Calificación actual: {current_grade}/10
Promedio anterior: {previous_average}/10
Número de calificaciones: {num_grades}
Intentos realizados: {attempts}
Trabajos entregados: {assignments_completed}/{assignments_total}
Días promedio de entrega: {delivery_days} días
Consultas de material: {material_queries} veces

=== ANÁLISIS DE MACHINE LEARNING ===
Nivel de Riesgo: {risk_level} (Score: {risk_score:.2f}/1.0)
Calificación Proyectada: {projected_grade:.1f}/10
Tendencia de Desempeño: {trend}
Confianza del Análisis: {confidence:.0%}

=== INSTRUCCIONES ===
Basándote en estos datos, genera recomendaciones en este EXACTO formato JSON:

{{
  "recommendation_type": "study_resource|tutoring|intervention|enrichment",
  "urgency": "immediate|normal|preventive",
  "primary_subject": "{subject}",
  "reason_short": "Breve razón en 1 línea",
  "reason_detailed": "Explicación detallada de por qué se recomienda esto",
  "actions": [
    "Acción 1 específica",
    "Acción 2 específica",
    "Acción 3 específica"
  ],
  "resources": [
    "resource_type: description"
  ],
  "timeline": "Cuando empezar (inmediato/esta semana/preventivo)",
  "success_indicators": [
    "Indicador 1 de éxito",
    "Indicador 2 de éxito"
  ],
  "confidence_level": "alto|medio|bajo"
}}

REGLAS IMPORTANTES:
1. Si Risk Level = HIGH: urgency DEBE ser "immediate"
2. Si Risk Level = LOW y Trend = improving: tipo DEBE ser "enrichment"
3. Las acciones deben ser ESPECÍFICAS y ACCIONABLES
4. Explica en lenguaje claro, no técnico
5. Sé empático pero directo
6. Responde SOLO con JSON válido

Genera la recomendación ahora:"""

# ============================================================
# PROMPT PARA EXPLICACIÓN PERSONALIZADA
# ============================================================

EXPLANATION_PROMPT = """Como experto educativo, escribe una explicación personalizada y motivadora para este estudiante.

Estudiante: {student_name}
Materia: {subject}
Situación: {situation}
Recomendación: {recommendation_type}

Escribe un párrafo (3-4 oraciones) que:
1. Reconozca su situación actual
2. Explique por qué necesita esta recomendación
3. Sea motivador y positivo
4. Indique el beneficio esperado

Tono: Amable, profesional, motivador (como un tutor comprensivo)

Escribe SOLO el párrafo sin prefijos:"""

# ============================================================
# PROMPT PARA ANÁLISIS DE PROGRESO
# ============================================================

PROGRESS_ANALYSIS_PROMPT = """Analiza el progreso académico de este estudiante:

Estudiante: {student_name}
Materia: {subject}
Calificaciones anteriores: {grade_history}
Calificación actual: {current_grade}
Proyección: {projected_grade}
Tendencia: {trend}

Proporciona:
1. Análisis del patrón de calificaciones (1-2 oraciones)
2. Factores positivos identificados (lista)
3. Áreas de mejora (lista)
4. Perspectiva hacia adelante (1 oración)

Formato:
ANÁLISIS:
[Tu análisis aquí]

FORTALEZAS:
- Punto 1
- Punto 2

ÁREAS DE MEJORA:
- Punto 1
- Punto 2

PERSPECTIVA:
[Tu perspectiva aquí]"""

# ============================================================
# PROMPT PARA RECURSOS ESPECÍFICOS
# ============================================================

RESOURCES_PROMPT = """Como especialista educativo, sugiere recursos específicos para este estudiante.

Estudiante: {student_name}
Materia: {subject}
Calificación: {current_grade}
Nivel de Riesgo: {risk_level}
Necesidad: {need}

Sugiere 3-4 recursos concretos:
Formato:
TIPO | DESCRIPCIÓN | DURACIÓN | DIFICULTAD | ENLACE/PROVEEDOR

Considera:
- Khan Academy (videos gratuitos)
- YouTube educativo
- Plataformas de ejercicios
- Tutoría
- Grupos de estudio

Prioriza recursos GRATUITOS o BAJO COSTO.
Sé específico con títulos y temas exactos."""

# ============================================================
# PROMPTS CORTOS PARA ANÁLISIS RÁPIDO
# ============================================================

QUICK_ANALYSIS = """Análisis rápido en 1 línea:
Estudiante {student_name}, calificación {current_grade}, riesgo {risk_level}.
¿Qué necesita? Responde en 1 oración corta."""

RISK_ASSESSMENT = """¿Es este nivel de riesgo correcto?
Estudiante: {student_name}
Score: {risk_score}
Level: {risk_level}
Responde: CORRECTO o REVISAR
Si revisar, explica por qué en 1 oración."""

# ============================================================
# UTILIDADES
# ============================================================

def format_recommendation_prompt(**kwargs):
    """Formatea el prompt principal con datos específicos"""
    return RECOMMENDATION_PROMPT.format(**kwargs)

def format_explanation_prompt(**kwargs):
    """Formatea el prompt de explicación"""
    return EXPLANATION_PROMPT.format(**kwargs)

def format_progress_prompt(**kwargs):
    """Formatea el prompt de progreso"""
    return PROGRESS_ANALYSIS_PROMPT.format(**kwargs)

def format_resources_prompt(**kwargs):
    """Formatea el prompt de recursos"""
    return RESOURCES_PROMPT.format(**kwargs)
