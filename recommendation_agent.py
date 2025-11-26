"""
Agente Inteligente de Recomendaciones Educativas
Usa Groq API con LLaMA para generar recomendaciones personalizadas
"""

import json
import logging
from typing import Dict, List, Optional
from datetime import datetime, timedelta
import hashlib

from langchain_groq import ChatGroq
from langchain_core.messages import HumanMessage

from config import (
    GROQ_API_KEY,
    GROQ_MODEL,
    RECOMMENDATION_TEMPERATURE,
    MAX_TOKENS,
    ENABLE_CACHE,
    CACHE_TTL_MINUTES,
    FALLBACK_RECOMMENDATIONS,
)
from prompts import (
    format_recommendation_prompt,
    format_explanation_prompt,
    format_resources_prompt,
)

# ============================================================
# CONFIGURAR LOGGING
# ============================================================

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# ============================================================
# CACHE SIMPLE EN MEMORIA
# ============================================================

class SimpleCache:
    """Cache simple en memoria para recomendaciones"""

    def __init__(self):
        self.cache: Dict[str, tuple] = {}

    def get(self, key: str) -> Optional[Dict]:
        """Obtener del cache"""
        if key in self.cache:
            value, timestamp = self.cache[key]
            # Verificar si no expiró
            if datetime.now() - timestamp < timedelta(minutes=CACHE_TTL_MINUTES):
                logger.info(f"Cache hit para: {key[:20]}...")
                return value
            else:
                del self.cache[key]
        return None

    def set(self, key: str, value: Dict) -> None:
        """Guardar en cache"""
        self.cache[key] = (value, datetime.now())

    def clear(self) -> None:
        """Limpiar cache"""
        self.cache.clear()


# ============================================================
# AGENTE PRINCIPAL
# ============================================================

class EducationalRecommendationAgent:
    """
    Agente inteligente que genera recomendaciones educativas personalizadas
    usando Groq API
    """

    def __init__(self):
        """Inicializar el agente"""
        try:
            self.llm = ChatGroq(
                model=GROQ_MODEL,
                api_key=GROQ_API_KEY,
                temperature=RECOMMENDATION_TEMPERATURE,
                max_tokens=MAX_TOKENS,
            )
            self.cache = SimpleCache() if ENABLE_CACHE else None
            logger.info(f"✓ Agente inicializado con Groq API")
            logger.info(f"  Modelo: {GROQ_MODEL}")
            logger.info(f"  Temperatura: {RECOMMENDATION_TEMPERATURE}")
        except Exception as e:
            logger.error(f"Error inicializando agente: {str(e)}")
            raise

    def _generate_cache_key(self, student_data: Dict, predictions: Dict) -> str:
        """Generar clave de cache"""
        key_str = f"{student_data.get('student_id')}:{predictions.get('risk_score')}"
        return hashlib.md5(key_str.encode()).hexdigest()

    def generate_recommendations(
        self,
        student_data: Dict,
        predictions: Dict
    ) -> Dict:
        """
        Generar recomendaciones personalizadas

        Args:
            student_data: Datos del estudiante {student_id, name, subject, current_grade, ...}
            predictions: Predicciones de ML {risk_score, risk_level, projected_grade, trend, ...}

        Returns:
            Dict con recomendaciones en formato JSON
        """
        try:
            # Verificar cache
            cache_key = self._generate_cache_key(student_data, predictions)
            if self.cache:
                cached = self.cache.get(cache_key)
                if cached:
                    logger.info(f"Recomendaciones obtenidas del cache")
                    return cached

            logger.info(f"Generando recomendaciones para estudiante {student_data.get('student_id')}")

            # Preparar datos para el prompt
            prompt_data = {
                'student_name': student_data.get('name', 'Estudiante'),
                'subject': student_data.get('subject', 'General'),
                'current_grade': student_data.get('current_grade', 0),
                'previous_average': student_data.get('previous_average', 0),
                'num_grades': student_data.get('num_calificaciones', 0),
                'attempts': student_data.get('num_trabajos', 0),
                'assignments_completed': student_data.get('trabajos_entregados', 0),
                'assignments_total': student_data.get('num_trabajos', 0),
                'delivery_days': student_data.get('dias_promedio_entrega', 0),
                'material_queries': student_data.get('promedio_consultas_material', 0),
                'risk_level': predictions.get('risk_level', 'UNKNOWN'),
                'risk_score': predictions.get('risk_score', 0),
                'projected_grade': predictions.get('projected_grade', 0),
                'trend': predictions.get('trend', 'stable'),
                'confidence': predictions.get('confidence', 0.5),
            }

            # Generar prompt
            prompt_text = format_recommendation_prompt(**prompt_data)

            # Llamar a Groq
            logger.info("Enviando request a Groq API...")
            response = self.llm.invoke([HumanMessage(content=prompt_text)])
            response_text = response.content

            logger.info(f"Respuesta recibida de Groq ({len(response_text)} caracteres)")

            # Parsear JSON
            try:
                # Limpiar respuesta si tiene markdown o prefijos
                if "```json" in response_text:
                    response_text = response_text.split("```json")[1].split("```")[0]
                elif "```" in response_text:
                    response_text = response_text.split("```")[1].split("```")[0]

                recommendations = json.loads(response_text)
                logger.info(f"✓ Recomendaciones generadas exitosamente")

            except json.JSONDecodeError as e:
                logger.warning(f"Error parseando JSON: {str(e)}")
                logger.warning(f"Respuesta: {response_text[:200]}")
                # Usar fallback
                recommendations = self._get_fallback_recommendation(predictions)
                logger.info("Usando recomendación fallback")

            # Agregar metadata
            recommendations['generated_at'] = datetime.now().isoformat()
            recommendations['student_id'] = student_data.get('student_id')
            recommendations['subject'] = student_data.get('subject')

            # Guardar en cache
            if self.cache:
                self.cache.set(cache_key, recommendations)

            return recommendations

        except Exception as e:
            logger.error(f"Error generando recomendaciones: {str(e)}", exc_info=True)
            # Retornar fallback en caso de error
            return self._get_fallback_recommendation(predictions)

    def _get_fallback_recommendation(self, predictions: Dict) -> Dict:
        """Obtener recomendación fallback si Groq falla"""
        risk_level = predictions.get('risk_level', 'MEDIUM').lower()

        if risk_level == 'high':
            fallback = FALLBACK_RECOMMENDATIONS['high_risk']
        elif risk_level == 'low':
            fallback = FALLBACK_RECOMMENDATIONS['low_risk']
        else:
            fallback = FALLBACK_RECOMMENDATIONS['medium_risk']

        return {
            **fallback,
            'fallback': True,
            'message': 'Recomendación generada con lógica predefinida',
            'generated_at': datetime.now().isoformat(),
        }

    def generate_explanation(
        self,
        student_name: str,
        subject: str,
        situation: str,
        recommendation_type: str
    ) -> str:
        """
        Generar explicación personalizada en lenguaje natural

        Args:
            student_name: Nombre del estudiante
            subject: Materia/asignatura
            situation: Descripción de la situación
            recommendation_type: Tipo de recomendación

        Returns:
            String con explicación personalizada
        """
        try:
            logger.info(f"Generando explicación para {student_name}")

            prompt_text = format_explanation_prompt(
                student_name=student_name,
                subject=subject,
                situation=situation,
                recommendation_type=recommendation_type,
            )

            response = self.llm.invoke([HumanMessage(content=prompt_text)])
            explanation = response.content.strip()

            logger.info(f"✓ Explicación generada")
            return explanation

        except Exception as e:
            logger.error(f"Error generando explicación: {str(e)}")
            return f"Se recomienda {recommendation_type} para mejorar en {subject}."

    def get_resources(
        self,
        student_name: str,
        subject: str,
        current_grade: float,
        risk_level: str,
        need: str
    ) -> str:
        """
        Obtener recursos educativos específicos

        Args:
            student_name: Nombre del estudiante
            subject: Materia
            current_grade: Calificación actual
            risk_level: Nivel de riesgo
            need: Tipo de necesidad (tutoring, resources, etc)

        Returns:
            String con lista de recursos
        """
        try:
            logger.info(f"Obteniendo recursos para {student_name} en {subject}")

            prompt_text = format_resources_prompt(
                student_name=student_name,
                subject=subject,
                current_grade=current_grade,
                risk_level=risk_level,
                need=need,
            )

            response = self.llm.invoke([HumanMessage(content=prompt_text)])
            resources = response.content.strip()

            logger.info(f"✓ Recursos obtenidos")
            return resources

        except Exception as e:
            logger.error(f"Error obteniendo recursos: {str(e)}")
            return "Consultar con tutor o usar Khan Academy para práctica adicional."

    def clear_cache(self) -> None:
        """Limpiar cache"""
        if self.cache:
            self.cache.clear()
            logger.info("Cache limpiado")


# ============================================================
# INSTANCIA GLOBAL
# ============================================================

agent = None


def init_agent():
    """Inicializar agente globalmente"""
    global agent
    agent = EducationalRecommendationAgent()
    logger.info("Agente global inicializado")
    return agent


def get_agent() -> EducationalRecommendationAgent:
    """Obtener instancia del agente"""
    global agent
    if agent is None:
        agent = init_agent()
    return agent


if __name__ == "__main__":
    # Test del agente
    logger.info("Iniciando test del agente...")

    agent = init_agent()

    test_student_data = {
        'student_id': 1,
        'name': 'Juan Pérez',
        'subject': 'Matemáticas',
        'current_grade': 6.5,
        'previous_average': 7.5,
        'num_calificaciones': 5,
        'num_trabajos': 10,
        'trabajos_entregados': 8,
        'dias_promedio_entrega': 3.5,
        'promedio_consultas_material': 2.0,
    }

    test_predictions = {
        'risk_score': 0.75,
        'risk_level': 'HIGH',
        'projected_grade': 5.5,
        'trend': 'declining',
        'confidence': 0.85,
    }

    print("\n" + "="*60)
    print("TEST DEL AGENTE DE RECOMENDACIONES")
    print("="*60 + "\n")

    recommendations = agent.generate_recommendations(test_student_data, test_predictions)

    print("RECOMENDACIONES GENERADAS:")
    print(json.dumps(recommendations, indent=2, ensure_ascii=False))

    print("\n" + "="*60)
