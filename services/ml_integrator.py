"""
ML Integrator - Orquesta llamadas a APIs de ML (supervisada y no supervisada)

Este módulo:
- Obtiene predicciones de supervisada (puerto 8001)
- Obtiene clustering de no_supervisado (puerto 8002)
- Agrega resultados en formato unificado
- Maneja errores y fallbacks gracefully
"""

import logging
import requests
import json
from typing import Dict, Any, Optional
from datetime import datetime

try:
    from shared.config import (
        SUPERVISADA_API_URL,
        NO_SUPERVISADA_API_URL,
        ML_API_TIMEOUT
    )
except ImportError:
    # Fallback si shared no está disponible
    SUPERVISADA_API_URL = "http://localhost:8001"
    NO_SUPERVISADA_API_URL = "http://localhost:8002"
    ML_API_TIMEOUT = 10

# ============================================================
# LOGGING
# ============================================================

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class MLIntegrator:
    """
    Integrador de APIs de ML
    Coordina llamadas a supervisada y no_supervisado
    """

    def __init__(self):
        """Inicializar integrador"""
        self.supervisada_url = SUPERVISADA_API_URL
        self.no_supervisada_url = NO_SUPERVISADA_API_URL
        self.timeout = ML_API_TIMEOUT

        logger.info(f"[MLIntegrator] Supervisada API: {self.supervisada_url}")
        logger.info(f"[MLIntegrator] No Supervisada API: {self.no_supervisada_url}")
        logger.info(f"[MLIntegrator] Timeout: {self.timeout}s")

    def get_student_ml_analysis(self, student_id: int) -> Dict[str, Any]:
        """
        Obtener análisis completo de ML para un estudiante

        Integra:
        - Predicciones de supervisada
        - Clustering de no_supervisado
        - Análisis combinado

        Args:
            student_id: ID del estudiante

        Returns:
            Dict con estructura:
            {
                'success': bool,
                'student_id': int,
                'predictions': {...},  # Datos de supervisada
                'discoveries': {...},  # Datos de no_supervisado
                'combined_analysis': {...},
                'timestamp': str,
                'errors': [...]  # Errores si los hay
            }
        """
        logger.info(f"[MLIntegrator] Obteniendo análisis para estudiante {student_id}")

        result = {
            'success': True,
            'student_id': student_id,
            'predictions': {},
            'discoveries': {},
            'combined_analysis': {},
            'timestamp': datetime.utcnow().isoformat() + 'Z',
            'errors': []
        }

        # ====================================================================
        # 1. OBTENER PREDICCIONES DE SUPERVISADA
        # ====================================================================

        try:
            logger.info(f"[Supervisada] Solicitando predicciones para estudiante {student_id}...")
            predictions = self._get_supervised_predictions(student_id)

            if predictions:
                result['predictions'] = predictions
                logger.info(f"[Supervisada] Predicciones obtenidas exitosamente")
            else:
                error_msg = "Supervisada: No se obtuvieron predicciones"
                result['errors'].append(error_msg)
                logger.warning(f"[ERROR] {error_msg}")

        except Exception as e:
            error_msg = f"Supervisada error: {str(e)}"
            result['errors'].append(error_msg)
            result['success'] = False
            logger.error(f"[ERROR] {error_msg}")

        # ====================================================================
        # 2. OBTENER CLUSTERING DE NO SUPERVISADA
        # ====================================================================

        try:
            logger.info(f"[No Supervisada] Solicitando clustering para estudiante {student_id}...")
            discoveries = self._get_unsupervised_discoveries(student_id)

            if discoveries:
                result['discoveries'] = discoveries
                logger.info(f"[No Supervisada] Clustering obtenido exitosamente")
            else:
                error_msg = "No Supervisada: No se obtuvo clustering"
                result['errors'].append(error_msg)
                logger.warning(f"[ERROR] {error_msg}")

        except Exception as e:
            error_msg = f"No Supervisada error: {str(e)}"
            result['errors'].append(error_msg)
            logger.error(f"[ERROR] {error_msg}")

        # ====================================================================
        # 3. COMBINAR ANÁLISIS
        # ====================================================================

        try:
            result['combined_analysis'] = self._combine_analysis(
                result['predictions'],
                result['discoveries']
            )
            logger.info(f"[MLIntegrator] Análisis combinado completado")
        except Exception as e:
            logger.error(f"[ERROR] Error combinando análisis: {str(e)}")
            result['errors'].append(f"Combinación de análisis falló: {str(e)}")

        # ====================================================================
        # 4. VALIDAR RESULTADO
        # ====================================================================

        if not result['predictions'] and not result['discoveries']:
            result['success'] = False
            logger.error(f"[ERROR] No se pudieron obtener datos de ML para estudiante {student_id}")

        logger.info(f"[MLIntegrator] Análisis completo - Success: {result['success']}, Errores: {len(result['errors'])}")

        return result

    def _get_supervised_predictions(self, student_id: int) -> Optional[Dict[str, Any]]:
        """
        Obtener predicciones de supervisada

        Endpoints llamados:
        - POST /predict/performance -> Predicción de calificación
        - POST /predict/career -> Recomendación de carrera
        - POST /predict/trend -> Predicción de tendencia
        - POST /predict/progress -> Análisis de progreso

        Args:
            student_id: ID del estudiante

        Returns:
            Dict con predicciones o None si falla
        """
        try:
            # Health check
            health_url = f"{self.supervisada_url}/health"
            logger.info(f"[Supervisada] Health check: {health_url}")

            response = requests.get(health_url, timeout=self.timeout)
            if response.status_code != 200:
                logger.warning(f"[Supervisada] Health check falló - Status: {response.status_code}")
                return None

            logger.info(f"[Supervisada] Servidor disponible")

            # Obtener predicciones
            payload = {"student_id": student_id}

            predictions = {}

            # 1. Performance Prediction
            try:
                perf_url = f"{self.supervisada_url}/predict/performance"
                logger.info(f"[Supervisada] GET {perf_url}")
                perf_response = requests.post(perf_url, json=payload, timeout=self.timeout)
                if perf_response.status_code == 200:
                    predictions['performance'] = perf_response.json()
                    logger.info(f"[Supervisada] Performance obtenida")
            except Exception as e:
                logger.warning(f"[Supervisada] Performance error: {str(e)}")

            # 2. Career Recommendation
            try:
                career_url = f"{self.supervisada_url}/predict/career"
                logger.info(f"[Supervisada] GET {career_url}")
                career_response = requests.post(career_url, json=payload, timeout=self.timeout)
                if career_response.status_code == 200:
                    predictions['career'] = career_response.json()
                    logger.info(f"[Supervisada] Career obtenida")
            except Exception as e:
                logger.warning(f"[Supervisada] Career error: {str(e)}")

            # 3. Trend Prediction
            try:
                trend_url = f"{self.supervisada_url}/predict/trend"
                logger.info(f"[Supervisada] GET {trend_url}")
                trend_response = requests.post(trend_url, json=payload, timeout=self.timeout)
                if trend_response.status_code == 200:
                    predictions['trend'] = trend_response.json()
                    logger.info(f"[Supervisada] Trend obtenida")
            except Exception as e:
                logger.warning(f"[Supervisada] Trend error: {str(e)}")

            # 4. Progress Analysis
            try:
                progress_url = f"{self.supervisada_url}/predict/progress"
                logger.info(f"[Supervisada] GET {progress_url}")
                progress_response = requests.post(progress_url, json=payload, timeout=self.timeout)
                if progress_response.status_code == 200:
                    predictions['progress'] = progress_response.json()
                    logger.info(f"[Supervisada] Progress obtenida")
            except Exception as e:
                logger.warning(f"[Supervisada] Progress error: {str(e)}")

            return predictions if predictions else None

        except Exception as e:
            logger.error(f"[Supervisada] Error general: {str(e)}")
            return None

    def _get_unsupervised_discoveries(self, student_id: int) -> Optional[Dict[str, Any]]:
        """
        Obtener clustering de no_supervisado

        NOTA: Actualmente desactivado porque el servidor unsupervised no tiene
        endpoint per-student. Los endpoints disponibles requieren arrays de datos.

        Esto se optimizará en versión futura con endpoint /student/{id}/cluster

        Args:
            student_id: ID del estudiante

        Returns:
            None (fallback silencioso para evitar timeouts)
        """
        logger.info(f"[No Supervisada] Clustering per-student no disponible actualmente")
        return None

    def _combine_analysis(self, predictions: Dict, discoveries: Dict) -> Dict[str, Any]:
        """
        Combinar predicciones y descubrimientos en un análisis unificado

        Args:
            predictions: Datos de supervisada
            discoveries: Datos de no_supervisado

        Returns:
            Análisis combinado
        """
        combined = {
            'has_predictions': bool(predictions),
            'has_discoveries': bool(discoveries),
            'summary': {}
        }

        # Summary de predicciones
        if predictions:
            perf = predictions.get('performance', {})
            if perf:
                combined['summary']['performance'] = {
                    'prediction': perf.get('prediction'),
                    'confidence': perf.get('confidence'),
                    'model_used': perf.get('model_used')
                }

            career = predictions.get('career', {})
            if career:
                combined['summary']['career_recommendation'] = career.get('prediction')

            trend = predictions.get('trend', {})
            if trend:
                combined['summary']['academic_trend'] = trend.get('prediction')

        # Summary de descubrimientos
        if discoveries:
            cluster = discoveries.get('cluster_assignment', {})
            if cluster:
                combined['summary']['cluster'] = {
                    'cluster_id': cluster.get('cluster_id'),
                    'cluster_name': cluster.get('cluster_name'),
                    'description': cluster.get('cluster_description')
                }

            analysis = discoveries.get('cluster_analysis', {})
            if analysis:
                combined['summary']['cluster_profiles'] = analysis.get('cluster_profiles')

        logger.info(f"[MLIntegrator] Análisis combinado completado")
        return combined


# ============================================================
# INSTANCIA GLOBAL
# ============================================================

_integrator = None


def get_integrator() -> MLIntegrator:
    """Obtener instancia singleton del integrador"""
    global _integrator
    if _integrator is None:
        _integrator = MLIntegrator()
    return _integrator


if __name__ == "__main__":
    # Test del integrador
    logger.info("=" * 70)
    logger.info("TEST DEL ML INTEGRATOR")
    logger.info("=" * 70)

    integrator = get_integrator()

    # Test con un estudiante
    result = integrator.get_student_ml_analysis(student_id=253)

    print("\n" + "=" * 70)
    print("RESULTADO DEL ANÁLISIS")
    print("=" * 70)
    print(json.dumps(result, indent=2, default=str))
