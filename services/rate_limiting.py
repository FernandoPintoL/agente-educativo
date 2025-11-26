"""
Rate Limiting Service
=====================

Servicio para implementar rate limiting en análisis de soluciones.

Funciones:
  - check_analysis_quota(): Verifica máximo 5 análisis por tarea
  - check_cooldown(): Verifica 5 minutos entre análisis
  - record_analysis(): Registra análisis en base de datos
  - detect_abuse_pattern(): Detecta patrones sospechosos
  - get_usage_stats(): Obtiene estadísticas de uso
"""

import json
import logging
from datetime import datetime, timedelta
from typing import Tuple, Dict, Any, Optional
from sqlalchemy.orm import Session
from sqlalchemy import func

# Importar modelo
try:
    from models.student_solution_analysis import StudentSolutionAnalysis
except ImportError:
    # Fallback si la importación relativa falla
    from agente.models.student_solution_analysis import StudentSolutionAnalysis

logger = logging.getLogger(__name__)

# ============================================================
# CONSTANTES
# ============================================================

MAX_ANALYSES_PER_TASK = 5
COOLDOWN_MINUTES = 5
ABUSE_THRESHOLD_24H = 50  # Máximo de análisis en 24 horas
ABUSE_THRESHOLD_SINGLE = 10  # Máximo por tarea individual


# ============================================================
# FUNCIONES PRINCIPALES
# ============================================================


def check_analysis_quota(
    db: Session,
    student_id: int,
    task_id: int,
    max_analyses: int = MAX_ANALYSES_PER_TASK
) -> Tuple[bool, str, int]:
    """
    Verifica si el estudiante puede hacer otro análisis (quota).

    Cuenta los análisis de esta tarea por el estudiante y verifica
    si ha llegado al máximo permitido.

    Args:
        db: Sesión de base de datos
        student_id: ID del estudiante
        task_id: ID de la tarea
        max_analyses: Máximo de análisis permitidos (default: 5)

    Returns:
        Tuple[bool, str, int]:
          - allowed (bool): Si se permite el análisis
          - message (str): Mensaje descriptivo
          - count (int): Análisis completados hasta ahora
    """
    try:
        # Contar todos los análisis de este estudiante en esta tarea
        count = db.query(func.count(StudentSolutionAnalysis.id)).filter(
            StudentSolutionAnalysis.student_id == student_id,
            StudentSolutionAnalysis.task_id == task_id
        ).scalar() or 0

        if count >= max_analyses:
            message = f"Ya completaste {count}/{max_analyses} análisis. Límite alcanzado."
            logger.warning(f"⚠️ Quota exceeded - Student: {student_id}, Task: {task_id}, Count: {count}")
            return False, message, count

        message = f"Análisis permitido: {count + 1}/{max_analyses}"
        return True, message, count

    except Exception as e:
        logger.error(f"Error checking quota: {str(e)}")
        # En caso de error, permitir (mejor UX que bloquear por error)
        return True, "Quota check failed, allowing analysis", 0


def check_cooldown(
    db: Session,
    student_id: int,
    task_id: int,
    cooldown_minutes: int = COOLDOWN_MINUTES
) -> Tuple[bool, int]:
    """
    Verifica si se ha completado el cooldown entre análisis.

    Obtiene el último análisis del estudiante en esta tarea
    y verifica si ya pasaron los minutos de cooldown.

    Args:
        db: Sesión de base de datos
        student_id: ID del estudiante
        task_id: ID de la tarea
        cooldown_minutes: Minutos de espera entre análisis (default: 5)

    Returns:
        Tuple[bool, int]:
          - allowed (bool): Si se permite el análisis
          - seconds_remaining (int): Segundos para permitir siguiente análisis
    """
    try:
        # Obtener último análisis
        last_analysis = db.query(StudentSolutionAnalysis).filter(
            StudentSolutionAnalysis.student_id == student_id,
            StudentSolutionAnalysis.task_id == task_id
        ).order_by(
            StudentSolutionAnalysis.created_at.desc()
        ).first()

        if not last_analysis:
            # No hay análisis previo - permitir
            return True, 0

        # Calcular tiempo transcurrido
        now = datetime.utcnow()
        elapsed = now - last_analysis.created_at
        cooldown_delta = timedelta(minutes=cooldown_minutes)

        if elapsed < cooldown_delta:
            remaining_delta = cooldown_delta - elapsed
            remaining_seconds = int(remaining_delta.total_seconds())

            logger.warning(
                f"⚠️ Cooldown active - Student: {student_id}, Task: {task_id}, "
                f"Remaining: {remaining_seconds}s"
            )
            return False, remaining_seconds

        return True, 0

    except Exception as e:
        logger.error(f"Error checking cooldown: {str(e)}")
        # En caso de error, permitir
        return True, 0


def record_analysis(
    db: Session,
    student_id: int,
    task_id: int,
    analysis_id: str,
    language: str,
    task_type: str,
    response_json: Dict[str, Any],
    analysis_duration_ms: int,
    ip_address: Optional[str] = None,
    user_agent: Optional[str] = None
) -> Optional[StudentSolutionAnalysis]:
    """
    Registra un análisis en la base de datos.

    Crea un nuevo registro de StudentSolutionAnalysis con todos los
    datos del análisis para auditoría y tracking.

    Args:
        db: Sesión de base de datos
        student_id: ID del estudiante
        task_id: ID de la tarea
        analysis_id: UUID único del análisis
        language: Lenguaje de programación
        task_type: Tipo de tarea ('tarea', 'evaluacion')
        response_json: Respuesta del LLM
        analysis_duration_ms: Duración del análisis en ms
        ip_address: IP del cliente (opcional)
        user_agent: User-Agent del navegador (opcional)

    Returns:
        StudentSolutionAnalysis: Record creado o None si hay error
    """
    try:
        # Convertir response_json a string JSON si es dict
        if isinstance(response_json, dict):
            response_json_str = json.dumps(response_json)
        else:
            response_json_str = str(response_json)

        # Crear nuevo registro
        record = StudentSolutionAnalysis(
            task_id=task_id,
            student_id=student_id,
            analysis_id=analysis_id,
            language=language,
            task_type=task_type,
            response_json=response_json_str,
            analysis_duration_ms=analysis_duration_ms,
            ip_address=ip_address,
            user_agent=user_agent
        )

        # Guardar en base de datos
        db.add(record)
        db.commit()
        db.refresh(record)

        logger.info(
            f"✅ Analysis recorded - Student: {student_id}, Task: {task_id}, "
            f"Duration: {analysis_duration_ms}ms"
        )

        return record

    except Exception as e:
        logger.error(f"Error recording analysis: {str(e)}")
        db.rollback()
        return None


def detect_abuse_pattern(
    db: Session,
    student_id: int,
    time_window_hours: int = 24
) -> Dict[str, Any]:
    """
    Detecta patrones de abuso en el uso de análisis.

    Analiza los análisis del estudiante en las últimas N horas
    para detectar comportamiento sospechoso.

    Args:
        db: Sesión de base de datos
        student_id: ID del estudiante
        time_window_hours: Ventana de tiempo a analizar (default: 24)

    Returns:
        Dict con indicadores de abuso:
          - total_analyses_window: Total de análisis en la ventana
          - max_analyses_single_task: Máximo en una sola tarea
          - unique_tasks: Número de tareas analizadas
          - is_suspicious: Si el patrón es sospechoso
          - reason: Razón del patrón sospechoso
          - severity: Nivel de severidad (low, medium, high)
    """
    try:
        # Calcular ventana de tiempo
        cutoff_time = datetime.utcnow() - timedelta(hours=time_window_hours)

        # Contar análisis totales en la ventana
        total_analyses = db.query(func.count(StudentSolutionAnalysis.id)).filter(
            StudentSolutionAnalysis.student_id == student_id,
            StudentSolutionAnalysis.created_at >= cutoff_time
        ).scalar() or 0

        # Contar análisis por tarea
        analyses_by_task = db.query(
            StudentSolutionAnalysis.task_id,
            func.count(StudentSolutionAnalysis.id).label('count')
        ).filter(
            StudentSolutionAnalysis.student_id == student_id,
            StudentSolutionAnalysis.created_at >= cutoff_time
        ).group_by(
            StudentSolutionAnalysis.task_id
        ).all()

        max_single = max([a[1] for a in analyses_by_task]) if analyses_by_task else 0
        unique_tasks = len(analyses_by_task)

        # Inicializar resultado
        result = {
            'total_analyses_window': total_analyses,
            'max_analyses_single_task': max_single,
            'unique_tasks': unique_tasks,
            'is_suspicious': False,
            'reason': 'normal',
            'severity': 'low'
        }

        # Detectar patrones sospechosos
        if total_analyses > ABUSE_THRESHOLD_24H:
            result['is_suspicious'] = True
            result['reason'] = f'Too many analyses in {time_window_hours}h ({total_analyses})'
            result['severity'] = 'high'

        elif max_single > ABUSE_THRESHOLD_SINGLE:
            result['is_suspicious'] = True
            result['reason'] = f'Too many analyses on single task ({max_single})'
            result['severity'] = 'medium'

        elif total_analyses > 20:
            result['severity'] = 'medium'

        # Logging
        if result['is_suspicious']:
            logger.warning(
                f"🚨 Suspicious pattern detected - Student: {student_id}, "
                f"Reason: {result['reason']}, Severity: {result['severity']}"
            )

        return result

    except Exception as e:
        logger.error(f"Error detecting abuse pattern: {str(e)}")
        return {
            'total_analyses_window': 0,
            'max_analyses_single_task': 0,
            'unique_tasks': 0,
            'is_suspicious': False,
            'reason': 'error_in_detection',
            'severity': 'unknown'
        }


def get_usage_stats(
    db: Session,
    student_id: int,
    task_id: int
) -> Dict[str, Any]:
    """
    Obtiene estadísticas de uso de análisis.

    Args:
        db: Sesión de base de datos
        student_id: ID del estudiante
        task_id: ID de la tarea

    Returns:
        Dict con estadísticas:
          - analyses_total: Total de análisis en esta tarea
          - analyses_today: Análisis en las últimas 24h
          - last_analysis: Timestamp del último análisis
          - avg_duration_ms: Duración promedio
    """
    try:
        today_start = datetime.utcnow() - timedelta(hours=24)

        # Total en esta tarea
        total = db.query(func.count(StudentSolutionAnalysis.id)).filter(
            StudentSolutionAnalysis.student_id == student_id,
            StudentSolutionAnalysis.task_id == task_id
        ).scalar() or 0

        # Hoy
        today = db.query(func.count(StudentSolutionAnalysis.id)).filter(
            StudentSolutionAnalysis.student_id == student_id,
            StudentSolutionAnalysis.task_id == task_id,
            StudentSolutionAnalysis.created_at >= today_start
        ).scalar() or 0

        # Último análisis
        last = db.query(StudentSolutionAnalysis).filter(
            StudentSolutionAnalysis.student_id == student_id,
            StudentSolutionAnalysis.task_id == task_id
        ).order_by(
            StudentSolutionAnalysis.created_at.desc()
        ).first()

        # Duración promedio
        avg_duration = db.query(
            func.avg(StudentSolutionAnalysis.analysis_duration_ms)
        ).filter(
            StudentSolutionAnalysis.student_id == student_id,
            StudentSolutionAnalysis.task_id == task_id
        ).scalar() or 0

        return {
            'analyses_total': total,
            'analyses_today': today,
            'last_analysis_at': last.created_at.isoformat() if last else None,
            'avg_duration_ms': int(avg_duration)
        }

    except Exception as e:
        logger.error(f"Error getting usage stats: {str(e)}")
        return {
            'analyses_total': 0,
            'analyses_today': 0,
            'last_analysis_at': None,
            'avg_duration_ms': 0
        }


# Exportar funciones públicas
__all__ = [
    'check_analysis_quota',
    'check_cooldown',
    'record_analysis',
    'detect_abuse_pattern',
    'get_usage_stats',
    'MAX_ANALYSES_PER_TASK',
    'COOLDOWN_MINUTES'
]
