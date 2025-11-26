"""
Audit Service
=============

Servicio para logging, auditoría y detección de abuso.

Funciones principales:
  - log_analysis() - Registra análisis en logs
  - create_abuse_alert() - Crea alerta de abuso
  - get_student_history() - Historial de estudiante
  - get_abuse_alerts() - Alertas activas
  - generate_report() - Reportes para profesores
"""

import json
import logging
from datetime import datetime, timedelta
from typing import List, Dict, Any, Optional
from sqlalchemy.orm import Session
from sqlalchemy import func, and_

logger = logging.getLogger(__name__)

# Importar modelos
try:
    from models.audit_models import AnalysisLog, AbuseAlert, StudentAnalyticsSummary
except ImportError:
    from agente.models.audit_models import AnalysisLog, AbuseAlert, StudentAnalyticsSummary


# ============================================================
# CONSTANTES
# ============================================================

ABUSE_PATTERNS = {
    'spam_24h': {
        'threshold': 50,
        'severity': 'high',
        'description': 'Más de 50 análisis en 24 horas'
    },
    'overuse_single_task': {
        'threshold': 10,
        'severity': 'medium',
        'description': 'Más de 10 análisis en una sola tarea'
    },
    'unusual_timing': {
        'threshold': 3,  # 3+ análisis entre 1-5 AM
        'severity': 'low',
        'description': 'Análisis a horas inusuales'
    }
}


# ============================================================
# FUNCIONES PRINCIPALES
# ============================================================


def log_analysis(
    db: Session,
    student_id: int,
    task_id: int,
    course_id: int,
    analysis_id: str,
    language: str,
    task_type: str,
    duration_ms: int,
    success: bool = True,
    concepts_found: int = 0,
    errors_found: int = 0,
    code_length: int = 0,
    feedback_json: Optional[Dict[str, Any]] = None,
    ip_address: Optional[str] = None,
    user_agent: Optional[str] = None,
    error_message: Optional[str] = None
) -> Optional[AnalysisLog]:
    """
    Registra un análisis en el log de auditoría.

    También verifica patrones de abuso y crea alertas si es necesario.

    Args:
        db: Sesión de base de datos
        student_id: ID del estudiante
        task_id: ID de la tarea
        course_id: ID del curso
        analysis_id: UUID del análisis
        language: Lenguaje de programación
        task_type: Tipo de tarea
        duration_ms: Duración en milisegundos
        success: Si el análisis fue exitoso
        concepts_found: Cantidad de conceptos encontrados
        errors_found: Cantidad de errores encontrados
        code_length: Longitud del código
        feedback_json: JSON de feedback (opcional)
        ip_address: IP del cliente (opcional)
        user_agent: User-Agent del navegador (opcional)
        error_message: Mensaje de error si no fue exitoso

    Returns:
        AnalysisLog record creado o None si hay error
    """
    try:
        # Convertir feedback_json si es dict
        if feedback_json and isinstance(feedback_json, dict):
            feedback_json_str = json.dumps(feedback_json)
        else:
            feedback_json_str = str(feedback_json) if feedback_json else None

        # Crear registro
        log_record = AnalysisLog(
            student_id=student_id,
            task_id=task_id,
            course_id=course_id,
            analysis_id=analysis_id,
            language=language,
            task_type=task_type,
            analysis_duration_ms=duration_ms,
            success=success,
            concepts_found=concepts_found,
            errors_found=errors_found,
            code_length=code_length,
            feedback_json=feedback_json_str,
            ip_address=ip_address,
            user_agent=user_agent,
            error_message=error_message
        )

        # Guardar en BD
        db.add(log_record)
        db.commit()
        db.refresh(log_record)

        logger.info(
            f"✅ Analysis logged - Student: {student_id}, Task: {task_id}, "
            f"Success: {success}, Duration: {duration_ms}ms"
        )

        # Detectar abuso después de registrar
        detect_and_create_alerts(db, student_id, course_id)

        return log_record

    except Exception as e:
        logger.error(f"Error logging analysis: {str(e)}")
        db.rollback()
        return None


def create_abuse_alert(
    db: Session,
    student_id: int,
    course_id: int,
    pattern_type: str,
    severity: str,
    message: str,
    description: Optional[str] = None,
    task_id: Optional[int] = None,
    metric_value: Optional[int] = None,
    metric_threshold: Optional[int] = None
) -> Optional[AbuseAlert]:
    """
    Crea una alerta de abuso.

    Args:
        db: Sesión de base de datos
        student_id: ID del estudiante
        course_id: ID del curso
        pattern_type: Tipo de patrón (spam, overuse, unusual, etc)
        severity: Severidad (low, medium, high)
        message: Mensaje corto de la alerta
        description: Descripción detallada (opcional)
        task_id: ID de la tarea (opcional)
        metric_value: Valor del métrico (ej: cantidad de análisis)
        metric_threshold: Umbral superado

    Returns:
        AbuseAlert creado o None si hay error
    """
    try:
        # Verificar si ya existe una alerta sin resolver
        existing = db.query(AbuseAlert).filter(
            AbuseAlert.student_id == student_id,
            AbuseAlert.pattern_type == pattern_type,
            AbuseAlert.resolved == False
        ).first()

        if existing:
            logger.warning(
                f"⚠️ Alert already exists - Student: {student_id}, "
                f"Pattern: {pattern_type}"
            )
            return existing

        # Crear nueva alerta
        alert = AbuseAlert(
            student_id=student_id,
            task_id=task_id,
            course_id=course_id,
            pattern_type=pattern_type,
            severity=severity,
            message=message,
            description=description,
            metric_value=metric_value,
            metric_threshold=metric_threshold,
            action_taken='logged',
            created_at=datetime.utcnow()
        )

        # Guardar
        db.add(alert)
        db.commit()
        db.refresh(alert)

        logger.warning(
            f"🚨 Abuse alert created - Student: {student_id}, "
            f"Pattern: {pattern_type}, Severity: {severity}"
        )

        return alert

    except Exception as e:
        logger.error(f"Error creating abuse alert: {str(e)}")
        db.rollback()
        return None


def detect_and_create_alerts(
    db: Session,
    student_id: int,
    course_id: int
) -> List[AbuseAlert]:
    """
    Detecta patrones de abuso y crea alertas automáticamente.

    Analiza:
    - Spam de análisis (50+ en 24h)
    - Sobre-utilización de tarea (10+ en una tarea)
    - Análisis a horas inusuales

    Args:
        db: Sesión de base de datos
        student_id: ID del estudiante
        course_id: ID del curso

    Returns:
        Lista de alertas creadas
    """
    alerts_created = []

    try:
        # PATRÓN 1: Spam en 24 horas
        last_24h = datetime.utcnow() - timedelta(hours=24)
        count_24h = db.query(func.count(AnalysisLog.id)).filter(
            AnalysisLog.student_id == student_id,
            AnalysisLog.course_id == course_id,
            AnalysisLog.created_at >= last_24h
        ).scalar() or 0

        if count_24h >= ABUSE_PATTERNS['spam_24h']['threshold']:
            alert = create_abuse_alert(
                db=db,
                student_id=student_id,
                course_id=course_id,
                pattern_type='spam_24h',
                severity='high',
                message=f'Spam detectado: {count_24h} análisis en 24 horas',
                description=f'El estudiante superó el umbral de {ABUSE_PATTERNS["spam_24h"]["threshold"]} análisis',
                metric_value=count_24h,
                metric_threshold=ABUSE_PATTERNS['spam_24h']['threshold']
            )
            if alert:
                alerts_created.append(alert)

        # PATRÓN 2: Sobre-utilización de tarea
        # (Obtener tarea más usada en últimas 24h)
        most_used_task = db.query(
            AnalysisLog.task_id,
            func.count(AnalysisLog.id).label('count')
        ).filter(
            AnalysisLog.student_id == student_id,
            AnalysisLog.course_id == course_id,
            AnalysisLog.created_at >= last_24h
        ).group_by(
            AnalysisLog.task_id
        ).order_by(
            func.count(AnalysisLog.id).desc()
        ).first()

        if most_used_task and most_used_task[1] >= ABUSE_PATTERNS['overuse_single_task']['threshold']:
            alert = create_abuse_alert(
                db=db,
                student_id=student_id,
                course_id=course_id,
                task_id=most_used_task[0],
                pattern_type='overuse_single_task',
                severity='medium',
                message=f'Sobre-utilización: {most_used_task[1]} análisis en una tarea',
                description=f'Superó el umbral de {ABUSE_PATTERNS["overuse_single_task"]["threshold"]} en tarea #{most_used_task[0]}',
                metric_value=most_used_task[1],
                metric_threshold=ABUSE_PATTERNS['overuse_single_task']['threshold']
            )
            if alert:
                alerts_created.append(alert)

        return alerts_created

    except Exception as e:
        logger.error(f"Error detecting abuse patterns: {str(e)}")
        return []


def get_student_history(
    db: Session,
    student_id: int,
    course_id: Optional[int] = None,
    limit: int = 50
) -> List[AnalysisLog]:
    """
    Obtiene historial de análisis de un estudiante.

    Args:
        db: Sesión de base de datos
        student_id: ID del estudiante
        course_id: ID del curso (opcional)
        limit: Máximo de registros

    Returns:
        Lista de AnalysisLog
    """
    try:
        query = db.query(AnalysisLog).filter(
            AnalysisLog.student_id == student_id
        )

        if course_id:
            query = query.filter(AnalysisLog.course_id == course_id)

        return query.order_by(
            AnalysisLog.created_at.desc()
        ).limit(limit).all()

    except Exception as e:
        logger.error(f"Error getting student history: {str(e)}")
        return []


def get_abuse_alerts(
    db: Session,
    course_id: int,
    severity: Optional[str] = None,
    resolved: bool = False,
    limit: int = 50
) -> List[AbuseAlert]:
    """
    Obtiene alertas de abuso para un curso.

    Args:
        db: Sesión de base de datos
        course_id: ID del curso
        severity: Filtrar por severidad (low, medium, high) - opcional
        resolved: Si se incluyen resueltas (default: False)
        limit: Máximo de registros

    Returns:
        Lista de AbuseAlert
    """
    try:
        query = db.query(AbuseAlert).filter(
            AbuseAlert.course_id == course_id,
            AbuseAlert.resolved == resolved
        )

        if severity:
            query = query.filter(AbuseAlert.severity == severity)

        return query.order_by(
            AbuseAlert.created_at.desc()
        ).limit(limit).all()

    except Exception as e:
        logger.error(f"Error getting abuse alerts: {str(e)}")
        return []


def get_professor_dashboard(
    db: Session,
    course_id: int
) -> Dict[str, Any]:
    """
    Genera datos para el dashboard del profesor.

    Incluye:
    - Estadísticas generales del curso
    - Alertas activas
    - Estudiantes problemáticos
    - Estadísticas por tarea

    Args:
        db: Sesión de base de datos
        course_id: ID del curso

    Returns:
        Dict con datos del dashboard
    """
    try:
        # Estadísticas generales
        total_logs = db.query(func.count(AnalysisLog.id)).filter(
            AnalysisLog.course_id == course_id
        ).scalar() or 0

        total_students = db.query(func.count(func.distinct(AnalysisLog.student_id))).filter(
            AnalysisLog.course_id == course_id
        ).scalar() or 0

        successful = db.query(func.count(AnalysisLog.id)).filter(
            AnalysisLog.course_id == course_id,
            AnalysisLog.success == True
        ).scalar() or 0

        # Alertas activas
        active_alerts = db.query(AbuseAlert).filter(
            AbuseAlert.course_id == course_id,
            AbuseAlert.resolved == False
        ).order_by(AbuseAlert.created_at.desc()).limit(10).all()

        # Estudiantes con más análisis
        top_students = db.query(
            AnalysisLog.student_id,
            func.count(AnalysisLog.id).label('count')
        ).filter(
            AnalysisLog.course_id == course_id
        ).group_by(
            AnalysisLog.student_id
        ).order_by(
            func.count(AnalysisLog.id).desc()
        ).limit(5).all()

        # Estadísticas por tarea
        task_stats = db.query(
            AnalysisLog.task_id,
            func.count(AnalysisLog.id).label('total'),
            func.count(func.distinct(AnalysisLog.student_id)).label('students')
        ).filter(
            AnalysisLog.course_id == course_id
        ).group_by(
            AnalysisLog.task_id
        ).order_by(
            func.count(AnalysisLog.id).desc()
        ).limit(5).all()

        return {
            'summary': {
                'total_analyses': total_logs,
                'total_students': total_students,
                'successful_rate': (successful / total_logs * 100) if total_logs > 0 else 0
            },
            'active_alerts': [alert.to_dict() for alert in active_alerts],
            'top_students': [
                {'student_id': s[0], 'analysis_count': s[1]}
                for s in top_students
            ],
            'task_statistics': [
                {'task_id': t[0], 'total_analyses': t[1], 'unique_students': t[2]}
                for t in task_stats
            ]
        }

    except Exception as e:
        logger.error(f"Error generating professor dashboard: {str(e)}")
        return {}


def get_task_usage_stats(
    db: Session,
    task_id: int,
    course_id: Optional[int] = None
) -> Dict[str, Any]:
    """
    Obtiene estadísticas de uso de una tarea.

    Args:
        db: Sesión de base de datos
        task_id: ID de la tarea
        course_id: ID del curso (opcional)

    Returns:
        Dict con estadísticas
    """
    try:
        query = db.query(AnalysisLog).filter(
            AnalysisLog.task_id == task_id
        )

        if course_id:
            query = query.filter(AnalysisLog.course_id == course_id)

        logs = query.all()

        if not logs:
            return {'error': 'No data found'}

        total = len(logs)
        students = len(set([log.student_id for log in logs]))
        successful = len([log for log in logs if log.success])
        avg_duration = sum([log.analysis_duration_ms for log in logs]) / total if total > 0 else 0

        # Max analyses by single student
        student_counts = {}
        for log in logs:
            student_counts[log.student_id] = student_counts.get(log.student_id, 0) + 1

        max_student_analyses = max(student_counts.values()) if student_counts else 0

        return {
            'task_id': task_id,
            'total_analyses': total,
            'unique_students': students,
            'successful_rate': (successful / total * 100) if total > 0 else 0,
            'avg_duration_ms': int(avg_duration),
            'max_analyses_single_student': max_student_analyses,
            'total_errors': sum([log.errors_found for log in logs]),
            'avg_errors_per_analysis': sum([log.errors_found for log in logs]) / total if total > 0 else 0
        }

    except Exception as e:
        logger.error(f"Error getting task usage stats: {str(e)}")
        return {}


# Exportar funciones públicas
__all__ = [
    'log_analysis',
    'create_abuse_alert',
    'detect_and_create_alerts',
    'get_student_history',
    'get_abuse_alerts',
    'get_professor_dashboard',
    'get_task_usage_stats'
]
