"""
Audit Models
============

Modelos SQLAlchemy para logging y auditoría de análisis de soluciones.

Modelos:
  - AnalysisLog: Registro de cada análisis
  - AbuseAlert: Alertas de abuso detectado
"""

from sqlalchemy import Column, Integer, String, Text, DateTime, Float, Boolean, func
from sqlalchemy.dialects.postgresql import UUID, JSON
from sqlalchemy.ext.declarative import declarative_base
from datetime import datetime
import logging

logger = logging.getLogger(__name__)

Base = declarative_base()


class AnalysisLog(Base):
    """
    Registro detallado de cada análisis de solución.

    Soporta:
    - Auditoría completa de uso
    - Historial por estudiante
    - Estadísticas por tarea
    - Análisis de patrones
    """

    __tablename__ = 'analysis_logs'

    # ==========================================
    # PRIMARY KEY
    # ==========================================
    id = Column(Integer, primary_key=True, autoincrement=True)

    # ==========================================
    # FOREIGN KEYS
    # ==========================================
    student_id = Column(Integer, nullable=False, index=True)
    task_id = Column(Integer, nullable=False, index=True)
    course_id = Column(Integer, nullable=False, index=True)

    # ==========================================
    # ANALYSIS TRACKING
    # ==========================================
    analysis_id = Column(
        UUID(as_uuid=True),
        unique=True,
        nullable=False,
        index=True
    )

    language = Column(String(50), default='python')
    task_type = Column(String(50), default='tarea')

    # ==========================================
    # TIMING
    # ==========================================
    created_at = Column(
        DateTime,
        default=datetime.utcnow,
        nullable=False,
        index=True
    )

    analysis_duration_ms = Column(Integer, default=0)

    # ==========================================
    # RESULTS
    # ==========================================
    concepts_found = Column(Integer, default=0)
    errors_found = Column(Integer, default=0)
    code_length = Column(Integer, default=0)

    success = Column(Boolean, default=True)
    error_message = Column(String(255))

    # ==========================================
    # CLIENT INFORMATION
    # ==========================================
    ip_address = Column(String(45))
    user_agent = Column(String(255))

    # ==========================================
    # FEEDBACK METADATA
    # ==========================================
    feedback_json = Column(Text)  # JSON serializado de la respuesta

    # ==========================================
    # COMPOSITE INDEXES
    # ==========================================
    __table_args__ = (
        # Índices para búsquedas comunes
        # Todos los análisis de un estudiante
        # Todos los análisis de una tarea
        # Análisis por fecha
    )

    def __repr__(self) -> str:
        return (
            f"<AnalysisLog("
            f"id={self.id}, "
            f"student_id={self.student_id}, "
            f"task_id={self.task_id}, "
            f"success={self.success}"
            f")>"
        )

    def to_dict(self) -> dict:
        """Convierte el modelo a diccionario"""
        return {
            'id': self.id,
            'student_id': self.student_id,
            'task_id': self.task_id,
            'course_id': self.course_id,
            'analysis_id': str(self.analysis_id) if self.analysis_id else None,
            'language': self.language,
            'task_type': self.task_type,
            'created_at': self.created_at.isoformat() if self.created_at else None,
            'analysis_duration_ms': self.analysis_duration_ms,
            'concepts_found': self.concepts_found,
            'errors_found': self.errors_found,
            'code_length': self.code_length,
            'success': self.success,
            'error_message': self.error_message,
            'ip_address': self.ip_address
        }


class AbuseAlert(Base):
    """
    Alertas de comportamiento sospechoso o abuso.

    Tipos de patrones:
    - spam: Demasiados análisis
    - overuse: Sobre-utilización de una tarea
    - bypass: Intento de evitar rate limiting
    - unusual: Comportamiento inusual
    - timing: Análisis a horas extrañas
    """

    __tablename__ = 'abuse_alerts'

    # ==========================================
    # PRIMARY KEY
    # ==========================================
    id = Column(Integer, primary_key=True, autoincrement=True)

    # ==========================================
    # FOREIGN KEYS
    # ==========================================
    student_id = Column(Integer, nullable=False, index=True)
    task_id = Column(Integer)
    course_id = Column(Integer, nullable=False, index=True)

    # ==========================================
    # ALERT DETAILS
    # ==========================================
    pattern_type = Column(
        String(50),
        nullable=False
        # spam, overuse, bypass, unusual, timing
    )

    severity = Column(
        String(20),
        nullable=False,
        default='low'
        # low, medium, high
    )

    message = Column(String(500), nullable=False)
    description = Column(Text)

    # ==========================================
    # METRICS
    # ==========================================
    metric_value = Column(Integer)  # Ej: cantidad de análisis
    metric_threshold = Column(Integer)  # Ej: umbral de 50

    # ==========================================
    # ACTION TRACKING
    # ==========================================
    action_taken = Column(
        String(50),
        default='logged'
        # logged, notified, blocked, reviewed
    )

    action_timestamp = Column(DateTime)
    action_by = Column(String(100))  # 'system' o professor_id

    resolved = Column(Boolean, default=False)
    resolved_at = Column(DateTime)
    resolution_notes = Column(Text)

    # ==========================================
    # AUDIT
    # ==========================================
    created_at = Column(
        DateTime,
        default=datetime.utcnow,
        nullable=False,
        index=True
    )

    updated_at = Column(
        DateTime,
        default=datetime.utcnow,
        onupdate=datetime.utcnow
    )

    # ==========================================
    # COMPOSITE INDEXES
    # ==========================================
    __table_args__ = (
        # Alertas activas por curso
        # Alertas por severidad
        # Alertas sin resolver
    )

    def __repr__(self) -> str:
        return (
            f"<AbuseAlert("
            f"id={self.id}, "
            f"student_id={self.student_id}, "
            f"pattern_type={self.pattern_type}, "
            f"severity={self.severity}"
            f")>"
        )

    def to_dict(self) -> dict:
        """Convierte el modelo a diccionario"""
        return {
            'id': self.id,
            'student_id': self.student_id,
            'task_id': self.task_id,
            'course_id': self.course_id,
            'pattern_type': self.pattern_type,
            'severity': self.severity,
            'message': self.message,
            'description': self.description,
            'metric_value': self.metric_value,
            'metric_threshold': self.metric_threshold,
            'action_taken': self.action_taken,
            'action_by': self.action_by,
            'resolved': self.resolved,
            'created_at': self.created_at.isoformat() if self.created_at else None,
            'resolved_at': self.resolved_at.isoformat() if self.resolved_at else None
        }


class StudentAnalyticsSummary(Base):
    """
    Resumen diario/semanal de analíticas de estudiante.

    Se actualiza una vez al día para mejorar performance
    de reportes y dashboards.
    """

    __tablename__ = 'student_analytics_summary'

    # ==========================================
    # PRIMARY KEY
    # ==========================================
    id = Column(Integer, primary_key=True, autoincrement=True)

    # ==========================================
    # FOREIGN KEYS
    # ==========================================
    student_id = Column(Integer, nullable=False, index=True)
    course_id = Column(Integer, nullable=False, index=True)

    # ==========================================
    # PERIOD
    # ==========================================
    date = Column(DateTime, nullable=False, index=True)  # Inicio del periodo
    period = Column(String(20), default='daily')  # daily, weekly, monthly

    # ==========================================
    # STATISTICS
    # ==========================================
    total_analyses = Column(Integer, default=0)
    successful_analyses = Column(Integer, default=0)
    failed_analyses = Column(Integer, default=0)

    avg_duration_ms = Column(Integer, default=0)
    avg_concepts_found = Column(Float, default=0.0)
    avg_errors_found = Column(Float, default=0.0)

    # ==========================================
    # ABUSE SCORING
    # ==========================================
    abuse_score = Column(Float, default=0.0)  # 0-100
    alerts_generated = Column(Integer, default=0)

    # ==========================================
    # AUDIT
    # ==========================================
    created_at = Column(
        DateTime,
        default=datetime.utcnow,
        nullable=False
    )

    updated_at = Column(
        DateTime,
        default=datetime.utcnow,
        onupdate=datetime.utcnow
    )

    def __repr__(self) -> str:
        return (
            f"<StudentAnalyticsSummary("
            f"student_id={self.student_id}, "
            f"date={self.date}, "
            f"analyses={self.total_analyses}"
            f")>"
        )

    def to_dict(self) -> dict:
        """Convierte el modelo a diccionario"""
        return {
            'id': self.id,
            'student_id': self.student_id,
            'course_id': self.course_id,
            'date': self.date.isoformat() if self.date else None,
            'period': self.period,
            'total_analyses': self.total_analyses,
            'successful_analyses': self.successful_analyses,
            'failed_analyses': self.failed_analyses,
            'avg_duration_ms': self.avg_duration_ms,
            'avg_concepts_found': self.avg_concepts_found,
            'avg_errors_found': self.avg_errors_found,
            'abuse_score': self.abuse_score,
            'alerts_generated': self.alerts_generated
        }


# Exportar para uso en otros módulos
__all__ = ['AnalysisLog', 'AbuseAlert', 'StudentAnalyticsSummary', 'Base']
