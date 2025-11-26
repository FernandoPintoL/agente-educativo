"""
Student Solution Analysis Model
================================

Modelo SQLAlchemy para rastrear análisis de soluciones de estudiantes.
Utilizado para rate limiting, auditoría y detección de abuso.

Campos:
  - id: Primary key
  - student_id: ID del estudiante (FK)
  - task_id: ID de la tarea (FK)
  - analysis_id: UUID único para cada análisis
  - language: Lenguaje del código (python, js, java, etc)
  - task_type: Tipo de tarea (tarea, evaluacion)
  - response_json: Respuesta completa del análisis
  - analysis_duration_ms: Tiempo que tardó el análisis
  - created_at: Timestamp de creación
  - updated_at: Timestamp de última actualización
  - ip_address: IP del cliente
  - user_agent: User-Agent del navegador
"""

from sqlalchemy import Column, Integer, String, Text, DateTime, func
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.ext.declarative import declarative_base
from datetime import datetime
import uuid
import logging

logger = logging.getLogger(__name__)

Base = declarative_base()


class StudentSolutionAnalysis(Base):
    """
    Modelo para rastrear análisis de soluciones de estudiantes.

    Soporta:
    - Rate limiting (quota + cooldown)
    - Auditoría de uso
    - Detección de abuso
    - Análisis de patrones
    """

    __tablename__ = 'student_solution_analyses'

    # ==========================================
    # PRIMARY KEY
    # ==========================================
    id = Column(Integer, primary_key=True, autoincrement=True)

    # ==========================================
    # FOREIGN KEYS
    # ==========================================
    task_id = Column(Integer, nullable=False, index=True)
    student_id = Column(Integer, nullable=False, index=True)

    # ==========================================
    # ANALYSIS TRACKING
    # ==========================================
    analysis_id = Column(
        UUID(as_uuid=True),
        default=uuid.uuid4,
        unique=True,
        nullable=False,
        index=True
    )

    language = Column(String(50), default='python')
    task_type = Column(String(50), default='tarea')  # 'tarea', 'evaluacion'

    # ==========================================
    # ANALYSIS RESULT
    # ==========================================
    response_json = Column(Text, nullable=False)
    analysis_duration_ms = Column(Integer, default=0)

    # ==========================================
    # TIMESTAMPS
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
    # AUDITING
    # ==========================================
    ip_address = Column(String(45))  # IPv4 o IPv6
    user_agent = Column(String(255))

    # ==========================================
    # COMPOSITE INDEXES
    # ==========================================
    __table_args__ = (
        # Índices para queries de rate limiting
        # Buscar análisis de un estudiante en una tarea
        # (usado en check_analysis_quota, check_cooldown)
    )

    def __repr__(self) -> str:
        return (
            f"<StudentSolutionAnalysis("
            f"id={self.id}, "
            f"student_id={self.student_id}, "
            f"task_id={self.task_id}, "
            f"created_at={self.created_at}"
            f")>"
        )

    def to_dict(self) -> dict:
        """Convierte el modelo a diccionario"""
        return {
            'id': self.id,
            'task_id': self.task_id,
            'student_id': self.student_id,
            'analysis_id': str(self.analysis_id),
            'language': self.language,
            'task_type': self.task_type,
            'response_json': self.response_json,
            'analysis_duration_ms': self.analysis_duration_ms,
            'created_at': self.created_at.isoformat() if self.created_at else None,
            'updated_at': self.updated_at.isoformat() if self.updated_at else None,
            'ip_address': self.ip_address,
            'user_agent': self.user_agent
        }


# Exportar para uso en otros módulos
__all__ = ['StudentSolutionAnalysis', 'Base']
