"""
Services Package
================

Servicios reutilizables para la plataforma educativa.
"""

try:
    from .rate_limiting import (
        check_analysis_quota,
        check_cooldown,
        record_analysis,
        detect_abuse_pattern,
        get_usage_stats
    )
except ImportError:
    # Fallback si las dependencias no están disponibles
    def check_analysis_quota(*args, **kwargs): return True
    def check_cooldown(*args, **kwargs): return True
    def record_analysis(*args, **kwargs): pass
    def detect_abuse_pattern(*args, **kwargs): return False
    def get_usage_stats(*args, **kwargs): return {}

__all__ = [
    'check_analysis_quota',
    'check_cooldown',
    'record_analysis',
    'detect_abuse_pattern',
    'get_usage_stats'
]
