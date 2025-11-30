"""
Configuración Centralizada del Agente LLM - v2.0.0
Soporta LOCAL (desarrollo) y RAILWAY (producción)

Variables de configuración en .env:
- ENVIRONMENT: development o production
- PORT: 8003 (local) o 8080 (Railway automático)
- DB_*: Credenciales de BD
- GROQ_*: Configuración Groq API
- ML_*: URLs de servicios ML
"""

import os
from dotenv import load_dotenv

# Cargar variables de entorno desde .env
agente_dir = os.path.dirname(__file__)
agente_env = os.path.join(agente_dir, '.env')

if os.path.exists(agente_env):
    load_dotenv(agente_env, override=True)

# ============================================================
# AMBIENTE Y PUERTO
# ============================================================

ENVIRONMENT = os.getenv('ENVIRONMENT', 'development').lower()
IS_PRODUCTION = ENVIRONMENT in ('production', 'railway')
IS_DEVELOPMENT = not IS_PRODUCTION

# Puerto automático basado en ambiente
if IS_PRODUCTION:
    # Railway asigna el puerto en la variable PORT
    PORT = int(os.getenv('PORT', 8080))
else:
    # Local usa 8003 (supervisado usa 8001, no_supervisado usa 8002)
    PORT = int(os.getenv('PORT', 8003))

HOST = os.getenv('HOST', '0.0.0.0')
DEBUG = os.getenv('DEBUG', 'true').lower() == 'true' if IS_DEVELOPMENT else False
LOG_LEVEL = os.getenv('LOG_LEVEL', 'DEBUG' if IS_DEVELOPMENT else 'INFO').upper()

# ============================================================
# BASE DE DATOS
# ============================================================

DB_HOST = os.getenv('DB_HOST', 'localhost')
DB_PORT = os.getenv('DB_PORT', '5432')
DB_DATABASE = os.getenv('DB_DATABASE', 'educativa')
DB_USERNAME = os.getenv('DB_USERNAME', 'postgres')
DB_PASSWORD = os.getenv('DB_PASSWORD', '1234')

# ============================================================
# URLs DE SERVICIOS ML (Para integración)
# ============================================================

ML_SUPERVISED_URL = os.getenv('ML_SUPERVISED_URL', 'http://127.0.0.1:8001')
ML_UNSUPERVISED_URL = os.getenv('ML_UNSUPERVISED_URL', 'http://127.0.0.1:8002')
ML_API_TIMEOUT = int(os.getenv('ML_API_TIMEOUT', '30'))

# ============================================================
# GROQ API CONFIGURATION
# ============================================================

# API Key de Groq (requerida en PRODUCTION, opcional en LOCAL)
# Obtén tu key en: https://console.groq.com/keys
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

if not GROQ_API_KEY and IS_PRODUCTION:
    raise ValueError(
        "\n[ERROR] GROQ_API_KEY es requerida en PRODUCTION\n"
        "Agrega GROQ_API_KEY a Railway Console\n"
    )
elif not GROQ_API_KEY and IS_DEVELOPMENT:
    print(
        "[INFO] GROQ_API_KEY no configurada - usando fallback en LOCAL"
    )

# Modelo de Groq a usar
# NOTA: mixtral-8x7b-32768 fue descontinuado en Nov 2025
GROQ_MODEL = os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile")

# Modelos disponibles actualmente en Groq (Nov 2025):
# - "llama-3.3-70b-versatile" (RECOMENDADO) - Nuevo modelo, rápido y preciso
# - "llama-3.1-8b-instant" - Muy rápido, menos preciso
# Ver todos: python check_groq_models.py

# ============================================================
# AGENT CONFIGURATION
# ============================================================

# Temperatura para recomendaciones (0=determinístico, 1=creativo)
# 0.3 recomendado para recomendaciones consistentes
RECOMMENDATION_TEMPERATURE = float(os.getenv("GROQ_TEMPERATURE", "0.3"))
EXPLANATION_TEMPERATURE = 0.7     # Explicaciones variadas

# Máximo de tokens en la respuesta
MAX_TOKENS = int(os.getenv("GROQ_MAX_TOKENS", "500"))

# Timeout para requests a Groq (segundos)
REQUEST_TIMEOUT = 30  # segundos

# ============================================================
# LOGGING
# ============================================================

# Ya definido arriba: LOG_LEVEL basado en ENVIRONMENT
LOG_FILE = os.getenv('LOG_FILE', 'agente/logs/agent.log')

# ============================================================
# CACHE (para evitar llamadas repetidas a Groq)
# ============================================================

ENABLE_CACHE = os.getenv("GROQ_CACHE_ENABLED", "true").lower() == "true"
CACHE_TTL_MINUTES = int(os.getenv("GROQ_CACHE_TTL", "60"))  # Por defecto 60 minutos

# ============================================================
# RECOMENDACIONES PREDEFINIDAS (fallback si API falla)
# ============================================================

FALLBACK_RECOMMENDATIONS = {
    "high_risk": {
        "type": "intervention",
        "urgency": "immediate",
        "resources": [
            "tutoring_1to1",
            "remedial_course",
            "parent_meeting"
        ]
    },
    "medium_risk": {
        "type": "tutoring",
        "urgency": "normal",
        "resources": [
            "group_tutoring",
            "practice_exercises",
            "peer_learning"
        ]
    },
    "low_risk": {
        "type": "enrichment",
        "urgency": "preventive",
        "resources": [
            "advanced_materials",
            "challenge_problems",
            "leadership_roles"
        ]
    }
}

# Configuración cargada (logs sin emojis para evitar encoding issues)
