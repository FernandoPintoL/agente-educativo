"""
Configuración del Agente Inteligente de Recomendaciones Educativas

Las variables de configuración se pueden establecer en el archivo .env:
- GROQ_API_KEY: API key de Groq
- GROQ_MODEL: Modelo a usar
- GROQ_TEMPERATURE: Temperatura (0-1)
- GROQ_MAX_TOKENS: Máximo de tokens
- GROQ_CACHE_ENABLED: Habilitar caché
- GROQ_CACHE_TTL: TTL en minutos
"""

import os
from dotenv import load_dotenv

# Cargar variables de entorno desde .env
# Busca .env en este orden:
# 1. ./agente/.env (específico del agente)
# 2. ../..env (en directorio ml_educativas)
# 3. Variables de entorno del sistema

agente_dir = os.path.dirname(__file__)
agente_env = os.path.join(agente_dir, '.env')
ml_api_env = os.path.join(agente_dir, '..', '.env')

# Cargar desde ./agente/.env primero (tiene prioridad)
if os.path.exists(agente_env):
    load_dotenv(agente_env, override=True)

# Luego cargar desde directorio padre (fallback)
if os.path.exists(ml_api_env):
    load_dotenv(ml_api_env, override=False)

# ============================================================
# GROQ API CONFIGURATION
# ============================================================

# API Key de Groq (requerida)
# Obtén tu key en: https://console.groq.com/keys
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

if not GROQ_API_KEY:
    print("\n[WARNING] GROQ_API_KEY no está configurada en .env")
    print("Por favor, añade tu API key de Groq al archivo .env:")
    print("GROQ_API_KEY=tu_api_key_aqui\n")

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

LOG_LEVEL = "INFO"
LOG_FILE = "agente/logs/agent.log"

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
