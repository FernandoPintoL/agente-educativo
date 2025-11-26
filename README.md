# 🤖 Agente Inteligente de Síntesis LLM

**Servicio de síntesis de descubrimientos ML usando LLM (Groq) para generar insights educativos inteligentes**

| Aspecto | Detalle |
|--------|---------|
| **Tecnología** | FastAPI + LangChain + Groq LLM |
| **Lenguaje** | Python 3.12+ |
| **Puerto Local** | 8003 |
| **Puerto Railway** | 8080 (dinámico `$PORT`) |
| **Base de Datos** | PostgreSQL (compartida) |
| **Cache** | Redis (opcional) |

---

## 📋 Descripción

Este servicio actúa como **orquestador inteligente** que:

✅ Sintetiza descubrimientos de los pipelines ML (supervisado y no supervisado)
✅ Genera recomendaciones personalizadas usando LLM (Groq)
✅ Crea estrategias de intervención educativa inteligentes
✅ Mantiene caché de respuestas para optimizar performance
✅ Se integra con la plataforma educativa vía API REST

**Flujo:**
```
[Datos Estudiante] → [ML Supervisado + No Supervisado] → [Agente LLM] → [Recomendaciones] → [Plataforma]
```

---

## 🚀 Quick Start (Inicio Rápido)

### Para Usuarios Locales

```bash
# 1. Ir al directorio agente
cd D:\PLATAFORMA\ EDUCATIVA\agente

# 2. Activar entorno virtual
.\venv\Scripts\activate

# 3. Instalar dependencias
pip install -r requirements.txt

# 4. Configurar variables de entorno
# Edita .env con tu GROQ_API_KEY

# 5. Iniciar servicio
python agent_service.py
```

**Resultado esperado:**
```
INFO:     Uvicorn running on http://0.0.0.0:8003
INFO:     Application startup complete
LLM Available: True ✅
```

Accede a: **http://localhost:8003/docs** para la interfaz interactiva

---

## ⚙️ Configuración

### Paso 1: Obtener GROQ_API_KEY

1. Ir a https://console.groq.com/keys
2. Crear nueva API key
3. Copiar la key

### Paso 2: Configurar .env

**Archivo:** `agente/.env`

```ini
# ====================================
# CONFIGURACIÓN CRÍTICA
# ====================================

# GROQ API (REQUERIDO)
GROQ_API_KEY=gsk_xxxxxxxxxxxxx  # Tu API key

# Puerto
PORT=8003  # Local: 8003, Railway: $PORT (env var)

# Base de Datos
DATABASE_URL=postgresql://usuario:password@localhost:5432/educativa_db

# ====================================
# CONFIGURACIÓN RECOMENDADA
# ====================================

# Modelo LLM
GROQ_MODEL=mixtral-8x7b-32768  # Rápido (recomendado)
# Alternativas:
# GROQ_MODEL=llama-3.3-70b-versatile  # Más potente

# Temperatura (0=determinista, 1=creativo)
GROQ_TEMPERATURE=0.3  # Para recomendaciones consistentes

# Ambiente
ENVIRONMENT=development
DEBUG=true
LOG_LEVEL=INFO

# ====================================
# CONFIGURACIÓN OPCIONAL
# ====================================

# Redis Cache
REDIS_URL=redis://localhost:6379
CACHE_ENABLED=true
CACHE_TTL=1800  # 30 minutos

# Integración con otros servicios ML
ML_SUPERVISED_URL=http://localhost:8001
ML_UNSUPERVISED_URL=http://localhost:8002

# API Key de Groq (debug)
GROQ_MAX_TOKENS=2048
GROQ_TIMEOUT=30
```

**Copiar desde ejemplo:**
```bash
cp .env.example .env
# Luego edita .env con tus valores
```

---

## 🛠️ Instalación Detallada

### Requisitos Previos

```bash
# Python 3.12+
python --version
# Output: Python 3.12.x

# pip
pip --version

# PostgreSQL (si no tienes una instancia compartida)
# Descargar desde: https://www.postgresql.org/download/
```

### Instalación Paso a Paso

#### 1. Clonar/Acceder al directorio

```bash
cd D:\PLATAFORMA\ EDUCATIVA\agente
```

#### 2. Crear entorno virtual (si no existe)

```bash
# Windows
python -m venv venv
.\venv\Scripts\activate

# macOS/Linux
python3 -m venv venv
source venv/bin/activate
```

#### 3. Actualizar pip

```bash
python -m pip install --upgrade pip
```

#### 4. Instalar dependencias

```bash
pip install -r requirements.txt
```

**Dependencias instaladas:**
- `langchain` - Framework para aplicaciones LLM
- `langchain-groq` - Integración Groq
- `fastapi` - Framework web
- `uvicorn` - Servidor ASGI
- `pydantic` - Validación de datos
- `python-dotenv` - Variables de entorno

#### 5. Verificar instalación

```bash
python -c "import langchain; import fastapi; print('✅ Instalación OK')"
```

---

## 🎯 Iniciar el Servicio

### Opción 1: Ejecución Directa

```bash
cd D:\PLATAFORMA\ EDUCATIVA\agente

# Activar entorno (si no está activo)
.\venv\Scripts\activate

# Ejecutar servicio
python agent_service.py
```

**Output esperado:**
```
INFO:     Application startup complete
LLM Available: True ✅
GROQ_MODEL: mixtral-8x7b-32768
INFO:     Uvicorn running on http://0.0.0.0:8003 (Press CTRL+C to quit)
```

### Opción 2: Con UV icorn Directo

```bash
uvicorn agent_service:app --host 0.0.0.0 --port 8003 --reload
```

### Opción 3: Docker (Producción)

```bash
# Construir imagen
docker build -t agente-service:latest .

# Ejecutar contenedor
docker run -p 8003:8003 \
  -e GROQ_API_KEY=gsk_xxxxx \
  -e PORT=8003 \
  agente-service:latest
```

---

## 📡 API Endpoints

### Documentation Automática

```
Swagger UI: http://localhost:8003/docs
ReDoc: http://localhost:8003/redoc
OpenAPI JSON: http://localhost:8003/openapi.json
```

### Endpoints Principales

#### 1. Health Check

```bash
GET /health
```

**Response:**
```json
{
  "status": "ok",
  "llm_available": true,
  "model": "mixtral-8x7b-32768"
}
```

#### 2. Síntesis de Descubrimientos

```bash
POST /api/synthesis
```

**Request:**
```json
{
  "student_id": 123,
  "supervised_results": {
    "risk_level": "high",
    "factors": ["bajo_rendimiento", "inasistencia"]
  },
  "unsupervised_results": {
    "cluster": "at_risk",
    "probability": 0.85
  }
}
```

**Response:**
```json
{
  "synthesis": "Descripción general de la situación del estudiante",
  "risk_assessment": "Análisis detallado del riesgo",
  "recommendations": [
    {
      "type": "tutoring",
      "urgency": "high",
      "action": "Tutoría individual inmediata"
    }
  ],
  "intervention_strategy": "Plan de intervención específico",
  "followup_date": "2025-12-01"
}
```

#### 3. Generar Recomendación Personalizada

```bash
POST /api/recommend
```

**Request:**
```json
{
  "student_id": 123,
  "context": "Estudiante con bajo rendimiento en matemáticas"
}
```

**Response:**
```json
{
  "recommendation": "Texto de recomendación generado por LLM",
  "resources": ["material_apoyo_1", "tutoria_online"],
  "priority": "high"
}
```

#### 4. Generar Plan de Intervención

```bash
POST /api/intervention-plan
```

**Request:**
```json
{
  "student_id": 123,
  "risk_profile": {
    "academic": 0.7,
    "behavioral": 0.3,
    "social": 0.5
  }
}
```

**Response:**
```json
{
  "plan": "Plan detallado de intervención",
  "phases": [
    {
      "phase": 1,
      "duration": "2 weeks",
      "actions": ["Tutoría", "Comunicación con padres"]
    }
  ],
  "success_metrics": ["Mejora de calificaciones", "Asistencia 100%"]
}
```

---

## 🔄 Integración con Otros Servicios

### Arquitectura General

```
┌─────────────────────────────────────────────────────────┐
│          Plataforma Educativa (Laravel)                 │
│  http://localhost:8000                                  │
└────────────────────┬────────────────────────────────────┘
                     │
        ┌────────────┼────────────┐
        │            │            │
        ▼            ▼            ▼
  ┌──────────┐ ┌──────────┐ ┌──────────────┐
  │ Agente   │ │ Supervisado     │ No Supervisado
  │ LLM      │ │ ML       │ ML
  │ 8003     │ │ 8001     │ 8002
  └──────────┘ └──────────┘ └──────────────┘
```

### Llamadas entre Servicios

**El Agente consume datos de:**
- `http://localhost:8001/api/prediction` - Predicciones supervisadas
- `http://localhost:8002/api/clustering` - Clustering no supervisado

**La Plataforma consume del Agente:**
- `http://localhost:8003/api/synthesis` - Síntesis
- `http://localhost:8003/api/recommend` - Recomendaciones

---

## 🧪 Testing

### Test de Salud

```bash
curl http://localhost:8003/health
```

**Output esperado:**
```json
{
  "status": "ok",
  "llm_available": true,
  "model": "mixtral-8x7b-32768"
}
```

### Test de Síntesis

```bash
curl -X POST http://localhost:8003/api/synthesis \
  -H "Content-Type: application/json" \
  -d '{
    "student_id": 123,
    "supervised_results": {
      "risk_level": "medium",
      "factors": ["bajo_rendimiento"]
    },
    "unsupervised_results": {
      "cluster": "at_risk",
      "probability": 0.7
    }
  }'
```

### Test desde Python

```python
import requests

# Health check
response = requests.get('http://localhost:8003/health')
print(response.json())

# Synthesis
response = requests.post('http://localhost:8003/api/synthesis', json={
    "student_id": 123,
    "supervised_results": {"risk_level": "high"},
    "unsupervised_results": {"cluster": "at_risk", "probability": 0.85}
})
print(response.json())
```

---

## 📊 Logs y Debugging

### Ver Logs en Tiempo Real

```bash
# Mientras el servicio está corriendo, en otra terminal:
tail -f logs/agent.log
```

### Habilitar Debug

**En `.env`:**
```ini
DEBUG=true
LOG_LEVEL=DEBUG
```

**Luego reinicia el servicio:**
```bash
python agent_service.py
```

### Verificar LLM Connection

```python
# Crear archivo test_llm.py
from langchain_groq import ChatGroq
from dotenv import load_dotenv

load_dotenv()

# Esto mostrará si Groq está disponible
chat = ChatGroq(model="mixtral-8x7b-32768")
response = chat.invoke("Hola, ¿estás funcionando?")
print(response.content)
```

**Ejecutar:**
```bash
python test_llm.py
```

---

## 🚨 Troubleshooting

### Error: "GROQ_API_KEY no está configurada"

**Solución:**
```bash
# 1. Edita .env
nano .env  # o abre en tu editor

# 2. Añade tu API key
GROQ_API_KEY=gsk_xxxxxxxxxxxxx

# 3. Reinicia el servicio
python agent_service.py
```

### Error: "LLM Available: False"

**Causas posibles:**
1. `GROQ_API_KEY` está vacía → Verifica .env
2. Clave API inválida → Obtén nueva desde https://console.groq.com
3. Sin conexión a internet → Verifica conectividad
4. Timeout de Groq → Intenta de nuevo

**Solución:**
```bash
# Verificar GROQ_API_KEY
echo %GROQ_API_KEY%  # Windows
# o
echo $GROQ_API_KEY   # macOS/Linux

# Probar conexión a Groq
curl https://api.groq.com/health  # Debería responder
```

### Error: "Port 8003 already in use"

**Solución:**
```bash
# Opción 1: Usar puerto diferente
python agent_service.py --port 8004

# Opción 2: Matar proceso anterior
# Windows:
netstat -ano | findstr :8003
taskkill /PID <PID> /F

# macOS/Linux:
lsof -ti:8003 | xargs kill -9
```

### Error: "Dependencies not found"

**Solución:**
```bash
# Reinstalar dependencias
pip install --upgrade -r requirements.txt

# Eliminar caché de pip
pip cache purge

# Reinstalar cleanly
pip uninstall langchain langchain-groq -y
pip install -r requirements.txt
```

### Error: "Database connection failed"

**Solución:**
```bash
# Verificar DATABASE_URL en .env
# Formato: postgresql://user:password@host:port/dbname

# Probar conexión con psql
psql postgresql://user:password@localhost:5432/educativa_db

# Si no funciona, usar SQLite localmente (desarrollo)
DATABASE_URL=sqlite:///./agent.db
```

---

## 🌐 Deployment (Railway)

### Configuración en Railway

1. **Railway Project** está en `agente/railway.json`:

```json
{
  "name": "agente-synthesis-service",
  "runtime": "python",
  "buildCommand": "pip install -r requirements.txt",
  "startCommand": "uvicorn agent_service:app --host 0.0.0.0 --port $PORT"
}
```

2. **Environment Variables en Railway:**

```
GROQ_API_KEY=gsk_xxxxxxxxxxxxx
PORT=8080  (Railway la establece automáticamente)
ENVIRONMENT=production
DEBUG=false
DATABASE_URL=postgresql://...  (Railway proporciona)
REDIS_URL=redis://...  (Railway proporciona)
```

3. **Desplegar:**

```bash
# Usando Railway CLI
railway login
railway up
```

### Monitorear Despliegue

- Logs: Dashboard de Railway
- Health: `https://tu-dominio-agente.railway.app/health`
- Docs: `https://tu-dominio-agente.railway.app/docs`

---

## 📈 Performance y Optimización

### Métricas Actuales

```
Tiempo de síntesis:   ~2-5 segundos (Groq)
Cached response:      <100ms
Concurrent requests:  10+
Memory usage:         ~200-300MB
```

### Optimizaciones

```python
# 1. Caché Redis (habilitada)
CACHE_ENABLED=true
CACHE_TTL=1800

# 2. Connection pooling
# Implementado en config.py

# 3. Async endpoints
# Usar FastAPI async para better concurrency
```

---

## 📚 Estructura del Proyecto

```
agente/
├── agent_service.py          # Main FastAPI app ⭐
├── recommendation_agent.py   # LLM agent logic
├── config.py                 # Configuration
├── prompts.py                # Prompt templates
├── requirements.txt          # Dependencies
├── .env                       # Variables (no commitear!)
├── .env.example              # Plantilla .env
├── Dockerfile                # Para Railway
├── railway.json              # Config Railway
├── README.md                 # Este archivo
└── logs/                     # Logs directory
    └── agent.log
```

### Archivos Clave

| Archivo | Propósito |
|---------|-----------|
| `agent_service.py` | Aplicación FastAPI principal, endpoints |
| `recommendation_agent.py` | Lógica del agente LLM, síntesis |
| `config.py` | Configuración, variables, defaults |
| `prompts.py` | Plantillas de prompts para LLM |

---

## 🔐 Seguridad

### Mejores Prácticas

```python
# ✅ NO commitear .env
git ignore .env

# ✅ Usar variables de entorno
GROQ_API_KEY=gsk_xxxxx

# ✅ CORS limitado en producción
CORS_ORIGINS=https://tu-dominio.com

# ✅ Token de autenticación (Sanctum)
LARAVEL_SANCTUM_ENABLED=true
```

### En Producción

```ini
# .env (producción)
ENVIRONMENT=production
DEBUG=false
CORS_ORIGINS=https://tu-dominio-educativa.com
SECRET_KEY=cambiar-esto-en-produccion
```

---

## 🤝 Integración con Plataforma

### Desde Laravel

```php
// En plataforma-educativa/app/Services/AgentService.php

class AgentService {
    public function synthesizeStudent($studentId) {
        $response = Http::post('http://localhost:8003/api/synthesis', [
            'student_id' => $studentId,
            'supervised_results' => [...],
            'unsupervised_results' => [...]
        ]);

        return $response->json();
    }
}
```

### Usar en React

```typescript
// En plataforma-educativa/resources/js/services/agentApi.ts

const agentApi = {
    async getSynthesis(studentId: number) {
        const response = await axios.post('http://localhost:8003/api/synthesis', {
            student_id: studentId,
            supervised_results: {...},
            unsupervised_results: {...}
        });
        return response.data;
    }
};
```

---

## 📞 Contacto y Soporte

### Verificar Status

```bash
# Health endpoint
curl http://localhost:8003/health

# Logs
tail -f logs/agent.log

# Groq API Status
curl https://status.groq.com
```

### Documentación Groq

- API Keys: https://console.groq.com/keys
- Modelos: https://console.groq.com/docs/models
- Rate Limits: https://console.groq.com/docs/rate-limits

### Documentación FastAPI

- Tutorial: https://fastapi.tiangolo.com/
- Deployment: https://fastapi.tiangolo.com/deployment/

---

## ✅ Checklist de Inicio

- [ ] Python 3.12+ instalado
- [ ] Entorno virtual creado
- [ ] Dependencias instaladas (`pip install -r requirements.txt`)
- [ ] GROQ_API_KEY obtenida desde console.groq.com
- [ ] `.env` configurado con GROQ_API_KEY
- [ ] `python agent_service.py` ejecutándose
- [ ] Health endpoint: `http://localhost:8003/health` respondiendo
- [ ] LLM Available: True en startup logs
- [ ] Swagger UI: `http://localhost:8003/docs` accesible

---

## 🎉 ¡Listo!

Tu servicio Agente está configurado y funcionando. Accede a:

```
Swagger UI (Interactive Docs):  http://localhost:8003/docs
ReDoc (Alternative Docs):       http://localhost:8003/redoc
Health Check:                   http://localhost:8003/health
```

**¿Problemas?** Revisa la sección [Troubleshooting](#-troubleshooting) o verifica los logs en `logs/agent.log`.
