# =====================================================
# Stage 1: Builder
# =====================================================

FROM python:3.11-slim AS builder

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .

RUN pip install --upgrade pip && \
    pip install --user -r requirements.txt

# =====================================================
# Stage 2: Runtime
# =====================================================

FROM python:3.11-slim

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PATH=/home/mluser/.local/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin

RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

RUN useradd -m -u 1000 mluser

COPY --from=builder /root/.local /home/mluser/.local
RUN chown -R mluser:mluser /home/mluser/.local

COPY --chown=mluser:mluser . .

# Asegurar permisos correctos para mluser
RUN chmod -R 755 /app && \
    mkdir -p /app/logs /app/__pycache__ && \
    chmod -R 777 /app/logs && \
    chown -R mluser:mluser /app /home/mluser && \
    chmod -R u+rwx /home/mluser && \
    chmod -R g+rx /app && \
    chmod -R o+rx /app

# Configuración de puerto (local=8003, producción=8080)
# Por defecto 8080 para producción, pero respeta variable PORT de docker-compose
ARG PORT=8080
ENV PORT=${PORT}

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=40s --retries=3 \
    CMD curl -f http://localhost:8080/health || exit 1

USER mluser

# Exponer puertos comunes (8003 para desarrollo, 8080 para producción)
EXPOSE 8003 8080

# Entry point: uvicorn respeta la variable PORT de entorno
# Local (docker-compose): PORT=8003
# Producción (Railway): PORT=8080 (por defecto o variable de Railway)
CMD sh -c 'uvicorn agent_service:app --host 0.0.0.0 --port ${PORT:-8080} --access-log'
