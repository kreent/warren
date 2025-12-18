#!/bin/bash

# Script de despliegue para Cloud Run - Warren Screener Optimizado
# ====================================================================

set -e  # Salir si hay algún error

# Configuración
PROJECT_ID="19630f83-fd0f-4611-8547-c6aea71e3893"  # CAMBIAR por tu project ID
REGION="europe-west1"
SERVICE_NAME="warren-screener"
IMAGE_NAME="gcr.io/${PROJECT_ID}/${SERVICE_NAME}:latest"

echo "========================================="
echo "Desplegando Warren Screener Optimizado"
echo "========================================="
echo ""
echo "Proyecto: ${PROJECT_ID}"
echo "Región: ${REGION}"
echo "Servicio: ${SERVICE_NAME}"
echo ""

# Paso 1: Construir la imagen Docker
echo "📦 Paso 1/3: Construyendo imagen Docker..."
docker build -t ${IMAGE_NAME} .

# Paso 2: Subir imagen a Google Container Registry
echo ""
echo "☁️  Paso 2/3: Subiendo imagen a GCR..."
docker push ${IMAGE_NAME}

# Paso 3: Desplegar a Cloud Run
echo ""
echo "🚀 Paso 3/3: Desplegando a Cloud Run..."
gcloud run deploy ${SERVICE_NAME} \
  --image ${IMAGE_NAME} \
  --platform managed \
  --region ${REGION} \
  --allow-unauthenticated \
  --cpu 2 \
  --memory 2Gi \
  --timeout 300 \
  --concurrency 10 \
  --max-instances 20 \
  --min-instances 0 \
  --port 8080 \
  --set-env-vars "PYTHONUNBUFFERED=1"

echo ""
echo "✅ Despliegue completado!"
echo ""
echo "URL del servicio:"
gcloud run services describe ${SERVICE_NAME} --region ${REGION} --format 'value(status.url)'
echo ""
echo "Endpoints disponibles:"
echo "  - GET / (información del servicio)"
echo "  - GET /analyze (análisis principal - con caché)"
echo "  - GET /analyze?force_refresh=1 (forzar nuevo análisis)"
echo "  - GET /health (health check)"
echo ""
echo "🔍 Para ver los logs:"
echo "gcloud logging read \"resource.type=cloud_run_revision AND resource.labels.service_name=${SERVICE_NAME}\" --limit 50 --format json"
