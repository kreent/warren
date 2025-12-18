#!/bin/bash

# ============================================================================
# Script de despliegue para Warren Screener con Cloud Storage Cache
# ============================================================================

set -e  # Salir si hay errores

# Colores para output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Función para imprimir con colores
print_info() { echo -e "${BLUE}ℹ ${NC}$1"; }
print_success() { echo -e "${GREEN}✓ ${NC}$1"; }
print_warning() { echo -e "${YELLOW}⚠ ${NC}$1"; }
print_error() { echo -e "${RED}✗ ${NC}$1"; }

# ============================================================================
# CONFIGURACIÓN - EDITA ESTOS VALORES
# ============================================================================

PROJECT_ID="tu-project-id"              # Tu Project ID de GCP
REGION="us-central1"                    # Región de despliegue
SERVICE_NAME="warren-screener"          # Nombre del servicio
BUCKET_NAME="warren-screener-cache"     # Nombre del bucket

# ============================================================================
# VALIDACIÓN
# ============================================================================

echo ""
print_info "🚀 Iniciando despliegue de Warren Screener con Cache"
echo ""

# Validar que gcloud esté instalado
if ! command -v gcloud &> /dev/null; then
    print_error "gcloud no está instalado. Instálalo desde: https://cloud.google.com/sdk"
    exit 1
fi

# Validar que gsutil esté instalado
if ! command -v gsutil &> /dev/null; then
    print_error "gsutil no está instalado. Viene con gcloud SDK."
    exit 1
fi

# Validar configuración
if [ "$PROJECT_ID" = "tu-project-id" ]; then
    print_error "Por favor, edita el script y configura tu PROJECT_ID"
    exit 1
fi

# Configurar proyecto
print_info "Configurando proyecto: $PROJECT_ID"
gcloud config set project $PROJECT_ID
print_success "Proyecto configurado"

# ============================================================================
# PASO 1: CREAR BUCKET DE CLOUD STORAGE
# ============================================================================

echo ""
print_info "📦 Paso 1/4: Verificando bucket de Cloud Storage..."

if gsutil ls -b gs://$BUCKET_NAME &>/dev/null; then
    print_success "Bucket ya existe: gs://$BUCKET_NAME"
else
    print_info "Creando bucket: gs://$BUCKET_NAME"
    gsutil mb -l $REGION gs://$BUCKET_NAME
    print_success "Bucket creado exitosamente"
fi

# ============================================================================
# PASO 2: CONFIGURAR PERMISOS
# ============================================================================

echo ""
print_info "🔐 Paso 2/4: Configurando permisos IAM..."

# Obtener el service account del proyecto
SERVICE_ACCOUNT=$(gcloud iam service-accounts list \
    --filter="displayName:Compute Engine default service account" \
    --format="value(email)")

if [ -z "$SERVICE_ACCOUNT" ]; then
    print_warning "No se encontró service account por defecto"
    SERVICE_ACCOUNT="$PROJECT_ID@appspot.gserviceaccount.com"
    print_info "Usando: $SERVICE_ACCOUNT"
fi

# Dar permisos al bucket
print_info "Otorgando permisos a: $SERVICE_ACCOUNT"
gsutil iam ch serviceAccount:$SERVICE_ACCOUNT:objectAdmin gs://$BUCKET_NAME
print_success "Permisos configurados"

# ============================================================================
# PASO 3: BUILD DE LA IMAGEN DOCKER
# ============================================================================

echo ""
print_info "🏗️  Paso 3/4: Building imagen Docker..."
print_info "Esto puede tardar 2-3 minutos..."

# Habilitar APIs necesarias
print_info "Habilitando APIs de GCP..."
gcloud services enable cloudbuild.googleapis.com --quiet
gcloud services enable run.googleapis.com --quiet
gcloud services enable storage-api.googleapis.com --quiet

# Build de la imagen
gcloud builds submit --tag gcr.io/$PROJECT_ID/$SERVICE_NAME --quiet

print_success "Imagen Docker creada: gcr.io/$PROJECT_ID/$SERVICE_NAME"

# ============================================================================
# PASO 4: DEPLOY EN CLOUD RUN
# ============================================================================

echo ""
print_info "☁️  Paso 4/4: Desplegando en Cloud Run..."
print_info "Esto puede tardar 1-2 minutos..."

gcloud run deploy $SERVICE_NAME \
    --image gcr.io/$PROJECT_ID/$SERVICE_NAME \
    --platform managed \
    --region $REGION \
    --allow-unauthenticated \
    --memory 2Gi \
    --timeout 600s \
    --cpu 2 \
    --min-instances 0 \
    --max-instances 10 \
    --set-env-vars GCS_BUCKET_NAME=$BUCKET_NAME \
    --quiet

# ============================================================================
# RESUMEN FINAL
# ============================================================================

echo ""
echo "============================================================================"
print_success "¡DESPLIEGUE COMPLETADO EXITOSAMENTE! 🎉"
echo "============================================================================"
echo ""

# Obtener URL del servicio
SERVICE_URL=$(gcloud run services describe $SERVICE_NAME \
    --region=$REGION \
    --format="get(status.url)")

print_info "📊 INFORMACIÓN DEL SERVICIO:"
echo ""
echo "  🌐 URL del servicio:"
echo "     $SERVICE_URL"
echo ""
echo "  💾 Bucket de caché:"
echo "     gs://$BUCKET_NAME"
echo ""
echo "  📍 Región:"
echo "     $REGION"
echo ""

echo "============================================================================"
print_info "📚 ENDPOINTS DISPONIBLES:"
echo "============================================================================"
echo ""
echo "  1. Análisis (con caché de 24h):"
echo "     $SERVICE_URL/analyze"
echo ""
echo "  2. Estado del caché:"
echo "     $SERVICE_URL/cache-status"
echo ""
echo "  3. Limpiar caché:"
echo "     $SERVICE_URL/clear-cache"
echo ""
echo "  4. Health check:"
echo "     $SERVICE_URL/health"
echo ""

echo "============================================================================"
print_info "🧪 COMANDOS DE PRUEBA:"
echo "============================================================================"
echo ""
echo "  # Ver estado del caché"
echo "  curl $SERVICE_URL/cache-status"
echo ""
echo "  # Ejecutar análisis"
echo "  curl $SERVICE_URL/analyze"
echo ""
echo "  # Ver logs en tiempo real"
echo "  gcloud run services logs tail $SERVICE_NAME --region=$REGION"
echo ""

echo "============================================================================"
print_info "💡 PRÓXIMOS PASOS:"
echo "============================================================================"
echo ""
echo "  1. La primera petición tardará ~4 minutos (sin caché)"
echo "  2. Las siguientes peticiones serán instantáneas (~200ms)"
echo "  3. El caché se renueva automáticamente cada 24 horas"
echo "  4. Puedes forzar actualización con /clear-cache"
echo ""

print_success "¡Listo para usar! 🚀"
echo ""
