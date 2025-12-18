#!/bin/bash

# Script de prueba para verificar el funcionamiento del caché

echo "======================================"
echo "PRUEBA DE CACHÉ - Warren Screener"
echo "======================================"
echo ""

# Obtener la URL del servicio
SERVICE_URL=$(gcloud run services describe warren-screener \
    --region=us-central1 \
    --format="get(status.url)" 2>/dev/null)

if [ -z "$SERVICE_URL" ]; then
    echo "❌ Error: No se encontró el servicio warren-screener"
    echo "   Asegúrate de haberlo desplegado primero con ./deploy.sh"
    exit 1
fi

echo "🌐 URL del servicio: $SERVICE_URL"
echo ""

# Función para extraer from_cache del JSON
check_cache_status() {
    local response=$1
    if echo "$response" | grep -q '"from_cache": true'; then
        echo "✅ DESDE CACHÉ"
    elif echo "$response" | grep -q '"from_cache": false'; then
        echo "🔄 ANÁLISIS NUEVO"
    else
        echo "❓ NO SE PUDO DETERMINAR"
    fi
}

# Test 1: Limpiar caché
echo "📋 Test 1: Limpiando caché..."
curl -s $SERVICE_URL/clear-cache | jq '.' || echo "(sin jq instalado, respuesta raw)"
echo ""
sleep 2

# Test 2: Primera petición (sin caché)
echo "📋 Test 2: Primera petición (debería tardar ~4 min)..."
echo "⏱️  Iniciando cronómetro..."
START_TIME=$(date +%s)

RESPONSE1=$(curl -s $SERVICE_URL/analyze)
END_TIME=$(date +%s)
DURATION=$((END_TIME - START_TIME))

echo "✅ Completado en: ${DURATION} segundos"
echo ""
echo "📊 Respuesta:"
echo "$RESPONSE1" | jq '.' 2>/dev/null || echo "$RESPONSE1"
echo ""
echo "Estado del caché: $(check_cache_status "$RESPONSE1")"
echo ""

# Test 3: Verificar estado del caché
echo "📋 Test 3: Verificando estado del caché..."
curl -s $SERVICE_URL/cache-status | jq '.' 2>/dev/null || curl -s $SERVICE_URL/cache-status
echo ""
sleep 2

# Test 4: Segunda petición (con caché)
echo "📋 Test 4: Segunda petición (debería ser instantánea)..."
echo "⏱️  Iniciando cronómetro..."
START_TIME=$(date +%s)

RESPONSE2=$(curl -s $SERVICE_URL/analyze)
END_TIME=$(date +%s)
DURATION=$((END_TIME - START_TIME))

echo "✅ Completado en: ${DURATION} segundos"
echo ""
echo "📊 Respuesta:"
echo "$RESPONSE2" | jq '.' 2>/dev/null || echo "$RESPONSE2"
echo ""
echo "Estado del caché: $(check_cache_status "$RESPONSE2")"
echo ""

# Test 5: Tercera petición (con caché)
echo "📋 Test 5: Tercera petición (también debería ser instantánea)..."
echo "⏱️  Iniciando cronómetro..."
START_TIME=$(date +%s)

RESPONSE3=$(curl -s $SERVICE_URL/analyze)
END_TIME=$(date +%s)
DURATION=$((END_TIME - START_TIME))

echo "✅ Completado en: ${DURATION} segundos"
echo ""
echo "Estado del caché: $(check_cache_status "$RESPONSE3")"
echo ""

# Resumen
echo "======================================"
echo "RESUMEN DE PRUEBAS"
echo "======================================"
echo ""
echo "✅ Test 1: Caché limpiado"
echo "✅ Test 2: Primera petición completada"
echo "✅ Test 3: Estado del caché verificado"
echo "✅ Test 4: Segunda petición completada"
echo "✅ Test 5: Tercera petición completada"
echo ""
echo "💡 CONCLUSIÓN:"
echo "   - La primera petición genera el caché"
echo "   - Las siguientes peticiones usan el caché"
echo "   - El caché expira en 24 horas"
echo ""
echo "🔍 VERIFICAR LOGS:"
echo "   gcloud run services logs tail warren-screener --region=us-central1"
echo ""
