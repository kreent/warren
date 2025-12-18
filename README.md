# Warren Screener - Versión Optimizada para Cloud Run

## 🚀 Mejoras Implementadas

### 1. **Caché en Memoria (Principal mejora)**
- ✅ Resultados del análisis completo se cachean por **1 hora**
- ✅ Listados de S&P500 y Nasdaq-100 se cachean permanentemente (se actualizan raramente)
- ✅ Fundamentales de cada ticker se cachean usando `@lru_cache`
- ✅ **Reducción estimada de tiempo: de 4 minutos a ~2-5 segundos** en requests subsecuentes

### 2. **Servidor de Producción (Gunicorn)**
- ✅ Reemplaza el servidor de desarrollo de Flask
- ✅ Configurado con 2 workers y 4 threads
- ✅ Worker class: `gthread` para aprovechar threading
- ✅ Timeout de 300 segundos para análisis inicial
- ✅ Mejor manejo de concurrencia y estabilidad

### 3. **Procesamiento Paralelo**
- ✅ Descarga de históricos usando `ThreadPoolExecutor`
- ✅ Hasta 10 descargas simultáneas (configurable)
- ✅ **Reducción de tiempo de descarga: ~60%**

### 4. **Recursos Mejorados**
- ✅ CPU: aumentado a **2 vCPU** (era 1)
- ✅ Memoria: aumentada a **2 GiB** (era 512 MiB)
- ✅ Concurrencia: reducida a **10** (era 80) - más adecuado para scripts CPU-intensivos
- ✅ Startup CPU Boost habilitado

### 5. **Health Checks y Monitoreo**
- ✅ Endpoint `/health` para verificar estado
- ✅ Información de caché en respuesta JSON
- ✅ Logs mejorados con timestamps

## 📊 Comparación de Tiempos Estimados

| Escenario | Versión Original | Versión Optimizada | Mejora |
|-----------|-----------------|-------------------|--------|
| Primera request (caché frío) | ~270 segundos | ~90-120 segundos | ~60% |
| Requests subsecuentes (caché caliente) | ~270 segundos | ~2-5 segundos | **~98%** |
| Descarga de históricos | ~120 segundos | ~40-50 segundos | ~60% |

## 🔧 Configuración del Caché

### Ajustar TTL (Time To Live)
En `main_optimized.py`, línea 54:
```python
CACHE_TTL_SECONDS = 3600  # 1 hora por defecto
```

**Recomendaciones según caso de uso:**
- Datos muy dinámicos (intraday): `300` (5 minutos)
- Datos diarios estándar: `3600` (1 hora) ✅ **Recomendado**
- Datos que cambian poco: `7200` (2 horas)
- Solo necesitas 1 análisis al día: `86400` (24 horas)

### Forzar Refresh del Caché
```bash
curl "https://TU-SERVICE-URL/analyze?force_refresh=1"
```

## 📦 Archivos del Proyecto

```
proyecto/
├── main_optimized.py      # Script principal optimizado
├── requirements.txt       # Dependencias Python
├── Dockerfile            # Configuración Docker
├── cloudrun-config.yaml  # Configuración Cloud Run (opcional)
├── deploy.sh             # Script de despliegue automatizado
└── README.md             # Este archivo
```

## 🚀 Instrucciones de Despliegue

### Opción 1: Usando el script de despliegue (Recomendado)

1. **Editar configuración:**
   ```bash
   nano deploy.sh
   # Cambiar PROJECT_ID por tu proyecto real
   ```

2. **Hacer ejecutable y desplegar:**
   ```bash
   chmod +x deploy.sh
   ./deploy.sh
   ```

### Opción 2: Paso a paso manual

1. **Configurar variables:**
   ```bash
   export PROJECT_ID="tu-proyecto-id"
   export REGION="europe-west1"
   export SERVICE_NAME="warren-screener"
   ```

2. **Construir imagen:**
   ```bash
   docker build -t gcr.io/${PROJECT_ID}/${SERVICE_NAME}:latest .
   ```

3. **Subir a GCR:**
   ```bash
   docker push gcr.io/${PROJECT_ID}/${SERVICE_NAME}:latest
   ```

4. **Desplegar:**
   ```bash
   gcloud run deploy ${SERVICE_NAME} \
     --image gcr.io/${PROJECT_ID}/${SERVICE_NAME}:latest \
     --platform managed \
     --region ${REGION} \
     --allow-unauthenticated \
     --cpu 2 \
     --memory 2Gi \
     --timeout 300 \
     --concurrency 10 \
     --max-instances 20 \
     --port 8080
   ```

## 🔍 Endpoints Disponibles

### 1. Información del servicio
```bash
GET /
```
Retorna información sobre el servicio y las optimizaciones aplicadas.

### 2. Análisis principal (con caché)
```bash
GET /analyze
```
Retorna el análisis. Si el caché es válido (< 1 hora), responde inmediatamente.

**Respuesta incluye:**
```json
{
  "total_analyzed": 150,
  "candidates_count": 25,
  "top_10": [...],
  "cached_at": "2024-01-15T10:30:00",
  "cache_ttl_seconds": 3600,
  "from_cache": true,
  "cache_age_seconds": 245
}
```

### 3. Forzar nuevo análisis
```bash
GET /analyze?force_refresh=1
```
Invalida el caché y ejecuta un nuevo análisis completo.

### 4. Health check
```bash
GET /health
```
Verifica que el servicio está funcionando.

## 📈 Monitoreo

### Ver logs en tiempo real:
```bash
gcloud logging tail "resource.type=cloud_run_revision AND resource.labels.service_name=warren-screener" --format json
```

### Ver últimos 50 logs:
```bash
gcloud logging read "resource.type=cloud_run_revision AND resource.labels.service_name=warren-screener" --limit 50
```

### Ver métricas en Cloud Console:
1. Ir a Cloud Run → warren-screener
2. Pestaña "METRICS"
3. Revisar:
   - Latencia (debería bajar drásticamente)
   - Request count
   - CPU y Memory utilization

## 🎯 Optimizaciones Adicionales Futuras

Si aún necesitas más velocidad, considera:

### 1. **Memorystore for Redis** (Caché distribuido)
- Para compartir caché entre múltiples instancias
- Persistencia del caché aunque las instancias se reinicien
- Ver guía en el documento original

### 2. **Cloud Tasks para análisis asíncrono**
- Ejecutar análisis en background
- Retornar inmediatamente un ID de tarea
- Consultar resultados después

### 3. **BigQuery para almacenar resultados**
- Guardar análisis históricos
- Queries rápidas sobre datos pasados
- Analytics avanzados

### 4. **Reducir universo de tickers**
- Cambiar `UNIVERSE_LIMIT = 250` a `150` o menos
- Análisis más rápido con menos tickers

## 🐛 Troubleshooting

### Error: Timeout después de 300 segundos
**Solución:** Incrementar timeout en deploy:
```bash
--timeout 600  # 10 minutos
```

### Error: Out of Memory
**Solución:** Aumentar memoria:
```bash
--memory 4Gi
```

### Caché no funciona entre requests
**Causa:** Cloud Run escala a cero y cada nueva instancia tiene caché vacío.
**Solución:** Configurar instancias mínimas:
```bash
--min-instances 1
```

### Análisis sigue siendo lento en primera request
**Opciones:**
1. Reducir `UNIVERSE_LIMIT` (línea 21 del código)
2. Reducir `MAX_FUND_REQS` (línea 23 del código)
3. Aumentar workers de Gunicorn en Dockerfile

## 💡 Consejos Pro

1. **Warm-up automático:** Configurar Cloud Scheduler para llamar `/analyze` cada 50 minutos mantiene el caché caliente.

2. **Monitorear uso de caché:** Revisar el campo `from_cache` en las respuestas para verificar efectividad.

3. **Ajustar concurrencia:** Si el análisis es muy CPU-intensivo, reducir a `--concurrency 5` o menos.

4. **Costos:** Con caché, la mayoría de requests responden en segundos → menores costos de CPU.

## 📞 Soporte

Si tienes dudas sobre la implementación o necesitas ajustes adicionales, revisa:
- Logs de Cloud Run
- Métricas de latencia
- Campo `from_cache` en respuestas

¡Buena suerte con tu screener optimizado! 🚀
