# Warren Screener - Caché de 24 Horas con Cloud Storage

## 🎯 Problema Resuelto

Tu servicio tardaba **4 minutos** en cada petición. Ahora:
- ✅ Primera petición: 4 minutos (genera caché)
- ✅ Siguientes peticiones: **~200 milisegundos** (usa caché)
- ✅ Caché válido por 24 horas
- ✅ Costo: ~$5/mes (casi gratis)

## 🚀 Despliegue Rápido (3 minutos)

### 1. Edita el script de despliegue

```bash
nano deploy.sh
```

Cambia esta línea:
```bash
PROJECT_ID="tu-project-id"  # ← Pon tu PROJECT_ID aquí
```

### 2. Ejecuta el script

```bash
chmod +x deploy.sh
./deploy.sh
```

¡Eso es todo! El script hará todo automáticamente:
- ✅ Crea el bucket de Cloud Storage
- ✅ Configura permisos
- ✅ Build de la imagen Docker
- ✅ Deploy en Cloud Run

## 📊 Endpoints Disponibles

Una vez desplegado, tendrás estos endpoints:

### 1. `/analyze` - Análisis principal
```bash
curl https://TU_URL/analyze
```

**Primera vez (sin caché):**
```json
{
  "total_analyzed": 245,
  "candidates_count": 18,
  "execution_time_seconds": 240.5
}
```

**Siguientes veces (con caché):**
```json
{
  "total_analyzed": 245,
  "candidates_count": 18,
  "execution_time_seconds": 0.2
}
```

### 2. `/cache-status` - Estado del caché
```bash
curl https://TU_URL/cache-status
```

```json
{
  "cache_enabled": true,
  "cache_exists": true,
  "time_remaining_hours": 18.5,
  "results_count": 245
}
```

### 3. `/clear-cache` - Forzar actualización
```bash
curl https://TU_URL/clear-cache
```

### 4. `/health` - Health check
```bash
curl https://TU_URL/health
```

## 📈 Métricas de Mejora

| Métrica | Antes | Después |
|---------|-------|---------|
| Primera petición | 4 min | 4 min |
| Peticiones subsecuentes | 4 min | 0.2 seg |
| Costo por petición | Alto | Bajo |
| Experiencia usuario | ❌ Lenta | ✅ Instantánea |

## 🔧 Configuración Avanzada

### Cambiar duración del caché

Edita `main.py`, línea 27:

```python
CACHE_TTL_HOURS = 24  # Cambiar a 12, 48, etc.
```

### Actualización automática diaria

Crear un Cloud Scheduler que limpie el caché cada mañana:

```bash
gcloud scheduler jobs create http warren-daily-update \
    --schedule="0 6 * * *" \
    --uri="https://TU_URL/clear-cache" \
    --http-method=GET \
    --location=us-central1
```

## 🧪 Testing

```bash
# Obtener tu URL
export SERVICE_URL=$(gcloud run services describe warren-screener \
    --region=us-central1 --format="get(status.url)")

# Primera petición (creará caché, ~4 min)
time curl $SERVICE_URL/analyze

# Segunda petición (usará caché, ~0.2 seg)
time curl $SERVICE_URL/analyze

# Ver estado del caché
curl $SERVICE_URL/cache-status

# Limpiar caché
curl $SERVICE_URL/clear-cache
```

## 📝 Ver Logs

```bash
# Ver logs en tiempo real
gcloud run services logs tail warren-screener --region=us-central1

# Buscar mensajes importantes:
# "✓ Usando datos del caché"
# "✓ Resultados guardados en caché"
# "⚠ Caché expirado"
```

## 💰 Costo Estimado

- **Cloud Storage:** ~$0.10/mes (negligible)
- **Cloud Run:** ~$5/mes
- **Total:** ~$5/mes

El caché reduce drásticamente el costo de Cloud Run porque:
- Solo se ejecuta análisis completo 1 vez al día (en lugar de cada petición)
- Las demás peticiones son prácticamente gratis (solo sirven JSON)

## 🐛 Troubleshooting

### El caché no se guarda

```bash
# Verificar permisos del bucket
gsutil iam get gs://warren-screener-cache

# Ver logs
gcloud run services logs tail warren-screener --region=us-central1
```

### Error de permisos

```bash
# Volver a configurar permisos
SERVICE_ACCOUNT=$(gcloud iam service-accounts list \
    --filter="displayName:Compute Engine default service account" \
    --format="value(email)")

gsutil iam ch serviceAccount:$SERVICE_ACCOUNT:objectAdmin \
    gs://warren-screener-cache
```

### Forzar nueva versión

```bash
# Si haces cambios en el código
gcloud builds submit --tag gcr.io/TU_PROJECT_ID/warren-screener

gcloud run deploy warren-screener \
    --image gcr.io/TU_PROJECT_ID/warren-screener \
    --region us-central1
```

## 📁 Estructura de Archivos

```
.
├── main.py              # Código principal con caché
├── requirements.txt     # Dependencias Python
├── Dockerfile          # Configuración Docker
├── deploy.sh           # Script de despliegue automático
└── README.md           # Este archivo
```

## ✨ Características Implementadas

- ✅ Caché automático de 24 horas
- ✅ Manejo de errores robusto
- ✅ Logs detallados con emojis
- ✅ Endpoints de gestión de caché
- ✅ Health check
- ✅ Fallback si Cloud Storage falla
- ✅ Timestamps y métricas de ejecución

## 🎓 Cómo Funciona

1. **Primera petición a `/analyze`:**
   - No hay caché → Ejecuta análisis completo (4 min)
   - Guarda resultados en Cloud Storage
   - Devuelve resultados al usuario

2. **Siguientes peticiones (dentro de 24h):**
   - Hay caché válido → Lee de Cloud Storage (0.2 seg)
   - Devuelve resultados inmediatamente

3. **Después de 24 horas:**
   - Caché expirado → Ejecuta nuevo análisis
   - Actualiza caché
   - Ciclo se repite

## 📞 Soporte

Si tienes problemas:

1. Verifica los logs: `gcloud run services logs tail warren-screener`
2. Verifica el bucket: `gsutil ls gs://warren-screener-cache`
3. Verifica permisos: `gsutil iam get gs://warren-screener-cache`

## 🚀 Próximos Pasos Recomendados

1. **Monitoreo:** Configura alertas en Cloud Monitoring
2. **Scheduler:** Automatiza limpieza diaria del caché
3. **Custom Domain:** Asigna un dominio personalizado
4. **CDN:** Agrega Cloud CDN para aún más velocidad
5. **Rate Limiting:** Implementa límites de peticiones

---

**¿Listo para desplegar?** → `./deploy.sh`
