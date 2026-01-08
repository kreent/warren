# 🧪 Guía de Pruebas: Endpoint Oracle V7

## 📝 Resumen

El endpoint `/oracle` replica la lógica exacta del script Oracle V7 de Colab, sin aplicar filtros técnicos que eliminaban stocks con buenos fundamentales.

## 🎯 Diferencia Clave

| Endpoint | Filtros | Filosofía | Resultados Esperados |
|----------|---------|-----------|---------------------|
| `/analyze` | Técnicos + Fundamentales | Value + Momentum | ~22 candidatos |
| `/oracle` | Solo Fundamentales | Pure Value | ~30 candidatos |

## 🔬 Pruebas a Realizar

### 1. Test Básico de Conectividad

```bash
# Verificar que el servicio está corriendo
curl https://your-service.com/

# Debería retornar:
{
  "status": "Warren Screener API v4.0 (Oracle V7 Enhanced)",
  "version": "4.0",
  "endpoints": {
    "/analyze": "...",
    "/oracle": "..."
  }
}
```

### 2. Test del Endpoint Oracle

```bash
# Ejecutar análisis Oracle V7
curl -X GET https://your-service.com/oracle

# O con mejor formato:
curl -X GET https://your-service.com/oracle | jq '.'
```

### 3. Verificación de Resultados

#### Estructura Esperada:

```json
{
  "total_analyzed": 250,
  "candidates_count": 30,
  "results": [
    {
      "Ticker": "SYF",
      "Price": 88.16,
      "Sector": "Financial Services",
      "ROIC": 0.1595,
      "Piotroski": 5,
      "Growth_Est": 0.0797,
      "Intrinsic": 560.05,
      "MOS": 0.8426
    }
  ],
  "summary": {
    "buy_zone_count": 16,
    "fair_zone_count": 5,
    "watch_zone_count": 9
  },
  "post_processed": {
    "sector_analysis": {...},
    "top_opportunities": {...},
    "alerts": [...],
    "watchlist_aggressive": [...],
    "portfolio_metrics": {...}
  }
}
```

### 4. Verificar Top 5 Stocks

Los primeros 5 stocks por MOS deberían ser (aproximadamente):

```bash
curl https://your-service.com/oracle | jq '.results[:5] | .[] | {Ticker, MOS}'
```

**Esperado:**
1. **SYF** - MOS ~84%
2. **AMP** - MOS ~73%
3. **VZ** - MOS ~66%
4. **KMB** - MOS ~60%
5. **FCX** - MOS ~52%

### 5. Verificar Stocks que Faltaban

Buscar si los stocks eliminados por filtros técnicos ahora aparecen:

```bash
# Verificar si IT (Gartner) está presente
curl https://your-service.com/oracle | jq '.results[] | select(.Ticker=="IT")'

# Esperado:
{
  "Ticker": "IT",
  "Price": 244.59,
  "Sector": "Technology",
  "ROIC": 0.516,
  "Piotroski": 5,
  "Growth_Est": 0.14,
  "Intrinsic": 484.25,
  "MOS": 0.495
}
```

Lista de tickers a verificar:
- IT, CLX, MKTX, ZBRA, LII, RL, HAS, DPZ, BR, HUBB

### 6. Comparar con /analyze

```bash
# Obtener resultados de ambos endpoints
curl https://your-service.com/analyze > warren_results.json
curl https://your-service.com/oracle > oracle_results.json

# Comparar número de candidatos
jq '.candidates_count' warren_results.json  # ~22
jq '.candidates_count' oracle_results.json  # ~30
```

### 7. Verificar Post-Processing

```bash
# Ver análisis por sector
curl https://your-service.com/oracle | jq '.post_processed.sector_analysis'

# Ver alertas
curl https://your-service.com/oracle | jq '.post_processed.alerts'

# Ver watchlist agresiva (MOS > 10%)
curl https://your-service.com/oracle | jq '.post_processed.watchlist_aggressive'
```

## ✅ Criterios de Éxito

El endpoint `/oracle` pasa la prueba si:

1. ✅ **Retorna ~30 candidatos** (vs ~22 de `/analyze`)
2. ✅ **Top 5 contiene SYF, AMP, VZ, KMB, FCX** en orden descendente por MOS
3. ✅ **Encuentra IT (Gartner)** con MOS ~49.5% y ROIC ~51.6%
4. ✅ **Encuentra otros stocks faltantes** (CLX, MKTX, ZBRA, etc.)
5. ✅ **Todos tienen ROIC >= 8%** (filtro Oracle V7)
6. ✅ **Todos tienen Piotroski >= 5** (filtro Oracle V7)
7. ✅ **MOS >= -20%** (watchlist threshold)
8. ✅ **Tiempo de ejecución < 120s** (más rápido que /analyze)

## 🐛 Troubleshooting

### Si faltan stocks esperados:

1. Verificar que `UNIVERSE_LIMIT` está en 250-500
2. Verificar que el universo incluye tanto S&P 500 como Nasdaq 100
3. Revisar logs para ver si hay errores de Yahoo Finance

### Si MOS difieren mucho:

Los MOS pueden variar ligeramente debido a:
- Precio actual diferente (mercado en tiempo real)
- Datos financieros actualizados en Yahoo Finance
- Diferencias menores en implementación de Piotroski

**Tolerancia aceptable**: ±5% en MOS

### Si el análisis es muy lento:

- Normal: 60-120 segundos para 250 tickers
- Lento: >180 segundos (posible throttling de Yahoo Finance)

## 📊 Ejemplo de Salida Exitosa

```bash
$ curl https://your-service.com/oracle | jq '.results[:3]'

[
  {
    "Ticker": "SYF",
    "Price": 88.16,
    "Sector": "Financial Services",
    "ROIC": 0.1595,
    "Piotroski": 5,
    "Growth_Est": 0.0797,
    "Intrinsic": 560.05,
    "MOS": 0.8426
  },
  {
    "Ticker": "AMP",
    "Price": 508.39,
    "Sector": "Financial Services",
    "ROIC": 1.6671,
    "Piotroski": 5,
    "Growth_Est": 0.14,
    "Intrinsic": 1885.35,
    "MOS": 0.7304
  },
  {
    "Ticker": "VZ",
    "Price": 40.46,
    "Sector": "Communication Services",
    "ROIC": 0.0876,
    "Piotroski": 5,
    "Growth_Est": 0.0438,
    "Intrinsic": 119.67,
    "MOS": 0.6619
  }
]

✅ PASSED: Top stocks correctos con MOS similares a Colab
```

## 🚀 Próximos Pasos

1. **Desplegar** los cambios a Cloud Run
2. **Ejecutar** las pruebas de esta guía
3. **Comparar** resultados con Oracle V7 Colab
4. **Validar** que los stocks faltantes ahora aparecen
5. **Documentar** cualquier diferencia encontrada

## 📞 Soporte

Si los resultados difieren significativamente del script Colab:
1. Verificar fechas de datos (pueden estar desactualizados)
2. Comparar precios actuales vs precios en Colab
3. Revisar logs del servidor para errores
4. Verificar que Yahoo Finance no está bloqueando requests
