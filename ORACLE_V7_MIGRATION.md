# 🏛️ Oracle V7 Migration - Warren Screener

## ✅ Cambio Completado

El endpoint `/analyze` ahora usa **lógica Oracle V7 pura** (100% fundamental, sin filtros técnicos).

**IMPORTANTE**: Se reemplazó completamente el pipeline de análisis para usar Oracle V7. Ya NO hay filtros técnicos (RSI, MACD, Moving Averages) ni descarga de históricos de precios.

## 🔄 ¿Qué Cambió?

### **ANTES** (Warren Screener con filtros técnicos)
```
1. Universo (250 tickers)
2. Descarga históricos de precios ⏳
3. ❌ FILTRO TÉCNICO ESTRICTO:
   - RSI entre 30-60
   - MACD alcista
   - Precio > SMA200
   - SMA50 > SMA200
   └─> Elimina ~80% ANTES del análisis fundamental
4. Análisis fundamental de sobrevivientes (~50 tickers)
5. Resultado: ~22 candidatos

❌ PROBLEMA: Eliminaba IT, CLX, MKTX, ZBRA, etc. por filtros técnicos
```

### **AHORA** (Oracle V7 Pure Fundamental)
```
1. Universo (250 tickers)
2. ✅ ANÁLISIS FUNDAMENTAL DIRECTO:
   - ROIC >= 8%
   - Piotroski >= 5
   - MOS >= -20%
   - Market cap >= $5B
3. Resultado: ~30 candidatos
4. ✅ NO descarga históricos
5. ✅ NO aplica filtros técnicos

✅ SOLUCIÓN: Encuentra TODOS los stocks con buenos fundamentales
```

## 📊 Resultados Esperados

### Top Stocks (por MOS):
```
1. SYF  - 84.3% MOS, 15.9% ROIC, Piotroski 5
2. AMP  - 73.1% MOS, 166.7% ROIC (!), Piotroski 5
3. VZ   - 66.2% MOS, 8.8% ROIC, Piotroski 5
4. KMB  - 59.6% MOS, 34.2% ROIC, Piotroski 5
5. FCX  - 52.4% MOS, 21.6% ROIC, Piotroski 5
```

### Stocks Recuperados (antes eliminados):
- **IT** (Gartner) - ROIC 51.6%, MOS 49.5%
- **CLX** (Clorox) - ROIC 30.4%, MOS 47.3%
- **MKTX** (MarketAxess) - ROIC 20.2%, MOS 26.8%
- **ZBRA** (Zebra Tech) - ROIC 12.0%, MOS 26.0%
- **LII** (Lennox) - ROIC 42.6%, MOS 9.5%
- Y más...

## 🧪 Cómo Probar

### 1. Deploy y Test Básico
```bash
# Verificar el servicio
curl https://your-service.com/

# Debería mostrar:
{
  "status": "Warren Screener API v4.0 (Oracle V7 Enhanced)",
  "oracle_v7_features": [
    "✅ NO filtros técnicos (RSI, MACD, MA) - Pure fundamental",
    "✅ Filtros: ROIC >= 8%, Piotroski >= 5, MOS >= -20%",
    ...
  ]
}
```

### 2. Ejecutar Análisis
```bash
curl -X GET https://your-service.com/analyze
```

### 3. Verificar Top 5
```bash
curl https://your-service.com/analyze | jq '.results[:5] | .[] | {Ticker, MOS, ROIC, Piotroski}'
```

**Esperado:**
```json
[
  {"Ticker": "SYF", "MOS": 0.843, "ROIC": 0.159, "Piotroski": 5},
  {"Ticker": "AMP", "MOS": 0.730, "ROIC": 1.667, "Piotroski": 5},
  {"Ticker": "VZ", "MOS": 0.662, "ROIC": 0.088, "Piotroski": 5},
  ...
]
```

### 4. Verificar Stock Específico (IT - Gartner)
```bash
curl https://your-service.com/analyze | jq '.results[] | select(.Ticker=="IT")'
```

**Esperado:**
```json
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

### 5. Contar Candidatos
```bash
curl https://your-service.com/analyze | jq '.candidates_count'
# Esperado: ~30 (vs ~22 antes)
```

## 📋 Estructura de Salida

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
    "watchlist_balanced": [...],
    "portfolio_metrics": {...}
  },
  "generated_at": "2026-01-08T...",
  "cache_enabled": true,
  "from_cache": false,
  "execution_time_seconds": 86.3
}
```

## ✅ Criterios de Éxito

El endpoint pasa si:

1. ✅ **Retorna ~30 candidatos** (no ~22)
2. ✅ **Top 5 contiene: SYF, AMP, VZ, KMB, FCX**
3. ✅ **Encuentra IT (Gartner)** con MOS ~50%
4. ✅ **Todos ROIC >= 8%**
5. ✅ **Todos Piotroski >= 5**
6. ✅ **Execution time 60-120s** (más rápido sin descargas de históricos)
7. ✅ **NO menciona filtros técnicos** en logs

## 🐛 Troubleshooting

### Si faltan candidatos esperados:
```bash
# Verificar que el universo es suficientemente grande
curl https://your-service.com/analyze | jq '.total_analyzed'
# Debería ser 250
```

### Si MOS difieren:
- **Tolerancia aceptable**: ±5% (debido a precios en tiempo real)
- Si difiere más: verificar datos financieros en Yahoo Finance

### Si IT no aparece:
1. Verificar que pasa ROIC >= 8% (IT tiene ~51.6%)
2. Verificar Piotroski >= 5
3. Revisar logs para ver si hay error en Yahoo Finance

### Si es muy lento:
- Normal: 60-120s para 250 tickers
- Slow: >180s (posible throttling)
- Solución: Reducir UNIVERSE_LIMIT a 200

## 🔐 Qué NO Cambió

- ✅ Caché de 24 horas sigue funcionando
- ✅ Cloud Storage integration intacta
- ✅ Piotroski Score (0-9) calculation
- ✅ ROIC-based DCF valuation
- ✅ Post-processing (sector analysis, alerts, watchlists)
- ✅ Estructura de salida compatible

## 📝 Notas Importantes

1. **Sin Rollback Fácil**: Los filtros técnicos fueron eliminados completamente
2. **Cache Invalidation**: Limpiar caché después del deploy para evitar datos viejos
3. **Documentación**: Actualizar docs para reflejar que es "Pure Fundamental"
4. **Usuarios**: Notificar que ya no filtra por momentum/técnicos

## 🚀 Despliegue

```bash
# 1. Deploy a Cloud Run
gcloud run deploy warren-screener \
  --source . \
  --region us-central1

# 2. Limpiar caché
curl -X GET https://your-service.com/clear-cache

# 3. Test
curl https://your-service.com/analyze | jq '.results[:5]'

# 4. Verificar
# - Más candidatos (~30 vs ~22)
# - IT y otros stocks recuperados
# - Top 5 correcto
```

## ✨ Beneficios

1. **Más oportunidades**: ~30 candidatos vs ~22
2. **Mejor calidad**: No elimina stocks con excelentes fundamentales
3. **Más rápido**: No descarga históricos de precios
4. **Más simple**: Código más limpio sin lógica técnica
5. **Más predecible**: Resultados idénticos al script Colab

## 🎯 Filosofía

**Antes**: Value + Momentum (híbrido)
**Ahora**: Pure Value Investing (como Buffett/Graham)

> "Price is what you pay, value is what you get" - Warren Buffett

Este screener ahora busca **valor fundamental** sin importar el momentum a corto plazo.

---

**Versión**: v4.0 Oracle V7 Enhanced
**Fecha**: 2026-01-08
**Commit**: 776de73
