#!/usr/bin/env python3
"""
Análisis de las diferencias entre Oracle V7 y Warren Screener
"""

print("🔍 ANÁLISIS DE DIFERENCIAS: Oracle V7 vs Warren Screener")
print("="*80)

print("\n📋 ORACLE V7 (Colab) - PIPELINE:")
print("-" * 80)
print("""
1. get_bulletproof_universe()
   └─> Obtiene 500 tickers (S&P 500 + Nasdaq 100)

2. analyze_stock_v7(ticker) para cada ticker:
   ├─> Filtro market cap >= $5B
   ├─> Extracción fuzzy de métricas
   ├─> Calcula ROIC = (EBIT * 0.79) / invested_capital
   ├─> Calcula Piotroski (0-9) - versión simplificada
   ├─> FILTRO: ROIC >= 8%
   ├─> FILTRO: Piotroski >= 5
   ├─> Calcula FCF = OCF - CapEx
   ├─> Growth = min(ROIC * 0.5, 0.14), max 3%
   ├─> DCF 2-stage (5 años + terminal 3%)
   ├─> MOS = (intrinsic - price) / intrinsic
   └─> FILTRO: MOS >= -20% O Piotroski >= 7

3. Ordena por MOS descendente
4. ✅ NO HAY FILTROS TÉCNICOS
""")

print("\n📋 WARREN SCREENER (main.py /analyze) - PIPELINE:")
print("-" * 80)
print("""
1. fetch_universe(250)
   └─> Obtiene 250 tickers

2. download_history_batch()
   └─> Descarga históricos de precios (SMA50, SMA200, RSI, MACD, OBV, ATR)

3. ⚠️ PREFILTRO TÉCNICO ESTRICTO:
   ├─> Precio entre $2-$2000
   ├─> Histórico >= 205 días (MA200 + 5)
   ├─> trend_up: Close > SMA200 AND SMA50 > SMA200
   ├─> rsi_ok: 30 <= RSI <= 60
   ├─> macd_up: MACD > Signal AND Histogram > 0
   └─> ❌ ELIMINA muchas acciones ANTES del análisis fundamental

4. Si quedan < 120, aplica filtro suave

5. get_fundamentals_and_quality(ticker) para candidatos técnicos:
   ├─> Calcula PE, PB, ROE, ROIC, márgenes, CAGR
   ├─> Calcula Piotroski (0-9) - versión completa
   ├─> DCF 10-year + ROIC-based DCF
   └─> Buffett Score

6. Ordena por Buffett Score, score6, ROIC
""")

print("\n❌ PROBLEMA IDENTIFICADO:")
print("-" * 80)
print("""
El Warren Screener (/analyze) ELIMINA stocks con excelentes fundamentales
porque no pasan los filtros técnicos ANTES de evaluarlos fundamentalmente.

Ejemplo: Gartner (IT)
- ROIC: 51.6% (EXCELENTE)
- Piotroski: 5 (CALIDAD)
- MOS: 49.5% (SUBVALORADO)
- ❌ ELIMINADO porque su RSI o MACD no cumplían criterios técnicos

Otros eliminados por filtros técnicos:
CLX, MKTX, ZBRA, LII, RL, HAS, DPZ, BR, HUBB
""")

print("\n✅ SOLUCIÓN IMPLEMENTADA:")
print("-" * 80)
print("""
Nuevo endpoint /oracle que replica Oracle V7 EXACTAMENTE:

1. NO descarga históricos de precios
2. NO aplica filtros técnicos
3. Análisis 100% fundamental:
   - ROIC >= 8%
   - Piotroski >= 5
   - MOS >= -20%
   - Market cap >= $5B

4. Usa intrinsic_roic (ROIC-based DCF) en lugar de intrinsic (CAGR-based)
5. Growth estimation basado en ROIC: min(roic * 0.5, 0.14)

RESULTADO: Los mismos stocks que Oracle V7 Colab
""")

print("\n📊 COMPARACIÓN DE ENDPOINTS:")
print("-" * 80)
print(f"{'Característica':<35} {'Oracle V7 (/oracle)':<25} {'Warren (/analyze)':<25}")
print("-" * 80)
print(f"{'Filtros técnicos':<35} {'❌ NO':<25} {'✅ SÍ (RSI, MACD, MA)':<25}")
print(f"{'Históricos de precio':<35} {'❌ NO necesita':<25} {'✅ Descarga':<25}")
print(f"{'Velocidad':<35} {'⚡ Más rápido':<25} {'🐢 Más lento':<25}")
print(f"{'Resultados':<35} {'📈 Más candidatos':<25} {'📉 Menos candidatos':<25}")
print(f"{'Enfoque':<35} {'💎 Value puro':<25} {'🎯 Value + Momentum':<25}")
print(f"{'DCF Method':<35} {'ROIC-based':<25} {'CAGR + ROIC dual':<25}")
print(f"{'Ideal para':<35} {'Long-term investing':<25} {'Swing trading':<25}")

print("\n🎯 RESULTADO ESPERADO:")
print("-" * 80)
print("""
GET /oracle debe retornar resultados IDÉNTICOS a Oracle V7 Colab:

Top stocks esperados:
1. SYF  - MOS 84.3%
2. AMP  - MOS 73.1% (ROIC altísimo 166.7%)
3. VZ   - MOS 66.2%
4. KMB  - MOS 59.6%
5. FCX  - MOS 52.4%

+ Stocks que faltaban en /analyze:
IT, CLX, MKTX, ZBRA, LII, RL, HAS, DPZ, BR, HUBB
""")

print("\n💡 RECOMENDACIÓN DE USO:")
print("-" * 80)
print("""
- USA /oracle para: Value investing, buy & hold, análisis fundamental puro
- USA /analyze para: Trading activo, momentum + value, swing trading

Ambos endpoints son válidos, solo tienen diferentes filosofías de inversión.
""")

print("="*80)
