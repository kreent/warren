#!/usr/bin/env python3
"""
Test script para verificar que el endpoint Oracle V7 genera resultados similares a Colab
"""

import sys
import json
from main import run_oracle_analysis, post_process_oracle_results

def main():
    print("🏛️ Testing Oracle V7 Analysis...")
    print("="*60)

    # Ejecutar análisis
    results = run_oracle_analysis()

    # Post-procesar
    results = post_process_oracle_results(results)

    # Mostrar resumen
    print("\n📊 RESUMEN DE RESULTADOS:")
    print(f"Total analizados: {results['total_analyzed']}")
    print(f"Candidatos encontrados: {results['candidates_count']}")
    print(f"Tiempo de ejecución: {results['execution_time_seconds']}s")

    if 'summary' in results:
        print(f"\n📈 Zonas:")
        print(f"  - Buy zone (MOS > 10%): {results['summary']['buy_zone_count']}")
        print(f"  - Fair zone (0-10%): {results['summary']['fair_zone_count']}")
        print(f"  - Watch zone (-20% a 0%): {results['summary']['watch_zone_count']}")

    # Mostrar top 10
    if results.get('results'):
        print("\n💎 TOP 10 POR MOS:")
        print("-" * 100)
        print(f"{'Ticker':<8} {'Price':>10} {'Sector':<25} {'ROIC':>8} {'Pio':>4} {'Growth':>8} {'Intrinsic':>12} {'MOS':>8}")
        print("-" * 100)

        for i, r in enumerate(results['results'][:10], 1):
            print(f"{r['Ticker']:<8} ${r['Price']:>9.2f} {r['Sector']:<25} "
                  f"{r['ROIC']*100:>7.1f}% {r['Piotroski']:>4} "
                  f"{r['Growth_Est']*100:>7.1f}% ${r['Intrinsic']:>11.2f} "
                  f"{r['MOS']*100:>7.1f}%")

    # Comparación con resultados esperados de Colab
    expected_top_5 = ['SYF', 'AMP', 'VZ', 'KMB', 'FCX']
    actual_top_5 = [r['Ticker'] for r in results['results'][:5]]

    print("\n🔍 VERIFICACIÓN vs COLAB:")
    print(f"Esperado (Colab): {expected_top_5}")
    print(f"Actual (Oracle):  {actual_top_5}")

    matches = sum(1 for ticker in expected_top_5 if ticker in actual_top_5)
    print(f"Coincidencias: {matches}/{len(expected_top_5)}")

    # Buscar tickers específicos que faltaban
    missing_tickers = ['IT', 'CLX', 'MKTX', 'ZBRA', 'LII', 'RL', 'HAS', 'DPZ', 'BR', 'HUBB']
    found_tickers = [r['Ticker'] for r in results['results']]

    print(f"\n🔎 TICKERS QUE FALTABAN EN /analyze:")
    for ticker in missing_tickers:
        if ticker in found_tickers:
            stock = next(r for r in results['results'] if r['Ticker'] == ticker)
            print(f"  ✅ {ticker}: Encontrado - MOS {stock['MOS']*100:.1f}%, ROIC {stock['ROIC']*100:.1f}%")
        else:
            print(f"  ❌ {ticker}: No encontrado")

    # Guardar resultados completos
    with open('/home/user/warren/oracle_test_results.json', 'w') as f:
        json.dump(results, f, indent=2, default=str)

    print("\n💾 Resultados completos guardados en: oracle_test_results.json")
    print("="*60)

    return 0

if __name__ == '__main__':
    sys.exit(main())
