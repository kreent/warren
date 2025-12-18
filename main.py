# =========================================
# Screener Value + Momentum + Buffett (Cloud Run) - OPTIMIZADO v2.1
# =========================================

import pandas as pd
import numpy as np
import yfinance as yf
import requests
import io
import sys
import time
import logging
from functools import lru_cache
from datetime import datetime, timedelta
from flask import Flask, jsonify, request
from concurrent.futures import ThreadPoolExecutor, as_completed

# Configurar logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Silencio de logs ruidosos
logging.getLogger("yfinance").setLevel(logging.CRITICAL)
logging.getLogger("urllib3").setLevel(logging.WARNING)

# -------- Parámetros generales --------
UNIVERSE_LIMIT   = 200     # Reducido para ser más rápido
BATCH_SIZE       = 60
MAX_FUND_REQS    = 120     # Reducido para evitar rate limits

PRICE_MIN, PRICE_MAX = 2.0, 2000.0

# Filtros más flexibles para obtener resultados
SCORE_MIN        = 4       # Reducido de 5 a 4
PE_MAX           = 25.0    # Aumentado de 15 a 25
PB_MAX           = 3.0     # Aumentado de 1.5 a 3
ROE_MIN          = 0.08    # Reducido de 0.12 a 0.08
DEBT_EBITDA_MAX  = 5.0     # Aumentado de 3.0 a 5.0

# Técnicos base
RSI_MIN, RSI_MAX = 25, 70  # Rango más amplio
VOL_LOOKBACK     = 20
MA_SHORT, MA_LONG= 50, 200

# Buffett / DCF
DISCOUNT_RATE    = 0.10
TERMINAL_G       = 0.02
MAX_GROWTH_CAP   = 0.12
MOS_THRESHOLD    = 0.30
FCF_SALES_PROXY  = 0.05

# -------- CONFIGURACIÓN DE CACHÉ --------
CACHE_TTL_SECONDS = 3600  # 1 hora
cached_analysis_result = None
cached_analysis_timestamp = None

def log(msg): 
    logger.info(msg)
    sys.stdout.flush()

# -------- Lectura robusta de listas (S&P500 ∪ Nasdaq-100) con CACHÉ --------
def try_read_html(url):
    r = requests.get(url, timeout=25, headers={"User-Agent":"Mozilla/5.0"})
    r.raise_for_status()
    return pd.read_html(io.StringIO(r.text))

def try_read_csv(url):
    try:
        r = requests.get(url, timeout=25, headers={"User-Agent":"Mozilla/5.0"})
        r.raise_for_status()
        return pd.read_csv(io.StringIO(r.text))
    except Exception:
        return None

def clean_symbols(series):
    s = series.astype(str).str.strip().str.upper().str.replace(".", "-", regex=False)
    s = s[~s.str.contains(r"[^\w\-]", regex=True)]
    return s

@lru_cache(maxsize=1)
def universe_sp500_cached():
    """Caché del listado S&P500"""
    try:
        for tb in try_read_html("https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"):
            cols = [str(c) for c in tb.columns]; lowers = [c.lower() for c in cols]
            if any(c in ("symbol","ticker") for c in lowers):
                idx = next(i for i,c in enumerate(lowers) if c in ("symbol","ticker"))
                log(f"✓ Obtenidos {len(tb)} tickers del S&P500")
                return clean_symbols(tb.iloc[:, idx])
    except Exception as e:
        log(f"⚠ Error obteniendo S&P500 desde Wikipedia: {e}")
    
    # Fallback a GitHub
    for gh in [
        "https://raw.githubusercontent.com/datasets/s-and-p-500-companies/master/data/constituents.csv",
        "https://raw.githubusercontent.com/plotly/dash-stock-tickers/master/symbols.csv"
    ]:
        g = try_read_csv(gh)
        if g is not None:
            for c in ["Symbol","symbol","Ticker","ticker","code"]:
                if c in g.columns:
                    log(f"✓ Obtenidos {len(g)} tickers del S&P500 (GitHub)")
                    return clean_symbols(g[c])
    
    log("⚠ No se pudo obtener S&P500 desde fuentes externas")
    return pd.Series([], dtype=str)

@lru_cache(maxsize=1)
def universe_nasdaq100_cached():
    """Caché del listado Nasdaq-100"""
    try:
        for tb in try_read_html("https://en.wikipedia.org/wiki/Nasdaq-100"):
            cols = [str(c) for c in tb.columns]; lowers = [c.lower() for c in cols]
            cand_idx = [i for i,c in enumerate(lowers) if ("ticker" in c) or ("symbol" in c)]
            if cand_idx:
                log(f"✓ Obtenidos {len(tb)} tickers del Nasdaq-100")
                return clean_symbols(tb.iloc[:, cand_idx[0]])
    except Exception as e:
        log(f"⚠ Error obteniendo Nasdaq-100 desde Wikipedia: {e}")
    
    # Fallback a GitHub
    for gh in [
        "https://raw.githubusercontent.com/nikbearbrown/Financial-Machine-Learning/master/data/nasdaq100list.csv",
        "https://raw.githubusercontent.com/sstrickx/yahoofinance-api/master/src/test/resources/nasdaq100.csv"
    ]:
        g = try_read_csv(gh)
        if g is not None:
            for c in ["Symbol","symbol","Ticker","ticker"]:
                if c in g.columns:
                    log(f"✓ Obtenidos {len(g)} tickers del Nasdaq-100 (GitHub)")
                    return clean_symbols(g[c])
    
    log("⚠ No se pudo obtener Nasdaq-100 desde fuentes externas")
    return pd.Series([], dtype=str)

def fetch_universe(limit=UNIVERSE_LIMIT):
    log("Construyendo universo (S&P500 ∪ Nasdaq-100)…")
    sp = universe_sp500_cached()
    ndx = universe_nasdaq100_cached()
    base = pd.Series(pd.concat([sp, ndx]).dropna().unique())
    
    if base.empty or len(base) < 50:
        log("⚠ Usando universo de respaldo (hardcoded)")
        base = pd.Series([
            "AAPL","MSFT","AMZN","GOOGL","META","NVDA","TSLA","JPM","BAC","WFC","C","GS","MS",
            "V","MA","PYPL","NFLX","PEP","KO","MCD","WMT","T","VZ","XOM","CVX","COP",
            "PFE","JNJ","MRK","ABT","TMO","BMY","UNH","HUM","CVS","CI","LLY",
            "ORCL","IBM","INTC","AMD","QCOM","AVGO","TXN","ADI","MU","CRM","NOW",
            "DIS","CMCSA","TGT","HD","LOW","NKE","COST","BKNG","ABNB","MAR","SBUX",
            "CAT","DE","BA","GE","HON","UPS","FDX","NOC","LMT","RTX","GD",
            "ADBE","PANW","FTNT","SNOW","ZS","OKTA","DDOG","NET","SQ","COIN","ROKU",
            "SHOP","UBER","LYFT","DASH","SPOT","ZM","DOCU","TWLO","PLTR","U",
            "F","GM","RIVN","LCID","NIO","XPEV","LI","BYDDY","TM","HMC"
        ])
    
    base = base.drop_duplicates().sample(min(limit, len(base)), random_state=42).reset_index(drop=True)
    log(f"✓ Universo final: {len(base)} tickers")
    return base.to_list()

# -------- Técnicos (SMA/RSI/MACD/OBV/ATR) --------
def ema(s, span): return s.ewm(span=span, adjust=False, min_periods=span).mean()

def rsi(series, length=14):
    delta = series.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(alpha=1/length, adjust=False, min_periods=length).mean()
    avg_loss = loss.ewm(alpha=1/length, adjust=False, min_periods=length).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    return 100 - (100/(1+rs))

def macd(series, fast=12, slow=26, signal=9):
    fast_ema = ema(series, fast); slow_ema = ema(series, slow)
    macd_line = fast_ema - slow_ema
    signal_line = ema(macd_line, signal)
    hist = macd_line - signal_line
    return macd_line, signal_line, hist

def obv(close, volume):
    direction = np.sign(close.diff()).fillna(0)
    return (direction * volume).cumsum()

def atr(high, low, close, length=14):
    prev_close = close.shift(1)
    tr = pd.concat([
        (high - low),
        (high - prev_close).abs(),
        (low  - prev_close).abs()
    ], axis=1).max(axis=1)
    return tr.rolling(length, min_periods=length).mean()

def compute_technicals(df):
    try:
        close = df["Close"].copy(); vol = df["Volume"].copy()
        high  = df["High"].copy();  low = df["Low"].copy()

        df["SMA50"]  = close.rolling(MA_SHORT, min_periods=MA_SHORT).mean()
        df["SMA200"] = close.rolling(MA_LONG,  min_periods=MA_LONG).mean()
        df["RSI14"]  = rsi(close, 14)
        m, s, h      = macd(close, 12, 26, 9)
        df["MACD"], df["MACDsig"], df["MACDh"] = m, s, h
        df["VolAvg20"] = vol.rolling(VOL_LOOKBACK, min_periods=VOL_LOOKBACK).mean()
        df["OBV"] = obv(close, vol)
        df["ATR14"] = atr(high, low, close, 14)

        valid = df.dropna()
        if valid.empty:
            return None, None
        last = valid.iloc[-1]
        conds = {
            "trend_up":  bool(last["Close"] > last["SMA200"] and last["SMA50"] > last["SMA200"]),
            "rsi_ok":    bool(RSI_MIN <= last["RSI14"] <= RSI_MAX),
            "macd_up":   bool(last["MACD"] > last["MACDsig"] and last["MACDh"] > 0),
            "volume_up": bool(last["Volume"] > last["VolAvg20"] and (last["Close"] > df["Close"].iloc[-2] if len(df)>1 else True)),
            "obv_up":    bool(valid["OBV"].iloc[-1] > valid["OBV"].iloc[-5] if len(valid) > 5 else False),
            "atr_pct":   float(last["ATR14"]/last["Close"]) if last["Close"] else np.nan,
        }
        return conds, last
    except Exception as e:
        log(f"⚠ Error en compute_technicals: {e}")
        return None, None

# -------- Descarga de históricos por lotes con paralelización --------
def _required_period_for(ma_long=MA_LONG, extra_days=40):
    days = ma_long + extra_days
    if days <= 250:   return "1y"
    elif days <= 500: return "2y"
    elif days <= 750: return "3y"
    else:             return "5y"

def _download_single_ticker(tk, period):
    """Descarga un ticker individual"""
    try:
        df = yf.download(tk, period=period, progress=False, threads=False)
        if df is None or df.empty or len(df) < 50:
            return tk, None
        df = df[["Open","High","Low","Close","Volume"]].copy()
        df = df.dropna()
        if len(df) < MA_LONG:
            return tk, None
        return tk, df
    except Exception as e:
        return tk, None

def download_history_batch(tickers, period=None, batch_size=BATCH_SIZE, sleep=0, retries=1, max_workers=8):
    """Descarga históricos usando ThreadPoolExecutor"""
    if period is None:
        period = _required_period_for(MA_LONG, 40)
    
    log(f"📊 Descargando históricos para {len(tickers)} tickers (period={period})...")
    
    results = {}
    failed = 0
    
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_to_ticker = {
            executor.submit(_download_single_ticker, tk, period): tk 
            for tk in tickers
        }
        
        for future in as_completed(future_to_ticker):
            tk, df = future.result()
            if df is not None and not df.empty:
                results[tk] = df
            else:
                failed += 1
    
    log(f"✓ Descargados: {len(results)} exitosos, {failed} fallidos")
    return results

# -------- Fundamentales y DCF con CACHÉ --------
@lru_cache(maxsize=500)
def get_fundamentals_and_quality_cached(ticker):
    """Versión cacheada de fundamentales"""
    try:
        t = yf.Ticker(ticker)
        info = t.info or {}
    except Exception as e:
        log(f"⚠ Error obteniendo info de {ticker}: {e}")
        info = {}

    pe = info.get("trailingPE") or info.get("forwardPE")
    pb = info.get("priceToBook")
    roe = info.get("returnOnEquity")
    debt = info.get("totalDebt")
    ebitda = info.get("ebitda")
    debt_ebitda = (float(debt)/float(ebitda)) if (debt not in (None,0) and ebitda not in (None,0)) else None

    gross_margin = info.get("grossMargins")
    op_margin    = info.get("operatingMargins")
    net_margin   = info.get("profitMargins")

    revenue = info.get("totalRevenue")
    price   = info.get("currentPrice") or info.get("regularMarketPrice")
    mktcap  = info.get("marketCap")
    shares  = info.get("sharesOutstanding")

    rev_cagr, ni_cagr = None, None
    try:
        fs = t.financials
        if fs is not None and not fs.empty and "Total Revenue" in fs.index:
            trev = fs.loc["Total Revenue"].dropna().sort_index()
            if len(trev) >= 2:
                yrs = (trev.index[-1] - trev.index[0]).days / 365.25
                if yrs > 0:
                    rev_cagr = ((trev.iloc[-1]/trev.iloc[0])**(1/yrs)) - 1
        if fs is not None and not fs.empty and "Net Income" in fs.index:
            tni = fs.loc["Net Income"].dropna().sort_index()
            if len(tni) >= 2:
                yrs2 = (tni.index[-1] - tni.index[0]).days / 365.25
                if yrs2 > 0 and tni.iloc[0] > 0:
                    ni_cagr = ((tni.iloc[-1]/tni.iloc[0])**(1/yrs2)) - 1
    except Exception:
        pass

    roic = None
    try:
        bs = t.balance_sheet
        nopat = None
        if fs is not None and not fs.empty and "Operating Income" in fs.index:
            oi = fs.loc["Operating Income"].dropna()
            if not oi.empty:
                nopat = oi.iloc[0] * 0.7
        ic = None
        if bs is not None and not bs.empty:
            equity_line = next((x for x in bs.index if "stockholder" in str(x).lower() or "shareholder" in str(x).lower()), None)
            debt_line   = next((x for x in bs.index if "total debt" in str(x).lower()), None)
            eq = bs.loc[equity_line].dropna().iloc[0] if equity_line else None
            dbt= bs.loc[debt_line].dropna().iloc[0] if debt_line else None
            if eq is not None and dbt is not None:
                ic = eq + dbt
        if nopat is not None and ic not in (None,0):
            roic = float(nopat) / float(ic)
    except Exception:
        pass

    net_debt = None
    try:
        cash_val = info.get("totalCash") or 0
        debt_val = info.get("totalDebt") or 0
        net_debt = float(debt_val) - float(cash_val)
    except Exception:
        pass

    ocf = info.get("operatingCashflow")
    capex = info.get("capitalExpenditures")
    fcf = None
    try:
        if ocf is not None and capex is not None: fcf = float(ocf) - float(capex)
    except Exception: pass

    if shares in (None, 0) and mktcap and price:
        try: shares = int(float(mktcap)/float(price))
        except Exception: pass

    intrinsic = None
    try:
        fcf_base = fcf if fcf not in (None, np.nan) else (FCF_SALES_PROXY * float(revenue) if revenue not in (None,0) else None)
        if fcf_base is not None and shares not in (None, 0):
            g_base = np.nanmean([x for x in [rev_cagr, ni_cagr] if x is not None]) if any([rev_cagr, ni_cagr]) else 0.05
            g_base = max(-0.05, min(float(g_base), MAX_GROWTH_CAP))
            fcfs, f = [], fcf_base
            for y in range(1, 11):
                growth = g_base * (0.9 ** (y-1))
                f = f * (1 + growth)
                fcfs.append(f / ((1+DISCOUNT_RATE)**y))
            terminal = fcfs[-1] * (1+TERMINAL_G) / (DISCOUNT_RATE - TERMINAL_G)
            ev = np.nansum(fcfs) + terminal/((1+DISCOUNT_RATE)**1)
            equity_val = ev - (net_debt if net_debt is not None else 0.0)
            intrinsic = float(equity_val) / float(shares)
    except Exception:
        pass

    return {
        "pe": (float(pe) if pe not in (None, "None", np.nan) else None),
        "pb": (float(pb) if pb not in (None, "None", np.nan) else None),
        "roe": (float(roe) if roe is not None else None),
        "debt_ebitda": (float(debt_ebitda) if debt_ebitda not in (None, "None", np.nan) else None),
        "fcf": fcf, "fcf_positive": (fcf is not None and fcf > 0),
        "gross_margin": (float(gross_margin) if gross_margin is not None else None),
        "op_margin":    (float(op_margin) if op_margin is not None else None),
        "net_margin":   (float(net_margin) if net_margin is not None else None),
        "rev_cagr":     (float(rev_cagr) if rev_cagr is not None else None),
        "ni_cagr":      (float(ni_cagr) if ni_cagr is not None else None),
        "roic":         (float(roic) if roic is not None else None),
        "shares_out":   (int(shares) if shares not in (None, np.nan) else None),
        "intrinsic":    (float(intrinsic) if intrinsic not in (None, np.nan) else None)
    }

def get_fundamentals_and_quality(ticker):
    return get_fundamentals_and_quality_cached(ticker)

# -------- Scoring --------
def evaluate_conditions(fund, tech):
    fundamentals = {
        "val_pe_pb": (fund.get("pe") is not None and fund["pe"] <= PE_MAX) and \
                     (fund.get("pb") is not None and fund["pb"] <= PB_MAX),
        "roe_ok":    (fund.get("roe") is not None and fund["roe"] >= ROE_MIN),
        "debt_ok":   (fund.get("debt_ebitda") is not None and fund["debt_ebitda"] <= DEBT_EBITDA_MAX),
        "fcf_pos":   fund.get("fcf_positive", False)
    }
    six = {
        "val_pe_pb": fundamentals["val_pe_pb"],
        "roe_ok": fundamentals["roe_ok"],
        "debt_ok": fundamentals["debt_ok"],
        "fcf_pos": fundamentals["fcf_pos"],
        "trend_up": tech["trend_up"],
        "rsi_macd": (tech["rsi_ok"] and tech["macd_up"])
    }
    score = int(sum(six.values()))
    bonus = int(tech["volume_up"])
    return six, fundamentals, tech, score, bonus

def buffett_score(fund, price, tech):
    pts = 0.0
    if fund.get("roic") is not None and fund["roic"] >= 0.12: pts += 1.5
    if fund.get("op_margin") is not None and fund["op_margin"] >= 0.15: pts += 1.0
    if fund.get("gross_margin") is not None and fund["gross_margin"] >= 0.40: pts += 0.5
    growth = np.nanmean([x for x in [fund.get("rev_cagr"), fund.get("ni_cagr")] if x is not None])
    if pd.notna(growth) and growth >= 0.05: pts += 1.0
    if fund.get("debt_ebitda") is not None and fund["debt_ebitda"] <= 2.5: pts += 1.0
    if fund.get("fcf") is not None and fund["fcf"] > 0: pts += 1.0
    intrinsic = fund.get("intrinsic")
    if intrinsic not in (None, np.nan) and price is not None:
        mos_ok = price <= intrinsic * (1 - MOS_THRESHOLD)
        if mos_ok: pts += 2.0
        elif price <= intrinsic * 0.9: pts += 1.0
    if fund.get("pe") is not None and fund["pe"] <= 20: pts += 0.5
    if fund.get("pb") is not None and fund["pb"] <= 3: pts += 0.5
    if tech["trend_up"] and (tech["rsi_ok"] or tech["macd_up"]) and tech.get("obv_up", False): pts += 1.0
    return round(min(10.0, pts), 2)

# -------------- ANÁLISIS CON MEJOR LOGGING --------------
def run_analysis():
    """Ejecuta el análisis completo con filtros flexibles"""
    log("=" * 60)
    log("INICIANDO ANÁLISIS")
    log("=" * 60)
    
    tickers = fetch_universe(limit=UNIVERSE_LIMIT)
    log(f"📋 Tickers a evaluar: {len(tickers)}")

    # 1) Históricos
    hist_map = download_history_batch(tickers, period=None, max_workers=8)
    
    if not hist_map:
        log("❌ ERROR: No se pudieron descargar históricos")
        return {
            "error": "No se pudieron descargar datos de mercado",
            "details": "Verifica tu conexión o los límites de Yahoo Finance"
        }

    log(f"✓ Históricos descargados: {len(hist_map)} tickers")

    # 2) Prefiltro técnico MUY flexible
    tech_ok = []
    for tk, hist in hist_map.items():
        try:
            price = float(hist["Close"].iloc[-1])
            if not (PRICE_MIN <= price <= PRICE_MAX): 
                continue
            if hist.shape[0] < MA_LONG + 5: 
                continue
            
            tech, _ = compute_technicals(hist)
            if tech is None:
                continue
            
            # Filtro SUPER flexible: solo necesita estar por encima de SMA200 O tener RSI ok O MACD ok
            if tech["trend_up"] or tech["rsi_ok"] or tech["macd_up"]:
                tech_ok.append((tk, price, tech))
        except Exception as e:
            log(f"⚠ Error procesando {tk}: {e}")
            continue

    log(f"✓ Pasaron filtro técnico: {len(tech_ok)} tickers")

    # Si aún son muy pocos, relajar más
    if len(tech_ok) < 30:
        log("⚠ Muy pocos resultados, aplicando filtro ultra-flexible...")
        tech_ok = []
        for tk, hist in hist_map.items():
            try:
                price = float(hist["Close"].iloc[-1])
                if not (PRICE_MIN <= price <= PRICE_MAX) or hist.shape[0] < 100:
                    continue
                tech, _ = compute_technicals(hist)
                if tech:
                    tech_ok.append((tk, price, tech))
            except:
                continue

    tech_ok = tech_ok[:MAX_FUND_REQS]
    log(f"📊 Analizando fundamentales de {len(tech_ok)} tickers...")

    # 3) Fundamentales
    rows = []
    for i, (tk, price, tech) in enumerate(tech_ok):
        if i % 20 == 0:
            log(f"  Procesando... {i}/{len(tech_ok)}")
        
        try:
            fund = get_fundamentals_and_quality(tk)
            six, fund_d, tech_d, score, bonus = evaluate_conditions(fund, tech)
            bscore = buffett_score(fund, price, tech)
            intrinsic = fund.get("intrinsic")
            mos = None
            if intrinsic not in (None, np.nan):
                try: 
                    mos = (intrinsic - price)/intrinsic
                except: 
                    mos = None
            
            rows.append({
                "ticker": tk, "price": round(price,2),
                "score6": score, "buffett_score": bscore,
                "pe": fund.get("pe"), "pb": fund.get("pb"), "roe": fund.get("roe"),
                "roic": fund.get("roic"), "gross_margin": fund.get("gross_margin"),
                "op_margin": fund.get("op_margin"), "net_margin": fund.get("net_margin"),
                "rev_cagr": fund.get("rev_cagr"), "ni_cagr": fund.get("ni_cagr"),
                "debt_ebitda": fund.get("debt_ebitda"), "fcf_positive": fund.get("fcf_positive"),
                "intrinsic": intrinsic, "mos": mos,
                "trend_up": tech["trend_up"], "rsi_ok": tech["rsi_ok"], "macd_up": tech["macd_up"],
                "obv_up": tech.get("obv_up", False), "atr_pct": tech.get("atr_pct", np.nan),
                "volume_up": tech["volume_up"]
            })
        except Exception as e:
            log(f"⚠ Error en fundamentales de {tk}: {e}")
            continue

    df = pd.DataFrame(rows)
    log(f"✓ DataFrame creado con {len(df)} registros")

    if df.empty:
        log("❌ DataFrame vacío después de análisis")
        return {
            "error": "Sin resultados después del análisis",
            "details": "Posibles causas: rate-limit de Yahoo Finance, problemas de red, o filtros muy estrictos",
            "suggestions": [
                "Espera 15 minutos y vuelve a intentar",
                "Verifica que el servicio tiene acceso a internet",
                "Revisa los logs para más detalles"
            ]
        }
    
    # Orden
    df = df.sort_values(["buffett_score","score6","roic"], ascending=[False,False,False]).reset_index(drop=True)

    # Candidatas con filtros MUY flexibles
    candidates = df[(df["buffett_score"] >= 5) | (df["score6"] >= 3)].copy()
    if candidates.empty:
        log("⚠ No hay candidatos con score alto, mostrando todos los resultados")
        candidates = df.head(20)  # Al menos mostrar top 20

    log(f"✓ Análisis completado: {len(candidates)} candidatos encontrados")
    log("=" * 60)

    result = {
        "total_analyzed": len(df),
        "candidates_count": len(candidates),
        "top_20": candidates.head(20).to_dict('records'),
        "all_results_count": len(df),
        "cached_at": datetime.now().isoformat(),
        "cache_ttl_seconds": CACHE_TTL_SECONDS,
        "filters_applied": {
            "price_range": f"${PRICE_MIN}-${PRICE_MAX}",
            "score_min": SCORE_MIN,
            "pe_max": PE_MAX,
            "pb_max": PB_MAX,
            "roe_min": ROE_MIN
        }
    }
    
    return result

def get_cached_analysis():
    """Retorna análisis cacheado o ejecuta uno nuevo"""
    global cached_analysis_result, cached_analysis_timestamp
    
    current_time = datetime.now()
    
    if cached_analysis_result is not None and cached_analysis_timestamp is not None:
        time_elapsed = (current_time - cached_analysis_timestamp).total_seconds()
        if time_elapsed < CACHE_TTL_SECONDS:
            log(f"✓ Retornando resultado cacheado (edad: {int(time_elapsed)}s)")
            result = cached_analysis_result.copy()
            result["from_cache"] = True
            result["cache_age_seconds"] = int(time_elapsed)
            return result
    
    log("Ejecutando nuevo análisis (caché expirado o no existe)...")
    cached_analysis_result = run_analysis()
    cached_analysis_timestamp = current_time
    
    result = cached_analysis_result.copy()
    result["from_cache"] = False
    result["cache_age_seconds"] = 0
    return result

# Flask app
app = Flask(__name__)

@app.route('/')
def home():
    return jsonify({
        "status": "Warren Screener API - OPTIMIZADO v2.1",
        "version": "2.1",
        "improvements": [
            "Filtros más flexibles para obtener más resultados",
            "Mejor logging y diagnóstico de errores",
            "Manejo robusto de rate-limits",
            "Universo de respaldo hardcoded"
        ],
        "endpoints": {
            "/analyze": "Análisis principal (con caché de 1 hora)",
            "/analyze?force_refresh=1": "Forzar nuevo análisis",
            "/health": "Health check",
            "/debug": "Información de debug"
        }
    })

@app.route('/health')
def health():
    return jsonify({
        "status": "healthy", 
        "timestamp": datetime.now().isoformat(),
        "cache_active": cached_analysis_result is not None
    })

@app.route('/debug')
def debug():
    """Endpoint de debug para diagnóstico"""
    cache_age = None
    if cached_analysis_timestamp:
        cache_age = int((datetime.now() - cached_analysis_timestamp).total_seconds())
    
    return jsonify({
        "cache_exists": cached_analysis_result is not None,
        "cache_age_seconds": cache_age,
        "cache_ttl": CACHE_TTL_SECONDS,
        "universe_limit": UNIVERSE_LIMIT,
        "filters": {
            "score_min": SCORE_MIN,
            "pe_max": PE_MAX,
            "pb_max": PB_MAX,
            "roe_min": ROE_MIN,
            "debt_ebitda_max": DEBT_EBITDA_MAX
        }
    })

@app.route('/analyze')
def analyze():
    try:
        force_refresh = request.args.get('force_refresh', '0') == '1'
        
        if force_refresh:
            global cached_analysis_result, cached_analysis_timestamp
            cached_analysis_result = None
            cached_analysis_timestamp = None
            log("🔄 Forzando refresh del caché...")
        
        results = get_cached_analysis()
        return jsonify(results)
    except Exception as e:
        log(f"❌ ERROR en /analyze: {str(e)}")
        import traceback
        return jsonify({
            "error": str(e),
            "traceback": traceback.format_exc()
        }), 500

if __name__ == "__main__":
    import os
    port = int(os.environ.get("PORT", 8080))
    app.run(host="0.0.0.0", port=port, threaded=True)
