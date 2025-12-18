# =========================================
# Screener Value + Momentum + Buffett (Cloud Run) - OPTIMIZADO
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
from flask import Flask, jsonify
from concurrent.futures import ThreadPoolExecutor, as_completed

# Silencio de logs ruidosos
logging.getLogger("yfinance").setLevel(logging.CRITICAL)
logging.getLogger("urllib3").setLevel(logging.WARNING)

# -------- Parámetros generales --------
UNIVERSE_LIMIT   = 250     # universo base (S&P500 ∪ Nasdaq-100)
BATCH_SIZE       = 60      # descarga por lotes para históricos (40–60 va bien)
MAX_FUND_REQS    = 160     # máximo de tickers a pedir fundamentals

PRICE_MIN, PRICE_MAX = 2.0, 2000.0

# Filtros "value+momentum" base (clásico)
SCORE_MIN        = 5
PE_MAX           = 15.0
PB_MAX           = 1.5
ROE_MIN          = 0.12
DEBT_EBITDA_MAX  = 3.0

# Técnicos base
RSI_MIN, RSI_MAX = 30, 60
VOL_LOOKBACK     = 20
MA_SHORT, MA_LONG= 50, 200

# Buffett / DCF
DISCOUNT_RATE    = 0.10    # 10% tasa de descuento
TERMINAL_G       = 0.02    # 2% crecimiento a perpetuidad
MAX_GROWTH_CAP   = 0.12    # 12% cap de crecimiento a 10 años
MOS_THRESHOLD    = 0.30    # margen de seguridad del 30%
FCF_SALES_PROXY  = 0.05    # proxy de FCF si falta: 5% de ventas

# -------- CONFIGURACIÓN DE CACHÉ --------
CACHE_TTL_SECONDS = 3600  # 1 hora - ajustar según necesidad
cached_analysis_result = None
cached_analysis_timestamp = None

def log(msg): 
    print(msg)
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

# CACHÉ para universe S&P500 (se actualiza raramente)
@lru_cache(maxsize=1)
def universe_sp500_cached():
    """Caché del listado S&P500 - se mantiene en memoria por vida de la instancia"""
    try:
        for tb in try_read_html("https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"):
            cols = [str(c) for c in tb.columns]; lowers = [c.lower() for c in cols]
            if any(c in ("symbol","ticker") for c in lowers):
                idx = next(i for i,c in enumerate(lowers) if c in ("symbol","ticker"))
                return clean_symbols(tb.iloc[:, idx])
    except Exception: pass
    for gh in [
        "https://raw.githubusercontent.com/datasets/s-and-p-500-companies/master/data/constituents.csv",
        "https://raw.githubusercontent.com/plotly/dash-stock-tickers/master/symbols.csv"
    ]:
        g = try_read_csv(gh)
        if g is not None:
            for c in ["Symbol","symbol","Ticker","ticker","code"]:
                if c in g.columns: return clean_symbols(g[c])
    try:
        if hasattr(yf, "tickers_sp500"):
            return pd.Series([t.replace(".","-").upper() for t in yf.tickers_sp500()])
    except Exception: pass
    return pd.Series([], dtype=str)

# CACHÉ para universe Nasdaq-100
@lru_cache(maxsize=1)
def universe_nasdaq100_cached():
    """Caché del listado Nasdaq-100 - se mantiene en memoria por vida de la instancia"""
    try:
        for tb in try_read_html("https://en.wikipedia.org/wiki/Nasdaq-100"):
            cols = [str(c) for c in tb.columns]; lowers = [c.lower() for c in cols]
            cand_idx = [i for i,c in enumerate(lowers) if ("ticker" in c) or ("symbol" in c)]
            if cand_idx: return clean_symbols(tb.iloc[:, cand_idx[0]])
    except Exception: pass
    for gh in [
        "https://raw.githubusercontent.com/nikbearbrown/Financial-Machine-Learning/master/data/nasdaq100list.csv",
        "https://raw.githubusercontent.com/sstrickx/yahoofinance-api/master/src/test/resources/nasdaq100.csv"
    ]:
        g = try_read_csv(gh)
        if g is not None:
            for c in ["Symbol","symbol","Ticker","ticker"]:
                if c in g.columns: return clean_symbols(g[c])
    return pd.Series([], dtype=str)

def fetch_universe(limit=UNIVERSE_LIMIT):
    log("Construyendo universo (S&P500 ∪ Nasdaq-100)…")
    sp = universe_sp500_cached()
    ndx = universe_nasdaq100_cached()
    base = pd.Series(pd.concat([sp, ndx]).dropna().unique())
    if base.empty:
        base = pd.Series([
            "AAPL","MSFT","AMZN","GOOGL","META","NVDA","TSLA","JPM","BAC","WFC","C","GS","MS",
            "V","MA","PYPL","NFLX","PEP","KO","MCD","WMT","T","VZ","XOM","CVX","COP",
            "PFE","JNJ","MRK","ABT","TMO","BMY","UNH","HUM","CVS","CI","LLY",
            "ORCL","IBM","INTC","AMD","QCOM","AVGO","TXN","ADI","MU","CRM","NOW",
            "DIS","CMCSA","TGT","HD","LOW","NKE","COST","BKNG","ABNB","MAR","SBUX",
            "CAT","DE","BA","GE","HON","UPS","FDX","NOC","LMT","RTX","GD",
            "ADBE","PANW","FTNT","SNOW","ZS","OKTA","DDOG","NET","SQ","COIN","ROKU"
        ])
    base = base.drop_duplicates().sample(min(limit, len(base)), random_state=42).reset_index(drop=True)
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

# -------- Descarga de históricos por lotes (adaptativo) con paralelización --------
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
        return tk, df
    except Exception:
        return tk, None

def download_history_batch(tickers, period=None, batch_size=BATCH_SIZE, sleep=0, retries=1, max_workers=10):
    """
    Descarga históricos usando ThreadPoolExecutor para paralelizar
    """
    if period is None:
        period = _required_period_for(MA_LONG, 40)
    
    log(f"Descargando históricos para {len(tickers)} tickers (period={period}) en paralelo...")
    
    results = {}
    
    # Usar ThreadPoolExecutor para descargas paralelas
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        # Enviar todas las tareas
        future_to_ticker = {
            executor.submit(_download_single_ticker, tk, period): tk 
            for tk in tickers
        }
        
        # Recoger resultados a medida que se completan
        for future in as_completed(future_to_ticker):
            tk, df = future.result()
            if df is not None and not df.empty:
                results[tk] = df
    
    log(f"✓ Descargados {len(results)}/{len(tickers)} tickers con datos válidos")
    return results

# -------- Fundamentales y DCF con CACHÉ por ticker --------
# Caché de 1 hora para fundamentales de cada ticker
@lru_cache(maxsize=500)
def get_fundamentals_and_quality_cached(ticker):
    """Versión cacheada de get_fundamentals_and_quality"""
    return _get_fundamentals_and_quality_impl(ticker)

def _get_fundamentals_and_quality_impl(ticker):
    try:
        t = yf.Ticker(ticker)
        info = t.info or {}
    except Exception:
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

    # DCF con proxy prudente si falta FCF
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
    except Exception: pass

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

# Wrapper para mantener compatibilidad
def get_fundamentals_and_quality(ticker):
    return get_fundamentals_and_quality_cached(ticker)

# -------- Scoring: clásico + Buffett Score --------
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
    """
    0–10 pts: Calidad (4), Fortaleza (2), Valoración (3), Técnica (1)
    """
    pts = 0.0
    # Calidad
    if fund.get("roic") is not None and fund["roic"] >= 0.12: pts += 1.5
    if fund.get("op_margin") is not None and fund["op_margin"] >= 0.15: pts += 1.0
    if fund.get("gross_margin") is not None and fund["gross_margin"] >= 0.40: pts += 0.5
    growth = np.nanmean([x for x in [fund.get("rev_cagr"), fund.get("ni_cagr")] if x is not None])
    if pd.notna(growth) and growth >= 0.05: pts += 1.0
    # Fortaleza
    if fund.get("debt_ebitda") is not None and fund["debt_ebitda"] <= 2.5: pts += 1.0
    if fund.get("fcf") is not None and fund["fcf"] > 0: pts += 1.0
    # Valoración
    intrinsic = fund.get("intrinsic")
    if intrinsic not in (None, np.nan) and price is not None:
        mos_ok = price <= intrinsic * (1 - MOS_THRESHOLD)
        if mos_ok: pts += 2.0
        elif price <= intrinsic * 0.9: pts += 1.0
    if fund.get("pe") is not None and fund["pe"] <= 20: pts += 0.5
    if fund.get("pb") is not None and fund["pb"] <= 3: pts += 0.5
    # Técnica
    if tech["trend_up"] and (tech["rsi_ok"] or tech["macd_up"]) and tech.get("obv_up", False): pts += 1.0
    return round(min(10.0, pts), 2)

# -------------- ANÁLISIS CON CACHÉ --------------
def run_analysis():
    """
    Ejecuta el análisis completo. Esta función es llamada cuando el caché expira.
    """
    tickers = fetch_universe(limit=UNIVERSE_LIMIT)
    log(f"Tickers a evaluar: {len(tickers)}")

    # 1) Históricos por lotes (adaptativo y paralelizado)
    hist_map = download_history_batch(tickers, period=None, batch_size=BATCH_SIZE, sleep=0, retries=1, max_workers=10)

    # 2) Prefiltro técnico + rango precio
    tech_ok = []
    for tk, hist in hist_map.items():
        price = float(hist["Close"].iloc[-1])
        if not (PRICE_MIN <= price <= PRICE_MAX): continue
        # asegurar que hay data suficiente para SMA200
        if hist.shape[0] < MA_LONG + 5: continue
        tech, _ = compute_technicals(hist)
        # Prefiltro "clásico": tendencia + (RSI y MACD)
        if tech and tech["trend_up"] and (tech["rsi_ok"] and tech["macd_up"]):
            tech_ok.append((tk, price, tech))

    # Prefiltro "suave" si quedaron pocos
    TARGET_MIN = 120
    if len(tech_ok) < TARGET_MIN:
        tech_ok = []
        for tk, hist in hist_map.items():
            price = float(hist["Close"].iloc[-1])
            if not (PRICE_MIN <= price <= PRICE_MAX) or hist.shape[0] < MA_LONG + 5: continue
            tech, _ = compute_technicals(hist)
            if tech and (hist["Close"].iloc[-1] > hist["Close"].rolling(MA_LONG).mean().iloc[-1]) and (tech["rsi_ok"] or tech["macd_up"]):
                tech_ok.append((tk, price, tech))

    tech_ok = tech_ok[:MAX_FUND_REQS]

    # 3) Fundamentales + DCF + Scorings
    log(f"Obteniendo fundamentales para {len(tech_ok)} tickers...")
    rows = []
    for tk, price, tech in tech_ok:
        fund = get_fundamentals_and_quality(tk)
        six, fund_d, tech_d, score, bonus = evaluate_conditions(fund, tech)
        bscore = buffett_score(fund, price, tech)
        intrinsic = fund.get("intrinsic")
        mos = None
        if intrinsic not in (None, np.nan):
            try: mos = (intrinsic - price)/intrinsic
            except Exception: mos = None
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

    df = pd.DataFrame(rows)

    if df.empty:
        return {"error": "Sin resultados (posible rate-limit o filtros muy estrictos)"}
    
    # Orden: primero Buffett score, luego score6, luego ROIC
    df = df.sort_values(["buffett_score","score6","roic"], ascending=[False,False,False]).reset_index(drop=True)

    # Mostrar candidatas: Buffett≥7 o score6≥SCORE_MIN
    candidates = df[(df["buffett_score"] >= 7) | (df["score6"] >= SCORE_MIN)].copy()
    if candidates.empty:
        candidates = df[(df["buffett_score"] >= 6) | (df["score6"] >= 4)].copy()

    # Convertir a formato JSON-serializable
    result = {
        "total_analyzed": len(df),
        "candidates_count": len(candidates),
        "top_10": candidates.head(10).to_dict('records'),
        "cached_at": datetime.now().isoformat(),
        "cache_ttl_seconds": CACHE_TTL_SECONDS
    }
    
    return result

def get_cached_analysis():
    """
    Retorna el análisis cacheado si está vigente, o ejecuta uno nuevo
    """
    global cached_analysis_result, cached_analysis_timestamp
    
    current_time = datetime.now()
    
    # Verificar si el caché es válido
    if cached_analysis_result is not None and cached_analysis_timestamp is not None:
        time_elapsed = (current_time - cached_analysis_timestamp).total_seconds()
        if time_elapsed < CACHE_TTL_SECONDS:
            log(f"✓ Retornando resultado cacheado (edad: {int(time_elapsed)}s)")
            result = cached_analysis_result.copy()
            result["from_cache"] = True
            result["cache_age_seconds"] = int(time_elapsed)
            return result
    
    # Caché expirado o no existe - ejecutar análisis
    log("Ejecutando nuevo análisis (caché expirado o no existe)...")
    cached_analysis_result = run_analysis()
    cached_analysis_timestamp = current_time
    
    result = cached_analysis_result.copy()
    result["from_cache"] = False
    result["cache_age_seconds"] = 0
    return result

# Flask app para Cloud Run
app = Flask(__name__)

@app.route('/')
def home():
    return jsonify({
        "status": "Warren Screener API - OPTIMIZADO",
        "version": "2.0",
        "endpoints": {
            "/analyze": "Análisis principal (con caché de 1 hora)",
            "/analyze?force_refresh=1": "Forzar nuevo análisis",
            "/health": "Health check"
        },
        "optimizations": [
            "Caché en memoria de resultados (1 hora TTL)",
            "Caché de listados S&P500 y Nasdaq-100",
            "Caché de fundamentales por ticker",
            "Descarga paralela de históricos",
            "Servidor Gunicorn recomendado"
        ]
    })

@app.route('/health')
def health():
    """Health check endpoint"""
    return jsonify({"status": "healthy", "timestamp": datetime.now().isoformat()})

@app.route('/analyze')
def analyze():
    try:
        # Permitir forzar refresh con parámetro
        from flask import request
        force_refresh = request.args.get('force_refresh', '0') == '1'
        
        if force_refresh:
            global cached_analysis_result, cached_analysis_timestamp
            cached_analysis_result = None
            cached_analysis_timestamp = None
            log("Forzando refresh del caché...")
        
        results = get_cached_analysis()
        return jsonify(results)
    except Exception as e:
        log(f"ERROR: {str(e)}")
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    import os
    port = int(os.environ.get("PORT", 8080))
    # NOTA: Para producción, usar Gunicorn en lugar de app.run()
    app.run(host="0.0.0.0", port=port, threaded=True)
