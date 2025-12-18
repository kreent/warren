# =========================================
# Screener Value + Momentum + Buffett (Cloud Run)
# CON CACHÉ EN CLOUD STORAGE DE 24 HORAS
# =========================================

import pandas as pd
import numpy as np
import yfinance as yf
import requests
import io
import sys
import time
import logging
import json
import os
from datetime import datetime, timedelta
from tqdm.auto import tqdm
from flask import Flask, jsonify
from google.cloud import storage

# Silencio de logs ruidosos
logging.getLogger("yfinance").setLevel(logging.CRITICAL)
logging.getLogger("urllib3").setLevel(logging.WARNING)

# -------- Configuración de Cloud Storage --------
GCS_BUCKET_NAME = os.environ.get("GCS_BUCKET_NAME", "warren-screener-cache")
CACHE_FILE_NAME = "screener_results.json"
CACHE_TTL_HOURS = 24

# Inicializar cliente de Cloud Storage
try:
    storage_client = storage.Client()
    bucket = storage_client.bucket(GCS_BUCKET_NAME)
    GCS_AVAILABLE = True
    print(f"✓ Cloud Storage conectado al bucket: {GCS_BUCKET_NAME}")
except Exception as e:
    print(f"⚠ Cloud Storage no disponible: {e}")
    GCS_AVAILABLE = False
    bucket = None

# -------- Parámetros generales --------
UNIVERSE_LIMIT   = 250
BATCH_SIZE       = 60
MAX_FUND_REQS    = 160

PRICE_MIN, PRICE_MAX = 2.0, 2000.0

SCORE_MIN        = 5
PE_MAX           = 15.0
PB_MAX           = 1.5
ROE_MIN          = 0.12
DEBT_EBITDA_MAX  = 3.0

RSI_MIN, RSI_MAX = 30, 60
VOL_LOOKBACK     = 20
MA_SHORT, MA_LONG= 50, 200

DISCOUNT_RATE    = 0.10
TERMINAL_G       = 0.02
MAX_GROWTH_CAP   = 0.12
MOS_THRESHOLD    = 0.30
FCF_SALES_PROXY  = 0.05

def log(msg): 
    print(msg)
    sys.stdout.flush()

# -------- Funciones de Caché con Cloud Storage --------
def get_cached_results():
    """Intenta obtener resultados del caché en Cloud Storage"""
    if not GCS_AVAILABLE:
        log("⚠ Cloud Storage no disponible, ejecutando sin caché")
        return None
    
    try:
        blob = bucket.blob(CACHE_FILE_NAME)
        
        if not blob.exists():
            log("⚠ No hay datos en caché, ejecutando análisis completo")
            return None
        
        # Descargar y parsear datos
        cache_content = blob.download_as_string()
        data = json.loads(cache_content)
        
        cache_time = datetime.fromisoformat(data.get("cached_at", ""))
        time_diff = datetime.now() - cache_time
        
        if time_diff < timedelta(hours=CACHE_TTL_HOURS):
            hours_ago = round(time_diff.total_seconds() / 3600, 1)
            log(f"✓ Usando datos del caché (generados hace {hours_ago} horas)")
            return data
        else:
            log(f"⚠ Caché expirado (más de {CACHE_TTL_HOURS}h), regenerando datos...")
            blob.delete()
            return None
            
    except Exception as e:
        log(f"⚠ Error leyendo caché: {e}")
        return None

def save_to_cache(results):
    """Guarda resultados en Cloud Storage"""
    if not GCS_AVAILABLE:
        log("⚠ Cloud Storage no disponible, no se guardará caché")
        return False
    
    try:
        cache_data = {
            "results": results,
            "cached_at": datetime.now().isoformat(),
            "expires_at": (datetime.now() + timedelta(hours=CACHE_TTL_HOURS)).isoformat()
        }
        
        blob = bucket.blob(CACHE_FILE_NAME)
        blob.upload_from_string(
            json.dumps(cache_data, default=str),
            content_type='application/json'
        )
        
        log(f"✓ Resultados guardados en caché por {CACHE_TTL_HOURS} horas")
        return True
        
    except Exception as e:
        log(f"⚠ Error guardando en caché: {e}")
        return False

# -------- Lectura de universo de tickers --------
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

def universe_sp500():
    try:
        for tb in try_read_html("https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"):
            cols = [str(c) for c in tb.columns]; lowers = [c.lower() for c in cols]
            if any(c in ("symbol","ticker") for c in lowers):
                idx = next(i for i,c in enumerate(lowers) if c in ("symbol","ticker"))
                return clean_symbols(tb.iloc[:, idx])
    except Exception: 
        pass
    
    for gh in [
        "https://raw.githubusercontent.com/datasets/s-and-p-500-companies/master/data/constituents.csv",
        "https://raw.githubusercontent.com/plotly/dash-stock-tickers/master/symbols.csv"
    ]:
        g = try_read_csv(gh)
        if g is not None:
            for c in ["Symbol","symbol","Ticker","ticker","code"]:
                if c in g.columns: 
                    return clean_symbols(g[c])
    
    return pd.Series([], dtype=str)

def universe_nasdaq100():
    try:
        for tb in try_read_html("https://en.wikipedia.org/wiki/Nasdaq-100"):
            cols = [str(c) for c in tb.columns]; lowers = [c.lower() for c in cols]
            cand_idx = [i for i,c in enumerate(lowers) if ("ticker" in c) or ("symbol" in c)]
            if cand_idx: 
                return clean_symbols(tb.iloc[:, cand_idx[0]])
    except Exception: 
        pass
    
    for gh in [
        "https://raw.githubusercontent.com/nikbearbrown/Financial-Machine-Learning/master/data/nasdaq100list.csv",
        "https://raw.githubusercontent.com/sstrickx/yahoofinance-api/master/src/test/resources/nasdaq100.csv"
    ]:
        g = try_read_csv(gh)
        if g is not None:
            for c in ["Symbol","symbol","Ticker","ticker"]:
                if c in g.columns: 
                    return clean_symbols(g[c])
    
    return pd.Series([], dtype=str)

def fetch_universe(limit=UNIVERSE_LIMIT):
    log("📊 Construyendo universo (S&P500 ∪ Nasdaq-100)…")
    sp, ndx = universe_sp500(), universe_nasdaq100()
    base = pd.Series(pd.concat([sp, ndx]).dropna().unique())
    
    if base.empty:
        log("⚠ No se pudo obtener el universo desde fuentes externas, usando fallback")
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

# -------- Indicadores técnicos --------
def ema(s, span): 
    return s.ewm(span=span, adjust=False, min_periods=span).mean()

def rsi(series, length=14):
    delta = series.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(alpha=1/length, adjust=False, min_periods=length).mean()
    avg_loss = loss.ewm(alpha=1/length, adjust=False, min_periods=length).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    return 100 - (100/(1+rs))

def macd(series, fast=12, slow=26, signal=9):
    fast_ema = ema(series, fast)
    slow_ema = ema(series, slow)
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
    close = df["Close"].copy()
    vol = df["Volume"].copy()
    high = df["High"].copy()
    low = df["Low"].copy()

    df["SMA50"] = close.rolling(MA_SHORT, min_periods=MA_SHORT).mean()
    df["SMA200"] = close.rolling(MA_LONG, min_periods=MA_LONG).mean()
    df["RSI14"] = rsi(close, 14)
    
    m, s, h = macd(close, 12, 26, 9)
    df["MACD"], df["MACDsig"], df["MACDh"] = m, s, h
    
    df["VolAvg20"] = vol.rolling(VOL_LOOKBACK, min_periods=VOL_LOOKBACK).mean()
    df["OBV"] = obv(close, vol)
    df["ATR14"] = atr(high, low, close, 14)

    valid = df.dropna()
    if valid.empty:
        return None, None
    
    last = valid.iloc[-1]
    conds = {
        "trend_up": bool(last["Close"] > last["SMA200"] and last["SMA50"] > last["SMA200"]),
        "rsi_ok": bool(RSI_MIN <= last["RSI14"] <= RSI_MAX),
        "macd_up": bool(last["MACD"] > last["MACDsig"] and last["MACDh"] > 0),
        "volume_up": bool(last["Volume"] > last["VolAvg20"] and (last["Close"] > df["Close"].iloc[-2] if len(df)>1 else True)),
        "obv_up": bool(valid["OBV"].iloc[-1] > valid["OBV"].iloc[-5] if len(valid) > 5 else False),
        "atr_pct": float(last["ATR14"]/last["Close"]) if last["Close"] else np.nan,
    }
    return conds, last

# -------- Descarga de históricos --------
def _required_period_for(ma_long=MA_LONG, extra_days=40):
    days = ma_long + extra_days
    if days <= 250: return "1y"
    elif days <= 500: return "2y"
    elif days <= 750: return "3y"
    else: return "5y"

def download_history_batch(tickers, period=None, batch_size=BATCH_SIZE, sleep=2.0, retries=2):
    if period is None:
        period = _required_period_for(MA_LONG, 40)
    
    log(f"📥 Descargando históricos: {len(tickers)} símbolos (lotes={batch_size}, period={period})…")
    batches = [tickers[i:i+batch_size] for i in range(0, len(tickers), batch_size)]
    hist_map = {}

    for batch_idx, batch in enumerate(tqdm(batches, desc="Lotes")):
        syms_str = " ".join(batch)
        attempt = 0
        
        while attempt < retries:
            try:
                data = yf.download(syms_str, period=period, group_by="ticker", progress=False, threads=True)
                
                if isinstance(data.columns, pd.MultiIndex):
                    for sym in batch:
                        try:
                            df = data[sym].dropna(how="all")
                            if df.empty or df.shape[0] < MA_LONG + 5: 
                                continue
                            hist_map[sym] = df
                        except Exception: 
                            pass
                else:
                    if len(batch) == 1 and not data.empty:
                        hist_map[batch[0]] = data
                break
                
            except Exception:
                attempt += 1
                if attempt < retries: 
                    time.sleep(sleep)
        
        if batch_idx < len(batches)-1: 
            time.sleep(sleep)
    
    log(f"✓ Históricos descargados: {len(hist_map)}/{len(tickers)}")
    return hist_map

# -------- Análisis fundamental --------
def get_fundamentals_and_quality(ticker):
    try:
        t = yf.Ticker(ticker)
        info = t.info
        
        pe = info.get("forwardPE") or info.get("trailingPE")
        pb = info.get("priceToBook")
        roe = info.get("returnOnEquity")
        debt_ebitda = info.get("debtToEbitda") or info.get("debtToEquity")
        
        gross_margin = info.get("grossMargins")
        op_margin = info.get("operatingMargins")
        net_margin = info.get("profitMargins")
        roic = info.get("returnOnCapital") or info.get("returnOnAssets")
        
        mktcap = info.get("marketCap")
        price = info.get("currentPrice") or info.get("regularMarketPrice")
        
        cashflow = t.cashflow if hasattr(t, "cashflow") else None
        income_stmt = t.financials if hasattr(t, "financials") else None
        balance = t.balance_sheet if hasattr(t, "balance_sheet") else None
        
        rev_cagr, ni_cagr = None, None
        if income_stmt is not None and not income_stmt.empty:
            try:
                rev_row = income_stmt.loc["Total Revenue"] if "Total Revenue" in income_stmt.index else None
                ni_row = income_stmt.loc["Net Income"] if "Net Income" in income_stmt.index else None
                
                if rev_row is not None and len(rev_row) >= 2:
                    r0, rN = rev_row.iloc[0], rev_row.iloc[-1]
                    if pd.notna(r0) and pd.notna(rN) and rN != 0:
                        rev_cagr = ((float(r0)/float(rN))**(1/(len(rev_row)-1))) - 1
                
                if ni_row is not None and len(ni_row) >= 2:
                    n0, nN = ni_row.iloc[0], ni_row.iloc[-1]
                    if pd.notna(n0) and pd.notna(nN) and nN > 0:
                        ni_cagr = ((float(n0)/float(nN))**(1/(len(ni_row)-1))) - 1
            except Exception: 
                pass
        
        ocf, capex, shares, revenue, net_debt = None, None, None, None, None
        
        if cashflow is not None and not cashflow.empty:
            try:
                ocf_row = cashflow.loc["Operating Cash Flow"] if "Operating Cash Flow" in cashflow.index else None
                if ocf_row is not None: 
                    ocf = ocf_row.iloc[0]
                
                capex_row = cashflow.loc["Capital Expenditure"] if "Capital Expenditure" in cashflow.index else None
                if capex_row is not None: 
                    capex = abs(capex_row.iloc[0])
            except Exception: 
                pass
        
        if balance is not None and not balance.empty:
            try:
                shares_row = balance.loc["Ordinary Shares Number"] if "Ordinary Shares Number" in balance.index else None
                if shares_row is None and "Share Issued" in balance.index:
                    shares_row = balance.loc["Share Issued"]
                if shares_row is not None: 
                    shares = shares_row.iloc[0]
                
                cash_row = balance.loc["Cash And Cash Equivalents"] if "Cash And Cash Equivalents" in balance.index else None
                debt_row = balance.loc["Total Debt"] if "Total Debt" in balance.index else None
                if cash_row is not None and debt_row is not None:
                    net_debt = float(debt_row.iloc[0]) - float(cash_row.iloc[0])
            except Exception: 
                pass
        
        if income_stmt is not None and not income_stmt.empty:
            try:
                rev_row = income_stmt.loc["Total Revenue"] if "Total Revenue" in income_stmt.index else None
                if rev_row is not None: 
                    revenue = rev_row.iloc[0]
            except Exception: 
                pass
        
        fcf = None
        try:
            if ocf is not None and capex is not None: 
                fcf = float(ocf) - float(capex)
        except Exception: 
            pass
        
        if shares in (None, 0) and mktcap and price:
            try: 
                shares = int(float(mktcap)/float(price))
            except Exception: 
                pass
        
        # DCF
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
            "fcf": fcf, 
            "fcf_positive": (fcf is not None and fcf > 0),
            "gross_margin": (float(gross_margin) if gross_margin is not None else None),
            "op_margin": (float(op_margin) if op_margin is not None else None),
            "net_margin": (float(net_margin) if net_margin is not None else None),
            "rev_cagr": (float(rev_cagr) if rev_cagr is not None else None),
            "ni_cagr": (float(ni_cagr) if ni_cagr is not None else None),
            "roic": (float(roic) if roic is not None else None),
            "shares_out": (int(shares) if shares not in (None, np.nan) else None),
            "intrinsic": (float(intrinsic) if intrinsic not in (None, np.nan) else None)
        }
    except Exception:
        return {}

# -------- Scoring --------
def evaluate_conditions(fund, tech):
    fundamentals = {
        "val_pe_pb": (fund.get("pe") is not None and fund["pe"] <= PE_MAX) and \
                     (fund.get("pb") is not None and fund["pb"] <= PB_MAX),
        "roe_ok": (fund.get("roe") is not None and fund["roe"] >= ROE_MIN),
        "debt_ok": (fund.get("debt_ebitda") is not None and fund["debt_ebitda"] <= DEBT_EBITDA_MAX),
        "fcf_pos": fund.get("fcf_positive", False)
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
    
    # Calidad (4 pts)
    if fund.get("roic") is not None and fund["roic"] >= 0.12: 
        pts += 1.5
    if fund.get("op_margin") is not None and fund["op_margin"] >= 0.15: 
        pts += 1.0
    if fund.get("gross_margin") is not None and fund["gross_margin"] >= 0.40: 
        pts += 0.5
    
    growth = np.nanmean([x for x in [fund.get("rev_cagr"), fund.get("ni_cagr")] if x is not None])
    if pd.notna(growth) and growth >= 0.05: 
        pts += 1.0
    
    # Fortaleza (2 pts)
    if fund.get("debt_ebitda") is not None and fund["debt_ebitda"] <= 2.5: 
        pts += 1.0
    if fund.get("fcf") is not None and fund["fcf"] > 0: 
        pts += 1.0
    
    # Valoración (3 pts)
    intrinsic = fund.get("intrinsic")
    if intrinsic not in (None, np.nan) and price is not None:
        mos_ok = price <= intrinsic * (1 - MOS_THRESHOLD)
        if mos_ok: 
            pts += 2.0
        elif price <= intrinsic * 0.9: 
            pts += 1.0
    
    if fund.get("pe") is not None and fund["pe"] <= 20: 
        pts += 0.5
    if fund.get("pb") is not None and fund["pb"] <= 3: 
        pts += 0.5
    
    # Técnica (1 pt)
    if tech["trend_up"] and (tech["rsi_ok"] or tech["macd_up"]) and tech.get("obv_up", False): 
        pts += 1.0
    
    return round(min(10.0, pts), 2)

# -------- Análisis principal --------
def run_analysis():
    """Ejecuta el análisis completo del screener"""
    
    # Intentar obtener del caché primero
    cached_results = get_cached_results()
    if cached_results:
        return cached_results["results"]
    
    # Si no hay caché válido, ejecutar análisis completo
    log("="*60)
    log("⏳ Iniciando análisis completo (sin caché válido)...")
    log("="*60)
    start_time = time.time()
    
    # 1. Obtener universo de tickers
    tickers = fetch_universe(limit=UNIVERSE_LIMIT)
    log(f"📋 Tickers a evaluar: {len(tickers)}")

    # 2. Descargar históricos
    hist_map = download_history_batch(tickers, period=None, batch_size=BATCH_SIZE, sleep=2.0, retries=2)

    # 3. Filtro técnico estricto
    log("🔍 Aplicando filtros técnicos...")
    tech_ok = []
    for tk, hist in hist_map.items():
        price = float(hist["Close"].iloc[-1])
        if not (PRICE_MIN <= price <= PRICE_MAX): 
            continue
        if hist.shape[0] < MA_LONG + 5: 
            continue
        
        tech, _ = compute_technicals(hist)
        if tech and tech["trend_up"] and (tech["rsi_ok"] and tech["macd_up"]):
            tech_ok.append((tk, price, tech))

    # 4. Filtro técnico suave si quedan pocos
    TARGET_MIN = 120
    if len(tech_ok) < TARGET_MIN:
        log(f"⚠ Solo {len(tech_ok)} candidatos con filtro estricto, aplicando filtro suave...")
        tech_ok = []
        for tk, hist in hist_map.items():
            price = float(hist["Close"].iloc[-1])
            if not (PRICE_MIN <= price <= PRICE_MAX) or hist.shape[0] < MA_LONG + 5: 
                continue
            
            tech, _ = compute_technicals(hist)
            if tech and (hist["Close"].iloc[-1] > hist["Close"].rolling(MA_LONG).mean().iloc[-1]) and \
               (tech["rsi_ok"] or tech["macd_up"]):
                tech_ok.append((tk, price, tech))

    tech_ok = tech_ok[:MAX_FUND_REQS]
    log(f"✓ Candidatos técnicos: {len(tech_ok)}")

    # 5. Análisis fundamental
    log("💰 Analizando fundamentales y calculando DCF...")
    rows = []
    for idx, (tk, price, tech) in enumerate(tech_ok):
        if (idx + 1) % 20 == 0:
            log(f"  Procesando {idx + 1}/{len(tech_ok)}...")
        
        fund = get_fundamentals_and_quality(tk)
        six, fund_d, tech_d, score, bonus = evaluate_conditions(fund, tech)
        bscore = buffett_score(fund, price, tech)
        
        intrinsic = fund.get("intrinsic")
        mos = None
        if intrinsic not in (None, np.nan):
            try: 
                mos = (intrinsic - price)/intrinsic
            except Exception: 
                mos = None
        
        rows.append({
            "ticker": tk, 
            "price": round(price, 2),
            "score6": score, 
            "buffett_score": bscore,
            "pe": fund.get("pe"), 
            "pb": fund.get("pb"), 
            "roe": fund.get("roe"),
            "roic": fund.get("roic"), 
            "gross_margin": fund.get("gross_margin"),
            "op_margin": fund.get("op_margin"), 
            "net_margin": fund.get("net_margin"),
            "rev_cagr": fund.get("rev_cagr"), 
            "ni_cagr": fund.get("ni_cagr"),
            "debt_ebitda": fund.get("debt_ebitda"), 
            "fcf_positive": fund.get("fcf_positive"),
            "intrinsic": intrinsic, 
            "mos": mos,
            "trend_up": tech["trend_up"], 
            "rsi_ok": tech["rsi_ok"], 
            "macd_up": tech["macd_up"],
            "obv_up": tech.get("obv_up", False), 
            "atr_pct": tech.get("atr_pct", np.nan),
            "volume_up": tech["volume_up"]
        })

    df = pd.DataFrame(rows)

    if df.empty:
        error_result = {"error": "Sin resultados (posible rate-limit o filtros muy estrictos)"}
        log("❌ Error: Sin resultados")
        return error_result
    
    # 6. Ordenar y filtrar candidatos
    log("🏆 Ordenando resultados...")
    df = df.sort_values(["buffett_score", "score6", "roic"], ascending=[False, False, False]).reset_index(drop=True)

    candidates = df[(df["buffett_score"] >= 7) | (df["score6"] >= SCORE_MIN)].copy()
    if candidates.empty:
        log("⚠ No hay candidatos con criterios estrictos, relajando filtros...")
        candidates = df[(df["buffett_score"] >= 6) | (df["score6"] >= 4)].copy()

    # 7. Preparar resultado
    execution_time = round(time.time() - start_time, 2)
    
    result = {
        "total_analyzed": len(df),
        "candidates_count": len(candidates),
        "top_10": candidates.head(10).to_dict('records'),
        "top_20": candidates.head(20).to_dict('records'),
        "generated_at": datetime.now().isoformat(),
        "cache_enabled": GCS_AVAILABLE,
        "execution_time_seconds": execution_time
    }
    
    log("="*60)
    log(f"✅ Análisis completado en {execution_time}s")
    log(f"📊 Total analizados: {len(df)}")
    log(f"⭐ Candidatos finales: {len(candidates)}")
    log("="*60)
    
    # Guardar en caché
    save_to_cache(result)
    
    return result

# -------- Flask App --------
app = Flask(__name__)

@app.route('/')
def home():
    cache_status = "enabled" if GCS_AVAILABLE else "disabled"
    return jsonify({
        "status": "Warren Screener API v2.0 (Cloud Storage Cache)",
        "version": "2.0",
        "cache": cache_status,
        "bucket": GCS_BUCKET_NAME if GCS_AVAILABLE else "not configured",
        "cache_ttl_hours": CACHE_TTL_HOURS,
        "endpoints": {
            "/analyze": "Run analysis (with 24h cache)",
            "/cache-status": "Check cache status",
            "/clear-cache": "Clear cache manually",
            "/health": "Health check"
        }
    })

@app.route('/analyze')
def analyze():
    """Endpoint principal de análisis"""
    try:
        log("\n" + "="*60)
        log("📊 Nueva petición de análisis recibida")
        log("="*60)
        
        results = run_analysis()
        return jsonify(results)
        
    except Exception as e:
        log(f"❌ Error en análisis: {str(e)}")
        import traceback
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500

@app.route('/cache-status')
def cache_status():
    """Verifica el estado del caché"""
    if not GCS_AVAILABLE:
        return jsonify({
            "cache_enabled": False, 
            "message": "Cloud Storage not available"
        })
    
    try:
        blob = bucket.blob(CACHE_FILE_NAME)
        
        if not blob.exists():
            return jsonify({
                "cache_enabled": True,
                "cache_exists": False,
                "message": "No cached data available"
            })
        
        cache_content = blob.download_as_string()
        data = json.loads(cache_content)
        
        cache_time = datetime.fromisoformat(data.get("cached_at", ""))
        expires_at = datetime.fromisoformat(data.get("expires_at", ""))
        time_remaining = expires_at - datetime.now()
        
        is_expired = time_remaining.total_seconds() <= 0
        
        return jsonify({
            "cache_enabled": True,
            "cache_exists": True,
            "is_expired": is_expired,
            "cached_at": cache_time.isoformat(),
            "expires_at": expires_at.isoformat(),
            "time_remaining_hours": round(time_remaining.total_seconds() / 3600, 2),
            "results_count": data["results"].get("total_analyzed", 0),
            "candidates_count": data["results"].get("candidates_count", 0),
            "file_size_kb": round(blob.size / 1024, 2)
        })
        
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/clear-cache')
def clear_cache():
    """Limpia el caché manualmente"""
    if not GCS_AVAILABLE:
        return jsonify({"status": "Cloud Storage not available"}), 503
    
    try:
        blob = bucket.blob(CACHE_FILE_NAME)
        if blob.exists():
            blob.delete()
            log("🗑️ Caché limpiado manualmente")
            return jsonify({
                "status": "success",
                "message": "Cache cleared successfully"
            })
        else:
            return jsonify({
                "status": "success",
                "message": "No cache to clear"
            })
    except Exception as e:
        log(f"❌ Error limpiando caché: {str(e)}")
        return jsonify({"error": str(e)}), 500

@app.route('/health')
def health():
    """Health check endpoint"""
    return jsonify({
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "cache_available": GCS_AVAILABLE
    })

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8080))
    log(f"🚀 Iniciando Warren Screener en puerto {port}")
    log(f"💾 Cache: {'Enabled' if GCS_AVAILABLE else 'Disabled'}")
    if GCS_AVAILABLE:
        log(f"🪣 Bucket: {GCS_BUCKET_NAME}")
    app.run(host="0.0.0.0", port=port)
