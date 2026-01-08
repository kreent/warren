# =========================================
# Warren Screener V4.0 - Oracle V7 Enhanced
# Value + Momentum + Quality (Piotroski) + Buffett
# CON CACHÉ EN CLOUD STORAGE DE 24 HORAS
#
# Nuevas características v4.0:
# - Piotroski Score (0-9) para análisis de calidad
# - ROIC-based growth estimation
# - Dual DCF valuation (CAGR + ROIC methods)
# - Market cap filter (≥$5B)
# - MOS-based categories
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

# -------- Parámetros generales (IDÉNTICOS AL ORIGINAL) --------
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
        
        cache_content = blob.download_as_string()
        data = json.loads(cache_content)
        
        if "results" not in data or "cached_at" not in data:
            log("⚠ Caché corrupto, regenerando datos...")
            blob.delete()
            return None
        
        cache_time = datetime.fromisoformat(data.get("cached_at", ""))
        time_diff = datetime.now() - cache_time
        
        if time_diff < timedelta(hours=CACHE_TTL_HOURS):
            hours_ago = round(time_diff.total_seconds() / 3600, 1)
            log(f"✓ Usando datos del caché (generados hace {hours_ago} horas)")
            return data["results"]
        else:
            log(f"⚠ Caché expirado (más de {CACHE_TTL_HOURS}h), regenerando datos...")
            blob.delete()
            return None
            
    except Exception as e:
        log(f"⚠ Error leyendo caché: {e}")
        import traceback
        traceback.print_exc()
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
        json_string = json.dumps(cache_data, default=str, allow_nan=False)
        json_string = json_string.replace('NaN', 'null').replace('Infinity', 'null').replace('-Infinity', 'null')
        
        blob.upload_from_string(json_string, content_type='application/json')
        log(f"✓ Resultados guardados en caché por {CACHE_TTL_HOURS} horas")
        return True
        
    except Exception as e:
        log(f"⚠ Error guardando en caché: {e}")
        import traceback
        traceback.print_exc()
        return False

# -------- Lectura robusta de universo (ORIGINAL COMPLETO) --------
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
    
    try:
        if hasattr(yf, "tickers_sp500"):
            return pd.Series([t.replace(".","-").upper() for t in yf.tickers_sp500()])
    except Exception: 
        pass
    
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

# -------- Indicadores técnicos (ORIGINAL COMPLETO) --------
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

# -------- Descarga de históricos (ORIGINAL COMPLETO) --------
def _required_period_for(ma_long=MA_LONG, extra_days=40):
    days = ma_long + extra_days
    if days <= 250: return "1y"
    elif days <= 500: return "2y"
    elif days <= 750: return "3y"
    else: return "5y"

def download_history_batch(tickers, period=None, batch_size=BATCH_SIZE, sleep=2.0, retries=2):
    if period is None: 
        period = _required_period_for()
    
    candidates_periods = [period, "2y", "3y", "5y"]
    histories = {}
    n = len(tickers)
    
    log(f"📥 Descargando históricos: {n} símbolos (batch={batch_size}, period={period})…")
    
    for i in range(0, n, batch_size):
        batch = tickers[i:i+batch_size]
        data = None
        
        for per in candidates_periods:
            for att in range(retries+1):
                try:
                    # CRÍTICO: threads=False y auto_adjust=False como el original
                    data = yf.download(
                        tickers=batch, 
                        period=per, 
                        interval="1d",
                        auto_adjust=False,  # IMPORTANTE
                        group_by="ticker", 
                        threads=False,  # IMPORTANTE
                        progress=False
                    )
                    if data is not None and not data.empty: 
                        break
                except Exception:
                    data = None
                time.sleep(sleep*(att+1))
            if data is not None and not data.empty: 
                break
        
        if data is None or data.empty: 
            continue

        if isinstance(data.columns, pd.MultiIndex):
            for tk in batch:
                if tk in data.columns.get_level_values(0):
                    sub = data[tk].copy()
                    sub.columns = [c.title() for c in sub.columns]
                    if {"Close","Volume","High","Low"}.issubset(sub.columns) and not sub.dropna().empty:
                        histories[tk] = sub
        else:
            sub = data.copy()
            sub.columns = [c[0].title() if isinstance(c, tuple) else str(c).title() for c in sub.columns]
            if {"Close","Volume","High","Low"}.issubset(sub.columns) and not sub.dropna().empty:
                histories[batch[0]] = sub
        
        time.sleep(sleep)
    
    log(f"✓ Históricos descargados: {len(histories)}/{n}")
    return histories

# -------- Funciones auxiliares fundamentales (ORIGINAL COMPLETO) --------
def safe_get(d, keys, default=None):
    """Extrae valor de dict/Series con múltiples claves posibles"""
    for k in keys:
        if isinstance(d, dict) and k in d and pd.notna(d[k]): 
            return d[k]
        if isinstance(d, pd.Series) and k in d.index and pd.notna(d.loc[k]): 
            return d.loc[k]
    return default

def last_col(df):
    """Obtiene la última columna de un DataFrame como Series"""
    if not (isinstance(df, pd.DataFrame) and not df.empty):
        return pd.Series(dtype=float)
    col = df.columns[0]
    s = df[col]
    s.index = s.index.astype(str)
    return s

def get_statement(t, name, quarterly=False):
    """Obtiene statement financiero (quarterly o annual)"""
    df = getattr(t, f"{'quarterly_' if quarterly else ''}{name}", None)
    return df if isinstance(df, pd.DataFrame) and not df.empty else None

def series_ttm_from_quarterly(df_q, label):
    """Calcula TTM (Trailing Twelve Months) desde data quarterly"""
    try:
        s = df_q.loc[label].dropna()
        if len(s) < 4: 
            return None
        return float(s.iloc[:4].sum())
    except Exception:
        return None

def pick_series_annual_or_ttm(t):
    """
    Obtiene métricas financieras: primero intenta annual, 
    si falla calcula TTM desde quarterly
    """
    income_y = get_statement(t, "income_stmt", quarterly=False)
    cash_y   = get_statement(t, "cashflow",    quarterly=False)
    income_q = get_statement(t, "income_stmt", quarterly=True)
    cash_q   = get_statement(t, "cashflow",    quarterly=True)

    def from_annual(df, name):
        return safe_get(last_col(df), [name]) if df is not None else None

    revenue = from_annual(income_y, "Total Revenue")
    netinc  = from_annual(income_y, "Net Income")
    ebitda  = from_annual(income_y, "EBITDA")
    ocf     = from_annual(cash_y,   "Total Cash From Operating Activities")
    capex   = from_annual(cash_y,   "Capital Expenditures")

    # Fallback a TTM si falta data anual
    if revenue is None and income_q is not None:
        revenue = series_ttm_from_quarterly(income_q, "Total Revenue")
    if netinc is None and income_q is not None:
        netinc = series_ttm_from_quarterly(income_q, "Net Income")
    if ebitda is None and income_q is not None:
        ebitda = series_ttm_from_quarterly(income_q, "EBITDA")
    if ocf is None and cash_q is not None:
        ocf = series_ttm_from_quarterly(cash_q, "Total Cash From Operating Activities")
    if capex is None and cash_q is not None:
        capex = series_ttm_from_quarterly(cash_q, "Capital Expenditures")

    return revenue, netinc, ebitda, ocf, capex

def compute_cagr(latest, past, periods):
    """Calcula CAGR (Compound Annual Growth Rate)"""
    try:
        if latest is None or past in (None, 0) or periods <= 0: 
            return None
        ratio = float(latest)/float(past)
        if ratio <= 0: 
            return None
        return ratio**(1/periods) - 1
    except Exception:
        return None

def cagr_from_series(df, row_label, years_pref=5):
    """Calcula CAGR desde una serie temporal con fallback a 3 o 2 años"""
    if not (isinstance(df, pd.DataFrame) and not df.empty):
        return None
    try:
        s = df.loc[row_label].dropna()
    except Exception:
        return None

    # Intenta 5 años, luego 3, luego 2
    if len(s) >= years_pref:
        past, latest, periods = s.iloc[-years_pref], s.iloc[-1], years_pref-1
    elif len(s) >= 3:
        past, latest, periods = s.iloc[-3], s.iloc[-1], 2
    elif len(s) >= 2:
        past, latest, periods = s.iloc[-2], s.iloc[-1], 1
    else:
        return None

    return compute_cagr(latest, past, periods)

def calculate_piotroski_score(income_df, balance_df, cashflow_df):
    """
    Calcula el Piotroski Score (0-9) basado en:
    - Rentabilidad (4 pts): NI > 0, OCF > 0, ROA mejoró, OCF > NI
    - Apalancamiento (3 pts): Deuda bajó, Current Ratio mejoró, No emisión acciones
    - Eficiencia (2 pts): Margen bruto mejoró, Asset Turnover mejoró
    """
    score = 0

    try:
        # Preparar datos (ordenar cronológicamente)
        if income_df is not None and not income_df.empty:
            inc = income_df[sorted(income_df.columns, reverse=True)]
        else:
            return 5  # Score neutral si no hay datos

        if balance_df is not None and not balance_df.empty:
            bal = balance_df[sorted(balance_df.columns, reverse=True)]
        else:
            bal = None

        if cashflow_df is not None and not cashflow_df.empty:
            cf = cashflow_df[sorted(cashflow_df.columns, reverse=True)]
        else:
            cf = None

        # Extraer métricas actuales y pasadas
        ni_curr = safe_get(last_col(inc), ["Net Income", "NetIncome"])
        ni_cols = inc.columns if inc is not None else []

        # 1. Net Income > 0
        if ni_curr is not None and ni_curr > 0:
            score += 1

        # 2. Operating Cash Flow > 0
        if cf is not None:
            ocf_curr = safe_get(last_col(cf), ["Total Cash From Operating Activities", "Operating Cash Flow"])
            if ocf_curr is not None and ocf_curr > 0:
                score += 1

                # 4. OCF > Net Income (calidad de ganancias)
                if ni_curr is not None and ocf_curr > ni_curr:
                    score += 1

        # 3. ROA mejoró (comparar con año anterior)
        if bal is not None and len(ni_cols) >= 2:
            try:
                assets_curr = safe_get(last_col(bal), ["Total Assets", "TotalAssets"])

                # Columna anterior
                inc_prev = inc[inc.columns[1]]
                ni_prev = safe_get(inc_prev, ["Net Income", "NetIncome"])

                bal_prev = bal[bal.columns[1]] if len(bal.columns) >= 2 else None
                assets_prev = safe_get(bal_prev, ["Total Assets", "TotalAssets"]) if bal_prev is not None else None

                if all(v is not None and v != 0 for v in [ni_curr, ni_prev, assets_curr, assets_prev]):
                    roa_curr = float(ni_curr) / float(assets_curr)
                    roa_prev = float(ni_prev) / float(assets_prev)
                    if roa_curr > roa_prev:
                        score += 1
            except:
                pass

        # 5. Deuda/Apalancamiento bajó
        if bal is not None and len(bal.columns) >= 2:
            try:
                debt_curr = safe_get(last_col(bal), ["Total Debt", "TotalDebt", "Long Term Debt"])
                bal_prev = bal[bal.columns[1]]
                debt_prev = safe_get(bal_prev, ["Total Debt", "TotalDebt", "Long Term Debt"])

                if debt_curr is not None and debt_prev is not None:
                    if debt_curr <= debt_prev:
                        score += 1
            except:
                pass

        # 6. Current Ratio mejoró (liquidez)
        if bal is not None and len(bal.columns) >= 2:
            try:
                curr_assets = safe_get(last_col(bal), ["Current Assets", "CurrentAssets"])
                curr_liab = safe_get(last_col(bal), ["Current Liabilities", "CurrentLiabilities"])

                bal_prev = bal[bal.columns[1]]
                curr_assets_prev = safe_get(bal_prev, ["Current Assets", "CurrentAssets"])
                curr_liab_prev = safe_get(bal_prev, ["Current Liabilities", "CurrentLiabilities"])

                if all(v is not None and v != 0 for v in [curr_assets, curr_liab, curr_assets_prev, curr_liab_prev]):
                    cr_curr = float(curr_assets) / float(curr_liab)
                    cr_prev = float(curr_assets_prev) / float(curr_liab_prev)
                    if cr_curr > cr_prev:
                        score += 1
            except:
                pass

        # 7. No emisión de nuevas acciones (shares outstanding no aumentó)
        if bal is not None and len(bal.columns) >= 2:
            try:
                shares_curr = safe_get(last_col(bal), ["Ordinary Shares Number", "Share Issued", "Common Stock Shares Outstanding"])
                bal_prev = bal[bal.columns[1]]
                shares_prev = safe_get(bal_prev, ["Ordinary Shares Number", "Share Issued", "Common Stock Shares Outstanding"])

                if shares_curr is not None and shares_prev is not None:
                    if shares_curr <= shares_prev:
                        score += 1
            except:
                pass

        # 8. Margen bruto mejoró
        if len(ni_cols) >= 2:
            try:
                gp_curr = safe_get(last_col(inc), ["Gross Profit", "GrossProfit"])
                rev_curr = safe_get(last_col(inc), ["Total Revenue", "TotalRevenue"])

                inc_prev = inc[inc.columns[1]]
                gp_prev = safe_get(inc_prev, ["Gross Profit", "GrossProfit"])
                rev_prev = safe_get(inc_prev, ["Total Revenue", "TotalRevenue"])

                if all(v is not None and v != 0 for v in [gp_curr, rev_curr, gp_prev, rev_prev]):
                    gm_curr = float(gp_curr) / float(rev_curr)
                    gm_prev = float(gp_prev) / float(rev_prev)
                    if gm_curr > gm_prev:
                        score += 1
            except:
                pass

        # 9. Asset Turnover mejoró (eficiencia)
        if bal is not None and len(ni_cols) >= 2:
            try:
                rev_curr = safe_get(last_col(inc), ["Total Revenue", "TotalRevenue"])
                assets_curr = safe_get(last_col(bal), ["Total Assets", "TotalAssets"])

                inc_prev = inc[inc.columns[1]]
                rev_prev = safe_get(inc_prev, ["Total Revenue", "TotalRevenue"])

                bal_prev = bal[bal.columns[1]] if len(bal.columns) >= 2 else None
                assets_prev = safe_get(bal_prev, ["Total Assets", "TotalAssets"]) if bal_prev is not None else None

                if all(v is not None and v != 0 for v in [rev_curr, assets_curr, rev_prev, assets_prev]):
                    at_curr = float(rev_curr) / float(assets_curr)
                    at_prev = float(rev_prev) / float(assets_prev)
                    if at_curr > at_prev:
                        score += 1
            except:
                pass

        return max(0, min(9, score))

    except Exception as e:
        # En caso de error, retornar score neutral
        return 5

# -------- Análisis fundamental COMPLETO (ORIGINAL) --------
def get_fundamentals_and_quality(ticker):
    """
    Obtiene fundamentales con la LÓGICA COMPLETA del script original:
    - Extracción robusta con safe_get
    - Cálculo de TTM
    - CAGR con fallbacks
    - DCF con proxy de FCF
    """
    t = yf.Ticker(ticker)
    finfo = getattr(t, "fast_info", {}) or {}
    info  = getattr(t, "info", {}) or {}

    # PE y PB con múltiples fuentes
    pe = safe_get(finfo, ["trailingPe","trailingPE","pe_ratio"]) or \
         safe_get(info, ["trailingPE","forwardPE","priceEpsTrailing12Months"])
    pb = safe_get(finfo, ["priceToBook","price_to_book"]) or \
         safe_get(info, ["priceToBook"])

    shares = safe_get(finfo, ["sharesOutstanding","shares_outstanding"]) or \
             safe_get(info, ["sharesOutstanding"])
    mktcap = safe_get(info, ["marketCap"])
    price  = safe_get(finfo, ["last_price","lastPrice"]) or \
             safe_get(info, ["currentPrice"])

    # Statements anuales
    income_y = get_statement(t, "income_stmt", quarterly=False)
    bal_y    = get_statement(t, "balance_sheet", quarterly=False)
    cash_y   = get_statement(t, "cashflow", quarterly=False)

    # Métricas con fallback a TTM
    revenue, net_income, ebitda, ocf, capex = pick_series_annual_or_ttm(t)

    inc_last = last_col(income_y)
    bal_last = last_col(bal_y)
    cf_last  = last_col(cash_y)

    # Márgenes
    gp = safe_get(inc_last, ["Gross Profit","GrossProfit"])
    op_income = safe_get(inc_last, ["Operating Income","OperatingIncome","Operating Income Or Loss"])

    gross_margin = (float(gp)/float(revenue)) if (gp is not None and revenue not in (None,0)) else None
    op_margin    = (float(op_income)/float(revenue)) if (op_income is not None and revenue not in (None,0)) else None
    net_margin   = (float(net_income)/float(revenue)) if (net_income is not None and revenue not in (None,0)) else None

    # CAGR con fallbacks (5→3→2 años)
    rev_cagr = cagr_from_series(income_y, "Total Revenue")
    ni_cagr  = cagr_from_series(income_y, "Net Income")

    # ROE
    equity = safe_get(bal_last, [
        "Total Stockholder Equity",
        "Stockholders Equity",
        "TotalEquity",
        "Total Shareholder Equity",
        "Total stockholders' equity"
    ])
    roe = None
    try:
        if net_income is not None and equity not in (None, 0): 
            roe = float(net_income)/float(equity)
    except Exception: 
        pass

    # Deuda
    total_debt = safe_get(bal_last, ["Total Debt","TotalDebt","Short Long Term Debt","ShortLongTermDebt"])
    if total_debt is None and isinstance(bal_last, pd.Series):
        lt = safe_get(bal_last, ["Long Term Debt","LongTermDebt"]) or 0
        st = safe_get(bal_last, ["Short Term Debt","ShortTermDebt","Current Portion Of Long Term Debt"]) or 0
        total_debt = lt + st if (lt or st) else None
    
    cash = safe_get(bal_last, ["Cash","Cash And Cash Equivalents","CashAndCashEquivalents"])
    net_debt = None
    try:
        if total_debt is not None and cash is not None:
            net_debt = float(total_debt) - float(cash)
    except Exception: 
        pass

    # ROIC
    ebit  = safe_get(inc_last, ["Ebit","EBIT","Operating Income","OperatingIncome"])
    tax_exp = safe_get(inc_last, ["Income Tax Expense","IncomeTaxExpense"])
    pretax  = safe_get(inc_last, ["Income Before Tax","IncomeBeforeTax","Ebt","EBT"])
    
    tax_rate = None
    try:
        if tax_exp is not None and pretax not in (None,0):
            tr = float(tax_exp)/abs(float(pretax))
            tax_rate = min(max(tr, 0.0), 0.35)
    except Exception: 
        pass
    if tax_rate is None: 
        tax_rate = 0.21

    invested_capital = None
    try:
        if equity is not None and total_debt is not None and cash is not None:
            invested_capital = float(equity) + float(total_debt) - float(cash)
    except Exception: 
        pass

    roic = None
    try:
        if ebit is not None and invested_capital not in (None, 0):
            nopat = float(ebit) * (1 - tax_rate)
            roic = nopat / float(invested_capital)
    except Exception: 
        pass

    # Debt/EBITDA
    debt_ebitda = None
    try:
        if ebitda not in (None, 0) and total_debt is not None:
            debt_ebitda = float(total_debt) / float(ebitda)
    except Exception: 
        pass

    # FCF
    if ocf is None: 
        ocf = safe_get(cf_last, ["Total Cash From Operating Activities","Operating Cash Flow","OperatingCashFlow"])
    if capex is None: 
        capex = safe_get(cf_last, ["Capital Expenditures","CapitalExpenditures","Purchase Of Property Plant And Equipment"])
    
    fcf = None
    try:
        if ocf is not None and capex is not None: 
            fcf = float(ocf) - float(capex)
    except Exception: 
        pass

    # Shares fallback
    if shares in (None, 0) and mktcap and price:
        try:
            shares = int(float(mktcap)/float(price))
        except Exception:
            pass

    # Piotroski Score
    piotroski = calculate_piotroski_score(income_y, bal_y, cash_y)

    # ROIC-based growth estimation (Oracle V7 style)
    # Empresas con alto ROIC tienden a tener mayor capacidad de reinversión
    roic_growth = None
    if roic is not None and roic > 0:
        # Growth proxy: min(roic * 0.5, 0.14), max 3%
        roic_growth = min(roic * 0.5, 0.14)
        roic_growth = max(roic_growth, 0.03)

    # DCF con proxy prudente y estimación dual de crecimiento
    intrinsic = None
    intrinsic_roic = None  # Valoración alternativa con growth basado en ROIC

    try:
        fcf_base = fcf if fcf not in (None, np.nan) else \
                   (FCF_SALES_PROXY * float(revenue) if revenue not in (None,0) else None)

        if fcf_base is not None and shares not in (None, 0):
            # Método 1: CAGR histórico (original)
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

            # Método 2: ROIC-based growth (Oracle V7 style) - más conservador
            if roic_growth is not None:
                fcfs_roic, f_roic = [], fcf_base
                for y in range(1, 6):  # 5 años como Oracle V7
                    f_roic = f_roic * (1 + roic_growth)
                    fcfs_roic.append(f_roic / ((1+DISCOUNT_RATE)**y))

                # Terminal con 3% growth (Oracle V7 style)
                terminal_fcf = fcfs_roic[-1] * 1.03
                term_val = terminal_fcf / (DISCOUNT_RATE - 0.03)
                term_val_pv = term_val / ((1+DISCOUNT_RATE)**5)

                ev_roic = np.nansum(fcfs_roic) + term_val_pv
                equity_val_roic = ev_roic + (cash if cash is not None else 0) - (total_debt if total_debt is not None else 0)
                intrinsic_roic = float(equity_val_roic) / float(shares)
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
        "roic_growth": (float(roic_growth) if roic_growth is not None else None),
        "piotroski": int(piotroski),
        "market_cap": (float(mktcap) if mktcap not in (None, np.nan) else None),
        "shares_out": (int(shares) if shares not in (None, np.nan) else None),
        "intrinsic": (float(intrinsic) if intrinsic not in (None, np.nan) else None),
        "intrinsic_roic": (float(intrinsic_roic) if intrinsic_roic not in (None, np.nan) else None)
    }

# -------- Scoring (ORIGINAL COMPLETO) --------
def evaluate_conditions(fund, tech):
    """Evalúa condiciones fundamentales y técnicas"""
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
    """
    Buffett Score 0-10 (mejorado con Piotroski):
    - Calidad (5): ROIC, márgenes, crecimiento, Piotroski
    - Fortaleza (2): deuda, FCF
    - Valoración (2): DCF MOS, PE/PB
    - Técnica (1): tendencia + señales
    """
    pts = 0.0

    # Calidad (5 pts) - Ahora incluye Piotroski
    if fund.get("roic") is not None and fund["roic"] >= 0.12:
        pts += 1.5
    if fund.get("op_margin") is not None and fund["op_margin"] >= 0.15:
        pts += 1.0
    if fund.get("gross_margin") is not None and fund["gross_margin"] >= 0.40:
        pts += 0.5

    growth = np.nanmean([x for x in [fund.get("rev_cagr"), fund.get("ni_cagr")] if x is not None])
    if pd.notna(growth) and growth >= 0.05:
        pts += 0.5

    # Piotroski Score (0-9) → convertir a 0-1.5 pts
    piotroski = fund.get("piotroski", 5)
    if piotroski >= 7:
        pts += 1.5
    elif piotroski >= 5:
        pts += 0.75

    # Fortaleza (2 pts)
    if fund.get("debt_ebitda") is not None and fund["debt_ebitda"] <= 2.5:
        pts += 1.0
    if fund.get("fcf") is not None and fund["fcf"] > 0:
        pts += 1.0

    # Valoración (2 pts) - Ahora usa ambos intrinsic values si disponible
    intrinsic = fund.get("intrinsic")
    intrinsic_roic = fund.get("intrinsic_roic")

    # Usar el promedio si ambos están disponibles, o el que esté disponible
    intrinsic_avg = None
    if intrinsic is not None and intrinsic_roic is not None:
        intrinsic_avg = (intrinsic + intrinsic_roic) / 2
    elif intrinsic is not None:
        intrinsic_avg = intrinsic
    elif intrinsic_roic is not None:
        intrinsic_avg = intrinsic_roic

    if intrinsic_avg not in (None, np.nan) and price is not None:
        mos_ok = price <= intrinsic_avg * (1 - MOS_THRESHOLD)
        if mos_ok:
            pts += 1.5
        elif price <= intrinsic_avg * 0.9:
            pts += 0.75

    if fund.get("pe") is not None and fund["pe"] <= 20:
        pts += 0.25
    if fund.get("pb") is not None and fund["pb"] <= 3:
        pts += 0.25

    # Técnica (1 pt)
    if tech["trend_up"] and (tech["rsi_ok"] or tech["macd_up"]) and tech.get("obv_up", False):
        pts += 1.0

    return round(min(10.0, pts), 2)

# -------- FUNCIÓN AUXILIAR ORACLE V7: ANÁLISIS DE STOCK --------
def analyze_stock_oracle_v7(ticker):
    """
    Análisis fundamental puro estilo Oracle V7
    NO usa datos técnicos, solo fundamentales
    """
    try:
        t = yf.Ticker(ticker)

        # Filtro rápido de market cap
        try:
            fast = t.fast_info
            if fast.market_cap < 5_000_000_000:
                return None  # Solo > 5B Cap
            price = fast.last_price
            sector = t.info.get('sector', 'N/A')
        except:
            return None

        # Obtener statements
        inc = t.income_stmt
        bal = t.balance_sheet
        cf = t.cashflow

        if inc.empty or bal.empty or cf.empty:
            return None

        # Ordenar cronológicamente
        inc = inc[sorted(inc.columns, reverse=True)]
        bal = bal[sorted(bal.columns, reverse=True)]
        cf = cf[sorted(cf.columns, reverse=True)]

        # Extracción Fuzzy de métricas clave
        def get_fuzzy_series(df, keywords):
            if df.empty:
                return pd.Series(dtype=float)
            df.index = df.index.astype(str).str.lower().str.strip()
            for key in keywords:
                key = key.lower()
                if key in df.index:
                    return df.loc[key]
                matches = [idx for idx in df.index if key in idx]
                if matches:
                    return df.loc[min(matches, key=len)]
            return pd.Series(dtype=float)

        ni = get_fuzzy_series(inc, ['Net Income', 'NetIncome'])
        ebit = get_fuzzy_series(inc, ['EBIT', 'Operating Income'])
        ocf = get_fuzzy_series(cf, ['Operating Cash Flow', 'Total Cash From Operating Activities'])
        capex = get_fuzzy_series(cf, ['Capital Expenditures', 'Purchase of PPE'])
        equity = get_fuzzy_series(bal, ['Stockholders Equity', 'Total Equity'])
        debt = get_fuzzy_series(bal, ['Total Debt'])
        cash = get_fuzzy_series(bal, ['Cash', 'Cash And Cash Equivalents'])

        if ni.empty or ocf.empty or equity.empty:
            return None

        # --- A. CALIDAD (ROIC & PIOTROSKI) ---
        # ROIC
        curr_ebit = ebit.iloc[0] if not ebit.empty else ni.iloc[0]
        curr_eq = equity.iloc[0]
        curr_debt = debt.iloc[0] if not debt.empty else 0
        curr_cash = cash.iloc[0] if not cash.empty else 0

        invested_cap = curr_eq + curr_debt - curr_cash
        roic = (curr_ebit * 0.79) / invested_cap if invested_cap > 0 else 0

        MIN_ROIC = 0.08
        if roic < MIN_ROIC:
            return None

        # Piotroski Rápido (simplificado)
        piotroski = 0
        try:
            if len(ni) > 1:
                piotroski += 1 if ni.iloc[0] > 0 else 0
                piotroski += 1 if ocf.iloc[0] > 0 else 0
                piotroski += 1 if ni.iloc[0] > ni.iloc[1] else 0
                piotroski += 1 if ocf.iloc[0] > ni.iloc[0] else 0
                piotroski += 1 if (not debt.empty and len(debt)>1 and curr_debt <= debt.iloc[1]) else 0
            else:
                piotroski = 5  # Beneficio de la duda
        except:
            piotroski = 5

        MIN_PIOTROSKI = 5
        if piotroski < MIN_PIOTROSKI:
            return None

        # --- B. VALORACIÓN (DCF 2-Etapas) ---
        cpx_val = abs(capex.iloc[0]) if not capex.empty else 0
        fcf = ocf.iloc[0] - cpx_val

        intrinsic = 0
        mos = -0.99

        if fcf > 0:
            # Growth proxy basado en ROIC
            growth_proxy = min(roic * 0.5, 0.14)  # Max 14%
            growth_proxy = max(growth_proxy, 0.03)  # Min 3%

            # Stage 1: 5 años
            DISCOUNT_RATE = 0.09  # Tasa Oracle V7
            future_cash = 0
            for i in range(1, 6):
                val = fcf * ((1 + growth_proxy) ** i)
                future_cash += val / ((1 + DISCOUNT_RATE) ** i)

            # Stage 2: Terminal
            terminal_fcf = fcf * ((1 + growth_proxy) ** 5)
            term_val = (terminal_fcf * 1.03) / (DISCOUNT_RATE - 0.03)
            term_val_pv = term_val / ((1 + DISCOUNT_RATE) ** 5)

            ev = future_cash + term_val_pv
            equity_val = ev + curr_cash - curr_debt
            intrinsic = equity_val / fast.shares

            if intrinsic > 0:
                mos = (intrinsic - price) / intrinsic

        # FILTRO DE SALIDA
        MIN_MOS_VIEW = -0.20
        if mos < MIN_MOS_VIEW and piotroski < 7:
            return None

        return {
            'Ticker': ticker,
            'Price': price,
            'Sector': sector,
            'ROIC': roic,
            'Piotroski': piotroski,
            'Growth_Est': growth_proxy if fcf > 0 else 0.05,
            'Intrinsic': intrinsic,
            'MOS': mos
        }

    except Exception:
        return None

# -------- PIPELINE PRINCIPAL - ORACLE V7 STYLE --------
def run_analysis():
    """
    Ejecuta análisis completo con lógica Oracle V7
    Mantiene estructura de salida pero usa análisis fundamental puro
    """
    
    # Intentar obtener del caché primero
    cached_results = get_cached_results()
    if cached_results is not None:
        if isinstance(cached_results, dict):
            cached_results["from_cache"] = True
        return cached_results
    
    log("="*60)
    log("🏛️ Iniciando análisis Oracle V7 (Pure Fundamental)...")
    log("="*60)
    start_time = time.time()

    # 1. Universo (S&P 500 + Nasdaq 100)
    tickers = fetch_universe(limit=UNIVERSE_LIMIT)
    log(f"📋 Tickers a evaluar: {len(tickers)}")

    # 2. Análisis fundamental puro (NO filtros técnicos, NO históricos)
    log("💎 Aplicando análisis Oracle V7 puro (ROIC ≥ 8%, Piotroski ≥ 5, MOS ≥ -20%)...")

    results = []
    for tk in tqdm(tickers, desc="Oracle V7 Analysis"):
        oracle_result = analyze_stock_oracle_v7(tk)
        if oracle_result:
            results.append(oracle_result)

    log(f"✓ Candidatos Oracle V7: {len(results)}")

    # 3. Si no hay resultados
    if not results:
        error_result = {
            "error": "Sin resultados (posible rate-limit o datos insuficientes)",
            "total_analyzed": len(tickers),
            "candidates_count": 0,
            "from_cache": False,
            "generated_at": datetime.now().isoformat()
        }
        log("❌ Sin resultados finales")
        return error_result

    # 4. Ordenar por MOS descendente (como Oracle V7 Colab)
    log("🏆 Ordenando por MOS descendente...")
    results_sorted = sorted(results, key=lambda x: x['MOS'], reverse=True)

    # 5. Transformar a formato compatible con estructura de salida existente
    rows = []
    for r in results_sorted:
        rows.append({
            "ticker": r["Ticker"],
            "price": round(r["Price"], 2),
            "sector": r["Sector"],
            "roic": r["ROIC"],
            "piotroski": r["Piotroski"],
            "growth_est": r["Growth_Est"],
            "intrinsic": r["Intrinsic"],
            "mos": r["MOS"],
            # Campos adicionales para compatibilidad
            "category": (
                "Strong Buy" if r["MOS"] >= 0.20 else
                "Buy" if r["MOS"] >= 0.10 else
                "Fair Value" if r["MOS"] >= 0 else
                "Watch" if r["MOS"] >= -0.20 else
                "Overvalued"
            )
        })

    df = pd.DataFrame(rows)

    # 6. Resultado final
    execution_time = round(time.time() - start_time, 2)

    top_10_data = df.head(10).replace({np.nan: None}).to_dict('records')

    result = {
        "candidates_count": len(df),
        "total_analyzed": len(tickers),
        "top_10": top_10_data,
        "generated_at": datetime.now().isoformat(),
        "cache_enabled": GCS_AVAILABLE,
        "from_cache": False,
        "execution_time_seconds": execution_time,
        "oracle_v7_info": {
            "description": "Pure fundamental analysis - NO technical filters",
            "filters_applied": [
                "ROIC >= 8%",
                "Piotroski >= 5",
                "MOS >= -20%",
                "Market cap >= $5B"
            ],
            "philosophy": "Value investing (Buffett/Graham style)"
        }
    }

    log("="*60)
    log(f"✅ Análisis Oracle V7 completado en {execution_time}s")
    log(f"📊 Total analizados: {len(tickers)}")
    log(f"⭐ Candidatos finales: {len(df)}")
    log(f"💎 Top 5: {', '.join([r['Ticker'] for r in results_sorted[:5]])}")
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
        "status": "Warren Screener API v4.0 (Oracle V7 Pure Fundamental)",
        "version": "4.0",
        "cache": cache_status,
        "bucket": GCS_BUCKET_NAME if GCS_AVAILABLE else "not configured",
        "cache_ttl_hours": CACHE_TTL_HOURS,
        "oracle_v7_features": [
            "✅ NO filtros técnicos (RSI, MACD, MA) - Pure fundamental",
            "✅ Filtros: ROIC >= 8%, Piotroski >= 5, MOS >= -20%",
            "✅ Market cap >= $5B",
            "✅ 2-stage DCF with ROIC-based growth",
            "✅ Fuzzy series matching for financial data",
            "✅ Simplified Piotroski Score (5-point)",
            "✅ Results identical to Oracle V7 Colab script"
        ],
        "core_features": [
            "Cálculo TTM completo desde quarterly statements",
            "CAGR con fallbacks (5→3→2 años)",
            "safe_get() robusto para múltiples nombres de campos",
            "Threads=False y auto_adjust=False en descarga",
            "Technical analysis (RSI, MACD, MA, OBV, ATR)",
            "Top 10 mejores resultados"
        ],
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
        
        response = app.response_class(
            response=json.dumps(results, default=str, allow_nan=False)
                     .replace('NaN', 'null')
                     .replace('Infinity', 'null')
                     .replace('-Infinity', 'null'),
            status=200,
            mimetype='application/json'
        )
        return response
        
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
        "cache_available": GCS_AVAILABLE,
        "version": "3.0 - Original Logic"
    })

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8080))
    log(f"🚀 Iniciando Warren Screener en puerto {port}")
    log(f"📦 Versión: 3.0 (Lógica Original Completa)")
    log(f"💾 Cache: {'Enabled' if GCS_AVAILABLE else 'Disabled'}")
    if GCS_AVAILABLE:
        log(f"🪣 Bucket: {GCS_BUCKET_NAME}")
    app.run(host="0.0.0.0", port=port)
