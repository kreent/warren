# =========================================
# Screener Value + Momentum + Buffett (Cloud Run)
# =========================================

import pandas as pd
import numpy as np
import yfinance as yf
import requests
import io
import sys
import time
import logging
from datetime import datetime, timedelta
from tqdm.auto import tqdm
from flask import Flask, jsonify, request

# Silencio de logs ruidosos
logging.getLogger("yfinance").setLevel(logging.CRITICAL)
logging.getLogger("urllib3").setLevel(logging.WARNING)

# -------- SISTEMA DE CACHÉ (24 HORAS) --------
CACHE_TTL_SECONDS = 86400  # 24 horas = 86400 segundos
cached_analysis_result = None
cached_analysis_timestamp = None

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

def log(msg): 
    print(msg)
    sys.stdout.flush()

# -------- Lectura robusta de listas (S&P500 ∪ Nasdaq-100) --------
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

def universe_nasdaq100():
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
    sp, ndx = universe_sp500(), universe_nasdaq100()
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

# -------- Descarga de históricos por lotes (adaptativo) --------
def _required_period_for(ma_long=MA_LONG, extra_days=40):
    days = ma_long + extra_days
    if days <= 250:   return "1y"
    elif days <= 500: return "2y"
    elif days <= 750: return "3y"
    else:             return "5y"

def download_history_batch(tickers, period=None, batch_size=BATCH_SIZE, sleep=2.0, retries=2):
    if period is None: period = _required_period_for()
    candidates_periods = [period, "2y", "3y", "5y"]
    histories = {}
    n = len(tickers)
    for i in range(0, n, batch_size):
        batch = tickers[i:i+batch_size]
        data = None
        for per in candidates_periods:
            for att in range(retries+1):
                try:
                    data = yf.download(
                        tickers=batch, period=per, interval="1d",
                        auto_adjust=False, group_by="ticker", threads=False, progress=False
                    )
                    if data is not None and not data.empty: break
                except Exception:
                    data = None
                time.sleep(sleep*(att+1))
            if data is not None and not data.empty: break
        if data is None or data.empty: continue

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
    return histories

# -------- Fundamentales robustos (TTM, CAGR, DCF) --------
def safe_get(d, keys, default=None):
    for k in keys:
        if isinstance(d, dict) and k in d and pd.notna(d[k]): return d[k]
        if isinstance(d, pd.Series) and k in d.index and pd.notna(d.loc[k]): return d.loc[k]
    return default

def last_col(df):
    if not (isinstance(df, pd.DataFrame) and not df.empty):
        return pd.Series(dtype=float)
    col = df.columns[0]; s = df[col]; s.index = s.index.astype(str); return s

def get_statement(t, name, quarterly=False):
    df = getattr(t, f"{'quarterly_' if quarterly else ''}{name}", None)
    return df if isinstance(df, pd.DataFrame) and not df.empty else None

def series_ttm_from_quarterly(df_q, label):
    try:
        s = df_q.loc[label].dropna()
        if len(s) < 4: return None
        return float(s.iloc[:4].sum())
    except Exception:
        return None

def pick_series_annual_or_ttm(t):
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
    try:
        if latest is None or past in (None, 0) or periods <= 0: return None
        ratio = float(latest)/float(past)
        if ratio <= 0: return None
        return ratio**(1/periods) - 1
    except Exception:
        return None

def cagr_from_series(df, row_label, years_pref=5):
    if not (isinstance(df, pd.DataFrame) and not df.empty): return None
    try:
        s = df.loc[row_label].dropna()
    except Exception:
        return None
    if len(s) >= years_pref:
        past, latest, periods = s.iloc[-years_pref], s.iloc[-1], years_pref-1
    elif len(s) >= 3:
        past, latest, periods = s.iloc[-3], s.iloc[-1], 2
    elif len(s) >= 2:
        past, latest, periods = s.iloc[-2], s.iloc[-1], 1
    else:
        return None
    return compute_cagr(latest, past, periods)

def get_fundamentals_and_quality(ticker):
    t = yf.Ticker(ticker)
    finfo = getattr(t, "fast_info", {}) or {}
    info  = getattr(t, "info", {}) or {}

    pe = safe_get(finfo, ["trailingPe","trailingPE","pe_ratio"]) or safe_get(info, ["trailingPE","forwardPE","priceEpsTrailing12Months"])
    pb = safe_get(finfo, ["priceToBook","price_to_book"]) or safe_get(info, ["priceToBook"])

    shares = safe_get(finfo, ["sharesOutstanding","shares_outstanding"]) or safe_get(info, ["sharesOutstanding"])
    mktcap = safe_get(info, ["marketCap"])
    price  = safe_get(finfo, ["last_price","lastPrice"]) or safe_get(info, ["currentPrice"])

    income_y = get_statement(t, "income_stmt", quarterly=False)
    bal_y    = get_statement(t, "balance_sheet", quarterly=False)
    cash_y   = get_statement(t, "cashflow", quarterly=False)

    revenue, net_income, ebitda, ocf, capex = pick_series_annual_or_ttm(t)

    inc_last = last_col(income_y); bal_last = last_col(bal_y); cf_last = last_col(cash_y)

    gp = safe_get(inc_last, ["Gross Profit","GrossProfit"])
    op_income = safe_get(inc_last, ["Operating Income","OperatingIncome","Operating Income Or Loss"])

    gross_margin = (float(gp)/float(revenue)) if (gp is not None and revenue not in (None,0)) else None
    op_margin    = (float(op_income)/float(revenue)) if (op_income is not None and revenue not in (None,0)) else None
    net_margin   = (float(net_income)/float(revenue)) if (net_income is not None and revenue not in (None,0)) else None

    rev_cagr = cagr_from_series(income_y, "Total Revenue")
    ni_cagr  = cagr_from_series(income_y, "Net Income")

    equity = safe_get(bal_last, ["Total Stockholder Equity","Stockholders Equity","TotalEquity","Total Shareholder Equity","Total stockholders' equity"])
    roe = None
    try:
        if net_income is not None and equity not in (None, 0): roe = float(net_income)/float(equity)
    except Exception: pass

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
    except Exception: pass

    ebit  = safe_get(inc_last, ["Ebit","EBIT","Operating Income","OperatingIncome"])
    tax_exp = safe_get(inc_last, ["Income Tax Expense","IncomeTaxExpense"])
    pretax  = safe_get(inc_last, ["Income Before Tax","IncomeBeforeTax","Ebt","EBT"])
    tax_rate = None
    try:
        if tax_exp is not None and pretax not in (None,0):
            tr = float(tax_exp)/abs(float(pretax))
            tax_rate = min(max(tr, 0.0), 0.35)
    except Exception: pass
    if tax_rate is None: tax_rate = 0.21

    invested_capital = None
    try:
        if equity is not None and total_debt is not None and cash is not None:
            invested_capital = float(equity) + float(total_debt) - float(cash)
    except Exception: pass

    roic = None
    try:
        if ebit is not None and invested_capital not in (None, 0):
            nopat = float(ebit) * (1 - tax_rate)
            roic = nopat / float(invested_capital)
    except Exception: pass

    debt_ebitda = None
    try:
        if ebitda not in (None, 0) and total_debt is not None:
            debt_ebitda = float(total_debt) / float(ebitda)
    except Exception: pass

    if ocf is None: ocf = safe_get(cf_last, ["Total Cash From Operating Activities","Operating Cash Flow","OperatingCashFlow"])
    if capex is None: capex = safe_get(cf_last, ["Capital Expenditures","CapitalExpenditures","Purchase Of Property Plant And Equipment"])
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

# -------------- ANÁLISIS --------------
def run_analysis():
    tickers = fetch_universe(limit=UNIVERSE_LIMIT)
    log(f"Tickers a evaluar: {len(tickers)}")

    # 1) Históricos por lotes (adaptativo)
    hist_map = download_history_batch(tickers, period=None, batch_size=BATCH_SIZE, sleep=2.0, retries=2)

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

# -------- FUNCIÓN DE CACHÉ --------
def get_cached_analysis():
    """
    Retorna el análisis cacheado si es válido (menos de 24 horas),
    o ejecuta un nuevo análisis y lo cachea.
    """
    global cached_analysis_result, cached_analysis_timestamp
    
    current_time = datetime.now()
    
    # Verificar si el caché es válido
    if cached_analysis_result is not None and cached_analysis_timestamp is not None:
        time_elapsed = (current_time - cached_analysis_timestamp).total_seconds()
        
        if time_elapsed < CACHE_TTL_SECONDS:
            # Caché válido, retornar resultado cacheado
            log(f"✓ Retornando resultado cacheado (edad: {int(time_elapsed/3600)} horas, {int((time_elapsed%3600)/60)} minutos)")
            result = cached_analysis_result.copy()
            result["from_cache"] = True
            result["cache_age_seconds"] = int(time_elapsed)
            result["cache_age_hours"] = round(time_elapsed / 3600, 2)
            return result
    
    # Caché expirado o no existe - ejecutar nuevo análisis
    log("🔄 Ejecutando nuevo análisis (caché expirado o no existe)...")
    log(f"Timestamp actual: {current_time.isoformat()}")
    
    cached_analysis_result = run_analysis()
    cached_analysis_timestamp = current_time
    
    result = cached_analysis_result.copy()
    result["from_cache"] = False
    result["cache_age_seconds"] = 0
    result["cache_age_hours"] = 0
    
    log(f"✓ Análisis completado y guardado en caché hasta: {(current_time + timedelta(seconds=CACHE_TTL_SECONDS)).isoformat()}")
    
    return result

# Flask app para Cloud Run
app = Flask(__name__)

@app.route('/')
def home():
    return jsonify({
        "status": "Warren Screener API con Caché 24h",
        "version": "1.1",
        "endpoints": {
            "/analyze": "Análisis principal (con caché de 24 horas)",
            "/analyze?force_refresh=1": "Forzar nuevo análisis ignorando caché",
            "/health": "Health check",
            "/cache-status": "Ver estado del caché"
        },
        "cache_info": {
            "ttl_hours": 24,
            "ttl_seconds": CACHE_TTL_SECONDS
        }
    })

@app.route('/health')
def health():
    """Endpoint de health check"""
    return jsonify({
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "cache_active": cached_analysis_result is not None
    })

@app.route('/cache-status')
def cache_status():
    """Ver estado del caché"""
    if cached_analysis_timestamp is None:
        return jsonify({
            "cache_exists": False,
            "message": "No hay datos en caché aún"
        })
    
    time_elapsed = (datetime.now() - cached_analysis_timestamp).total_seconds()
    time_remaining = CACHE_TTL_SECONDS - time_elapsed
    
    return jsonify({
        "cache_exists": True,
        "cached_at": cached_analysis_timestamp.isoformat(),
        "cache_age_seconds": int(time_elapsed),
        "cache_age_hours": round(time_elapsed / 3600, 2),
        "cache_valid": time_remaining > 0,
        "time_remaining_seconds": max(0, int(time_remaining)),
        "time_remaining_hours": max(0, round(time_remaining / 3600, 2)),
        "expires_at": (cached_analysis_timestamp + timedelta(seconds=CACHE_TTL_SECONDS)).isoformat()
    })

@app.route('/analyze')
def analyze():
    try:
        # Permitir forzar refresh con parámetro ?force_refresh=1
        force_refresh = request.args.get('force_refresh', '0') == '1'
        
        if force_refresh:
            global cached_analysis_result, cached_analysis_timestamp
            log("🔄 Forzando nuevo análisis (invalidando caché)...")
            cached_analysis_result = None
            cached_analysis_timestamp = None
        
        results = get_cached_analysis()
        return jsonify(results)
    except Exception as e:
        log(f"❌ ERROR: {str(e)}")
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    import os
    port = int(os.environ.get("PORT", 8080))
    app.run(host="0.0.0.0", port=port)
