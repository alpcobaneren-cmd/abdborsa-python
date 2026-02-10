from fastapi import FastAPI, Query
from fastapi.responses import PlainTextResponse
import os, re, time, json, math
from dataclasses import dataclass
from datetime import datetime, timezone, timedelta

import numpy as np
import pandas as pd
import yfinance as yf

app = FastAPI()

# ============ AYARLAR (servis tarafı) ============
USE_SPY = True
USE_IWM = True

TRIGGER_5D_DROP = -0.15
LOOKBACK_DD_DAYS = 60

PRICE_PERIOD_1MO = "1mo"
PRICE_PERIOD_5Y  = "5y"
BACKTEST_HORIZONS = [20, 40, 60]

ATR_N = 14
RSI_N = 14
STOP_ATR_MULT = 1.5
TP_ATR_MULT   = 3.0

EDGE_MIN_N       = 50
EDGE_MIN_PUP     = 0.60
EDGE_MIN_MED     = 0.10
EDGE_MIN_TP_HIT  = 0.30
EDGE_MAX_SL_HIT  = 0.50

SCAN_MAX_STRONG  = 60

BATCH_1MO = 90
BATCH_5Y  = 25
SLEEP_BETWEEN_BATCH = 0.10

CACHE_DIR = "/tmp"
IWM_CACHE_PATH = os.path.join(CACHE_DIR, "iwm_tickers_cache.json")
SPY_CACHE_PATH = os.path.join(CACHE_DIR, "spy_tickers_cache.json")

UA = "Mozilla/5.0 (compatible; ScanBot/1.0)"
TIMEOUT = 25


# ============ HELPERS ============
def now_tr_str():
    dt = datetime.now(timezone.utc) + timedelta(hours=3)
    return dt.strftime("%d.%m.%Y %H:%M")

def clean_ticker(t):
    t = (t or "").strip().upper()
    if not t: return ""
    if t in ["-", "—"]: return ""
    if not re.fullmatch(r"[A-Z0-9\.\-]{1,15}", t):
        return ""
    return t

def uniq_keep_order(xs):
    seen=set()
    out=[]
    for x in xs:
        if x not in seen:
            out.append(x)
            seen.add(x)
    return out

def chunked(lst, n):
    for i in range(0, len(lst), n):
        yield lst[i:i+n]

def fmt_pct(x, digits=1):
    if x is None: return "?"
    try:
        if isinstance(x, (float,int,np.floating,np.integer)):
            if np.isnan(x) or np.isinf(x): return "?"
            return f"{float(x)*100:.{digits}f}%"
    except Exception:
        pass
    return "?"

def fmt_num(x, digits=2):
    if x is None: return "?"
    try:
        x = float(x)
        if np.isnan(x) or np.isinf(x): return "?"
        return f"{x:.{digits}f}"
    except Exception:
        return "?"

def fmt_money(x):
    if x is None: return "?"
    try:
        x = float(x)
        if np.isnan(x) or np.isinf(x): return "?"
        sign = "-" if x < 0 else ""
        x = abs(x)
        if x >= 1e12: return f"{sign}{x/1e12:.2f}T$"
        if x >= 1e9:  return f"{sign}{x/1e9:.2f}B$"
        if x >= 1e6:  return f"{sign}{x/1e6:.0f}M$"
        if x >= 1e3:  return f"{sign}{x/1e3:.0f}K$"
        return f"{sign}{x:.0f}$"
    except Exception:
        return "?"

def load_json(path):
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return None

def save_json(path, obj):
    try:
        with open(path, "w", encoding="utf-8") as f:
            json.dump(obj, f, ensure_ascii=False, indent=2)
    except Exception:
        pass


# ============ SPY / IWM tickers ============
import requests

def get_spy_tickers():
    url = "https://raw.githubusercontent.com/datasets/s-and-p-500-companies/master/data/constituents.csv"
    try:
        r = requests.get(url, headers={"User-Agent": UA}, timeout=TIMEOUT)
        r.raise_for_status()
        txt = r.text.lstrip("\ufeff")
        df = pd.read_csv(pd.io.common.StringIO(txt))
        sym = []
        for x in df.get("Symbol", []):
            t = clean_ticker(str(x))
            if t: sym.append(t)
        sym = uniq_keep_order(sym)
        if len(sym) >= 450:
            save_json(SPY_CACHE_PATH, {"ts": time.time(), "tickers": sym, "source": "github_sp500"})
            return sym, "github_sp500", "OK"
    except Exception:
        pass

    cached = load_json(SPY_CACHE_PATH)
    if cached and isinstance(cached, dict) and "tickers" in cached:
        sym = [clean_ticker(x) for x in cached["tickers"]]
        sym = [x for x in sym if x]
        return sym, "cache", "cache"
    return [], "none", "FAIL"

def parse_blackrock_holdings_csv(csv_text):
    csv_text = (csv_text or "").lstrip("\ufeff")
    lines = csv_text.splitlines()
    header_idx = None
    candidates = ["Ticker", "Symbol", "Holding Ticker", "Issuer Ticker"]
    for i, line in enumerate(lines[:100]):
        if any(h in line for h in candidates) and ("," in line):
            header_idx = i
            break
    if header_idx is None:
        header_idx = 0
    data = "\n".join(lines[header_idx:])
    df = pd.read_csv(pd.io.common.StringIO(data), engine="python", on_bad_lines="skip")

    col = None
    for c in df.columns:
        lc = str(c).strip().lower()
        if lc in ["ticker", "symbol", "holding ticker", "issuer ticker"]:
            col = c
            break
    if col is None:
        for c in df.columns:
            lc = str(c).lower()
            if "tick" in lc or "sym" in lc:
                col = c
                break
    if col is None:
        raise ValueError("ticker kolonu bulunamadı")

    tickers=[]
    for x in df[col].astype(str).tolist():
        t = clean_ticker(x)
        if t:
            tickers.append(t)
    tickers = uniq_keep_order(tickers)
    return tickers

def get_iwm_tickers(use_cache_if_fail=True):
    errors=[]
    url = "https://www.blackrock.com/us/individual/products/239710/ishares-russell-2000-etf/1464253357814.ajax?fileType=csv&fileName=IWM_holdings&dataType=fund"
    try:
        r = requests.get(url, headers={"User-Agent": UA}, timeout=TIMEOUT)
        if r.status_code == 200 and r.text and len(r.text) > 5000:
            tickers = parse_blackrock_holdings_csv(r.text)
            if len(tickers) >= 1500:
                meta = {"ts": time.time(), "tickers": tickers, "source": "blackrock_product_ajax", "note": f"OK ({len(tickers)})"}
                save_json(IWM_CACHE_PATH, meta)
                return tickers, "blackrock_product_ajax", meta["note"]
            else:
                errors.append(f"blackrock_product_ajax short ({len(tickers)})")
        else:
            errors.append(f"blackrock_product_ajax HTTP {r.status_code}")
    except Exception as e:
        errors.append(f"blackrock_product_ajax {e}")

    if use_cache_if_fail:
        cached = load_json(IWM_CACHE_PATH)
        if cached and isinstance(cached, dict) and "tickers" in cached:
            tickers = [clean_ticker(x) for x in cached["tickers"]]
            tickers = [x for x in tickers if x]
            if len(tickers) >= 1500:
                fresh = (time.time()-cached.get("ts",0) < 6*3600)
                return tickers, "cache", "fresh cache" if fresh else "cache"
            errors.append(f"cache short ({len(tickers)})")

    raise RuntimeError("IWM tickers alınamadı: " + " | ".join(errors))


# ============ Indicators / Data ============
def rsi(close, n=14):
    close = pd.Series(close).astype(float)
    delta = close.diff()
    gain = (delta.where(delta > 0, 0.0)).rolling(n).mean()
    loss = (-delta.where(delta < 0, 0.0)).rolling(n).mean()
    rs = gain / loss.replace(0, np.nan)
    return 100 - (100 / (1 + rs))

def atr_pct(high, low, close, n=14):
    high = pd.Series(high).astype(float)
    low  = pd.Series(low).astype(float)
    close= pd.Series(close).astype(float)
    prev_close = close.shift(1)
    tr = pd.concat([(high-low), (high-prev_close).abs(), (low-prev_close).abs()], axis=1).max(axis=1)
    atr = tr.rolling(n).mean()
    return atr / close

def yf_download_batch(tickers, period, interval="1d"):
    return yf.download(
        tickers,
        period=period,
        interval=interval,
        group_by="ticker",
        progress=False,
        threads=True,
        auto_adjust=False
    )

def extract_ohlcv(df, t):
    if df is None or len(df) == 0:
        return None
    try:
        if isinstance(df.columns, pd.MultiIndex):
            if t not in df.columns.get_level_values(0):
                return None
            d = df[t].copy()
        else:
            d = df.copy()
        for c in ["Open","High","Low","Close","Volume"]:
            if c not in d.columns:
                return None
        d = d.dropna(subset=["Close"])
        if len(d) < 10:
            return None
        return d
    except Exception:
        return None

@dataclass
class QuickRow:
    ticker: str
    close: float
    r5: float
    dd60: float
    cd_days: int
    peak60: float
    peak60_date: str
    rsi14: float
    atrp14: float
    adv20: float

def quick_metrics_from_1mo(d):
    c = d["Close"].astype(float)
    if len(c) < 15:
        return None

    r5 = (c.iloc[-1]/c.iloc[-6]-1.0) if len(c) >= 6 else np.nan

    window = c.iloc[-min(LOOKBACK_DD_DAYS, len(c)):]
    peak_val = float(window.max())
    peak_pos = window.index[window == peak_val][0]
    peak_date = str(pd.to_datetime(peak_pos).date())
    dd = float(c.iloc[-1]/peak_val - 1.0) if peak_val>0 else np.nan
    cd = int((pd.to_datetime(c.index[-1]) - pd.to_datetime(peak_pos)).days)

    rsi14 = float(rsi(c, RSI_N).iloc[-1])
    atrp = float(atr_pct(d["High"], d["Low"], c, ATR_N).iloc[-1])

    v = d["Volume"].astype(float)
    adv = float((c*v).rolling(min(20, len(c))).mean().iloc[-1])

    return float(c.iloc[-1]), float(r5), dd, cd, peak_val, peak_date, rsi14, atrp, adv

@dataclass
class BtStats:
    horizon: int
    med: float
    pup: float
    n: int
    tp_hit: float
    sl_hit: float

def is_strong(st: BtStats):
    return (
        st.n >= EDGE_MIN_N and
        st.pup >= EDGE_MIN_PUP and
        st.med >= EDGE_MIN_MED and
        st.tp_hit >= EDGE_MIN_TP_HIT and
        st.sl_hit <= EDGE_MAX_SL_HIT
    )

def compute_backtest_from_df(df5):
    if df5 is None or len(df5) < 300:
        return []
    df5 = df5.dropna(subset=["Close"])
    if len(df5) < 300:
        return []

    c = df5["Close"].astype(float)
    h = df5["High"].astype(float)
    l = df5["Low"].astype(float)

    r5 = c / c.shift(5) - 1.0
    trig_mask = (r5 <= TRIGGER_5D_DROP)
    trig_idx = df5.index[trig_mask.fillna(False)]
    if len(trig_idx) < 5:
        return []

    atrp_ser = atr_pct(h, l, c, ATR_N)

    out=[]
    for horizon in BACKTEST_HORIZONS:
        rets=[]
        tp_hits=0
        sl_hits=0
        n=0

        for t0 in trig_idx:
            i0 = df5.index.get_loc(t0)
            i1 = i0 + horizon
            if i1 >= len(df5):
                continue

            entry = float(c.iloc[i0])
            atrp0 = float(atrp_ser.iloc[i0]) if not np.isnan(atrp_ser.iloc[i0]) else np.nan
            if np.isnan(atrp0) or entry <= 0:
                continue

            sl_pct = STOP_ATR_MULT*atrp0
            tp_pct = TP_ATR_MULT*atrp0
            sl_level = entry * (1.0 - sl_pct)
            tp_level = entry * (1.0 + tp_pct)

            window_h = h.iloc[i0+1:i1+1]
            window_l = l.iloc[i0+1:i1+1]
            if len(window_h)==0 or len(window_l)==0:
                continue

            tp_hit = bool(window_h.max() >= tp_level)
            sl_hit = bool(window_l.min() <= sl_level)
            ret = float(c.iloc[i1]/entry - 1.0)

            rets.append(ret)
            tp_hits += 1 if tp_hit else 0
            sl_hits += 1 if sl_hit else 0
            n += 1

        if n == 0:
            out.append(BtStats(horizon, np.nan, np.nan, 0, np.nan, np.nan))
            continue

        rets = np.array(rets, dtype=float)
        med = float(np.nanmedian(rets))
        pup = float(np.mean(rets > 0))
        out.append(BtStats(horizon, med, pup, n, float(tp_hits/n), float(sl_hits/n)))

    return out

def pick_best_stats(stats):
    strong = [s for s in stats if is_strong(s)]
    if strong:
        strong.sort(key=lambda x: (x.med, x.pup, x.n), reverse=True)
        return strong[0]
    ok = [s for s in stats if s.n > 0 and not (np.isnan(s.med) or np.isnan(s.pup))]
    if not ok:
        return None
    ok.sort(key=lambda x: (x.med, x.pup, x.n), reverse=True)
    return ok[0]

def market_summary(etf_ticker):
    try:
        df = yf.download(etf_ticker, period="3mo", interval="1d", progress=False, auto_adjust=False)
        if df is None or len(df) < 12:
            return np.nan, np.nan, np.nan
        c = df["Close"].dropna()
        if len(c) < 12:
            return np.nan, np.nan, np.nan
        r1 = (c.iloc[-1]/c.iloc[-2]-1.0)
        r5 = (c.iloc[-1]/c.iloc[-6]-1.0) if len(c) >= 6 else np.nan
        vol10 = c.pct_change().iloc[-10:].std()
        return float(r1), float(r5), float(vol10)
    except Exception:
        return np.nan, np.nan, np.nan

def build_scan_report():
    spy, spy_src, spy_note = ([], "none", "")
    iwm, iwm_src, iwm_note = ([], "none", "")
    errors = []

    if USE_SPY:
        spy, spy_src, spy_note = get_spy_tickers()
        if not spy: errors.append("SPY alınamadı")

    if USE_IWM:
        try:
            iwm, iwm_src, iwm_note = get_iwm_tickers()
        except Exception as e:
            iwm = []
            iwm_src, iwm_note = "fail", str(e).splitlines()[0]
            errors.append("IWM alınamadı")

    universe = uniq_keep_order([*spy, *iwm])
    if not universe:
        return "⚠️ Evren boş. SPY/IWM alınamadı."

    # 1mo quick
    rows=[]
    got=0
    for batch in chunked(universe, BATCH_1MO):
        try:
            df = yf_download_batch(batch, period=PRICE_PERIOD_1MO, interval="1d")
        except Exception:
            df = None
        for t in batch:
            d = extract_ohlcv(df, t) if df is not None else None
            if d is None:
                continue
            m = quick_metrics_from_1mo(d)
            if m is None:
                continue
            close, r5, dd, cd, peak, peak_date, rsi14, atrp, adv = m
            rows.append(QuickRow(t, close, r5, dd, cd, peak, peak_date, rsi14, atrp, adv))
            got += 1
        time.sleep(SLEEP_BETWEEN_BATCH)

    triggered = [r for r in rows if (not np.isnan(r.r5) and r.r5 <= TRIGGER_5D_DROP)]
    triggered_sorted = sorted(triggered, key=lambda x: x.r5)

    cand = [r.ticker for r in triggered_sorted]
    strong_lines=[]
    total_batches = max(1, math.ceil(len(cand)/BATCH_5Y))

    for bi, batch in enumerate(chunked(cand, BATCH_5Y), start=1):
        try:
            df5 = yf_download_batch(batch, period=PRICE_PERIOD_5Y, interval="1d")
        except Exception:
            df5 = None

        for t in batch:
            d5 = extract_ohlcv(df5, t) if df5 is not None else None
            if d5 is None:
                continue
            stats = compute_backtest_from_df(d5)
            if not stats:
                continue
            best = pick_best_stats(stats)
            if best is None:
                continue
            if not is_strong(best):
                continue

            qr = next((x for x in rows if x.ticker == t), None)
            if not qr:
                continue

            strong_lines.append(
                f"• {t}: 5g {fmt_pct(qr.r5,1)} | DD {fmt_pct(qr.dd60,1)} | RSI {fmt_num(qr.rsi14,1)} | ADV {fmt_money(qr.adv20)} | "
                f"GÜÇLÜ 💎 → {best.horizon}g med {fmt_pct(best.med,1)} p(up) {best.pup*100:.1f}% "
                f"TP {best.tp_hit*100:.1f}% SL {best.sl_hit*100:.1f}% n {best.n}"
            )
            if len(strong_lines) >= SCAN_MAX_STRONG:
                break
        if len(strong_lines) >= SCAN_MAX_STRONG:
            break
        time.sleep(SLEEP_BETWEEN_BATCH)

    spy_r1, spy_r5, spy_vol10 = market_summary("SPY")
    iwm_r1, iwm_r5, iwm_vol10 = market_summary("IWM")

    market_alarm = (not np.isnan(spy_r5) and spy_r5 <= -0.03) or (not np.isnan(spy_vol10) and spy_vol10 >= 0.02)

    lines=[]
    lines.append("🚨 OTOMATİK ALARM RAPORU")
    lines.append("🔎 SCAN / RAPOR\n")
    lines.append("🧺 Evren")
    lines.append(f"• Toplam: {len(universe)}")
    lines.append(f"• SPY: {len(spy)}")
    lines.append(f"• IWM: {len(iwm)}")
    if USE_IWM:
        if iwm:
            lines.append(f"• IWM Notu: ✅ IWM OK ({iwm_src}) ({iwm_note})")
        else:
            lines.append(f"• IWM Notu: ⚠️ IWM çekilemedi ({iwm_src}).")
    lines.append(f"• 1 aylık veri gelen: {got} / {len(universe)}")
    if errors:
        lines.append("⚠️ Not: " + " | ".join(errors))
    lines.append("")
    lines.append("🧭 Piyasa Özeti")
    lines.append(f"• SPY 1g: {fmt_pct(spy_r1,1)} | 5g: {fmt_pct(spy_r5,1)} | 10g Vol: {fmt_pct(spy_vol10,1)}")
    lines.append(f"• IWM 1g: {fmt_pct(iwm_r1,1)} | 5g: {fmt_pct(iwm_r5,1)} | 10g Vol: {fmt_pct(iwm_vol10,1)}")
    lines.append(f"✅ Piyasa Alarmı: {'VAR ⚠️' if market_alarm else 'YOK ✅'}\n")
    lines.append(f"🎯 Tetiklenen (5g ≤ -15%): {len(triggered_sorted)}")
    lines.append("")
    lines.append("💎 FIRSATLAR (sadece GÜÇLÜ 💎)")
    if not strong_lines:
        lines.append("• Şu an kriterlere uyan yok.")
    else:
        lines.extend(strong_lines)
    lines.append("")
    lines.append("🧩 Detay için: /detay TICKER  (örn: /detay EU)")
    lines.append(f"⏱️ {now_tr_str()}")
    return "\n".join(lines)

def build_detay_report(ticker):
    t = clean_ticker(ticker)
    if not t:
        return "⚠️ Kullanım: /detay TICKER  (örn: /detay EU)"

    df1 = yf_download_batch([t], period=PRICE_PERIOD_1MO, interval="1d")
    d1  = extract_ohlcv(df1, t)
    if d1 is None:
        return "⚠️ 1 aylık veri çekilemedi (Close yok / delisted olabilir)."

    m = quick_metrics_from_1mo(d1)
    if m is None:
        return "⚠️ 1 aylık metrik hesaplanamadı."
    close, r5, dd, cd, peak, peak_date, rsi14, atrp, adv = m

    entry = close
    sl = entry * (1.0 - STOP_ATR_MULT*atrp) if not np.isnan(atrp) else np.nan
    tp = entry * (1.0 + TP_ATR_MULT*atrp)   if not np.isnan(atrp) else np.nan

    df5 = yf_download_batch([t], period=PRICE_PERIOD_5Y, interval="1d")
    d5  = extract_ohlcv(df5, t)
    if d5 is None:
        return "⚠️ 5 yıllık veri çekilemedi (Close yok / delisted olabilir)."

    stats = compute_backtest_from_df(d5)
    if not stats:
        stats = [BtStats(h, np.nan, np.nan, 0, np.nan, np.nan) for h in BACKTEST_HORIZONS]

    best = pick_best_stats(stats)
    best_h = "?" if best is None else best.horizon
    best_lab = "?" if best is None else ("GÜÇLÜ 💎" if is_strong(best) else "NÖTR 🟡")

    info = {}
    try:
        info = yf.Ticker(t).info or {}
    except Exception:
        info = {}

    name = (info.get("shortName") or info.get("longName") or "").strip()
    sector = (info.get("sector") or "?").strip() if isinstance(info.get("sector"), str) else "?"
    industry = (info.get("industry") or "?").strip() if isinstance(info.get("industry"), str) else "?"

    lines=[]
    lines.append("💎 DETAY RAPOR (FIRSAT ŞABLONU - FULL)")
    lines.append(f"• Hisse: {t}  ({name})" if name else f"• Hisse: {t}")
    lines.append(f"• Sektör: {sector} | Endüstri: {industry}\n")

    lines.append("🧠 Güncel Durum")
    lines.append(f"• Fiyat: {fmt_num(close,2)}")
    lines.append(f"• 5g değişim: {fmt_pct(r5,1)}")
    lines.append(f"• Drawdown (son ~60g tepeye göre): {fmt_pct(dd,1)}")
    lines.append(f"• Tepe (60g): {fmt_num(peak,2)} @ {peak_date} | Tepe üstünden geçen süre (CD): {cd} gün")
    lines.append(f"• RSI (14): {fmt_num(rsi14,1)}")
    lines.append(f"• ATR% (14): {fmt_pct(atrp,1)}")
    lines.append(f"• Ortalama $ hacim (ADV~20g): {fmt_money(adv)}\n")

    lines.append("🎯 Örnek Risk/Ödül (ATR tabanlı)")
    lines.append(f"• 🟢 Giriş: {fmt_num(entry,2)} | 🔴 Stop: {fmt_num(sl,2)} | 🟣 TP(mid): {fmt_num(tp,2)} | ⚖️ RR: 2.00\n")

    lines.append("📊 Geçmiş Benzer Durum İstatistiği (5y) — trigger: 5g ≤ -15%")
    for st in stats:
        lines.append(
            f"• {st.horizon}g: med {fmt_pct(st.med,1)} | p(up) {(st.pup*100):.1f}% | "
            f"TP {(st.tp_hit*100):.1f}% | SL {(st.sl_hit*100):.1f}% | n {st.n}"
        )
    lines.append(f"• En mantıklı vade: {best_h}g → EDGE: {best_lab}")
    lines.append("")
    lines.append(f"⏱️ {now_tr_str()}")
    return "\n".join(lines)

@app.get("/health", response_class=PlainTextResponse)
def health():
    return "OK"

@app.get("/scan", response_class=PlainTextResponse)
def scan():
    return build_scan_report()

@app.get("/detay", response_class=PlainTextResponse)
def detay(ticker: str = Query(..., min_length=1, max_length=15)):
    return build_detay_report(ticker)
