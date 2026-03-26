"""
Combined Prompt Generator — Macro + OHLCV (Option A + B)
=========================================================
Kết hợp hai nguồn thông tin bổ sung cho FFT:

  [VĨ MÔ]   lãi suất SBV, sự kiện lớn, calendar (Tết/earnings)
  [NẾN+VOL] candlestick pattern + volume signal từ OHLCV
  [ĐỒNG THUẬN] macro vs price-action có align không?

Nguyên tắc thiết kế:
  1. Ngắn gọn: ~120-200 tokens (tránh choán context window của LLM)
  2. Không trùng FFT: không mention Close price hay technical indicator (RSI, MACD...)
  3. Mỗi section chỉ giữ 1-2 signal quan trọng nhất
  4. Section [ĐỒNG THUẬN] giúp LLM biết hai nguồn có mâu thuẫn không

Output: prompts_v2_combined_{short|mid}_term.json
Key:    by_date (YYYY-MM-DD) + by_index (str(i))

Usage:
    python generate_combined_prompts.py               # cả short + mid term
    python generate_combined_prompts.py --pred_len 1  # chỉ short-term
"""

import argparse
import json
import os
from datetime import date
from typing import Optional, List, Tuple

import numpy as np
import pandas as pd


# ══════════════════════════════════════════════════════════════════════════════
# PHẦN 1 — MACRO DATA (từ generate_macro_prompts_v2.py, rút gọn)
# ══════════════════════════════════════════════════════════════════════════════

SBV_RATE_PERIODS = [
    (date(2009,  1,  1), date(2009, 11, 30),  7.0, "stable",
     "phuc hoi post-GFC"),
    (date(2009, 12,  1), date(2010, 10, 31),  8.0, "tightening",
     "lam phat bat dau tang"),
    (date(2010, 11,  1), date(2011,  2, 28),  9.0, "tightening",
     "lam phat vuot 11%"),
    (date(2011,  3,  1), date(2011, 10, 31), 14.0, "tightening",
     "KHUNG HOANG LAM PHAT CPI 23%"),
    (date(2011, 11,  1), date(2012,  5, 31), 13.0, "easing",
     "cat lai suat sau lam phat"),
    (date(2012,  6,  1), date(2012, 12, 31),  9.0, "easing",
     "xu ly no xau ngan hang"),
    (date(2013,  1,  1), date(2013, 12, 31),  7.0, "easing",
     "kinh te phuc hoi cham"),
    (date(2014,  1,  1), date(2015,  9, 30),  6.5, "stable",
     "lam phat thap, on dinh"),
    (date(2015, 10,  1), date(2016,  6, 30),  6.5, "stable",
     "Trung Quoc pha gia CNY, ty gia bien dong"),
    (date(2016,  7,  1), date(2017,  6, 30),  6.25, "easing",
     "co che ty gia trung tam moi"),
    (date(2017,  7,  1), date(2019,  8, 31),  6.25, "stable",
     "tang truong manh 6.8-7.1%, Basel II"),
    (date(2019,  9,  1), date(2020,  2, 29),  6.0, "easing",
     "rui ro chien tranh thuong mai My-Trung"),
    (date(2020,  3,  1), date(2020,  9, 30),  4.5, "easing",
     "COVID-19 SBV cat -150bps ho tro"),
    (date(2020, 10,  1), date(2021,  9, 30),  4.0, "stable",
     "lai suat lich su thap, TTCK bung no"),
    (date(2021, 10,  1), date(2022,  8, 31),  4.0, "stable",
     "phuc hoi post-COVID, FED bat dau tang lai"),
    (date(2022,  9,  1), date(2022, 12, 31),  6.0, "tightening",
     "KHUNG HOANG TRAI PHIEU FLC/Van Thinh Phat"),
    (date(2023,  1,  1), date(2023,  5, 31),  5.5, "tightening",
     "khung hoang BDS, SCB, tin dung dong bang"),
    (date(2023,  6,  1), date(2023, 12, 31),  4.5, "easing",
     "SBV cat 4 lan, ho tro tang truong"),
    (date(2024,  1,  1), date(2024, 12, 31),  4.5, "stable",
     "on dinh, GDP 7.1%, xuat khau manh"),
    (date(2025,  1,  1), date(2025, 12, 31),  4.5, "stable",
     "on dinh, FDI tang, muc tieu 6.5-7% GDP"),
    (date(2026,  1,  1), date(2026, 12, 31),  4.5, "stable",
     "chinh sach on dinh"),
]

# Sự kiện đặc biệt — chỉ giữ crisis/shock lớn
CRISIS_EVENTS = [
    (date(2011,  1,  1), date(2011, 12, 31), "KHUNG HOANG LAM PHAT VN (CPI 23%)"),
    (date(2012,  1,  1), date(2014, 12, 31), "xu ly no xau, VAMC"),
    (date(2020,  1, 23), date(2020,  9, 30), "COVID-19 DAI DICH lan 1"),
    (date(2021,  8,  1), date(2021, 10, 31), "COVID DELTA phong toa TPHCM"),
    (date(2022,  3,  1), date(2022,  4, 30), "FLC scandal, bat Trinh Van Quyet"),
    (date(2022,  4,  1), date(2022,  7, 31), "Tan Hoang Minh huy trai phieu 10k ty"),
    (date(2022,  9,  1), date(2023,  3, 31), "Van Thinh Phat-SCB scandal, khung hoang niem tin"),
    (date(2023,  1,  1), date(2023,  6, 30), "tin dung dong bang, BDS kho khan"),
]

TET_DATES = {
    2009: date(2009, 1, 26), 2010: date(2010, 2, 14),
    2011: date(2011, 2,  3), 2012: date(2012, 1, 23),
    2013: date(2013, 2, 10), 2014: date(2014, 1, 31),
    2015: date(2015, 2, 19), 2016: date(2016, 2,  8),
    2017: date(2017, 1, 28), 2018: date(2018, 2, 16),
    2019: date(2019, 2,  5), 2020: date(2020, 1, 25),
    2021: date(2021, 2, 12), 2022: date(2022, 2,  1),
    2023: date(2023, 1, 22), 2024: date(2024, 2, 10),
    2025: date(2025, 1, 29), 2026: date(2026, 2, 17),
}


def get_macro_signal(d: date) -> Tuple[str, float]:
    """
    Trả về (macro_text, macro_score).
    macro_score: +1 bullish, -1 bearish, 0 neutral
    """
    # SBV rate
    rate, direction, regime_desc = 4.5, "stable", "on dinh"
    for s, e, r, dr, desc in SBV_RATE_PERIODS:
        if s <= d <= e:
            rate, direction, regime_desc = r, dr, desc
            break

    dir_map  = {"tightening": "that chat", "easing": "no long", "stable": "on dinh"}
    dir_score = {"tightening": -0.5, "easing": +0.5, "stable": 0.0}
    score = dir_score.get(direction, 0.0)

    # Crisis events
    crisis = None
    for s, e, evt in CRISIS_EVENTS:
        if s <= d <= e:
            crisis = evt
            score -= 1.0
            break

    # Tet calendar
    tet_txt = None
    for y in [d.year - 1, d.year, d.year + 1]:
        tet = TET_DATES.get(y)
        if tet is None:
            continue
        delta = (tet - d).days
        if 1 <= delta <= 20:
            tet_txt = f"truoc Tet {delta}n"
            score += 0.3
            break
        elif -15 <= delta <= 0:
            tet_txt = f"vua qua Tet {-delta}n"
            break

    # Earnings season (VCB Q4: Feb, Q1: Apr-May, Q2: Jul-Aug, Q3: Oct-Nov)
    EARNINGS_MONTHS = {2: "KQKD Q4", 4: "KQKD Q1", 5: "KQKD Q1",
                       7: "KQKD Q2", 8: "KQKD Q2", 10: "KQKD Q3", 11: "KQKD Q3"}
    earnings_txt = EARNINGS_MONTHS.get(d.month)

    # Build compact text
    parts = [f"SBV {rate}% ({dir_map[direction]}), {regime_desc}"]
    if crisis:
        parts.append(crisis)
    extras = [x for x in [tet_txt, earnings_txt] if x]
    if extras:
        parts.append(" | ".join(extras))

    return ". ".join(parts), score


# ══════════════════════════════════════════════════════════════════════════════
# PHẦN 2 — OHLCV SIGNAL (từ generate_ohlcv_prompts.py, rút gọn)
# ══════════════════════════════════════════════════════════════════════════════

def candle_parts(o, h, l, c):
    body  = abs(c - o)
    rng   = h - l if h > l else 1e-9
    upper = h - max(o, c)
    lower = min(o, c) - l
    return body, upper, lower, rng


def classify_single(o, h, l, c):
    """Trả về (name, score) hoặc None. score: +2=strong_bull ... -2=strong_bear"""
    body, upper, lower, rng = candle_parts(o, h, l, c)
    bull = c >= o

    if body < 0.10 * rng:                          # Doji
        if lower > 2 * upper:   return ("Dragonfly Doji", +0.5)
        if upper > 2 * lower:   return ("Gravestone Doji", -0.5)
        return ("Doji", 0.0)
    if body < 1e-9 or rng < 1e-9:
        return None
    if lower >= 2*body and upper <= 0.3*body:       # Hammer / Hanging Man
        return ("Hammer", +1.0) if bull else ("Hanging Man", -1.0)
    if upper >= 2*body and lower <= 0.3*body:       # Shooting Star / Inv Hammer
        return ("Shooting Star", -1.0) if not bull else ("Inverted Hammer", +0.5)
    if (upper + lower) < 0.05*rng and body > 0.7*rng:  # Marubozu
        return ("Bull Marubozu", +2.0) if bull else ("Bear Marubozu", -2.0)
    if body > 0.6 * rng:
        return ("Strong Bull", +1.0) if bull else ("Strong Bear", -1.0)
    return None


def detect_multi_candle(df5: pd.DataFrame):
    """
    Kiểm tra 5 nến cuối, trả về (name, score) của pattern quan trọng nhất.
    Ưu tiên: Engulfing > Morning/Evening Star > 3 Soldiers/Crows > Inside/Outside
    """
    rows = df5.values  # shape (5,6): O,H,L,C,...
    O, H, L, C = 0, 1, 2, 3

    def body(r): return abs(r[C] - r[O])
    def rng(r):  return r[H] - r[L] if r[H] > r[L] else 1e-9
    def bull(r): return r[C] >= r[O]

    # ── Engulfing (2 nến cuối) ────────────────────────────────────────────
    p, c = rows[-2], rows[-1]
    if not bull(p) and bull(c):
        if c[O] <= min(p[O], p[C]) and c[C] >= max(p[O], p[C]):
            return ("Bullish Engulfing", +2.5)
    if bull(p) and not bull(c):
        if c[O] >= max(p[O], p[C]) and c[C] <= min(p[O], p[C]):
            return ("Bearish Engulfing", -2.5)

    # ── Morning / Evening Star (3 nến cuối) ──────────────────────────────
    c1, c2, c3 = rows[-3], rows[-2], rows[-1]
    small_c2 = body(c2) < 0.3 * rng(c2)
    if (not bull(c1) and body(c1) > 0.5*rng(c1) and small_c2
            and bull(c3) and body(c3) > 0.5*rng(c3)):
        return ("Morning Star", +2.5)
    if (bull(c1) and body(c1) > 0.5*rng(c1) and small_c2
            and not bull(c3) and body(c3) > 0.5*rng(c3)):
        return ("Evening Star", -2.5)

    # ── Three White Soldiers / Three Black Crows ──────────────────────────
    last3 = rows[-3:]
    strong = all(body(r) > 0.5*rng(r) for r in last3)
    if strong and all(bull(r) for r in last3):
        return ("3 White Soldiers", +2.0)
    if strong and all(not bull(r) for r in last3):
        return ("3 Black Crows", -2.0)

    # ── Gap (2 nến cuối) ──────────────────────────────────────────────────
    gap_pct = (rows[-1][O] - rows[-2][C]) / (rows[-2][C] + 1e-9) * 100
    if gap_pct >= 0.5:  return ("Gap Up",   +1.0)
    if gap_pct <= -0.5: return ("Gap Down", -1.0)

    # ── Inside Bar ────────────────────────────────────────────────────────
    if rows[-1][H] <= rows[-2][H] and rows[-1][L] >= rows[-2][L]:
        return ("Inside Bar", 0.0)

    return None


def get_ohlcv_signal(window: pd.DataFrame) -> Tuple[str, float]:
    """
    Trả về (ohlcv_text, ohlcv_score).
    """
    O = window["Open"].values
    H = window["High"].values
    L = window["Low"].values
    C = window["Close"].values
    V = window["Volume"].values.astype(float)
    n = len(C)

    score = 0.0
    candle_parts_list = []

    # ── Multi-candle (ưu tiên cao nhất) ──────────────────────────────────
    df5 = window.iloc[-5:][["Open","High","Low","Close","Adj Close","Volume"]]
    df5.columns = ["Open","High","Low","Close","AdjClose","Volume"]
    multi = detect_multi_candle(df5[["Open","High","Low","Close"]].values.tolist()
                                 if False else df5)

    # Gọi lại với DataFrame
    df5_arr = window.iloc[-5:].copy()
    df5_arr.columns = window.columns
    multi = _detect_multi_df(df5_arr)

    if multi:
        candle_parts_list.append(multi[0])
        score += multi[1]

    # ── Single candle hôm qua + hôm nay (nếu chưa có multi) ──────────────
    if not multi:
        for i in [-2, -1]:
            r = classify_single(O[i], H[i], L[i], C[i])
            if r:
                candle_parts_list.append(r[0])
                score += r[1]

    # ── Volume signal ─────────────────────────────────────────────────────
    vol_ma20 = np.mean(V[-20:]) if n >= 20 else np.mean(V)
    vol_ratio = V[-1] / vol_ma20 if vol_ma20 > 0 else 1.0
    price_5d = (C[-1] - C[-5]) / C[-5] * 100 if n >= 5 else 0
    vol_5d_trend = (np.mean(V[-5:]) - np.mean(V[-10:-5])) / (np.mean(V[-10:-5]) + 1e-9) * 100 if n >= 10 else 0

    vol_signal = None
    if price_5d > 1.0 and vol_5d_trend < -20:
        vol_signal = "vol giam khi gia tang (DIVERGENCE)"
        score -= 1.0
    elif price_5d > 1.0 and vol_5d_trend > 25:
        vol_signal = "vol tang xac nhan da tang"
        score += 0.8
    elif price_5d < -1.0 and vol_5d_trend > 25:
        vol_signal = "vol tang khi gia giam (SELL-OFF)"
        score -= 0.8
    elif vol_ratio > 2.5:
        direction = "tang" if C[-1] > C[-2] else "giam"
        vol_signal = f"volume spike x{vol_ratio:.1f} khi gia {direction}"
        score += 0.5 if C[-1] > C[-2] else -0.5
    elif vol_ratio < 0.35:
        vol_signal = "volume can kiet (cho dot pha)"

    # ── Price structure (ngắn gọn nhất) ──────────────────────────────────
    slope = np.polyfit(np.arange(n), C, 1)[0]
    slope_pct = slope / C[0] * 100
    if slope_pct > 0.12:
        struct = "uptrend"
        score += 0.5
    elif slope_pct < -0.12:
        struct = "downtrend"
        score -= 0.5
    else:
        struct = "sideway"

    # HH/HL check
    mid = n // 2
    hh = np.max(H[mid:]) > np.max(H[:mid])
    hl = np.min(L[mid:]) > np.min(L[:mid])
    if hh and hl:
        struct += " HH/HL"
        score += 0.3
    elif not hh and not hl:
        struct += " LH/LL"
        score -= 0.3

    # Breakout
    resist = np.percentile(H[:-5], 90) if n > 5 else H[-1]
    support = np.percentile(L[:-5], 10) if n > 5 else L[-1]
    if C[-1] > resist:
        struct += " BREAKOUT"
        score += 0.8
    elif C[-1] < support:
        struct += " BREAKDOWN"
        score -= 0.8

    # ── Build text ────────────────────────────────────────────────────────
    parts = []
    if candle_parts_list:
        parts.append("; ".join(candle_parts_list))
    parts.append(struct)
    if vol_signal:
        parts.append(vol_signal)

    return ". ".join(parts), score


def _detect_multi_df(df5: pd.DataFrame) -> Optional[Tuple[str, float]]:
    """Helper: detect_multi_candle nhận DataFrame."""
    O = df5["Open"].values
    H = df5["High"].values
    L = df5["Low"].values
    C = df5["Close"].values

    def body(i): return abs(C[i] - O[i])
    def rng(i):  return H[i] - L[i] if H[i] > L[i] else 1e-9
    def bull(i): return C[i] >= O[i]

    # Engulfing
    if not bull(-2) and bull(-1):
        if O[-1] <= min(O[-2], C[-2]) and C[-1] >= max(O[-2], C[-2]):
            return ("Bullish Engulfing", +2.5)
    if bull(-2) and not bull(-1):
        if O[-1] >= max(O[-2], C[-2]) and C[-1] <= min(O[-2], C[-2]):
            return ("Bearish Engulfing", -2.5)

    # Morning/Evening Star
    small_mid = body(-2) < 0.3 * rng(-2)
    if not bull(-3) and body(-3) > 0.5*rng(-3) and small_mid and bull(-1) and body(-1) > 0.5*rng(-1):
        return ("Morning Star", +2.5)
    if bull(-3) and body(-3) > 0.5*rng(-3) and small_mid and not bull(-1) and body(-1) > 0.5*rng(-1):
        return ("Evening Star", -2.5)

    # Three Soldiers/Crows
    strong_all = all(body(i) > 0.5*rng(i) for i in [-3, -2, -1])
    if strong_all and all(bull(i) for i in [-3, -2, -1]):
        return ("3 White Soldiers", +2.0)
    if strong_all and all(not bull(i) for i in [-3, -2, -1]):
        return ("3 Black Crows", -2.0)

    # Gap
    if C[-2] > 0:
        gap_pct = (O[-1] - C[-2]) / C[-2] * 100
        if gap_pct >= 0.5:  return ("Gap Up",   +1.0)
        if gap_pct <= -0.5: return ("Gap Down", -1.0)

    # Inside Bar
    if H[-1] <= H[-2] and L[-1] >= L[-2]:
        return ("Inside Bar", 0.0)

    return None


# ══════════════════════════════════════════════════════════════════════════════
# PHẦN 3 — COMBINE
# ══════════════════════════════════════════════════════════════════════════════

SCORE_LABEL = [
    (+3.0, "MANH BULLISH"),
    (+1.5, "BULLISH"),
    (+0.5, "NHE BULLISH"),
    (-0.5, "TRUNG LAP"),
    (-1.5, "NHE BEARISH"),
    (-3.0, "BEARISH"),
    (-99,  "MANH BEARISH"),
]

def score_to_label(s: float) -> str:
    for threshold, label in SCORE_LABEL:
        if s >= threshold:
            return label
    return "MANH BEARISH"


def build_combined_prompt(end_date: date, ohlcv_window: pd.DataFrame,
                          pred_len: int = 1) -> str:
    """
    Kết hợp macro + OHLCV thành prompt ngắn gọn.
    """
    macro_text, macro_score = get_macro_signal(end_date)
    ohlcv_text, ohlcv_score = get_ohlcv_signal(ohlcv_window)
    total_score = macro_score + ohlcv_score

    # Consensus / conflict
    same_sign = (macro_score >= 0) == (ohlcv_score >= 0)
    if abs(macro_score) < 0.3 and abs(ohlcv_score) < 0.3:
        consensus = "ca hai trung lap"
    elif same_sign:
        consensus = f"vi mo va price-action DONG THUAN {score_to_label(total_score)}"
    else:
        stronger = "vi mo" if abs(macro_score) >= abs(ohlcv_score) else "price-action"
        weaker   = "price-action" if stronger == "vi mo" else "vi mo"
        consensus = f"{stronger} {score_to_label(macro_score if stronger=='vi mo' else ohlcv_score)} nhung {weaker} trai chieu → THAN TRONG"

    horizon = "ngay tiep theo" if pred_len == 1 else f"{pred_len} ngay tiep theo"

    return (
        f"[VI MO] {macro_text}. "
        f"[GIA+VOL] {ohlcv_text}. "
        f"[TONG HOP] {consensus}. Du bao {horizon}."
    )


# ══════════════════════════════════════════════════════════════════════════════
# PHẦN 4 — MAIN LOOP
# ══════════════════════════════════════════════════════════════════════════════

def generate_prompts(ohlcv_path: str, output_path: str,
                     seq_len: int = 60, pred_len: int = 1,
                     align_path: str = None) -> dict:
    """
    align_path: nếu truyền vào (vd: vcb_stock_indicators_v2.csv), chỉ gen prompts
                cho các dates có trong file đó → tránh index mismatch.
    """
    df = pd.read_csv(ohlcv_path, parse_dates=["date"])
    df = df.sort_values("date").reset_index(drop=True)

    # Align: chỉ giữ rows có date >= start date của indicators file
    if align_path is not None:
        df_align = pd.read_csv(align_path, parse_dates=["date"])
        align_start = df_align["date"].min()
        df = df[df["date"] >= align_start].reset_index(drop=True)
        print(f"  Aligned to {os.path.basename(align_path)}: "
              f"start={align_start.date()}, rows={len(df)}")

    total      = len(df) - seq_len - pred_len + 1
    dates_all  = df["date"].dt.date.tolist()
    dates_str  = df["date"].dt.strftime("%Y-%m-%d").tolist()

    print(f"OHLCV : {ohlcv_path}")
    print(f"  Rows   : {len(df)}  ({dates_str[0]} → {dates_str[-1]})")
    print(f"  Windows: {total}  (seq_len={seq_len}, pred_len={pred_len})")
    print(f"  Output : {output_path}")

    by_date  = {}
    by_index = {}

    for i in range(total):
        window   = df.iloc[i: i + seq_len]
        end_date = dates_all[i + seq_len - 1]
        end_str  = dates_str[i + seq_len - 1]

        prompt = build_combined_prompt(end_date, window, pred_len=pred_len)

        by_date[end_str] = prompt
        by_index[str(i)] = prompt

        if (i + 1) % 500 == 0 or i == total - 1:
            print(f"  [{i+1}/{total}] {end_str}  len={len(prompt)}")

    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(by_index, f, ensure_ascii=False, indent=2)

    print(f"  Saved {len(by_index)} prompts → {output_path}\n")
    return {"by_date": by_date, "by_index": by_index}


def print_samples(prompts: dict):
    bi = prompts["by_index"]
    keys = sorted(bi.keys(), key=lambda x: int(x))
    samples = [keys[0], keys[len(keys)//5], keys[len(keys)//2],
               keys[len(keys)*4//5], keys[-1]]
    for k in samples:
        bd_key = list(prompts["by_date"].keys())[int(k)]
        print(f"\n[{bd_key}]\n  {bi[k]}")


def main():
    parser = argparse.ArgumentParser(
        description="Generate Combined (Macro + OHLCV) prompts for VCB"
    )
    parser.add_argument("--ohlcv_path", type=str,
                        default="./dataset/dataset/stock/vcb_raw_ohlcv.csv")
    parser.add_argument("--align_path", type=str,
                        default="./dataset/dataset/stock/vcb_stock_indicators_v2.csv",
                        help="Align prompt indices to this CSV start date (fix mismatch)")
    parser.add_argument("--output_dir", type=str,
                        default="./dataset/dataset/stock")
    parser.add_argument("--seq_len",  type=int, default=60)
    parser.add_argument("--pred_len", type=int, default=None,
                        help="1=short, 60=mid, None=ca hai")
    args = parser.parse_args()

    pred_lens = [1, 60] if args.pred_len is None else [args.pred_len]

    for pl in pred_lens:
        suffix   = "short_term" if pl == 1 else "mid_term"
        out_path = os.path.join(args.output_dir, f"prompts_v2_combined_{suffix}.json")
        prompts  = generate_prompts(
            ohlcv_path=args.ohlcv_path,
            output_path=out_path,
            seq_len=args.seq_len,
            pred_len=pl,
            align_path=args.align_path,
        )
        print_samples(prompts)
        print("─" * 72)

    print("\nDone! Combined prompt structure:")
    print("  [VI MO]   lai suat SBV + su kien + calendar")
    print("  [GIA+VOL] candlestick pattern + volume signal + structure")
    print("  [TONG HOP] macro vs price-action dong thuan hay trai chieu?")


if __name__ == "__main__":
    main()
