"""
OHLCV Candlestick & Price-Action Prompt Generator — Option B
=============================================================
Sinh dynamic prompts từ Open/High/Low/Close/Volume.

Tại sao khác FFT?
  - FFT chỉ thấy FREQUENCY của Close price
  - Không thấy: body/shadow nến, volume pattern, breakout structure
  - Không thấy: engulfing (cần 2 nến liên tiếp), gap, inside/outside bar

Nội dung prompt (4 lớp):
  1. Candlestick patterns  : mô hình nến đơn/đôi/ba trong 5 ngày gần nhất
  2. Price structure       : trend, HH/HL, support/resistance, breakout
  3. Volume-price analysis : divergence, spike, climax, dry-up
  4. Overall signal        : BULLISH / BEARISH / NEUTRAL + confidence

Output JSON:
  { "by_date": {"YYYY-MM-DD": prompt}, "by_index": {"0": prompt, ...} }

Key = ngày CUỐI window (ngày mà từ đó ta dự báo)

Usage:
    python generate_ohlcv_prompts.py
    python generate_ohlcv_prompts.py --pred_len 60
"""

import argparse
import json
import os
from typing import Optional, List, Tuple

import numpy as np
import pandas as pd


# ─────────────────────────────────────────────────────────────────────────────
# 1. Single-candle classifiers
# ─────────────────────────────────────────────────────────────────────────────

def candle_parts(o, h, l, c):
    """Trả về (body, upper_shadow, lower_shadow, total_range)."""
    body         = abs(c - o)
    total_range  = h - l if h > l else 1e-9
    upper_shadow = h - max(o, c)
    lower_shadow = min(o, c) - l
    return body, upper_shadow, lower_shadow, total_range


def is_doji(o, h, l, c, threshold=0.10):
    """Body < threshold * total_range → indecision."""
    body, _, _, rng = candle_parts(o, h, l, c)
    return body < threshold * rng


def is_hammer(o, h, l, c):
    """
    Hammer (bullish reversal từ downtrend):
      - lower shadow >= 2x body
      - upper shadow <= 0.3 * body
      - body ở nửa trên của range
    """
    body, upper, lower, rng = candle_parts(o, h, l, c)
    if body < 1e-9 or rng < 1e-9:
        return False
    return (lower >= 2 * body) and (upper <= 0.3 * body)


def is_shooting_star(o, h, l, c):
    """
    Shooting Star (bearish reversal từ uptrend):
      - upper shadow >= 2x body
      - lower shadow <= 0.3 * body
      - body ở nửa dưới của range
    """
    body, upper, lower, rng = candle_parts(o, h, l, c)
    if body < 1e-9 or rng < 1e-9:
        return False
    return (upper >= 2 * body) and (lower <= 0.3 * body)


def is_marubozu(o, h, l, c, threshold=0.05):
    """
    Marubozu: rất ít shadow → strong conviction.
      - upper + lower shadow < threshold * total_range
    """
    body, upper, lower, rng = candle_parts(o, h, l, c)
    return (upper + lower) < threshold * rng and body > 0.7 * rng


def classify_single_candle(o, h, l, c):
    """Returns (pattern_name, direction) hoặc None."""
    body, upper, lower, rng = candle_parts(o, h, l, c)
    is_bull = c >= o

    if is_doji(o, h, l, c):
        if lower > 2 * upper:
            return ("Dragonfly Doji", "neutral_bull")   # long lower shadow → support
        if upper > 2 * lower:
            return ("Gravestone Doji", "neutral_bear")  # long upper shadow → resistance
        return ("Doji", "neutral")

    if is_hammer(o, h, l, c):
        return ("Hammer", "bullish") if is_bull else ("Hanging Man", "bearish")

    if is_shooting_star(o, h, l, c):
        return ("Shooting Star", "bearish") if not is_bull else ("Inverted Hammer", "bullish")

    if is_marubozu(o, h, l, c):
        return ("Bullish Marubozu", "strong_bull") if is_bull else ("Bearish Marubozu", "strong_bear")

    # Long body without marubozu
    if body > 0.6 * rng:
        return ("Strong Bull Candle", "bullish") if is_bull else ("Strong Bear Candle", "bearish")

    return None


# ─────────────────────────────────────────────────────────────────────────────
# 2. Multi-candle pattern detectors
# ─────────────────────────────────────────────────────────────────────────────

def detect_engulfing(candles: pd.DataFrame) -> Optional[Tuple[str, str]]:
    """
    Bullish/Bearish Engulfing từ 2 nến cuối.
    candles: DataFrame với cột O, H, L, C — lấy 2 hàng cuối.
    """
    if len(candles) < 2:
        return None
    prev = candles.iloc[-2]
    curr = candles.iloc[-1]
    po, pc = prev["Open"], prev["Close"]
    co, cc = curr["Open"], curr["Close"]

    prev_bull = pc > po
    curr_bull = cc > co

    # Bullish engulfing: prev bearish, curr bull, curr body engulfs prev body
    if not prev_bull and curr_bull:
        if co <= min(po, pc) and cc >= max(po, pc):
            return ("Bullish Engulfing", "strong_bull")

    # Bearish engulfing: prev bull, curr bear, curr body engulfs prev body
    if prev_bull and not curr_bull:
        if co >= max(po, pc) and cc <= min(po, pc):
            return ("Bearish Engulfing", "strong_bear")

    return None


def detect_morning_evening_star(candles: pd.DataFrame) -> Optional[Tuple[str, str]]:
    """
    Morning Star (bullish) / Evening Star (bearish) — 3 nến.
    """
    if len(candles) < 3:
        return None
    c1, c2, c3 = candles.iloc[-3], candles.iloc[-2], candles.iloc[-1]

    def body(c):
        return abs(c["Close"] - c["Open"])

    def is_small(c):
        _, _, _, rng = candle_parts(c["Open"], c["High"], c["Low"], c["Close"])
        return body(c) < 0.3 * rng if rng > 0 else False

    # Morning Star: c1 bearish large, c2 small, c3 bullish large
    if (c1["Close"] < c1["Open"]
            and body(c1) > 0.5 * (c1["High"] - c1["Low"])
            and is_small(c2)
            and c3["Close"] > c3["Open"]
            and body(c3) > 0.5 * (c3["High"] - c3["Low"])):
        return ("Morning Star", "strong_bull")

    # Evening Star: c1 bullish large, c2 small, c3 bearish large
    if (c1["Close"] > c1["Open"]
            and body(c1) > 0.5 * (c1["High"] - c1["Low"])
            and is_small(c2)
            and c3["Close"] < c3["Open"]
            and body(c3) > 0.5 * (c3["High"] - c3["Low"])):
        return ("Evening Star", "strong_bear")

    return None


def detect_three_soldiers_crows(candles: pd.DataFrame) -> Optional[Tuple[str, str]]:
    """
    Three White Soldiers (bullish) / Three Black Crows (bearish) — 3 nến cuối.
    """
    if len(candles) < 3:
        return None
    last3 = candles.iloc[-3:]
    bodies = [abs(r["Close"] - r["Open"]) for _, r in last3.iterrows()]
    is_bulls = [r["Close"] > r["Open"] for _, r in last3.iterrows()]
    ranges  = [(r["High"] - r["Low"]) for _, r in last3.iterrows()]

    # Kiểm tra cả 3 nến có body đủ lớn
    strong = all(b > 0.5 * rng for b, rng in zip(bodies, ranges) if rng > 0)

    if all(is_bulls) and strong:
        return ("Three White Soldiers", "strong_bull")
    if all(not b for b in is_bulls) and strong:
        return ("Three Black Crows", "strong_bear")

    return None


def detect_inside_outside_bar(candles: pd.DataFrame) -> Optional[Tuple[str, str]]:
    """
    Inside bar (compression) / Outside bar (expansion) — 2 nến cuối.
    """
    if len(candles) < 2:
        return None
    prev = candles.iloc[-2]
    curr = candles.iloc[-1]

    # Inside bar: curr H/L entirely within prev H/L
    if curr["High"] <= prev["High"] and curr["Low"] >= prev["Low"]:
        return ("Inside Bar", "neutral")   # compression → breakout pending

    # Outside bar: curr H/L entirely outside prev H/L
    if curr["High"] >= prev["High"] and curr["Low"] <= prev["Low"]:
        direction = "bullish" if curr["Close"] > curr["Open"] else "bearish"
        return ("Outside Bar", direction)  # expansion

    return None


def detect_gap(candles: pd.DataFrame, min_gap_pct: float = 0.5) -> Optional[Tuple[str, str]]:
    """Gap up / Gap down giữa close hôm trước và open hôm nay (%)."""
    if len(candles) < 2:
        return None
    prev_close = candles.iloc[-2]["Close"]
    curr_open  = candles.iloc[-1]["Open"]
    gap_pct = (curr_open - prev_close) / prev_close * 100

    if gap_pct >= min_gap_pct:
        return ("Gap Up", "bullish")
    if gap_pct <= -min_gap_pct:
        return ("Gap Down", "bearish")
    return None


# ─────────────────────────────────────────────────────────────────────────────
# 3. Price structure analysis (full window)
# ─────────────────────────────────────────────────────────────────────────────

def analyze_price_structure(window: pd.DataFrame) -> dict:
    """
    Phân tích cấu trúc giá trong 60 ngày:
      - Trend direction & strength (linear regression slope)
      - Higher highs / lower lows pattern
      - Current price vs window range (price position)
      - Breakout detection
      - Support / resistance levels
    """
    closes = window["Close"].values
    highs  = window["High"].values
    lows   = window["Low"].values
    n = len(closes)

    # Linear regression slope → trend
    x = np.arange(n)
    slope, _ = np.polyfit(x, closes, 1)
    slope_pct = slope / closes[0] * 100   # % per day

    if slope_pct > 0.15:
        trend = "uptrend manh"
        trend_dir = "bullish"
    elif slope_pct > 0.05:
        trend = "uptrend nhe"
        trend_dir = "bullish"
    elif slope_pct < -0.15:
        trend = "downtrend manh"
        trend_dir = "bearish"
    elif slope_pct < -0.05:
        trend = "downtrend nhe"
        trend_dir = "bearish"
    else:
        trend = "sideway"
        trend_dir = "neutral"

    # Higher highs / lower lows (so sanh 2 nua window)
    mid = n // 2
    first_half_high = np.max(highs[:mid])
    second_half_high = np.max(highs[mid:])
    first_half_low  = np.min(lows[:mid])
    second_half_low  = np.min(lows[mid:])

    if second_half_high > first_half_high and second_half_low > first_half_low:
        hh_hl = "HH/HL (higher highs & higher lows — uptrend confirmed)"
    elif second_half_high < first_half_high and second_half_low < first_half_low:
        hh_hl = "LH/LL (lower highs & lower lows — downtrend confirmed)"
    else:
        hh_hl = "mixed structure"

    # Price position trong range
    w_high = np.max(highs)
    w_low  = np.min(lows)
    price_pos = (closes[-1] - w_low) / (w_high - w_low) if w_high > w_low else 0.5

    if price_pos > 0.85:
        pos_str = f"gan dinh 60 ngay ({price_pos:.0%})"
    elif price_pos < 0.15:
        pos_str = f"gan day 60 ngay ({price_pos:.0%})"
    elif price_pos > 0.6:
        pos_str = f"nua tren range ({price_pos:.0%})"
    else:
        pos_str = f"nua duoi range ({price_pos:.0%})"

    # Breakout: close > 90th percentile high or < 10th percentile low (trừ 5 ngày cuối)
    hist_highs = highs[:-5]
    hist_lows  = lows[:-5]
    resist = np.percentile(hist_highs, 90)
    support = np.percentile(hist_lows, 10)
    recent_close = closes[-1]

    if recent_close > resist:
        breakout = f"BREAKOUT len tren vung khang cu {resist:.0f}"
    elif recent_close < support:
        breakout = f"BREAKDOWN duoi vung ho tro {support:.0f}"
    else:
        breakout = None

    # ATR (Average True Range) 14 ngày — measure volatility
    tr_list = []
    for i in range(1, min(15, n)):
        tr = max(highs[i] - lows[i],
                 abs(highs[i] - closes[i-1]),
                 abs(lows[i]  - closes[i-1]))
        tr_list.append(tr)
    atr = np.mean(tr_list) if tr_list else 0
    atr_pct = atr / closes[-1] * 100

    if atr_pct > 3.0:
        vol_regime = "bien dong cao (ATR >3%)"
    elif atr_pct > 1.5:
        vol_regime = "bien dong trung binh"
    else:
        vol_regime = "bien dong thap (ATR <1.5%)"

    return {
        "trend": trend,
        "trend_dir": trend_dir,
        "hh_hl": hh_hl,
        "price_pos": pos_str,
        "breakout": breakout,
        "vol_regime": vol_regime,
        "slope_pct": slope_pct,
    }


# ─────────────────────────────────────────────────────────────────────────────
# 4. Volume-price analysis
# ─────────────────────────────────────────────────────────────────────────────

def analyze_volume(window: pd.DataFrame) -> dict:
    """
    Volume-price relationship trong window:
      - Volume trend (tăng/giảm theo giá → confirm/diverge)
      - Volume spike (đột biến volume)
      - Volume climax (extreme high)
      - Volume dry-up (cạn volume)
    """
    closes  = window["Close"].values
    volumes = window["Volume"].values.astype(float)
    n = len(closes)

    # Rolling 20-ngày average để normalize
    vol_ma20 = np.mean(volumes[-20:]) if n >= 20 else np.mean(volumes)
    last_vol = volumes[-1]
    vol_ratio = last_vol / vol_ma20 if vol_ma20 > 0 else 1.0

    # Volume trend (5 ngày cuối vs 5 ngày trước đó)
    vol_recent  = np.mean(volumes[-5:])
    vol_prev    = np.mean(volumes[-10:-5]) if n >= 10 else vol_recent
    vol_trend   = (vol_recent - vol_prev) / (vol_prev + 1e-9) * 100

    # Price trend 5 ngày cuối
    price_trend_5 = (closes[-1] - closes[-5]) / closes[-5] * 100 if n >= 5 else 0

    # Volume-price divergence
    signals = []

    if price_trend_5 > 1.0 and vol_trend < -20:
        signals.append("DIVERGENCE: gia tang nhung volume giam → rally yeu, co the dao chieu")
    elif price_trend_5 < -1.0 and vol_trend < -20:
        signals.append("pullback voi volume thap → co the la correction nhe, khong phai reversal lon")
    elif price_trend_5 > 1.0 and vol_trend > 30:
        signals.append("gia tang kem volume tang → BREAKOUT co xac nhan")
    elif price_trend_5 < -1.0 and vol_trend > 30:
        signals.append("gia giam kem volume tang → SELL-OFF manh, ap luc ban lon")

    # Volume spike (ngày cuối)
    if vol_ratio > 3.0:
        direction = "tang" if closes[-1] > closes[-2] else "giam"
        signals.append(f"VOLUME SPIKE x{vol_ratio:.1f} lan MA20 khi gia {direction} → su kien lon")
    elif vol_ratio > 2.0:
        signals.append(f"volume cao bat thuong x{vol_ratio:.1f} lan MA20")
    elif vol_ratio < 0.3:
        signals.append("volume can kiet (dry-up) → thi truong thieu dong luc, cho dot pha")

    # Volume climax: highest volume in window in last 5 days
    if np.max(volumes[-5:]) == np.max(volumes) and np.max(volumes) > 2 * vol_ma20:
        signals.append("dinh volume trong 60 ngay xuat hien gan day → co the la exhaustion point")

    # Accumulation/Distribution: consistently high volume on up days
    recent = window.iloc[-10:]
    up_vol   = recent.loc[recent["Close"] >= recent["Open"], "Volume"].mean()
    down_vol = recent.loc[recent["Close"] <  recent["Open"], "Volume"].mean()
    if not np.isnan(up_vol) and not np.isnan(down_vol) and down_vol > 0:
        ratio = up_vol / down_vol
        if ratio > 1.8:
            signals.append("volume ngay tang lon hon ngay giam x{:.1f} → tich luy (accumulation)".format(ratio))
        elif ratio < 0.55:
            signals.append("volume ngay giam lon hon ngay tang x{:.1f} → phan phat (distribution)".format(1/ratio))

    if not signals:
        signals.append("volume on dinh, khong co tin hieu bất thuong")

    return {
        "signals": signals,
        "vol_ratio": vol_ratio,
        "vol_trend": vol_trend,
    }


# ─────────────────────────────────────────────────────────────────────────────
# 5. Aggregate all signals → prompt
# ─────────────────────────────────────────────────────────────────────────────

DIRECTION_SCORE = {
    "strong_bull": +2,
    "bullish":     +1,
    "neutral_bull":+0.5,
    "neutral":      0,
    "neutral_bear":-0.5,
    "bearish":     -1,
    "strong_bear": -2,
}


def build_ohlcv_prompt(window: pd.DataFrame, pred_len: int = 1) -> str:
    """
    Tổng hợp tất cả signals từ OHLCV window thành prompt.
    window: DataFrame 60 hàng với cột Open/High/Low/Close/Adj Close/Volume.
    """
    # ── Candlestick patterns (5 ngày gần nhất) ──────────────────────────────
    recent5 = window.iloc[-5:].rename(columns={"Open": "Open", "High": "High",
                                               "Low": "Low", "Close": "Close"})

    candle_signals = []
    score = 0.0

    # Single candle patterns (3 ngày cuối)
    for i in [-3, -2, -1]:
        row = window.iloc[i]
        result = classify_single_candle(row["Open"], row["High"], row["Low"], row["Close"])
        if result:
            name, direction = result
            day_label = {-3: "3 ngay truoc", -2: "hom qua", -1: "hom nay"}[i]
            candle_signals.append(f"{name} ({day_label})")
            score += DIRECTION_SCORE.get(direction, 0)

    # Multi-candle patterns
    for detector in [detect_engulfing, detect_morning_evening_star,
                     detect_three_soldiers_crows, detect_inside_outside_bar,
                     detect_gap]:
        result = detector(recent5)
        if result:
            name, direction = result
            candle_signals.append(name)
            score += DIRECTION_SCORE.get(direction, 0) * 1.5   # multi-candle có weight cao hơn

    # ── Price structure ──────────────────────────────────────────────────────
    structure = analyze_price_structure(window)

    # ── Volume ──────────────────────────────────────────────────────────────
    vol_info = analyze_volume(window)

    # ── Score tổng → overall signal ─────────────────────────────────────────
    if score >= 3:
        overall = "BULLISH MANH"
    elif score >= 1.5:
        overall = "BULLISH"
    elif score >= 0.5:
        overall = "BULLISH NHE"
    elif score <= -3:
        overall = "BEARISH MANH"
    elif score <= -1.5:
        overall = "BEARISH"
    elif score <= -0.5:
        overall = "BEARISH NHE"
    else:
        overall = "TRUNG LAP"

    # ── Assemble ─────────────────────────────────────────────────────────────
    parts = []

    # Layer 1: Candlestick
    if candle_signals:
        parts.append("[Nen] " + "; ".join(candle_signals) + ".")
    else:
        parts.append("[Nen] Khong co mo hinh nen dac biet.")

    # Layer 2: Structure
    struct_parts = [structure["trend"], structure["hh_hl"], structure["price_pos"]]
    if structure["breakout"]:
        struct_parts.append(structure["breakout"])
    struct_parts.append(structure["vol_regime"])
    parts.append("[Cau truc] " + " | ".join(struct_parts) + ".")

    # Layer 3: Volume
    vol_str = " ".join(vol_info["signals"][:2])  # giữ tối đa 2 signals để ngắn gọn
    parts.append("[Volume] " + vol_str + ".")

    # Layer 4: Overall signal + task
    horizon = "ngay tiep theo" if pred_len == 1 else f"{pred_len} ngay tiep theo"
    parts.append(f"[Tin hieu tong hop] {overall}. Du bao {horizon}.")

    return " ".join(parts)


# ─────────────────────────────────────────────────────────────────────────────
# 6. Main generation loop
# ─────────────────────────────────────────────────────────────────────────────

def generate_prompts(ohlcv_path: str, output_path: str,
                     seq_len: int = 60, pred_len: int = 1) -> dict:
    """
    Sinh prompts cho toan bo dataset OHLCV.

    Output JSON:
      { "by_date": {"YYYY-MM-DD": prompt}, "by_index": {"0": prompt, ...} }
    """
    df = pd.read_csv(ohlcv_path, parse_dates=["date"])
    df = df.sort_values("date").reset_index(drop=True)

    total = len(df) - seq_len - pred_len + 1
    dates_all = df["date"].dt.strftime("%Y-%m-%d").tolist()

    print(f"OHLCV: {ohlcv_path}")
    print(f"  Rows   : {len(df)}  ({dates_all[0]} → {dates_all[-1]})")
    print(f"  Windows: {total}  (seq_len={seq_len}, pred_len={pred_len})")
    print(f"  Output : {output_path}")

    by_date  = {}
    by_index = {}

    for i in range(total):
        window   = df.iloc[i: i + seq_len]
        end_date = dates_all[i + seq_len - 1]   # last day of input
        prompt   = build_ohlcv_prompt(window, pred_len=pred_len)

        by_date[end_date] = prompt
        by_index[str(i)]  = prompt

        if (i + 1) % 500 == 0 or i == total - 1:
            print(f"  [{i+1}/{total}] {end_date}  len={len(prompt)}")

    out = {"by_date": by_date, "by_index": by_index}
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)

    print(f"Saved {len(by_date)} prompts → {output_path}\n")
    return out


def print_samples(prompts: dict, n: int = 6):
    by_index = prompts["by_index"]
    keys = sorted(by_index.keys(), key=lambda x: int(x))
    sample_keys = [keys[0], keys[1], keys[len(keys)//4],
                   keys[len(keys)//2], keys[-2], keys[-1]]
    for k in sample_keys[:n]:
        print(f"\n[idx={k}]\n  {by_index[k]}")


# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Generate OHLCV candlestick prompts for VCB stock"
    )
    parser.add_argument(
        "--ohlcv_path", type=str,
        default="./dataset/dataset/stock/vcb_raw_ohlcv.csv",
    )
    parser.add_argument(
        "--output_dir", type=str,
        default="./dataset/dataset/stock",
    )
    parser.add_argument("--seq_len",  type=int, default=60)
    parser.add_argument("--pred_len", type=int, default=None,
                        help="1=short-term, 60=mid-term, None=ca hai")
    args = parser.parse_args()

    pred_lens = [1, 60] if args.pred_len is None else [args.pred_len]

    for pl in pred_lens:
        suffix   = "short_term" if pl == 1 else "mid_term"
        out_path = os.path.join(args.output_dir, f"prompts_v2_ohlcv_{suffix}.json")
        prompts  = generate_prompts(
            ohlcv_path=args.ohlcv_path,
            output_path=out_path,
            seq_len=args.seq_len,
            pred_len=pl,
        )
        print_samples(prompts)
        print("─" * 70)

    print("\nDone! Prompts OHLCV chứa:")
    print("  - Candlestick patterns (Doji, Hammer, Engulfing, Soldiers...)")
    print("  - Price structure (trend, HH/HL, breakout, ATR regime)")
    print("  - Volume-price analysis (divergence, spike, accumulation/distribution)")
    print("  => Bổ sung những gì FFT KHÔNG thể học từ Close price đơn thuần.")


if __name__ == "__main__":
    main()
