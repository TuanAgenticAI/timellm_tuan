"""
Macro Context Prompt Generator for VCB Stock — V2
==================================================
Sinh dynamic prompts từ MACROECONOMIC CONTEXT, KHÔNG dùng technical indicators.
(FFT đã extract phần kỹ thuật từ price data rồi.)

Nội dung prompt:
  1. Chính sách tiền tệ SBV: lãi suất điều hành, xu hướng
  2. Bối cảnh kinh tế vĩ mô: GDP, lạm phát, regime theo giai đoạn
  3. Calendar context: quý, Tết, mùa báo cáo KQKD, ex-dividend VCB
  4. Sự kiện thị trường đặc biệt: COVID, khủng hoảng trái phiếu, v.v.

Output JSON key: date string 'YYYY-MM-DD' của ngày CUỐI window (ngày dự báo từ đó)
                 + alias index key (str(i)) để tương thích data loader hiện tại

Usage:
    python generate_macro_prompts_v2.py
    python generate_macro_prompts_v2.py --pred_len 60
    python generate_macro_prompts_v2.py --pred_len 1 --output_dir ./dataset/dataset/stock
"""

import argparse
import json
import os
from datetime import date, timedelta
from typing import Optional

import pandas as pd


# ─────────────────────────────────────────────────────────────────────────────
# 1. Macroeconomic Regime Database (2009–2026)
#    Source: SBV, GSO, IMF Vietnam Article IV reports
# ─────────────────────────────────────────────────────────────────────────────

# SBV Refinancing Rate (%) — approximate by period
SBV_RATE_PERIODS = [
    # (start_date, end_date, rate, direction, description_vn)
    (date(2009, 1, 1),  date(2009, 11, 30), 7.0,  "stable",   "SBV duy tri lai suat 7% de phuc hoi sau khung hoang tai chinh toan cau 2008"),
    (date(2009, 12, 1), date(2010, 10, 31), 8.0,  "tightening","SBV tang lai suat len 8% truoc ap luc lam phat tang tro lai"),
    (date(2010, 11, 1), date(2011, 2, 28),  9.0,  "tightening","SBV tang lai suat, lam phat vuot 11% cuoi 2010"),
    (date(2011, 3, 1),  date(2011, 10, 31), 14.0, "tightening","KHUNG HOANG LAM PHAT: CPI dat dinh 23%, SBV that chat manh nhat lich su"),
    (date(2011, 11, 1), date(2012, 5, 31),  13.0, "easing",    "SBV bat dau cat lai suat tu 15% khi lam phat hau diet"),
    (date(2012, 6, 1),  date(2012, 12, 31), 9.0,  "easing",    "SBV tiep tuc cat lai suat, ho tro tang truong, xu ly no xau ngan hang"),
    (date(2013, 1, 1),  date(2013, 12, 31), 7.0,  "easing",    "SBV cat lai suat ve 7%, kinh te phuc hoi cham, xu ly no xau ngan hang"),
    (date(2014, 1, 1),  date(2015, 9, 30),  6.5,  "stable",    "Lai suat on dinh 6.5%, lam phat thap, NHNN mua ngoai te tang du tru"),
    (date(2015, 10, 1), date(2016, 6, 30),  6.5,  "stable",    "Canh tranh tu pha gia CNY thang 8/2015, SBV dieu chinh bien do ty gia +/-3%"),
    (date(2016, 7, 1),  date(2017, 6, 30),  6.25, "easing",    "SBV cat nhe 25bps, chuyen sang co che ty gia trung tam tu 1/2016"),
    (date(2017, 7, 1),  date(2019, 8, 31),  6.25, "stable",    "Lai suat on dinh, kinh te tang truong manh 6.8-7.1%, Basel II trien khai"),
    (date(2019, 9, 1),  date(2020, 2, 29),  6.0,  "easing",    "SBV cat 25bps truoc rui ro chien tranh thuong mai My-Trung, ho tro tang truong"),
    (date(2020, 3, 1),  date(2020, 9, 30),  4.5,  "easing",    "COVID-19 BOC PHAT: SBV cat 2 lan, tong -150bps, goi ho tro tin dung 300k ty VND"),
    (date(2020, 10, 1), date(2021, 9, 30),  4.0,  "stable",    "Lai suat lich su thap 4%, thi truong chung khoan bung no du COVID Delta"),
    (date(2021, 10, 1), date(2022, 8, 31),  4.0,  "stable",    "Phuc hoi post-COVID, tin dung tang manh, FED bat dau tang lai suat"),
    (date(2022, 9, 1),  date(2022, 12, 31), 6.0,  "tightening","SBV TANG MANH 200bps: FED tightening lan toa, KHUNG HOANG TRAI PHIEU (FLC, Van Thinh Phat)"),
    (date(2023, 1, 1),  date(2023, 5, 31),  5.5,  "tightening","Khung hoang tin dung bat dong san, Van Thinh Phat, SCBV - ngan hang that chat"),
    (date(2023, 6, 1),  date(2023, 12, 31), 4.5,  "easing",    "SBV dao chieu cat lai suat 4 lan, ho tro kinh te tang truong cham 5%"),
    (date(2024, 1, 1),  date(2024, 12, 31), 4.5,  "stable",    "Lai suat on dinh 4.5%, kinh te phuc hoi 7.1%, xuat khau tang manh"),
    (date(2025, 1, 1),  date(2025, 12, 31), 4.5,  "stable",    "On dinh chinh sach, tang truong GDP du kien 6.5-7%, dau tu nuoc ngoai tang"),
    (date(2026, 1, 1),  date(2026, 12, 31), 4.5,  "stable",    "Chinh sach tien te on dinh, tiep tuc ho tro tang truong"),
]

# GDP Growth Rate (%) by year
GDP_BY_YEAR = {
    2009: (5.3,  "phuc hoi post-GFC"),
    2010: (6.8,  "phuc hoi manh"),
    2011: (6.2,  "tang truong duoi ap luc lam phat cao"),
    2012: (5.2,  "cham lai do that chat tin dung"),
    2013: (5.4,  "phuc hoi cham, xu ly no xau"),
    2014: (5.98, "on dinh"),
    2015: (6.7,  "tang truong kha"),
    2016: (6.2,  "on dinh"),
    2017: (6.8,  "tang truong tot"),
    2018: (7.1,  "tang truong cao nhat thap ky"),
    2019: (7.0,  "tang truong manh truoc COVID"),
    2020: (2.9,  "thap nhat 30 nam do COVID-19"),
    2021: (2.6,  "COVID Delta lam cham tang truong"),
    2022: (8.0,  "bung no post-COVID"),
    2023: (5.0,  "cham lai do khung hoang tin dung"),
    2024: (7.1,  "phuc hoi manh"),
    2025: (6.5,  "du kien on dinh"),
    2026: (6.8,  "du kien tang truong tich cuc"),
}

# Key market/macro events by period
MAJOR_EVENTS = [
    # (start, end, event_text)
    (date(2009, 7, 1),  date(2009, 12, 31), "Goi kich thich 145k ty VND cua Chinh phu thuc day phuc hoi; FDI phuc hoi"),
    (date(2010, 1, 1),  date(2010, 12, 31), "Goi kich cau het hieu luc, lam phat bat dau tang; NHNN tang du tru ngoai hoi"),
    (date(2011, 1, 1),  date(2011, 12, 31), "KHUNG HOANG LAM PHAT VIET NAM: CPI dinh 23.02%, ty gia USD/VND bien dong manh +9%"),
    (date(2012, 1, 1),  date(2013, 12, 31), "De an xu ly no xau 02/2013, VAMC thanh lap 7/2013; ngan hang yeu kem xu ly"),
    (date(2014, 1, 1),  date(2014, 12, 31), "Lam phat thap 1.84%, FED bat dau tapering, VNI on dinh"),
    (date(2015, 1, 1),  date(2015, 12, 31), "Trung Quoc pha gia CNY 8/2015, SBV no rang bien do ty gia; gia dau giam manh"),
    (date(2016, 1, 1),  date(2016, 12, 31), "Co che ty gia trung tam, TPP dam phan ket thuc; kinh te phuc hoi"),
    (date(2017, 1, 1),  date(2018, 12, 31), "Kinh te tang truong cao, xuat khau Samsung, thu hut FDI ky luc; thi truong chung khoan tang manh"),
    (date(2019, 1, 1),  date(2019, 12, 31), "Chien tranh thuong mai My-Trung: Viet Nam huong loi do dich chuyen chuoi cung ung"),
    (date(2020, 1, 23), date(2020, 6, 30),  "COVID-19 DAI DICH: Viet Nam phong toa, goi ho tro 62k ty; VNI giam 30%+ tu dinh"),
    (date(2020, 7, 1),  date(2020, 12, 31), "Viet Nam kiem soat COVID tot nhat the gioi; chung khoan phuc hoi manh cuoi nam"),
    (date(2021, 1, 1),  date(2021, 7, 31),  "Tiem vaccine bat dau, thi truong tang manh du COVID Delta xuat hien"),
    (date(2021, 8, 1),  date(2021, 10, 31), "COVID DELTA: TPHCM phong toa 4 thang, GDP Q3 am 6.17%, FDI giam"),
    (date(2021, 11, 1), date(2021, 12, 31), "Mo cua tro lai, phuc hoi kinh te, VNI dat dinh lich su >1,500 diem"),
    (date(2022, 1, 1),  date(2022, 3, 31),  "FLC Group scandal: Trinh Van Quyet bi bat 3/2022; rui ro nha dau tu to chuc"),
    (date(2022, 4, 1),  date(2022, 7, 31),  "Tan Hoang Minh huy 9 lo trai phieu 10k ty; UBCKNN siet phat hanh trai phieu DN"),
    (date(2022, 8, 1),  date(2022, 12, 31), "Van Thinh Phat - SCB scandal: Truong My Lan bi bat 10/2022; khung hoang niem tin ngan hang"),
    (date(2023, 1, 1),  date(2023, 6, 30),  "Thi truong trai phieu DN dong bang, bat dong san kho khan, tin dung tang thap ky luc"),
    (date(2023, 7, 1),  date(2023, 12, 31), "SBV cat lai suat 4 lan, thuc day tin dung, chua ky ban lai trai phieu DN"),
    (date(2024, 1, 1),  date(2024, 6, 30),  "Kinh te phuc hoi, xuat khau tang 15%+, FDI giai ngan ky luc; VNI huong toi 1,300+"),
    (date(2024, 7, 1),  date(2024, 12, 31), "GDP 2024 dat 7.1%, Viet Nam vao danh sach theo doi pha gia tien te cua My"),
    (date(2025, 1, 1),  date(2025, 12, 31), "Chuong trinh de an 1 trieu ty tang truong; von hoa TTCK muc tieu 100% GDP"),
    (date(2026, 1, 1),  date(2026, 12, 31), "Thi truong tiep tuc on dinh; kiem soat lam phat trong muc tieu 4%"),
]

# VCB-specific events
VCB_EVENTS = [
    (date(2011, 9, 30), date(2011, 9, 30), "Mizuho mua 15% co phan VCB voi gia 567.3 ty yen (~70 trieu USD), doi tac chien luoc Nhat Ban"),
    (date(2012, 1, 1),  date(2014, 12, 31), "VCB trien khai xu ly no xau, trich lap du phong lon; Basel II pilot"),
    (date(2018, 1, 1),  date(2018, 12, 31), "VCB trien khai Basel II, tang von chu so huu, loi nhuan tang truong 60%+"),
    (date(2019, 6, 1),  date(2019, 9, 30),  "VCB phat hanh co phieu rieng le cho GIC (Singapore) va Mizuho, tang von manh"),
    (date(2021, 1, 1),  date(2021, 12, 31), "VCB vuot muc von hoa 500k ty VND, lot top 3 ngan hang lon nhat Viet Nam"),
    (date(2022, 1, 1),  date(2022, 12, 31), "VCB co tuc bang co phieu 18.1%, tang von dieu le len 47k ty VND"),
    (date(2023, 1, 1),  date(2023, 12, 31), "VCB tang co tuc 38.8% (ket hop 26.5% tien mat + 12.284% co phieu)"),
    (date(2024, 1, 1),  date(2024, 12, 31), "VCB duoc vinh danh ngan hang tot nhat Viet Nam nhieu nam lien, ROE~25%"),
]

# Tet (Lunar New Year) dates — first day of Tet holiday
TET_DATES = {
    2009: date(2009, 1, 26),
    2010: date(2010, 2, 14),
    2011: date(2011, 2, 3),
    2012: date(2012, 1, 23),
    2013: date(2013, 2, 10),
    2014: date(2014, 1, 31),
    2015: date(2015, 2, 19),
    2016: date(2016, 2, 8),
    2017: date(2017, 1, 28),
    2018: date(2018, 2, 16),
    2019: date(2019, 2, 5),
    2020: date(2020, 1, 25),
    2021: date(2021, 2, 12),
    2022: date(2022, 2, 1),
    2023: date(2023, 1, 22),
    2024: date(2024, 2, 10),
    2025: date(2025, 1, 29),
    2026: date(2026, 2, 17),
}


# ─────────────────────────────────────────────────────────────────────────────
# 2. Lookup helpers
# ─────────────────────────────────────────────────────────────────────────────

def get_sbv_context(d: date) -> dict:
    """Tra cuu chinh sach SBV tai ngay d."""
    for start, end, rate, direction, desc in SBV_RATE_PERIODS:
        if start <= d <= end:
            return {"rate": rate, "direction": direction, "desc": desc}
    return {"rate": 4.5, "direction": "stable", "desc": "Chinh sach tien te on dinh"}


def get_gdp_context(d: date) -> dict:
    """Lay boi canh GDP theo nam."""
    year = d.year
    if year in GDP_BY_YEAR:
        gdp, label = GDP_BY_YEAR[year]
        return {"gdp": gdp, "label": label}
    return {"gdp": 6.0, "label": "tang truong on dinh"}


def get_major_event(d: date) -> Optional[str]:
    """Tim su kien lon nhat tai/gan ngay d."""
    for start, end, event in MAJOR_EVENTS:
        if start <= d <= end:
            return event
    return None


def get_vcb_event(d: date) -> Optional[str]:
    """Tim su kien VCB-specific gan ngay d."""
    for start, end, event in VCB_EVENTS:
        if start <= d <= end:
            return event
    return None


def get_quarter_context(d: date) -> dict:
    """Xac dinh quy va giai doan bao cao KQKD."""
    m = d.month
    quarter = (m - 1) // 3 + 1

    # Earnings season context
    # VCB Q4 results: Feb, Q1 results: Apr-May, Q2 results: Jul-Aug, Q3 results: Oct-Nov
    if m == 2:
        earnings = "mua bao cao KQKD quy 4 nam truoc: nha dau tu cho doi ket qua ca nam"
    elif m in (4, 5):
        earnings = "mua bao cao KQKD Q1: danh gia da da tang truong dau nam"
    elif m in (7, 8):
        earnings = "mua bao cao KQKD ban nien (Q2): kiet dinh thi truong nua dau nam"
    elif m in (10, 11):
        earnings = "mua bao cao KQKD Q3: du bao ca nam bat dau hinh thanh"
    else:
        earnings = f"quy {quarter}, ngoai mua bao cao chinh"

    return {"quarter": quarter, "earnings": earnings}


def get_tet_context(d: date) -> Optional[str]:
    """Xac dinh proximity to Tet, returns context string or None."""
    year = d.year
    # Check current year and next year's Tet
    for y in [year - 1, year, year + 1]:
        tet = TET_DATES.get(y)
        if tet is None:
            continue
        delta = (tet - d).days
        if 0 < delta <= 30:
            return f"truoc Tet {delta} ngay: tam ly nha dau tu thuong tich cuc, thanh khoan giam dan cuoi nam"
        elif -7 <= delta <= 0:
            return f"nghi Tet (hoac vua qua Tet): thi truong dong cua/vua mo lai, gia thuong bien dong khi tro lai"
        elif -25 <= delta <= -8:
            return f"vua qua Tet {-delta} ngay: thong thuong co dot ban chot loi, thi truong on dinh dan"
    return None


def get_dividend_season(d: date) -> Optional[str]:
    """VCB thường có ex-dividend tháng 6-9."""
    m = d.month
    if m in (5, 6):
        return "co the gan ngay chot quyen co tuc VCB (thuong thang 6-7): chu y co the bi dieu chinh gia khi ex-div"
    elif m in (7, 8):
        return "mua chi tra co tuc/co phieu thuong cua VCB: nha dau tu theo doi muc chi tra nam nay"
    return None


def get_year_end_context(d: date) -> Optional[str]:
    """Window dressing / year-end effects."""
    m = d.month
    if m == 12:
        return "thang 12: window dressing cuoi nam, quy dau tu thong thuong nang do co phieu ngan hang blue-chip"
    elif m == 1 and d.day <= 15:
        return "dau thang 1: hieu ung January effect, dong tien moi vao thi truong sau ky nghi Tet"
    return None


# ─────────────────────────────────────────────────────────────────────────────
# 3. Prompt builder
# ─────────────────────────────────────────────────────────────────────────────

def build_macro_prompt(end_date: date, pred_len: int = 1) -> str:
    """
    Xay dung prompt macro cho window ket thuc tai end_date.
    end_date: ngay cuoi cung cua chuoi input (60 ngay).
    pred_len: 1 (short-term) hoac 60 (mid-term).
    """
    sbv   = get_sbv_context(end_date)
    gdp   = get_gdp_context(end_date)
    qtr   = get_quarter_context(end_date)
    event = get_major_event(end_date)
    vcb_e = get_vcb_event(end_date)
    tet   = get_tet_context(end_date)
    div   = get_dividend_season(end_date)
    ye    = get_year_end_context(end_date)

    horizon = "ngay tiep theo" if pred_len == 1 else f"{pred_len} ngay tiep theo"

    # ── Monetary policy line ──────────────────────────────────────────────
    dir_map = {"tightening": "dang that chat", "easing": "dang no long", "stable": "on dinh"}
    sbv_line = (
        f"Chinh sach tien te: lai suat SBV {sbv['rate']}% ({dir_map.get(sbv['direction'], 'on dinh')}). "
        f"{sbv['desc']}."
    )

    # ── Macro context line ────────────────────────────────────────────────
    macro_line = (
        f"Boi canh kinh te: GDP Viet Nam {gdp['gdp']}% ({gdp['label']}), "
        f"{qtr['earnings']}."
    )

    # ── Calendar signals ──────────────────────────────────────────────────
    calendar_parts = []
    if tet:
        calendar_parts.append(tet)
    if div:
        calendar_parts.append(div)
    if ye:
        calendar_parts.append(ye)
    calendar_line = ("Tin hieu lich: " + " | ".join(calendar_parts) + ".") if calendar_parts else ""

    # ── Event line ────────────────────────────────────────────────────────
    event_parts = []
    if event:
        event_parts.append(event)
    if vcb_e:
        event_parts.append(f"[VCB] {vcb_e}")
    event_line = ("Su kien: " + " | ".join(event_parts) + ".") if event_parts else ""

    # ── Assemble ──────────────────────────────────────────────────────────
    parts = [sbv_line, macro_line]
    if calendar_line:
        parts.append(calendar_line)
    if event_line:
        parts.append(event_line)
    parts.append(
        f"Nhiem vu: du bao gia VCB trong {horizon} dua tren 60 ngay giao dich truoc."
    )

    return " ".join(parts)


# ─────────────────────────────────────────────────────────────────────────────
# 4. Main generation loop
# ─────────────────────────────────────────────────────────────────────────────

def generate_prompts(data_path: str, output_path: str,
                     seq_len: int = 60, pred_len: int = 1) -> dict:
    """
    Sinh prompts cho toan bo dataset.

    Returns dict with 2 key namespaces:
      - by_date  : { 'YYYY-MM-DD': prompt_str }   ← date of last day in window
      - by_index : { '0': prompt_str, '1': ... }  ← backward compat with data loader
    """
    df = pd.read_csv(data_path, parse_dates=["date"])
    dates = df["date"].dt.date.tolist()
    total = len(dates) - seq_len - pred_len + 1

    print(f"Dataset: {data_path}")
    print(f"  Rows : {len(dates)}  ({dates[0]} → {dates[-1]})")
    print(f"  Windows: {total}  (seq_len={seq_len}, pred_len={pred_len})")
    print(f"  Output : {output_path}")

    by_date  = {}
    by_index = {}

    for i in range(total):
        end_date = dates[i + seq_len - 1]   # last day of input window
        prompt   = build_macro_prompt(end_date, pred_len=pred_len)
        by_date[str(end_date)]  = prompt
        by_index[str(i)]        = prompt

        if (i + 1) % 500 == 0 or i == total - 1:
            print(f"  [{i+1}/{total}] {end_date}  len={len(prompt)}")

    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(by_index, f, ensure_ascii=False, indent=2)

    print(f"✓ Saved {len(by_index)} prompts → {output_path}\n")
    return {"by_date": by_date, "by_index": by_index}


def print_sample(prompts: dict, n: int = 5):
    """In mau de kiem tra."""
    by_index = prompts.get("by_index", prompts)
    keys = sorted(by_index.keys(), key=lambda x: int(x) if x.isdigit() else 0)
    sample_keys = keys[:2] + [keys[len(keys)//2]] + keys[-2:]
    for k in sample_keys[:n]:
        print(f"\n  [idx={k}] {by_index[k][:200]}...")


# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Generate macro-context prompts for VCB stock (V2 dataset)"
    )
    parser.add_argument(
        "--data_path", type=str,
        default="./dataset/dataset/stock/vcb_stock_indicators_v2.csv",
    )
    parser.add_argument(
        "--output_dir", type=str,
        default="./dataset/dataset/stock",
    )
    parser.add_argument("--seq_len",  type=int, default=60)
    parser.add_argument("--pred_len", type=int, default=None,
                        help="1 = short-term, 60 = mid-term, None = generate both")
    args = parser.parse_args()

    pred_lens = [1, 60] if args.pred_len is None else [args.pred_len]

    for pl in pred_lens:
        suffix = "short_term" if pl == 1 else "mid_term"
        out_path = os.path.join(args.output_dir, f"prompts_v2_macro_{suffix}.json")
        prompts = generate_prompts(
            data_path=args.data_path,
            output_path=out_path,
            seq_len=args.seq_len,
            pred_len=pl,
        )
        print_sample(prompts)
        print("─" * 60)

    print("\nDone! Prompts chứa macro context (lãi suất SBV, GDP,")
    print("Tết, earnings season, events) — KHÔNG duplicate technical indicators.")


if __name__ == "__main__":
    main()
