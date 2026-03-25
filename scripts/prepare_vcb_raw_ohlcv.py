import argparse
from pathlib import Path
import pandas as pd


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input",
        default="gia_VCB_2009-2026.csv",
        help="Input raw OHLCV csv (expects 'Date' column).",
    )
    parser.add_argument(
        "--output",
        default="dataset/dataset/stock/vcb_raw_ohlcv.csv",
        help="Output csv (will use 'date' column).",
    )
    args = parser.parse_args()

    in_path = Path(args.input)
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(in_path)

    if "Date" not in df.columns:
        raise ValueError(f"Missing 'Date' column in {in_path}. Columns: {list(df.columns)}")

    df = df.rename(columns={"Date": "date"})

    # Normalize date to ISO format and sort ascending (oldest -> newest)
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values("date").reset_index(drop=True)
    df["date"] = df["date"].dt.strftime("%Y-%m-%d")

    expected = ["date", "Open", "High", "Low", "Close", "Adj Close", "Volume"]
    missing = [c for c in expected if c not in df.columns]
    if missing:
        raise ValueError(f"Missing columns {missing} in {in_path}. Columns: {list(df.columns)}")

    df = df[expected]
    df.to_csv(out_path, index=False)

    print(f"Wrote {len(df)} rows to {out_path}")


if __name__ == "__main__":
    main()

