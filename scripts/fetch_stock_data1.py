import os
import yfinance as yf

def main(symbol_file, output_dir):
    # Read stock symbols
    with open(symbol_file, 'r') as f:
        symbols = [line.strip() for line in f.readlines() if line.strip()]

    os.makedirs(output_dir, exist_ok=True)

    # Fetch data from Yahoo Finance
    for ticker in symbols:
        print(f"Fetching {ticker} data...")
        data = yf.download(ticker, period="1y")
        csv_path = os.path.join(output_dir, f"{ticker}.csv")
        data.to_csv(csv_path)
        print(f"✅ Saved {csv_path}")

if __name__ == "__main__":
    import sys
    if len(sys.argv) < 3:
        print("Usage: python scripts/fetch_stock_data.py input/symbols.txt output/")
    else:
        main(sys.argv[1], sys.argv[2])
