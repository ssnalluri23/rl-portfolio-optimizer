import pandas as pd
import os
import pandas_ta as ta

TICKERS = ['AAPL', 'MSFT', 'SPY', 'AMZN', 'GOOG']
DATA_DIR = 'data'
OUTPUT_FILE = f'{DATA_DIR}/processed_data.csv'

def load_and_merge_close_prices():
    dfs = []
    for ticker in TICKERS:
        df = pd.read_csv(f"{DATA_DIR}/{ticker}.csv", index_col=0, parse_dates=True, dayfirst=False)
        df.index.name = 'Date'  # Optional: renames index for clarity
        df = df[['Close']].rename(columns={'Close': ticker})
        dfs.append(df)
    merged = pd.concat(dfs, axis=1)
    return merged.dropna()

def add_indicators(df):
    result = df.copy()
    for ticker in TICKERS:
        price = pd.to_numeric(df[ticker], errors='coerce')  # 🔥 Fix: ensure float dtype
        result[f'{ticker}_rsi'] = ta.rsi(price, length=14)
        macd = ta.macd(price)
        result[f'{ticker}_macd'] = macd['MACD_12_26_9']
        result[f'{ticker}_signal'] = macd['MACDs_12_26_9']
        result[f'{ticker}_sma'] = ta.sma(price, length=20)
    return result.dropna()


def normalize(df):
    numeric_df = df.select_dtypes(include=['number'])  # Only numeric columns
    return (numeric_df - numeric_df.mean()) / numeric_df.std()

if __name__ == "__main__":
    print("📥 Merging close prices...")
    close_df = load_and_merge_close_prices()

    print("🧮 Adding indicators...")
    full_df = add_indicators(close_df)

    print("📏 Normalizing...")
    normalized_df = normalize(full_df)

    print(f"💾 Saving to {OUTPUT_FILE}")
    normalized_df.to_csv(OUTPUT_FILE)
    print("✅ Done.")
