import yfinance as yf
import pandas as pd
import os

def download_data(tickers, start, end, save_to='data'):
    os.makedirs(save_to, exist_ok=True)
    for ticker in tickers:
        print(f"Downloading {ticker}...")
        df = yf.download(ticker, start=start, end=end)
        df.to_csv(f"{save_to}/{ticker}.csv")
        print(f"Saved {ticker} to {save_to}/{ticker}.csv")

if __name__ == "__main__":
    tickers = ['AAPL', 'MSFT', 'SPY', 'AMZN', 'GOOG']
    start = '2010-01-01'
    end = '2020-12-31'
    download_data(tickers, start, end)
