import os
import pandas as pd
import yfinance as yf

def fetch_data(ticker: str="^NSEI", start_date: str= "2015-08-31", end_date: str= "2026-08-31", save_raw: bool= True) -> pd.DataFrame:

    clean_ticker = ticker.replace("^", "").replace(".", "_")
    filename = f"{clean_ticker}_{start_date}_to_{end_date}.csv"
    cached_path = os.path.join("data", "raw", filename)

    # If already downloaded, load from disk directly!
    if os.path.exists(cached_path):
        print(f"--> Found cached dataset at: {cached_path}. Loading from disk...")
        return pd.read_csv(cached_path, index_col=0, parse_dates=True)
    
    '''For loading stock market data from Yahoo Finance and saving it as a CSV after cleaning he columns'''
    print (f"--> Fetching data for ticker: '{ticker}' from {start_date} to {end_date}...")
    
    #Download Data
    df = yf.download(ticker, start=start_date, end=end_date)

    #For successful collction of data
    if df.empty:
        raise ValueError(f"No data returned for ticker '{ticker}'. Please check the symbol or your internet connection.")

    # Clean MultiIndex Columns
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = [col[0] for col in df.columns]

    # Standardize column names (remove any accidental whitspace)
    df.columns = [str(col).strip() for col in df.columns]

    # Ensure 'Adj Close' exists (fallback to 'Close' if missing)
    if 'Adj Close' not in df.columns and 'Close' in df.columns:
        df['Adj Close'] = df['Close']

    df = df.dropna()

    if save_raw:
        os.makedirs("data/raw", exist_ok=True)

        file_path = os.path.join("data", "raw", filename)

        df.to_csv(file_path)
        print(f"--> Saved raw data to: {file_path}")

    return df

if __name__ == "__main__":
    data = fetch_data(ticker = "^NSEI", start_date="2015-01-01", end_date="2026-08-01")

    print("\n--- Download Success! Preview of the dataset ---")
    print(data.head())
    print("\nDataset SHape (Rows, Columns):", data.shape)