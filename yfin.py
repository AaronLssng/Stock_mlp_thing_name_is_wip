

import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import warnings
import requests
import time
import os
import stock_data

warnings.filterwarnings('ignore')


#  CONFIGURATION 

MAX_RETRIES = 3
BATCH_SIZE = 50
VERBOSE = True


def log(message):
    """Print message if verbose mode is enabled."""
    if VERBOSE:
        print(message)


#  TICKER FUNCTIONS 

def get_tickers_from_wikipedia(index_type='sp500'):
    """
    Fetch tickers from Wikipedia.
    
    Parameters:
    - index_type: 'sp500', 'dow', or 'nasdaq100'
    """
    try:
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
        }
        
        index_map = {
            'sp500': ('https://en.wikipedia.org/wiki/List_of_S%26P_500_companies', 0, 'Symbol'),
            'dow': ('https://en.wikipedia.org/wiki/Dow_Jones_Industrial_Average', 0, 'Symbol'),
            'nasdaq100': ('https://en.wikipedia.org/wiki/Nasdaq-100', 0, 'Ticker'),
        }
        
        if index_type not in index_map:
            log(f"Unknown index type: {index_type}")
            return None
        
        url, table_idx, symbol_col = index_map[index_type]
        response = requests.get(url, headers=headers, timeout=10)
        
        if response.status_code == 200:
            tables = pd.read_html(response.text)
            tickers = tables[table_idx][symbol_col].tolist()
            log(f"Fetched {len(tickers)} tickers from Wikipedia")
            return tickers
        else:
            log(f"Wikipedia returned status {response.status_code}")
            return None
            
    except Exception as e:
        log(f"Error fetching from Wikipedia: {e}")
        return None


def get_hardcoded_tickers(index_type='sp500'):
    """Return hardcoded ticker lists as fallback."""
    hardcoded = {
        'sp500': [
            'AAPL', 'MSFT', 'AMZN', 'NVDA', 'META', 'GOOGL', 'BRK-B', 'LLY',
            'JPM', 'UNH', 'XOM', 'V', 'PG', 'JNJ', 'MA', 'HD', 'CVX', 'MRK', 'ABBV',
            'KO', 'PEP', 'COST', 'TMO', 'MCD', 'DIS', 'CSCO', 'ABT', 'WMT', 'ACN',
            'NKE', 'NEE', 'NFLX', 'TXN', 'GS', 'VZ', 'PM', 'PFE', 'COP', 'ORCL',
            'HON', 'ADBE', 'QCOM', 'MS', 'UPS', 'CAT', 'IBM', 'DHR', 'UNP',
            'LMT', 'INTU', 'T', 'GE', 'BMY', 'AMGN', 'PLD', 'CVS', 'C', 'MDT',
            'AMAT', 'CMCSA', 'RTX', 'TGT', 'GSK', 'SPGI', 'BLK', 'GILD', 'SYK',
            'SCHW', 'BA', 'DE', 'VRTX', 'NOW', 'MO', 'MMC', 'DUK', 'SO', 'INTC',
            'SAP', 'HUM', 'PNC', 'SLB', 'TJX', 'CL', 'ZTS', 'AEP', 'FIS', 'CI',
            'TFC', 'SHW', 'USB', 'AIG', 'PYPL', 'ELV', 'MDLZ', 'CHTR', 'CB', 'ADI'
        ],
        'dow': [
            'AAPL', 'AMGN', 'AXP', 'BA', 'CAT', 'CRM', 'CSCO', 'CVX', 'DIS',
            'DOW', 'GS', 'HD', 'HON', 'IBM', 'INTC', 'JNJ', 'JPM', 'KO',
            'MCD', 'MMM', 'MRK', 'MSFT', 'NKE', 'PG', 'TRV', 'UNH', 'VZ',
            'WMT', 'WBA', 'NEE'
        ],
        'nasdaq100': [
            'AAPL', 'MSFT', 'AMZN', 'NVDA', 'META', 'GOOGL', 'GOOG', 'AVGO',
            'PEP', 'COST', 'CSCO', 'ADBE', 'TXN', 'QCOM', 'AMGN', 'INTU',
            'AMD', 'ISRG', 'CMCSA', 'HON', 'NFLX', 'INTC', 'SBUX', 'MDLZ',
            'GILD', 'BKNG', 'ADI', 'VRTX', 'REGN', 'LRCX', 'MRNA', 'PANW',
            'KLAC', 'SNPS', 'CDNS', 'MELI', 'FTNT', 'ADP', 'ASML', 'KDP'
        ]
    }
    return hardcoded.get(index_type, hardcoded['sp500'])


def get_tickers(index_type='sp500', use_fallback=True):
    
    tickers = get_tickers_from_wikipedia(index_type)
    
    if tickers is None and use_fallback:
        log(f"Using fallback hardcoded list for {index_type}...")
        tickers = get_hardcoded_tickers(index_type)
    
    if tickers:
        tickers = clean_tickers(tickers)
    
    return tickers


def clean_tickers(tickers):
    
    ticker_map = {
        'BRK.B': 'BRK-B',
        'BF.B': 'BF-B',
        'BRK-B': 'BRK-B',
    }
    return [ticker_map.get(t, t) for t in tickers]


#  DOWNLOAD FUNCTIONS 

def download_with_retry(symbols, start_date, end_date):
    """Download data with retry logic."""
    for attempt in range(MAX_RETRIES):
        try:
            return yf.download(
                symbols,
                start=start_date,
                end=end_date,
                auto_adjust=True,
                progress=False,
                threads=True
            )
        except Exception as e:
            if attempt < MAX_RETRIES - 1:
                log(f"    Retry {attempt+1}/{MAX_RETRIES}...")
                time.sleep(2)
            else:
                log(f"    ✗ Failed after {MAX_RETRIES} attempts: {e}")
                return None
    return None


def fetch_index_data(index_symbol='^GSPC', start_date=None, end_date=None, years=10):
    
    if end_date is None:
        end_date = datetime.now()
    if start_date is None:
        start_date = end_date - timedelta(days=years*365)
    
    log(f"Fetching {index_symbol} from {start_date.strftime('%Y-%m-%d')} to {end_date.strftime('%Y-%m-%d')}")
    
    try:
        data = yf.download(
            index_symbol,
            start=start_date,
            end=end_date,
            auto_adjust=True,
            progress=False
        )
        log(f"Downloaded {len(data)} days of {index_symbol} data")
        return data
    except Exception as e:
        log(f"Error downloading {index_symbol}: {e}")
        return None


def fetch_stock_data(symbols, start_date=None, end_date=None, years=10):
   
    if end_date is None:
        end_date = datetime.now()
    if start_date is None:
        start_date = end_date - timedelta(days=years*365)
    
    log(f"Fetching data for {len(symbols)} stocks from {start_date.strftime('%Y-%m-%d')} to {end_date.strftime('%Y-%m-%d')}")
    
    all_close_prices = []
    all_volumes = []
    failed_tickers = []
    
    for i in range(0, len(symbols), BATCH_SIZE):
        batch = symbols[i:i+BATCH_SIZE]
        log(f"  Batch {i//BATCH_SIZE + 1}/{(len(symbols)//BATCH_SIZE)+1}...")
        
        batch_data = download_with_retry(batch, start_date, end_date)
        
        if batch_data is None:
            failed_tickers.extend(batch)
            continue
        
        # Extract close prices and volumes
        if hasattr(batch_data.columns, 'levels'):
            # Multi-index case
            close_prices = batch_data['Close'] if 'Close' in batch_data.columns.levels[0] else batch_data
            volumes = batch_data['Volume'] if 'Volume' in batch_data.columns.levels[0] else None
        else:
            close_prices = batch_data
            volumes = None
        
        all_close_prices.append(close_prices)
        if volumes is not None:
            all_volumes.append(volumes)
    
    if not all_close_prices:
        log("No data downloaded")
        return None, None
    
    close_prices = pd.concat(all_close_prices, axis=1)
    volumes = pd.concat(all_volumes, axis=1) if all_volumes else None
    
    log(f"Downloaded data for {len(close_prices.columns)} stocks")
    if failed_tickers:
        log(f"Failed to download: {len(failed_tickers)} tickers")
    
    return close_prices, volumes


def fetch_individual_stock(symbol, years=10, timeframe='1d'):
    
    end_date = datetime.now()
    start_date = end_date - timedelta(days=years*365)
    
    try:
        data = yf.download(
            symbol,
            start=start_date,
            end=end_date,
            interval=timeframe,
            auto_adjust=True,
            progress=False
        )
        return data
    except Exception as e:
        log(f"Error fetching {symbol}: {e}")
        return pd.DataFrame()


def fetch_multiple_stocks(symbols, years=10, timeframe='1d'):
    
    result = {}
    for symbol in symbols:
        data = fetch_individual_stock(symbol, years, timeframe)
        if not data.empty:
            result[symbol] = data
            log(f"✓ {symbol}: {len(data)} bars")
        else:
            log(f"✗ {symbol}: No data")
    return result


#  FEATURE ENGINEERING 

def add_features(df, price_column='Close', volume_column='Volume'):
    
    df = df.copy()
    
    # 1. Returns
    df['Return_Pct'] = df[price_column].pct_change() * 100
    
    # 2. Lagged returns
    for lag in [1, 2, 3, 5, 10, 20]:
        df[f'Return_Lag_{lag}'] = df['Return_Pct'].shift(lag)
    
    # 3. Rolling statistics
    for window in [5, 10, 20]:
        df[f'Volatility_{window}d'] = df['Return_Pct'].rolling(window).std()
        if volume_column in df.columns:
            df[f'Volume_MA_{window}'] = df[volume_column].rolling(window).mean()
    
    # 4. Price ratios
    if 'High' in df.columns and 'Low' in df.columns:
        df['High_Low_Ratio'] = (df['High'] / df['Low'] - 1) * 100
    if 'Close' in df.columns and 'Open' in df.columns:
        df['Close_Open_Ratio'] = (df['Close'] / df['Open'] - 1) * 100
    
    # 5. Moving average ratios
    for ma in [20, 50, 200]:
        if len(df) >= ma:
            df[f'MA_{ma}'] = df[price_column].rolling(ma).mean()
            df[f'Price_to_MA_{ma}'] = df[price_column] / df[f'MA_{ma}'] - 1
    
    # 6. ATR (Average True Range)
    if all(col in df.columns for col in ['High', 'Low']):
        df['True_Range'] = np.maximum(
            df['High'] - df['Low'],
            np.maximum(
                abs(df['High'] - df[price_column].shift(1)),
                abs(df['Low'] - df[price_column].shift(1))
            )
        )
        df['ATR_14'] = df['True_Range'].rolling(14).mean()
    
    return df


def add_cyclical_features(df, date_column='Date'):
    """Add cyclical time features (day of week, month, etc.)."""
    df = df.copy()
    
    if date_column in df.columns:
        dates = pd.to_datetime(df[date_column])
        df['Year'] = dates.dt.year
        df['Month'] = dates.dt.month
        df['Day'] = dates.dt.day
        df['DayOfWeek'] = dates.dt.dayofweek
        df['DayOfYear'] = dates.dt.dayofyear
        df['WeekOfYear'] = dates.dt.isocalendar().week
        
        # Cyclical encoding
        df['Month_Sin'] = np.sin(2 * np.pi * df['Month'] / 12)
        df['Month_Cos'] = np.cos(2 * np.pi * df['Month'] / 12)
        df['DayOfWeek_Sin'] = np.sin(2 * np.pi * df['DayOfWeek'] / 7)
        df['DayOfWeek_Cos'] = np.cos(2 * np.pi * df['DayOfWeek'] / 7)
    
    return df


#  SUMMARY FUNCTIONS 

def create_summary(close_prices, volumes=None, index_data=None, index_name='Index'):
   
    summary = []
    
    for idx, date in enumerate(close_prices.index):
        row = {'Date': date.strftime('%Y-%m-%d') if hasattr(date, 'strftime') else str(date)[:10]}
        
        # Add index data if provided
        if index_data is not None and date in index_data.index:
            for col in ['Open', 'High', 'Low', 'Close', 'Volume']:
                if col in index_data.columns:
                    row[f'{index_name}_{col}'] = float(index_data.loc[date, col])
        
        # Add stock statistics
        try:
            daily_prices = close_prices.loc[date]
            if isinstance(daily_prices, pd.Series):
                clean_prices = daily_prices.dropna()
                if len(clean_prices) > 0:
                    row['Stocks_Count'] = int(len(clean_prices))
                    row['Stocks_Mean'] = float(clean_prices.mean())
                    row['Stocks_Median'] = float(clean_prices.median())
                    row['Stocks_Std'] = float(clean_prices.std())
                    row['Stocks_Min'] = float(clean_prices.min())
                    row['Stocks_Max'] = float(clean_prices.max())
                    row['Stocks_Sum'] = float(clean_prices.sum())
                    
                    # Calculate returns
                    if idx > 0:
                        prev_prices = close_prices.iloc[idx-1].dropna()
                        common = clean_prices.index.intersection(prev_prices.index)
                        if len(common) > 0:
                            returns = (clean_prices[common] / prev_prices[common] - 1) * 100
                            row['Stocks_Mean_Return'] = float(returns.mean())
                            row['Stocks_Median_Return'] = float(returns.median())
                            row['Stocks_Std_Return'] = float(returns.std())
                            row['Stocks_Min_Return'] = float(returns.min())
                            row['Stocks_Max_Return'] = float(returns.max())
                
                if volumes is not None and date in volumes.index:
                    daily_vol = volumes.loc[date]
                    if isinstance(daily_vol, pd.Series):
                        row['Stocks_Total_Volume'] = float(daily_vol.sum())
                        row['Stocks_Mean_Volume'] = float(daily_vol.mean())
        except:
            row['Stocks_Count'] = 0
        
        summary.append(row)
        
        if idx % 100 == 0:
            log(f"  Processed {idx+1}/{len(close_prices)} days...")
    
    df = pd.DataFrame(summary)
    df['Date'] = pd.to_datetime(df['Date'])
    
    return df


def fetch_summary_data(index_type='sp500', years=10, save_csv=None):
    
    index_symbols = {
        'sp500': '^GSPC',
        'dow': '^DJI',
        'nasdaq100': '^NDX'
    }
    
    index_names = {
        'sp500': 'SP500',
        'dow': 'DOW',
        'nasdaq100': 'NAS100'
    }
    
    log("=" * 60)
    log(f"FETCHING {index_type.upper()} DATA FOR LAST {years} YEARS")
    log("=" * 60)
    
    # Get tickers
    log("\n[1/4] Getting ticker list...")
    tickers = get_tickers(index_type)
    if not tickers:
        log("Failed to get tickers")
        return None
    log(f"Working with {len(tickers)} stocks")
    
    # Set date range
    end_date = datetime.now()
    start_date = end_date - timedelta(days=years*365)
    
    log(f"\n[2/4] Date range: {start_date.strftime('%Y-%m-%d')} to {end_date.strftime('%Y-%m-%d')}")
    
    # Fetch index data
    log(f"\n[3/4] Fetching {index_type.upper()} index data...")
    index_data = fetch_index_data(index_symbols.get(index_type, '^GSPC'), 
                                   start_date, end_date)
    
    # Fetch stock data
    log(f"\n[4/4] Fetching stock data for {len(tickers)} stocks...")
    close_prices, volumes = fetch_stock_data(tickers, start_date, end_date)
    
    if close_prices is None:
        log("✗ Failed to fetch stock data")
        return None
    
    # Create summary
    log("\nCreating summary...")
    index_name = index_names.get(index_type, 'Index')
    df = create_summary(close_prices, volumes, index_data, index_name)
    
    # Add features
    log("\nAdding technical features...")
    price_col = f'{index_name}_Close'
    volume_col = f'{index_name}_Volume'
    
    df = add_features(df, price_col, volume_col)
    df = add_cyclical_features(df)
    
    # Save to CSV if requested
    if save_csv:
        df.to_csv(save_csv, index=False)
        log(f"\nData saved to: {save_csv}")
    
    log("\n" + "=" * 60)
    log("DATA FETCH COMPLETE!")
    log("=" * 60)
    log(f"Total days: {len(df)}")
    log(f"Date range: {df['Date'].iloc[0]} to {df['Date'].iloc[-1]}")
    if 'Stocks_Count' in df.columns:
        log(f"Average stocks/day: {df['Stocks_Count'].mean():.0f}")
    log(f"Total features: {len(df.columns) - 1}")
    
    return df


#  BINARY STORAGE FUNCTIONS (C Integration) 

def save_stock_to_binary(symbol, df, timeframe='1d'):
    
    if df is None or df.empty:
        log(f"✗ No data to save for {symbol}")
        return False
    
    # Create directory
    os.makedirs(f'data/{symbol}', exist_ok=True)
    filepath = f'data/{symbol}/{timeframe}.bin'
    
    # Convert to storage format
    df_storage = pd.DataFrame({
        'timestamp': pd.to_datetime(df.index).astype(np.int64) // 10**9,
        'open': df['Open'].values,
        'high': df['High'].values,
        'low': df['Low'].values,
        'close': df['Close'].values,
        'volume': df['Volume'].values.astype(np.float64)
    })
    
    # Remove NaN rows
    df_storage = df_storage.dropna()
    
    if df_storage.empty:
        log(f"✗ No valid data after cleaning for {symbol}")
        return False
    
    # Save to binary
    try:
        if os.path.exists(filepath):
            stock_data.append_bars(filepath, df_storage)
            log(f"✓ Appended {len(df_storage)} bars to {filepath}")
        else:
            stock_data.create_stock_file(filepath, symbol, timeframe)
            stock_data.append_bars(filepath, df_storage)
            log(f"Created {filepath} with {len(df_storage)} bars")
        return True
    except Exception as e:
        log(f"Error saving {symbol}: {e}")
        return False


def load_stock_from_binary(symbol, timeframe='1d'):
   
    filepath = f'data/{symbol}/{timeframe}.bin'
    if not os.path.exists(filepath):
        log(f"✗ No data for {symbol}")
        return None
    
    try:
        np_array = stock_data.read_all_stock_data(filepath)
        if len(np_array) == 0:
            return pd.DataFrame()
        
        df = pd.DataFrame(np_array)
        df['Date'] = pd.to_datetime(df['timestamp'], unit='s')
        df.set_index('Date', inplace=True)
        # Rename columns to match expected format
        df.rename(columns={
            'open': 'Open',
            'high': 'High',
            'low': 'Low',
            'close': 'Close',
            'volume': 'Volume'
        }, inplace=True)
        return df
    except Exception as e:
        log(f"✗ Error loading {symbol}: {e}")
        return None


def fetch_and_save_stock(symbol, timeframe='1d', years=5):
    
    log(f"Fetching {symbol}...")
    df = fetch_individual_stock(symbol, years=years, timeframe=timeframe)
    
    if df.empty:
        log(f"No data for {symbol}")
        return False
    
    return save_stock_to_binary(symbol, df, timeframe)


def save_sp500_to_binary(df, timeframe='1d'):
    
    if df is None or df.empty:
        log("No data to save for SP500")
        return False
    
    os.makedirs('data/SP500', exist_ok=True)
    filepath = f'data/SP500/{timeframe}.bin'
    
    # Prepare data
    df_storage = pd.DataFrame({
        'timestamp': pd.to_datetime(df['Date']).astype(np.int64) // 10**9,
        'open': df['SP500_Open'],
        'high': df['SP500_High'],
        'low': df['SP500_Low'],
        'close': df['SP500_Close'],
        'volume': df['SP500_Volume']
    }).dropna()
    
    try:
        if os.path.exists(filepath):
            stock_data.append_bars(filepath, df_storage)
            log(f"Appended to {filepath}")
        else:
            stock_data.create_stock_file(filepath, 'SP500', timeframe)
            stock_data.append_bars(filepath, df_storage)
            log(f"Created {filepath} with {len(df_storage)} bars")
        return True
    except Exception as e:
        log(f"Error saving SP500: {e}")
        return False


#  CONVENIENCE FUNCTIONS 

def fetch_sp500_summary(years=10, save_csv='sp500_summary.csv', save_binary=True):
    
    df = fetch_summary_data('sp500', years, save_csv)
    if df is not None and save_binary:
        save_sp500_to_binary(df)
    return df


def fetch_dow_summary(years=10, save_csv='dow_summary.csv', save_binary=True):
 
    df = fetch_summary_data('dow', years, save_csv)
    return df


def fetch_nasdaq_summary(years=10, save_csv='nasdaq_summary.csv', save_binary=True):
    
    df = fetch_summary_data('nasdaq100', years, save_csv)
    return df


def fetch_and_store_stocks(symbols, timeframe='1d', years=5):
    
    results = {}
    for symbol in symbols:
        success = fetch_and_save_stock(symbol, timeframe, years)
        results[symbol] = success
    return results


def load_multiple_stocks(symbols, timeframe='1d'):
    
    results = {}
    for symbol in symbols:
        df = load_stock_from_binary(symbol, timeframe)
        if df is not None:
            results[symbol] = df
    return results


# MAIN (for testing) 
if __name__ == "__main__":
    print("\n" + "=" * 60)
    print("STOCK DATA FETCHER - LIBRARY MODE")
    print("=" * 60)
    
    print("\nThis is now a library. Import it in your own scripts:")
    print("  from yfin import fetch_and_save_stock, load_stock_from_binary, fetch_sp500_summary")
    print("\nExample usage:")
    print("  # Fetch and save individual stock")
    print("  fetch_and_save_stock('AAPL', timeframe='1d', years=3)")
    print("")
    print("  # Load from binary")
    print("  df = load_stock_from_binary('AAPL')")
    print("")
    print("  # Fetch S&P 500 data")
    print("  df = fetch_sp500_summary(years=5, save_csv='sp500_5y.csv')")
    print("")
    print("  # Fetch multiple stocks")
    print("  fetch_and_store_stocks(['AAPL', 'MSFT', 'GOOGL'], years=3)")
    
    # Test if user wants to run
    print("\n" + "=" * 60)
    response = input("Run test fetch? (y/n): ").strip().lower()
    
    if response == 'y':
        symbols = ['AAPL', 'MSFT', 'GOOGL']
        for symbol in symbols:
            fetch_and_save_stock(symbol, timeframe='1d', years=2)
        
        print("\n" + "=" * 60)
        print("Loading back from binary...")
        for symbol in symbols:
            df = load_stock_from_binary(symbol)
            if df is not None:
                print(f"{symbol}: {len(df)} bars loaded")
                print(df.head(2))
                print()