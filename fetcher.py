import os
import sys
from datetime import datetime, timedelta

import numpy as np
import pandas as pd
import yfinance as yf

# Import library
try:
    from yfin import load_stock_from_binary, save_stock_to_binary
except ImportError:
    # Fallback to direct import
    import stock_data

    def save_stock_to_binary(symbol, df, timeframe="1d"):

        if df.empty:
            return False

        os.makedirs(f"data/{symbol}", exist_ok=True)
        filepath = f"data/{symbol}/{timeframe}.bin"

        # Flatten multi-level columns if they exist
        if isinstance(df.columns, pd.MultiIndex):
            # Extract the price data and create a flat DataFrame
            df_flat = pd.DataFrame()

            # Map the multi-level columns to simple column names
            col_mapping = {
                "Open": "Open",
                "High": "High",
                "Low": "Low",
                "Close": "Close",
                "Volume": "Volume",
            }

            for col_name in col_mapping:
                if (col_name, symbol) in df.columns:
                    df_flat[col_name] = df[(col_name, symbol)]
                elif col_name in df.columns:
                    df_flat[col_name] = df[col_name]

            df = df_flat

        # Ensure we have the required columns
        required_cols = ["Open", "High", "Low", "Close", "Volume"]
        if not all(col in df.columns for col in required_cols):
            print(f"✗ Missing required columns. Available: {df.columns.tolist()}")
            return False

        df_storage = pd.DataFrame(
            {
                "timestamp": df.index.astype(np.int64) // 10**9,
                "open": df["Open"].values,
                "high": df["High"].values,
                "low": df["Low"].values,
                "close": df["Close"].values,
                "volume": df["Volume"].values.astype(np.float64),
            }
        ).dropna()

        try:
            if os.path.exists(filepath):
                stock_data.append_bars(filepath, df_storage)
                print(f"Appended {len(df_storage)} bars to {filepath}")
            else:
                stock_data.create_stock_file(filepath, symbol, timeframe)
                stock_data.append_bars(filepath, df_storage)
                print(f"Created {filepath} with {len(df_storage)} bars")
            return True
        except Exception as e:
            print(f"✗ Error saving {symbol}: {e}")
            return False


def is_valid_stock(symbol):

    if not symbol or not isinstance(symbol, str):
        return False, None

    symbol = symbol.strip().upper()

    try:
        ticker = yf.Ticker(symbol)
        info = ticker.info

        if info and isinstance(info, dict):
            if "symbol" in info and info["symbol"] == symbol:
                return True, symbol
            if "longName" in info or "shortName" in info:
                return True, symbol

        data = ticker.history(period="1d")
        if not data.empty:
            return True, symbol

        return False, None
    except:
        return False, None


def get_stock_choice():

    while True:
        choice = input("What stock do you want to fetch? (e.g., AAPL): ").strip()

        if not choice:
            print("Empty input, please enter a stock symbol!")
            continue

        valid, symbol = is_valid_stock(choice)
        if valid:
            return symbol
        else:
            print(f"'{choice}' is not a valid stock symbol. Please try again.")


def get_timeframe():

    print("\nTimeframe options:")
    print("1. Specific number of years")
    print("2. Yesterday only")

    while True:
        choice = input("Enter 1 or 2: ").strip()

        if choice == "1":
            return get_years()
        elif choice == "2":
            return "yesterday"
        else:
            print("Invalid choice. Please enter 1 or 2.")


def get_years():

    while True:
        try:
            years = int(input("How many years of data? (1-20): ").strip())
            if 1 <= years <= 20:
                return years
            else:
                print("Please enter a number between 1 and 20.")
        except ValueError:
            print("Please enter a valid number.")


def get_action():

    print("\n" + "=" * 50)
    print("STOCK DATA FETCHER")
    print("=" * 50)
    print("1. Fetch stock data")
    print("2. Help")
    print("3. Exit")

    while True:
        try:
            choice = int(input("\nEnter your choice (1-3): ").strip())
            if 1 <= choice <= 3:
                return choice
            else:
                print("Please enter 1, 2, or 3.")
        except ValueError:
            print("Please enter a valid number.")


def fetch_stock_data(symbol, timeframe):

    print(f"\nFetching data for {symbol}...")

    try:
        if timeframe == "yesterday":
            # Get yesterday's data
            today = datetime.today()
            yesterday = today - timedelta(days=1)

            # Use auto_adjust=False to get simpler column structure
            data = yf.download(
                symbol,
                start=yesterday - timedelta(days=5),
                end=today,
                auto_adjust=False,  # Changed to False to avoid multi-index columns
                progress=False,
            )

            if not data.empty:
                # Get the last trading day
                data = data.iloc[[-1]]

                # If we still have multi-index columns, flatten them
                if isinstance(data.columns, pd.MultiIndex):
                    # Create a flat DataFrame with simple column names
                    data_flat = pd.DataFrame(index=data.index)
                    for col in ["Open", "High", "Low", "Close", "Volume"]:
                        if (col, symbol) in data.columns:
                            data_flat[col] = data[(col, symbol)]
                        elif col in data.columns:
                            data_flat[col] = data[col]
                    data = data_flat

                # Adjust prices manually if needed
                if "Adj Close" in data.columns:
                    data["Close"] = data["Adj Close"]

                print(f"✓ Got data for {data.index[-1].strftime('%Y-%m-%d')}")
                return data, "1d"
            else:
                print("✗ No data found")
                return None, None
        else:
            # Fetch specified years
            data = yf.download(
                symbol,
                period=f"{timeframe}y",
                auto_adjust=False,  # Changed to False
                progress=False,
            )

            if not data.empty:
                # Flatten multi-index columns if necessary
                if isinstance(data.columns, pd.MultiIndex):
                    data_flat = pd.DataFrame(index=data.index)
                    for col in ["Open", "High", "Low", "Close", "Volume"]:
                        if (col, symbol) in data.columns:
                            data_flat[col] = data[(col, symbol)]
                        elif col in data.columns:
                            data_flat[col] = data[col]
                    data = data_flat

                print(f"✓ Downloaded {len(data)} days of data")
                print(
                    f"  Date range: {data.index[0].strftime('%Y-%m-%d')} to {data.index[-1].strftime('%Y-%m-%d')}"
                )
                return data, f"{timeframe}y"
            else:
                print("✗ No data found")
                return None, None
    except Exception as e:
        print(f"✗ Error fetching data: {e}")
        return None, None


def main():

    while True:
        action = get_action()

        if action == 1:
            # Fetch stock data
            symbol = get_stock_choice()
            timeframe = get_timeframe()

            if timeframe == "yesterday":
                data, tf_str = fetch_stock_data(symbol, "yesterday")
                if data is not None:
                    timeframe_str = "1d"
            else:
                data, tf_str = fetch_stock_data(symbol, timeframe)
                timeframe_str = tf_str

            if data is not None and not data.empty:
                print("\n" + "=" * 50)
                print("DATA PREVIEW")
                print("=" * 50)
                print(data.tail(5))

                # Ask to save
                save_choice = input("\nSave to binary file? (y/n): ").strip().lower()
                if save_choice == "y":
                    # Print debug info about the DataFrame
                    print(f"DataFrame shape: {data.shape}")
                    print(f"Column names: {data.columns.tolist()}")
                    print(f"Column types: {type(data.columns)}")

                    success = save_stock_to_binary(symbol, data, timeframe_str)
                    if success:
                        print("Data saved successfully!")

                        # Offer to load back
                        load_choice = (
                            input("Load data back to verify? (y/n): ").strip().lower()
                        )
                        if load_choice == "y":
                            loaded_df = load_stock_from_binary(symbol, timeframe_str)
                            if loaded_df is not None:
                                print(f"✓ Loaded {len(loaded_df)} bars")
                                print(loaded_df.tail())

                input("\nPress Enter to continue...")

        elif action == 2:
            # Help
            print("\n" + "=" * 50)
            print("HELP")
            print("=" * 50)
            print("This program fetches stock data from Yahoo Finance")
            print("and saves it in an efficient binary format.")
            print("\nFeatures:")
            print("- Fetch historical data for any stock")
            print("- Save to optimized binary files")
            print("- Fast loading with C library")
            print("\nSupported timeframes:")
            print("- Yesterday only (for quick updates)")
            print("- Multiple years (1-20)")
            input("\nPress Enter to continue...")

        elif action == 3:
            # Exit
            print("\nGoodbye!")
            break


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\nProgram interrupted. Goodbye!")
    except Exception as e:
        print(f"\nAn error occurred: {e}")
