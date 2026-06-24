import ctypes
import numpy as np
import os


dll_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'libstockdata.dll')
if not os.path.exists(dll_path):
    dll_path = 'libstockdata.dll'

lib = ctypes.CDLL(dll_path)


class StockHeader(ctypes.Structure):
    _fields_ = [
        ('stock_name', ctypes.c_char * 16),
        ('timeframe', ctypes.c_char * 8),
        ('num_bars', ctypes.c_uint64),
        ('start_timestamp', ctypes.c_int64),
        ('end_timestamp', ctypes.c_int64),
        ('reserved', ctypes.c_char * 16),
    ]


class StockBar(ctypes.Structure):
    _fields_ = [
        ('timestamp', ctypes.c_int64),
        ('open', ctypes.c_float),
        ('high', ctypes.c_float),
        ('low', ctypes.c_float),
        ('close', ctypes.c_float),
        ('volume', ctypes.c_double),
    ]


BAR_DTYPE = np.dtype([
    ('timestamp', np.int64),
    ('open', np.float32),
    ('high', np.float32),
    ('low', np.float32),
    ('close', np.float32),
    ('volume', np.float64),
])


lib.stock_create_file.argtypes = [ctypes.c_char_p, ctypes.c_char_p, ctypes.c_char_p]
lib.stock_create_file.restype = ctypes.c_int

lib.stock_append_bars.argtypes = [ctypes.c_char_p, ctypes.POINTER(StockBar), ctypes.c_size_t]
lib.stock_append_bars.restype = ctypes.c_int

lib.stock_read_all.argtypes = [ctypes.c_char_p, ctypes.POINTER(ctypes.POINTER(StockBar)), ctypes.POINTER(ctypes.c_size_t)]
lib.stock_read_all.restype = ctypes.c_int

lib.stock_free_bars.argtypes = [ctypes.POINTER(StockBar)]
lib.stock_free_bars.restype = None


def create_stock_file(filepath, stock_name, timeframe):
    """Create a new stock data file."""
    result = lib.stock_create_file(
        filepath.encode('utf-8'),
        stock_name.encode('utf-8'),
        timeframe.encode('utf-8')
    )
    
    if result != 0:
        error_messages = {
            -1: "File not found",
            -2: "IO error",
            -3: "Invalid header",
            -4: "Out of range",
            -5: "Memory error",
            -6: "Invalid argument",
            -7: "File already exists"
        }
        error_msg = error_messages.get(result, f"Unknown error (code: {result})")
        raise RuntimeError(f"Failed to create file: {filepath} - {error_msg}")


def append_bars(filepath, bars_df):
    """
    Append bars from a pandas DataFrame to an existing file.
    bars_df must have columns: timestamp, open, high, low, close, volume
    """
    if bars_df.empty:
        return
    
    df = bars_df[['timestamp', 'open', 'high', 'low', 'close', 'volume']].copy()
    df['timestamp'] = df['timestamp'].astype(np.int64)
    df['open'] = df['open'].astype(np.float32)
    df['high'] = df['high'].astype(np.float32)
    df['low'] = df['low'].astype(np.float32)
    df['close'] = df['close'].astype(np.float32)
    df['volume'] = df['volume'].astype(np.float64)
    
    n_bars = len(df)
    bar_array = (StockBar * n_bars)()
    
    for i in range(n_bars):
        bar_array[i].timestamp = int(df.iloc[i]['timestamp'])
        bar_array[i].open = float(df.iloc[i]['open'])
        bar_array[i].high = float(df.iloc[i]['high'])
        bar_array[i].low = float(df.iloc[i]['low'])
        bar_array[i].close = float(df.iloc[i]['close'])
        bar_array[i].volume = float(df.iloc[i]['volume'])
    
    result = lib.stock_append_bars(
        filepath.encode('utf-8'),
        bar_array,
        n_bars
    )
    
    if result != 0:
        raise RuntimeError(f"Failed to append bars to: {filepath} (error code: {result})")


def read_all_stock_data(filepath):
    """Read all bars from a file and return as numpy array."""
    filepath_bytes = filepath.encode('utf-8')
    bars_ptr = ctypes.POINTER(StockBar)()
    count = ctypes.c_size_t()
    
    result = lib.stock_read_all(filepath_bytes, ctypes.byref(bars_ptr), ctypes.byref(count))
    if result != 0:
        raise RuntimeError(f"Failed to read file: {filepath} (error code: {result})")
    
    n_bars = count.value
    if n_bars == 0:
        return np.empty(0, dtype=BAR_DTYPE)
    
    np_array = np.zeros(n_bars, dtype=BAR_DTYPE)
    
    for i in range(n_bars):
        bar = bars_ptr[i]
        np_array[i] = (bar.timestamp, bar.open, bar.high, bar.low, bar.close, bar.volume)
    
    lib.stock_free_bars(bars_ptr)
    
    return np_array