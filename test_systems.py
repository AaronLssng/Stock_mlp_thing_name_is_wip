"""
Quick test script to verify the entire system works. (AI generated!!!!!)
"""

import os
import sys
import pandas as pd
import numpy as np

print("=" * 60)
print("SYSTEM TEST")
print("=" * 60)

# TEST 1: Check stock_data module
print("\n[TEST 1] Checking stock_data module...")
try:
    import stock_data
    print("✓ stock_data imported successfully")
    
    # Check all attributes (not just top-level)
    import inspect
    functions = [name for name, obj in inspect.getmembers(stock_data, inspect.isfunction)]
    print(f"  Available functions: {functions}")
    
    # Check for required functions
    required = ['create_stock_file', 'append_bars', 'read_all_stock_data']
    missing = [r for r in required if not hasattr(stock_data, r)]
    
    if missing:
        print(f"✗ Missing functions: {missing}")
        print("  These functions should be defined in stock_data.py")
        sys.exit(1)
    else:
        print("All required functions found")
        
except ImportError as e:
    print(f"Failed to import stock_data: {e}")
    sys.exit(1)

# TEST 2: Create Binary File
print("\n[TEST 2] Creating binary file")
try:
    # Clean up from previous test
    import shutil
    if os.path.exists('data/TEST'):
        shutil.rmtree('data/TEST')
    
    os.makedirs('data/TEST', exist_ok=True)
    filepath = 'data/TEST/1d.bin'
    stock_data.create_stock_file(filepath, 'TEST', '1d')
    print(f"✓ Created {filepath}")
except Exception as e:
    print(f"✗ Failed to create file: {e}")
    sys.exit(1)

# TEST 3: Append and Read Bars

print("\n[TEST 3] Appending and reading bars...")
try:
    # Create test data
    test_data = pd.DataFrame({
        'timestamp': [1609459200, 1609545600, 1609632000],
        'open': [100.0, 101.0, 102.0],
        'high': [101.0, 102.0, 103.0],
        'low': [99.0, 100.0, 101.0],
        'close': [100.5, 101.5, 102.5],
        'volume': [1000000.0, 1100000.0, 1200000.0]
    })
    
    # Append
    stock_data.append_bars(filepath, test_data)
    print(f"✓ Appended {len(test_data)} bars")
    
    # Read back
    result = stock_data.read_all_stock_data(filepath)
    print(f"✓ Read back {len(result)} bars")
    
    if len(result) > 0:
        print(f"  First bar: timestamp={result[0]['timestamp']}, close={result[0]['close']}")
    
    if len(result) == len(test_data):
        print("✓ Data length matches")
    else:
        print(f"✗ Data length mismatch: expected {len(test_data)}, got {len(result)}")
        
except Exception as e:
    print(f"✗ Test failed: {e}")
    import traceback
    traceback.print_exc()

# TEST 4: Fetch from yfinance
print("\n[TEST 4] Fetching from yfinance...")
try:
    import yfinance as yf
    df = yf.download('AAPL', period='1mo', progress=False)
    if not df.empty:
        print(f"✓ Fetched {len(df)} bars for AAPL")
        print(f"  Date range: {df.index[0]} to {df.index[-1]}")
    else:
        print("✗ No data returned")
except Exception as e:
    print(f"✗ Test failed: {e}")

# SUMMARY
print("\n" + "=" * 60)
print("TEST COMPLETE")
print("=" * 60)

# Check for binary files
print("\nFiles in data/ directory:")
if os.path.exists('data'):
    for root, dirs, files in os.walk('data'):
        for f in files:
            filepath = os.path.join(root, f)
            size = os.path.getsize(filepath)
            print(f"  {filepath} ({size} bytes)")
        if not files:
            print("  (no files found)")
else:
    print("  data/ directory does not exist")