#!/usr/bin/env python3
"""
Test script to verify yfinance data fetching works correctly
"""

import yfinance as yf
import pandas as pd
from datetime import datetime, timedelta
import pytz

def test_fetch_data():
    """Test fetching data for common tickers"""
    tickers = ['AAPL', 'GOOGL', 'MSFT', 'BTC-USD', 'ETH-USD']
    start_date = datetime.now() - timedelta(days=30)
    end_date = datetime.now()
    
    print("Testing yfinance data fetching...")
    print(f"Date range: {start_date.date()} to {end_date.date()}")
    print(f"Tickers: {', '.join(tickers)}")
    print("-" * 50)
    
    successful = []
    failed = []
    
    for ticker in tickers:
        try:
            # Add timezone info
            if start_date.tzinfo is None:
                start_date = pytz.UTC.localize(start_date)
            if end_date.tzinfo is None:
                end_date = pytz.UTC.localize(end_date)
            
            ticker_obj = yf.Ticker(ticker)
            data = ticker_obj.history(
                start=start_date,
                end=end_date,
                auto_adjust=True,
                repair=True,
                keepna=False,
                actions=False
            )
            
            if not data.empty and len(data) > 5:
                successful.append(ticker)
                print(f"✅ {ticker}: {len(data)} days of data")
            else:
                failed.append(ticker)
                print(f"❌ {ticker}: No data or insufficient data")
                
        except Exception as e:
            failed.append(ticker)
            print(f"❌ {ticker}: Error - {str(e)}")
    
    print("-" * 50)
    print(f"✅ Successful: {len(successful)}")
    print(f"❌ Failed: {len(failed)}")
    
    if successful:
        print(f"Working tickers: {', '.join(successful)}")
    if failed:
        print(f"Failed tickers: {', '.join(failed)}")
    
    return len(successful) > 0

if __name__ == "__main__":
    success = test_fetch_data()
    if success:
        print("\n🎉 Data fetching test passed!")
    else:
        print("\n💥 Data fetching test failed!")