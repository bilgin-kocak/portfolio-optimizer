#!/usr/bin/env python3
"""
Test script to verify yfinance data fetching works correctly
"""

import yfinance as yf
import pandas as pd
from datetime import datetime, timedelta
import time

def test_individual_tickers():
    """Test fetching data for individual tickers"""
    tickers = ['AAPL', 'GOOGL', 'MSFT', 'BTC-USD', 'ETH-USD']
    start_date = (datetime.now() - timedelta(days=30)).strftime('%Y-%m-%d')
    end_date = datetime.now().strftime('%Y-%m-%d')
    
    print("Testing individual ticker fetching...")
    print(f"Date range: {start_date} to {end_date}")
    print(f"Tickers: {', '.join(tickers)}")
    print("-" * 50)
    
    successful = []
    failed = []
    
    for ticker in tickers:
        try:
            data = yf.download(
                ticker,
                start=start_date,
                end=end_date,
                progress=False
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
    
    return successful, failed

def test_batch_download():
    """Test fetching data for multiple tickers at once"""
    tickers = ['AAPL', 'GOOGL', 'MSFT']
    start_date = (datetime.now() - timedelta(days=30)).strftime('%Y-%m-%d')
    end_date = datetime.now().strftime('%Y-%m-%d')
    
    print("\nTesting batch download...")
    print(f"Date range: {start_date} to {end_date}")
    print(f"Tickers: {', '.join(tickers)}")
    print("-" * 50)
    
    try:
        data = yf.download(
            tickers,
            start=start_date,
            end=end_date,
            progress=False,
            threads=True,
            group_by='ticker'
        )
        
        if not data.empty:
            print(f"✅ Batch download successful: {data.shape}")
            print(f"✅ Columns: {list(data.columns)}")
            return True
        else:
            print("❌ Batch download returned empty data")
            return False
            
    except Exception as e:
        print(f"❌ Batch download failed: {str(e)}")
        return False

def test_crypto_and_commodities():
    """Test fetching crypto and commodity data"""
    crypto_tickers = ['BTC-USD', 'ETH-USD']
    commodity_tickers = ['GLD', 'SLV']
    
    print("\nTesting crypto and commodities...")
    print("-" * 50)
    
    all_successful = True
    
    for category, tickers in [("Crypto", crypto_tickers), ("Commodities", commodity_tickers)]:
        print(f"\n{category}:")
        for ticker in tickers:
            try:
                data = yf.download(
                    ticker,
                    period="1mo",
                    progress=False
                )
                
                if not data.empty:
                    print(f"  ✅ {ticker}: {len(data)} days")
                else:
                    print(f"  ❌ {ticker}: No data")
                    all_successful = False
                    
            except Exception as e:
                print(f"  ❌ {ticker}: Error - {str(e)}")
                all_successful = False
    
    return all_successful

def main():
    """Main test function"""
    print("🧪 yfinance Data Fetching Test")
    print("=" * 60)
    
    # Test 1: Individual tickers
    successful_individual, failed_individual = test_individual_tickers()
    
    # Test 2: Batch download
    batch_success = test_batch_download()
    
    # Test 3: Crypto and commodities
    crypto_commodity_success = test_crypto_and_commodities()
    
    # Summary
    print("\n" + "=" * 60)
    print("📊 TEST SUMMARY")
    print("-" * 60)
    print(f"Individual tickers successful: {len(successful_individual)}")
    print(f"Individual tickers failed: {len(failed_individual)}")
    print(f"Batch download: {'✅ PASSED' if batch_success else '❌ FAILED'}")
    print(f"Crypto & Commodities: {'✅ PASSED' if crypto_commodity_success else '❌ FAILED'}")
    
    overall_success = (len(successful_individual) > 0 and 
                      batch_success and 
                      crypto_commodity_success)
    
    if overall_success:
        print("\n🎉 All tests passed! yfinance should work in your app.")
    else:
        print("\n⚠️ Some tests failed. You may need to:")
        print("   - Check your internet connection")
        print("   - Try upgrading yfinance: pip install --upgrade yfinance")
        print("   - Wait a few minutes and try again (rate limiting)")
    
    return overall_success

if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)