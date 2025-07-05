# Portfolio Optimizer

A Streamlit-based portfolio optimization application that implements Modern Portfolio Theory (MPT) to help users create and analyze investment portfolios across multiple asset classes.

## Features

### 1. Returns Distribution (Feature 1)
- View histogram of daily returns for any selected asset
- See expected return and volatility metrics
- Understand the risk-return profile of individual assets

### 2. Portfolio Analysis (Feature 2)
- Set custom portfolio weights using interactive sliders
- Calculate portfolio expected return and risk
- View portfolio composition with pie charts
- See key metrics like Sharpe ratio

### 3. Efficient Frontier (Feature 3)
- Generate efficient frontier using Monte Carlo simulation
- Find optimal portfolios (Maximum Sharpe Ratio and Minimum Variance)
- Visualize risk-return tradeoffs across thousands of portfolio combinations

## Installation

1. Clone this repository or download the files
2. Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage

Run the Streamlit app:
```bash
streamlit run app.py
```

The app will open in your browser. Then:

1. **Select Assets**: 
   - Choose from categories: Crypto, Stocks, ETFs, Commodities
   - Select multiple assets for your portfolio
   - Add custom tickers if needed

2. **Set Date Range**: 
   - Choose the historical period for analysis
   - Longer periods provide more data but may not reflect recent market conditions

3. **Explore Tabs**:
   - **Overview**: Price performance, correlations, and basic statistics
   - **Returns Distribution**: Histogram of daily returns for individual assets
   - **Portfolio Weights**: Set weights and see portfolio metrics
   - **Efficient Frontier**: Find optimal portfolio allocations

## Available Assets

- **Cryptocurrencies**: BTC-USD, ETH-USD, BNB-USD, ADA-USD, SOL-USD
- **Stocks**: AAPL, GOOGL, MSFT, AMZN, TSLA, META, NVDA
- **ETFs**: SPY (S&P 500), QQQ (NASDAQ), GLD (Gold), SLV (Silver), USO (Oil)
- **Commodities**: GC=F (Gold Futures), SI=F (Silver Futures), CL=F (Oil Futures)

## Key Metrics Explained

- **Expected Return**: Annualized average return based on historical data
- **Risk (Standard Deviation)**: Annualized volatility measure
- **Sharpe Ratio**: Risk-adjusted return metric (higher is better)
- **Correlation**: Measure of how assets move together (-1 to 1)

## Data Source

Historical price data is fetched from Yahoo Finance using the `yfinance` library. Free and no API key required.

## Tips

- Diversification across uncorrelated assets can reduce portfolio risk
- The efficient frontier shows the best possible risk-return combinations
- Past performance doesn't guarantee future results
- Consider transaction costs and taxes in real-world implementation

## Limitations

- Uses historical data which may not predict future performance
- Assumes normal distribution of returns
- Doesn't account for transaction costs or taxes
- Limited to assets available on Yahoo Finance