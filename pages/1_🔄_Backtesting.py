import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime, timedelta
import warnings
import time
warnings.filterwarnings('ignore')

st.title("🔄 Backtesting Engine with Rebalancing Simulation")
st.markdown("Test how your portfolio would have performed with different rebalancing strategies")

POPULAR_ASSETS = {
    "Stocks": ["AAPL", "GOOGL", "MSFT", "AMZN", "TSLA", "META", "NVDA"],
    "ETFs": ["SPY", "QQQ", "GLD", "SLV", "VTI", "BND"],
    "Crypto": ["BTC-USD", "ETH-USD", "BNB-USD"],
}

def validate_ticker(ticker):
    """Validate ticker format and common patterns"""
    if not ticker or not isinstance(ticker, str):
        return False
    
    ticker = ticker.upper().strip()
    
    # Basic validation - should be 1-5 characters for stocks, longer for other assets
    if len(ticker) < 1 or len(ticker) > 10:
        return False
    
    # Check for common patterns
    invalid_chars = [' ', '<', '>', '&', '"', "'", '\\', '/', '?', '#']
    if any(char in ticker for char in invalid_chars):
        return False
    
    return True

@st.cache_data(ttl=3600)
def fetch_historical_data(tickers, start_date, end_date):
    """Fetch historical data for backtesting - using the working method from main app"""
    if isinstance(tickers, str):
        tickers = [tickers]
    
    # Validate tickers first
    valid_tickers = []
    invalid_tickers = []
    
    for ticker in tickers:
        if validate_ticker(ticker):
            valid_tickers.append(ticker.upper().strip())
        else:
            invalid_tickers.append(ticker)
    
    if invalid_tickers:
        st.warning(f"⚠️ Invalid ticker format: {', '.join(invalid_tickers)}")
    
    if not valid_tickers:
        st.error("❌ No valid tickers to fetch data for!")
        return None
    
    # Convert dates to string format for yf.download()
    if hasattr(start_date, 'strftime'):
        start_str = start_date.strftime('%Y-%m-%d')
    else:
        start_str = str(start_date)
    
    if hasattr(end_date, 'strftime'):
        end_str = end_date.strftime('%Y-%m-%d')
    else:
        end_str = str(end_date)
    
    # Try to fetch data with retries
    max_retries = 3
    data = None
    
    for attempt in range(max_retries):
        try:
            # Use yf.download() with threading for better performance
            data = yf.download(
                tickers=valid_tickers,
                start=start_str,
                end=end_str,
                progress=False,
                threads=True if len(valid_tickers) > 1 else False,
                group_by='ticker' if len(valid_tickers) > 1 else None
            )
            
            if data is not None and not data.empty:
                break
            else:
                raise ValueError("No data returned")
                
        except Exception as e:
            error_msg = str(e).lower()
            if attempt < max_retries - 1:
                if "json" in error_msg or "expecting value" in error_msg:
                    # Wait and retry for JSON decode errors
                    time.sleep(2 ** attempt)  # Exponential backoff
                    continue
                elif "429" in error_msg or "rate limit" in error_msg:
                    # Rate limiting - wait longer
                    time.sleep(5 * (attempt + 1))
                    continue
            else:
                st.error(f"❌ Failed to fetch data after {max_retries} attempts: {str(e)}")
                return None
    
    if data is None or data.empty:
        st.error("❌ No data received from Yahoo Finance")
        return None
    
    # Process the data based on structure
    try:
        if len(valid_tickers) == 1:
            # Single ticker - data is a simple DataFrame
            if 'Adj Close' in data.columns:
                result = data[['Adj Close']].copy()
                result.columns = valid_tickers
            elif 'Close' in data.columns:
                result = data[['Close']].copy()
                result.columns = valid_tickers
            else:
                st.error("❌ No price data found in the response")
                return None
        else:
            # Multiple tickers - data is grouped
            result = pd.DataFrame()
            successful_tickers = []
            failed_tickers = []
            
            for ticker in valid_tickers:
                try:
                    if ('Adj Close', ticker) in data.columns:
                        result[ticker] = data[('Adj Close', ticker)]
                        successful_tickers.append(ticker)
                    elif ('Close', ticker) in data.columns:
                        result[ticker] = data[('Close', ticker)]
                        successful_tickers.append(ticker)
                    else:
                        # Try to find ticker data in any form
                        ticker_cols = [col for col in data.columns if ticker in str(col)]
                        if ticker_cols:
                            # Use the first available column for this ticker
                            result[ticker] = data[ticker_cols[0]]
                            successful_tickers.append(ticker)
                        else:
                            failed_tickers.append(ticker)
                except Exception:
                    failed_tickers.append(ticker)
            
            if failed_tickers:
                st.warning(f"⚠️ Could not fetch data for: {', '.join(failed_tickers)}")
            
            if successful_tickers:
                st.success(f"✅ Successfully loaded: {', '.join(successful_tickers)}")
            
            if result.empty:
                st.error("❌ No data could be processed for any ticker")
                return None
        
        # Clean the data
        result = result.dropna()
        
        if len(result) < 10:
            st.warning("⚠️ Limited data available - results may be less accurate")
        
        return result
        
    except Exception as e:
        st.error(f"❌ Error processing data: {str(e)}")
        return None

def calculate_portfolio_value(prices, weights, initial_investment):
    """Calculate portfolio value over time without rebalancing"""
    initial_prices = prices.iloc[0]
    shares = (initial_investment * weights) / initial_prices
    portfolio_values = (prices * shares).sum(axis=1)
    return portfolio_values

def rebalance_portfolio(current_prices, target_weights, portfolio_value, transaction_cost=0.001):
    """Rebalance portfolio to target weights"""
    current_values = portfolio_value * target_weights
    shares = current_values / current_prices
    
    transaction_costs = portfolio_value * transaction_cost
    adjusted_portfolio_value = portfolio_value - transaction_costs
    
    return shares, adjusted_portfolio_value

def simulate_rebalancing(prices, target_weights, initial_investment, 
                        rebalance_frequency='quarterly', threshold=None, transaction_cost=0.001):
    """Simulate portfolio with rebalancing"""
    portfolio_values = []
    rebalance_dates = []
    
    current_shares = (initial_investment * target_weights) / prices.iloc[0]
    
    frequency_map = {
        'monthly': 'ME',
        'quarterly': 'QE',
        'semi-annual': '6ME',
        'annual': 'YE'
    }
    
    if rebalance_frequency != 'threshold':
        rebalance_schedule = pd.date_range(
            start=prices.index[0],
            end=prices.index[-1],
            freq=frequency_map.get(rebalance_frequency, 'QE')
        )
    
    for date, row in prices.iterrows():
        current_value = (row * current_shares).sum()
        portfolio_values.append(current_value)
        
        should_rebalance = False
        
        if rebalance_frequency == 'threshold' and threshold:
            current_weights = (row * current_shares) / current_value
            weight_drift = np.abs(current_weights - target_weights)
            if (weight_drift > threshold).any():
                should_rebalance = True
        elif rebalance_frequency != 'threshold' and date in rebalance_schedule:
            should_rebalance = True
        
        if should_rebalance and current_value > 0:
            current_shares, adjusted_value = rebalance_portfolio(
                row, target_weights, current_value, transaction_cost
            )
            portfolio_values[-1] = adjusted_value
            rebalance_dates.append(date)
    
    return pd.Series(portfolio_values, index=prices.index), rebalance_dates

def calculate_performance_metrics(returns_series):
    """Calculate key performance metrics"""
    total_return = (returns_series.iloc[-1] / returns_series.iloc[0] - 1) * 100
    
    years = (returns_series.index[-1] - returns_series.index[0]).days / 365.25
    cagr = ((returns_series.iloc[-1] / returns_series.iloc[0]) ** (1/years) - 1) * 100
    
    daily_returns = returns_series.pct_change().dropna()
    volatility = daily_returns.std() * np.sqrt(252) * 100
    
    sharpe_ratio = (cagr - 2) / volatility if volatility > 0 else 0
    
    rolling_max = returns_series.expanding().max()
    drawdown = (returns_series - rolling_max) / rolling_max
    max_drawdown = drawdown.min() * 100
    
    return {
        'Total Return': total_return,
        'CAGR': cagr,
        'Volatility': volatility,
        'Sharpe Ratio': sharpe_ratio,
        'Max Drawdown': max_drawdown
    }

# Initialize default dates
default_start_date = datetime.now() - timedelta(days=365*5)
default_end_date = datetime.now()

# Initialize variables to store widget values
selected_assets = []
weights = {}

with st.sidebar:
    st.header("Backtesting Configuration")
    
    st.subheader("1. Select Assets")
    
    asset_categories = st.multiselect(
        "Asset Categories:",
        list(POPULAR_ASSETS.keys()),
        default=["Stocks"]
    )
    
    # If no categories selected, use Stocks as default
    if not asset_categories:
        asset_categories = ["Stocks"]
        st.info("Please select at least one asset category above")
    
    available_assets = []
    for category in asset_categories:
        available_assets.extend(POPULAR_ASSETS[category])
    
    # Ensure we have default selections
    if available_assets:
        default_assets = available_assets[:3]
    else:
        default_assets = []
    
    selected_assets = st.multiselect(
        "Select assets:",
        available_assets,
        default=default_assets,
        help="Select at least 2 assets for backtesting"
    )
    
    if len(selected_assets) < 2:
        st.warning("Please select at least 2 assets for portfolio backtesting")
    
    st.subheader("2. Target Allocation")
    weights = {}
    if selected_assets:
        st.info("Set target weights (must sum to 100%)")
        remaining_weight = 100.0
        
        for i, asset in enumerate(selected_assets):
            if i == len(selected_assets) - 1:
                weights[asset] = remaining_weight
                st.number_input(
                    f"{asset} (%)",
                    value=remaining_weight,
                    disabled=True,
                    key=f"weight_{asset}"
                )
            else:
                default_weight = remaining_weight / (len(selected_assets) - i)
                weight = st.number_input(
                    f"{asset} (%)",
                    min_value=0.0,
                    max_value=remaining_weight,
                    value=default_weight,
                    step=1.0,
                    key=f"weight_{asset}"
                )
                weights[asset] = weight
                remaining_weight -= weight
    
    st.subheader("3. Investment Settings")
    initial_investment = st.number_input(
        "Initial Investment ($)",
        min_value=1000,
        value=10000,
        step=1000
    )
    
    col1, col2 = st.columns(2)
    with col1:
        start_date = st.date_input(
            "Start Date",
            value=default_start_date.date(),
            max_value=default_end_date.date()
        )
    with col2:
        end_date = st.date_input(
            "End Date",
            value=default_end_date.date(),
            max_value=default_end_date.date()
        )
    
    st.subheader("4. Rebalancing Strategy")
    rebalance_type = st.radio(
        "Rebalancing Type:",
        ["Periodic", "Threshold-based", "No Rebalancing"]
    )
    
    if rebalance_type == "Periodic":
        rebalance_frequency = st.selectbox(
            "Rebalancing Frequency:",
            ["monthly", "quarterly", "semi-annual", "annual"],
            index=1
        )
        threshold = None
    elif rebalance_type == "Threshold-based":
        rebalance_frequency = "threshold"
        threshold = st.slider(
            "Rebalance when drift exceeds:",
            min_value=0.01,
            max_value=0.20,
            value=0.05,
            format="%.2f",
            help="Rebalance when any asset's weight drifts by this percentage"
        )
    else:
        rebalance_frequency = None
        threshold = None
    
    transaction_cost = st.number_input(
        "Transaction Cost (%)",
        min_value=0.0,
        max_value=1.0,
        value=0.1,
        step=0.01,
        help="Cost per rebalancing as percentage of portfolio value"
    ) / 100
    
    include_benchmark = st.checkbox("Include S&P 500 (SPY) benchmark", value=True)

# Debug section
st.write("### Debug Information:")
st.write(f"- Selected assets: {selected_assets}")
st.write(f"- Number of selected: {len(selected_assets)}")
st.write(f"- Weights: {weights}")
st.write(f"- Sum of weights: {sum(weights.values())}")
st.write(f"- Start date: {start_date}")
st.write(f"- End date: {end_date}")
st.write(f"- Date comparison: {start_date < end_date}")

if selected_assets and len(selected_assets) >= 2 and start_date < end_date:
    if abs(sum(weights.values()) - 100.0) > 0.01:
        st.error("Portfolio weights must sum to 100%")
        st.write(f"Current weight sum: {sum(weights.values())}")
    else:
        st.write("✅ Conditions met, fetching data...")
        tickers_to_fetch = selected_assets.copy()
        if include_benchmark and 'SPY' not in tickers_to_fetch:
            tickers_to_fetch.append('SPY')
        
        st.write(f"Fetching data for: {tickers_to_fetch}")
        prices = fetch_historical_data(tickers_to_fetch, start_date, end_date)
        
        st.write(f"Data fetched: {prices is not None}")
        if prices is not None:
            st.write(f"Data shape: {prices.shape}")
            st.write(f"Data columns: {list(prices.columns)}")
        
        if prices is not None and not prices.empty:
            portfolio_prices = prices[selected_assets]
            target_weights = np.array([weights[asset] / 100 for asset in selected_assets])
            
            buy_hold_values = calculate_portfolio_value(
                portfolio_prices, target_weights, initial_investment
            )
            
            if rebalance_type != "No Rebalancing":
                rebalanced_values, rebalance_dates = simulate_rebalancing(
                    portfolio_prices, target_weights, initial_investment,
                    rebalance_frequency, threshold, transaction_cost
                )
            else:
                rebalanced_values = buy_hold_values
                rebalance_dates = []
            
            if include_benchmark and 'SPY' in prices.columns:
                benchmark_values = (initial_investment / prices['SPY'].iloc[0]) * prices['SPY']
            
            tab1, tab2, tab3, tab4 = st.tabs([
                "📈 Performance Comparison", 
                "📊 Portfolio Drift", 
                "📅 Rebalancing Events",
                "📊 Performance Metrics"
            ])
            
            with tab1:
                st.header("Portfolio Performance Comparison")
                
                fig_performance = go.Figure()
                
                fig_performance.add_trace(go.Scatter(
                    x=buy_hold_values.index,
                    y=buy_hold_values,
                    name="Buy & Hold",
                    line=dict(color='blue', width=2)
                ))
                
                if rebalance_type != "No Rebalancing":
                    fig_performance.add_trace(go.Scatter(
                        x=rebalanced_values.index,
                        y=rebalanced_values,
                        name=f"Rebalanced ({rebalance_type})",
                        line=dict(color='green', width=2)
                    ))
                
                if include_benchmark and 'SPY' in prices.columns:
                    fig_performance.add_trace(go.Scatter(
                        x=benchmark_values.index,
                        y=benchmark_values,
                        name="S&P 500 (SPY)",
                        line=dict(color='red', width=2, dash='dash')
                    ))
                
                for date in rebalance_dates:
                    fig_performance.add_vline(
                        x=date,
                        line_dash="dot",
                        line_color="gray",
                        opacity=0.5
                    )
                
                fig_performance.update_layout(
                    title="Portfolio Value Over Time",
                    xaxis_title="Date",
                    yaxis_title="Portfolio Value ($)",
                    hovermode='x unified',
                    height=600
                )
                
                st.plotly_chart(fig_performance, use_container_width=True)
                
                col1, col2, col3 = st.columns(3)
                with col1:
                    buy_hold_return = (buy_hold_values.iloc[-1] / initial_investment - 1) * 100
                    st.metric("Buy & Hold Return", f"{buy_hold_return:.2f}%")
                
                with col2:
                    if rebalance_type != "No Rebalancing":
                        rebalanced_return = (rebalanced_values.iloc[-1] / initial_investment - 1) * 100
                        st.metric("Rebalanced Return", f"{rebalanced_return:.2f}%")
                
                with col3:
                    if include_benchmark and 'SPY' in prices.columns:
                        benchmark_return = (benchmark_values.iloc[-1] / initial_investment - 1) * 100
                        st.metric("S&P 500 Return", f"{benchmark_return:.2f}%")
            
            with tab2:
                st.header("Portfolio Weight Drift Analysis")
                
                if rebalance_type != "No Rebalancing":
                    buy_hold_weights = pd.DataFrame()
                    for i, asset in enumerate(selected_assets):
                        asset_values = buy_hold_values * (portfolio_prices[asset] / portfolio_prices.iloc[0] * target_weights[i])
                        buy_hold_weights[asset] = asset_values / buy_hold_values * 100
                    
                    fig_drift = go.Figure()
                    for asset in selected_assets:
                        fig_drift.add_trace(go.Scatter(
                            x=buy_hold_weights.index,
                            y=buy_hold_weights[asset],
                            name=asset,
                            mode='lines',
                            stackgroup='one'
                        ))
                    
                    for date in rebalance_dates:
                        fig_drift.add_vline(
                            x=date,
                            line_dash="dot",
                            line_color="black",
                            opacity=0.5,
                            annotation_text="Rebalance"
                        )
                    
                    fig_drift.update_layout(
                        title="Portfolio Weight Evolution (Buy & Hold)",
                        xaxis_title="Date",
                        yaxis_title="Weight (%)",
                        hovermode='x unified',
                        height=500
                    )
                    
                    st.plotly_chart(fig_drift, use_container_width=True)
                else:
                    st.info("Enable rebalancing to see drift analysis")
            
            with tab3:
                st.header("Rebalancing Events")
                
                if rebalance_dates:
                    st.write(f"Total rebalancing events: {len(rebalance_dates)}")
                    st.write(f"Estimated total transaction costs: ${len(rebalance_dates) * initial_investment * transaction_cost:.2f}")
                    
                    rebalance_df = pd.DataFrame({
                        'Date': rebalance_dates,
                        'Portfolio Value': [rebalanced_values.loc[date] for date in rebalance_dates]
                    })
                    st.dataframe(rebalance_df, use_container_width=True)
                else:
                    st.info("No rebalancing events with current strategy")
            
            with tab4:
                st.header("Performance Metrics Comparison")
                
                metrics_data = {}
                
                metrics_data['Buy & Hold'] = calculate_performance_metrics(buy_hold_values)
                
                if rebalance_type != "No Rebalancing":
                    metrics_data['Rebalanced'] = calculate_performance_metrics(rebalanced_values)
                
                if include_benchmark and 'SPY' in prices.columns:
                    metrics_data['S&P 500'] = calculate_performance_metrics(benchmark_values)
                
                metrics_df = pd.DataFrame(metrics_data).T
                metrics_df = metrics_df.round(2)
                
                st.dataframe(metrics_df, use_container_width=True)
                
                fig_metrics = go.Figure()
                
                x = list(metrics_df.index)
                metrics_to_plot = ['CAGR', 'Volatility', 'Sharpe Ratio']
                
                for metric in metrics_to_plot:
                    fig_metrics.add_trace(go.Bar(
                        name=metric,
                        x=x,
                        y=metrics_df[metric],
                        text=metrics_df[metric],
                        textposition='auto',
                    ))
                
                fig_metrics.update_layout(
                    title="Key Performance Metrics Comparison",
                    barmode='group',
                    xaxis_title="Strategy",
                    yaxis_title="Value",
                    height=400
                )
                
                st.plotly_chart(fig_metrics, use_container_width=True)
                
                st.subheader("Summary")
                if rebalance_type != "No Rebalancing":
                    performance_diff = rebalanced_return - buy_hold_return
                    if performance_diff > 0:
                        st.success(f"✅ Rebalancing improved returns by {performance_diff:.2f}%")
                    else:
                        st.warning(f"⚠️ Rebalancing reduced returns by {abs(performance_diff):.2f}%")
                    
                    st.info(f"💡 Transaction costs totaled ${len(rebalance_dates) * initial_investment * transaction_cost:.2f} ({len(rebalance_dates)} rebalances)")

else:
    st.info("👈 Please configure your backtesting parameters in the sidebar")
    
    st.subheader("How to use the Backtesting Engine:")
    st.markdown("""
    1. **Select Assets**: Choose from stocks, ETFs, or cryptocurrencies
    2. **Set Target Allocation**: Define your ideal portfolio weights
    3. **Choose Investment Settings**: Set initial amount and date range
    4. **Select Rebalancing Strategy**:
        - **Periodic**: Rebalance at fixed intervals
        - **Threshold-based**: Rebalance when weights drift too far
        - **No Rebalancing**: Buy and hold comparison
    5. **Analyze Results**: Compare performance across strategies
    """)
    
    st.subheader("Key Features:")
    st.markdown("""
    - 📊 Visual comparison of rebalanced vs buy-and-hold strategies
    - 📈 Benchmark comparison against S&P 500
    - 💰 Transaction cost consideration
    - 📉 Portfolio drift visualization
    - 📊 Comprehensive performance metrics (CAGR, Sharpe, Max Drawdown)
    """)