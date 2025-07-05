import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime, timedelta
from scipy.optimize import minimize
import warnings
warnings.filterwarnings('ignore')

st.set_page_config(
    page_title="Portfolio Optimizer",
    page_icon="📊",
    layout="wide"
)

st.title("📊 Portfolio Optimizer")
st.markdown("Build and analyze your investment portfolio using Modern Portfolio Theory")

POPULAR_ASSETS = {
    "Crypto": ["BTC-USD", "ETH-USD", "BNB-USD", "ADA-USD", "SOL-USD"],
    "Stocks": ["AAPL", "GOOGL", "MSFT", "AMZN", "TSLA", "META", "NVDA"],
    "ETFs": ["SPY", "QQQ", "GLD", "SLV", "USO"],
    "Commodities": ["GC=F", "SI=F", "CL=F"]  # Gold, Silver, Oil futures
}

def fetch_data(tickers, start_date, end_date):
    """Fetch historical data for given tickers"""
    try:
        data = yf.download(tickers, start=start_date, end=end_date)['Adj Close']
        if isinstance(data, pd.Series):
            data = data.to_frame()
        return data
    except Exception as e:
        st.error(f"Error fetching data: {str(e)}")
        return None

def calculate_portfolio_stats(returns, weights):
    """Calculate portfolio return and risk"""
    portfolio_return = np.sum(returns.mean() * weights) * 252
    portfolio_std = np.sqrt(np.dot(weights.T, np.dot(returns.cov() * 252, weights)))
    return portfolio_return, portfolio_std

def calculate_sharpe_ratio(returns, weights, risk_free_rate=0.02):
    """Calculate Sharpe ratio"""
    portfolio_return, portfolio_std = calculate_portfolio_stats(returns, weights)
    sharpe = (portfolio_return - risk_free_rate) / portfolio_std
    return sharpe

def generate_efficient_frontier(returns, n_portfolios=10000):
    """Generate efficient frontier using Monte Carlo simulation"""
    n_assets = len(returns.columns)
    results = np.zeros((3, n_portfolios))
    weights_record = []
    
    for i in range(n_portfolios):
        weights = np.random.random(n_assets)
        weights /= np.sum(weights)
        weights_record.append(weights)
        
        portfolio_return, portfolio_std = calculate_portfolio_stats(returns, weights)
        results[0, i] = portfolio_return
        results[1, i] = portfolio_std
        results[2, i] = calculate_sharpe_ratio(returns, weights)
    
    return results, weights_record

def optimize_portfolio(returns, target='sharpe'):
    """Optimize portfolio weights"""
    n_assets = len(returns.columns)
    args = (returns,)
    constraints = ({'type': 'eq', 'fun': lambda x: np.sum(x) - 1})
    bounds = tuple((0, 1) for asset in range(n_assets))
    initial_guess = n_assets * [1. / n_assets]
    
    if target == 'sharpe':
        def neg_sharpe(weights, returns):
            return -calculate_sharpe_ratio(returns, weights)
        result = minimize(neg_sharpe, initial_guess, args=args,
                         method='SLSQP', bounds=bounds, constraints=constraints)
    else:  # minimize variance
        def portfolio_variance(weights, returns):
            return np.dot(weights.T, np.dot(returns.cov() * 252, weights))
        result = minimize(portfolio_variance, initial_guess, args=args,
                         method='SLSQP', bounds=bounds, constraints=constraints)
    
    return result.x

with st.sidebar:
    st.header("Portfolio Settings")
    
    st.subheader("1. Select Assets")
    asset_categories = st.multiselect(
        "Choose categories:",
        list(POPULAR_ASSETS.keys()),
        default=["Stocks"]
    )
    
    available_assets = []
    for category in asset_categories:
        available_assets.extend(POPULAR_ASSETS[category])
    
    selected_assets = st.multiselect(
        "Select assets for your portfolio:",
        available_assets,
        default=available_assets[:3] if available_assets else []
    )
    
    custom_ticker = st.text_input("Add custom ticker (e.g., IBM):")
    if custom_ticker and st.button("Add Custom Ticker"):
        if custom_ticker.upper() not in selected_assets:
            selected_assets.append(custom_ticker.upper())
    
    st.subheader("2. Date Range")
    col1, col2 = st.columns(2)
    with col1:
        start_date = st.date_input(
            "Start Date",
            datetime.now() - timedelta(days=365),
            max_value=datetime.now()
        )
    with col2:
        end_date = st.date_input(
            "End Date",
            datetime.now(),
            max_value=datetime.now()
        )
    
    if start_date >= end_date:
        st.error("Start date must be before end date!")

if selected_assets and start_date < end_date:
    data = fetch_data(selected_assets, start_date, end_date)
    
    if data is not None and not data.empty:
        returns = data.pct_change().dropna()
        
        tab1, tab2, tab3, tab4 = st.tabs(["📊 Overview", "📈 Returns Distribution", "⚖️ Portfolio Weights", "🎯 Efficient Frontier"])
        
        with tab1:
            st.header("Portfolio Overview")
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.subheader("Price Performance")
                normalized_data = data / data.iloc[0] * 100
                fig_price = go.Figure()
                for column in normalized_data.columns:
                    fig_price.add_trace(go.Scatter(
                        x=normalized_data.index,
                        y=normalized_data[column],
                        name=column,
                        mode='lines'
                    ))
                fig_price.update_layout(
                    title="Normalized Price Performance (Base = 100)",
                    xaxis_title="Date",
                    yaxis_title="Normalized Price",
                    hovermode='x unified'
                )
                st.plotly_chart(fig_price, use_container_width=True)
            
            with col2:
                st.subheader("Asset Statistics")
                annual_returns = returns.mean() * 252
                annual_std = returns.std() * np.sqrt(252)
                sharpe_ratios = annual_returns / annual_std
                
                stats_df = pd.DataFrame({
                    'Annual Return (%)': annual_returns * 100,
                    'Annual Volatility (%)': annual_std * 100,
                    'Sharpe Ratio': sharpe_ratios
                }).round(2)
                
                st.dataframe(stats_df, use_container_width=True)
            
            st.subheader("Correlation Matrix")
            corr_matrix = returns.corr()
            fig_corr = px.imshow(
                corr_matrix,
                text_auto=True,
                aspect="auto",
                color_continuous_scale='RdBu_r',
                zmin=-1,
                zmax=1
            )
            fig_corr.update_layout(title="Asset Correlation Matrix")
            st.plotly_chart(fig_corr, use_container_width=True)
        
        with tab2:
            st.header("Feature 1: Returns Distribution")
            
            selected_asset_hist = st.selectbox(
                "Select asset to view returns distribution:",
                selected_assets
            )
            
            asset_returns = returns[selected_asset_hist]
            
            col1, col2 = st.columns([2, 1])
            
            with col1:
                fig_hist = go.Figure()
                fig_hist.add_trace(go.Histogram(
                    x=asset_returns * 100,
                    nbinsx=50,
                    name=selected_asset_hist,
                    marker_color='blue',
                    opacity=0.7
                ))
                
                mean_return = asset_returns.mean() * 100
                std_return = asset_returns.std() * 100
                
                fig_hist.add_vline(
                    x=mean_return,
                    line_dash="dash",
                    line_color="red",
                    annotation_text=f"Mean: {mean_return:.2f}%"
                )
                
                fig_hist.update_layout(
                    title=f"Daily Returns Distribution - {selected_asset_hist}",
                    xaxis_title="Daily Return (%)",
                    yaxis_title="Frequency",
                    showlegend=False
                )
                st.plotly_chart(fig_hist, use_container_width=True)
            
            with col2:
                st.metric("Expected Daily Return", f"{mean_return:.3f}%")
                st.metric("Daily Volatility", f"{std_return:.3f}%")
                st.metric("Annualized Return", f"{mean_return * 252:.2f}%")
                st.metric("Annualized Volatility", f"{std_return * np.sqrt(252):.2f}%")
        
        with tab3:
            st.header("Feature 2: Portfolio Analysis")
            
            st.subheader("Set Portfolio Weights")
            
            weights = {}
            remaining_weight = 100.0
            
            for i, asset in enumerate(selected_assets):
                if i == len(selected_assets) - 1:
                    weights[asset] = remaining_weight
                    st.slider(
                        f"{asset} (%)",
                        0.0,
                        100.0,
                        remaining_weight,
                        disabled=True,
                        key=f"weight_{asset}"
                    )
                else:
                    weight = st.slider(
                        f"{asset} (%)",
                        0.0,
                        remaining_weight,
                        remaining_weight / (len(selected_assets) - i),
                        key=f"weight_{asset}"
                    )
                    weights[asset] = weight
                    remaining_weight -= weight
            
            weights_array = np.array([weights[asset] / 100 for asset in selected_assets])
            
            if np.sum(weights_array) == 1.0:
                portfolio_return, portfolio_std = calculate_portfolio_stats(returns, weights_array)
                portfolio_sharpe = calculate_sharpe_ratio(returns, weights_array)
                
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Expected Annual Return", f"{portfolio_return * 100:.2f}%")
                with col2:
                    st.metric("Annual Risk (Std Dev)", f"{portfolio_std * 100:.2f}%")
                with col3:
                    st.metric("Sharpe Ratio", f"{portfolio_sharpe:.3f}")
                
                st.subheader("Portfolio Composition")
                fig_pie = px.pie(
                    values=list(weights.values()),
                    names=list(weights.keys()),
                    title="Portfolio Allocation"
                )
                st.plotly_chart(fig_pie, use_container_width=True)
        
        with tab4:
            st.header("Feature 3: Efficient Frontier")
            
            with st.spinner("Generating efficient frontier..."):
                results, weights_record = generate_efficient_frontier(returns)
                
                max_sharpe_idx = np.argmax(results[2])
                max_sharpe_return = results[0, max_sharpe_idx]
                max_sharpe_std = results[1, max_sharpe_idx]
                max_sharpe_weights = weights_record[max_sharpe_idx]
                
                min_var_idx = np.argmin(results[1])
                min_var_return = results[0, min_var_idx]
                min_var_std = results[1, min_var_idx]
                min_var_weights = weights_record[min_var_idx]
            
            fig_frontier = go.Figure()
            
            fig_frontier.add_trace(go.Scatter(
                x=results[1] * 100,
                y=results[0] * 100,
                mode='markers',
                marker=dict(
                    size=3,
                    color=results[2],
                    colorscale='Viridis',
                    showscale=True,
                    colorbar=dict(title="Sharpe Ratio")
                ),
                text=[f"Sharpe: {s:.3f}" for s in results[2]],
                hovertemplate='Risk: %{x:.2f}%<br>Return: %{y:.2f}%<br>%{text}<extra></extra>',
                name='Portfolios'
            ))
            
            fig_frontier.add_trace(go.Scatter(
                x=[max_sharpe_std * 100],
                y=[max_sharpe_return * 100],
                mode='markers',
                marker=dict(size=15, color='red', symbol='star'),
                name='Max Sharpe Ratio',
                text=[f"Sharpe: {results[2, max_sharpe_idx]:.3f}"]
            ))
            
            fig_frontier.add_trace(go.Scatter(
                x=[min_var_std * 100],
                y=[min_var_return * 100],
                mode='markers',
                marker=dict(size=15, color='green', symbol='diamond'),
                name='Minimum Variance',
                text=[f"Sharpe: {results[2, min_var_idx]:.3f}"]
            ))
            
            fig_frontier.update_layout(
                title="Efficient Frontier",
                xaxis_title="Risk (Standard Deviation %)",
                yaxis_title="Expected Return (%)",
                hovermode='closest'
            )
            
            st.plotly_chart(fig_frontier, use_container_width=True)
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.subheader("Maximum Sharpe Ratio Portfolio")
                sharpe_df = pd.DataFrame({
                    'Asset': selected_assets,
                    'Weight (%)': max_sharpe_weights * 100
                }).round(2)
                st.dataframe(sharpe_df, use_container_width=True)
                st.metric("Expected Return", f"{max_sharpe_return * 100:.2f}%")
                st.metric("Risk", f"{max_sharpe_std * 100:.2f}%")
                st.metric("Sharpe Ratio", f"{results[2, max_sharpe_idx]:.3f}")
            
            with col2:
                st.subheader("Minimum Variance Portfolio")
                var_df = pd.DataFrame({
                    'Asset': selected_assets,
                    'Weight (%)': min_var_weights * 100
                }).round(2)
                st.dataframe(var_df, use_container_width=True)
                st.metric("Expected Return", f"{min_var_return * 100:.2f}%")
                st.metric("Risk", f"{min_var_std * 100:.2f}%")
                st.metric("Sharpe Ratio", f"{results[2, min_var_idx]:.3f}")

else:
    st.info("👈 Please select assets and date range from the sidebar to begin")
    
    st.subheader("How to use this app:")
    st.markdown("""
    1. **Select Assets**: Choose from popular cryptocurrencies, stocks, ETFs, and commodities
    2. **Set Date Range**: Pick your analysis period
    3. **Explore Features**:
        - **Overview**: See price performance, statistics, and correlations
        - **Returns Distribution**: Analyze individual asset return patterns
        - **Portfolio Weights**: Set custom weights and see portfolio metrics
        - **Efficient Frontier**: Find optimal portfolio allocations
    """)
    
    st.subheader("Available Assets:")
    for category, assets in POPULAR_ASSETS.items():
        st.write(f"**{category}**: {', '.join(assets[:5])}...")