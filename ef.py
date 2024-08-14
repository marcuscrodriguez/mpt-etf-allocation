import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize
from datetime import datetime, timedelta
from sklearn.linear_model import LinearRegression
#by: Marcus C. Rodriguez, ETF Asset Allocation MPT 
st.title("Modern Portfolio Theory: ")

# Sidebar for user input
st.sidebar.header("Select Parameters")

# Fixed ETF list and simplified categories
etfs = ["BIL", "IBIT", "LQD", "TLT", "HYG", "SPY", "QQQ", "IWM", "EFA", "SCZ", "DBC", "IAU", "XLRE"]
categories_pie = {
    "Cash Equivalents": ["BIL"],
    "Crypto": ["IBIT"],
    "Fixed Income": ["LQD", "TLT", "HYG"],
    "Equities": ["SPY", "QQQ", "IWM", "EFA", "SCZ"],
    "Commodities": ["DBC", "IAU"],
    "Real Estate": ["XLRE"],
}
categories = {
    "Cash Equivalents - T-Bills": ["BIL"],
    "Crypto - Bitcoin": ["IBIT"],
    "Fixed Income - US$ Corporate Investment Grade": ["LQD"],
    "Fixed Income - US Government Bonds": ["TLT"],
    "Fixed Income - US$ High Yield Corporate": ["HYG"],    
    "Domestic Equity S&P500 - Large Cap": ["SPY"],
    "Domestic Equity NASDAQ 100 - Large/Mid Cap": ["QQQ"],
    "Domestic Equity Russell 2000 - Small Cap": ["IWM"],
    "Foreign Equity MSCI EAFE - Large/Mid Cap": ["EFA"],
    "Foreign Equity MSCI EAFE - Small Cap": ["SCZ"],
    "Commodities - Ag, Oil, Metals": ["DBC"],
    "Commodities - Gold": ["IAU"],   
    "Real Estate - REITs": ["XLRE"],
}

# Input for risk-free rate
risk_free_rate = st.sidebar.number_input("Enter Risk-Free Rate (in %)", min_value=0.0, value=5.48) / 100

# Automatically calculate dates for the last 10 years
end_date = datetime.today().strftime('%Y-%m-%d')
start_date = (datetime.today() - timedelta(days=365 * 10)).strftime('%Y-%m-%d')

# Get data
data = yf.download(etfs, start=start_date, end=end_date)['Adj Close']

# Calculate log returns
returns = np.log(data / data.shift(1))

# Calculate expected returns, covariance matrix
mean_returns = returns.mean() * 252  # Annualize the returns
cov_matrix = returns.cov() * 252  # Annualize the covariance matrix

# Calculate variance of each ETF
etf_variances = np.diag(cov_matrix)

# Define portfolio performance metrics
def portfolio_performance(weights, mean_returns, cov_matrix):
    returns = np.dot(weights, mean_returns)
    std_dev = np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights)))
    sharpe_ratio = (returns - risk_free_rate) / std_dev
    return returns, std_dev, sharpe_ratio

# Define the negative Sharpe ratio function to minimize
def negative_sharpe_ratio(weights, mean_returns, cov_matrix, risk_free_rate):
    p_returns, p_std_dev, _ = portfolio_performance(weights, mean_returns, cov_matrix)
    return -(p_returns - risk_free_rate) / p_std_dev

# Define constraints and bounds
constraints = ({'type': 'eq', 'fun': lambda x: np.sum(x) - 1})  # Sum of weights = 1
bounds = tuple((0, 1) for _ in range(len(etfs)))  # Allow weights to be between 0 and 1

# Efficient frontier calculation
def calculate_efficient_frontier(mean_returns, cov_matrix, num_portfolios=100):
    results = np.zeros((3, num_portfolios))
    weights_record = []

    for i in range(num_portfolios):
        weights = np.random.random(len(etfs))
        weights /= np.sum(weights)
        weights_record.append(weights)

        portfolio_return, portfolio_std_dev, portfolio_sharpe = portfolio_performance(weights, mean_returns, cov_matrix)
        results[0, i] = portfolio_std_dev
        results[1, i] = portfolio_return
        results[2, i] = portfolio_sharpe

    return results, weights_record

# Calculate efficient frontier
results, weights_record = calculate_efficient_frontier(mean_returns, cov_matrix)

# Sort the results by volatility (ascending order)
sorted_indices = np.argsort(results[0])
sorted_volatility = results[0][sorted_indices]
sorted_returns = results[1][sorted_indices]
sorted_sharpe_ratios = results[2][sorted_indices]
sorted_weights = [weights_record[i] for i in sorted_indices]

# Regression line calculation
x = sorted_volatility.reshape(-1, 1)  # Volatility (Std. Deviation)
y = sorted_returns  # Expected Returns
regressor = LinearRegression()
regressor.fit(x, y)
regression_line = regressor.predict(x)

# Optimize for maximum Sharpe ratio (tangency portfolio)
opt_sharpe = minimize(negative_sharpe_ratio, len(etfs) * [1./len(etfs)], args=(mean_returns, cov_matrix, risk_free_rate),
                      method='SLSQP', bounds=bounds, constraints=constraints)
optimal_weights = opt_sharpe.x
optimal_return, optimal_volatility, optimal_sharpe_ratio = portfolio_performance(optimal_weights, mean_returns, cov_matrix)

# Slider to adjust allocation on the efficient frontier
st.sidebar.header("Adjust Allocation")
allocation_slider = st.sidebar.slider("Select the desired portfolio position (0 = min risk, 1 = max return)", 0.0, 1.0, 0.0)
allocation_index = int(allocation_slider * (len(regression_line) - 1))
st.sidebar.header("ETF Links", divider="rainbow")
url1 = "https://www.ssga.com/us/en/intermediary/etfs/funds/spdr-bloomberg-1-3-month-t-bill-etf-bil"
url2 = "https://www.ishares.com/us/products/333011/ishares-bitcoin-trust"
url3 = "https://www.ishares.com/us/products/239566/LQD"
url4 = "https://www.ishares.com/us/products/239454/ishares-20-year-treasury-bond-etf"
url5 = "https://www.ishares.com/us/products/239565/ishares-iboxx-high-yield-corporate-bond-etf"
url6 = "https://www.ssga.com/us/en/intermediary/etfs/funds/spdr-sp-500-etf-trust-spy"
url7 = "https://www.invesco.com/qqq-etf/en/about.html"
url8 = "https://www.ishares.com/us/products/239710/ishares-russell-2000-etf"
url9 = "https://www.ishares.com/us/products/239623/ishares-msci-eafe-etf"
url10 = "https://www.ishares.com/us/products/239627/ishares-msci-eafe-smallcap-etf"
url11 = "https://www.ishares.com/us/products/239561/ishares-gold-trust-fund"
url12 = "https://www.invesco.com/us/financial-products/etfs/product-detail?audienceType=Investor&ticker=DBC"
url13 = "https://www.ssga.com/us/en/intermediary/etfs/funds/the-real-estate-select-sector-spdr-fund-xlre"
st.sidebar.write("[BIL](%s)" % url1)
st.sidebar.write("[IBIT](%s)" % url2)
st.sidebar.write("[LQD](%s)" % url3)
st.sidebar.write("[TLT](%s)" % url4)
st.sidebar.write("[HYG](%s)" % url5)
st.sidebar.write("[SPY](%s)" % url6)
st.sidebar.write("[QQQ](%s)" % url7)
st.sidebar.write("[IWM](%s)" % url8)
st.sidebar.write("[EFA](%s)" % url9)
st.sidebar.write("[SCZ](%s)" % url10)
st.sidebar.write("[IAU](%s)" % url11)
st.sidebar.write("[DBC](%s)" % url12)
st.sidebar.write("[XLRE](%s)" % url13)

# Show selected portfolio's details
selected_portfolio_return = regression_line[allocation_index]
selected_portfolio_volatility = x[allocation_index][0]
selected_portfolio_sharpe = sorted_sharpe_ratios[allocation_index]
selected_portfolio_weights = sorted_weights[allocation_index]
selected_portfolio_weights_df = pd.DataFrame(selected_portfolio_weights, index=etfs, columns=['Weight'])

# Plot efficient frontier with regression line and visual point
st.header("Efficient Frontier | ETF Asset Allocation", divider="rainbow")
fig, ax = plt.subplots()
ax.plot(x, regression_line, color='blue', linestyle='-', label="Regression Line")
ax.scatter([selected_portfolio_volatility], [selected_portfolio_return], color='red', label="Selected Portfolio")
ax.set_xlabel('Volatility (Std. Deviation)')
ax.set_ylabel('Expected Returns')
ax.set_title('Efficient Frontier')

st.pyplot(fig)

# Display selected portfolio details
st.header("Selected Portfolio Details", divider="rainbow")
st.write(f"Expected Return: {selected_portfolio_return:.2%}")
st.write(f"Volatility (Risk): {selected_portfolio_volatility:.2%}")
st.write(f"Sharpe Ratio: {selected_portfolio_sharpe:.2f}")
st.dataframe(selected_portfolio_weights_df.T)

# Aggregate weights by simplified category
simplified_category_weights = {
    "Equities": 0.0,
    "Cash Equivalents": 0.0,
    "Fixed Income": 0.0,
    "Commodities": 0.0,
    "Crypto": 0.0,
    "Real Estate": 0.0
}

for category, assets in categories_pie.items():
    weight_sum = sum(selected_portfolio_weights_df.loc[assets].sum())
    simplified_category_weights[category] += weight_sum

# Plot a pie chart for the selected portfolio allocation by simplified category
st.header("Portfolio Asset Allocation", divider="rainbow")
fig2, ax2 = plt.subplots()
ax2.pie(simplified_category_weights.values(), labels=simplified_category_weights.keys(), autopct='%1.1f%%', startangle=90)
ax2.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.
st.pyplot(fig2)

# Create and display a table with the broader categories, ETFs, percentage allocations, expected returns, variance, and Sharpe ratio
st.header("Detailed Portfolio Allocation", divider="rainbow")
allocation_table = []
for category, etf_list in categories.items():
    for etf in etf_list:
        allocation_table.append([
            category,
            etf,
            f"{selected_portfolio_weights_df.loc[etf].values[0] * 100:.2f}%",
            f"{mean_returns.loc[etf] * 100:.2f}%",
            f"{etf_variances[etfs.index(etf)] * 100:.2f}%",
            f"{(mean_returns.loc[etf] - risk_free_rate) / etf_variances[etfs.index(etf)]:.2f}"
        ])

allocation_df = pd.DataFrame(allocation_table, columns=["Category", "ETF", "Percentage", "Expected Return (10Y)", "Variance (10Y)", "Sharpe Ratio"])
st.dataframe(allocation_df)

# Calculate and display the correlation matrix of the selected ETFs
st.header("Correlation Matrix of Selected ETFs", divider="rainbow")
correlation_matrix = returns.corr()
st.dataframe(correlation_matrix)

