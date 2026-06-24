import yfinance as yf                # fetches historical market data from Yahoo Finance
import numpy as np                   # numerical operations and linear algebra
import pandas as pd                  # DataFrame handling for tabular data
import matplotlib.pyplot as plt      # plotting
from datetime import datetime        # used in input validation to parse date strings
from scipy.optimize import minimize  # analytical portfolio optimisation
import seaborn as sns                # correlation heatmap

# Build a mock portfolio that has been trading for two years.
# Use MPT to minimise risk by optimising weights across assets.

use_fixed_seed = True

# Number of random portfolios to sample
num_portfolios = 10000

# Number of trading days in a year which is used to annualise statistics
trading_days = 252
 
# Minimum number of trading days required to compute a meaningful covariance matrix.
# Too few observations produce an unreliable covariance estimate.
minimum_trading_days = 60
 
# Daily return magnitude above which a value is flagged as a likely data error.
return_spike_threshold = 0.5

# =============================================================================
# Validation functions
# =============================================================================
 
def validate_inputs(tickers, start_date, end_date):
    """
    Check that the user-supplied configuration is correct before
    making any network calls or performing computation.
    Raises ValueError with a message if anything is incorrect.
    """
    if not tickers:
        raise ValueError("Tickers list is empty. Provide at least two assets.")
 
    if len(tickers) < 2:
        raise ValueError(
            f"At least two tickers are required to compute a covariance matrix. Got: {tickers}"
        )
 
    # Attempt to parse both dates to catch typos or wrong formats early
    try:
        start = datetime.strptime(start_date, '%Y-%m-%d')
    except ValueError:
        raise ValueError(f"start_date '{start_date}' is not in YYYY-MM-DD format.")
 
    try:
        end = datetime.strptime(end_date, '%Y-%m-%d')
    except ValueError:
        raise ValueError(f"end_date '{end_date}' is not in YYYY-MM-DD format.")
 
    if start >= end:
        raise ValueError(
            f"start_date ({start_date}) must be before end_date ({end_date})."
        )
 
    print("Input validation passed.")
 
 
def validate_market_data(df, tickers):
    """
    Check the raw price DataFrame returned by yfinance for common failure modes.
    Raises ValueError if the data is unusable.
    """
    if df.empty:
        raise ValueError(
            "Downloaded price data is empty. Check tickers and date range."
        )
 
    # yfinance silently drops tickers it cannot find — check all are present
    missing = [t for t in tickers if t not in df.columns]
    if missing:
        raise ValueError(
            f"The following tickers were not returned by yfinance: {missing}. "
            "Check for delisted or misspelled symbols."
        )
 
    # A column that is entirely NaN means yfinance returned a column header
    # but no actual data — treat it as missing
    all_nan = [col for col in df.columns if df[col].isna().all()]
    if all_nan:
        raise ValueError(
            f"The following tickers have no price data in the given date range: {all_nan}"
        )
 
    # Too few rows means the covariance matrix will be statistically unreliable
    if len(df) < minimum_trading_days:
        raise ValueError(
            f"Only {len(df)} trading days found. "
            f"At least {minimum_trading_days} are required for a reliable covariance estimate."
        )
 
    print(f"Market data validation passed: {len(df)} trading days, {len(df.columns)} assets.")
 
 
def validate_returns(returns):
    """
    Check the daily returns DataFrame for anomalies that would
    corrupt the covariance matrix or Monte Carlo results.
    Raises ValueError for structural problems; prints warnings for suspicious values.
    """
    # If any column is entirely NaN, something went wrong during pct_change/dropna
    all_nan = [col for col in returns.columns if returns[col].isna().all()]
    if all_nan:
        raise ValueError(
            f"Returns are all NaN for: {all_nan}. "
            "This can happen if a ticker has no overlapping data with the date range."
        )
 
    # Check for any remaining NaN values (partial gaps in the data)
    nan_counts = returns.isna().sum()
    cols_with_nans = nan_counts[nan_counts > 0]
    if not cols_with_nans.empty:
        print(f"Warning: NaN values remain in returns after dropna():\n{cols_with_nans}")
        print("This may indicate gaps in the price data for those assets.")
 
    # Flag single-day return spikes that likely indicate data errors rather than
    # real market moves — these would distort the covariance matrix significantly
    spikes = (returns.abs() > return_spike_threshold)
    spike_counts = spikes.sum()
    flagged = spike_counts[spike_counts > 0]
    if not flagged.empty:
        print(
            f"Warning: The following assets have daily returns exceeding "
            f"{return_spike_threshold * 100:.0f}% — possible data errors or stock splits:\n{flagged}"
        )
 
    print("Returns validation passed.")
 
 
def validate_weights(weights, num_assets):
    """
    Confirm that a weight vector is valid for use in portfolio calculations.
    Raises ValueError if weights are the wrong length, negative, or don't sum to unity.
    """
    if len(weights) != num_assets:
        raise ValueError(
            f"Weight vector has {len(weights)} elements but portfolio has {num_assets} assets."
        )
 
    if np.any(weights < 0):
        raise ValueError(
            f"All weights must be non-negative (no short selling). Got: {weights}"
        )
 
    # Use a tolerance rather than exact equality because floating point
    # arithmetic means weights won't sum to exactly 1.0
    if not np.isclose(np.sum(weights), 1.0, atol=1e-6):
        raise ValueError(
            f"Weights must sum to 1. Current sum: {np.sum(weights):.6f}"
        )
 
    print(f"Weight validation passed: sum = {np.sum(weights):.6f}")
 
 
def validate_covariance_matrix(cov):
    """
    Check that the covariance matrix is valid for use in the variance formula.
    A valid covariance matrix must be symmetric and positive semi-definite (PSD).
    PSD means all eigenvalues are >= 0; if any are negative, the matrix is
    malformed and sqrt(w^T Cov w) could produce NaN or imaginary values.
    Raises ValueError if the matrix fails the PSD check.
    """
    # Compute eigenvalues. For a real symmetric matrix values are all real
    eigenvalues = np.linalg.eigvalsh(cov.values)
 
    # Allow a small negative tolerance to account for floating point rounding
    if np.any(eigenvalues < -1e-8):
        raise ValueError(
            f"Covariance matrix is not positive semi-definite. "
            f"Minimum eigenvalue: {eigenvalues.min():.6e}. "
            "This can occur with very short date ranges or perfectly correlated assets."
        )
 
    print(f"Covariance matrix validation passed. Min eigenvalue: {eigenvalues.min():.6e}")
 
 
def validate_monte_carlo_results(portfolio_df):
    """
    Check the Monte Carlo output DataFrame for degenerate values that would
    corrupt the Sharpe ratio calculation or produce misleading plots.
    Raises ValueError for structural problems; prints warnings for edge cases.
    """
    # NaN in volatility or returns would silently corrupt idxmin/idxmax
    if portfolio_df['Volatility'].isna().any():
        raise ValueError("NaN values found in simulated portfolio volatilities.")
 
    if portfolio_df['Returns'].isna().any():
        raise ValueError("NaN values found in simulated portfolio returns.")
 
    # Negative volatility is mathematically impossible. Indicator of a bad covariance matrix
    if (portfolio_df['Volatility'] < 0).any():
        raise ValueError("Negative volatility values found. Check covariance matrix.")
 
    # Zero volatility would cause division by zero in the Sharpe ratio
    zero_vol = (portfolio_df['Volatility'] == 0).sum()
    if zero_vol > 0:
        print(
            f"Warning: {zero_vol} portfolios have zero volatility. "
            "These will be excluded from the Sharpe ratio calculation."
        )
 
    print(f"Monte Carlo results validation passed: {len(portfolio_df)} portfolios simulated.")
 
# =============================================================================
# Analytical optimisation functions
# =============================================================================
 
def portfolio_volatility(weights, cov_matrix):
    """
    Compute annualised portfolio volatility for a given weight vector.
    This is the objective function minimised by scipy.
    """
    variance = weights @ cov_matrix @ weights
    return np.sqrt(variance)
 
 
def portfolio_sharpe(weights, cov_matrix, expected_returns, risk_free_rate):
    """
    Compute the negative Sharpe ratio for a given weight vector.
    Negative because scipy.minimize minimises. Minimising negative Sharpe
    is equivalent to maximising Sharpe.
    """
    ret = np.dot(weights, expected_returns)
    vol = portfolio_volatility(weights, cov_matrix)
    return -(ret - risk_free_rate) / vol 

def portfolio_sortino(weights, returns, expected_returns, risk_free_rate):
    """
    Compute the negative Sortino ratio for a given weight vector.
    Negative because scipy.minimize minimises. Minimising negative Sortino
    is equivalent to maximising it.
    Uses the risk-free rate as the downside threshold.
    """
    ret = np.dot(weights, expected_returns)
    downside_dev = portfolio_downside_deviation(weights, returns, target_return=risk_free_rate / trading_days)
    
    return -(ret - risk_free_rate) / downside_dev

def portfolio_downside_deviation(weights, returns, target_return=None):
    """
    Annualised downside deviation.

    weights: portfolio weights
    returns: daily returns DataFrame
    target_return: daily target return
    """
    if target_return is None:
        target_return = risk_free_rate / trading_days
    
    portfolio_returns = returns.dot(weights)
    downside_returns = np.minimum(portfolio_returns - target_return, 0)
    downside_variance = np.mean(downside_returns ** 2)
    
    return np.sqrt(downside_variance * trading_days)
 
def find_min_volatility_portfolio(cov_matrix, num_assets):
    """
    Use scipy.optimize.minimize to find the exact minimum variance portfolio.
    This is an analytical solution, not a Monte Carlo approximation.
 
    Constraints:
      - weights sum to 1
      - each weight is between 0 and 1 (no short selling)
 
    Returns the optimised weight vector.
    """
    # Start from an equal-weight portfolio as the initial guess
    initial_guess = np.ones(num_assets) / num_assets
 
    # Constraint: weights must sum to 1
    constraints = {'type': 'eq', 'fun': lambda w: np.sum(w) - 1}
 
    # Bounds: each weight between 0 (no position) and 1 (fully invested in one asset)
    bounds = tuple((0, 1) for _ in range(num_assets))
 
    result = minimize(
        portfolio_volatility,
        initial_guess,
        args=(cov_matrix),
        method='SLSQP',        # Sequential Least Squares Programming
        bounds=bounds,
        constraints=constraints,
        options={'ftol': 1e-12, 'maxiter': 1000}
    )
 
    if not result.success:
        raise RuntimeError(f"Minimum volatility optimisation failed: {result.message}")
 
    return result.x
 
 
def find_max_sharpe_portfolio(cov_matrix, expected_returns, risk_free_rate, num_assets):
    """
    Use scipy.optimize.minimize to find the exact maximum Sharpe ratio portfolio.
 
    Constraints and bounds are the same as the minimum volatility.
    Returns the optimised weight vector.
    """
    initial_guess = np.ones(num_assets) / num_assets
    constraints = {'type': 'eq', 'fun': lambda w: np.sum(w) - 1}
    bounds = tuple((0, 1) for _ in range(num_assets))
 
    result = minimize(
        portfolio_sharpe,
        initial_guess,
        args=(cov_matrix, expected_returns, risk_free_rate),
        method='SLSQP',
        bounds=bounds,
        constraints=constraints,
        options={'ftol': 1e-12, 'maxiter': 1000}
    )
 
    if not result.success:
        raise RuntimeError(f"Maximum Sharpe optimisation failed: {result.message}")
 
    return result.x

def find_max_sortino_portfolio(returns, expected_returns, risk_free_rate, num_assets):
    """
    Use scipy.optimize.minimize to find the exact maximum Sortino ratio portfolio.
    The Sortino ratio penalises only downside volatility. The Sharpe ratio
    treats upside and downside variance equally.
    Constraints and bounds are the same as the other optimisation functions.
    Returns the optimised weight vector.
    """
    
    initial_guess = np.ones(num_assets) / num_assets
    constraints = {'type': 'eq', 'fun': lambda w: np.sum(w) - 1}
    bounds = tuple((0, 1) for _ in range(num_assets))

    result = minimize(portfolio_sortino, 
        initial_guess,
        args=(returns, expected_returns, risk_free_rate),
        method='SLSQP',
        bounds=bounds,
        constraints=constraints
    )

    if not result.success:
        raise RuntimeError(
            f"Maximum Sortino optimisation failed: {result.message}"
        )

    return result.x
 
# =============================================================================
# Output Summary Functions
# =============================================================================

def build_comparison_table(portfolio_entries, returns_df, cov_matrix_values, expected_returns_values, risk_free_rate):
    """
    Builds a summary DataFrame comparing return, volatility, Sharpe, and Sortino
    across all portfolios. Each entry in portfolio_entries is a (label, weights) tuple.
    """
    rows = []
    for label, weights in portfolio_entries:
        ret = np.dot(weights, expected_returns_values)
        vol = portfolio_volatility(weights, cov_matrix_values)
        sharpe = (ret - risk_free_rate) / vol
        sortino = (ret - risk_free_rate) / portfolio_downside_deviation(weights, returns_df)
        rows.append({
            'Portfolio'      : label,
            'Return (%)'     : round(ret * 100, 2),
            'Volatility (%)' : round(vol * 100, 2),
            'Sharpe'         : round(sharpe, 3),
            'Sortino'        : round(sortino, 3)
        })
    return pd.DataFrame(rows).set_index('Portfolio')

def build_weights_table(portfolio_entries, ticker_names):
    """
    Builds a DataFrame showing the percentage allocation to each asset
    across all portfolios. Each entry in portfolio_entries is a (label, weights) tuple.
    Tickers are sorted alphabetically so the column order is consistent across portfolios.
    """
    rows = []
    for label, weights in portfolio_entries:
        row = {'Portfolio': label}
        # zip ticker names with their corresponding weights, then sort by ticker name
        # so column order matches the sorted() used in the printed summaries
        for name, w in sorted(zip(ticker_names, weights), key=lambda x: x[0]):
            row[name] = f"{w * 100:.1f}%"
        rows.append(row)
    return pd.DataFrame(rows).set_index('Portfolio')

def extract_mc_weights(portfolio_row, ticker_names):
    """
    Pulls the weight vector out of a Monte Carlo portfolio Series row,
    in the same column order as ticker_names.
    """
    return np.array([portfolio_row[name + ' Weight'] for name in ticker_names])

# =============================================================================
# Configuration
# =============================================================================

# The six stocks in our portfolio
tickers = ['IBM', 'AAPL', 'NKE', 'NFLX', 'GOOGL', 'AMZN']
start_date = '2024-01-01'
end_date = '2026-01-01'

risk_free_rate = 0.04  # approximate current annualised risk-free rate
                       # in production environment this would be sourced from live data
                       # (e.g. 3-month US T-bill yield or SONIA for GBP portfolios)


# Validate inputs
validate_inputs(tickers, start_date, end_date)
 
# =============================================================================
# Data acquisition
# =============================================================================

# Download OHLCV data for all tickers in a single API call.
# yfinance returns a MultiIndex DataFrame: (price_type, ticker)
raw = yf.download(tickers, start=start_date, end=end_date, auto_adjust=True)

# Filter for closing price as that is all that is needed. 
# Result is a DataFrame with one column per ticker, indexed by date.
stock_data = raw['Close'] # Select close of the MultiIndex.

validate_market_data(stock_data, tickers)

print('Stock Data Close: \n{}'.format(stock_data.head()))

# =============================================================================
# Returns
# =============================================================================

# Compute daily percentage change for each stock:
# pct_change() = (price_today - price_yesterday) / price_yesterday
# The first row will be NaN because there is no prior day. Removed with .dropna().
returns = stock_data.pct_change().dropna()

validate_returns(returns)
print('Returns: \n{}'.format(returns.head()))

# =============================================================================
# Correlation heatmap
# =============================================================================
# The correlation matrix normalises covariance to the range [-1, 1].
# Easier to interpret diversification benefits.
# A value close to +1 means the two assets are highly correlated providing
# little diversification. A value close to 0 or negative means
# assets are largely independent or inversely correlated, reducing portfolio risk.
 
correlation_matrix = returns.corr()
 
fig, ax = plt.subplots(figsize=(7, 6))
sns.heatmap(
    correlation_matrix,
    annot=True,          # print the correlation value inside each cell
    fmt='.2f',           # two decimal places
    cmap='coolwarm',     # blue = negative correlation, red = positive
    vmin=-1, vmax=1,     # fix the colour scale to the full correlation range
    linewidths=0.5,
    ax=ax
)
ax.set_title('Asset Return Correlation Matrix')
plt.tight_layout()
plt.show()


# =============================================================================
# Initial fixed-weight portfolio
# =============================================================================
 
# Define an initial arbitrary set of portfolio weights.
# These must sum to 1: 15% IBM, 25% AAPL, 20% NKE, 15% NFLX, 05% GOOGL, 20% AMZN.
# Using a numpy array so matrix operations work correctly.
initial_weights = np.array([0.15, 0.25, 0.2, 0.15, 0.05, 0.2])

validate_weights(initial_weights, num_assets=len(tickers))

# Compute the daily portfolio return as the weighted sum of individual returns.
# dot product: portfolio_return = w1*r1 + w2*r2 + w3*r3 + w4*r4 for each day
portfolio_return_series = returns.dot(initial_weights)

print('Portfolio Returns: \n{}'.format(portfolio_return_series.head()))

# =============================================================================
# Covariance matrix
# =============================================================================

# cov() computes the pairwise covariance between each pair of assets' daily returns.
# A high positive value means two assets tend to move in the same direction.
# A negative value means they tend to move in opposite directions (useful for diversification).
# Multiply by trading_days to annualise (scale from daily to yearly).
covariance_matrix = returns.cov() * trading_days

validate_covariance_matrix(covariance_matrix)

print('Covariance Matrix: \n{}'.format(covariance_matrix))

# =============================================================================
# Initial portfolio variance and volatility
# =============================================================================
 
# Variance measures total portfolio risk.
# Formula: sigma^2 = w^T * Cov * w
# w^T is the transpose of the weights vector; @ is the matrix multiplication operator.
# Scales each asset's variance by its weight squared, and each pair's
# covariance by both their weights. Demonstrates how diversification reduces risk.
initial_variance = initial_weights @ covariance_matrix @ initial_weights

# Volatility is just the standard deviation: the square root of variance.
# It is in the same units as the returns (annualised).
# High volatility = high risk.
initial_volatility = np.sqrt(initial_variance)

print('Initial Portfolio Variance: ', initial_variance)
print('Initial Portfolio Volatility: ', initial_volatility)

# =============================================================================
# Annualised expected returns
# =============================================================================
 
# Mean daily return scaled to annual. This is a noisy estimate over short windows.
# In production, expected returns would typically come from a factor model or
# analyst forecasts rather than simple historical averaging.
expected_returns = returns.mean() * trading_days

# =============================================================================
# Efficient frontier via Monte Carlo
# =============================================================================

# Generate 10,000 random portfolios (each with a different set of weights) and plot them.
# The resulting scatter plot traces out the efficient frontier:
# the set of portfolios offering the highest return for a given level of risk.
#
# Formal problem:
#   Minimise:   w^T * Cov * w          (minimise portfolio variance)
#   Subject to: w^T * mu >= mu_target  (return must meet some target)
#               sum(w) = 1             (weights must sum to 1)
#               w_i >= 0               (no short selling)

mc_returns = []     # stores the expected return of each random portfolio
mc_volatility = []  # stores the volatility of each random portfolio
mc_weights = []     # stores the weight vector of each random portfolio
mc_sortino = []     # stores the sortino ratio of each random portfolio

# Number of assets
num_assets = len(stock_data.columns)

# Fix the random seed so results are reproducible across runs
if use_fixed_seed:
    np.random.seed(42)

for _ in range(num_portfolios):
    # Assign random positive numbers
    weights = np.random.random(num_assets)
    # Normalise so they sum to unity (i.e. full allocation across assets)
    weights = weights / np.sum(weights)
    mc_weights.append(weights)

    # Expected portfolio return: weighted average of individual asset returns
    port_return = np.dot(weights, expected_returns)
    mc_returns.append(port_return)
    
    # Sortino ratio for given weights
    sortino = (port_return - risk_free_rate) / portfolio_downside_deviation(weights, returns)
    mc_sortino.append(sortino)
    
    # Portfolio variance via the quadratic form w^T * Cov * w, computed using
    # element-wise scaling: multiply each row by its weight, each column by its weight,
    # then sum all elements. Equivalent to the matrix formula that works with DataFrames.
    variance = covariance_matrix.mul(weights, axis=0).mul(weights, axis=1).sum().sum()
    mc_volatility.append(np.sqrt(variance)) # Annualised prior

# Build a results DataFrame with returns, volatility, and each asset's weight
data = {'Returns': mc_returns, 'Volatility': mc_volatility, 'Sortino': mc_sortino}
for i, symbol in enumerate(stock_data.columns.tolist()):
    # Extract the weight for this asset from every simulated portfolio
    data[symbol + ' Weight'] = [w[i] for w in mc_weights]

portfolio_df = pd.DataFrame(data)

validate_monte_carlo_results(portfolio_df)

# =============================================================================
# Montecarlo based Sharpe and Minimum Volatility 
# =============================================================================

# Minimum volatility portfolio
min_vol_portfolio = portfolio_df.iloc[portfolio_df['Volatility'].idxmin()]

# Optimal portfolio: highest Sharpe ratio
# Sharpe Ratio = (Rp - Rf) / SDp
# Rp: portfolio return, Rf: risk-free rate, SDp: portfolio standard deviation
 
sharpe_ratios = (portfolio_df['Returns'] - risk_free_rate) / portfolio_df['Volatility']
optimal_portfolio_sharpe = portfolio_df.iloc[sharpe_ratios.idxmax()]

# Optimal portfolio: highest Sortino ratio
# Sharpe Ratio = (Rp - Rf) / SDp
# Rp: portfolio return, Rf: risk-free rate, SDp: portfolio downside standard deviation
optimal_portfolio_sortino = portfolio_df.iloc[portfolio_df['Sortino'].idxmax()]


# Extract weight vectors from Monte Carlo portfolio Series rows
mc_min_vol_weights      = extract_mc_weights(min_vol_portfolio, stock_data.columns.tolist())
mc_max_sharpe_weights   = extract_mc_weights(optimal_portfolio_sharpe, stock_data.columns.tolist())
mc_max_sortino_weights  = extract_mc_weights(optimal_portfolio_sortino, stock_data.columns.tolist())


# =============================================================================
# Analytical optimisation via scipy
# =============================================================================
# The Monte Carlo approach approximates the efficient frontier by random sampling.
# scipy.optimize.minimize finds the exact solution. Treating this as a
# constrained optimisation problem solved precisely rather than by random sampling.
 
# Exact minimum volatility portfolio
analytical_min_vol_weights = find_min_volatility_portfolio(
    covariance_matrix.values, num_assets
)
validate_weights(analytical_min_vol_weights, num_assets)
 
# Exact maximum Sharpe ratio portfolio
analytical_max_sharpe_weights = find_max_sharpe_portfolio(
    covariance_matrix.values, expected_returns.values, risk_free_rate, num_assets
)
validate_weights(analytical_max_sharpe_weights, num_assets)

# Exact maximum Sortino ratio portfolio
analytical_max_sortino_weights = find_max_sortino_portfolio(
    returns, expected_returns.values, risk_free_rate, num_assets
)
validate_weights(analytical_max_sortino_weights, num_assets)

# Assign portfolio entries so they can be tabulated for easier comparrison 
portfolio_entries = [
    ('MC Min Volatility',      mc_min_vol_weights),
    ('MC Max Sharpe',          mc_max_sharpe_weights),
    ('MC Max Sortino',         mc_max_sortino_weights),
    ('Analytical Min Vol',     analytical_min_vol_weights),
    ('Analytical Max Sharpe',  analytical_max_sharpe_weights),
    ('Analytical Max Sortino', analytical_max_sortino_weights),
]

ticker_names = stock_data.columns.tolist()

comparison_table = build_comparison_table(
    portfolio_entries,
    returns,
    covariance_matrix.values,
    expected_returns.values,
    risk_free_rate
)

weights_table = build_weights_table(portfolio_entries, ticker_names)

print("\n--- Portfolio Comparison ---")
print(comparison_table.to_string())
print("\n--- Asset Allocations ---")
print(weights_table.to_string())

# Compute volatility and return for the analytical portfolios so they can be
# marked on the efficient frontier plot
analytical_min_vol = portfolio_volatility(analytical_min_vol_weights, covariance_matrix.values)
analytical_min_vol_ret = np.dot(analytical_min_vol_weights, expected_returns.values)
 
analytical_max_sharpe_vol = portfolio_volatility(analytical_max_sharpe_weights, covariance_matrix.values)
analytical_max_sharpe_ret = np.dot(analytical_max_sharpe_weights, expected_returns.values)

analytical_max_sortino_vol = portfolio_volatility(analytical_max_sortino_weights, covariance_matrix.values)
analytical_max_sortino_ret = np.dot(analytical_max_sortino_weights, expected_returns.values)

# =============================================================================
# Final plot: efficient frontier with Monte Carlo cloud and analytical solutions
# =============================================================================
# The Monte Carlo scatter shows the shape of the feasible region (efficient frontier).
# The analytical points (stars) show exactly where the optimal portfolios sit —
# they should lie on or very close to the left edge of the cloud.
#
# Green + = analytical minimum volatility (lowest possible risk)
# Blue +  = analytical maximum Sharpe ratio (best risk-adjusted return)
# Red +   = analytical maximum Sortino ratio (best downside risk-adjusted return)
#
# Green Star = Monte Carlo minimum volatility (lowest possible risk)
# Blue Star  = Monte Carlo maximum Sharpe ratio (best risk-adjusted return)
# Red Star   = Monte Carlo maximum Sortino ratio (best downside risk-adjusted return)
 
fig, ax = plt.subplots(figsize=(10, 10))
 
# Monte Carlo cloud: yellow dots, semi-transparent
portfolio_df.plot.scatter(
    x='Volatility', y='Returns', marker='o', color='y',
    s=15, alpha=0.3, grid=True, ax=ax, label='Monte Carlo Portfolios'
)
 
# Analytical minimum volatility
ax.scatter(
    analytical_min_vol, analytical_min_vol_ret,
    color='g', marker='+', s=300, zorder=5, label='Analytical Min Volatility'
)
 
# Analytical maximum Sharpe ratio
ax.scatter(
    analytical_max_sharpe_vol, analytical_max_sharpe_ret,
    color='b', marker='+', s=300, zorder=5, label='Analytical Max Sharpe Ratio'
)

# Analytical maximum Sortino ratio
ax.scatter(
    analytical_max_sortino_vol, analytical_max_sortino_ret,
    color='r', marker='+', s=300, zorder=5, label='Analytical Max Sortino Ratio'
)

# Montecarlo minimum volatility
ax.scatter(
    min_vol_portfolio['Volatility'], min_vol_portfolio['Returns'],
    color='g', marker='*', s=300, zorder=5, label='Montecarlo Min Volatility'
)
 
# Montecarlo maximum Sharpe ratio
ax.scatter(
    optimal_portfolio_sharpe['Volatility'], optimal_portfolio_sharpe['Returns'],
    color='b', marker='*', s=300, zorder=5, label='Montecarlo Max Sharpe Ratio'
)

# Montecarlo maximum Sortino ratio
ax.scatter(
    optimal_portfolio_sortino['Volatility'], optimal_portfolio_sortino['Returns'],
    color='r', marker='*', s=200, zorder=5, label='Montecarlo Max Sortino Ratio'
)
 
ax.set_xlabel('Risk (Volatility)')
ax.set_ylabel('Expected Returns')
ax.set_title('Efficient Frontier — Markowitz Portfolio Optimisation')
ax.legend()
plt.show()
