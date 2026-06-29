# Markowitz Portfolio Optimisation (MPT)

Mean-variance portfolio optimisation and efficient frontier analysis across a basket of equities, implemented in Python.


## Overview

This project applies Modern Portfolio Theory to a four-asset equity portfolio, finding the allocation of weights across assets that either minimises risk or maximises risk-adjusted return. The efficient frontier is approximated via Monte Carlo simulation and solved exactly using constrained optimisation through scipy. Both the Sharpe and Sortino ratios are used as measures of risk-adjusted performance, with the Sortino ratio penalising only downside volatility rather than total volatility.


## Features

Efficient frontier generated via 10,000 randomly weighted Monte Carlo portfolios

Analytical minimum volatility portfolio solved exactly using scipy SLSQP optimisation

Analytical maximum Sharpe ratio portfolio solved exactly using scipy SLSQP optimisation

Analytical maximum Sortino ratio portfolio solved exactly using scipy SLSQP optimisation

Correlation heatmap of asset returns to visualise diversification between holdings

Input validation, market data validation, and returns sanity checks throughout

Readable per-portfolio summaries including return, volatility, Sharpe, and Sortino


## Installation

Clone the repository and install the required dependencies:


```bash

git clone https://github.com/Efox1991/Markowitz-Portfolio-Optimization-Model.git

cd Markowitz-Portfolio-Optimization-Model

pip install -r requirements.txt

```

The project was developed and tested with the following package versions:

| Package    | Version |

|------------|---------|

| matplotlib | 3.8.4   |

| numpy      | 2.5.0   |

| pandas     | 3.0.3   |

| scipy      | 1.18.0  |

| seaborn    | 0.13.2  |

| yfinance   | 0.2.66  |

Compatible versions at or above those listed should work without issue.


## Usage


All user-facing configuration sits at the top of the script, below the function definitions:

pythontickers = \["AAPL", "NKE", "GOOGL", "AMZN"]  # assets to include in the portfolio

start\_date = '2024-01-01'                         # start of the historical window

end\_date   = '2026-01-01'                         # end of the historical window

risk\_free\_rate = 0.04                             # annualised risk-free rate

use\_fixed\_seed = True                             # set False for non-deterministic simulation

num\_portfolios = 10000			          # Number of random portfolios to sample

Swap in any tickers supported by Yahoo Finance. The risk-free rate should reflect the current annualised yield on a short-dated government bond — in practice a 3-month US T-bill yield for a USD portfolio, or SONIA for a GBP portfolio.


## Run the script directly:

bashpython markowitz\_mpt.py


## Output

Running the script produces the following:

Printed to console


**Validation confirmations at each stage of the pipeline**

Annualised covariance matrix

Initial fixed-weight portfolio variance and volatility

Per-portfolio summaries for all six optimal portfolios (three Monte Carlo, three analytical), each showing asset allocations, expected annual return, annual volatility, Sharpe ratio, and Sortino ratio


**Plots**

Correlation heatmap of daily asset returns

Efficient frontier scatter plot with all 10,000 Monte Carlo portfolios shown as a cloud, and the three analytical optimal portfolios marked as stars


## Limitations

Expected returns are estimated as the annualised mean of historical daily returns. This is a standard simplification but a noisy one — mean daily returns are highly sensitive to the chosen date window and are not reliable predictors of future performance. In a production setting, expected returns would typically come from a factor model or analyst forecasts rather than simple historical averaging.

The model assumes no short selling, meaning all weights are constrained to be non-negative. Transaction costs, taxes, and liquidity constraints are not modelled.

The covariance structure is assumed to be stationary over the chosen date range. In practice, asset correlations shift over time, particularly during periods of market stress, which can cause the realised frontier to differ significantly from the estimated one.
For assets with approximately symmetric return distributions the Sortino-optimal portfolio will converge toward the Sharpe-optimal portfolio.


## Dependencies

Python 3.x

yfinance

numpy

pandas

matplotlib

seaborn

scipy


## License

MIT

