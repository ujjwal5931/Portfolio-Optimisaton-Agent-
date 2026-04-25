from dataclasses import dataclass
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.optimize import minimize

from utils import get_logger


logger = get_logger(__name__)


@dataclass
class PortfolioResult:
    """
    Container for the optimisation output.

    This keeps the FastAPI layer clean and makes it easy to extend the
    optimisation logic without touching the HTTP handler.
    """

    tickers: List[str]
    weights: np.ndarray
    expected_return: float
    volatility: float
    sharpe_ratio: float
    risk_profile: str
    efficient_frontier_path: str


def compute_statistics(prices: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
    """
    Given a DataFrame of historical prices, compute annualised mean returns
    and the annualised covariance matrix using daily log-returns.
    """
    log_returns = np.log(prices / prices.shift(1)).dropna()
    mean_daily = log_returns.mean().values
    cov_daily = log_returns.cov().values

    trading_days = 252
    mean_annual = mean_daily * trading_days
    cov_annual = cov_daily * trading_days

    return mean_annual, cov_annual


def _portfolio_performance(
    weights: np.ndarray,
    mean_returns: np.ndarray,
    cov_matrix: np.ndarray,
    risk_free_rate: float,
) -> Tuple[float, float, float]:
    """
    Compute portfolio expected return, volatility, and Sharpe ratio for the
    given weights and asset statistics.
    """
    port_return = float(np.dot(weights, mean_returns))
    port_vol = float(np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights))))
    if port_vol == 0:
        sharpe = 0.0
    else:
        sharpe = (port_return - risk_free_rate) / port_vol
    return port_return, port_vol, sharpe


def _objective_factory(
    mean_returns: np.ndarray,
    cov_matrix: np.ndarray,
    risk_free_rate: float,
    risk_profile: str,
):
    """
    Build the optimisation objective based on the risk profile mapping.

    - 'low'      -> minimise volatility.
    - 'moderate' -> maximise Sharpe ratio (implemented as minimising negative Sharpe).
    - 'high'     -> maximise expected return (implemented as minimising negative return).
    """

    def obj(weights: np.ndarray) -> float:
        ret, vol, sharpe = _portfolio_performance(
            weights, mean_returns, cov_matrix, risk_free_rate
        )
        if risk_profile == "low":
            return vol
        if risk_profile == "high":
            return -ret
        # default to Sharpe-maximisation
        return -sharpe

    return obj


def _monte_carlo_simulation(
    num_portfolios: int,
    mean_returns: np.ndarray,
    cov_matrix: np.ndarray,
    risk_free_rate: float,
) -> pd.DataFrame:
    """
    Run a Monte Carlo simulation over random portfolios to approximate
    the efficient frontier and provide visual intuition.

    Returns a DataFrame with columns: ['return', 'volatility', 'sharpe'] and
    one column per asset weight.
    """
    num_assets = len(mean_returns)
    results = {
        "return": [],
        "volatility": [],
        "sharpe": [],
    }
    for i in range(num_assets):
        results[f"w_{i}"] = []

    for _ in range(num_portfolios):
        weights = np.random.random(num_assets)
        weights /= np.sum(weights)
        ret, vol, sharpe = _portfolio_performance(
            weights, mean_returns, cov_matrix, risk_free_rate
        )
        results["return"].append(ret)
        results["volatility"].append(vol)
        results["sharpe"].append(sharpe)
        for i in range(num_assets):
            results[f"w_{i}"].append(weights[i])

    return pd.DataFrame(results)


def _generate_efficient_frontier_plot(
    mc_results: pd.DataFrame,
    optimal_result: PortfolioResult,
    output_path: str,
) -> None:
    """
    Generate and save an efficient frontier scatter plot along with the
    selected optimal portfolio highlighted in a different colour.
    """
    plt.figure(figsize=(10, 6))
    plt.scatter(
        mc_results["volatility"],
        mc_results["return"],
        c=mc_results["sharpe"],
        cmap="viridis",
        alpha=0.4,
        label="Random Portfolios",
    )

    plt.colorbar(label="Sharpe Ratio")

    plt.scatter(
        optimal_result.volatility,
        optimal_result.expected_return,
        color="red",
        marker="*",
        s=300,
        label="Optimised Portfolio",
    )

    plt.title("Efficient Frontier with Optimised Portfolio")
    plt.xlabel("Annualised Volatility")
    plt.ylabel("Annualised Expected Return")
    plt.legend()
    plt.grid(True, linestyle="--", alpha=0.5)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()

    logger.info("Efficient frontier plot saved to %s", output_path)


def optimise_portfolio(
    prices: pd.DataFrame,
    risk_profile: str,
    risk_free_rate: float = 0.06,
    num_portfolios: int = 5000,
    frontier_path: str = "efficient_frontier.png",
) -> PortfolioResult:
    """
    Core Markowitz mean-variance optimisation entry point.

    - Computes annualised statistics from historical prices.
    - Runs a Monte Carlo simulation to approximate the efficient frontier.
    - Uses SciPy's SLSQP optimiser with constraints:
        * Sum of weights == 1
        * No short-selling (0 <= weight <= 1 for each asset)
    - Selects the objective based on the risk_profile mapping.
    - Returns a PortfolioResult object plus generates a frontier PNG.
    """
    tickers = list(prices.columns)
    mean_returns, cov_matrix = compute_statistics(prices)

    logger.info(
        "Running Monte Carlo simulation for %d portfolios.", num_portfolios
    )
    mc_results = _monte_carlo_simulation(
        num_portfolios, mean_returns, cov_matrix, risk_free_rate
    )

    num_assets = len(tickers)
    bounds = tuple((0.0, 1.0) for _ in range(num_assets))
    constraints = (
        {"type": "eq", "fun": lambda w: np.sum(w) - 1.0},
    )
    x0 = np.array(num_assets * [1.0 / num_assets])

    objective = _objective_factory(mean_returns, cov_matrix, risk_free_rate, risk_profile)

    logger.info("Starting Markowitz optimisation with risk profile '%s'.", risk_profile)
    optimisation = minimize(
        objective,
        x0=x0,
        method="SLSQP",
        bounds=bounds,
        constraints=constraints,
        options={"maxiter": 500, "ftol": 1e-9},
    )

    if not optimisation.success:
        logger.warning(
            "Optimisation did not fully converge: %s", optimisation.message
        )

    weights = optimisation.x
    exp_return, vol, sharpe = _portfolio_performance(
        weights, mean_returns, cov_matrix, risk_free_rate
    )

    result = PortfolioResult(
        tickers=tickers,
        weights=weights,
        expected_return=exp_return,
        volatility=vol,
        sharpe_ratio=sharpe,
        risk_profile=risk_profile,
        efficient_frontier_path=frontier_path,
    )

    _generate_efficient_frontier_plot(mc_results, result, frontier_path)

    return result

