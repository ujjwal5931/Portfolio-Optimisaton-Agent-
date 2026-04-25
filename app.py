from pathlib import Path
from typing import Any, Dict, List, Optional

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field, validator

from data_fetcher import default_tickers_for_geography, fetch_price_history
from optimizer import PortfolioResult, optimise_portfolio
from planner import build_execution_plan, parse_investment_goal
from report_generator import generate_human_readable_report
from utils import get_groq_client, get_logger, sanitize_geography, sanitize_risk_level


logger = get_logger(__name__)

# Resolve paths for static assets and generated plots
BASE_DIR = Path(__file__).resolve().parent
STATIC_DIR = BASE_DIR / "static"

app = FastAPI(
    title="Portfolio Optimisation Agent",
    description=(
        "Autonomous AI agent that turns a natural language investment goal into an "
        "optimised stock portfolio using Markowitz mean-variance optimisation."
    ),
    version="1.0.0",
)

# Serve static assets (CSS, JS, etc.)
app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")


class PortfolioRequest(BaseModel):
    """
    Request model for the optimisation API.

    - goal: Natural-language investment goal from the end-user.
    - tickers: Optional custom ticker universe; when omitted, a sensible default
      is chosen based on the inferred geography.
    """

    goal: str = Field(
        ...,
        description=(
            "Natural-language investment goal, e.g. "
            "'Invest ₹50,000 in Indian stocks with moderate risk for 1 year'."
        ),
    )
    tickers: Optional[List[str]] = Field(
        default=None,
        description="Optional custom list of stock tickers to consider.",
    )

    @validator("tickers")
    def _strip_tickers(cls, v: Optional[List[str]]) -> Optional[List[str]]:
        if v is None:
            return v
        cleaned = [t.strip().upper() for t in v if t and t.strip()]
        return cleaned or None


class PortfolioResponse(BaseModel):
    """
    Canonical JSON response from the optimisation endpoint.
    """

    allocation: Dict[str, float]
    expected_return: float
    volatility: float
    sharpe_ratio: float
    risk_level: str
    geography: str
    efficient_frontier_path: str
    explanation: str
    execution_plan: Dict[str, Any]


@app.get("/", response_class=HTMLResponse, tags=["ui"])
def serve_ui() -> HTMLResponse:
    """Serve the main portfolio optimisation UI."""
    index_path = STATIC_DIR / "index.html"
    if not index_path.exists():
        raise HTTPException(status_code=404, detail="UI not found")
    return HTMLResponse(content=index_path.read_text(encoding="utf-8"))


@app.get("/efficient-frontier.png", tags=["portfolio"])
def serve_efficient_frontier() -> FileResponse:
    """Serve the most recently generated efficient frontier plot."""
    plot_path = BASE_DIR / "efficient_frontier.png"
    if not plot_path.exists():
        raise HTTPException(status_code=404, detail="Efficient frontier plot not yet generated")
    return FileResponse(plot_path, media_type="image/png")


@app.get("/health", tags=["meta"])
def health_check() -> Dict[str, str]:
    """
    Lightweight health endpoint useful for readiness/liveness probes.
    """
    return {"status": "ok"}


@app.post("/optimize-portfolio", response_model=PortfolioResponse, tags=["portfolio"])
def optimize_portfolio(request: PortfolioRequest) -> PortfolioResponse:
    """
    Core endpoint: accepts a natural-language investment goal, delegates to an LLM
    for interpretation, runs portfolio optimisation, and returns a structured,
    production-ready JSON response.
    """
    try:
        client = get_groq_client()
    except RuntimeError as exc:
        logger.exception("Groq client initialisation failed.")
        raise HTTPException(status_code=500, detail=str(exc)) from exc

    # 1. Parse the unstructured goal into a machine-readable representation.
    try:
        structured_goal = parse_investment_goal(request.goal, client)
    except Exception as exc:  # noqa: BLE001
        logger.exception("Failed to parse investment goal via LLM.")
        raise HTTPException(
            status_code=400,
            detail="Unable to interpret the investment goal. Please rephrase and try again.",
        ) from exc

    # Normalise core fields for downstream use.
    risk_level = sanitize_risk_level(structured_goal.get("risk_level"))
    geography = sanitize_geography(structured_goal.get("geography"))
    duration_years = float(structured_goal.get("duration_years", 1.0))

    # 2. Decide ticker universe: use user-provided tickers if supplied, otherwise
    #    fall back to geography-specific defaults.
    if request.tickers:
        tickers = request.tickers
        logger.info(
            "Using custom ticker universe from request (%d tickers).", len(tickers)
        )
    else:
        tickers = default_tickers_for_geography(geography)
        logger.info(
            "Using default tickers for geography '%s' (%d tickers).",
            geography,
            len(tickers),
        )

    # 3. Fetch historical price data for the chosen universe.
    try:
        prices, _ = fetch_price_history(tickers, duration_years)
    except ValueError as exc:
        logger.exception("Failed to fetch price history from yfinance.")
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:  # noqa: BLE001
        logger.exception("Unexpected error while fetching price history.")
        raise HTTPException(
            status_code=500,
            detail="Unexpected error while fetching historical market data.",
        ) from exc

    # 4. Run Markowitz optimisation with constraints and Monte Carlo simulation.
    try:
        portfolio: PortfolioResult = optimise_portfolio(
            prices=prices,
            risk_profile=risk_level,
            risk_free_rate=0.06,
            num_portfolios=5000,
            frontier_path="efficient_frontier.png",
        )
    except Exception as exc:  # noqa: BLE001
        logger.exception("Portfolio optimisation failed.")
        raise HTTPException(
            status_code=500,
            detail="Portfolio optimisation failed. Please try again later.",
        ) from exc

    # 5. Ask the LLM to turn the numbers into a narrative explanation.
    try:
        explanation = generate_human_readable_report(
            client=client,
            original_goal=request.goal,
            structured_goal=structured_goal,
            portfolio=portfolio,
        )
    except Exception as exc:  # noqa: BLE001
        logger.exception("Failed to generate human-readable explanation.")
        # The numeric output is still useful to the caller even if the LLM step fails.
        explanation = (
            "An error occurred while generating the natural language explanation. "
            "The allocation and metrics are still valid numeric outputs."
        )

    # 6. Build an explicit execution plan artefact for transparency.
    execution_plan = build_execution_plan(structured_goal)

    allocation = {
        ticker: float(weight) for ticker, weight in zip(portfolio.tickers, portfolio.weights)
    }

    response = PortfolioResponse(
        allocation=allocation,
        expected_return=float(portfolio.expected_return),
        volatility=float(portfolio.volatility),
        sharpe_ratio=float(portfolio.sharpe_ratio),
        risk_level=risk_level,
        geography=geography,
        efficient_frontier_path=portfolio.efficient_frontier_path,
        explanation=explanation,
        execution_plan=execution_plan,
    )

    return response


if __name__ == "__main__":
    # Allow running directly via `python app.py` during development.
    import uvicorn

    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=True)

