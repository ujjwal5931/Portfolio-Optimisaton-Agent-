from typing import Any, Dict

from groq import Groq

from utils import get_logger, sanitize_geography, sanitize_risk_level


logger = get_logger(__name__)


def parse_investment_goal(goal: str, client: Groq) -> Dict[str, Any]:
    """
    Use the Groq LLM to transform an unstructured natural-language investment goal
    into a structured JSON payload that the rest of the pipeline can rely on.

    The model is instructed to emit STRICT JSON so we can safely parse the response
    without brittle string post-processing.
    """
    system_prompt = (
        "You are a financial planning assistant. "
        "Your job is to read a user's natural-language investment goal and convert it "
        "into a STRICT JSON object with the following keys:\n"
        "  - investment_amount: float, total amount to invest in numeric form (no currency symbol).\n"
        "  - currency: string, ISO-like currency code inferred from the text (e.g. INR, USD).\n"
        "  - risk_level: one of 'low', 'moderate', 'high'.\n"
        "  - duration_years: float, investment horizon in years.\n"
        "  - geography: one of 'india', 'us', 'global' based on the market focus.\n"
        "  - notes: short string summarising the user's goal.\n\n"
        "Return ONLY valid JSON. Do not include any explanation or markdown."
    )

    user_prompt = (
        "User goal:\n"
        f"{goal}\n\n"
        "Extract and normalise the goal into the specified JSON format."
    )

    logger.info("Calling Groq LLM to parse investment goal.")

    completion = client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        temperature=0.1,
        max_tokens=512,
    )

    content = completion.choices[0].message.content
    if not content:
        raise ValueError("LLM returned empty content while parsing goal.")

    import json  # Local import to keep module import side-effects minimal

    try:
        data: Dict[str, Any] = json.loads(content)
    except json.JSONDecodeError as exc:
        logger.error("Failed to decode LLM JSON output: %s", content)
        raise ValueError("LLM did not return valid JSON for goal parsing.") from exc

    # Basic post-processing to ensure consistent downstream types.
    investment_amount = float(data.get("investment_amount", 0.0))
    duration_years = float(data.get("duration_years", 1.0))
    risk_level = sanitize_risk_level(data.get("risk_level", "moderate"))
    geography = sanitize_geography(data.get("geography", "global"))
    currency = data.get("currency", "INR")
    notes = data.get("notes") or ""

    structured = {
        "investment_amount": investment_amount,
        "currency": currency,
        "risk_level": risk_level,
        "duration_years": duration_years,
        "geography": geography,
        "notes": notes,
        "raw": data,
    }

    logger.info(
        "Parsed goal - amount=%s %s, risk=%s, duration=%s years, geography=%s",
        investment_amount,
        currency,
        risk_level,
        duration_years,
        geography,
    )

    return structured


def build_execution_plan(structured_goal: Dict[str, Any]) -> Dict[str, Any]:
    """
    Build a simple, machine-readable execution plan from the parsed goal.

    This makes it easy to debug and extend the pipeline while also providing
    a clear artefact to log or expose in responses if desired.
    """
    plan = {
        "steps": [
            "Validate parsed goal fields (amount, risk, duration, geography).",
            "Select universe of eligible stocks based on geography and user tickers.",
            "Fetch historical price data for the chosen tickers via yfinance.",
            "Compute daily returns, annualised mean returns, and covariance matrix.",
            "Run Monte Carlo simulation to explore 5000 random portfolios.",
            "Run Markowitz mean-variance optimisation with no short-selling.",
            "Apply risk profile mapping to select objective (volatility / Sharpe / return).",
            "Generate efficient frontier plot as PNG.",
            "Call LLM to generate a human-readable investment explanation.",
        ],
        "goal_summary": structured_goal.get("notes"),
        "risk_level": structured_goal.get("risk_level"),
        "geography": structured_goal.get("geography"),
        "duration_years": structured_goal.get("duration_years"),
    }
    return plan

