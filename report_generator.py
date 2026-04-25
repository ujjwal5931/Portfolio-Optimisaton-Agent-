from typing import Any, Dict

from groq import Groq

from optimizer import PortfolioResult
from utils import get_logger


logger = get_logger(__name__)


def generate_human_readable_report(
    client: Groq,
    original_goal: str,
    structured_goal: Dict[str, Any],
    portfolio: PortfolioResult,
) -> str:
    """
    Use the Groq LLM to turn the numeric optimisation output into a clear,
    human-readable explanation tailored to the user's original goal.

    The explanation is careful not to promise guaranteed returns and instead
    frames outputs as scenarios based on historical data.
    """
    system_prompt = (
        "You are an experienced portfolio manager explaining investment plans to "
        "a non-technical client. You must:\n"
        "- Clearly explain the proposed allocation across stocks.\n"
        "- Briefly describe the risk/return trade-off in plain language.\n"
        "- Emphasise that numbers are estimates based on historical data, not guarantees.\n"
        "- Keep the tone professional but friendly.\n"
        "- Avoid giving tax or legal advice.\n"
    )

    weights_pct = [
        f"{ticker}: {weight * 100:.2f}%" for ticker, weight in zip(portfolio.tickers, portfolio.weights)
    ]

    user_prompt = (
        "Original user goal:\n"
        f"{original_goal}\n\n"
        "Structured goal (parsed):\n"
        f"{structured_goal}\n\n"
        "Optimised portfolio summary:\n"
        f"- Risk profile used: {portfolio.risk_profile}\n"
        f"- Expected annual return (approx): {portfolio.expected_return*100:.2f}%\n"
        f"- Annualised volatility (approx): {portfolio.volatility*100:.2f}%\n"
        f"- Sharpe ratio (approx): {portfolio.sharpe_ratio:.2f}\n"
        f"- Allocation by ticker:\n"
        + "\n".join(f"  * {item}" for item in weights_pct)
        + "\n\n"
        "Write a concise explanation (3–6 short paragraphs) that the user can "
        "easily understand. Do not output JSON or bullet lists in the final answer."
    )

    logger.info("Calling Groq LLM to generate human-readable explanation.")

    completion = client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        temperature=0.5,
        max_tokens=800,
    )

    content = completion.choices[0].message.content
    if not content:
        raise ValueError("LLM returned empty content while generating explanation.")

    return content.strip()

