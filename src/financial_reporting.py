"""Conservative portfolio-performance calculations from tabular holdings data."""
from __future__ import annotations
import pandas as pd

ALIASES = {
    "starting_capital": ["starting capital", "starting_capital", "initial capital", "initial_capital", "beginning portfolio value", "opening portfolio value"],
    "ending_portfolio_value": ["ending portfolio value", "ending_portfolio_value", "final portfolio value", "portfolio end value", "portfolio_value_end", "ending value"],
    "cost_basis": ["cost basis", "cost_basis", "invested amount", "invested_amount", "purchase cost", "stock cost basis", "stock_cost_basis"],
    "market_value": ["market value", "market_value", "current value", "current_value", "final value", "ending market value"],
    "fees": ["fees", "brokerage", "brokerage costs", "brokerage_cost", "commission", "transaction costs", "transaction_costs"],
    "security": ["ticker", "symbol", "security", "asset", "holding", "instrument"],
    "benchmark_return": ["benchmark return", "benchmark_return", "benchmark performance"],
}

def _normalize(value):
    return " ".join(str(value).strip().lower().replace("_", " ").replace("-", " ").split())

def _find_column(df, key):
    aliases = {_normalize(x) for x in ALIASES[key]}
    return next((col for col in df.columns if _normalize(col) in aliases), None)

def _numeric(series):
    return pd.to_numeric(series, errors="coerce")

def _single_value(series):
    values = _numeric(series).dropna().unique()
    return float(values[0]) if len(values) == 1 else None

def _usd(value):
    return "USD " + format(value, ",.2f")

def build_financial_performance_analysis(df: pd.DataFrame) -> dict:
    """Calculate only metrics supported by explicit, recognizable columns."""
    if not isinstance(df, pd.DataFrame) or df.empty:
        return {"available": False, "reason": "No non-empty dataset was supplied."}
    cols = {key: _find_column(df, key) for key in ALIASES}
    metrics = {}
    start_col, end_col = cols["starting_capital"], cols["ending_portfolio_value"]
    cost_col, value_col = cols["cost_basis"], cols["market_value"]
    fee_col, security_col = cols["fees"], cols["security"]

    starting = _single_value(df[start_col]) if start_col else None
    ending = _single_value(df[end_col]) if end_col else None
    if starting is not None and starting > 0 and ending is not None:
        pnl = ending - starting
        metrics.update(starting_capital=starting, ending_portfolio_value=ending,
                       portfolio_pnl=pnl, portfolio_return_pct=100 * pnl / starting)

    cost_total = _numeric(df[cost_col]).sum(min_count=1) if cost_col else None
    market_total = _numeric(df[value_col]).sum(min_count=1) if value_col else None
    if cost_total is not None and pd.notna(cost_total) and cost_total > 0 and market_total is not None and pd.notna(market_total):
        sleeve_pnl = float(market_total - cost_total)
        metrics.update(invested_sleeve_cost_basis=float(cost_total),
                       invested_sleeve_market_value=float(market_total),
                       invested_sleeve_pnl=sleeve_pnl,
                       invested_sleeve_return_pct=100 * sleeve_pnl / float(cost_total))
    if fee_col:
        fee_values = _numeric(df[fee_col]).dropna()
        if not fee_values.empty:
            metrics["reported_fees"] = float(fee_values.sum())
    benchmark_col = cols["benchmark_return"]
    if benchmark_col:
        values = _numeric(df[benchmark_col]).dropna().unique()
        if len(values) == 1:
            metrics["benchmark_return_pct"] = float(values[0])

    holdings = pd.DataFrame()
    if security_col and cost_col and value_col:
        work = df[[security_col, cost_col, value_col]].copy()
        work[cost_col], work[value_col] = _numeric(work[cost_col]), _numeric(work[value_col])
        work = work.dropna(subset=[security_col])
        holdings = work.groupby(security_col, dropna=False).agg(
            cost_basis=(cost_col, "sum"), market_value=(value_col, "sum")
        ).reset_index()
        holdings["pnl"] = holdings["market_value"] - holdings["cost_basis"]
        holdings["return_pct"] = holdings.apply(
            lambda row: 100 * row["pnl"] / row["cost_basis"] if row["cost_basis"] > 0 else float("nan"), axis=1
        )
        holdings = holdings.sort_values("pnl", ascending=True)

    lines = ["## Financial and investment performance analysis"]
    if "portfolio_return_pct" in metrics:
        lines.append("- **Total portfolio:** " + _usd(metrics["starting_capital"]) + " starting capital to " + _usd(metrics["ending_portfolio_value"]) + " ending value; net change " + _usd(metrics["portfolio_pnl"]) + " ({:.2f}%).".format(metrics["portfolio_return_pct"]))
    else:
        lines.append("- **Total portfolio return:** Not calculated. A unique, explicit starting-capital and ending-portfolio-value pair was not available.")
    if "invested_sleeve_return_pct" in metrics:
        lines.append("- **Invested sleeve:** cost basis " + _usd(metrics["invested_sleeve_cost_basis"]) + "; market value " + _usd(metrics["invested_sleeve_market_value"]) + "; P&L " + _usd(metrics["invested_sleeve_pnl"]) + " ({:.2f}%).".format(metrics["invested_sleeve_return_pct"]))
    else:
        lines.append("- **Invested-sleeve return:** Not calculated. Both cost-basis and market-value columns are required.")
    if "reported_fees" in metrics:
        lines.append("- **Reported fees / brokerage:** " + _usd(metrics["reported_fees"]) + ". Whether these costs are already reflected in ending values is not established; they are not deducted again.")
    if not holdings.empty:
        lines.append("- **Holdings attribution:** Grouped cost basis versus market value; not a time-weighted or cash-flow-adjusted return.")
        headers = [str(c) for c in holdings.columns]
        lines.extend(["| " + " | ".join(headers) + " |", "| " + " | ".join(["---"] * len(headers)) + " |"])
        for row in holdings.itertuples(index=False, name=None):
            cells = [format(v, ",.2f") if isinstance(v, (int, float)) else str(v) for v in row]
            lines.append("| " + " | ".join(cells) + " |")
    if "benchmark_return_pct" in metrics:
        lines.append("- **Benchmark return:** {:.2f}% (relative performance is not inferred without matching periods and definitions).".format(metrics["benchmark_return_pct"]))
    lines += ["", "**Interpretation limits:** Returns are simple value changes based on detected columns. Compatible dates, currency, cash flows, dividends, corporate actions, and fee treatment must be established. No annualization, time-weighted return, money-weighted return, risk-adjusted return, or causal attribution is inferred."]
    return {"available": bool(metrics), "detected_columns": {k: str(v) for k, v in cols.items() if v is not None},
            "metrics": metrics, "holdings": holdings, "markdown": "\n".join(lines),
            **({"reason": "No sufficiently explicit portfolio or invested-sleeve measures were detected."} if not metrics else {})}
