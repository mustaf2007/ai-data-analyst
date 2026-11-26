import pandas as pd
import pytest
from financial_reporting import build_financial_performance_analysis

def test_financial_analysis_separates_portfolio_and_stock_sleeve_returns():
    df = pd.DataFrame({
        "ticker": ["AAA"], "stock cost basis": [28672.43],
        "market value": [27276.98], "starting capital": [100000.00],
        "ending portfolio value": [98575.87], "brokerage costs": [28.67],
    })
    result = build_financial_performance_analysis(df)
    metrics = result["metrics"]
    assert metrics["portfolio_return_pct"] == pytest.approx(-1.42413, abs=1e-4)
    assert metrics["invested_sleeve_pnl"] == pytest.approx(-1395.45, abs=0.01)
    assert metrics["invested_sleeve_return_pct"] == pytest.approx(-4.8649, abs=1e-3)
    assert metrics["reported_fees"] == pytest.approx(28.67)
    assert "not deducted again" in result["markdown"]

def test_financial_analysis_does_not_invent_total_portfolio_return():
    result = build_financial_performance_analysis(pd.DataFrame({"cost_basis": [100], "market_value": [90]}))
    assert "portfolio_return_pct" not in result["metrics"]
    assert "Total portfolio return:** Not calculated" in result["markdown"]
