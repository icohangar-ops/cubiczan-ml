//! REIT financial ratio engine.
//!
//! Computes key REIT-specific metrics including FFO, AFFO, NAV, Debt/EBITDA,
//! Interest Coverage, Dividend Yield, P/FFO, Cap Rate, and Same-Store NOI Growth.

use chrono::Utc;
use crate::types::{FinancialRatios, REIT};

/// Errors that can occur during metric computation.
#[derive(Debug, thiserror::Error)]
pub enum MetricsError {
    #[error("Missing balance sheet data for {ticker}")]
    MissingBalanceSheet { ticker: String },

    #[error("Missing income statement data for {ticker}")]
    MissingIncomeStatement { ticker: String },

    #[error("Division by zero computing {metric}")]
    DivisionByZero { metric: String },

    #[error("Insufficient data for metric: {0}")]
    InsufficientData(String),
}

/// Compute FFO (Funds From Operations) per share.
///
/// FFO = Net Income + Depreciation/Amortization - Gains on Sales
/// FFO per share = FFO / Weighted Average Shares
pub fn compute_ffo_per_share(
    net_income: f64,
    depreciation: f64,
    gains_on_sales: f64,
    weighted_avg_shares: f64,
) -> Result<f64, MetricsError> {
    if weighted_avg_shares.abs() < 1e-9 {
        return Err(MetricsError::DivisionByZero {
            metric: "FFO per share".into(),
        });
    }
    let ffo = net_income + depreciation - gains_on_sales;
    Ok(ffo / weighted_avg_shares)
}

/// Compute total FFO in dollars.
pub fn compute_ffo(
    net_income: f64,
    depreciation: f64,
    gains_on_sales: f64,
) -> f64 {
    net_income + depreciation - gains_on_sales
}

/// Compute AFFO (Adjusted Funds From Operations) per share.
///
/// AFFO = FFO - Recurring Capital Expenditures - Straight-Line Rent Adjustment
pub fn compute_affo_per_share(
    ffo_per_share: f64,
    recurring_capex: f64,
    weighted_avg_shares: f64,
    rent_adjustment: f64,
) -> Result<f64, MetricsError> {
    if weighted_avg_shares.abs() < 1e-9 {
        return Err(MetricsError::DivisionByZero {
            metric: "AFFO per share".into(),
        });
    }
    let recurring_capex_per_share = recurring_capex / weighted_avg_shares;
    let rent_adj_per_share = rent_adjustment / weighted_avg_shares;
    Ok(ffo_per_share - recurring_capex_per_share - rent_adj_per_share)
}

/// Compute NAV (Net Asset Value) per share.
///
/// NAV = (Total Assets - Total Liabilities) / Shares Outstanding
/// More precisely for REITs: NAV = (Real Estate Value - Debt + Other Assets) / Shares
pub fn compute_nav_per_share(
    total_assets: f64,
    total_liabilities: f64,
    shares_outstanding: f64,
) -> Result<f64, MetricsError> {
    if shares_outstanding.abs() < 1e-9 {
        return Err(MetricsError::DivisionByZero {
            metric: "NAV per share".into(),
        });
    }
    let equity = total_assets - total_liabilities;
    Ok(equity / shares_outstanding)
}

/// Compute Debt/EBITDA ratio.
pub fn compute_debt_to_ebitda(total_debt: f64, ebit: f64) -> Result<f64, MetricsError> {
    if ebit.abs() < 1e-9 {
        return Err(MetricsError::DivisionByZero {
            metric: "Debt/EBITDA".into(),
        });
    }
    Ok(total_debt / ebit)
}

/// Compute Interest Coverage Ratio (EBIT / Interest Expense).
pub fn compute_interest_coverage(ebit: f64, interest_expense: f64) -> Result<f64, MetricsError> {
    if interest_expense.abs() < 1e-9 {
        return Err(MetricsError::DivisionByZero {
            metric: "Interest Coverage".into(),
        });
    }
    Ok(ebit / interest_expense)
}

/// Compute Dividend Yield (annualized).
///
/// Dividend Yield = Annual Dividends Per Share / Share Price
pub fn compute_dividend_yield(dividends_per_share: f64, share_price: f64) -> Result<f64, MetricsError> {
    if share_price.abs() < 1e-9 {
        return Err(MetricsError::DivisionByZero {
            metric: "Dividend Yield".into(),
        });
    }
    Ok(dividends_per_share / share_price)
}

/// Compute P/FFO (Price to Funds From Operations) ratio.
pub fn compute_price_to_ffo(share_price: f64, ffo_per_share: f64) -> Result<f64, MetricsError> {
    if ffo_per_share.abs() < 1e-9 {
        return Err(MetricsError::DivisionByZero {
            metric: "P/FFO".into(),
        });
    }
    Ok(share_price / ffo_per_share)
}

/// Compute Capitalization Rate.
///
/// Cap Rate = NOI / Property Value (approximated by Net RE Assets)
pub fn compute_cap_rate(noi: f64, net_real_estate_assets: f64) -> Result<f64, MetricsError> {
    if net_real_estate_assets.abs() < 1e-9 {
        return Err(MetricsError::DivisionByZero {
            metric: "Cap Rate".into(),
        });
    }
    Ok(noi / net_real_estate_assets)
}

/// Compute Current Ratio.
pub fn compute_current_ratio(current_assets: f64, current_liabilities: f64) -> Result<f64, MetricsError> {
    if current_liabilities.abs() < 1e-9 {
        return Err(MetricsError::DivisionByZero {
            metric: "Current Ratio".into(),
        });
    }
    Ok(current_assets / current_liabilities)
}

/// Compute Debt-to-Equity ratio.
pub fn compute_debt_to_equity(total_debt: f64, shareholders_equity: f64) -> Result<f64, MetricsError> {
    if shareholders_equity.abs() < 1e-9 {
        return Err(MetricsError::DivisionByZero {
            metric: "Debt-to-Equity".into(),
        });
    }
    Ok(total_debt / shareholders_equity)
}

/// Compute Return on Equity.
pub fn compute_return_on_equity(net_income: f64, shareholders_equity: f64) -> Result<f64, MetricsError> {
    if shareholders_equity.abs() < 1e-9 {
        return Err(MetricsError::DivisionByZero {
            metric: "Return on Equity".into(),
        });
    }
    Ok(net_income / shareholders_equity)
}

/// Compute Operating Margin.
pub fn compute_operating_margin(ebit: f64, total_revenue: f64) -> Result<f64, MetricsError> {
    if total_revenue.abs() < 1e-9 {
        return Err(MetricsError::DivisionByZero {
            metric: "Operating Margin".into(),
        });
    }
    Ok(ebit / total_revenue)
}

/// Compute Same-Store NOI Growth (percentage change).
pub fn compute_same_store_noi_growth(
    current_noi: f64,
    prior_noi: f64,
) -> Result<f64, MetricsError> {
    if prior_noi.abs() < 1e-9 {
        return Err(MetricsError::DivisionByZero {
            metric: "Same-Store NOI Growth".into(),
        });
    }
    Ok((current_noi - prior_noi) / prior_noi.abs())
}

/// Compute all financial ratios for a REIT given its financial statements.
pub fn compute_all_ratios(
    reit: &REIT,
) -> Result<FinancialRatios, MetricsError> {
    let bs = reit.balance_sheet.as_ref().ok_or(MetricsError::MissingBalanceSheet {
        ticker: reit.ticker.clone(),
    })?;
    let is = reit.income_statement.as_ref().ok_or(MetricsError::MissingIncomeStatement {
        ticker: reit.ticker.clone(),
    })?;

    let gains = is.gains_losses_on_sales.unwrap_or(0.0);
    let weighted_shares = is.weighted_avg_shares.unwrap_or(bs.shares_outstanding);

    let ffo = compute_ffo(is.net_income, is.depreciation_amortization, gains);
    let ffo_per_share = compute_ffo_per_share(
        is.net_income,
        is.depreciation_amortization,
        gains,
        weighted_shares,
    )?;

    // AFFO with zero recurring capex and rent adjustment (simplified)
    let affo_per_share = compute_affo_per_share(ffo_per_share, 0.0, weighted_shares, 0.0)?;

    let nav_per_share = compute_nav_per_share(
        bs.total_assets,
        bs.total_liabilities,
        bs.shares_outstanding,
    )?;

    let price_to_ffo = if reit.share_price.abs() > 1e-9 {
        compute_price_to_ffo(reit.share_price, ffo_per_share).unwrap_or(0.0)
    } else {
        0.0
    };

    let dividend_yield = match is.dividends_per_share {
        Some(dps) if reit.share_price.abs() > 1e-9 => {
            compute_dividend_yield(dps, reit.share_price).unwrap_or(0.0)
        }
        _ => 0.0,
    };

    let debt_to_ebitda = if is.ebit.abs() > 1e-9 {
        compute_debt_to_ebitda(bs.total_debt, is.ebit).unwrap_or(0.0)
    } else {
        0.0
    };

    let interest_coverage = if is.interest_expense.abs() > 1e-9 {
        compute_interest_coverage(is.ebit, is.interest_expense).unwrap_or(0.0)
    } else {
        0.0
    };

    let cap_rate = if bs.net_real_estate_assets.abs() > 1e-9 {
        compute_cap_rate(is.noi, bs.net_real_estate_assets).unwrap_or(0.0)
    } else {
        0.0
    };

    let same_store_noi = is.same_store_noi_growth.unwrap_or(0.0);

    let current_ratio = if bs.current_liabilities.abs() > 1e-9 {
        compute_current_ratio(bs.current_assets, bs.current_liabilities).unwrap_or(0.0)
    } else {
        0.0
    };

    let debt_to_equity = if bs.shareholders_equity.abs() > 1e-9 {
        compute_debt_to_equity(bs.total_debt, bs.shareholders_equity).unwrap_or(0.0)
    } else {
        0.0
    };

    let return_on_equity = if bs.shareholders_equity.abs() > 1e-9 {
        compute_return_on_equity(is.net_income, bs.shareholders_equity).unwrap_or(0.0)
    } else {
        0.0
    };

    let operating_margin = if is.total_revenue.abs() > 1e-9 {
        compute_operating_margin(is.ebit, is.total_revenue).unwrap_or(0.0)
    } else {
        0.0
    };

    Ok(FinancialRatios {
        computed_at: Utc::now(),
        ffo,
        ffo_per_share,
        affo_per_share,
        nav_per_share,
        price_to_ffo,
        dividend_yield,
        debt_to_ebitda,
        interest_coverage,
        cap_rate,
        same_store_noi_growth: same_store_noi,
        current_ratio,
        debt_to_equity,
        return_on_equity,
        operating_margin,
    })
}

/// Quick health check scoring for a REIT based on key ratios.
/// Returns a score from 0.0 (poor) to 100.0 (excellent).
pub fn health_check_score(ratios: &FinancialRatios) -> f64 {
    let mut score: f64 = 50.0;

    // Interest coverage: higher is better; >4.0 is strong
    if ratios.interest_coverage >= 4.0 {
        score += 15.0;
    } else if ratios.interest_coverage >= 2.5 {
        score += 10.0;
    } else if ratios.interest_coverage >= 1.5 {
        score += 5.0;
    } else if ratios.interest_coverage < 1.0 {
        score -= 15.0;
    }

    // Debt/EBITDA: lower is better; <5x is good
    if ratios.debt_to_ebitda <= 3.0 {
        score += 10.0;
    } else if ratios.debt_to_ebitda <= 5.0 {
        score += 5.0;
    } else if ratios.debt_to_ebitda > 7.0 {
        score -= 10.0;
    }

    // P/FFO: lower is better for value; 10-14x is fair value
    if ratios.price_to_ffo > 0.0 {
        if ratios.price_to_ffo <= 12.0 {
            score += 10.0;
        } else if ratios.price_to_ffo <= 16.0 {
            score += 5.0;
        } else if ratios.price_to_ffo > 22.0 {
            score -= 10.0;
        }
    }

    // Dividend yield: 3-6% is typical sweet spot
    if ratios.dividend_yield > 0.0 {
        if ratios.dividend_yield >= 0.03 && ratios.dividend_yield <= 0.06 {
            score += 10.0;
        } else if ratios.dividend_yield > 0.08 {
            score -= 5.0; // Potentially unsustainable
        }
    }

    // Same-store NOI growth: positive is good
    if ratios.same_store_noi_growth > 0.0 {
        score += 5.0;
    } else if ratios.same_store_noi_growth < -0.02 {
        score -= 10.0;
    }

    score.max(0.0).min(100.0)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::types::{BalanceSheet, IncomeStatement, REIT, REITSector};
    use chrono::Utc;

    fn make_test_reit() -> REIT {
        REIT {
            ticker: "O".into(),
            name: "Realty Income".into(),
            cik: "0001002047".into(),
            sector: REITSector::Residential,
            inception_date: None,
            market_cap: 48_000_000_000.0,
            share_price: 55.0,
            shares_outstanding: 872_727_273.0,
            balance_sheet: Some(BalanceSheet {
                period_end: Utc::now(),
                real_estate_assets: 15_000_000.0,
                accumulated_depreciation: 3_000_000.0,
                net_real_estate_assets: 12_000_000.0,
                total_assets: 20_000_000.0,
                current_assets: 500_000.0,
                total_liabilities: 10_000_000.0,
                current_liabilities: 300_000.0,
                mortgage_debt: 5_000_000.0,
                unsecured_debt: 3_000_000.0,
                total_debt: 8_000_000.0,
                shareholders_equity: 10_000_000.0,
                cash: 400_000.0,
                restricted_cash: 100_000.0,
                shares_outstanding: 872_727_273.0,
            }),
            income_statement: Some(IncomeStatement {
                period_start: Utc::now(),
                period_end: Utc::now(),
                rental_revenue: 2_000_000.0,
                total_revenue: 2_200_000.0,
                same_store_noi_growth: Some(0.035),
                operating_expenses: 800_000.0,
                noi: 1_200_000.0,
                depreciation_amortization: 400_000.0,
                general_admin_expenses: 200_000.0,
                interest_expense: 250_000.0,
                interest_income: Some(10_000.0),
                ebit: 600_000.0,
                income_tax_expense: 70_000.0,
                net_income: 380_000.0,
                gains_losses_on_sales: Some(50_000.0),
                ffo_per_share: Some(3.80),
                dividends_per_share: Some(3.08),
                weighted_avg_shares: Some(872_727_273.0),
            }),
            ratios: None,
        }
    }

    #[test]
    fn test_compute_ffo() {
        let ffo = compute_ffo(1_000_000.0, 500_000.0, 100_000.0);
        assert!((ffo - 1_400_000.0).abs() < 1.0);
    }

    #[test]
    fn test_compute_ffo_per_share() {
        let ffo_ps = compute_ffo_per_share(1_400_000.0, 500_000.0, 100_000.0, 1_000_000.0).unwrap();
        assert!((ffo_ps - 1.8).abs() < 1e-6);
    }

    #[test]
    fn test_compute_ffo_per_share_zero_shares() {
        let result = compute_ffo_per_share(100.0, 50.0, 0.0, 0.0);
        assert!(result.is_err());
    }

    #[test]
    fn test_compute_affo_per_share() {
        let affo = compute_affo_per_share(1.80, 200_000.0, 1_000_000.0, 50_000.0).unwrap();
        // AFFO = 1.80 - 0.20 - 0.05 = 1.55
        assert!((affo - 1.55).abs() < 1e-6);
    }

    #[test]
    fn test_compute_nav_per_share() {
        let nav = compute_nav_per_share(20_000_000.0, 10_000_000.0, 500_000.0).unwrap();
        assert!((nav - 20.0).abs() < 1e-6);
    }

    #[test]
    fn test_compute_debt_to_ebitda() {
        let ratio = compute_debt_to_ebitda(8_000_000.0, 600_000.0).unwrap();
        assert!((ratio - 13.333).abs() < 0.01);
    }

    #[test]
    fn test_compute_debt_to_ebitda_zero_ebit() {
        let result = compute_debt_to_ebitda(5_000_000.0, 0.0);
        assert!(result.is_err());
    }

    #[test]
    fn test_compute_interest_coverage() {
        let ratio = compute_interest_coverage(600_000.0, 250_000.0).unwrap();
        assert!((ratio - 2.4).abs() < 0.01);
    }

    #[test]
    fn test_compute_dividend_yield() {
        let yield_val = compute_dividend_yield(3.08, 55.0).unwrap();
        assert!((yield_val - 0.056).abs() < 0.001);
    }

    #[test]
    fn test_compute_price_to_ffo() {
        let ratio = compute_price_to_ffo(55.0, 3.80).unwrap();
        assert!((ratio - 14.473).abs() < 0.01);
    }

    #[test]
    fn test_compute_cap_rate() {
        let rate = compute_cap_rate(1_200_000.0, 12_000_000.0).unwrap();
        assert!((rate - 0.10).abs() < 0.001);
    }

    #[test]
    fn test_compute_current_ratio() {
        let ratio = compute_current_ratio(500_000.0, 300_000.0).unwrap();
        assert!((ratio - 1.6667).abs() < 0.001);
    }

    #[test]
    fn test_compute_debt_to_equity() {
        let ratio = compute_debt_to_equity(8_000_000.0, 10_000_000.0).unwrap();
        assert!((ratio - 0.8).abs() < 0.01);
    }

    #[test]
    fn test_compute_return_on_equity() {
        let roe = compute_return_on_equity(380_000.0, 10_000_000.0).unwrap();
        assert!((roe - 0.038).abs() < 0.001);
    }

    #[test]
    fn test_compute_operating_margin() {
        let margin = compute_operating_margin(600_000.0, 2_200_000.0).unwrap();
        assert!((margin - 0.2727).abs() < 0.001);
    }

    #[test]
    fn test_compute_same_store_noi_growth() {
        let growth = compute_same_store_noi_growth(1_200_000.0, 1_100_000.0).unwrap();
        assert!((growth - 0.0909).abs() < 0.001);
    }

    #[test]
    fn test_compute_all_ratios() {
        let reit = make_test_reit();
        let ratios = compute_all_ratios(&reit).unwrap();
        assert!(ratios.ffo > 0.0);
        assert!(ratios.interest_coverage > 0.0);
        assert!(ratios.dividend_yield > 0.0);
        assert!(ratios.price_to_ffo > 0.0);
    }

    #[test]
    fn test_compute_all_ratios_missing_bs() {
        let mut reit = make_test_reit();
        reit.balance_sheet = None;
        let result = compute_all_ratios(&reit);
        assert!(result.is_err());
    }

    #[test]
    fn test_compute_all_ratios_missing_is() {
        let mut reit = make_test_reit();
        reit.income_statement = None;
        let result = compute_all_ratios(&reit);
        assert!(result.is_err());
    }

    #[test]
    fn test_health_check_score() {
        let reit = make_test_reit();
        let ratios = compute_all_ratios(&reit).unwrap();
        let score = health_check_score(&ratios);
        assert!(score >= 0.0 && score <= 100.0);
    }

    #[test]
    fn test_health_check_score_perfect() {
        let ratios = FinancialRatios {
            computed_at: Utc::now(),
            ffo: 500.0,
            ffo_per_share: 5.0,
            affo_per_share: 4.5,
            nav_per_share: 80.0,
            price_to_ffo: 11.0,
            dividend_yield: 0.045,
            debt_to_ebitda: 2.5,
            interest_coverage: 5.0,
            cap_rate: 0.06,
            same_store_noi_growth: 0.04,
            current_ratio: 2.0,
            debt_to_equity: 0.5,
            return_on_equity: 0.10,
            operating_margin: 0.50,
        };
        let score = health_check_score(&ratios);
        assert!(score >= 70.0); // Should score high for great metrics
    }
}
