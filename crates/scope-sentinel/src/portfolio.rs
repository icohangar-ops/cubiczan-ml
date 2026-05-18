//! Portfolio analytics for REIT holdings.
//!
//! Provides portfolio construction, sector allocation analysis, diversification
//! scoring, benchmark comparison, and risk-adjusted return calculations
//! (Sharpe, Sortino, Treynor ratios).

use std::collections::HashMap;
use ndarray::{Array1, Array2};
use crate::types::{FinancialRatios, PortfolioAnalytics, PortfolioPosition, REITSector, RiskRating};

/// Errors that can occur during portfolio operations.
#[derive(Debug, thiserror::Error)]
pub enum PortfolioError {
    #[error("Empty portfolio — no positions provided")]
    EmptyPortfolio,

    #[error("Position weight exceeds 1.0: {weight}")]
    InvalidWeight { weight: f64 },

    #[error("Total portfolio weight is {total}, expected 1.0")]
    WeightSumError { total: f64 },

    #[error("Cannot compute ratio: {reason}")]
    RatioError { reason: String },

    #[error("Missing financial ratios for ticker: {0}")]
    MissingRatios(String),
}

/// Compute portfolio sector weights as a map from sector to weight fraction.
pub fn compute_sector_weights(
    positions: &[PortfolioPosition],
) -> HashMap<REITSector, f64> {
    let mut weights: HashMap<REITSector, f64> = HashMap::new();
    for pos in positions {
        *weights.entry(pos.sector).or_insert(0.0) += pos.weight;
    }
    weights
}

/// Compute Herfindahl-Hirschman Index (HHI) based diversification score.
/// Returns a score from 0.0 (perfectly diversified) to 1.0 (concentrated).
///
/// diversification_score = 1 - HHI
pub fn compute_diversification_score(positions: &[PortfolioPosition]) -> Result<f64, PortfolioError> {
    if positions.is_empty() {
        return Err(PortfolioError::EmptyPortfolio);
    }
    let hhi: f64 = positions.iter().map(|p| p.weight * p.weight).sum();
    Ok(1.0 - hhi)
}

/// Compute sector-based diversification score (HHI across sectors).
pub fn compute_sector_diversification_score(
    positions: &[PortfolioPosition],
) -> Result<f64, PortfolioError> {
    if positions.is_empty() {
        return Err(PortfolioError::EmptyPortfolio);
    }
    let sector_weights = compute_sector_weights(positions);
    let hhi: f64 = sector_weights.values().map(|w| w * w).sum();
    Ok(1.0 - hhi)
}

/// Validate that portfolio weights sum to approximately 1.0.
pub fn validate_weights(positions: &[PortfolioPosition], tolerance: f64) -> Result<(), PortfolioError> {
    let total: f64 = positions.iter().map(|p| p.weight).sum();
    if (total - 1.0).abs() > tolerance {
        return Err(PortfolioError::WeightSumError { total });
    }
    for pos in positions {
        if pos.weight < 0.0 || pos.weight > 1.0 {
            return Err(PortfolioError::InvalidWeight { weight: pos.weight });
        }
    }
    Ok(())
}

/// Compute weighted average of a ratio across portfolio positions.
pub fn weighted_average_ratio(
    positions: &[PortfolioPosition],
    ratio_fn: fn(&FinancialRatios) -> f64,
) -> Result<f64, PortfolioError> {
    if positions.is_empty() {
        return Err(PortfolioError::EmptyPortfolio);
    }

    let mut weighted_sum = 0.0;
    let mut total_weight = 0.0;

    for pos in positions {
        if let Some(ratios) = &pos.ratios {
            weighted_sum += pos.weight * ratio_fn(ratios);
            total_weight += pos.weight;
        }
    }

    if total_weight.abs() < 1e-9 {
        return Err(PortfolioError::RatioError {
            reason: "No positions with financial ratios".into(),
        });
    }

    Ok(weighted_sum / total_weight)
}

/// Compute the Sharpe ratio for a series of returns.
///
/// Sharpe = (mean_return - risk_free_rate) / std_dev_return
pub fn compute_sharpe_ratio(
    returns: &[f64],
    risk_free_rate: f64,
) -> Result<f64, PortfolioError> {
    if returns.len() < 2 {
        return Err(PortfolioError::RatioError {
            reason: "Need at least 2 return observations".into(),
        });
    }

    let n = returns.len() as f64;
    let mean: f64 = returns.iter().sum::<f64>() / n;
    let variance: f64 = returns.iter().map(|r| (r - mean).powi(2)).sum::<f64>() / (n - 1.0);
    let std_dev = variance.sqrt();

    if std_dev.abs() < 1e-12 {
        return Ok(0.0);
    }

    Ok((mean - risk_free_rate) / std_dev)
}

/// Compute the Sortino ratio (uses downside deviation instead of total std dev).
///
/// Sortino = (mean_return - risk_free_rate) / downside_deviation
pub fn compute_sortino_ratio(
    returns: &[f64],
    risk_free_rate: f64,
    minimum_acceptable_return: f64,
) -> Result<f64, PortfolioError> {
    if returns.len() < 2 {
        return Err(PortfolioError::RatioError {
            reason: "Need at least 2 return observations".into(),
        });
    }

    let n = returns.len() as f64;
    let mean: f64 = returns.iter().sum::<f64>() / n;

    // Downside deviation: only consider returns below MAR
    let downside: f64 = returns
        .iter()
        .map(|r| {
            let diff = r - minimum_acceptable_return;
            if diff < 0.0 { diff * diff } else { 0.0 }
        })
        .sum::<f64>();
    let downside_dev = (downside / n).sqrt();

    if downside_dev.abs() < 1e-12 {
        return Ok(0.0);
    }

    Ok((mean - risk_free_rate) / downside_dev)
}

/// Compute the Treynor ratio.
///
/// Treynor = (mean_return - risk_free_rate) / beta
pub fn compute_treynor_ratio(
    returns: &[f64],
    benchmark_returns: &[f64],
    risk_free_rate: f64,
) -> Result<f64, PortfolioError> {
    if returns.len() < 2 || returns.len() != benchmark_returns.len() {
        return Err(PortfolioError::RatioError {
            reason: "Returns and benchmark must have equal length >= 2".into(),
        });
    }

    let beta = compute_beta(returns, benchmark_returns)?;

    if beta.abs() < 1e-12 {
        return Err(PortfolioError::RatioError {
            reason: "Beta is near zero".into(),
        });
    }

    let mean: f64 = returns.iter().sum::<f64>() / returns.len() as f64;
    Ok((mean - risk_free_rate) / beta)
}

/// Compute portfolio beta relative to a benchmark.
pub fn compute_beta(portfolio_returns: &[f64], benchmark_returns: &[f64]) -> Result<f64, PortfolioError> {
    if portfolio_returns.len() < 2 || portfolio_returns.len() != benchmark_returns.len() {
        return Err(PortfolioError::RatioError {
            reason: "Mismatched or insufficient return data".into(),
        });
    }

    let p = Array1::from_vec(portfolio_returns.to_vec());
    let b = Array1::from_vec(benchmark_returns.to_vec());
    let n = p.len() as f64;

    let p_mean = p.sum() / n;
    let b_mean = b.sum() / n;

    let p_centered = &p - p_mean;
    let b_centered = &b - b_mean;

    let covariance: f64 = (&p_centered * &b_centered).sum() / (n - 1.0);
    let variance: f64 = (&b_centered * &b_centered).sum() / (n - 1.0);

    if variance.abs() < 1e-12 {
        return Err(PortfolioError::RatioError {
            reason: "Benchmark variance is zero".into(),
        });
    }

    Ok(covariance / variance)
}

/// Compute portfolio alpha (Jensen's alpha).
pub fn compute_alpha(
    portfolio_returns: &[f64],
    benchmark_returns: &[f64],
    risk_free_rate: f64,
) -> Result<f64, PortfolioError> {
    let beta = compute_beta(portfolio_returns, benchmark_returns)?;
    let p_mean: f64 = portfolio_returns.iter().sum::<f64>() / portfolio_returns.len() as f64;
    let b_mean: f64 = benchmark_returns.iter().sum::<f64>() / benchmark_returns.len() as f64;

    let expected = risk_free_rate + beta * (b_mean - risk_free_rate);
    Ok(p_mean - expected)
}

/// Compute correlation matrix for a set of return series.
pub fn compute_correlation_matrix(returns_matrix: &[Vec<f64>]) -> Result<Array2<f64>, PortfolioError> {
    let n_series = returns_matrix.len();
    if n_series == 0 {
        return Err(PortfolioError::EmptyPortfolio);
    }
    let n_obs = returns_matrix[0].len();
    if n_obs < 2 {
        return Err(PortfolioError::RatioError {
            reason: "Need at least 2 observations".into(),
        });
    }

    // Verify all series have the same length
    for series in returns_matrix {
        if series.len() != n_obs {
            return Err(PortfolioError::RatioError {
                reason: "All return series must have equal length".into(),
            });
        }
    }

    let mut corr = Array2::zeros((n_series, n_series));
    let means: Vec<f64> = returns_matrix
        .iter()
        .map(|s| s.iter().sum::<f64>() / n_obs as f64)
        .collect();

    // Compute covariance matrix
    let mut cov = Array2::zeros((n_series, n_series));
    for i in 0..n_series {
        for j in i..n_series {
            let mut cov_ij = 0.0;
            for k in 0..n_obs {
                cov_ij += (returns_matrix[i][k] - means[i]) * (returns_matrix[j][k] - means[j]);
            }
            cov_ij /= (n_obs - 1) as f64;
            cov[[i, j]] = cov_ij;
            cov[[j, i]] = cov_ij;
        }
    }

    // Convert to correlation
    for i in 0..n_series {
        for j in 0..n_series {
            let denom = (cov[[i, i]] * cov[[j, j]]).sqrt();
            if denom.abs() > 1e-12 {
                corr[[i, j]] = cov[[i, j]] / denom;
            } else {
                corr[[i, j]] = 0.0;
            }
        }
    }

    Ok(corr)
}

/// Determine the overall risk rating for a portfolio based on metrics.
pub fn assess_portfolio_risk(
    weighted_avg_debt_to_ebitda: f64,
    diversification_score: f64,
    weighted_avg_dividend_yield: f64,
) -> RiskRating {
    let mut risk_score = 3.0f64;

    // High leverage increases risk
    if weighted_avg_debt_to_ebitda > 7.0 {
        risk_score += 1.5;
    } else if weighted_avg_debt_to_ebitda > 5.0 {
        risk_score += 0.5;
    } else if weighted_avg_debt_to_ebitda < 3.0 {
        risk_score -= 0.5;
    }

    // Poor diversification increases risk
    if diversification_score < 0.3 {
        risk_score += 1.0;
    } else if diversification_score > 0.7 {
        risk_score -= 0.5;
    }

    // Very high yield may indicate risk
    if weighted_avg_dividend_yield > 0.08 {
        risk_score += 0.5;
    }

    let rounded = risk_score.round().clamp(1.0, 5.0) as u8;
    RiskRating::from_score(rounded).unwrap_or(RiskRating::Medium)
}

/// Generate a full portfolio analytics summary.
pub fn compute_portfolio_analytics(
    positions: &[PortfolioPosition],
    benchmark_returns: Option<&[f64]>,
    portfolio_returns: Option<&[f64]>,
    risk_free_rate: f64,
) -> Result<PortfolioAnalytics, PortfolioError> {
    if positions.is_empty() {
        return Err(PortfolioError::EmptyPortfolio);
    }

    let total_value: f64 = positions.iter().map(|p| p.market_value).sum();
    let total_cost: f64 = positions.iter().map(|p| p.cost_basis * p.shares).sum();
    let total_pnl = total_value - total_cost;
    let total_return_pct = if total_cost.abs() > 1e-9 {
        (total_pnl / total_cost) * 100.0
    } else {
        0.0
    };

    let sector_weights = compute_sector_weights(positions);
    let div_score = compute_diversification_score(positions)?;
    let sector_div = compute_sector_diversification_score(positions)?;

    // Use the average of individual and sector diversification
    let diversification_score = (div_score + sector_div) / 2.0;

    let w_avg_div_yield = weighted_average_ratio(positions, |r| r.dividend_yield).unwrap_or(0.0);
    let w_avg_ffo_yield = weighted_average_ratio(positions, |r| {
        if r.price_to_ffo.abs() > 1e-9 { 1.0 / r.price_to_ffo } else { 0.0 }
    }).unwrap_or(0.0);
    let w_avg_d_ebitda = weighted_average_ratio(positions, |r| r.debt_to_ebitda).unwrap_or(0.0);

    let risk_rating = assess_portfolio_risk(w_avg_d_ebitda, diversification_score, w_avg_div_yield);

    let (sharpe, sortino, treynor) = match (portfolio_returns, benchmark_returns) {
        (Some(pr), Some(br)) => {
            let s = compute_sharpe_ratio(pr, risk_free_rate).ok();
            let so = compute_sortino_ratio(pr, risk_free_rate, risk_free_rate).ok();
            let tr = compute_treynor_ratio(pr, br, risk_free_rate).ok();
            (s, so, tr)
        }
        (Some(pr), None) => {
            let s = compute_sharpe_ratio(pr, risk_free_rate).ok();
            let so = compute_sortino_ratio(pr, risk_free_rate, risk_free_rate).ok();
            (s, so, None)
        }
        _ => (None, None, None),
    };

    Ok(PortfolioAnalytics {
        total_value,
        total_cost_basis: total_cost,
        total_pnl,
        total_return_pct,
        sector_weights,
        diversification_score,
        risk_rating,
        weighted_avg_dividend_yield: w_avg_div_yield,
        weighted_avg_ffo_yield: w_avg_ffo_yield,
        weighted_avg_debt_to_ebitda: w_avg_d_ebitda,
        sharpe_ratio: sharpe,
        sortino_ratio: sortino,
        treynor_ratio: treynor,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::types::FinancialRatios;
    use chrono::Utc;

    fn make_test_ratios(
        div_yield: f64,
        p_ffo: f64,
        d_ebitda: f64,
        ic: f64,
    ) -> FinancialRatios {
        FinancialRatios {
            computed_at: Utc::now(),
            ffo: 500.0,
            ffo_per_share: 5.0,
            affo_per_share: 4.5,
            nav_per_share: 80.0,
            price_to_ffo: p_ffo,
            dividend_yield: div_yield,
            debt_to_ebitda: d_ebitda,
            interest_coverage: ic,
            cap_rate: 0.055,
            same_store_noi_growth: 0.03,
            current_ratio: 2.0,
            debt_to_equity: 0.8,
            return_on_equity: 0.08,
            operating_margin: 0.45,
        }
    }

    fn sample_positions() -> Vec<PortfolioPosition> {
        vec![
            PortfolioPosition {
                ticker: "O".into(),
                sector: REITSector::Residential,
                shares: 100.0,
                cost_basis: 50.0,
                current_price: 55.0,
                weight: 0.4,
                market_value: 5500.0,
                unrealized_pnl: 500.0,
                unrealized_return_pct: 10.0,
                ratios: Some(make_test_ratios(0.04, 14.0, 5.0, 3.5)),
            },
            PortfolioPosition {
                ticker: "AMT".into(),
                sector: REITSector::DataCenter,
                shares: 50.0,
                cost_basis: 150.0,
                current_price: 180.0,
                weight: 0.35,
                market_value: 9000.0,
                unrealized_pnl: 1500.0,
                unrealized_return_pct: 20.0,
                ratios: Some(make_test_ratios(0.03, 20.0, 6.0, 4.0)),
            },
            PortfolioPosition {
                ticker: "PSA".into(),
                sector: REITSector::SelfStorage,
                shares: 30.0,
                cost_basis: 280.0,
                current_price: 300.0,
                weight: 0.25,
                market_value: 9000.0,
                unrealized_pnl: 600.0,
                unrealized_return_pct: 7.14,
                ratios: Some(make_test_ratios(0.035, 18.0, 4.0, 5.0)),
            },
        ]
    }

    #[test]
    fn test_compute_sector_weights() {
        let positions = sample_positions();
        let weights = compute_sector_weights(&positions);
        assert!((weights[&REITSector::Residential] - 0.4).abs() < 1e-9);
        assert!((weights[&REITSector::DataCenter] - 0.35).abs() < 1e-9);
        assert!((weights[&REITSector::SelfStorage] - 0.25).abs() < 1e-9);
    }

    #[test]
    fn test_compute_diversification_score() {
        let positions = sample_positions();
        let score = compute_diversification_score(&positions).unwrap();
        // HHI = 0.4^2 + 0.35^2 + 0.25^2 = 0.16 + 0.1225 + 0.0625 = 0.345
        // div = 1 - 0.345 = 0.655
        assert!((score - 0.655).abs() < 0.001);
    }

    #[test]
    fn test_compute_diversification_score_empty() {
        let result = compute_diversification_score(&[]);
        assert!(result.is_err());
    }

    #[test]
    fn test_compute_sector_diversification_score() {
        let positions = sample_positions();
        let score = compute_sector_diversification_score(&positions).unwrap();
        // Each position is in a unique sector, so sector HHI = same as position HHI
        assert!((score - 0.655).abs() < 0.001);
    }

    #[test]
    fn test_validate_weights_valid() {
        let positions = sample_positions();
        assert!(validate_weights(&positions, 0.01).is_ok());
    }

    #[test]
    fn test_validate_weights_invalid_sum() {
        let mut positions = sample_positions();
        positions[0].weight = 0.5;
        // Total = 0.5 + 0.35 + 0.25 = 1.1
        assert!(validate_weights(&positions, 0.01).is_err());
    }

    #[test]
    fn test_weighted_average_ratio() {
        let positions = sample_positions();
        let avg_div = weighted_average_ratio(&positions, |r| r.dividend_yield).unwrap();
        // 0.4*0.04 + 0.35*0.03 + 0.25*0.035 = 0.016 + 0.0105 + 0.00875 = 0.03525
        assert!((avg_div - 0.03525).abs() < 0.0001);
    }

    #[test]
    fn test_weighted_average_no_ratios() {
        let positions = vec![PortfolioPosition {
            ticker: "X".into(),
            sector: REITSector::Specialty,
            shares: 10.0,
            cost_basis: 100.0,
            current_price: 100.0,
            weight: 1.0,
            market_value: 1000.0,
            unrealized_pnl: 0.0,
            unrealized_return_pct: 0.0,
            ratios: None,
        }];
        let result = weighted_average_ratio(&positions, |r| r.dividend_yield);
        assert!(result.is_err());
    }

    #[test]
    fn test_compute_sharpe_ratio() {
        let returns = vec![0.01, 0.02, 0.03, -0.01, 0.015, 0.025, 0.005, -0.005, 0.02, 0.01];
        let sharpe = compute_sharpe_ratio(&returns, 0.02).unwrap();
        assert!(sharpe.is_finite());
    }

    #[test]
    fn test_compute_sharpe_ratio_insufficient_data() {
        let returns = vec![0.01];
        let result = compute_sharpe_ratio(&returns, 0.02);
        assert!(result.is_err());
    }

    #[test]
    fn test_compute_sortino_ratio() {
        let returns = vec![0.01, 0.02, 0.03, -0.01, 0.015, 0.025, 0.005, -0.005, 0.02, 0.01];
        let sortino = compute_sortino_ratio(&returns, 0.02, 0.0).unwrap();
        assert!(sortino.is_finite());
        // Sanity: Sharpe should also be finite.
        let sharpe = compute_sharpe_ratio(&returns, 0.02).unwrap();
        assert!(sharpe.is_finite());
    }

    #[test]
    fn test_compute_treynor_ratio() {
        let returns = vec![0.01, 0.02, 0.03, -0.01, 0.015, 0.025, 0.005, -0.005, 0.02, 0.01];
        let benchmark = vec![0.005, 0.01, 0.015, -0.005, 0.01, 0.02, 0.0, -0.003, 0.015, 0.008];
        let treynor = compute_treynor_ratio(&returns, &benchmark, 0.02).unwrap();
        assert!(treynor.is_finite());
    }

    #[test]
    fn test_compute_beta() {
        let portfolio = vec![0.02, 0.04, 0.06, -0.02, 0.03, 0.05, 0.01, -0.01, 0.04, 0.02];
        let benchmark = vec![0.01, 0.02, 0.03, -0.01, 0.015, 0.025, 0.005, -0.005, 0.02, 0.01];
        let beta = compute_beta(&portfolio, &benchmark).unwrap();
        assert!(beta > 0.0);
        assert!((beta - 2.0).abs() < 0.1); // Should be close to 2x
    }

    #[test]
    fn test_compute_alpha() {
        let portfolio = vec![0.02, 0.04, 0.06, -0.02, 0.03];
        let benchmark = vec![0.01, 0.02, 0.03, -0.01, 0.015];
        let alpha = compute_alpha(&portfolio, &benchmark, 0.01).unwrap();
        assert!(alpha.is_finite());
    }

    #[test]
    fn test_compute_correlation_matrix() {
        let returns = vec![
            vec![0.01, 0.02, 0.03, -0.01, 0.015],
            vec![0.005, 0.01, 0.015, -0.005, 0.008],
            vec![0.03, -0.01, 0.02, 0.01, 0.04],
        ];
        let corr = compute_correlation_matrix(&returns).unwrap();
        assert_eq!(corr.nrows(), 3);
        assert_eq!(corr.ncols(), 3);

        // Diagonal should be ~1.0
        assert!((corr[[0, 0]] - 1.0).abs() < 1e-6);
        assert!((corr[[1, 1]] - 1.0).abs() < 1e-6);

        // Matrix should be symmetric
        assert!((corr[[0, 1]] - corr[[1, 0]]).abs() < 1e-9);
    }

    #[test]
    fn test_assess_portfolio_risk() {
        let rating = assess_portfolio_risk(5.0, 0.7, 0.04);
        assert!(rating <= RiskRating::Medium);

        let high_risk = assess_portfolio_risk(8.0, 0.2, 0.09);
        assert!(high_risk >= RiskRating::High);
    }

    #[test]
    fn test_compute_portfolio_analytics() {
        let positions = sample_positions();
        let analytics = compute_portfolio_analytics(&positions, None, None, 0.02).unwrap();
        assert!(analytics.total_value > 0.0);
        assert!(analytics.diversification_score > 0.0);
        assert!(analytics.weighted_avg_dividend_yield > 0.0);
    }

    #[test]
    fn test_compute_portfolio_analytics_with_returns() {
        let positions = sample_positions();
        let pr = vec![0.01, 0.02, -0.005, 0.015, 0.008];
        let br = vec![0.005, 0.01, -0.002, 0.008, 0.004];
        let analytics = compute_portfolio_analytics(&positions, Some(&br), Some(&pr), 0.02).unwrap();
        assert!(analytics.sharpe_ratio.is_some());
        assert!(analytics.sortino_ratio.is_some());
        assert!(analytics.treynor_ratio.is_some());
    }

    #[test]
    fn test_compute_portfolio_analytics_empty() {
        let result = compute_portfolio_analytics(&[], None, None, 0.02);
        assert!(result.is_err());
    }
}
