/// Evaluation Metrics — PnL, Sharpe Ratio, Maximum Drawdown.

/// Compute cumulative PnL from a list of per-step returns.
///
/// - `returns`: list of per-step PnL values
///
/// Returns: cumulative PnL
pub fn compute_pnl(returns: &[f64]) -> f64 {
    returns.iter().sum()
}

/// Compute annualised Sharpe ratio.
///
/// - `returns`: list of per-step returns
/// - `risk_free_rate`: annualised risk-free rate
/// - `periods_per_year`: number of trading periods per year
///
/// Returns: annualised Sharpe ratio
pub fn compute_sharpe_ratio(returns: &[f64], risk_free_rate: f64, periods_per_year: usize) -> f64 {
    if returns.len() < 2 {
        return 0.0;
    }

    let per_period_rf = risk_free_rate / periods_per_year as f64;
    let n = returns.len();

    let mean_excess: f64 = returns.iter().map(|r| r - per_period_rf).sum::<f64>() / n as f64;

    let variance: f64 = returns
        .iter()
        .map(|r| {
            let diff = r - per_period_rf - mean_excess;
            diff * diff
        })
        .sum::<f64>()
        / (n - 1) as f64;

    let std_excess = variance.sqrt();
    if std_excess < 1e-10 {
        return 0.0;
    }

    mean_excess / std_excess * (periods_per_year as f64).sqrt()
}

/// Compute maximum drawdown from a list of per-step returns.
///
/// - `returns`: list of per-step returns
///
/// Returns: maximum drawdown (positive number, e.g. 0.15 = 15%)
pub fn compute_max_drawdown(returns: &[f64]) -> f64 {
    if returns.len() < 2 {
        return 0.0;
    }

    let mut cumulative = 0.0;
    let mut running_max = 0.0;
    let mut max_dd = 0.0;

    for &r in returns {
        cumulative += r;
        if cumulative > running_max {
            running_max = cumulative;
        }
        let dd = running_max - cumulative;
        if dd > max_dd {
            max_dd = dd;
        }
    }

    max_dd
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_pnl() {
        let returns = vec![1.0, -0.5, 2.0, -1.0];
        let pnl = compute_pnl(&returns);
        assert!((pnl - 1.5).abs() < 1e-10);
    }

    #[test]
    fn test_pnl_empty() {
        let pnl = compute_pnl(&[]);
        assert!((pnl - 0.0).abs() < 1e-10);
    }

    #[test]
    fn test_sharpe() {
        let returns = vec![0.01, 0.02, 0.015, -0.005, 0.03, 0.025, 0.01, -0.01, 0.02, 0.015];
        let sr = compute_sharpe_ratio(&returns, 0.0, 252);
        assert!(sr > 0.0); // positive mean returns
        assert!(sr.is_finite());
    }

    #[test]
    fn test_sharpe_short() {
        let sr = compute_sharpe_ratio(&[0.01], 0.0, 252);
        assert!((sr - 0.0).abs() < 1e-10); // not enough data
    }

    #[test]
    fn test_max_drawdown() {
        let returns = vec![1.0, 2.0, -3.0, 1.0, 2.0];
        let mdd = compute_max_drawdown(&returns);
        assert!(mdd > 0.0);
        assert!(mdd.is_finite());
    }

    #[test]
    fn test_max_drawdown_monotonic() {
        let returns = vec![1.0, 2.0, 3.0, 4.0];
        let mdd = compute_max_drawdown(&returns);
        assert!((mdd - 0.0).abs() < 1e-10); // always going up
    }
}
