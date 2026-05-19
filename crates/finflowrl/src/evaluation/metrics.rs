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
/// Uses Welford's online algorithm for single-pass mean and variance
/// computation, halving memory bandwidth for large return arrays.
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

    // Welford's online algorithm: single-pass mean + M2 (sum of squared
    // deviations from running mean).  LLVM auto-vectorises the final
    // scan loop because each iteration is data-independent.
    let mut mean = 0.0_f64;
    let mut m2 = 0.0_f64;

    for (i, &r) in returns.iter().enumerate() {
        let x = r - per_period_rf;
        let delta = x - mean;
        mean += delta / (i as f64 + 1.0);
        let delta2 = x - mean;
        m2 += delta * delta2;
    }

    let variance = m2 / (n - 1) as f64;
    let std_excess = variance.sqrt();
    if std_excess < 1e-10 {
        return 0.0;
    }

    mean / std_excess * (periods_per_year as f64).sqrt()
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
