//! Vault Analyzer — MegaVault PnL tracking and yield comparison.
//!
//! Provides tools for monitoring the dYdX MegaVault's historical performance,
//! computing yield metrics, comparing against benchmarks, and generating
//! risk-adjusted return analytics.
//!
//! # Features
//!
//! - **PnL Tracking**: Realized + unrealized PnL with fee attribution
//! - **Yield Comparison**: APY vs benchmark (risk-free rate, BTC hold)
//! - **Drawdown Analysis**: Peak-to-trough, underwater curve
//! - **Risk Metrics**: Volatility, Sortino, max drawdown, Calmar ratio

use crate::types::*;
use serde::{Deserialize, Serialize};

/// A single vault performance snapshot.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VaultSnapshot {
    /// Vault address or identifier.
    pub vault_id: String,
    /// Total value locked in USD.
    pub tvl_usd: f64,
    /// Net asset value per share.
    pub nav_per_share: f64,
    /// Cumulative PnL since inception (USD).
    pub cumulative_pnl: f64,
    /// PnL since last snapshot (USD).
    pub pnl_delta: f64,
    /// 24h yield as a percentage.
    pub yield_24h_pct: f64,
    /// 7d annualized yield (APY).
    pub yield_7d_apy: f64,
    /// 30d annualized yield (APY).
    pub yield_30d_apy: f64,
    /// Since inception APY.
    pub yield_inception_apy: f64,
    /// Number of active positions.
    pub open_positions: u32,
    /// Total commission earned by the vault (USD).
    pub commission_earned: f64,
    /// Unix timestamp (ms).
    pub timestamp_ms: i64,
}

/// Vault composition — how capital is allocated.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VaultComposition {
    pub vault_id: String,
    /// Breakdown by asset: amount allocated.
    pub allocations: Vec<AssetAllocation>,
    /// Cash/USDC reserve percentage.
    pub cash_reserve_pct: f64,
    /// Leverage factor.
    pub leverage_factor: f64,
    /// Unix timestamp (ms).
    pub timestamp_ms: i64,
}

/// Individual asset allocation within a vault.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AssetAllocation {
    pub asset: String,
    pub notional_usd: f64,
    pub weight_pct: f64,
    pub unrealized_pnl: f64,
    pub entry_price: f64,
    pub current_price: f64,
}

/// Yield comparison against benchmarks.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct YieldComparison {
    pub vault_id: String,
    pub vault_30d_apy: f64,
    pub vault_7d_apy: f64,
    /// BTC hold return (APY).
    pub btc_benchmark_apy: f64,
    /// ETH hold return (APY).
    pub eth_benchmark_apy: f64,
    /// Risk-free rate (e.g., T-bill APY).
    pub risk_free_apy: f64,
    /// Vault APY minus risk-free rate.
    pub excess_return_30d: f64,
    /// Sharpe-like ratio (excess return / vault volatility).
    pub sharpe_like: f64,
    /// Sortino-like ratio.
    pub sortino_like: f64,
    /// Vault outperforming benchmarks?
    pub is_outperforming: bool,
}

/// A full vault analytics report.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VaultReport {
    pub vault_id: String,
    pub current_snapshot: Option<VaultSnapshot>,
    pub composition: Option<VaultComposition>,
    pub yield_comparison: Option<YieldComparison>,
    /// Historical NAV curve.
    pub nav_history: Vec<NavPoint>,
    /// Drawdown analysis.
    pub drawdown_analysis: DrawdownAnalysis,
    /// Risk metrics.
    pub risk_metrics: VaultRiskMetrics,
}

/// NAV history point for charting.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NavPoint {
    pub timestamp_ms: i64,
    pub nav: f64,
    /// Drawdown from peak at this point (%).
    pub drawdown_pct: f64,
}

/// Drawdown analysis.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DrawdownAnalysis {
    pub current_drawdown_pct: f64,
    pub max_drawdown_pct: f64,
    pub max_drawdown_duration_bars: u32,
    pub average_drawdown_pct: f64,
    pub recovery_factor: f64, // CAGR / Max DD
    pub number_of_drawdowns: u32,
}

/// Risk metrics for the vault.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VaultRiskMetrics {
    /// Annualized volatility of daily returns.
    pub annualized_volatility: f64,
    /// Sharpe ratio (excess return / volatility).
    pub sharpe_ratio: f64,
    /// Sortino ratio (excess return / downside vol).
    pub sortino_ratio: f64,
    /// Calmar ratio (CAGR / max drawdown).
    pub calmar_ratio: f64,
    /// Worst single-day return (%).
    pub worst_day_pct: f64,
    /// Best single-day return (%).
    pub best_day_pct: f64,
    /// VaR at 95% confidence (daily %).
    pub var_95_pct: f64,
}

/// The vault analyzer engine.
pub struct VaultAnalyzer {
    /// Historical NAV snapshots.
    nav_history: Vec<NavPoint>,
    /// Historical daily returns.
    daily_returns: Vec<f64>,
}

impl Default for VaultAnalyzer {
    fn default() -> Self {
        Self::new()
    }
}

impl VaultAnalyzer {
    pub fn new() -> Self {
        Self {
            nav_history: Vec::new(),
            daily_returns: Vec::new(),
        }
    }

    /// Add a NAV observation.
    pub fn add_nav(&mut self, timestamp_ms: i64, nav: f64) {
        // Compute return from previous
        if let Some(prev) = self.nav_history.last() {
            let daily_ret = (nav - prev.nav) / prev.nav;
            self.daily_returns.push(daily_ret);
        }

        // Compute drawdown from peak
        let peak = self
            .nav_history
            .iter()
            .map(|p| p.nav)
            .fold(0.0_f64, f64::max)
            .max(nav);
        let drawdown = if peak > 0.0 {
            (peak - nav) / peak * 100.0
        } else {
            0.0
        };

        self.nav_history.push(NavPoint {
            timestamp_ms,
            nav,
            drawdown_pct: drawdown,
        });
    }

    /// Run a full analysis of the vault's performance.
    pub fn analyze(&self, vault_id: &str) -> VaultReport {
        let risk_metrics = self.compute_risk_metrics();
        let drawdown = self.compute_drawdown_analysis();

        VaultReport {
            vault_id: vault_id.to_string(),
            current_snapshot: None,
            composition: None,
            yield_comparison: None,
            nav_history: self.nav_history.clone(),
            drawdown_analysis: drawdown,
            risk_metrics,
        }
    }

    /// Compare vault yield against benchmarks.
    pub fn compare_yield(
        &self,
        vault_id: &str,
        vault_30d_apy: f64,
        vault_7d_apy: f64,
        btc_30d_apy: f64,
        eth_30d_apy: f64,
        risk_free_apy: f64,
    ) -> YieldComparison {
        let excess_return = vault_30d_apy - risk_free_apy;
        let sharpe_like = if risk_metrics_vol(&self.daily_returns) > f64::EPSILON {
            excess_return / risk_metrics_vol(&self.daily_returns)
        } else {
            0.0
        };
        let sortino_like = if downside_dev(&self.daily_returns) > f64::EPSILON {
            excess_return / downside_dev(&self.daily_returns)
        } else {
            0.0
        };

        YieldComparison {
            vault_id: vault_id.to_string(),
            vault_30d_apy,
            vault_7d_apy,
            btc_benchmark_apy: btc_30d_apy,
            eth_benchmark_apy: eth_30d_apy,
            risk_free_apy,
            excess_return_30d: excess_return,
            sharpe_like,
            sortino_like,
            is_outperforming: vault_30d_apy > btc_30d_apy && vault_30d_apy > eth_30d_apy,
        }
    }

    /// Compute drawdown analysis from NAV history.
    fn compute_drawdown_analysis(&self) -> DrawdownAnalysis {
        if self.nav_history.is_empty() {
            return DrawdownAnalysis::zero();
        }

        let mut max_dd = 0.0_f64;
        let mut _current_dd = 0.0_f64;
        let mut dd_sum = 0.0_f64;
        let mut dd_count = 0_u32;
        let mut peak = 0.0_f64;
        let mut max_dd_duration = 0_u32;
        let mut current_dd_duration = 0_u32;
        let mut in_drawdown = false;

        for point in &self.nav_history {
            if point.nav > peak {
                peak = point.nav;
                if in_drawdown {
                    in_drawdown = false;
                    max_dd_duration = max_dd_duration.max(current_dd_duration);
                    current_dd_duration = 0;
                }
            } else {
                let dd = (peak - point.nav) / peak * 100.0;
                max_dd = max_dd.max(dd);
                _current_dd = dd;

                if dd > 0.01 {
                    dd_sum += dd;
                    dd_count += 1;
                    if !in_drawdown {
                        in_drawdown = true;
                    }
                    current_dd_duration += 1;
                }
            }
        }
        max_dd_duration = max_dd_duration.max(current_dd_duration);

        let current_dd = self
            .nav_history
            .last()
            .map(|p| p.drawdown_pct)
            .unwrap_or(0.0);

        // Recovery factor: need total return and max DD
        let total_return = if self.nav_history.len() > 1 {
            (self.nav_history.last().unwrap().nav - self.nav_history.first().unwrap().nav)
                / self.nav_history.first().unwrap().nav
                * 100.0
        } else {
            0.0
        };
        let recovery_factor = if max_dd > 0.0 { total_return / max_dd } else { 0.0 };

        DrawdownAnalysis {
            current_drawdown_pct: current_dd,
            max_drawdown_pct: max_dd,
            max_drawdown_duration_bars: max_dd_duration,
            average_drawdown_pct: if dd_count > 0 { dd_sum / dd_count as f64 } else { 0.0 },
            recovery_factor,
            number_of_drawdowns: dd_count,
        }
    }

    /// Compute risk metrics from daily returns.
    fn compute_risk_metrics(&self) -> VaultRiskMetrics {
        if self.daily_returns.len() < 2 {
            return VaultRiskMetrics::zero();
        }

        let n = self.daily_returns.len() as f64;
        let mean = self.daily_returns.iter().sum::<f64>() / n;

        let variance = self
            .daily_returns
            .iter()
            .map(|r| (r - mean).powi(2))
            .sum::<f64>()
            / (n - 1.0);
        let annualized_vol = variance.sqrt() * (365.0_f64).sqrt() * 100.0;

        let worst = self.daily_returns.iter().cloned().fold(f64::INFINITY, f64::min) * 100.0;
        let best = self.daily_returns.iter().cloned().fold(f64::NEG_INFINITY, f64::max) * 100.0;

        // VaR 95%: 5th percentile of returns
        let mut sorted = self.daily_returns.clone();
        sorted.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
        let var_idx = ((5.0 / 100.0) * n).ceil() as usize;
        let var_95 = sorted.get(var_idx.min(sorted.len() - 1)).copied().unwrap_or(0.0) * 100.0;

        let dd_analysis = self.compute_drawdown_analysis();

        let cagr = if self.nav_history.len() > 1 {
            let first_nav = self.nav_history.first().unwrap().nav;
            let last_nav = self.nav_history.last().unwrap().nav;
            let days = self.nav_history.len() as f64;
            ((last_nav / first_nav).powf(365.0 / days) - 1.0) * 100.0
        } else {
            0.0
        };

        let sharpe = if annualized_vol > f64::EPSILON {
            cagr / annualized_vol
        } else {
            0.0
        };
        let sortino = if downside_dev(&self.daily_returns) > f64::EPSILON {
            cagr / (downside_dev(&self.daily_returns) * 100.0)
        } else {
            0.0
        };
        let calmar = if dd_analysis.max_drawdown_pct > 0.0 {
            cagr / dd_analysis.max_drawdown_pct
        } else {
            0.0
        };

        VaultRiskMetrics {
            annualized_volatility: annualized_vol,
            sharpe_ratio: sharpe,
            sortino_ratio: sortino,
            calmar_ratio: calmar,
            worst_day_pct: worst,
            best_day_pct: best,
            var_95_pct: var_95,
        }
    }
}

impl DrawdownAnalysis {
    pub fn zero() -> Self {
        Self {
            current_drawdown_pct: 0.0,
            max_drawdown_pct: 0.0,
            max_drawdown_duration_bars: 0,
            average_drawdown_pct: 0.0,
            recovery_factor: 0.0,
            number_of_drawdowns: 0,
        }
    }
}

impl VaultRiskMetrics {
    pub fn zero() -> Self {
        Self {
            annualized_volatility: 0.0,
            sharpe_ratio: 0.0,
            sortino_ratio: 0.0,
            calmar_ratio: 0.0,
            worst_day_pct: 0.0,
            best_day_pct: 0.0,
            var_95_pct: 0.0,
        }
    }
}

/// Compute annualized volatility from daily returns.
fn risk_metrics_vol(returns: &[f64]) -> f64 {
    if returns.len() < 2 {
        return 0.0;
    }
    let mean = returns.iter().sum::<f64>() / returns.len() as f64;
    let variance = returns
        .iter()
        .map(|r| (r - mean).powi(2))
        .sum::<f64>()
        / (returns.len() - 1) as f64;
    variance.sqrt() * (365.0_f64).sqrt()
}

/// Compute downside deviation from daily returns.
fn downside_dev(returns: &[f64]) -> f64 {
    if returns.is_empty() {
        return 0.0;
    }
    let downside: Vec<f64> = returns.iter().map(|r| r.min(0.0)).collect();
    let mean = downside.iter().sum::<f64>() / downside.len() as f64;
    downside
        .iter()
        .map(|r| (r - mean).powi(2))
        .sum::<f64>()
        .sqrt()
}

/// Generate mock NAV history for testing.
pub fn generate_mock_nav_history(days: u32, base_nav: f64, daily_return_mean: f64, daily_return_std: f64) -> Vec<(i64, f64)> {
    let mut nav = base_nav;
    let mut points = Vec::with_capacity(days as usize);

    for i in 0..days {
        let ret = daily_return_mean + (rand::random::<f64>() - 0.5) * 2.0 * daily_return_std;
        nav *= 1.0 + ret;
        let ts = chrono::Utc::now().timestamp_millis() - (days - i) as i64 * 86_400_000;
        points.push((ts, nav));
    }

    points
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_vault_analyzer_empty() {
        let analyzer = VaultAnalyzer::new();
        let report = analyzer.analyze("test-vault");
        assert_eq!(report.vault_id, "test-vault");
        assert!(report.nav_history.is_empty());
        assert_eq!(report.risk_metrics.annualized_volatility, 0.0);
    }

    #[test]
    fn test_vault_analyzer_with_data() {
        let mut analyzer = VaultAnalyzer::new();
        let points = generate_mock_nav_history(100, 100.0, 0.001, 0.01);
        for (ts, nav) in points {
            analyzer.add_nav(ts, nav);
        }

        let report = analyzer.analyze("megavault");
        assert_eq!(report.nav_history.len(), 100);
        assert!(report.risk_metrics.annualized_volatility > 0.0);
        assert!(report.drawdown_analysis.max_drawdown_pct >= 0.0);
    }

    #[test]
    fn test_yield_comparison() {
        let mut analyzer = VaultAnalyzer::new();
        for (ts, nav) in generate_mock_nav_history(50, 100.0, 0.001, 0.005) {
            analyzer.add_nav(ts, nav);
        }

        let comparison = analyzer.compare_yield(
            "vault-1", 25.0, 30.0, 15.0, 10.0, 5.0,
        );
        assert_eq!(comparison.vault_30d_apy, 25.0);
        assert_eq!(comparison.excess_return_30d, 20.0);
        assert!(comparison.is_outperforming); // 25 > 15 and 25 > 10
    }

    #[test]
    fn test_yield_comparison_underperforming() {
        let analyzer = VaultAnalyzer::new();
        let comparison = analyzer.compare_yield(
            "vault-2", 8.0, 10.0, 25.0, 15.0, 5.0,
        );
        assert!(!comparison.is_outperforming);
    }

    #[test]
    fn test_vault_snapshot_serde() {
        let snap = VaultSnapshot {
            vault_id: "mega-1".into(),
            tvl_usd: 50_000_000.0,
            nav_per_share: 1.15,
            cumulative_pnl: 5_000_000.0,
            pnl_delta: 50_000.0,
            yield_24h_pct: 0.1,
            yield_7d_apy: 30.0,
            yield_30d_apy: 25.0,
            yield_inception_apy: 20.0,
            open_positions: 12,
            commission_earned: 250_000.0,
            timestamp_ms: 1704067200000,
        };
        let json = serde_json::to_string(&snap).unwrap();
        let restored: VaultSnapshot = serde_json::from_str(&json).unwrap();
        assert_eq!(restored.vault_id, "mega-1");
        assert_eq!(restored.tvl_usd, 50_000_000.0);
    }

    #[test]
    fn test_vault_composition_serde() {
        let comp = VaultComposition {
            vault_id: "mega-1".into(),
            allocations: vec![
                AssetAllocation {
                    asset: "BTC".into(),
                    notional_usd: 20_000_000.0,
                    weight_pct: 40.0,
                    unrealized_pnl: 500_000.0,
                    entry_price: 65000.0,
                    current_price: 67500.0,
                },
            ],
            cash_reserve_pct: 10.0,
            leverage_factor: 1.5,
            timestamp_ms: 0,
        };
        let json = serde_json::to_string(&comp).unwrap();
        assert!(json.contains("BTC"));
    }

    #[test]
    fn test_drawdown_analysis_zero() {
        let dd = DrawdownAnalysis::zero();
        assert_eq!(dd.max_drawdown_pct, 0.0);
        assert_eq!(dd.number_of_drawdowns, 0);
    }

    #[test]
    fn test_risk_metrics_zero() {
        let rm = VaultRiskMetrics::zero();
        assert_eq!(rm.sharpe_ratio, 0.0);
    }

    #[test]
    fn test_mock_nav_history_valid() {
        let points = generate_mock_nav_history(50, 100.0, 0.001, 0.01);
        assert_eq!(points.len(), 50);
        for (ts, nav) in &points {
            assert!(*nav > 0.0);
            assert!(*ts > 0);
        }
    }

    #[test]
    fn test_nav_point_serde() {
        let np = NavPoint {
            timestamp_ms: 1704067200000,
            nav: 105.5,
            drawdown_pct: 0.5,
        };
        let json = serde_json::to_string(&np).unwrap();
        let restored: NavPoint = serde_json::from_str(&json).unwrap();
        assert_eq!(restored.nav, 105.5);
    }

    #[test]
    fn test_vault_report_serde() {
        let report = VaultReport {
            vault_id: "test".into(),
            current_snapshot: None,
            composition: None,
            yield_comparison: None,
            nav_history: vec![],
            drawdown_analysis: DrawdownAnalysis::zero(),
            risk_metrics: VaultRiskMetrics::zero(),
        };
        let json = serde_json::to_string(&report).unwrap();
        assert!(json.contains("test"));
    }
}
