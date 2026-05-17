//! # Backtesting Engine
//!
//! Full backtesting engine for RL agents against historical data,
//! with performance reporting, trade logging, and multi-strategy support.

use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};

/// A completed trade record.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Trade {
    pub id: u64,
    pub timestamp: DateTime<Utc>,
    pub direction: TradeDirection,
    pub asset: String,
    pub entry_price: f64,
    pub exit_price: f64,
    pub quantity: f64,
    pub pnl: f64,
    pub pnl_pct: f64,
    pub commission: f64,
    pub slippage: f64,
    pub hold_bars: u64,
    pub entry_reason: String,
    pub exit_reason: String,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum TradeDirection {
    Long,
    Short,
}

impl std::fmt::Display for TradeDirection {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            TradeDirection::Long => write!(f, "LONG"),
            TradeDirection::Short => write!(f, "SHORT"),
        }
    }
}

/// Trade log that records all trades.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TradeLog {
    trades: Vec<Trade>,
    next_id: u64,
}

impl TradeLog {
    pub fn new() -> Self {
        Self { trades: Vec::new(), next_id: 1 }
    }

    pub fn record(&mut self, trade: Trade) {
        self.trades.push(trade);
    }

    pub fn trades(&self) -> &[Trade] {
        &self.trades
    }

    pub fn len(&self) -> usize {
        self.trades.len()
    }

    pub fn is_empty(&self) -> bool {
        self.trades.is_empty()
    }

    pub fn total_pnl(&self) -> f64 {
        self.trades.iter().map(|t| t.pnl).sum()
    }

    pub fn winning_trades(&self) -> Vec<&Trade> {
        self.trades.iter().filter(|t| t.pnl > 0.0).collect()
    }

    pub fn losing_trades(&self) -> Vec<&Trade> {
        self.trades.iter().filter(|t| t.pnl <= 0.0).collect()
    }

    pub fn win_rate(&self) -> f64 {
        if self.trades.is_empty() { return 0.0; }
        self.winning_trades().len() as f64 / self.trades.len() as f64
    }

    pub fn avg_win(&self) -> f64 {
        let wins = self.winning_trades();
        if wins.is_empty() { return 0.0; }
        wins.iter().map(|t| t.pnl).sum::<f64>() / wins.len() as f64
    }

    pub fn avg_loss(&self) -> f64 {
        let losses = self.losing_trades();
        if losses.is_empty() { return 0.0; }
        losses.iter().map(|t| t.pnl).sum::<f64>() / losses.len() as f64
    }

    /// Profit factor (gross profit / gross loss).
    pub fn profit_factor(&self) -> f64 {
        let gross_profit: f64 = self.winning_trades().iter().map(|t| t.pnl).sum();
        let gross_loss: f64 = self.losing_trades().iter().map(|t| t.pnl.abs()).sum();
        if gross_loss == 0.0 { return f64::INFINITY; }
        gross_profit / gross_loss
    }
}

/// Slippage model for realistic execution.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SlippageModel {
    /// Fixed slippage in basis points.
    pub fixed_bps: f64,
    /// Proportional slippage (fraction of price).
    pub proportional: f64,
    /// Maximum slippage as fraction of price.
    pub max_slippage: f64,
}

impl SlippageModel {
    pub fn new(fixed_bps: f64, proportional: f64) -> Self {
        Self { fixed_bps, proportional, max_slippage: 0.01 }
    }

    pub fn none() -> Self {
        Self::new(0.0, 0.0)
    }

    /// Calculate slippage for a trade.
    pub fn calculate(&self, price: f64, quantity: f64) -> f64 {
        let fixed = price * self.fixed_bps / 10_000.0;
        let prop = price * self.proportional * (quantity / 1_000.0).min(1.0);
        let slippage = fixed + prop;
        slippage.min(price * self.max_slippage)
    }
}

impl Default for SlippageModel {
    fn default() -> Self {
        Self::new(1.0, 0.001) // 1 bps fixed + 0.1% proportional
    }
}

/// Commission model.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CommissionModel {
    /// Per-share commission.
    pub per_share: f64,
    /// Minimum commission per trade.
    pub min_commission: f64,
    /// Commission as percentage of notional value.
    pub percentage: f64,
}

impl CommissionModel {
    pub fn new(per_share: f64, percentage: f64) -> Self {
        Self { per_share, min_commission: 0.0, percentage }
    }

    pub fn none() -> Self {
        Self::new(0.0, 0.0)
    }

    /// Calculate commission for a trade.
    pub fn calculate(&self, price: f64, quantity: f64) -> f64 {
        let per_share_cost = self.per_share * quantity.abs();
        let pct_cost = price * quantity.abs() * self.percentage;
        (per_share_cost + pct_cost).max(self.min_commission)
    }
}

impl Default for CommissionModel {
    fn default() -> Self {
        Self::new(0.005, 0.001) // $0.005/share + 0.1%
    }
}

/// Performance report from a backtest.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PerformanceReport {
    pub total_return: f64,
    pub annualized_return: f64,
    pub sharpe_ratio: f64,
    pub sortino_ratio: f64,
    pub max_drawdown: f64,
    pub max_drawdown_duration_bars: u64,
    pub win_rate: f64,
    pub profit_factor: f64,
    pub avg_win: f64,
    pub avg_loss: f64,
    pub total_trades: u64,
    pub avg_hold_bars: f64,
    pub calmar_ratio: f64,
    pub volatility: f64,
    pub downside_deviation: f64,
}

impl PerformanceReport {
    /// Generate a performance report from an equity curve.
    pub fn from_equity_curve(equity: &[f64], risk_free_rate: f64) -> Self {
        if equity.len() < 2 {
            return Self::empty();
        }

        let returns: Vec<f64> = equity
            .windows(2)
            .map(|w| (w[1] - w[0]) / w[0])
            .collect();

        let total_return = (equity.last().unwrap() - equity.first().unwrap()) / equity.first().unwrap();
        let n_bars = equity.len() as f64;
        let annualized_return = if total_return > 0.0 {
            (1.0 + total_return).powf(252.0 / n_bars) - 1.0
        } else {
            -((1.0 - total_return).powf(252.0 / n_bars) - 1.0)
        };

        let mean_return: f64 = returns.iter().sum::<f64>() / returns.len() as f64;
        let variance: f64 = returns
            .iter()
            .map(|r| (r - mean_return).powi(2))
            .sum::<f64>() / returns.len() as f64;
        let vol = variance.sqrt() * (252.0_f64).sqrt();

        let sharpe = if vol > 0.0 {
            ((mean_return * 252.0 - risk_free_rate) / vol) as f64
        } else { 0.0 };

        // Sortino
        let downside: Vec<f64> = returns.iter().filter(|&&r| r < 0.0).cloned().collect();
        let downside_dev = if !downside.is_empty() {
            let ds_mean = downside.iter().sum::<f64>() / downside.len() as f64;
            (downside.iter().map(|r| (r - ds_mean).powi(2)).sum::<f64>() / downside.len() as f64)
                .sqrt() * (252.0_f64).sqrt()
        } else { 0.0 };
        let sortino = if downside_dev > 0.0 {
            (mean_return * 252.0 - risk_free_rate) / downside_dev
        } else { 0.0 };

        // Max drawdown
        let mut peak = equity[0];
        let mut max_dd = 0.0;
        let mut max_dd_duration = 0u64;
        let mut current_dd_duration = 0u64;

        for &val in equity {
            if val > peak {
                peak = val;
                current_dd_duration = 0;
            }
            let dd = (peak - val) / peak;
            if dd > max_dd { max_dd = dd; }
            if val < peak { current_dd_duration += 1; }
            if current_dd_duration > max_dd_duration { max_dd_duration = current_dd_duration; }
        }

        let calmar = if max_dd > 0.0 { annualized_return / max_dd } else { 0.0 };

        Self {
            total_return,
            annualized_return,
            sharpe_ratio: sharpe,
            sortino_ratio: sortino,
            max_drawdown: max_dd,
            max_drawdown_duration_bars: max_dd_duration,
            win_rate: 0.0,
            profit_factor: 0.0,
            avg_win: 0.0,
            avg_loss: 0.0,
            total_trades: 0,
            avg_hold_bars: 0.0,
            calmar_ratio: calmar,
            volatility: vol,
            downside_deviation: downside_dev,
        }
    }

    pub fn empty() -> Self {
        Self {
            total_return: 0.0, annualized_return: 0.0, sharpe_ratio: 0.0,
            sortino_ratio: 0.0, max_drawdown: 0.0, max_drawdown_duration_bars: 0,
            win_rate: 0.0, profit_factor: 0.0, avg_win: 0.0, avg_loss: 0.0,
            total_trades: 0, avg_hold_bars: 0.0, calmar_ratio: 0.0,
            volatility: 0.0, downside_deviation: 0.0,
        }
    }
}

impl std::fmt::Display for PerformanceReport {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        writeln!(f, "╔══════════════════════════════════════╗")?;
        writeln!(f, "║       BACKTEST PERFORMANCE            ║")?;
        writeln!(f, "╠══════════════════════════════════════╣")?;
        writeln!(f, "║ Total Return:     {:>10.2}%      ║", self.total_return * 100.0)?;
        writeln!(f, "║ Annualized:       {:>10.2}%      ║", self.annualized_return * 100.0)?;
        writeln!(f, "║ Sharpe Ratio:     {:>10.2}        ║", self.sharpe_ratio)?;
        writeln!(f, "║ Sortino Ratio:    {:>10.2}        ║", self.sortino_ratio)?;
        writeln!(f, "║ Max Drawdown:     {:>10.2}%      ║", self.max_drawdown * 100.0)?;
        writeln!(f, "║ Calmar Ratio:     {:>10.2}        ║", self.calmar_ratio)?;
        writeln!(f, "║ Volatility:       {:>10.2}%      ║", self.volatility * 100.0)?;
        writeln!(f, "║ Win Rate:         {:>10.2}%      ║", self.win_rate * 100.0)?;
        writeln!(f, "║ Profit Factor:    {:>10.2}        ║", self.profit_factor)?;
        writeln!(f, "║ Total Trades:     {:>10}         ║", self.total_trades)?;
        writeln!(f, "╚══════════════════════════════════════╝")
    }
}

/// Result from a backtest run.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BacktestResult {
    pub report: PerformanceReport,
    pub equity_curve: Vec<f64>,
    pub trade_log: TradeLog,
    pub strategy_name: String,
    pub start_date: DateTime<Utc>,
    pub end_date: DateTime<Utc>,
}

impl BacktestResult {
    /// Enrich the report with trade-level metrics.
    pub fn with_trade_metrics(mut self) -> Self {
        self.report.total_trades = self.trade_log.len() as u64;
        self.report.win_rate = self.trade_log.win_rate();
        self.report.avg_win = self.trade_log.avg_win();
        self.report.avg_loss = self.trade_log.avg_loss();
        self.report.profit_factor = self.trade_log.profit_factor();
        self.report.avg_hold_bars = if !self.trade_log.is_empty() {
            self.trade_log.trades().iter().map(|t| t.hold_bars).sum::<u64>() as f64
                / self.trade_log.len() as f64
        } else { 0.0 };
        self
    }
}

/// Multi-strategy backtesting across multiple strategies.
pub struct MultiStrategyBacktest {
    results: Vec<BacktestResult>,
}

impl MultiStrategyBacktest {
    pub fn new() -> Self {
        Self { results: Vec::new() }
    }

    pub fn add_result(&mut self, result: BacktestResult) {
        self.results.push(result);
    }

    pub fn results(&self) -> &[BacktestResult] {
        &self.results
    }

    /// Get the best strategy by Sharpe ratio.
    pub fn best_by_sharpe(&self) -> Option<&BacktestResult> {
        self.results
            .iter()
            .max_by(|a, b| a.report.sharpe_ratio.partial_cmp(&b.report.sharpe_ratio).unwrap())
    }

    /// Get the best strategy by total return.
    pub fn best_by_return(&self) -> Option<&BacktestResult> {
        self.results
            .iter()
            .max_by(|a, b| a.report.total_return.partial_cmp(&b.report.total_return).unwrap())
    }

    /// Correlation matrix of strategy returns.
    pub fn correlation_matrix(&self) -> Vec<Vec<f64>> {
        let n = self.results.len();
        let mut matrix = vec![vec![1.0; n]; n];
        for i in 0..n {
            for j in (i + 1)..n {
                let corr = Self::pearson_corr(&self.results[i].equity_curve, &self.results[j].equity_curve);
                matrix[i][j] = corr;
                matrix[j][i] = corr;
            }
        }
        matrix
    }

    fn pearson_corr(a: &[f64], b: &[f64]) -> f64 {
        let n = a.len().min(b.len());
        if n < 2 { return 0.0; }
        let a_mean: f64 = a[..n].iter().sum::<f64>() / n as f64;
        let b_mean: f64 = b[..n].iter().sum::<f64>() / n as f64;
        let cov: f64 = a[..n].iter().zip(b[..n].iter())
            .map(|(x, y)| (x - a_mean) * (y - b_mean))
            .sum::<f64>() / (n - 1) as f64;
        let var_a: f64 = a[..n].iter().map(|x| (x - a_mean).powi(2)).sum::<f64>() / (n - 1) as f64;
        let var_b: f64 = b[..n].iter().map(|x| (x - b_mean).powi(2)).sum::<f64>() / (n - 1) as f64;
        let denom = (var_a * var_b).sqrt();
        if denom < 1e-10 { return 0.0; }
        cov / denom
    }
}

/// Simple backtesting engine (synchronous, single-threaded).
pub struct BacktestEngine;

impl BacktestEngine {
    /// Run a simple buy-hold-sell backtest with pre-computed signals.
    pub fn run(
        prices: &[f64],
        signals: &[f64], // -1.0 = sell, 0.0 = hold, 1.0 = buy
        initial_capital: f64,
        commission: &CommissionModel,
        slippage: &SlippageModel,
    ) -> BacktestResult {
        let mut equity = vec![initial_capital];
        let mut cash = initial_capital;
        let mut position = 0.0;
        let mut trade_log = TradeLog::new();

        for i in 0..prices.len() {
            let price = prices[i];
            let signal = signals.get(i).copied().unwrap_or(0.0);

            let prev_pos = position;

            if signal > 0.1 && position <= 0.0 {
                // Buy
                let slip = slippage.calculate(price, position);
                let exec_price = price + slip;
                let afford = cash / exec_price;
                let comm = commission.calculate(exec_price, afford);
                position = afford;
                cash -= position * exec_price + comm;
            } else if signal < -0.1 && position > 0.0 {
                // Sell
                let slip = slippage.calculate(price, position);
                let exec_price = price - slip;
                let comm = commission.calculate(exec_price, position);
                let pnl = position * (exec_price - prices.get(i.saturating_sub(1)).copied().unwrap_or(exec_price));
                trade_log.record(Trade {
                    id: trade_log.next_id,
                    timestamp: Utc::now(),
                    direction: TradeDirection::Long,
                    asset: "DEFAULT".to_string(),
                    entry_price: prices.get(i.saturating_sub(1)).copied().unwrap_or(exec_price),
                    exit_price: exec_price,
                    quantity: position,
                    pnl,
                    pnl_pct: if prev_pos > 0.0 { pnl / (prev_pos * prices.get(i.saturating_sub(1)).copied().unwrap_or(exec_price)) } else { 0.0 },
                    commission: comm,
                    slippage: slip,
                    hold_bars: 1,
                    entry_reason: "signal".to_string(),
                    exit_reason: "signal".to_string(),
                });
                cash += position * exec_price - comm;
                position = 0.0;
            }

            equity.push(cash + position * price);
        }

        let mut report = PerformanceReport::from_equity_curve(&equity, 0.05);
        report.total_trades = trade_log.len() as u64;
        report.win_rate = trade_log.win_rate();
        report.profit_factor = trade_log.profit_factor();

        BacktestResult {
            report,
            equity_curve: equity,
            trade_log,
            strategy_name: "signal_based".to_string(),
            start_date: Utc::now(),
            end_date: Utc::now(),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn sample_prices() -> Vec<f64> {
        vec![100.0, 101.0, 99.0, 102.0, 103.0, 101.0, 105.0, 104.0, 107.0, 110.0]
    }

    #[test]
    fn test_trade_log() {
        let mut log = TradeLog::new();
        log.record(Trade {
            id: 1, timestamp: Utc::now(), direction: TradeDirection::Long,
            asset: "AAPL".to_string(), entry_price: 100.0, exit_price: 110.0,
            quantity: 10.0, pnl: 100.0, pnl_pct: 0.1, commission: 1.0,
            slippage: 0.5, hold_bars: 5, entry_reason: "signal".to_string(),
            exit_reason: "signal".to_string(),
        });
        assert_eq!(log.len(), 1);
        assert!((log.total_pnl() - 100.0).abs() < 0.01);
        assert!((log.win_rate() - 1.0).abs() < 0.01);
    }

    #[test]
    fn test_slippage_model() {
        let slip = SlippageModel::new(1.0, 0.001);
        let cost = slip.calculate(100.0, 1000.0);
        assert!(cost > 0.0);
    }

    #[test]
    fn test_commission_model() {
        let comm = CommissionModel::new(0.005, 0.001);
        let cost = comm.calculate(100.0, 100.0);
        assert!(cost > 0.0);
    }

    #[test]
    fn test_performance_report() {
        let equity: Vec<f64> = (0..100).map(|i| {
            100_000.0 * (1.0 + 0.0001 * i as f64 + 0.001 * (i as f64 % 10.0 - 5.0))
        }).collect();
        let report = PerformanceReport::from_equity_curve(&equity, 0.05);
        let display = format!("{}", report);
        assert!(display.contains("Sharpe"));
    }

    #[test]
    fn test_backtest_engine() {
        let prices = sample_prices();
        let signals = vec![0.0, 1.0, 0.0, 0.0, 0.0, 0.0, -1.0, 0.0, 0.0, 0.0];
        let result = BacktestEngine::run(
            &prices, &signals, 100_000.0,
            &CommissionModel::default(),
            &SlippageModel::default(),
        );
        assert!(result.equity_curve.len() > 1);
    }

    #[test]
    fn test_multi_strategy() {
        let mut ms = MultiStrategyBacktest::new();
        let prices = sample_prices();
        let signals = vec![0.0, 1.0, 0.0, 0.0, 0.0, 0.0, -1.0, 0.0, 0.0, 0.0];
        let result = BacktestEngine::run(&prices, &signals, 100_000.0, &CommissionModel::none(), &SlippageModel::none());
        ms.add_result(result);
        assert!(ms.best_by_return().is_some());
    }
}
