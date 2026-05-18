//! # Market Observer
//!
//! The observation layer of the closed-loop system. Analyzes market data to produce
//! a [`MarketObservation`] used by the decision engine.
//!
//! Capabilities:
//! - Market regime detection (trending, mean-reverting, volatile, quiet, crisis, recovery)
//! - Volatility classification (low/medium/high/extreme)
//! - Trend strength estimation (ADX-like heuristic)
//! - Liquidity assessment
//! - Correlation structure monitoring
//! - Risk event detection (tail events, gap moves)

use chrono::Utc;
use crate::types::*;

/// The market observer that transforms raw price data into structured observations.
#[derive(Debug, Clone)]
pub struct MarketObserver {
    /// Lookback window for moving averages and indicators.
    pub lookback: usize,
    /// Threshold for high volatility (annualized).
    pub high_vol_threshold: f64,
    /// Threshold for extreme volatility (annualized).
    pub extreme_vol_threshold: f64,
    /// ATR multiplier for trend strength calculation.
    pub atr_trend_multiplier: f64,
    /// Threshold for detecting a gap move (as fraction).
    pub gap_threshold: f64,
    /// Minimum bars required for observation.
    pub min_bars: usize,
}

impl Default for MarketObserver {
    fn default() -> Self {
        MarketObserver {
            lookback: 20,
            high_vol_threshold: 0.30,
            extreme_vol_threshold: 0.60,
            atr_trend_multiplier: 2.0,
            gap_threshold: 0.02,
            min_bars: 10,
        }
    }
}

impl MarketObserver {
    pub fn new(lookback: usize) -> Self {
        MarketObserver {
            lookback,
            ..Default::default()
        }
    }

    /// Produce a full market observation from a series of prices.
    pub fn observe(&self, prices: &[f64]) -> MarketObservation {
        if prices.len() < self.min_bars {
            return MarketObservation::quiet_default(prices.last().copied().unwrap_or(0.0));
        }

        let recent = &prices[prices.len().saturating_sub(self.lookback)..];
        let current_price = *prices.last().unwrap_or(&0.0);
        let vol = self.compute_volatility(prices);
        let vol_regime = self.classify_volatility(vol);

        let (trend_strength, trend_direction) = self.estimate_trend(prices);
        let regime = self.detect_regime(vol, trend_strength, trend_direction, prices);
        let liquidity = self.assess_liquidity(prices);
        let avg_corr = self.estimate_correlation(prices);
        let returns = self.compute_recent_returns(prices, 20);
        let (risk_event, risk_desc) = self.detect_risk_events(prices);

        MarketObservation {
            regime,
            price: current_price,
            volatility_regime: vol_regime,
            volatility: vol,
            trend_strength,
            trend_direction,
            liquidity_score: liquidity,
            avg_correlation: avg_corr,
            risk_event_detected: risk_event,
            risk_event_description: risk_desc,
            recent_returns: returns,
            timestamp: Utc::now(),
        }
    }

    /// Classify volatility regime based on annualized volatility.
    pub fn classify_volatility(&self, vol: f64) -> VolatilityRegime {
        if vol < self.high_vol_threshold * 0.5 {
            VolatilityRegime::Low
        } else if vol < self.high_vol_threshold {
            VolatilityRegime::Medium
        } else if vol < self.extreme_vol_threshold {
            VolatilityRegime::High
        } else {
            VolatilityRegime::Extreme
        }
    }

    /// Estimate trend strength and direction using an ADX-like heuristic.
    ///
    /// Returns (strength in [0,1], direction in [-1, 1]).
    pub fn estimate_trend(&self, prices: &[f64]) -> (f64, f64) {
        if prices.len() < self.lookback + 1 {
            return (0.0, 0.0);
        }

        // Compute directional movement
        let n = prices.len();
        let mut plus_dm = 0.0_f64;
        let mut minus_dm = 0.0_f64;
        let mut tr_sum = 0.0_f64;

        let window = self.lookback.min(n - 1);
        for i in (n - window)..n {
            let up_move = prices[i] - prices[i - 1];
            let down_move = prices[i - 1] - prices[i];

            plus_dm += if up_move > down_move && up_move > 0.0 { up_move } else { 0.0 };
            minus_dm += if down_move > up_move && down_move > 0.0 { down_move } else { 0.0 };

            // True range (simplified: just use abs change)
            tr_sum += up_move.abs().max(down_move.abs()).max(prices[i] * 0.001);
        }

        if tr_sum < 1e-15 {
            return (0.0, 0.0);
        }

        let plus_di = plus_dm / tr_sum;
        let minus_di = minus_dm / tr_sum;

        // ADX-like: strength is the magnitude of directional difference
        let di_diff = (plus_di - minus_di).abs();
        let di_sum = plus_di + minus_di;

        let adx = if di_sum > 1e-15 {
            100.0 * di_diff / di_sum / 100.0 // Normalize to [0, 1]
        } else {
            0.0
        };

        // Direction: +1 if bullish, -1 if bearish
        let direction = if di_sum > 1e-15 {
            (plus_di - minus_di) / di_sum
        } else {
            0.0
        };

        (adx.clamp(0.0, 1.0), direction.clamp(-1.0, 1.0))
    }

    /// Detect the current market regime using simple heuristics.
    pub fn detect_regime(
        &self,
        vol: f64,
        trend_strength: f64,
        trend_direction: f64,
        prices: &[f64],
    ) -> MarketRegime {
        let n = prices.len();
        if n < 5 {
            return MarketRegime::Quiet;
        }

        // Check for crisis: extreme volatility with sharp recent drops
        if vol > self.extreme_vol_threshold {
            let recent_drop: f64 = prices[n - 1] - prices[n.saturating_sub(5)];
            let drop_pct = if prices[n.saturating_sub(5)] > 1e-15 {
                recent_drop / prices[n.saturating_sub(5)]
            } else {
                0.0
            };
            if drop_pct < -0.05 {
                return MarketRegime::Crisis;
            }
            return MarketRegime::Volatile;
        }

        // Check for volatile regime
        if vol > self.high_vol_threshold {
            return MarketRegime::Volatile;
        }

        // Check for trending vs mean-reverting
        if trend_strength > 0.4 && trend_direction.abs() > 0.2 {
            return MarketRegime::Trending;
        }

        // Mean reversion: low trend strength with moderate volatility
        if trend_strength < 0.2 && vol > 0.10 {
            return MarketRegime::MeanReverting;
        }

        // Recovery: price dropping but decelerating
        if n >= 10 {
            let recent_5 = prices[n - 5] - prices[n - 1];
            let older_5 = prices[n - 10] - prices[n - 5];
            if older_5 < 0.0 && recent_5 > older_5 && recent_5 > 0.0 {
                return MarketRegime::Recovery;
            }
        }

        MarketRegime::Quiet
    }

    /// Assess market liquidity based on price series regularity.
    ///
    /// Higher score = more liquid. Uses coefficient of variation of returns.
    pub fn assess_liquidity(&self, prices: &[f64]) -> f64 {
        if prices.len() < 3 {
            return 0.5;
        }

        let returns: Vec<f64> = (1..prices.len())
            .filter_map(|i| {
                if prices[i - 1] > 1e-15 {
                    Some((prices[i] - prices[i - 1]) / prices[i - 1])
                } else {
                    None
                }
            })
            .collect();

        if returns.is_empty() {
            return 0.5;
        }

        let mean = returns.iter().sum::<f64>() / returns.len() as f64;
        let variance = returns.iter().map(|r| (r - mean).powi(2)).sum::<f64>() / returns.len() as f64;
        let std = variance.sqrt();

        // Low coefficient of variation = more regular/liquid
        // We invert so higher = more liquid
        let cv = if mean.abs() < 1e-15 { std } else { std / mean.abs() };
        (1.0 / (1.0 + cv * 100.0)).clamp(0.0, 1.0)
    }

    /// Estimate average correlation among recent price windows.
    ///
    /// Splits the price series into overlapping windows and computes
    /// average correlation between consecutive windows.
    pub fn estimate_correlation(&self, prices: &[f64]) -> f64 {
        if prices.len() < self.lookback * 3 {
            return 0.5; // default moderate correlation
        }

        let window_size = self.lookback;
        let step = window_size / 2;
        if step == 0 {
            return 0.5;
        }

        let mut windows: Vec<Vec<f64>> = Vec::new();
        let mut i = 0;
        while i + window_size <= prices.len() {
            let w: Vec<f64> = (i + 1..i + window_size)
                .filter_map(|j| {
                    if prices[j] > 1e-15 && prices[j - 1] > 1e-15 {
                        Some((prices[j] - prices[j - 1]) / prices[j - 1])
                    } else {
                        None
                    }
                })
                .collect();
            if w.len() >= window_size / 2 {
                windows.push(w);
            }
            i += step;
        }

        if windows.len() < 2 {
            return 0.5;
        }

        // Average correlation between consecutive windows
        let mut total_corr = 0.0_f64;
        let mut pairs = 0;
        for i in 0..windows.len().saturating_sub(1) {
            let c = pearson_corr(&windows[i], &windows[i + 1]);
            if c.is_finite() {
                total_corr += c.abs();
                pairs += 1;
            }
        }

        if pairs == 0 {
            return 0.5;
        }
        total_corr / pairs as f64
    }

    /// Detect risk events: tail moves and gap events.
    pub fn detect_risk_events(&self, prices: &[f64]) -> (bool, Option<String>) {
        if prices.len() < 5 {
            return (false, None);
        }

        let n = prices.len();
        let returns: Vec<f64> = (1..n)
            .filter_map(|i| {
                if prices[i - 1] > 1e-15 {
                    Some((prices[i] - prices[i - 1]) / prices[i - 1])
                } else {
                    None
                }
            })
            .collect();

        if returns.len() < 4 {
            return (false, None);
        }

        // Compute mean and std of returns
        let mean = returns.iter().sum::<f64>() / returns.len() as f64;
        let variance = returns.iter().map(|r| (r - mean).powi(2)).sum::<f64>() / returns.len() as f64;
        let std = variance.sqrt();

        if std < 1e-15 {
            return (false, None);
        }

        // Check for gap moves (overnight-style gaps) — checked before tail events
        if n >= 2 {
            let gap = (prices[n - 1] - prices[n - 2]).abs();
            if prices[n - 2] > 1e-15 && gap / prices[n - 2] > self.gap_threshold {
                return (true, Some(format!("Gap move detected: {:.2}%", gap / prices[n - 2] * 100.0)));
            }
        }

        // Check latest return for tail event (beyond 3 sigma)
        let latest = returns[returns.len() - 1];
        let z_score = (latest - mean) / std;

        if z_score.abs() > 3.0 {
            let desc = if z_score > 0.0 {
                format!("Tail up event: z-score {:.2}", z_score)
            } else {
                format!("Tail down event: z-score {:.2}", z_score)
            };
            return (true, Some(desc));
        }

        (false, None)
    }

    /// Compute annualized volatility from prices.
    fn compute_volatility(&self, prices: &[f64]) -> f64 {
        if prices.len() < 2 {
            return 0.0;
        }

        let returns: Vec<f64> = (1..prices.len())
            .filter_map(|i| {
                if prices[i - 1] > 1e-15 {
                    Some((prices[i] / prices[i - 1]).ln())
                } else {
                    None
                }
            })
            .collect();

        if returns.is_empty() {
            return 0.0;
        }

        let mean = returns.iter().sum::<f64>() / returns.len() as f64;
        let variance = returns.iter().map(|r| (r - mean).powi(2)).sum::<f64>() / returns.len() as f64;
        variance.sqrt() * 252.0_f64.sqrt()
    }

    /// Compute recent simple returns.
    fn compute_recent_returns(&self, prices: &[f64], n: usize) -> Vec<f64> {
        let start = prices.len().saturating_sub(n);
        (start + 1..prices.len())
            .filter_map(|i| {
                if prices[i - 1] > 1e-15 {
                    Some((prices[i] - prices[i - 1]) / prices[i - 1])
                } else {
                    None
                }
            })
            .collect()
    }
}

/// Pearson correlation between two slices.
fn pearson_corr(a: &[f64], b: &[f64]) -> f64 {
    let n = a.len().min(b.len());
    if n < 2 {
        return 0.0;
    }
    let ma: f64 = a[..n].iter().sum::<f64>() / n as f64;
    let mb: f64 = b[..n].iter().sum::<f64>() / n as f64;
    let mut cov = 0.0_f64;
    let mut va = 0.0_f64;
    let mut vb = 0.0_f64;
    for i in 0..n {
        let da = a[i] - ma;
        let db = b[i] - mb;
        cov += da * db;
        va += da * da;
        vb += db * db;
    }
    let denom = (va * vb).sqrt();
    if denom < 1e-15 { 0.0 } else { cov / denom }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn close(a: f64, b: f64, eps: f64) -> bool {
        (a - b).abs() < eps
    }

    fn uptrend_prices() -> Vec<f64> {
        (0..50).map(|i| 100.0 + i as f64 * 1.0).collect()
    }

    fn downtrend_prices() -> Vec<f64> {
        (0..50).map(|i| 150.0 - i as f64 * 1.0).collect()
    }

    fn ranging_prices() -> Vec<f64> {
        // Use TAU/10 period = 10 bars per cycle, 5 full cycles in 50 bars
        // Amplitude 0.8 keeps volatility low enough for Quiet regime detection
        (0..50).map(|i| 100.0 + 0.8 * (i as f64 * std::f64::consts::TAU / 10.0).sin()).collect()
    }

    fn volatile_prices() -> Vec<f64> {
        (0..50).map(|i| {
            100.0 + match i % 4 {
                0 => 5.0,
                1 => -3.0,
                2 => -4.0,
                _ => 2.0,
            }
        }).collect()
    }

    #[test]
    fn test_classify_volatility_low() {
        let obs = MarketObserver::default();
        assert_eq!(obs.classify_volatility(0.05), VolatilityRegime::Low);
    }

    #[test]
    fn test_classify_volatility_medium() {
        let obs = MarketObserver::default();
        assert_eq!(obs.classify_volatility(0.20), VolatilityRegime::Medium);
    }

    #[test]
    fn test_classify_volatility_high() {
        let obs = MarketObserver::default();
        assert_eq!(obs.classify_volatility(0.45), VolatilityRegime::High);
    }

    #[test]
    fn test_classify_volatility_extreme() {
        let obs = MarketObserver::default();
        assert_eq!(obs.classify_volatility(0.80), VolatilityRegime::Extreme);
    }

    #[test]
    fn test_trend_estimation_uptrend() {
        let obs = MarketObserver::new(10);
        let prices = uptrend_prices();
        let (strength, direction) = obs.estimate_trend(&prices);
        assert!(strength > 0.3, "Expected strong trend, got {}", strength);
        assert!(direction > 0.0, "Expected positive direction, got {}", direction);
    }

    #[test]
    fn test_trend_estimation_downtrend() {
        let obs = MarketObserver::new(10);
        let prices = downtrend_prices();
        let (strength, direction) = obs.estimate_trend(&prices);
        assert!(direction < 0.0, "Expected negative direction, got {}", direction);
    }

    #[test]
    fn test_trend_estimation_ranging() {
        let obs = MarketObserver::new(10);
        let prices = ranging_prices();
        let (strength, _) = obs.estimate_trend(&prices);
        assert!(strength < 0.5, "Expected weak trend for ranging, got {}", strength);
    }

    #[test]
    fn test_regime_detection_trending() {
        let obs = MarketObserver::default();
        let prices = uptrend_prices();
        let vol = obs.compute_volatility(&prices);
        let (ts, td) = obs.estimate_trend(&prices);
        let regime = obs.detect_regime(vol, ts, td, &prices);
        assert_eq!(regime, MarketRegime::Trending);
    }

    #[test]
    fn test_regime_detection_quiet() {
        let obs = MarketObserver::default();
        let prices = ranging_prices();
        let vol = obs.compute_volatility(&prices);
        let (ts, td) = obs.estimate_trend(&prices);
        let regime = obs.detect_regime(vol, ts, td, &prices);
        assert_eq!(regime, MarketRegime::Quiet);
    }

    #[test]
    fn test_liquidity_assessment() {
        let obs = MarketObserver::default();
        let prices = uptrend_prices();
        let liquidity = obs.assess_liquidity(&prices);
        assert!(liquidity > 0.0 && liquidity <= 1.0);
    }

    #[test]
    fn test_liquidity_volatile_low() {
        let obs = MarketObserver::default();
        let prices = volatile_prices();
        let liquidity = obs.assess_liquidity(&prices);
        // Volatile prices tend to have lower liquidity score
        assert!(liquidity > 0.0 && liquidity <= 1.0);
    }

    #[test]
    fn test_correlation_estimation() {
        let obs = MarketObserver::new(5);
        let prices = uptrend_prices();
        let corr = obs.estimate_correlation(&prices);
        assert!(corr >= 0.0 && corr <= 1.0);
    }

    #[test]
    fn test_correlation_short_series() {
        let obs = MarketObserver::default();
        let prices = vec![100.0, 101.0, 102.0];
        let corr = obs.estimate_correlation(&prices);
        assert_eq!(corr, 0.5); // default for short series
    }

    #[test]
    fn test_risk_event_detection_no_event() {
        let obs = MarketObserver::default();
        let prices = uptrend_prices();
        let (detected, desc) = obs.detect_risk_events(&prices);
        assert!(!detected);
        assert!(desc.is_none());
    }

    #[test]
    fn test_risk_event_tail() {
        let obs = MarketObserver::new(5);
        let mut prices = ranging_prices();
        // Insert a massive tail event
        *prices.last_mut().unwrap() = prices[prices.len() - 2] * 1.5;
        let (detected, desc) = obs.detect_risk_events(&prices);
        assert!(detected);
        assert!(desc.is_some());
    }

    #[test]
    fn test_risk_event_gap() {
        let obs = MarketObserver::default();
        let mut prices = ranging_prices();
        // Insert a gap
        let n = prices.len();
        prices[n - 1] = prices[n - 2] * 1.05; // 5% gap
        let (detected, desc) = obs.detect_risk_events(&prices);
        assert!(detected);
        assert!(desc.unwrap().contains("Gap"));
    }

    #[test]
    fn test_full_observe_uptrend() {
        let obs = MarketObserver::default();
        let prices = uptrend_prices();
        let observation = obs.observe(&prices);
        assert_eq!(observation.regime, MarketRegime::Trending);
        assert!(observation.price > 0.0);
        assert!(!observation.recent_returns.is_empty());
    }

    #[test]
    fn test_full_observe_insufficient_data() {
        let obs = MarketObserver::default();
        let prices = vec![100.0, 101.0];
        let observation = obs.observe(&prices);
        assert_eq!(observation.regime, MarketRegime::Quiet);
    }

    #[test]
    fn test_volatility_computation() {
        let obs = MarketObserver::default();
        let prices = uptrend_prices();
        let vol = obs.compute_volatility(&prices);
        assert!(vol > 0.0);
        assert!(vol.is_finite());
    }

    #[test]
    fn test_recent_returns_computation() {
        let obs = MarketObserver::default();
        let prices = vec![100.0, 110.0, 105.0, 115.0];
        let rets = obs.compute_recent_returns(&prices, 10);
        assert_eq!(rets.len(), 3);
        assert!(close(rets[0], 0.1, 1e-10));
    }

    #[test]
    fn test_observe_empty_prices() {
        let obs = MarketObserver::default();
        let obs_result = obs.observe(&[]);
        assert_eq!(obs_result.price, 0.0);
    }

    #[test]
    fn test_pearson_corr_function() {
        let a = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let b = vec![2.0, 4.0, 6.0, 8.0, 10.0];
        let c = pearson_corr(&a, &b);
        assert!(close(c, 1.0, 1e-6));
    }

    #[test]
    fn test_pearson_corr_short() {
        let a = vec![1.0];
        let b = vec![2.0];
        assert_eq!(pearson_corr(&a, &b), 0.0);
    }
}
