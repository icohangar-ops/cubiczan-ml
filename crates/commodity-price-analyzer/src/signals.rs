//! # Trading Signal Generation
//!
//! Implements technical indicator computation and multi-signal scoring
//! for commodity trading decisions. Includes RSI, MACD, Bollinger Bands,
//! ATR, Stochastic oscillator, composite scoring, and position sizing.

use chrono::Utc;
use crate::types::{
    CommoditySignal, CommodityType, PositionRecommendation, PricePoint,
    SignalStrength,
};

// ---------------------------------------------------------------------------
// Technical Indicators Bundle
// ---------------------------------------------------------------------------

/// A snapshot of computed technical indicators.
#[derive(Debug, Clone)]
pub struct TechnicalIndicators {
    pub rsi: Option<f64>,
    pub macd_line: Option<f64>,
    pub macd_signal: Option<f64>,
    pub macd_histogram: Option<f64>,
    pub bollinger_upper: Option<f64>,
    pub bollinger_middle: Option<f64>,
    pub bollinger_lower: Option<f64>,
    pub atr: Option<f64>,
    pub stochastic_k: Option<f64>,
    pub stochastic_d: Option<f64>,
}

impl TechnicalIndicators {
    pub fn new() -> Self {
        TechnicalIndicators {
            rsi: None,
            macd_line: None,
            macd_signal: None,
            macd_histogram: None,
            bollinger_upper: None,
            bollinger_middle: None,
            bollinger_lower: None,
            atr: None,
            stochastic_k: None,
            stochastic_d: None,
        }
    }
}

impl Default for TechnicalIndicators {
    fn default() -> Self {
        Self::new()
    }
}

// ---------------------------------------------------------------------------
// Signal Generator
// ---------------------------------------------------------------------------

/// Multi-indicator trading signal generator for commodity prices.
pub struct SignalGenerator {
    /// Period for RSI calculation (default 14).
    pub rsi_period: usize,
    /// Fast EMA period for MACD (default 12).
    pub macd_fast: usize,
    /// Slow EMA period for MACD (default 26).
    pub macd_slow: usize,
    /// Signal line period for MACD (default 9).
    pub macd_signal_period: usize,
    /// Period for Bollinger Bands (default 20).
    pub bollinger_period: usize,
    /// Number of standard deviations for Bollinger Bands (default 2.0).
    pub bollinger_std_devs: f64,
    /// Period for ATR calculation (default 14).
    pub atr_period: usize,
    /// %K period for Stochastic (default 14).
    pub stochastic_k_period: usize,
    /// %D smoothing period for Stochastic (default 3).
    pub stochastic_d_period: usize,
}

impl Default for SignalGenerator {
    fn default() -> Self {
        SignalGenerator {
            rsi_period: 14,
            macd_fast: 12,
            macd_slow: 26,
            macd_signal_period: 9,
            bollinger_period: 20,
            bollinger_std_devs: 2.0,
            atr_period: 14,
            stochastic_k_period: 14,
            stochastic_d_period: 3,
        }
    }
}

impl SignalGenerator {
    /// Create a new signal generator with standard (12/26/9) MACD settings.
    pub fn new() -> Self {
        Self::default()
    }

    // -----------------------------------------------------------------------
    // RSI (Wilder's Smoothing)
    // -----------------------------------------------------------------------

    /// Compute RSI using Wilder's exponential moving average smoothing.
    /// Returns RSI in [0, 100], or None if insufficient data.
    pub fn compute_rsi(&self, prices: &[PricePoint]) -> Option<f64> {
        let closes: Vec<f64> = prices.iter().map(|p| p.close).collect();
        if closes.len() < self.rsi_period + 1 {
            return None;
        }

        let period = self.rsi_period;
        let alpha = 1.0 / period as f64; // Wilder's smoothing factor

        // Initial average gain/loss over first `period` changes
        let mut avg_gain = 0.0;
        let mut avg_loss = 0.0;
        for i in 1..=period {
            let change = closes[i] - closes[i - 1];
            if change > 0.0 {
                avg_gain += change;
            } else {
                avg_loss += change.abs();
            }
        }
        avg_gain /= period as f64;
        avg_loss /= period as f64;

        // Wilder's smoothing for remaining periods
        for i in (period + 1)..closes.len() {
            let change = closes[i] - closes[i - 1];
            let gain = if change > 0.0 { change } else { 0.0 };
            let loss = if change < 0.0 { change.abs() } else { 0.0 };
            avg_gain = alpha * gain + (1.0 - alpha) * avg_gain;
            avg_loss = alpha * loss + (1.0 - alpha) * avg_loss;
        }

        if avg_loss.abs() < 1e-15 {
            return Some(100.0);
        }
        let rs = avg_gain / avg_loss;
        Some(100.0 - 100.0 / (1.0 + rs))
    }

    // -----------------------------------------------------------------------
    // MACD (12/26/9 Standard)
    // -----------------------------------------------------------------------

    /// Compute MACD line, signal line, and histogram.
    /// Returns (macd_line, signal_line, histogram).
    pub fn compute_macd(&self, prices: &[PricePoint]) -> (Option<f64>, Option<f64>, Option<f64>) {
        let closes: Vec<f64> = prices.iter().map(|p| p.close).collect();

        if closes.len() < self.macd_slow + self.macd_signal_period {
            return (None, None, None);
        }

        let fast_ema = self.ema(&closes, self.macd_fast);
        let slow_ema = self.ema(&closes, self.macd_slow);

        // MACD line = fast EMA - slow EMA
        let macd_values: Vec<f64> = fast_ema.iter().zip(slow_ema.iter())
            .map(|(f, s)| f - s)
            .collect();

        // Signal line = EMA of MACD values
        let signal_values = self.ema(&macd_values, self.macd_signal_period);

        let macd_line = *macd_values.last().unwrap_or(&0.0);
        let signal_line = *signal_values.last().unwrap_or(&0.0);
        let histogram = macd_line - signal_line;

        (Some(macd_line), Some(signal_line), Some(histogram))
    }

    // -----------------------------------------------------------------------
    // Bollinger Bands (20-period, 2 std dev)
    // -----------------------------------------------------------------------

    /// Compute Bollinger Bands.
    /// Returns (upper_band, middle_band, lower_band).
    pub fn compute_bollinger_bands(&self, prices: &[PricePoint]) -> (Option<f64>, Option<f64>, Option<f64>) {
        let closes: Vec<f64> = prices.iter().map(|p| p.close).collect();

        if closes.len() < self.bollinger_period {
            return (None, None, None);
        }

        let slice = &closes[closes.len() - self.bollinger_period..];
        let middle = slice.iter().sum::<f64>() / slice.len() as f64;
        let variance = slice.iter().map(|c| (c - middle).powi(2)).sum::<f64>() / slice.len() as f64;
        let std_dev = variance.sqrt();

        let upper = middle + self.bollinger_std_devs * std_dev;
        let lower = middle - self.bollinger_std_devs * std_dev;

        (Some(upper), Some(middle), Some(lower))
    }

    // -----------------------------------------------------------------------
    // ATR (Average True Range)
    // -----------------------------------------------------------------------

    /// Compute Average True Range using Wilder's smoothing.
    pub fn compute_atr(&self, prices: &[PricePoint]) -> Option<f64> {
        if prices.len() < self.atr_period + 1 {
            return None;
        }

        let period = self.atr_period;
        let alpha = 1.0 / period as f64;

        // True Range series
        let mut tr_values: Vec<f64> = Vec::new();
        for i in 1..prices.len() {
            let high = prices[i].high;
            let low = prices[i].low;
            let prev_close = prices[i - 1].close;
            let tr = (high - low)
                .max((high - prev_close).abs())
                .max((low - prev_close).abs());
            tr_values.push(tr);
        }

        // Initial ATR as simple average of first `period` TR values
        let mut atr = tr_values[..period].iter().sum::<f64>() / period as f64;

        // Wilder's smoothing
        for i in period..tr_values.len() {
            atr = alpha * tr_values[i] + (1.0 - alpha) * atr;
        }

        Some(atr)
    }

    // -----------------------------------------------------------------------
    // Stochastic Oscillator (%K, %D)
    // -----------------------------------------------------------------------

    /// Compute Stochastic oscillator.
    /// Returns (%K, %D) in [0, 100].
    pub fn compute_stochastic(&self, prices: &[PricePoint]) -> (Option<f64>, Option<f64>) {
        if prices.len() < self.stochastic_k_period {
            return (None, None);
        }

        let period = self.stochastic_k_period;
        let slice = &prices[prices.len() - period..];

        let highest_high = slice.iter().map(|p| p.high).fold(f64::NEG_INFINITY, f64::max);
        let lowest_low = slice.iter().map(|p| p.low).fold(f64::INFINITY, f64::min);
        let current_close = prices.last().unwrap().close;

        let range = highest_high - lowest_low;
        if range.abs() < 1e-15 {
            return (Some(50.0), Some(50.0));
        }

        let k = (current_close - lowest_low) / range * 100.0;

        // %D is the SMA of %K values; for simplicity we compute it as a simple
        // approximation using only the current %K
        let d = k; // simplified; a full implementation would track %K history

        (Some(k), Some(d))
    }

    // -----------------------------------------------------------------------
    // Compute all indicators at once
    // -----------------------------------------------------------------------

    /// Compute all technical indicators for a price series.
    pub fn compute_all(&self, prices: &[PricePoint]) -> TechnicalIndicators {
        let rsi = self.compute_rsi(prices);
        let (macd_line, macd_signal, macd_histogram) = self.compute_macd(prices);
        let (bb_upper, bb_middle, bb_lower) = self.compute_bollinger_bands(prices);
        let atr = self.compute_atr(prices);
        let (stoch_k, stoch_d) = self.compute_stochastic(prices);

        TechnicalIndicators {
            rsi,
            macd_line,
            macd_signal,
            macd_histogram,
            bollinger_upper: bb_upper,
            bollinger_middle: bb_middle,
            bollinger_lower: bb_lower,
            atr,
            stochastic_k: stoch_k,
            stochastic_d: stoch_d,
        }
    }

    // -----------------------------------------------------------------------
    // Multi-signal scoring system (0-100)
    // -----------------------------------------------------------------------

    /// Generate a composite signal score from 0 (strong sell) to 100 (strong buy).
    /// 50 represents neutral.
    pub fn composite_score(&self, prices: &[PricePoint]) -> f64 {
        let indicators = self.compute_all(prices);
        let mut score = 50.0; // start neutral

        // RSI contribution (-25 to +25)
        if let Some(rsi) = indicators.rsi {
            if rsi < 30.0 {
                // Oversold → bullish
                score += (30.0 - rsi) / 30.0 * 25.0;
            } else if rsi > 70.0 {
                // Overbought → bearish
                score -= (rsi - 70.0) / 30.0 * 25.0;
            }
        }

        // MACD contribution (-25 to +25)
        if let (Some(hist), Some(_macd)) = (indicators.macd_histogram, indicators.macd_line) {
            // Normalize histogram: capped contribution
            let macd_signal = (hist * 100.0).clamp(-25.0, 25.0);
            score += macd_signal;
        }

        // Bollinger Bands contribution (-25 to +25)
        if let (Some(upper), Some(_middle), Some(lower)) = (indicators.bollinger_upper, indicators.bollinger_middle, indicators.bollinger_lower) {
            let last_close = prices.last().map(|p| p.close).unwrap_or(0.0);
            let band_range = upper - lower;
            if band_range.abs() > 1e-15 {
                let position = (last_close - lower) / band_range; // 0 to 1
                // Near lower band → bullish; near upper → bearish
                let bb_signal = (0.5 - position) * 50.0;
                score += bb_signal.clamp(-25.0, 25.0);
            }
        }

        // Stochastic contribution (-25 to +25)
        if let Some(k) = indicators.stochastic_k {
            if k < 20.0 {
                score += (20.0 - k) / 20.0 * 25.0;
            } else if k > 80.0 {
                score -= (k - 80.0) / 20.0 * 25.0;
            }
        }

        score.clamp(0.0, 100.0)
    }

    /// Convert a composite score (0-100) to a `SignalStrength`.
    pub fn score_to_signal(score: f64) -> SignalStrength {
        if score >= 75.0 {
            SignalStrength::StrongBuy
        } else if score >= 55.0 {
            SignalStrength::Buy
        } else if score >= 45.0 {
            SignalStrength::Neutral
        } else if score >= 25.0 {
            SignalStrength::Sell
        } else {
            SignalStrength::StrongSell
        }
    }

    /// Generate a full `CommoditySignal` with reasoning.
    pub fn generate_signal(&self, commodity: CommodityType, prices: &[PricePoint]) -> CommoditySignal {
        let score = self.composite_score(prices);
        let signal = Self::score_to_signal(score);
        let indicators = self.compute_all(prices);
        let confidence = (score - 50.0).abs() / 50.0; // 0 at neutral, 1 at extremes

        let mut reasoning_parts = Vec::new();
        let mut indicator_names = Vec::new();

        if let Some(rsi) = indicators.rsi {
            indicator_names.push(format!("RSI={:.1}", rsi));
            if rsi < 30.0 {
                reasoning_parts.push(format!("RSI at {:.1} indicates oversold conditions", rsi));
            } else if rsi > 70.0 {
                reasoning_parts.push(format!("RSI at {:.1} indicates overbought conditions", rsi));
            }
        }

        if let Some(hist) = indicators.macd_histogram {
            indicator_names.push(format!("MACD_hist={:.4}", hist));
            if hist > 0.0 {
                reasoning_parts.push("MACD histogram is positive (bullish momentum)".into());
            } else {
                reasoning_parts.push("MACD histogram is negative (bearish momentum)".into());
            }
        }

        if let Some(atr) = indicators.atr {
            indicator_names.push(format!("ATR={:.4}", atr));
        }

        if let Some(k) = indicators.stochastic_k {
            indicator_names.push(format!("Stoch%K={:.1}", k));
        }

        let reasoning = if reasoning_parts.is_empty() {
            "Mixed signals, no strong directional bias".into()
        } else {
            reasoning_parts.join("; ")
        };

        CommoditySignal {
            commodity,
            signal,
            confidence: confidence.clamp(0.0, 1.0),
            reasoning,
            indicators: indicator_names,
            timestamp: Utc::now(),
        }
    }

    // -----------------------------------------------------------------------
    // Position Sizing
    // -----------------------------------------------------------------------

    /// Compute a position sizing recommendation based on signal score and ATR.
    pub fn position_sizing(
        &self,
        prices: &[PricePoint],
        portfolio_value: f64,
        risk_per_trade: f64,
    ) -> PositionRecommendation {
        let score = self.composite_score(prices);
        let signal = Self::score_to_signal(score);
        let last_price = prices.last().map(|p| p.close).unwrap_or(0.0);
        let atr = self.compute_atr(prices).unwrap_or(last_price * 0.02);

        // Determine direction multiplier
        let direction = match signal {
            SignalStrength::StrongBuy => 1.0,
            SignalStrength::Buy => 1.0,
            SignalStrength::Neutral => 0.0,
            SignalStrength::Sell => -1.0,
            SignalStrength::StrongSell => -1.0,
        };

        // Position size based on risk budget and ATR
        let atr_dollars = atr * last_price; // approximate dollar risk per unit
        let size_fraction = if atr_dollars.abs() > 1e-15 {
            let units = (portfolio_value * risk_per_trade) / atr_dollars;
            let position_value = units * last_price;
            (position_value / portfolio_value).clamp(0.0, 0.25) // max 25% of portfolio
        } else {
            0.0
        };

        // Scale by signal strength
        let signal_scale = (score - 50.0).abs() / 50.0; // 0 to 1
        let size_fraction = size_fraction * signal_scale;

        let stop_loss = if direction > 0.0 {
            last_price - 2.0 * atr
        } else if direction < 0.0 {
            last_price + 2.0 * atr
        } else {
            last_price * 0.95
        };

        let take_profit = if direction > 0.0 {
            last_price + 3.0 * atr
        } else if direction < 0.0 {
            last_price - 3.0 * atr
        } else {
            last_price * 1.05
        };

        let risk_reward = if (take_profit - stop_loss).abs() > 1e-15 {
            ((take_profit - stop_loss) * direction / (stop_loss - last_price).abs()).abs()
        } else {
            0.0
        };

        let reasoning = match signal {
            SignalStrength::StrongBuy => format!(
                "Strong buy signal (score={:.0}). Risking {:.1}% of portfolio.",
                score, size_fraction * 100.0
            ),
            SignalStrength::Buy => format!(
                "Buy signal (score={:.0}). Risking {:.1}% of portfolio.",
                score, size_fraction * 100.0
            ),
            SignalStrength::Neutral => "No actionable signal. No position recommended.".into(),
            SignalStrength::Sell => format!(
                "Sell signal (score={:.0}). Consider reducing position.",
                score
            ),
            SignalStrength::StrongSell => format!(
                "Strong sell signal (score={:.0}). Consider exiting or shorting.",
                score
            ),
        };

        PositionRecommendation {
            size_fraction,
            stop_loss: stop_loss.max(0.0),
            take_profit: take_profit.max(0.0),
            risk_reward_ratio: risk_reward,
            reasoning,
        }
    }

    // -----------------------------------------------------------------------
    // Helpers
    // -----------------------------------------------------------------------

    /// Compute EMA (exponential moving average) over a series of values.
    /// Returns a vector of EMA values, same length as input.
    fn ema(&self, data: &[f64], period: usize) -> Vec<f64> {
        if data.is_empty() || period == 0 {
            return vec![];
        }

        let alpha = 2.0 / (period as f64 + 1.0);
        let mut ema_values = Vec::with_capacity(data.len());

        // Initialize with SMA of first `period` values
        if data.len() < period {
            // Not enough data; return simple values
            return data.to_vec();
        }

        let initial_sma: f64 = data[..period].iter().sum::<f64>() / period as f64;
        for _ in 0..period {
            ema_values.push(initial_sma);
        }

        // EMA for remaining values
        for i in period..data.len() {
            let prev = *ema_values.last().unwrap();
            ema_values.push(alpha * data[i] + (1.0 - alpha) * prev);
        }

        ema_values
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use chrono::{Duration, Utc};

    fn make_trending_up_prices(n: usize) -> Vec<PricePoint> {
        let now = Utc::now();
        (0..n)
            .map(|i| {
                let base = 100.0 + i as f64 * 0.5;
                let noise = (i as f64 * 1.3).sin() * 1.0;
                let close = base + noise;
                PricePoint::new(
                    now - Duration::days((n - 1 - i) as i64),
                    close - 0.5,
                    close + 1.0,
                    close - 1.0,
                    close,
                    1000.0 + i as f64 * 50.0,
                )
            })
            .collect()
    }

    fn make_trending_down_prices(n: usize) -> Vec<PricePoint> {
        let now = Utc::now();
        (0..n)
            .map(|i| {
                let base = 200.0 - i as f64 * 0.5;
                let noise = (i as f64 * 1.7).sin() * 1.0;
                let close = base + noise;
                PricePoint::new(
                    now - Duration::days((n - 1 - i) as i64),
                    close + 0.5,
                    close + 1.0,
                    close - 1.0,
                    close,
                    1000.0 + i as f64 * 50.0,
                )
            })
            .collect()
    }

    fn make_volatile_prices(n: usize) -> Vec<PricePoint> {
        let now = Utc::now();
        (0..n)
            .map(|i| {
                let close = 100.0 + 10.0 * (i as f64 * 0.3).sin();
                PricePoint::new(
                    now - Duration::days((n - 1 - i) as i64),
                    close - 1.0,
                    close + 3.0,
                    close - 3.0,
                    close,
                    1000.0,
                )
            })
            .collect()
    }

    #[test]
    fn test_rsi_overbought_trending_up() {
        let gen = SignalGenerator::new();
        let prices = make_trending_up_prices(50);
        let rsi = gen.compute_rsi(&prices).unwrap();
        // Sustained uptrend should give RSI above 50
        assert!(rsi > 50.0);
    }

    #[test]
    fn test_rsi_insufficient_data() {
        let gen = SignalGenerator::new();
        let prices = make_trending_up_prices(5);
        assert!(gen.compute_rsi(&prices).is_none());
    }

    #[test]
    fn test_rsi_range() {
        let gen = SignalGenerator::new();
        let prices = make_trending_up_prices(50);
        let rsi = gen.compute_rsi(&prices).unwrap();
        assert!(rsi >= 0.0 && rsi <= 100.0);
    }

    #[test]
    fn test_macd_trending_up() {
        let gen = SignalGenerator::new();
        let prices = make_trending_up_prices(50);
        let (macd, signal, hist) = gen.compute_macd(&prices);
        assert!(macd.is_some());
        assert!(signal.is_some());
        assert!(hist.is_some());
        // Uptrend: MACD line should be above signal
        assert!(hist.unwrap() >= 0.0);
    }

    #[test]
    fn test_macd_insufficient_data() {
        let gen = SignalGenerator::new();
        let prices = make_trending_up_prices(10);
        let (macd, signal, hist) = gen.compute_macd(&prices);
        assert!(macd.is_none());
        assert!(signal.is_none());
        assert!(hist.is_none());
    }

    #[test]
    fn test_bollinger_bands() {
        let gen = SignalGenerator::new();
        let prices = make_trending_up_prices(50);
        let (upper, middle, lower) = gen.compute_bollinger_bands(&prices);
        assert!(upper.is_some());
        assert!(middle.is_some());
        assert!(lower.is_some());
        assert!(upper.unwrap() > middle.unwrap());
        assert!(middle.unwrap() > lower.unwrap());
    }

    #[test]
    fn test_bollinger_bands_insufficient_data() {
        let gen = SignalGenerator::new();
        let prices = make_trending_up_prices(5);
        let (upper, middle, lower) = gen.compute_bollinger_bands(&prices);
        assert!(upper.is_none());
        assert!(middle.is_none());
        assert!(lower.is_none());
    }

    #[test]
    fn test_atr_positive() {
        let gen = SignalGenerator::new();
        let prices = make_volatile_prices(50);
        let atr = gen.compute_atr(&prices).unwrap();
        assert!(atr > 0.0);
    }

    #[test]
    fn test_atr_insufficient_data() {
        let gen = SignalGenerator::new();
        let prices = make_volatile_prices(5);
        assert!(gen.compute_atr(&prices).is_none());
    }

    #[test]
    fn test_stochastic_range() {
        let gen = SignalGenerator::new();
        let prices = make_volatile_prices(50);
        let (k, d) = gen.compute_stochastic(&prices);
        assert!(k.is_some());
        assert!(d.is_some());
        let k_val = k.unwrap();
        assert!(k_val >= 0.0 && k_val <= 100.0);
    }

    #[test]
    fn test_stochastic_insufficient_data() {
        let gen = SignalGenerator::new();
        let prices = make_volatile_prices(5);
        let (k, d) = gen.compute_stochastic(&prices);
        assert!(k.is_none());
        assert!(d.is_none());
    }

    #[test]
    fn test_compute_all_indicators() {
        let gen = SignalGenerator::new();
        let prices = make_trending_up_prices(50);
        let indicators = gen.compute_all(&prices);
        assert!(indicators.rsi.is_some());
        assert!(indicators.macd_line.is_some());
        assert!(indicators.bollinger_upper.is_some());
        assert!(indicators.atr.is_some());
        assert!(indicators.stochastic_k.is_some());
    }

    #[test]
    fn test_composite_score_range() {
        let gen = SignalGenerator::new();
        let prices = make_trending_up_prices(50);
        let score = gen.composite_score(&prices);
        assert!(score >= 0.0 && score <= 100.0);
    }

    #[test]
    fn test_score_to_signal() {
        assert_eq!(SignalGenerator::score_to_signal(80.0), SignalStrength::StrongBuy);
        assert_eq!(SignalGenerator::score_to_signal(60.0), SignalStrength::Buy);
        assert_eq!(SignalGenerator::score_to_signal(48.0), SignalStrength::Neutral);
        assert_eq!(SignalGenerator::score_to_signal(35.0), SignalStrength::Sell);
        assert_eq!(SignalGenerator::score_to_signal(10.0), SignalStrength::StrongSell);
    }

    #[test]
    fn test_generate_signal_structure() {
        let gen = SignalGenerator::new();
        let prices = make_trending_up_prices(50);
        let signal = gen.generate_signal(CommodityType::Gold, &prices);
        assert_eq!(signal.commodity, CommodityType::Gold);
        assert!(!signal.indicators.is_empty());
        assert!(!signal.reasoning.is_empty());
    }

    #[test]
    fn test_position_sizing_neutral() {
        let gen = SignalGenerator::new();
        // Create flat prices → neutral signal → small position
        let now = Utc::now();
        let prices: Vec<PricePoint> = (0..50)
            .map(|i| {
                PricePoint::new(now - Duration::days((49 - i) as i64), 100.0, 101.0, 99.0, 100.0, 1000.0)
            })
            .collect();
        let rec = gen.position_sizing(&prices, 100_000.0, 0.02);
        assert!(rec.size_fraction >= 0.0);
    }

    #[test]
    fn test_position_sizing_non_negative() {
        let gen = SignalGenerator::new();
        let prices = make_trending_up_prices(50);
        let rec = gen.position_sizing(&prices, 100_000.0, 0.02);
        assert!(rec.stop_loss >= 0.0);
        assert!(rec.take_profit >= 0.0);
    }

    #[test]
    fn test_position_sizing_reasoning() {
        let gen = SignalGenerator::new();
        let prices = make_trending_up_prices(50);
        let rec = gen.position_sizing(&prices, 100_000.0, 0.02);
        assert!(!rec.reasoning.is_empty());
    }

    #[test]
    fn test_technical_indicators_default() {
        let ti = TechnicalIndicators::default();
        assert!(ti.rsi.is_none());
        assert!(ti.macd_line.is_none());
    }

    #[test]
    fn test_signal_generator_default() {
        let gen = SignalGenerator::default();
        assert_eq!(gen.rsi_period, 14);
        assert_eq!(gen.macd_fast, 12);
        assert_eq!(gen.macd_slow, 26);
        assert_eq!(gen.macd_signal_period, 9);
        assert_eq!(gen.bollinger_period, 20);
        assert!((gen.bollinger_std_devs - 2.0).abs() < 1e-10);
    }

    #[test]
    fn test_bollinger_width() {
        let gen = SignalGenerator::new();
        let volatile = make_volatile_prices(50);
        let flat: Vec<PricePoint> = {
            let now = Utc::now();
            (0..50).map(|i| {
                PricePoint::new(now - Duration::days((49 - i) as i64), 100.0, 100.5, 99.5, 100.0, 1000.0)
            }).collect()
        };

        let (v_upper, _, v_lower) = gen.compute_bollinger_bands(&volatile);
        let (f_upper, _, f_lower) = gen.compute_bollinger_bands(&flat);

        let v_width = v_upper.unwrap() - v_lower.unwrap();
        let f_width = f_upper.unwrap() - f_lower.unwrap();
        // Volatile data should have wider bands
        assert!(v_width > f_width);
    }
}
