//! # Seasonal Analysis
//!
//! Detects seasonal patterns in commodity prices including monthly patterns,
//! day-of-week effects, inventory cycle phase estimation, and seasonal strength
//! scoring.

use chrono::{Datelike, Utc};
use crate::types::{CommodityType, PricePoint, SeasonalPattern};

// ---------------------------------------------------------------------------
// Seasonal Analyzer
// ---------------------------------------------------------------------------

/// Analyzer for detecting and scoring seasonal patterns in commodity prices.
pub struct SeasonalAnalyzer {
    /// Minimum number of data points required for analysis.
    pub min_data_points: usize,
}

impl Default for SeasonalAnalyzer {
    fn default() -> Self {
        SeasonalAnalyzer {
            min_data_points: 30,
        }
    }
}

impl SeasonalAnalyzer {
    /// Create a new seasonal analyzer with default settings.
    pub fn new() -> Self {
        Self::default()
    }

    // -----------------------------------------------------------------------
    // Monthly Seasonal Patterns
    // -----------------------------------------------------------------------

    /// Detect monthly seasonal patterns by grouping returns by calendar month.
    /// Returns 12 `SeasonalPattern` entries (one per month), sorted by month.
    pub fn monthly_patterns(&self, commodity: CommodityType, prices: &[PricePoint]) -> Vec<SeasonalPattern> {
        if prices.len() < self.min_data_points {
            return vec![];
        }

        // Group returns by month
        let mut monthly_returns: [[Vec<f64>; 12]; 12] = Default::default(); // [year_offset][month]

        for i in 1..prices.len() {
            let prev_close = prices[i - 1].close;
            if prev_close.abs() < 1e-15 {
                continue;
            }
            let ret = (prices[i].close - prev_close) / prev_close;
            let month = prices[i].timestamp.month() as usize; // 1-12
            monthly_returns[0][month - 1].push(ret);
        }

        let mut patterns = Vec::new();
        for month in 0..12 {
            let returns = &monthly_returns[0][month];
            if returns.is_empty() {
                patterns.push(SeasonalPattern {
                    commodity,
                    month: (month + 1) as u32,
                    avg_return: 0.0,
                    volatility: 0.0,
                    historical_occurrences: 0,
                });
                continue;
            }

            let avg_return: f64 = returns.iter().sum::<f64>() / returns.len() as f64;
            let variance = returns.iter().map(|r| (r - avg_return).powi(2)).sum::<f64>()
                / returns.len().max(1) as f64;
            let volatility = variance.sqrt();

            patterns.push(SeasonalPattern {
                commodity,
                month: (month + 1) as u32,
                avg_return,
                volatility,
                historical_occurrences: returns.len() as u32,
            });
        }

        patterns
    }

    // -----------------------------------------------------------------------
    // Day-of-Week Effects
    // -----------------------------------------------------------------------

    /// Compute average returns by day of week.
    /// Returns a vector of 7 (f64, f64) tuples: (avg_return, count) for Mon=0 .. Sun=6.
    pub fn day_of_week_effects(&self, prices: &[PricePoint]) -> Vec<(f64, usize)> {
        let mut dow_returns: [[Vec<f64>; 7]; 1] = Default::default();

        for i in 1..prices.len() {
            let prev_close = prices[i - 1].close;
            if prev_close.abs() < 1e-15 {
                continue;
            }
            let ret = (prices[i].close - prev_close) / prev_close;
            let dow = prices[i].timestamp.weekday().num_days_from_monday() as usize;
            dow_returns[0][dow].push(ret);
        }

        (0..7)
            .map(|dow| {
                let returns = &dow_returns[0][dow];
                if returns.is_empty() {
                    (0.0, 0)
                } else {
                    let avg = returns.iter().sum::<f64>() / returns.len() as f64;
                    (avg, returns.len())
                }
            })
            .collect()
    }

    /// Find the best and worst performing day of the week.
    /// Returns (best_dow, best_avg_return, worst_dow, worst_avg_return).
    pub fn best_worst_day_of_week(&self, prices: &[PricePoint]) -> (usize, f64, usize, f64) {
        let effects = self.day_of_week_effects(prices);
        let mut best_dow = 0;
        let mut best_ret = f64::NEG_INFINITY;
        let mut worst_dow = 0;
        let mut worst_ret = f64::INFINITY;

        for (dow, (avg, count)) in effects.iter().enumerate() {
            if *count > 0 {
                if *avg > best_ret {
                    best_ret = *avg;
                    best_dow = dow;
                }
                if *avg < worst_ret {
                    worst_ret = *avg;
                    worst_dow = dow;
                }
            }
        }

        (best_dow, best_ret, worst_dow, worst_ret)
    }

    // -----------------------------------------------------------------------
    // Inventory Cycle Phase Estimation
    // -----------------------------------------------------------------------

    /// Inventory cycle phase based on recent price trend heuristic.
    /// Phases: "accumulation", "markup", "distribution", "markdown".
    pub fn inventory_cycle_phase(&self, prices: &[PricePoint]) -> String {
        if prices.len() < 20 {
            return "unknown".into();
        }

        // Use 20-period SMA and recent trend
        let closes: Vec<f64> = prices.iter().map(|p| p.close).collect();
        let n = closes.len();

        // Short-term trend (last 5 periods)
        let short_trend = if n >= 6 {
            (closes[n - 1] - closes[n - 6]) / closes[n - 6]
        } else {
            0.0
        };

        // Medium-term trend (last 20 periods)
        let medium_trend = if n >= 20 {
            (closes[n - 1] - closes[n - 20]) / closes[n - 20]
        } else {
            0.0
        };

        // Recent momentum (ratio of short to medium)
        let momentum_ratio = if medium_trend.abs() < 1e-15 {
            0.0
        } else {
            short_trend / medium_trend
        };

        // For a linear trend, short/medium ≈ 5/20 * P_start/P_mid ≈ 0.2-0.25.
        // Values above 0.15 indicate acceleration. Below indicates deceleration.
        if medium_trend > 0.02 && momentum_ratio > 0.15 {
            "markup".into() // Price rising and accelerating
        } else if medium_trend > 0.02 && momentum_ratio <= 0.15 {
            "distribution".into() // Price rising but decelerating
        } else if medium_trend < -0.02 && momentum_ratio > 0.15 {
            "markdown".into() // Price falling and accelerating (both negative, ratio positive & high)
        } else {
            "accumulation".into() // Price flat or slowly recovering
        }
    }

    // -----------------------------------------------------------------------
    // Seasonal Strength Scoring
    // -----------------------------------------------------------------------

    /// Compute a seasonal strength score from -1.0 to 1.0.
    /// Positive means the current month is seasonally favorable.
    pub fn seasonal_strength_score(&self, commodity: CommodityType, prices: &[PricePoint]) -> f64 {
        let patterns = self.monthly_patterns(commodity, prices);
        if patterns.is_empty() {
            return 0.0;
        }

        let current_month = Utc::now().month() as usize; // 1-12
        let current_pattern = &patterns[current_month - 1];

        if current_pattern.historical_occurrences == 0 {
            return 0.0;
        }

        // Score based on how much the average return for the current month
        // deviates from the average of all months
        let all_avg: f64 = patterns.iter()
            .filter(|p| p.historical_occurrences > 0)
            .map(|p| p.avg_return)
            .sum::<f64>()
            / patterns.iter().filter(|p| p.historical_occurrences > 0).count().max(1) as f64;

        if all_avg.abs() < 1e-15 {
            return 0.0;
        }

        // Normalize to [-1, 1]
        let score = (current_pattern.avg_return - all_avg) / all_avg.abs();
        score.clamp(-1.0, 1.0)
    }

    /// Find the historically best and worst months.
    /// Returns (best_month, best_return, worst_month, worst_return).
    pub fn best_worst_months(&self, commodity: CommodityType, prices: &[PricePoint]) -> (u32, f64, u32, f64) {
        let patterns = self.monthly_patterns(commodity, prices);
        if patterns.is_empty() {
            return (1, 0.0, 1, 0.0);
        }

        let mut best_month = 1u32;
        let mut best_ret = f64::NEG_INFINITY;
        let mut worst_month = 1u32;
        let mut worst_ret = f64::INFINITY;

        for p in &patterns {
            if p.historical_occurrences > 0 {
                if p.avg_return > best_ret {
                    best_ret = p.avg_return;
                    best_month = p.month;
                }
                if p.avg_return < worst_ret {
                    worst_ret = p.avg_return;
                    worst_month = p.month;
                }
            }
        }

        (best_month, best_ret, worst_month, worst_ret)
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use chrono::{Duration};

    fn make_prices_with_monthly_seasonality(total_days: usize) -> Vec<PricePoint> {
        let now = Utc::now();
        let mut prices = Vec::with_capacity(total_days);
        let mut price = 100.0;

        for i in 0..total_days {
            let ts = now - Duration::days((total_days - 1 - i) as i64);
            let month = ts.month();

            // Seasonal bias: Jan (1) up, Jul (7) down
            let seasonal_bias = match month {
                1 => 0.005,   // January: bullish
                7 => -0.005,  // July: bearish
                _ => 0.0,
            };

            price *= 1.0 + seasonal_bias + (i as f64 * 0.1).sin() * 0.002;
            price = price.max(50.0).min(200.0);

            prices.push(PricePoint::new(ts, price - 0.5, price + 1.0, price - 1.0, price, 1000.0));
        }

        prices
    }

    #[test]
    fn test_monthly_patterns_length() {
        let analyzer = SeasonalAnalyzer::new();
        let prices = make_prices_with_monthly_seasonality(400);
        let patterns = analyzer.monthly_patterns(CommodityType::Gold, &prices);
        assert_eq!(patterns.len(), 12);
    }

    #[test]
    fn test_monthly_patterns_commodity() {
        let analyzer = SeasonalAnalyzer::new();
        let prices = make_prices_with_monthly_seasonality(400);
        let patterns = analyzer.monthly_patterns(CommodityType::Silver, &prices);
        for p in &patterns {
            assert_eq!(p.commodity, CommodityType::Silver);
            assert!(p.month >= 1 && p.month <= 12);
        }
    }

    #[test]
    fn test_monthly_patterns_insufficient_data() {
        let analyzer = SeasonalAnalyzer::new();
        let prices = make_prices_with_monthly_seasonality(10);
        let patterns = analyzer.monthly_patterns(CommodityType::Gold, &prices);
        assert!(patterns.is_empty());
    }

    #[test]
    fn test_monthly_patterns_have_occurrences() {
        let analyzer = SeasonalAnalyzer::new();
        let prices = make_prices_with_monthly_seasonality(400);
        let patterns = analyzer.monthly_patterns(CommodityType::Gold, &prices);
        let with_data: Vec<_> = patterns.iter().filter(|p| p.historical_occurrences > 0).collect();
        assert!(!with_data.is_empty());
    }

    #[test]
    fn test_day_of_week_effects_length() {
        let analyzer = SeasonalAnalyzer::new();
        let prices = make_prices_with_monthly_seasonality(100);
        let effects = analyzer.day_of_week_effects(&prices);
        assert_eq!(effects.len(), 7);
    }

    #[test]
    fn test_day_of_week_effects_sum() {
        let analyzer = SeasonalAnalyzer::new();
        let prices = make_prices_with_monthly_seasonality(100);
        let effects = analyzer.day_of_week_effects(&prices);
        let total: usize = effects.iter().map(|(_, count)| *count).sum();
        assert!(total > 0);
    }

    #[test]
    fn test_best_worst_day_of_week() {
        let analyzer = SeasonalAnalyzer::new();
        let prices = make_prices_with_monthly_seasonality(100);
        let (best_dow, best_ret, worst_dow, worst_ret) = analyzer.best_worst_day_of_week(&prices);
        assert!(best_dow < 7);
        assert!(worst_dow < 7);
        assert!(best_ret >= worst_ret);
    }

    #[test]
    fn test_inventory_cycle_markup() {
        let analyzer = SeasonalAnalyzer::new();
        // Accelerating uptrend (quadratic)
        let now = Utc::now();
        let prices: Vec<PricePoint> = (0..30)
            .map(|i| {
                let price = 100.0 + 0.1 * (i as f64).powi(2);
                PricePoint::new(now - Duration::days((29 - i) as i64), price - 1.0, price + 1.0, price - 1.0, price, 1000.0)
            })
            .collect();
        let phase = analyzer.inventory_cycle_phase(&prices);
        assert_eq!(phase, "markup");
    }

    #[test]
    fn test_inventory_cycle_markdown() {
        let analyzer = SeasonalAnalyzer::new();
        // Accelerating downtrend (quadratic)
        let now = Utc::now();
        let prices: Vec<PricePoint> = (0..30)
            .map(|i| {
                let price = 200.0 - 0.15 * (i as f64).powi(2);
                PricePoint::new(now - Duration::days((29 - i) as i64), price + 1.0, price + 1.0, price - 1.0, price, 1000.0)
            })
            .collect();
        let phase = analyzer.inventory_cycle_phase(&prices);
        assert_eq!(phase, "markdown");
    }

    #[test]
    fn test_inventory_cycle_insufficient_data() {
        let analyzer = SeasonalAnalyzer::new();
        let now = Utc::now();
        let prices = vec![PricePoint::new(now, 100.0, 101.0, 99.0, 100.0, 1000.0)];
        let phase = analyzer.inventory_cycle_phase(&prices);
        assert_eq!(phase, "unknown");
    }

    #[test]
    fn test_seasonal_strength_score_range() {
        let analyzer = SeasonalAnalyzer::new();
        let prices = make_prices_with_monthly_seasonality(400);
        let score = analyzer.seasonal_strength_score(CommodityType::Gold, &prices);
        assert!(score >= -1.0 && score <= 1.0);
    }

    #[test]
    fn test_seasonal_strength_no_data() {
        let analyzer = SeasonalAnalyzer::new();
        let prices = make_prices_with_monthly_seasonality(5);
        let score = analyzer.seasonal_strength_score(CommodityType::Gold, &prices);
        assert!((score - 0.0).abs() < 1e-10);
    }

    #[test]
    fn test_best_worst_months() {
        let analyzer = SeasonalAnalyzer::new();
        let prices = make_prices_with_monthly_seasonality(400);
        let (best_m, best_r, worst_m, worst_r) = analyzer.best_worst_months(CommodityType::Gold, &prices);
        assert!(best_m >= 1 && best_m <= 12);
        assert!(worst_m >= 1 && worst_m <= 12);
        assert!(best_r >= worst_r);
    }

    #[test]
    fn test_best_worst_months_empty() {
        let analyzer = SeasonalAnalyzer::new();
        let prices = make_prices_with_monthly_seasonality(5);
        let (best_m, best_r, worst_m, worst_r) = analyzer.best_worst_months(CommodityType::Gold, &prices);
        assert_eq!(best_m, 1);
        assert_eq!(worst_m, 1);
        assert!((best_r - 0.0).abs() < 1e-10);
        assert!((worst_r - 0.0).abs() < 1e-10);
    }

    #[test]
    fn test_seasonal_analyzer_default() {
        let analyzer = SeasonalAnalyzer::default();
        assert_eq!(analyzer.min_data_points, 30);
    }
}
