//! Price forecasting using linear regression on log prices.

use serde::{Deserialize, Serialize};

/// A single price data point in a time series.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PricePoint {
    pub period: usize,
    pub price: f64,
}

/// Result of a price forecast.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ForecastResult {
    pub forecast_price: f64,
    pub price_deviation_pct: f64,
    pub confidence_bps: u32,
}

/// Compute price forecast using simple linear regression on log prices.
///
/// Returns `(forecast_price, price_deviation_pct, confidence_bps)`.
pub fn compute_price_forecast(history: &[PricePoint]) -> ForecastResult {
    if history.len() < 3 {
        let current = history.last().map(|h| h.price).unwrap_or(0.0);
        return ForecastResult {
            forecast_price: current * 1.05,
            price_deviation_pct: 5.0,
            confidence_bps: 500,
        };
    }

    let prices: Vec<f64> = history.iter().map(|h| h.price).collect();
    let log_prices: Vec<f64> = prices.iter().map(|p| p.ln()).collect();
    let n = log_prices.len() as f64;

    // Least-squares linear regression
    let x_mean = (n - 1.0) / 2.0;
    let y_mean = log_prices.iter().sum::<f64>() / n;

    let mut numerator = 0.0_f64;
    let mut denominator = 0.0_f64;
    for (i, lp) in log_prices.iter().enumerate() {
        let xi = i as f64;
        numerator += (xi - x_mean) * (lp - y_mean);
        denominator += (xi - x_mean).powi(2);
    }

    let slope = if denominator.abs() < 1e-12 {
        0.0
    } else {
        numerator / denominator
    };

    // Extrapolate 12 periods ahead
    let forecast_log = log_prices.last().unwrap() + slope * 12.0;
    let forecast_price = forecast_log.exp();

    let current_price = *prices.last().unwrap();
    let deviation_pct = ((forecast_price - current_price) / current_price) * 100.0;

    // R-squared for confidence
    let mut ss_res = 0.0_f64;
    let mut ss_tot = 0.0_f64;
    for (i, lp) in log_prices.iter().enumerate() {
        let xi = i as f64;
        let predicted = y_mean + slope * (xi - x_mean);
        ss_res += (lp - predicted).powi(2);
        ss_tot += (lp - y_mean).powi(2);
    }
    let r_squared = if ss_tot.abs() < 1e-12 {
        0.0
    } else {
        1.0 - ss_res / ss_tot
    };
    let confidence_bps = (r_squared * 10_000.0).clamp(100.0, 9_500.0) as u32;

    ForecastResult {
        forecast_price,
        price_deviation_pct: deviation_pct,
        confidence_bps,
    }
}

/// Generate mock historical price data for model training.
pub fn generate_mock_price_history(mineral: &str, periods: usize) -> Vec<PricePoint> {
    use crate::config::MINERALS;

    let config = MINERALS.iter()
        .find(|(s, _)| *s == mineral)
        .map(|(_, c)| c)
        .expect("Unknown mineral");

    let mid = (config.typical_price_range.0 + config.typical_price_range.1) / 2.0;

    let mut history = Vec::with_capacity(periods);
    let mut price = mid;

    // Simple seeded PRNG (LCG)
    let mut seed = mineral.bytes().fold(0u64, |acc, b| acc.wrapping_mul(31).wrapping_add(b as u64));

    for i in 0..periods {
        seed = seed.wrapping_mul(1_103_515_245).wrapping_add(12_345);
        let random_val = (seed >> 16) as f64 / 65_536.0; // 0..1
        let change = -0.08 + random_val * 0.18; // -0.08..0.10
        price *= 1.0 + change;
        price = price.clamp(config.typical_price_range.0 * 0.7, config.typical_price_range.1 * 1.3);
        history.push(PricePoint { period: i, price });
    }

    history
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_forecast_basic() {
        let history = vec![
            PricePoint { period: 0, price: 100.0 },
            PricePoint { period: 1, price: 102.0 },
            PricePoint { period: 2, price: 104.0 },
            PricePoint { period: 3, price: 106.0 },
            PricePoint { period: 4, price: 108.0 },
        ];
        let result = compute_price_forecast(&history);
        assert!(result.forecast_price > 108.0, "Expected upward forecast");
        assert!(result.confidence_bps > 0);
    }

    #[test]
    fn test_forecast_short_history() {
        let history = vec![
            PricePoint { period: 0, price: 100.0 },
        ];
        let result = compute_price_forecast(&history);
        assert_eq!(result.forecast_price, 105.0);
        assert_eq!(result.confidence_bps, 500);
    }

    #[test]
    fn test_mock_price_history() {
        let history = generate_mock_price_history("LITHIUM", 24);
        assert_eq!(history.len(), 24);
        for point in &history {
            assert!(point.price > 0.0);
        }
    }
}
