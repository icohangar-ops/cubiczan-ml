use crate::types::*;
use chrono::{Datelike, Duration};
use rand::distributions::Distribution;
use serde::{Deserialize, Serialize};

/// Basic price statistics.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PriceStats {
    pub min: f64,
    pub max: f64,
    pub mean: f64,
    pub median: f64,
    pub std_dev: f64,
}

impl PriceStats {
    /// Calculate statistics from a slice of prices.
    pub fn from_prices(prices: &[f64]) -> Self {
        if prices.is_empty() {
            return PriceStats {
                min: 0.0,
                max: 0.0,
                mean: 0.0,
                median: 0.0,
                std_dev: 0.0,
            };
        }

        let min = prices.iter().cloned().fold(f64::INFINITY, f64::min);
        let max = prices.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
        let mean = prices.iter().sum::<f64>() / prices.len() as f64;

        let mut sorted = prices.to_vec();
        sorted.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));

        let median = if sorted.len() % 2 == 0 {
            (sorted[sorted.len() / 2 - 1] + sorted[sorted.len() / 2]) / 2.0
        } else {
            sorted[sorted.len() / 2]
        };

        let variance = if prices.len() > 1 {
            prices.iter().map(|p| (p - mean).powi(2)).sum::<f64>() / (prices.len() - 1) as f64
        } else {
            0.0
        };
        let std_dev = variance.sqrt();

        PriceStats {
            min,
            max,
            mean,
            median,
            std_dev,
        }
    }
}

/// Calculate the change between the last two prices.
pub fn calculate_price_change(prices: &[f64]) -> Option<PriceChange> {
    if prices.len() < 2 {
        return None;
    }
    let prev = prices[prices.len() - 2];
    let curr = prices[prices.len() - 1];
    let absolute = curr - prev;
    let percentage = if prev.abs() < f64::EPSILON {
        0.0
    } else {
        (absolute / prev.abs()) * 100.0
    };
    let direction = if absolute > f64::EPSILON {
        PriceDirection::Up
    } else if absolute < -f64::EPSILON {
        PriceDirection::Down
    } else {
        PriceDirection::Flat
    };
    Some(PriceChange {
        absolute,
        percentage,
        direction,
    })
}

/// Calculate simple moving average with the given window size.
pub fn calculate_sma(prices: &[f64], window: usize) -> Vec<f64> {
    if window == 0 || prices.len() < window {
        return vec![];
    }
    let mut sma = Vec::with_capacity(prices.len() - window + 1);
    let mut window_sum: f64 = prices[..window].iter().sum();
    sma.push(window_sum / window as f64);

    for i in window..prices.len() {
        window_sum += prices[i] - prices[i - window];
        sma.push(window_sum / window as f64);
    }
    sma
}

/// Calculate volatility as the standard deviation of log returns.
pub fn calculate_volatility(prices: &[f64]) -> f64 {
    if prices.len() < 2 {
        return 0.0;
    }
    let log_returns: Vec<f64> = prices
        .windows(2)
        .map(|w| (w[1] / w[0]).ln())
        .collect();

    let mean = log_returns.iter().sum::<f64>() / log_returns.len() as f64;
    let variance = log_returns
        .iter()
        .map(|r| (r - mean).powi(2))
        .sum::<f64>()
        / (log_returns.len() - 1) as f64;
    variance.sqrt()
}

/// Generate historical prices using Geometric Brownian Motion.
///
/// `initial` — starting price
/// `days` — number of days to simulate
/// `annual_vol` — annual volatility (σ)
/// `seed` — RNG seed for reproducibility
///
/// Model: S(t+1) = S(t) * exp((μ - σ²/2)*dt + σ*√dt*Z)
/// where μ = 0.05 (drift), dt = 1/252, Z ~ N(0,1)
pub fn generate_historical_prices(
    initial: f64,
    days: usize,
    annual_vol: f64,
    seed: u64,
) -> Vec<PricePoint> {
    if days == 0 {
        return vec![];
    }

    use rand::SeedableRng;
    let mut rng = rand::rngs::StdRng::seed_from_u64(seed);

    let dt: f64 = 1.0 / 252.0; // trading days per year
    let drift = 0.05; // annual drift (μ)
    let vol_sqrt_dt = annual_vol * dt.sqrt();

    let start_date = chrono::NaiveDate::from_ymd_opt(2024, 1, 2).unwrap();
    let mut prices = Vec::with_capacity(days);
    let mut price = initial;

    for i in 0..days {
        let date = start_date + Duration::days(i as i64);

        // Skip weekends (Saturday = 6, Sunday = 7)
        if date.weekday().num_days_from_sunday() >= 6 {
            continue;
        }

        let z: f64 = rand::distributions::Standard.sample(&mut rng);
        let log_return = (drift - annual_vol.powi(2) / 2.0) * dt + vol_sqrt_dt * z;
        price *= log_return.exp();

        prices.push(PricePoint {
            date,
            price,
            volume: None,
        });
    }

    prices
}

/// Calculate comprehensive price statistics.
pub fn calculate_price_stats(prices: &[f64]) -> PriceStats {
    PriceStats::from_prices(prices)
}

/// Calculate the max drawdown from a price series.
pub fn calculate_max_drawdown(prices: &[f64]) -> f64 {
    if prices.is_empty() {
        return 0.0;
    }
    let mut peak = prices[0];
    let mut max_dd = 0.0;
    for &price in prices {
        if price > peak {
            peak = price;
        }
        let dd = (peak - price) / peak;
        if dd > max_dd {
            max_dd = dd;
        }
    }
    max_dd
}

/// Calculate annualized return from a price series.
pub fn calculate_annualized_return(prices: &[f64]) -> f64 {
    if prices.len() < 2 {
        return 0.0;
    }
    let total_return = prices[prices.len() - 1] / prices[0];
    let years = (prices.len() - 1) as f64 / 252.0;
    if years <= 0.0 {
        return 0.0;
    }
    total_return.powf(1.0 / years) - 1.0
}

/// Calculate the Sharpe ratio assuming a risk-free rate.
pub fn calculate_sharpe_ratio(prices: &[f64], risk_free_rate: f64) -> f64 {
    if prices.len() < 2 {
        return 0.0;
    }
    let vol = calculate_volatility(prices);
    if vol < f64::EPSILON {
        return 0.0;
    }
    let log_returns: Vec<f64> = prices
        .windows(2)
        .map(|w| (w[1] / w[0]).ln())
        .collect();
    let mean_return = log_returns.iter().sum::<f64>() / log_returns.len() as f64;
    let annualized_mean = mean_return * 252.0;
    let annualized_vol = vol * (252.0_f64).sqrt();
    (annualized_mean - risk_free_rate) / annualized_vol
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_price_change_up() {
        let prices = vec![100.0, 110.0];
        let change = calculate_price_change(&prices).unwrap();
        assert_eq!(change.absolute, 10.0);
        assert!((change.percentage - 10.0).abs() < 0.001);
        assert_eq!(change.direction, PriceDirection::Up);
    }

    #[test]
    fn test_price_change_down() {
        let prices = vec![100.0, 90.0];
        let change = calculate_price_change(&prices).unwrap();
        assert_eq!(change.absolute, -10.0);
        assert_eq!(change.direction, PriceDirection::Down);
    }

    #[test]
    fn test_price_change_flat() {
        let prices = vec![100.0, 100.0];
        let change = calculate_price_change(&prices).unwrap();
        assert_eq!(change.absolute, 0.0);
        assert_eq!(change.direction, PriceDirection::Flat);
    }

    #[test]
    fn test_price_change_insufficient() {
        assert!(calculate_price_change(&[100.0]).is_none());
        assert!(calculate_price_change(&[]).is_none());
    }

    #[test]
    fn test_sma_basic() {
        let prices = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let sma = calculate_sma(&prices, 3);
        assert_eq!(sma.len(), 3);
        assert!((sma[0] - 2.0).abs() < 0.001);
        assert!((sma[1] - 3.0).abs() < 0.001);
        assert!((sma[2] - 4.0).abs() < 0.001);
    }

    #[test]
    fn test_sma_window_too_large() {
        let prices = vec![1.0, 2.0];
        let sma = calculate_sma(&prices, 5);
        assert!(sma.is_empty());
    }

    #[test]
    fn test_sma_zero_window() {
        let prices = vec![1.0, 2.0, 3.0];
        let sma = calculate_sma(&prices, 0);
        assert!(sma.is_empty());
    }

    #[test]
    fn test_volatility_constant_prices() {
        let prices = vec![100.0; 10];
        let vol = calculate_volatility(&prices);
        assert!((vol - 0.0).abs() < 0.001);
    }

    #[test]
    fn test_volatility_positive() {
        let prices = vec![100.0, 105.0, 95.0, 110.0, 90.0];
        let vol = calculate_volatility(&prices);
        assert!(vol > 0.0);
    }

    #[test]
    fn test_volatility_insufficient() {
        assert!((calculate_volatility(&[100.0]) - 0.0).abs() < 0.001);
    }

    #[test]
    fn test_generate_historical_prices_reproducible() {
        let p1 = generate_historical_prices(100.0, 20, 0.3, 42);
        let p2 = generate_historical_prices(100.0, 20, 0.3, 42);
        assert_eq!(p1.len(), p2.len());
        for (a, b) in p1.iter().zip(p2.iter()) {
            assert!((a.price - b.price).abs() < 0.0001);
        }
    }

    #[test]
    fn test_generate_historical_prices_empty() {
        let prices = generate_historical_prices(100.0, 0, 0.3, 42);
        assert!(prices.is_empty());
    }

    #[test]
    fn test_generate_historical_prices_positive() {
        let prices = generate_historical_prices(100.0, 50, 0.2, 1);
        assert!(!prices.is_empty());
        // All prices should be positive
        for pp in &prices {
            assert!(pp.price > 0.0);
        }
    }

    #[test]
    fn test_price_stats_basic() {
        let prices = vec![10.0, 20.0, 30.0, 40.0, 50.0];
        let stats = calculate_price_stats(&prices);
        assert!((stats.min - 10.0).abs() < 0.001);
        assert!((stats.max - 50.0).abs() < 0.001);
        assert!((stats.mean - 30.0).abs() < 0.001);
        assert!((stats.median - 30.0).abs() < 0.001);
    }

    #[test]
    fn test_price_stats_empty() {
        let stats = calculate_price_stats(&[]);
        assert!((stats.min - 0.0).abs() < 0.001);
        assert!((stats.std_dev - 0.0).abs() < 0.001);
    }

    #[test]
    fn test_price_stats_single() {
        let stats = calculate_price_stats(&[42.0]);
        assert!((stats.mean - 42.0).abs() < 0.001);
        assert!((stats.std_dev - 0.0).abs() < 0.001); // No variance with single value
    }

    #[test]
    fn test_max_drawdown() {
        let prices = vec![100.0, 120.0, 80.0, 110.0, 70.0];
        // Peak 120, trough 80 → DD = 40/120 = 0.333
        // Peak 120, trough 70 → DD = 50/120 = 0.417
        let dd = calculate_max_drawdown(&prices);
        assert!((dd - 50.0 / 120.0).abs() < 0.001);
    }

    #[test]
    fn test_max_drawdown_no_drawdown() {
        let prices = vec![100.0, 110.0, 120.0];
        let dd = calculate_max_drawdown(&prices);
        assert!((dd - 0.0).abs() < 0.001);
    }

    #[test]
    fn test_annualized_return() {
        // Double over ~252 days → ~100% return
        let prices: Vec<f64> = (0..253).map(|i| 100.0 * (1.0 + i as f64 / 252.0)).collect();
        let ann_ret = calculate_annualized_return(&prices);
        assert!(ann_ret > 0.0);
    }

    #[test]
    fn test_sharpe_ratio() {
        let prices = vec![100.0, 101.0, 102.0, 101.0, 103.0, 104.0, 105.0];
        let sharpe = calculate_sharpe_ratio(&prices, 0.02);
        // With positive drift, should be positive
        assert!(sharpe.is_finite());
    }
}
