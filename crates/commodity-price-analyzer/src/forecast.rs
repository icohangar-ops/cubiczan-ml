//! # Price Forecasting Engine
//!
//! Implements multiple forecasting methods for commodity prices including
//! linear regression, moving averages, mean reversion, exponential smoothing
//! (single, double, triple), and ensemble forecasting.

use chrono::Utc;
use crate::types::{CommodityType, PriceForecast, PricePoint};

// ---------------------------------------------------------------------------
// Forecast Configuration
// ---------------------------------------------------------------------------

/// Configuration for the forecasting engine.
#[derive(Debug, Clone)]
pub struct ForecastConfig {
    /// Number of historical data points to look back.
    pub lookback_period: usize,
    /// Number of periods to forecast ahead.
    pub forecast_horizon: u32,
    /// Minimum confidence threshold for a forecast to be considered valid.
    pub confidence_threshold: f64,
    /// Smoothing factor for exponential smoothing (0 < alpha < 1).
    pub smoothing_alpha: f64,
}

impl Default for ForecastConfig {
    fn default() -> Self {
        ForecastConfig {
            lookback_period: 60,
            forecast_horizon: 5,
            confidence_threshold: 0.3,
            smoothing_alpha: 0.3,
        }
    }
}

impl ForecastConfig {
    /// Create a new forecast config with custom parameters.
    pub fn new(lookback_period: usize, forecast_horizon: u32, confidence_threshold: f64, smoothing_alpha: f64) -> Self {
        ForecastConfig {
            lookback_period,
            forecast_horizon,
            confidence_threshold: confidence_threshold.clamp(0.0, 1.0),
            smoothing_alpha: smoothing_alpha.clamp(0.01, 0.99),
        }
    }
}

// ---------------------------------------------------------------------------
// Forecast Method Weights
// ---------------------------------------------------------------------------

/// Weights for ensemble forecast combination.
#[derive(Debug, Clone)]
pub struct EnsembleWeights {
    pub linear_regression: f64,
    pub moving_average: f64,
    pub mean_reversion: f64,
    pub exp_smoothing: f64,
}

impl Default for EnsembleWeights {
    fn default() -> Self {
        EnsembleWeights {
            linear_regression: 0.25,
            moving_average: 0.25,
            mean_reversion: 0.25,
            exp_smoothing: 0.25,
        }
    }
}

impl EnsembleWeights {
    /// Normalize weights so they sum to 1.0.
    pub fn normalize(&mut self) {
        let sum = self.linear_regression + self.moving_average + self.mean_reversion + self.exp_smoothing;
        if sum.abs() < 1e-15 {
            self.linear_regression = 0.25;
            self.moving_average = 0.25;
            self.mean_reversion = 0.25;
            self.exp_smoothing = 0.25;
        } else {
            self.linear_regression /= sum;
            self.moving_average /= sum;
            self.mean_reversion /= sum;
            self.exp_smoothing /= sum;
        }
    }
}

// ---------------------------------------------------------------------------
// Forecast Result (internal detail)
// ---------------------------------------------------------------------------

/// Internal result from a single forecasting method.
#[derive(Debug, Clone)]
struct MethodForecast {
    predicted_price: f64,
    confidence: f64,
    method_name: String,
}

// ---------------------------------------------------------------------------
// Forecast Engine
// ---------------------------------------------------------------------------

/// The main forecasting engine for commodity price prediction.
pub struct ForecastEngine {
    config: ForecastConfig,
    weights: EnsembleWeights,
}

impl ForecastEngine {
    /// Create a new forecasting engine with default configuration.
    pub fn new() -> Self {
        ForecastEngine {
            config: ForecastConfig::default(),
            weights: EnsembleWeights::default(),
        }
    }

    /// Create a new forecasting engine with a custom configuration.
    pub fn with_config(config: ForecastConfig) -> Self {
        ForecastEngine {
            config,
            weights: EnsembleWeights::default(),
        }
    }

    /// Create a new forecasting engine with custom config and ensemble weights.
    pub fn with_config_and_weights(config: ForecastConfig, mut weights: EnsembleWeights) -> Self {
        weights.normalize();
        ForecastEngine { config, weights }
    }

    // -----------------------------------------------------------------------
    // Public forecast methods
    // -----------------------------------------------------------------------

    /// Linear regression trend projection.
    /// Fits y = a + b*x to the close prices and extrapolates.
    pub fn linear_regression_forecast(&self, prices: &[PricePoint]) -> PriceForecast {
        let commodity = CommodityType::Gold; // default; caller can override
        self.linear_regression_forecast_for(commodity, prices)
    }

    /// Linear regression forecast for a specific commodity.
    pub fn linear_regression_forecast_for(&self, commodity: CommodityType, prices: &[PricePoint]) -> PriceForecast {
        let data = self.slice_data(prices);
        if data.len() < 2 {
            return PriceForecast {
                commodity,
                timestamp: Utc::now(),
                predicted_price: prices.last().map(|p| p.close).unwrap_or(0.0),
                confidence: 0.0,
                horizon: self.config.forecast_horizon,
                model_version: "linear_regression".into(),
            };
        }

        let n = data.len() as f64;
        let closes: Vec<f64> = data.iter().map(|p| p.close).collect();

        // Compute means
        let mean_x = (n - 1.0) / 2.0;
        let mean_y = closes.iter().sum::<f64>() / n;

        // Compute slope (b) and intercept (a)
        let mut sum_xy = 0.0;
        let mut sum_xx = 0.0;
        let mut sum_yy = 0.0;
        for (i, y) in closes.iter().enumerate() {
            let xi = i as f64;
            let dx = xi - mean_x;
            let dy = y - mean_y;
            sum_xy += dx * dy;
            sum_xx += dx * dx;
            sum_yy += dy * dy;
        }

        let slope = if sum_xx.abs() < 1e-15 { 0.0 } else { sum_xy / sum_xx };
        let intercept = mean_y - slope * mean_x;

        // Predict at horizon
        let predicted = intercept + slope * (n + self.config.forecast_horizon as f64 - 1.0);
        let predicted = predicted.max(0.0);

        // R-squared as confidence proxy
        let total_ss = sum_yy;
        let residual_ss = closes.iter().enumerate()
            .map(|(i, y)| {
                let yhat = intercept + slope * i as f64;
                (y - yhat).powi(2)
            })
            .sum::<f64>();
        let r_squared = if total_ss.abs() < 1e-15 { 0.0 } else { 1.0 - residual_ss / total_ss };
        let confidence = r_squared.clamp(0.0, 1.0);

        PriceForecast {
            commodity,
            timestamp: Utc::now(),
            predicted_price: predicted,
            confidence,
            horizon: self.config.forecast_horizon,
            model_version: "linear_regression".into(),
        }
    }

    /// Moving average crossover signal detection.
    /// Returns a forecast based on short/long moving average crossover.
    pub fn moving_average_crossover(&self, commodity: CommodityType, prices: &[PricePoint], short_period: usize, long_period: usize) -> PriceForecast {
        let data = self.slice_data(prices);
        let last_price = data.last().map(|p| p.close).unwrap_or(0.0);

        if data.len() < long_period {
            return PriceForecast {
                commodity,
                timestamp: Utc::now(),
                predicted_price: last_price,
                confidence: 0.0,
                horizon: self.config.forecast_horizon,
                model_version: "ma_crossover".into(),
            };
        }

        let short_ma = self.compute_sma(&data, short_period);
        let long_ma = self.compute_sma(&data, long_period);

        let (predicted, confidence) = if short_ma > long_ma {
            // Bullish crossover — trend up
            let spread = (short_ma - long_ma) / long_ma;
            let predicted = last_price * (1.0 + spread * 0.5 * self.config.forecast_horizon as f64);
            let confidence = (spread * 10.0).clamp(0.1, 0.9);
            (predicted.max(0.0), confidence)
        } else if short_ma < long_ma {
            // Bearish crossover — trend down
            let spread = (long_ma - short_ma) / long_ma;
            let predicted = last_price * (1.0 - spread * 0.5 * self.config.forecast_horizon as f64);
            let confidence = (spread * 10.0).clamp(0.1, 0.9);
            (predicted.max(0.0), confidence)
        } else {
            (last_price, 0.3)
        };

        PriceForecast {
            commodity,
            timestamp: Utc::now(),
            predicted_price: predicted,
            confidence,
            horizon: self.config.forecast_horizon,
            model_version: "ma_crossover".into(),
        }
    }

    /// Mean reversion detection using z-score.
    /// Measures how far the current price deviates from the historical mean
    /// in terms of standard deviations.
    pub fn mean_reversion_forecast(&self, commodity: CommodityType, prices: &[PricePoint]) -> PriceForecast {
        let data = self.slice_data(prices);
        let last_price = data.last().map(|p| p.close).unwrap_or(0.0);

        if data.len() < 2 {
            return PriceForecast {
                commodity,
                timestamp: Utc::now(),
                predicted_price: last_price,
                confidence: 0.0,
                horizon: self.config.forecast_horizon,
                model_version: "mean_reversion".into(),
            };
        }

        let closes: Vec<f64> = data.iter().map(|p| p.close).collect();
        let mean = closes.iter().sum::<f64>() / closes.len() as f64;
        let variance = closes.iter().map(|c| (c - mean).powi(2)).sum::<f64>() / (closes.len() as f64 - 1.0);
        let std_dev = variance.sqrt();

        let z_score = if std_dev.abs() < 1e-15 { 0.0 } else { (last_price - mean) / std_dev };

        // Mean reversion: if z > 0 (overvalued), forecast down toward mean; if z < 0 (undervalued), forecast up
        let reversion_factor = 0.1; // how quickly it reverts per period
        let expected_return = -z_score * std_dev / last_price * reversion_factor * self.config.forecast_horizon as f64;
        let predicted = last_price * (1.0 + expected_return);
        let predicted = predicted.max(0.0);

        // Higher confidence when z-score is more extreme (stronger mean reversion signal)
        let confidence = (z_score.abs() * 0.2).clamp(0.1, 0.85);

        PriceForecast {
            commodity,
            timestamp: Utc::now(),
            predicted_price: predicted,
            confidence,
            horizon: self.config.forecast_horizon,
            model_version: "mean_reversion".into(),
        }
    }

    /// Compute the z-score for the last price in the series.
    pub fn compute_z_score(&self, prices: &[PricePoint]) -> f64 {
        let data = self.slice_data(prices);
        if data.len() < 2 { return 0.0; }

        let closes: Vec<f64> = data.iter().map(|p| p.close).collect();
        let mean = closes.iter().sum::<f64>() / closes.len() as f64;
        let variance = closes.iter().map(|c| (c - mean).powi(2)).sum::<f64>() / (closes.len() as f64 - 1.0);
        let std_dev = variance.sqrt();
        if std_dev.abs() < 1e-15 { return 0.0; }
        let last = *closes.last().unwrap();
        (last - mean) / std_dev
    }

    /// Simple (single) exponential smoothing.
    /// Uses the formula: S(t) = alpha * Y(t) + (1-alpha) * S(t-1)
    pub fn exponential_smoothing_single(&self, commodity: CommodityType, prices: &[PricePoint]) -> PriceForecast {
        let data = self.slice_data(prices);
        let last_price = data.last().map(|p| p.close).unwrap_or(0.0);

        if data.is_empty() {
            return PriceForecast {
                commodity,
                timestamp: Utc::now(),
                predicted_price: 0.0,
                confidence: 0.0,
                horizon: self.config.forecast_horizon,
                model_version: "exp_smoothing_single".into(),
            };
        }

        let closes: Vec<f64> = data.iter().map(|p| p.close).collect();
        let alpha = self.config.smoothing_alpha;

        // Initialize with first value
        let mut s = closes[0];
        for i in 1..closes.len() {
            s = alpha * closes[i] + (1.0 - alpha) * s;
        }

        // For single ES, the forecast is flat at the last smoothed value
        let predicted = s.max(0.0);

        // Confidence based on recent error
        let errors: Vec<f64> = closes.iter().skip(1).enumerate().map(|(i, &y)| {
            let mut s_i = closes[0];
            for j in 1..=i {
                s_i = alpha * closes[j] + (1.0 - alpha) * s_i;
            }
            (y - s_i).abs()
        }).collect();

        let mae = if errors.is_empty() { 0.0 } else { errors.iter().sum::<f64>() / errors.len() as f64 };
        let confidence = (1.0 - mae / last_price.abs().max(1e-15) * 10.0).clamp(0.1, 0.9);

        PriceForecast {
            commodity,
            timestamp: Utc::now(),
            predicted_price: predicted,
            confidence,
            horizon: self.config.forecast_horizon,
            model_version: "exp_smoothing_single".into(),
        }
    }

    /// Double exponential smoothing (Brown's method).
    /// Adds a trend component to single exponential smoothing.
    pub fn exponential_smoothing_double(&self, commodity: CommodityType, prices: &[PricePoint]) -> PriceForecast {
        let data = self.slice_data(prices);
        let last_price = data.last().map(|p| p.close).unwrap_or(0.0);

        if data.len() < 2 {
            return PriceForecast {
                commodity,
                timestamp: Utc::now(),
                predicted_price: last_price,
                confidence: 0.0,
                horizon: self.config.forecast_horizon,
                model_version: "exp_smoothing_double".into(),
            };
        }

        let closes: Vec<f64> = data.iter().map(|p| p.close).collect();
        let alpha = self.config.smoothing_alpha;

        // Single smoothing
        let mut s: Vec<f64> = vec![closes[0]];
        for i in 1..closes.len() {
            s.push(alpha * closes[i] + (1.0 - alpha) * s[i - 1]);
        }

        // Double smoothing (smooth the smoothed series)
        let mut s2: Vec<f64> = vec![s[0]];
        for i in 1..s.len() {
            s2.push(alpha * s[i] + (1.0 - alpha) * s2[i - 1]);
        }

        // Level and trend at the end
        let a = 2.0 * s[s.len() - 1] - s2[s2.len() - 1]; // level
        let b = (alpha / (1.0 - alpha)) * (s[s.len() - 1] - s2[s2.len() - 1]); // trend

        let h = self.config.forecast_horizon as f64;
        let predicted = (a + b * h).max(0.0);

        // Confidence based on trend strength
        let trend_strength = if last_price.abs() < 1e-15 { 0.0 } else { (b / last_price).abs() };
        let confidence = (1.0 - trend_strength * 5.0).clamp(0.1, 0.9);

        PriceForecast {
            commodity,
            timestamp: Utc::now(),
            predicted_price: predicted,
            confidence,
            horizon: self.config.forecast_horizon,
            model_version: "exp_smoothing_double".into(),
        }
    }

    /// Triple exponential smoothing (Holt-Winters method, additive).
    /// Adds a seasonal component with configurable season length.
    pub fn exponential_smoothing_triple(&self, commodity: CommodityType, prices: &[PricePoint], season_length: usize) -> PriceForecast {
        let data = self.slice_data(prices);

        if data.len() < season_length * 2 {
            // Not enough data for triple smoothing, fall back to double
            return self.exponential_smoothing_double(commodity, prices);
        }

        let closes: Vec<f64> = data.iter().map(|p| p.close).collect();
        let alpha = self.config.smoothing_alpha;
        let beta = self.config.smoothing_alpha; // use same alpha for simplicity
        let gamma = self.config.smoothing_alpha;
        let m = season_length;

        // Initialize seasonality with averages for each season position
        let n_seasons = closes.len() / m;
        let mut seasonals: Vec<f64> = vec![0.0; m];
        for s in 0..m {
            let mut sum = 0.0;
            for k in 0..n_seasons {
                sum += closes[s + k * m];
            }
            seasonals[s] = sum / n_seasons as f64;
        }
        // Normalize seasonals to sum to 0
        let s_mean = seasonals.iter().sum::<f64>() / m as f64;
        for s in seasonals.iter_mut() {
            *s -= s_mean;
        }

        // Initialize level and trend
        let mut level = closes[0] - seasonals[0];
        let mut trend = if closes.len() >= m { (closes[m] - closes[0]) / m as f64 } else { 0.0 };

        // Run Holt-Winters
        for i in 1..closes.len() {
            let prev_seasonal = seasonals[(i - 1) % m];
            let new_level = alpha * (closes[i] - prev_seasonal) + (1.0 - alpha) * (level + trend);
            let new_trend = beta * (new_level - level) + (1.0 - beta) * trend;
            let new_seasonal = gamma * (closes[i] - new_level) + (1.0 - gamma) * prev_seasonal;

            seasonals[(i - 1) % m] = new_seasonal;
            level = new_level;
            trend = new_trend;
        }

        let h = self.config.forecast_horizon as f64;
        let seasonal_idx = (closes.len() - 1) % m;
        let predicted = (level + trend * h + seasonals[seasonal_idx]).max(0.0);

        let confidence = 0.6; // moderate confidence for Holt-Winters

        PriceForecast {
            commodity,
            timestamp: Utc::now(),
            predicted_price: predicted,
            confidence,
            horizon: self.config.forecast_horizon,
            model_version: "exp_smoothing_triple".into(),
        }
    }

    /// Ensemble forecast combining multiple methods with configurable weights.
    pub fn ensemble_forecast(&self, commodity: CommodityType, prices: &[PricePoint]) -> PriceForecast {
        let lr = self.linear_regression_forecast_for(commodity, prices);
        let ma = self.moving_average_crossover(commodity, prices, 10, 30);
        let mr = self.mean_reversion_forecast(commodity, prices);
        let es = self.exponential_smoothing_single(commodity, prices);

        let weighted_price =
            self.weights.linear_regression * lr.predicted_price
            + self.weights.moving_average * ma.predicted_price
            + self.weights.mean_reversion * mr.predicted_price
            + self.weights.exp_smoothing * es.predicted_price;

        let weighted_confidence =
            self.weights.linear_regression * lr.confidence
            + self.weights.moving_average * ma.confidence
            + self.weights.mean_reversion * mr.confidence
            + self.weights.exp_smoothing * es.confidence;

        PriceForecast {
            commodity,
            timestamp: Utc::now(),
            predicted_price: weighted_price.max(0.0),
            confidence: weighted_confidence.clamp(0.0, 1.0),
            horizon: self.config.forecast_horizon,
            model_version: "ensemble".into(),
        }
    }

    /// Generate multiple forecasts for different horizons.
    pub fn multi_horizon_forecast(&self, commodity: CommodityType, prices: &[PricePoint], horizons: &[u32]) -> Vec<PriceForecast> {
        let original_horizon = self.config.forecast_horizon;
        let mut results = Vec::new();
        for &h in horizons {
            let mut engine = self.clone();
            engine.config.forecast_horizon = h;
            results.push(engine.ensemble_forecast(commodity, prices));
        }
        let _ = original_horizon;
        results
    }

    // -----------------------------------------------------------------------
    // Helper methods
    // -----------------------------------------------------------------------

    /// Slice data to the configured lookback period.
    fn slice_data<'a>(&self, prices: &'a [PricePoint]) -> &'a [PricePoint] {
        if prices.len() <= self.config.lookback_period {
            prices
        } else {
            &prices[prices.len() - self.config.lookback_period..]
        }
    }

    /// Compute simple moving average over the last `period` data points.
    fn compute_sma(&self, data: &[PricePoint], period: usize) -> f64 {
        if data.len() < period || period == 0 {
            return data.last().map(|p| p.close).unwrap_or(0.0);
        }
        let sum: f64 = data[data.len() - period..].iter().map(|p| p.close).sum();
        sum / period as f64
    }
}

impl Default for ForecastEngine {
    fn default() -> Self {
        Self::new()
    }
}

impl Clone for ForecastEngine {
    fn clone(&self) -> Self {
        ForecastEngine {
            config: self.config.clone(),
            weights: self.weights.clone(),
        }
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use chrono::{Duration, Utc};

    /// Helper: create linearly increasing price data.
    fn make_linear_prices(n: usize, start: f64, step: f64) -> Vec<PricePoint> {
        let now = Utc::now();
        (0..n)
            .map(|i| {
                let price = start + i as f64 * step;
                PricePoint::new(
                    now - Duration::days((n - 1 - i) as i64),
                    price - 0.5,
                    price + 1.0,
                    price - 1.0,
                    price,
                    1000.0,
                )
            })
            .collect()
    }

    /// Helper: create flat (constant) price data.
    fn make_flat_prices(n: usize, price: f64) -> Vec<PricePoint> {
        let now = Utc::now();
        (0..n)
            .map(|i| {
                PricePoint::new(
                    now - Duration::days((n - 1 - i) as i64),
                    price,
                    price + 0.5,
                    price - 0.5,
                    price,
                    1000.0,
                )
            })
            .collect()
    }

    /// Helper: create volatile oscillating price data.
    fn make_volatile_prices(n: usize, base: f64, amplitude: f64) -> Vec<PricePoint> {
        let now = Utc::now();
        (0..n)
            .map(|i| {
                let price = base + amplitude * (i as f64 * 0.5).sin();
                PricePoint::new(
                    now - Duration::days((n - 1 - i) as i64),
                    price - 1.0,
                    price + 2.0,
                    price - 2.0,
                    price,
                    1000.0,
                )
            })
            .collect()
    }

    #[test]
    fn test_default_config() {
        let config = ForecastConfig::default();
        assert_eq!(config.lookback_period, 60);
        assert_eq!(config.forecast_horizon, 5);
        assert!((config.smoothing_alpha - 0.3).abs() < 1e-10);
    }

    #[test]
    fn test_config_new_clamps() {
        let config = ForecastConfig::new(30, 10, 1.5, 2.0);
        assert!((config.confidence_threshold - 1.0).abs() < 1e-10);
        assert!((config.smoothing_alpha - 0.99).abs() < 1e-10);
    }

    #[test]
    fn test_ensemble_weights_normalize() {
        let mut w = EnsembleWeights { linear_regression: 2.0, moving_average: 1.0, mean_reversion: 1.0, exp_smoothing: 0.0 };
        w.normalize();
        assert!((w.linear_regression - 0.5).abs() < 1e-10);
        assert!((w.moving_average - 0.25).abs() < 1e-10);
    }

    #[test]
    fn test_ensemble_weights_zero_sum() {
        let mut w = EnsembleWeights { linear_regression: 0.0, moving_average: 0.0, mean_reversion: 0.0, exp_smoothing: 0.0 };
        w.normalize();
        assert!((w.linear_regression - 0.25).abs() < 1e-10);
    }

    #[test]
    fn test_linear_regression_trending_up() {
        let engine = ForecastEngine::new();
        let prices = make_linear_prices(30, 100.0, 2.0);
        let forecast = engine.linear_regression_forecast_for(CommodityType::Gold, &prices);
        // Trending up, so predicted should be above last price
        assert!(forecast.predicted_price > prices.last().unwrap().close);
        assert!(forecast.confidence > 0.0);
    }

    #[test]
    fn test_linear_regression_trending_down() {
        let engine = ForecastEngine::new();
        let prices = make_linear_prices(30, 200.0, -2.0);
        let forecast = engine.linear_regression_forecast_for(CommodityType::Silver, &prices);
        // Trending down
        assert!(forecast.predicted_price < prices.last().unwrap().close);
    }

    #[test]
    fn test_linear_regression_flat() {
        let engine = ForecastEngine::new();
        let prices = make_flat_prices(30, 100.0);
        let forecast = engine.linear_regression_forecast_for(CommodityType::Copper, &prices);
        // Flat data: predicted should be very close to the flat price
        assert!((forecast.predicted_price - 100.0).abs() < 1.0);
    }

    #[test]
    fn test_linear_regression_insufficient_data() {
        let engine = ForecastEngine::new();
        let prices = make_flat_prices(1, 100.0);
        let forecast = engine.linear_regression_forecast_for(CommodityType::Gold, &prices);
        assert!((forecast.confidence - 0.0).abs() < 1e-10);
    }

    #[test]
    fn test_ma_crossover_bullish() {
        let engine = ForecastEngine::with_config(ForecastConfig::new(60, 5, 0.3, 0.3));
        // Create data where recent prices are higher (bullish)
        let mut prices = make_flat_prices(40, 100.0);
        for i in 30..40 {
            prices[i] = PricePoint::new(
                prices[i].timestamp, 150.0, 155.0, 145.0, 150.0, 1000.0
            );
        }
        let forecast = engine.moving_average_crossover(CommodityType::Gold, &prices, 5, 20);
        // Short MA should be above long MA → bullish
        assert!(forecast.predicted_price > 100.0);
    }

    #[test]
    fn test_ma_crossover_insufficient_data() {
        let engine = ForecastEngine::new();
        let prices = make_flat_prices(5, 100.0);
        let forecast = engine.moving_average_crossover(CommodityType::Gold, &prices, 5, 20);
        assert!((forecast.confidence - 0.0).abs() < 1e-10);
    }

    #[test]
    fn test_mean_reversion_overvalued() {
        let engine = ForecastEngine::new();
        // Price jumps far above historical mean
        let mut prices = make_flat_prices(25, 100.0);
        prices.push(PricePoint::new(Utc::now(), 200.0, 210.0, 195.0, 200.0, 1000.0));
        let forecast = engine.mean_reversion_forecast(CommodityType::Gold, &prices);
        // Should forecast downward (reversion to mean)
        assert!(forecast.predicted_price < 200.0);
    }

    #[test]
    fn test_mean_reversion_undervalued() {
        let engine = ForecastEngine::new();
        let mut prices = make_flat_prices(25, 100.0);
        prices.push(PricePoint::new(Utc::now(), 50.0, 55.0, 45.0, 50.0, 1000.0));
        let forecast = engine.mean_reversion_forecast(CommodityType::Gold, &prices);
        // Should forecast upward
        assert!(forecast.predicted_price > 50.0);
    }

    #[test]
    fn test_z_score_computation() {
        let engine = ForecastEngine::new();
        let prices = make_flat_prices(30, 100.0);
        let z = engine.compute_z_score(&prices);
        // Flat data → z-score near 0
        assert!(z.abs() < 1.0);
    }

    #[test]
    fn test_z_score_extreme() {
        let engine = ForecastEngine::new();
        let mut prices = make_flat_prices(25, 100.0);
        prices.push(PricePoint::new(Utc::now(), 500.0, 510.0, 495.0, 500.0, 1000.0));
        let z = engine.compute_z_score(&prices);
        assert!(z > 2.0); // way above mean
    }

    #[test]
    fn test_exp_smoothing_single() {
        let engine = ForecastEngine::with_config(ForecastConfig::new(60, 1, 0.3, 0.5));
        let prices = make_linear_prices(30, 100.0, 2.0);
        let forecast = engine.exponential_smoothing_single(CommodityType::Gold, &prices);
        assert!(forecast.predicted_price > 0.0);
        assert!(forecast.confidence > 0.0);
    }

    #[test]
    fn test_exp_smoothing_double() {
        let engine = ForecastEngine::with_config(ForecastConfig::new(60, 3, 0.3, 0.3));
        let prices = make_linear_prices(30, 100.0, 2.0);
        let forecast = engine.exponential_smoothing_double(CommodityType::Gold, &prices);
        assert!(forecast.predicted_price > 0.0);
        // Double smoothing should capture trend → higher than last
        assert!(forecast.predicted_price >= prices.last().unwrap().close - 10.0);
    }

    #[test]
    fn test_exp_smoothing_triple() {
        let engine = ForecastEngine::with_config(ForecastConfig::new(100, 5, 0.3, 0.3));
        let prices = make_volatile_prices(60, 100.0, 10.0);
        let forecast = engine.exponential_smoothing_triple(CommodityType::Gold, &prices, 12);
        assert!(forecast.predicted_price > 0.0);
        assert_eq!(forecast.model_version, "exp_smoothing_triple");
    }

    #[test]
    fn test_exp_smoothing_triple_fallback() {
        let engine = ForecastEngine::new();
        let prices = make_flat_prices(5, 100.0);
        let forecast = engine.exponential_smoothing_triple(CommodityType::Gold, &prices, 12);
        // Should fall back to double smoothing
        assert_eq!(forecast.model_version, "exp_smoothing_double");
    }

    #[test]
    fn test_ensemble_forecast() {
        let engine = ForecastEngine::with_config_and_weights(
            ForecastConfig::new(30, 5, 0.3, 0.3),
            EnsembleWeights::default(),
        );
        let prices = make_linear_prices(40, 100.0, 1.0);
        let forecast = engine.ensemble_forecast(CommodityType::Gold, &prices);
        assert!(forecast.predicted_price > 0.0);
        assert_eq!(forecast.model_version, "ensemble");
        assert!(forecast.confidence >= 0.0 && forecast.confidence <= 1.0);
    }

    #[test]
    fn test_multi_horizon_forecast() {
        let engine = ForecastEngine::new();
        let prices = make_linear_prices(40, 100.0, 1.0);
        let forecasts = engine.multi_horizon_forecast(CommodityType::Gold, &prices, &[1, 5, 10]);
        assert_eq!(forecasts.len(), 3);
        assert_eq!(forecasts[0].horizon, 1);
        assert_eq!(forecasts[1].horizon, 5);
        assert_eq!(forecasts[2].horizon, 10);
    }

    #[test]
    fn test_forecast_engine_default() {
        let engine = ForecastEngine::default();
        let prices = make_linear_prices(20, 100.0, 1.0);
        let forecast = engine.linear_regression_forecast(&prices);
        assert!(forecast.predicted_price > 0.0);
    }

    #[test]
    fn test_forecast_non_negative() {
        let engine = ForecastEngine::new();
        let prices = make_linear_prices(10, 100.0, -50.0);
        let forecast = engine.linear_regression_forecast_for(CommodityType::Gold, &prices);
        assert!(forecast.predicted_price >= 0.0);
    }
}
