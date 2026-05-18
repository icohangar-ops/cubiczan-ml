//! Forecasting models: Holt-Winters, linear regression, confidence intervals, accuracy metrics.

use crate::timeseries::{mean, std_dev};
use statrs::distribution::ContinuousCDF;
use crate::types::*;
use chrono::{DateTime, Duration, Utc};

/// Holt-Winters exponential smoothing model parameters.
#[derive(Debug, Clone)]
pub struct HoltWintersParams {
    pub alpha: f64, // Level smoothing
    pub beta: f64,  // Trend smoothing
    pub gamma: f64, // Seasonal smoothing
    pub period: usize,
}

impl Default for HoltWintersParams {
    fn default() -> Self {
        HoltWintersParams {
            alpha: 0.3,
            beta: 0.1,
            gamma: 0.1,
            period: 12,
        }
    }
}

impl HoltWintersParams {
    pub fn new(alpha: f64, beta: f64, gamma: f64, period: usize) -> Self {
        HoltWintersParams {
            alpha: alpha.clamp(0.0, 1.0),
            beta: beta.clamp(0.0, 1.0),
            gamma: gamma.clamp(0.0, 1.0),
            period,
        }
    }
}

/// Holt-Winters model state after fitting.
#[derive(Debug, Clone)]
pub struct HoltWintersModel {
    pub params: HoltWintersParams,
    pub level: f64,
    pub trend: f64,
    pub seasonal: Vec<f64>,
    pub fitted_values: Vec<f64>,
    pub residuals: Vec<f64>,
    pub residual_std: f64,
}

impl HoltWintersModel {
    /// Fits the Holt-Winters additive model to the given data.
    pub fn fit(values: &[f64], params: HoltWintersParams) -> Result<Self> {
        let n = values.len();
        let period = params.period;

        if n < 2 * period {
            return Err(GlacierError::InsufficientData {
                required: 2 * period,
                actual: n,
            });
        }

        // Initialize level and trend from first period average
        let initial_level = values[..period].iter().sum::<f64>() / period as f64;

        // Initial trend from difference of first two period averages
        let second_level = values[period..2 * period].iter().sum::<f64>() / period as f64;
        let initial_trend = (second_level - initial_level) / period as f64;

        // Initialize seasonal indices
        let mut seasonal = Vec::with_capacity(period);
        for i in 0..period {
            seasonal.push(values[i] - initial_level);
        }
        // Normalize seasonal to sum to zero
        let s_mean: f64 = seasonal.iter().sum::<f64>() / period as f64;
        for s in seasonal.iter_mut() {
            *s -= s_mean;
        }

        // Run the smoothing iterations
        let mut level = initial_level;
        let mut trend = initial_trend;
        let mut fitted_values = Vec::with_capacity(n);
        let mut residuals = Vec::with_capacity(n);

        for i in 0..n {
            let s_idx = i % period;
            let fitted = level + trend + seasonal[s_idx];
            fitted_values.push(fitted);
            residuals.push(values[i] - fitted);

            // Update components
            let new_level = params.alpha * (values[i] - seasonal[s_idx])
                + (1.0 - params.alpha) * (level + trend);
            let new_trend = params.beta * (new_level - level)
                + (1.0 - params.beta) * trend;
            let new_seasonal = params.gamma * (values[i] - new_level - new_trend)
                + (1.0 - params.gamma) * seasonal[s_idx];

            level = new_level;
            trend = new_trend;
            seasonal[s_idx] = new_seasonal;
        }

        let residual_std = std_dev(&residuals);

        Ok(HoltWintersModel {
            params,
            level,
            trend,
            seasonal,
            fitted_values,
            residuals,
            residual_std,
        })
    }

    /// Produces forecasts for `horizon` steps ahead with confidence intervals.
    pub fn forecast(
        &self,
        horizon: usize,
        last_timestamp: DateTime<Utc>,
        confidence: f64,
    ) -> ForecastResult {
        let z_score = normal_z_score(confidence);
        let mut predictions = Vec::with_capacity(horizon);
        let mut lower = Vec::with_capacity(horizon);
        let mut upper = Vec::with_capacity(horizon);
        let mut timestamps = Vec::with_capacity(horizon);

        for h in 1..=horizon {
            let s_idx = (self.params.period + (h - 1) % self.params.period) % self.params.period;
            let point = self.level + h as f64 * self.trend + self.seasonal[s_idx];

            // Confidence interval widens with horizon
            let ci_width = z_score * self.residual_std * (h as f64).sqrt();

            predictions.push(point);
            lower.push(point - ci_width);
            upper.push(point + ci_width);
            timestamps.push(last_timestamp + Duration::days(h as i64));
        }

        ForecastResult {
            commodity: EnergyCommodity::NaturalGas, // placeholder
            model: ForecastModel::HoltWinters,
            predictions,
            lower_bound: lower,
            upper_bound: upper,
            timestamps,
            mae: 0.0,
            rmse: 0.0,
            mape: 0.0,
        }
    }
}

/// Linear regression model: y = intercept + slope * x.
#[derive(Debug, Clone)]
pub struct LinearRegression {
    pub intercept: f64,
    pub slope: f64,
    pub r_squared: f64,
    pub residual_std: f64,
}

impl LinearRegression {
    /// Fits a simple linear regression y = a + b*x.
    pub fn fit(y_values: &[f64]) -> Result<Self> {
        if y_values.len() < 2 {
            return Err(GlacierError::InsufficientData {
                required: 2,
                actual: y_values.len(),
            });
        }

        let n = y_values.len() as f64;
        let x_mean = (n - 1.0) / 2.0;
        let y_mean = mean(y_values);

        let mut ss_xy = 0.0_f64;
        let mut ss_xx = 0.0_f64;

        for (i, y) in y_values.iter().enumerate() {
            let x = i as f64;
            ss_xy += (x - x_mean) * (y - y_mean);
            ss_xx += (x - x_mean).powi(2);
        }

        if ss_xx.abs() < f64::EPSILON {
            return Err(GlacierError::NumericalError(
                "Cannot fit regression: zero variance in x".into(),
            ));
        }

        let slope = ss_xy / ss_xx;
        let intercept = y_mean - slope * x_mean;

        // Compute R-squared
        let ss_tot: f64 = y_values.iter().map(|y| (y - y_mean).powi(2)).sum();
        let ss_res: f64 = y_values
            .iter()
            .enumerate()
            .map(|(i, y)| {
                let pred = intercept + slope * i as f64;
                (y - pred).powi(2)
            })
            .sum();

        let r_squared = if ss_tot.abs() > f64::EPSILON {
            1.0 - ss_res / ss_tot
        } else {
            1.0
        };

        let residual_std = (ss_res / (n - 2.0).max(1.0)).sqrt();

        Ok(LinearRegression {
            intercept,
            slope,
            r_squared: r_squared.clamp(0.0, 1.0),
            residual_std,
        })
    }

    /// Predicts values for future steps.
    pub fn predict(&self, n_steps: usize, start_index: usize) -> Vec<f64> {
        (0..n_steps)
            .map(|i| self.intercept + self.slope * (start_index + i) as f64)
            .collect()
    }

    /// Predicts with confidence intervals.
    pub fn predict_with_ci(
        &self,
        n_steps: usize,
        start_index: usize,
        confidence: f64,
        n_train: usize,
    ) -> (Vec<f64>, Vec<f64>, Vec<f64>) {
        let z = normal_z_score(confidence);
        let x_mean = (n_train as f64 - 1.0) / 2.0;
        let ss_xx: f64 = (0..n_train)
            .map(|i| (i as f64 - x_mean).powi(2))
            .sum();

        let mut predictions = Vec::with_capacity(n_steps);
        let mut lower = Vec::with_capacity(n_steps);
        let mut upper = Vec::with_capacity(n_steps);

        for i in 0..n_steps {
            let x = (start_index + i) as f64;
            let pred = self.intercept + self.slope * x;
            let se = self.residual_std
                * (1.0 + 1.0 / n_train as f64 + (x - x_mean).powi(2) / ss_xx).sqrt();
            let width = z * se;

            predictions.push(pred);
            lower.push(pred - width);
            upper.push(pred + width);
        }

        (predictions, lower, upper)
    }

    /// Returns fitted values for the training data.
    pub fn fitted_values(&self, n: usize) -> Vec<f64> {
        (0..n).map(|i| self.intercept + self.slope * i as f64).collect()
    }
}

/// Computes the Mean Absolute Error (MAE) between two series.
pub fn mae(actual: &[f64], predicted: &[f64]) -> f64 {
    if actual.len() != predicted.len() || actual.is_empty() {
        return f64::NAN;
    }
    actual
        .iter()
        .zip(predicted.iter())
        .map(|(a, p)| (a - p).abs())
        .sum::<f64>()
        / actual.len() as f64
}

/// Computes the Root Mean Square Error (RMSE) between two series.
pub fn rmse(actual: &[f64], predicted: &[f64]) -> f64 {
    if actual.len() != predicted.len() || actual.is_empty() {
        return f64::NAN;
    }
    (actual
        .iter()
        .zip(predicted.iter())
        .map(|(a, p)| (a - p).powi(2))
        .sum::<f64>()
        / (actual.len() as f64))
        .sqrt()
}

/// Computes the Mean Absolute Percentage Error (MAPE) between two series.
pub fn mape(actual: &[f64], predicted: &[f64]) -> f64 {
    if actual.len() != predicted.len() || actual.is_empty() {
        return f64::NAN;
    }
    actual
        .iter()
        .zip(predicted.iter())
        .filter(|(a, _)| a.abs() > f64::EPSILON)
        .map(|(a, p)| ((a - p) / a).abs())
        .sum::<f64>()
        / (actual.len() as f64)
        * 100.0
}

/// Returns the approximate z-score for a given confidence level.
/// Uses a lookup table for common values, linear interpolation otherwise.
pub fn normal_z_score(confidence: f64) -> f64 {
    let table: &[(f64, f64)] = &[
        (0.50, 0.674),
        (0.60, 0.842),
        (0.70, 1.036),
        (0.80, 1.282),
        (0.85, 1.440),
        (0.90, 1.645),
        (0.95, 1.960),
        (0.975, 2.241),
        (0.99, 2.576),
        (0.995, 2.807),
        (0.999, 3.291),
    ];

    // Try statrs for exact value if available
    if let Ok(dist) = statrs::distribution::Normal::new(0.0, 1.0) {
        let z = dist.inverse_cdf((1.0 + confidence) / 2.0);
        if z.is_finite() {
            return z;
        }
    }

    // Fallback to table lookup with linear interpolation
    if confidence <= table[0].0 {
        return table[0].1;
    }
    if confidence >= table[table.len() - 1].0 {
        return table[table.len() - 1].1;
    }

    for window in table.windows(2) {
        let (c0, z0) = window[0];
        let (c1, z1) = window[1];
        if confidence >= c0 && confidence <= c1 {
            let frac = (confidence - c0) / (c1 - c0);
            return z0 + frac * (z1 - z0);
        }
    }

    1.96 // Default 95% confidence
}

/// Performs in-sample backtesting: fits on training data, evaluates on test data.
pub fn backtest_forecast(
    values: &[f64],
    train_ratio: f64,
    model_type: ForecastModel,
    horizon: usize,
    commodity: EnergyCommodity,
) -> Result<ForecastResult> {
    let split = (values.len() as f64 * train_ratio) as usize;
    if split < 2 || split >= values.len() {
        return Err(GlacierError::InvalidInput(
            "Train ratio produces invalid split".into(),
        ));
    }

    let train = &values[..split];
    let test = &values[split..];

    let (predictions, lower, upper) = match model_type {
        ForecastModel::HoltWinters => {
            let params = HoltWintersParams::default();
            let hw = HoltWintersModel::fit(train, params)?;
            let actual_preds = hw.forecast(horizon.min(test.len()), Utc::now(), 0.95);
            (
                actual_preds.predictions,
                actual_preds.lower_bound,
                actual_preds.upper_bound,
            )
        }
        ForecastModel::Regression => {
            let lr = LinearRegression::fit(train)?;
            let (preds, lo, hi) = lr.predict_with_ci(horizon.min(test.len()), split, 0.95, train.len());
            (preds, lo, hi)
        }
        ForecastModel::ARIMA => {
            // Simplified ARIMA-like: difference + linear regression on differences
            let diffs: Vec<f64> = train.windows(2).map(|w| w[1] - w[0]).collect();
            let lr = LinearRegression::fit(&diffs)?;
            let last = train[train.len() - 1];
            let preds: Vec<f64> = lr
                .predict(horizon.min(test.len()), diffs.len())
                .iter()
                .scan(last, |state, d| {
                    *state += d;
                    Some(*state)
                })
                .collect();
            let lo: Vec<f64> = preds.iter().map(|p| p - lr.residual_std * 1.96).collect();
            let hi: Vec<f64> = preds.iter().map(|p| p + lr.residual_std * 1.96).collect();
            (preds, lo, hi)
        }
    };

    let eval_len = predictions.len().min(test.len());
    let test_eval = &test[..eval_len];
    let pred_eval = &predictions[..eval_len];

    let timestamps: Vec<DateTime<Utc>> = (0..eval_len)
        .map(|i| Utc::now() + Duration::days(i as i64))
        .collect();

    Ok(ForecastResult {
        commodity,
        model: model_type,
        predictions: predictions[..eval_len].to_vec(),
        lower_bound: lower[..eval_len].to_vec(),
        upper_bound: upper[..eval_len].to_vec(),
        timestamps,
        mae: mae(test_eval, pred_eval),
        rmse: rmse(test_eval, pred_eval),
        mape: mape(test_eval, pred_eval),
    })
}

/// Cross-validates a forecasting model using walk-forward validation.
pub fn walk_forward_validation(
    values: &[f64],
    train_window: usize,
    test_window: usize,
    step_size: usize,
) -> Vec<(f64, f64, f64)> {
    let mut results = Vec::new();

    let mut start = 0;
    while start + train_window + test_window <= values.len() {
        let train = &values[start..start + train_window];
        let test = &values[start + train_window..start + train_window + test_window];

        if let Ok(lr) = LinearRegression::fit(train) {
            let preds = lr.predict(test_window, train_window);
            let mae_val = mae(test, &preds);
            let rmse_val = rmse(test, &preds);
            let mape_val = mape(test, &preds);
            if mae_val.is_finite() && rmse_val.is_finite() && mape_val.is_finite() {
                results.push((mae_val, rmse_val, mape_val));
            }
        }

        start += step_size;
        if start + train_window + test_window > values.len() && start + step_size <= values.len() {
            break;
        }
    }

    results
}

/// Computes a combined forecast by averaging multiple model predictions.
pub fn ensemble_forecast(forecasts: &[ForecastResult]) -> Option<ForecastResult> {
    if forecasts.is_empty() {
        return None;
    }

    let min_len = forecasts.iter().map(|f| f.predictions.len()).min()?;
    let commodity = forecasts[0].commodity;
    let timestamps = forecasts[0].timestamps.clone();

    let mut avg_preds = vec![0.0f64; min_len];
    let mut avg_lower = vec![0.0f64; min_len];
    let mut avg_upper = vec![0.0f64; min_len];

    for f in forecasts {
        for i in 0..min_len {
            avg_preds[i] += f.predictions[i];
            avg_lower[i] += f.lower_bound[i];
            avg_upper[i] += f.upper_bound[i];
        }
    }

    let n = forecasts.len() as f64;
    for i in 0..min_len {
        avg_preds[i] /= n;
        avg_lower[i] /= n;
        avg_upper[i] /= n;
    }

    let avg_mae = forecasts.iter().map(|f| f.mae).sum::<f64>() / n;
    let avg_rmse = forecasts.iter().map(|f| f.rmse).sum::<f64>() / n;
    let avg_mape = forecasts.iter().map(|f| f.mape).sum::<f64>() / n;

    Some(ForecastResult {
        commodity,
        model: ForecastModel::Regression, // Ensemble
        predictions: avg_preds,
        lower_bound: avg_lower,
        upper_bound: avg_upper,
        timestamps,
        mae: avg_mae,
        rmse: avg_rmse,
        mape: avg_mape,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_holt_winters_fit() {
        // Build a series with clear trend + seasonality
        let values: Vec<f64> = (0..36)
            .map(|i| {
                let trend = 100.0 + i as f64 * 2.0;
                let seasonal = if i % 12 < 6 { 10.0 } else { -10.0 };
                trend + seasonal + (i as f64 * 0.3).sin() * 2.0
            })
            .collect();

        let model = HoltWintersModel::fit(&values, HoltWintersParams::default()).unwrap();
        assert!(!model.fitted_values.is_empty());
        assert!(model.residual_std.is_finite());
    }

    #[test]
    fn test_holt_winters_insufficient_data() {
        let values = vec![1.0, 2.0, 3.0];
        let result = HoltWintersModel::fit(&values, HoltWintersParams::default());
        assert!(result.is_err());
    }

    #[test]
    fn test_holt_winters_forecast() {
        let values: Vec<f64> = (0..36)
            .map(|i| 100.0 + i as f64 * 2.0 + (i % 12) as f64 * 5.0)
            .collect();
        let model = HoltWintersModel::fit(&values, HoltWintersParams::default()).unwrap();
        let fc = model.forecast(5, Utc::now(), 0.95);
        assert_eq!(fc.predictions.len(), 5);
        assert_eq!(fc.lower_bound.len(), 5);
        assert_eq!(fc.upper_bound.len(), 5);
        // Upper bound should be above predictions
        for i in 0..5 {
            assert!(fc.lower_bound[i] <= fc.predictions[i]);
            assert!(fc.upper_bound[i] >= fc.predictions[i]);
        }
    }

    #[test]
    fn test_linear_regression_fit() {
        let values: Vec<f64> = (0..20).map(|i| 10.0 + i as f64 * 2.0).collect();
        let lr = LinearRegression::fit(&values).unwrap();
        assert!((lr.slope - 2.0).abs() < 1e-8);
        assert!((lr.intercept - 10.0).abs() < 1e-8);
        assert!((lr.r_squared - 1.0).abs() < 1e-8);
    }

    #[test]
    fn test_linear_regression_noisy() {
        let values: Vec<f64> = (0..50)
            .map(|i| 5.0 + i as f64 * 0.5 + (i as f64 * 1.7).sin() * 2.0)
            .collect();
        let lr = LinearRegression::fit(&values).unwrap();
        assert!((lr.slope - 0.5).abs() < 0.1); // Close to 0.5
        assert!(lr.r_squared < 1.0); // Not perfect fit
        assert!(lr.r_squared > 0.0); // But some explanatory power
    }

    #[test]
    fn test_linear_regression_insufficient() {
        let result = LinearRegression::fit(&[42.0]);
        assert!(result.is_err());
    }

    #[test]
    fn test_linear_regression_predict() {
        let values = vec![10.0, 20.0, 30.0, 40.0];
        let lr = LinearRegression::fit(&values).unwrap();
        let preds = lr.predict(3, 4);
        assert_eq!(preds.len(), 3);
        assert!((preds[0] - 50.0).abs() < 1e-8);
    }

    #[test]
    fn test_linear_regression_predict_with_ci() {
        // Use noisy data so residual_std > 0
        let values: Vec<f64> = (0..30).map(|i| 100.0 + i as f64 * 1.0 + (i as f64 * 1.3).sin() * 3.0).collect();
        let lr = LinearRegression::fit(&values).unwrap();
        let (preds, lower, upper) = lr.predict_with_ci(5, 30, 0.95, 30);
        assert_eq!(preds.len(), 5);
        for i in 0..5 {
            assert!(lower[i] <= preds[i]);
            assert!(upper[i] >= preds[i]);
            assert!(upper[i] - lower[i] > 0.0);
        }
    }

    #[test]
    fn test_mae() {
        let actual = vec![1.0, 2.0, 3.0, 4.0];
        let predicted = vec![1.1, 1.9, 3.2, 3.8];
        let err = mae(&actual, &predicted);
        assert!(err.is_finite());
        assert!(err > 0.0);
        assert!(err < 0.5);
    }

    #[test]
    fn test_rmse() {
        let actual = vec![1.0, 2.0, 3.0];
        let predicted = vec![2.0, 3.0, 4.0];
        let err = rmse(&actual, &predicted);
        assert!(err > 0.99 && err < 1.01, "RMSE should be ~1.0, got {}", err);
    }

    #[test]
    fn test_mape() {
        let actual = vec![100.0, 200.0, 300.0];
        let predicted = vec![110.0, 190.0, 310.0];
        let err = mape(&actual, &predicted);
        assert!(err.is_finite());
        assert!(err > 0.0);
    }

    #[test]
    fn test_mape_mismatched_lengths() {
        let err = mape(&[1.0, 2.0], &[1.0]);
        assert!(err.is_nan());
    }

    #[test]
    fn test_normal_z_score() {
        let z95 = normal_z_score(0.95);
        assert!((z95 - 1.96).abs() < 0.05);

        let z99 = normal_z_score(0.99);
        assert!((z99 - 2.576).abs() < 0.05);
    }

    #[test]
    fn test_backtest_forecast_regression() {
        let values: Vec<f64> = (0..60)
            .map(|i| 50.0 + i as f64 * 0.5 + (i as f64 * 0.8).sin() * 3.0)
            .collect();
        let result = backtest_forecast(
            &values,
            0.7,
            ForecastModel::Regression,
            10,
            EnergyCommodity::CrudeOil,
        )
        .unwrap();
        assert!(!result.predictions.is_empty());
        assert!(result.mae.is_finite());
        assert!(result.rmse.is_finite());
        assert!(result.mape.is_finite());
    }

    #[test]
    fn test_backtest_forecast_holt_winters() {
        let values: Vec<f64> = (0..60)
            .map(|i| 100.0 + i as f64 * 0.3 + if i % 12 < 6 { 5.0 } else { -5.0 })
            .collect();
        let result = backtest_forecast(
            &values,
            0.7,
            ForecastModel::HoltWinters,
            10,
            EnergyCommodity::NaturalGas,
        )
        .unwrap();
        assert!(!result.predictions.is_empty());
    }

    #[test]
    fn test_backtest_invalid_ratio() {
        let values = vec![1.0, 2.0, 3.0];
        // train_ratio=0.99 with 3 values gives split=2, test=1, so it doesn't error.
        // Use a ratio that makes split too close to len.
        let result = backtest_forecast(&values, 0.5, ForecastModel::Regression, 5, EnergyCommodity::CrudeOil);
        // split=1, train=[1.0], test=[2.0,3.0]; LR needs at least 2 points for train
        assert!(result.is_err());
    }

    #[test]
    fn test_walk_forward_validation() {
        let values: Vec<f64> = (0..50).map(|i| 10.0 + i as f64 * 0.5).collect();
        let results = walk_forward_validation(&values, 20, 5, 5);
        assert!(!results.is_empty());
        for (mae_v, rmse_v, mape_v) in &results {
            assert!(mae_v.is_finite());
            assert!(rmse_v.is_finite());
            assert!(mape_v.is_finite());
        }
    }

    #[test]
    fn test_ensemble_forecast() {
        let fc1 = ForecastResult {
            commodity: EnergyCommodity::CrudeOil,
            model: ForecastModel::Regression,
            predictions: vec![100.0, 101.0, 102.0],
            lower_bound: vec![99.0, 100.0, 101.0],
            upper_bound: vec![101.0, 102.0, 103.0],
            timestamps: vec![],
            mae: 1.0,
            rmse: 1.5,
            mape: 1.0,
        };
        let fc2 = ForecastResult {
            commodity: EnergyCommodity::CrudeOil,
            model: ForecastModel::HoltWinters,
            predictions: vec![102.0, 103.0, 104.0],
            lower_bound: vec![100.0, 101.0, 102.0],
            upper_bound: vec![104.0, 105.0, 106.0],
            timestamps: vec![],
            mae: 2.0,
            rmse: 2.5,
            mape: 2.0,
        };

        let ensemble = ensemble_forecast(&[fc1, fc2]).unwrap();
        assert_eq!(ensemble.predictions.len(), 3);
        assert!((ensemble.predictions[0] - 101.0).abs() < 1e-10);
        assert!((ensemble.mae - 1.5).abs() < 1e-10);
    }

    #[test]
    fn test_ensemble_forecast_empty() {
        assert!(ensemble_forecast(&[]).is_none());
    }

    #[test]
    fn test_holt_winters_params_clamping() {
        let params = HoltWintersParams::new(1.5, -0.5, 2.0, 12);
        assert!((params.alpha - 1.0).abs() < 1e-10);
        assert!((params.beta - 0.0).abs() < 1e-10);
        assert!((params.gamma - 1.0).abs() < 1e-10);
    }
}
