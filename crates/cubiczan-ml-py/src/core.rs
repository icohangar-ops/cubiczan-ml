//! # CubicZan ML Core — Python Bindings
//!
//! PyO3 module exposing cubiczan-ml-core financial math, time series,
//! risk management, preprocessing, and utility functions to Python.

use ndarray::{Array1, Array2};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::PyDict;

use cubiczan_ml_core::{
    DrawdownTracker, KellyCriterion, MinMaxScaler, MovingAverage, MovingAverageType,
    PortfolioMetrics, StandardScaler, Volatility,
};

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

/// Convert a `Vec<Vec<f64>>` (row-major, jagged-safe) to an `Array2<f64>`.
fn vec_to_array2(data: &[Vec<f64>]) -> Result<Array2<f64>, PyErr> {
    if data.is_empty() {
        return Err(PyValueError::new_err("empty data"));
    }
    let ncols = data[0].len();
    if ncols == 0 {
        return Err(PyValueError::new_err("empty rows"));
    }
    for (i, row) in data.iter().enumerate() {
        if row.len() != ncols {
            return Err(PyValueError::new_err(format!(
                "row 0 has {} columns but row {} has {} columns",
                ncols,
                i,
                row.len()
            )));
        }
    }
    let nrows = data.len();
    let flat: Vec<f64> = data.iter().flat_map(|row| row.iter().copied()).collect();
    Array2::from_shape_vec((nrows, ncols), flat)
        .map_err(|e| PyValueError::new_err(format!("invalid shape: {}", e)))
}

/// Convert an `Array2<f64>` to `Vec<Vec<f64>>`.
fn array2_to_vec(data: &Array2<f64>) -> Vec<Vec<f64>> {
    let (nrows, _) = data.dim();
    (0..nrows).map(|r| data.row(r).to_vec()).collect()
}

/// Compute RSI using Wilder's smoothing method.
///
/// The core crate does not expose a standalone RSI function, so we implement
/// it here following the standard 14-period convention.
fn compute_rsi(prices: &[f64], period: usize) -> Result<Vec<f64>, String> {
    if period == 0 {
        return Err("period must be > 0".into());
    }
    if prices.len() < period + 1 {
        return Err(format!(
            "need at least {} prices for period {}, got {}",
            period + 1,
            period,
            prices.len()
        ));
    }
    let n = prices.len();
    let mut result = vec![f64::NAN; n];

    // Price changes
    let changes: Vec<f64> = (1..n).map(|i| prices[i] - prices[i - 1]).collect();

    // Initial average gain / loss
    let mut avg_gain = changes[..period].iter().map(|&c| c.max(0.0)).sum::<f64>() / period as f64;
    let mut avg_loss = changes[..period].iter().map(|&c| (-c).max(0.0)).sum::<f64>() / period as f64;

    let rs = if avg_loss < 1e-15 {
        f64::INFINITY
    } else {
        avg_gain / avg_loss
    };
    result[period] = 100.0 - 100.0 / (1.0 + rs);

    // Wilder's exponential smoothing for subsequent values
    for i in period..changes.len() {
        let gain = changes[i].max(0.0);
        let loss = (-changes[i]).max(0.0);
        avg_gain = (avg_gain * (period - 1) as f64 + gain) / period as f64;
        avg_loss = (avg_loss * (period - 1) as f64 + loss) / period as f64;
        let rs = if avg_loss < 1e-15 {
            f64::INFINITY
        } else {
            avg_gain / avg_loss
        };
        result[i + 1] = 100.0 - 100.0 / (1.0 + rs);
    }

    Ok(result)
}

// ===========================================================================
//  PyFunctions
// ===========================================================================

/// Simple Moving Average.
#[pyfunction]
pub fn py_sma(prices: Vec<f64>, window: usize) -> PyResult<Vec<f64>> {
    let data = Array1::from_vec(prices);
    MovingAverage::sma(&data, window)
        .map(|arr| arr.to_vec())
        .map_err(|e| PyValueError::new_err(e.to_string()))
}

/// Exponential Moving Average.
#[pyfunction]
pub fn py_ema(prices: Vec<f64>, window: usize) -> PyResult<Vec<f64>> {
    let data = Array1::from_vec(prices);
    MovingAverage::ema(&data, window)
        .map(|arr| arr.to_vec())
        .map_err(|e| PyValueError::new_err(e.to_string()))
}

/// Relative Strength Index (Wilder's smoothing).
#[pyfunction]
pub fn py_rsi(prices: Vec<f64>, period: usize) -> PyResult<Vec<f64>> {
    compute_rsi(&prices, period).map_err(|e| PyValueError::new_err(e))
}

/// Bollinger Bands — returns `(upper, middle, lower)`.
#[pyfunction]
pub fn py_bollinger_bands(
    prices: Vec<f64>,
    window: usize,
    num_std: f64,
) -> PyResult<(Vec<f64>, Vec<f64>, Vec<f64>)> {
    let data = Array1::from_vec(prices.clone());
    let middle = MovingAverage::sma(&data, window)
        .map_err(|e| PyValueError::new_err(e.to_string()))?;

    let rolling_std_vec = cubiczan_ml_core::time_series::rolling_std(&prices, window)
        .map_err(|e| PyValueError::new_err(e.to_string()))?;

    let n = prices.len();
    let mut upper = vec![f64::NAN; n];
    let mut lower = vec![f64::NAN; n];
    for i in 0..n {
        if !middle[i].is_nan() && !rolling_std_vec[i].is_nan() {
            upper[i] = middle[i] + num_std * rolling_std_vec[i];
            lower[i] = middle[i] - num_std * rolling_std_vec[i];
        }
    }
    Ok((upper, middle.to_vec(), lower))
}

/// Annualized realized volatility from close prices.
#[pyfunction]
#[pyo3(signature = (prices, annual_factor=252.0))]
pub fn py_volatility(prices: Vec<f64>, annual_factor: f64) -> PyResult<f64> {
    let data = Array1::from_vec(prices);
    Volatility::realized(&data, annual_factor)
        .map_err(|e| PyValueError::new_err(e.to_string()))
}

/// Weighted portfolio return: `sum(weights * asset_returns)`.
#[pyfunction]
pub fn py_portfolio_returns(weights: Vec<f64>, asset_returns: Vec<f64>) -> PyResult<f64> {
    if weights.len() != asset_returns.len() {
        return Err(PyValueError::new_err(format!(
            "weights length ({}) != asset_returns length ({})",
            weights.len(),
            asset_returns.len()
        )));
    }
    Ok(weights
        .iter()
        .zip(asset_returns.iter())
        .map(|(w, r)| w * r)
        .sum())
}

/// Sharpe ratio: `(mean(returns) - risk_free) / std(returns)`.
#[pyfunction]
pub fn py_sharpe_ratio(returns: Vec<f64>, risk_free: f64) -> PyResult<f64> {
    if returns.len() < 2 {
        return Err(PyValueError::new_err("need at least 2 returns"));
    }
    let n = returns.len() as f64;
    let mean = returns.iter().sum::<f64>() / n;
    let variance =
        returns.iter().map(|&x| (x - mean).powi(2)).sum::<f64>() / (n - 1.0);
    let std = variance.sqrt();
    if std < 1e-15 {
        return Ok(0.0);
    }
    Ok((mean - risk_free) / std)
}

/// Maximum drawdown from a price series.
#[pyfunction]
pub fn py_max_drawdown(prices: Vec<f64>) -> PyResult<f64> {
    if prices.len() < 2 {
        return Ok(0.0);
    }
    // Convert prices → simple returns
    let returns: Vec<f64> = (1..prices.len())
        .map(|i| {
            if prices[i - 1].abs() < 1e-15 {
                0.0
            } else {
                (prices[i] - prices[i - 1]) / prices[i - 1]
            }
        })
        .collect();
    let arr = Array1::from_vec(returns);
    Ok(PortfolioMetrics::max_drawdown(&arr))
}

/// Rolling mean over a sliding window.
#[pyfunction]
pub fn py_rolling_mean(prices: Vec<f64>, window: usize) -> PyResult<Vec<f64>> {
    cubiczan_ml_core::time_series::rolling_mean(&prices, window)
        .map_err(|e| PyValueError::new_err(e.to_string()))
}

/// Rolling standard deviation over a sliding window.
#[pyfunction]
pub fn py_rolling_std(prices: Vec<f64>, window: usize) -> PyResult<Vec<f64>> {
    cubiczan_ml_core::time_series::rolling_std(&prices, window)
        .map_err(|e| PyValueError::new_err(e.to_string()))
}

/// Log returns from a price series.
#[pyfunction]
pub fn py_log_returns(prices: Vec<f64>) -> PyResult<Vec<f64>> {
    cubiczan_ml_core::time_series::compute_returns(
        &prices,
        cubiczan_ml_core::time_series::ReturnType::Log,
    )
    .map_err(|e| PyValueError::new_err(e.to_string()))
}

/// Min-max normalize a 1-D array to [0, 1].
#[pyfunction]
pub fn py_normalize(data: Vec<f64>) -> PyResult<Vec<f64>> {
    if data.is_empty() {
        return Ok(vec![]);
    }
    let min_val = data.iter().cloned().fold(f64::INFINITY, f64::min);
    let max_val = data.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
    let range = max_val - min_val;
    if range.abs() < 1e-15 {
        // Constant feature → map to midpoint 0.5
        return Ok(vec![0.5; data.len()]);
    }
    Ok(data.iter().map(|&x| (x - min_val) / range).collect())
}

/// Numerically stable softmax.
#[pyfunction]
pub fn py_softmax(values: Vec<f64>) -> PyResult<Vec<f64>> {
    let mut v = values;
    cubiczan_ml_core::utils::softmax(&mut v);
    Ok(v)
}

/// Logistic sigmoid function.
#[pyfunction]
pub fn py_sigmoid(x: f64) -> PyResult<f64> {
    Ok(cubiczan_ml_core::utils::sigmoid(x))
}

/// Kelly criterion optimal fraction.
///
/// Given `win_rate` and `payoff_ratio` (win/loss ratio *b*), computes:
/// `f* = p - (1-p) / b` (full Kelly).
#[pyfunction]
pub fn py_kelly_fraction(win_rate: f64, payoff_ratio: f64) -> PyResult<f64> {
    if !(0.0..=1.0).contains(&win_rate) {
        return Err(PyValueError::new_err(format!(
            "win_rate must be in [0, 1], got {}",
            win_rate
        )));
    }
    if payoff_ratio <= 0.0 {
        return Err(PyValueError::new_err(
            "payoff_ratio must be > 0",
        ));
    }
    // Map to core API: avg_win = payoff_ratio, avg_loss = 1.0, fraction = 1.0
    // so that b = avg_win / avg_loss = payoff_ratio
    KellyCriterion::compute_fraction(win_rate, payoff_ratio, 1.0, 1.0)
        .map_err(|e| PyValueError::new_err(e.to_string()))
}

/// Kelly criterion position sizing — returns a dict with position details.
#[pyfunction]
pub fn py_kelly_position(
    py: Python<'_>,
    capital: f64,
    price: f64,
    win_rate: f64,
    avg_win: f64,
    avg_loss: f64,
    fraction: f64,
) -> PyResult<PyObject> {
    let pos = KellyCriterion::compute_position(
        capital, win_rate, avg_win, avg_loss, fraction, price,
    )
    .map_err(|e| PyValueError::new_err(e.to_string()))?;

    let d = PyDict::new(py);
    d.set_item("shares", pos.units)?;
    d.set_item("dollar_amount", pos.size)?;
    d.set_item("fraction", pos.fraction)?;
    d.set_item("risk_amount", pos.risk_amount)?;
    d.set_item("method", &pos.method)?;
    Ok(d.unbind().into_any())
}

// ===========================================================================
//  PyClasses
// ===========================================================================

// ---------------------------------------------------------------------------
// PyMovingAverage
// ---------------------------------------------------------------------------

#[pyclass(name = "MovingAverage")]
pub struct PyMovingAverage;

#[pymethods]
impl PyMovingAverage {
    #[new]
    fn new() -> Self {
        PyMovingAverage
    }

    /// Compute a moving average.
    ///
    /// * `kind` — `"sma"`, `"ema"`, or `"wma"`.
    #[pyo3(signature = (prices, window, kind))]
    fn compute(&self, prices: Vec<f64>, window: usize, kind: &str) -> PyResult<Vec<f64>> {
        let data = Array1::from_vec(prices);
        let ma_type = match kind.to_lowercase().as_str() {
            "sma" => MovingAverageType::SMA,
            "ema" => MovingAverageType::EMA,
            "wma" => MovingAverageType::WMA,
            _ => {
                return Err(PyValueError::new_err(format!(
                    "unknown moving average kind: '{}'. Expected 'sma', 'ema', or 'wma'",
                    kind
                )))
            }
        };
        MovingAverage::compute(&data, window, ma_type)
            .map(|arr| arr.to_vec())
            .map_err(|e| PyValueError::new_err(e.to_string()))
    }
}

// ---------------------------------------------------------------------------
// PyMinMaxScaler
// ---------------------------------------------------------------------------

#[pyclass(name = "MinMaxScaler")]
pub struct PyMinMaxScaler {
    inner: MinMaxScaler,
}

#[pymethods]
impl PyMinMaxScaler {
    #[new]
    fn new() -> Self {
        PyMinMaxScaler {
            inner: MinMaxScaler::new(),
        }
    }

    /// Fit the scaler to 2-D data (list of rows).
    fn fit(&mut self, data: Vec<Vec<f64>>) -> PyResult<()> {
        let arr = vec_to_array2(&data)?;
        self.inner
            .fit(&arr)
            .map_err(|e| PyValueError::new_err(e.to_string()))
    }

    /// Transform data using the fitted scaler.
    fn transform(&self, data: Vec<Vec<f64>>) -> PyResult<Vec<Vec<f64>>> {
        let arr = vec_to_array2(&data)?;
        let result = self
            .inner
            .transform(&arr)
            .map_err(|e| PyValueError::new_err(e.to_string()))?;
        Ok(array2_to_vec(&result))
    }

    /// Fit and transform in one step.
    fn fit_transform(&mut self, data: Vec<Vec<f64>>) -> PyResult<Vec<Vec<f64>>> {
        let arr = vec_to_array2(&data)?;
        let result = self
            .inner
            .fit_transform(&arr)
            .map_err(|e| PyValueError::new_err(e.to_string()))?;
        Ok(array2_to_vec(&result))
    }
}

// ---------------------------------------------------------------------------
// PyStandardScaler
// ---------------------------------------------------------------------------

#[pyclass(name = "StandardScaler")]
pub struct PyStandardScaler {
    inner: StandardScaler,
}

#[pymethods]
impl PyStandardScaler {
    #[new]
    fn new() -> Self {
        PyStandardScaler {
            inner: StandardScaler::new(),
        }
    }

    /// Fit the scaler to 2-D data (list of rows).
    fn fit(&mut self, data: Vec<Vec<f64>>) -> PyResult<()> {
        let arr = vec_to_array2(&data)?;
        self.inner
            .fit(&arr)
            .map_err(|e| PyValueError::new_err(e.to_string()))
    }

    /// Transform data using the fitted scaler.
    fn transform(&self, data: Vec<Vec<f64>>) -> PyResult<Vec<Vec<f64>>> {
        let arr = vec_to_array2(&data)?;
        let result = self
            .inner
            .transform(&arr)
            .map_err(|e| PyValueError::new_err(e.to_string()))?;
        Ok(array2_to_vec(&result))
    }

    /// Fit and transform in one step.
    fn fit_transform(&mut self, data: Vec<Vec<f64>>) -> PyResult<Vec<Vec<f64>>> {
        let arr = vec_to_array2(&data)?;
        let result = self
            .inner
            .fit_transform(&arr)
            .map_err(|e| PyValueError::new_err(e.to_string()))?;
        Ok(array2_to_vec(&result))
    }
}

// ---------------------------------------------------------------------------
// PyRiskMetrics
// ---------------------------------------------------------------------------

#[pyclass(name = "RiskMetrics")]
pub struct PyRiskMetrics;

#[pymethods]
impl PyRiskMetrics {
    #[new]
    fn new() -> Self {
        PyRiskMetrics
    }

    /// Compute portfolio risk metrics from periodic returns.
    ///
    /// Returns a dict with: `sharpe`, `sortino`, `max_dd`, `annualized_return`,
    /// `annualized_vol`, `var`, `cvar`, `calmar`, `skewness`, `kurtosis`.
    #[pyo3(signature = (returns, risk_free))]
    fn compute(
        &self,
        py: Python<'_>,
        returns: Vec<f64>,
        risk_free: f64,
    ) -> PyResult<PyObject> {
        let data = Array1::from_vec(returns);
        // Use 252 periods/year for annualization, 95% VaR confidence
        let metrics = PortfolioMetrics::compute(&data, 252.0, risk_free, 0.95)
            .map_err(|e| PyValueError::new_err(e.to_string()))?;

        let d = PyDict::new(py);
        d.set_item("sharpe", metrics.sharpe_ratio)?;
        d.set_item("sortino", metrics.sortino_ratio)?;
        d.set_item("max_dd", metrics.max_drawdown)?;
        d.set_item("annualized_return", metrics.annualized_return)?;
        d.set_item("annualized_vol", metrics.annualized_volatility)?;
        d.set_item("var", metrics.var)?;
        d.set_item("cvar", metrics.cvar)?;
        d.set_item("calmar", metrics.calmar_ratio)?;
        d.set_item("skewness", metrics.skewness)?;
        d.set_item("kurtosis", metrics.kurtosis)?;
        Ok(d.unbind().into_any())
    }
}

// ---------------------------------------------------------------------------
// PyDrawdownTracker
// ---------------------------------------------------------------------------

#[pyclass(name = "DrawdownTracker")]
pub struct PyDrawdownTracker {
    inner: DrawdownTracker,
}

#[pymethods]
impl PyDrawdownTracker {
    #[new]
    fn new(initial_value: f64, limit: f64) -> Self {
        PyDrawdownTracker {
            inner: DrawdownTracker::new(initial_value, limit),
        }
    }

    /// Update the tracker with a new portfolio value.
    ///
    /// Raises `ValueError` if the drawdown limit is breached.
    fn update(&mut self, value: f64) -> PyResult<()> {
        self.inner
            .update(value)
            .map_err(|e| PyValueError::new_err(e.to_string()))
    }

    /// Current drawdown as a positive fraction (e.g. 0.15 = 15%).
    #[getter]
    fn current_drawdown(&self) -> f64 {
        self.inner.current_drawdown
    }

    /// Maximum observed drawdown as a positive fraction.
    #[getter]
    fn max_drawdown(&self) -> f64 {
        self.inner.max_drawdown
    }

    /// Recovery percentage needed to get back to the high-water mark.
    fn recovery_needed(&self) -> f64 {
        self.inner.recovery_needed()
    }
}
