use pyo3::prelude::*;

mod core;
mod nlp;
mod rl;
mod dl;
mod tf;

/// CubicZan ML — High-performance ML library for finance/DeFi.
///
/// Python bindings for the Cubiczan Rust ML ecosystem.
///
/// Submodules:
/// - `cubiczan_ml.core` — Financial math, time series, risk, preprocessing
/// - `cubiczan_ml.nlp` — Sentiment analysis, NER, classification, tokenization
/// - `cubiczan_ml.rl` — Reinforcement learning agents and trading environments
/// - `cubiczan_ml.dl` — Deep learning model configs and inference
/// - `cubiczan_ml.tf` — TensorFlow session management and bridges
#[pymodule]
fn cubiczan_ml(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(version, m)?)?;
    m.add_class::<core::PyMovingAverage>()?;
    m.add_class::<core::PyRiskMetrics>()?;
    m.add_class::<core::PyDrawdownTracker>()?;
    m.add_class::<core::PyMinMaxScaler>()?;
    m.add_class::<core::PyStandardScaler>()?;
    m.add_function(wrap_pyfunction!(core::py_kelly_fraction, m)?)?;
    m.add_function(wrap_pyfunction!(core::py_kelly_position, m)?)?;
    m.add_function(wrap_pyfunction!(core::py_sma, m)?)?;
    m.add_function(wrap_pyfunction!(core::py_ema, m)?)?;
    m.add_function(wrap_pyfunction!(core::py_rsi, m)?)?;
    m.add_function(wrap_pyfunction!(core::py_bollinger_bands, m)?)?;
    m.add_function(wrap_pyfunction!(core::py_volatility, m)?)?;
    m.add_function(wrap_pyfunction!(core::py_portfolio_returns, m)?)?;
    m.add_function(wrap_pyfunction!(core::py_sharpe_ratio, m)?)?;
    m.add_function(wrap_pyfunction!(core::py_max_drawdown, m)?)?;
    m.add_function(wrap_pyfunction!(core::py_normalize, m)?)?;
    m.add_function(wrap_pyfunction!(core::py_softmax, m)?)?;
    m.add_function(wrap_pyfunction!(core::py_sigmoid, m)?)?;
    m.add_function(wrap_pyfunction!(core::py_rolling_mean, m)?)?;
    m.add_function(wrap_pyfunction!(core::py_rolling_std, m)?)?;
    m.add_function(wrap_pyfunction!(core::py_log_returns, m)?)?;
    m.add_class::<nlp::PyFinSentimentAnalyzer>()?;
    m.add_class::<nlp::PyFinancialNER>()?;
    m.add_class::<nlp::PyZeroShotClassifier>()?;
    m.add_function(wrap_pyfunction!(nlp::py_sentiment_label, m)?)?;
    m.add_class::<rl::PyTradingEnv>()?;
    m.add_class::<rl::PyQLearningAgent>()?;
    m.add_class::<rl::PyBacktestResult>()?;
    m.add_function(wrap_pyfunction!(rl::run_backtest, m)?)?;
    m.add_class::<dl::PyLstmConfig>()?;
    m.add_class::<dl::PyTransformerConfig>()?;
    m.add_class::<dl::PyAutoencoderConfig>()?;
    m.add_class::<dl::PyTrainingConfig>()?;
    m.add_class::<tf::PyTfSession>()?;
    m.add_class::<tf::PyPyTfBridge>()?;
    Ok(())
}

#[pyfunction]
fn version() -> &'static str {
    env!("CARGO_PKG_VERSION")
}
