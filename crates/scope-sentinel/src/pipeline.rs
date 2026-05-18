//! End-to-end integration pipeline for the scope-sentinel REIT analytics platform.
//!
//! Wires together the four major stages:
//! 1. **EDGAR parsing** (`edgar` module) — extract financial statements from SEC filings.
//! 2. **Metrics computation** (`metrics` module) — derive key REIT ratios (FFO, AFFO,
//!    NAV, Debt/EBITDA, …).
//! 3. **Portfolio analytics** (`portfolio` module) — sector weights, diversification,
//!    risk-adjusted returns.
//! 4. **Signal generation** (`signal` module) — value screening, momentum, sector
//!    rotation, composite scoring.
//!
//! The pipeline can operate at two levels:
//! - **Per-REIT enrichment** — compute ratios and generate a single signal.
//! - **Batch pipeline** — process a universe of REITs, attach signals, and optionally
//!   produce portfolio-level analytics.

use chrono::{DateTime, Utc};

use crate::metrics::compute_all_ratios;
use crate::portfolio::compute_portfolio_analytics;
use crate::signal::{
    composite_signal, value_screen, CompositeConfig, ValueScreenConfig,
};
use crate::types::{
    PortfolioAnalytics, PortfolioPosition, REIT, SignalResult,
};

// ---------------------------------------------------------------------------
// Configuration
// ---------------------------------------------------------------------------

/// Top-level pipeline configuration.
#[derive(Debug, Clone)]
pub struct PipelineConfig {
    /// Thresholds for the value-screening stage.
    pub value_screen_config: ValueScreenConfig,
    /// Weights and thresholds for the composite signal stage.
    pub composite_config: CompositeConfig,
    /// Risk-free rate used in Sharpe / Sortino / Treynor (default 0.02).
    pub risk_free_rate: f64,
}

impl Default for PipelineConfig {
    fn default() -> Self {
        Self {
            value_screen_config: ValueScreenConfig::default(),
            composite_config: CompositeConfig::default(),
            risk_free_rate: 0.02,
        }
    }
}

// ---------------------------------------------------------------------------
// Errors
// ---------------------------------------------------------------------------

/// Errors produced during pipeline execution.
#[derive(Debug, thiserror::Error)]
pub enum PipelineError {
    #[error("Metrics computation failed for {ticker}: {reason}")]
    MetricsFailed { ticker: String, reason: String },

    #[error("Portfolio analytics failed: {0}")]
    PortfolioFailed(String),

    #[error("No signals generated")]
    NoSignals,
}

// ---------------------------------------------------------------------------
// Result
// ---------------------------------------------------------------------------

/// Aggregated result of a full pipeline run.
#[derive(Debug, Clone)]
pub struct PipelineResult {
    /// The (possibly enriched) REITs that were processed.
    pub reits: Vec<REIT>,
    /// One signal per REIT that could be evaluated.
    pub signals: Vec<SignalResult>,
    /// Portfolio-level analytics (set when portfolio positions are provided).
    pub portfolio_analytics: Option<PortfolioAnalytics>,
    /// Non-fatal errors encountered along the way.
    pub errors: Vec<String>,
    /// Timestamp when the pipeline was executed.
    pub run_at: DateTime<Utc>,
}

// ---------------------------------------------------------------------------
// Core pipeline stages
// ---------------------------------------------------------------------------

/// Enrich a single REIT with computed [`FinancialRatios`].
///
/// Returns the same REIT with `ratios` populated, or an error if the required
/// financial statements are missing.
pub fn enrich_reit(reit: &REIT) -> Result<REIT, PipelineError> {
    let ratios = compute_all_ratios(reit).map_err(|e| PipelineError::MetricsFailed {
        ticker: reit.ticker.clone(),
        reason: e.to_string(),
    })?;
    let mut enriched = reit.clone();
    enriched.ratios = Some(ratios);
    Ok(enriched)
}

/// Generate a [`SignalResult`] for a REIT that already has ratios.
///
/// Uses the default [`CompositeConfig`]; for custom thresholds use
/// [`composite_signal`] directly.
pub fn generate_signal(reit: &REIT, config: &PipelineConfig) -> Option<SignalResult> {
    let ratios = reit.ratios.as_ref()?;
    let vs = value_screen(
        &reit.ticker,
        reit.share_price,
        ratios,
        &config.value_screen_config,
    );
    Some(composite_signal(
        reit,
        vs.overall_value_score,
        None, // basic pipeline has no price history
        &config.composite_config,
    ))
}

// ---------------------------------------------------------------------------
// Batch pipeline
// ---------------------------------------------------------------------------

/// Run the full pipeline over a vector of REITs.
///
/// Each REIT is enriched with financial ratios, value-screened, and turned into
/// a composite signal.  REITs that cannot be processed (e.g. missing financials)
/// are recorded in `errors` but do not halt the pipeline.
pub fn run_pipeline(reits: Vec<REIT>, config: &PipelineConfig) -> PipelineResult {
    let mut enriched = Vec::with_capacity(reits.len());
    let mut signals = Vec::with_capacity(reits.len());
    let mut errors = Vec::new();

    for reit in &reits {
        match enrich_reit(reit) {
            Ok(e) => {
                if let Some(sig) = generate_signal(&e, config) {
                    signals.push(sig);
                }
                enriched.push(e);
            }
            Err(err) => {
                errors.push(err.to_string());
                enriched.push(reit.clone());
            }
        }
    }

    PipelineResult {
        reits: enriched,
        signals,
        portfolio_analytics: None,
        errors,
        run_at: Utc::now(),
    }
}

/// Run the pipeline and also compute portfolio-level analytics from a set of
/// [`PortfolioPosition`] entries.
///
/// The portfolio positions are enriched with signals (via their attached ratios)
/// and a full [`PortfolioAnalytics`] summary is produced.
pub fn run_pipeline_with_portfolio(
    positions: Vec<PortfolioPosition>,
    config: &PipelineConfig,
) -> PipelineResult {
    // Derive a synthetic REIT from each position (for signal generation).
    let mut signals = Vec::with_capacity(positions.len());
    let mut errors = Vec::new();

    for pos in &positions {
        let reit = REIT {
            ticker: pos.ticker.clone(),
            name: pos.ticker.clone(),
            cik: String::new(),
            sector: pos.sector,
            inception_date: None,
            market_cap: pos.market_value,
            share_price: pos.current_price,
            shares_outstanding: pos.shares,
            balance_sheet: None,
            income_statement: None,
            ratios: pos.ratios.clone(),
        };

        if let Some(sig) = generate_signal(&reit, config) {
            signals.push(sig);
        }
    }

    // Compute portfolio analytics.
    let analytics = match compute_portfolio_analytics(&positions, None, None, config.risk_free_rate)
    {
        Ok(a) => Some(a),
        Err(e) => {
            errors.push(format!("Portfolio analytics: {}", e));
            None
        }
    };

    PipelineResult {
        reits: Vec::new(), // no enriched REITs in portfolio mode
        signals,
        portfolio_analytics: analytics,
        errors,
        run_at: Utc::now(),
    }
}

// ===========================================================================
// Tests
// ===========================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use crate::types::{BalanceSheet, FinancialRatios, IncomeStatement, REITSector};
    use chrono::Utc;

    // ---- helpers -------------------------------------------------------

    fn healthy_reit(ticker: &str, price: f64) -> REIT {
        REIT {
            ticker: ticker.into(),
            name: format!("{} Inc.", ticker),
            cik: "0000000001".into(),
            sector: REITSector::Residential,
            inception_date: None,
            market_cap: price * 1_000_000_000.0,
            share_price: price,
            shares_outstanding: 1_000_000_000.0,
            balance_sheet: Some(BalanceSheet {
                period_end: Utc::now(),
                real_estate_assets: 15_000_000.0,
                accumulated_depreciation: 3_000_000.0,
                net_real_estate_assets: 12_000_000.0,
                total_assets: 20_000_000.0,
                current_assets: 500_000.0,
                total_liabilities: 10_000_000.0,
                current_liabilities: 300_000.0,
                mortgage_debt: 5_000_000.0,
                unsecured_debt: 3_000_000.0,
                total_debt: 8_000_000.0,
                shareholders_equity: 10_000_000.0,
                cash: 400_000.0,
                restricted_cash: 100_000.0,
                shares_outstanding: 1_000_000_000.0,
            }),
            income_statement: Some(IncomeStatement {
                period_start: Utc::now(),
                period_end: Utc::now(),
                rental_revenue: 2_000_000.0,
                total_revenue: 2_200_000.0,
                same_store_noi_growth: Some(0.035),
                operating_expenses: 800_000.0,
                noi: 1_200_000.0,
                depreciation_amortization: 400_000.0,
                general_admin_expenses: 200_000.0,
                interest_expense: 250_000.0,
                interest_income: Some(10_000.0),
                ebit: 600_000.0,
                income_tax_expense: 70_000.0,
                net_income: 380_000.0,
                gains_losses_on_sales: Some(50_000.0),
                ffo_per_share: Some(3.80),
                dividends_per_share: Some(3.08),
                weighted_avg_shares: Some(1_000_000_000.0),
            }),
            ratios: None,
        }
    }

    fn bare_reit(ticker: &str) -> REIT {
        REIT {
            ticker: ticker.into(),
            name: ticker.into(),
            cik: "0000000000".into(),
            sector: REITSector::Specialty,
            inception_date: None,
            market_cap: 0.0,
            share_price: 0.0,
            shares_outstanding: 1_000_000.0,
            balance_sheet: None,
            income_statement: None,
            ratios: None,
        }
    }

    fn make_position(ticker: &str, sector: REITSector, weight: f64) -> PortfolioPosition {
        PortfolioPosition {
            ticker: ticker.into(),
            sector,
            shares: 100.0,
            cost_basis: 50.0,
            current_price: 60.0,
            weight,
            market_value: 6_000.0,
            unrealized_pnl: 1_000.0,
            unrealized_return_pct: 20.0,
            ratios: Some(FinancialRatios {
                computed_at: Utc::now(),
                ffo: 500.0,
                ffo_per_share: 5.0,
                affo_per_share: 4.5,
                nav_per_share: 80.0,
                price_to_ffo: 12.0,
                dividend_yield: 0.045,
                debt_to_ebitda: 4.0,
                interest_coverage: 4.5,
                cap_rate: 0.055,
                same_store_noi_growth: 0.03,
                current_ratio: 2.0,
                debt_to_equity: 0.8,
                return_on_equity: 0.08,
                operating_margin: 0.45,
            }),
        }
    }

    // ---- tests ---------------------------------------------------------

    #[test]
    fn test_enrich_reit_success() {
        let reit = healthy_reit("O", 55.0);
        let enriched = enrich_reit(&reit).unwrap();
        assert!(enriched.ratios.is_some());
        let ratios = enriched.ratios.unwrap();
        assert!(ratios.ffo_per_share > 0.0);
        assert!(ratios.dividend_yield > 0.0);
    }

    #[test]
    fn test_enrich_reit_missing_data() {
        let reit = bare_reit("EMPTY");
        let result = enrich_reit(&reit);
        assert!(result.is_err());
        let err = result.unwrap_err().to_string();
        assert!(err.contains("EMPTY"), "error should reference the ticker");
    }

    #[test]
    fn test_generate_signal() {
        let reit = healthy_reit("O", 55.0);
        let enriched = enrich_reit(&reit).unwrap();
        let config = PipelineConfig::default();
        let sig = generate_signal(&enriched, &config).unwrap();
        assert_eq!(sig.ticker, "O");
        // Confidence should be in [0, 1].
        assert!(sig.confidence >= 0.0 && sig.confidence <= 1.0);
    }

    #[test]
    fn test_run_pipeline_basic() {
        let reits = vec![healthy_reit("O", 55.0), healthy_reit("AMT", 180.0)];
        let config = PipelineConfig::default();
        let result = run_pipeline(reits, &config);

        assert_eq!(result.reits.len(), 2);
        assert_eq!(result.signals.len(), 2);
        assert!(result.errors.is_empty());
        // All enriched REITs should have ratios.
        for reit in &result.reits {
            assert!(reit.ratios.is_some());
        }
    }

    #[test]
    fn test_run_pipeline_with_incomplete_reit() {
        let reits = vec![healthy_reit("O", 55.0), bare_reit("X")];
        let config = PipelineConfig::default();
        let result = run_pipeline(reits, &config);

        assert_eq!(result.reits.len(), 2);
        assert_eq!(result.signals.len(), 1, "only the healthy REIT should get a signal");
        assert_eq!(result.errors.len(), 1, "one error expected for the bare REIT");
    }

    #[test]
    fn test_run_pipeline_with_portfolio() {
        let positions = vec![
            make_position("O", REITSector::Residential, 0.4),
            make_position("AMT", REITSector::DataCenter, 0.35),
            make_position("PSA", REITSector::SelfStorage, 0.25),
        ];
        let config = PipelineConfig::default();
        let result = run_pipeline_with_portfolio(positions, &config);

        assert_eq!(result.signals.len(), 3, "one signal per position");
        assert!(result.portfolio_analytics.is_some());
        let analytics = result.portfolio_analytics.unwrap();
        assert!(analytics.total_value > 0.0);
        assert!(analytics.diversification_score > 0.0);
        assert!(result.errors.is_empty());
    }

    #[test]
    fn test_pipeline_result_structure() {
        let reits = vec![healthy_reit("A", 50.0)];
        let config = PipelineConfig::default();
        let result = run_pipeline(reits, &config);

        // Verify timestamps are recent.
        let now = Utc::now();
        let age = (now - result.run_at).num_seconds().abs();
        assert!(age < 5, "run_at should be very recent");

        // Default pipeline should not have portfolio analytics.
        assert!(result.portfolio_analytics.is_none());
    }

    #[test]
    fn test_pipeline_config_default() {
        let config = PipelineConfig::default();
        assert!((config.risk_free_rate - 0.02).abs() < 1e-9);
        assert!((config.value_screen_config.max_price_to_ffo - 14.0).abs() < 1e-9);
        assert!((config.composite_config.value_weight - 0.40).abs() < 1e-9);
    }
}
