//! REIT signal generation module.
//!
//! Provides value screening (P/FFO, NAV discount, dividend yield), momentum signals
//! (price trend, volume changes), sector rotation detection, and a composite scoring
//! system that combines multiple signal dimensions into actionable Buy/Sell/Hold
//! recommendations.

use chrono::Utc;
use std::collections::HashMap;

use crate::metrics::health_check_score;
use crate::types::{FinancialRatios, REIT, REITSector, Signal, SignalResult};

// ---------------------------------------------------------------------------
// Value Screening
// ---------------------------------------------------------------------------

/// Configuration for value screening thresholds.
#[derive(Debug, Clone)]
pub struct ValueScreenConfig {
    /// Maximum P/FFO considered "cheap" (default 14.0).
    pub max_price_to_ffo: f64,
    /// Minimum dividend yield for value (default 0.03).
    pub min_dividend_yield: f64,
    /// Minimum NAV discount considered attractive (default 0.15 → 15 %).
    pub min_nav_discount: f64,
    /// Maximum acceptable Debt/EBITDA (default 6.0).
    pub max_debt_to_ebitda: f64,
    /// Minimum interest coverage (default 2.0).
    pub min_interest_coverage: f64,
}

impl Default for ValueScreenConfig {
    fn default() -> Self {
        Self {
            max_price_to_ffo: 14.0,
            min_dividend_yield: 0.03,
            min_nav_discount: 0.15,
            max_debt_to_ebitda: 6.0,
            min_interest_coverage: 2.0,
        }
    }
}

/// Result of value-screening a single REIT.
#[derive(Debug, Clone)]
pub struct ValueScreenResult {
    pub ticker: String,
    /// Whether the REIT passes all value criteria.
    pub passes: bool,
    /// Individual sub-scores in [-1, 1].
    pub p_ffo_score: f64,
    pub nav_discount_score: f64,
    pub dividend_yield_score: f64,
    pub leverage_score: f64,
    /// Weighted overall value score in [-1, 1].
    pub overall_value_score: f64,
    /// Human-readable rationale.
    pub rationale: String,
}

/// Compute NAV discount as a fraction.
///
/// Positive value means the share trades below NAV (undervalued).
/// `nav_discount = (nav_per_share - share_price) / nav_per_share`
pub fn nav_discount(share_price: f64, nav_per_share: f64) -> f64 {
    if nav_per_share.abs() < 1e-9 {
        return 0.0;
    }
    (nav_per_share - share_price) / nav_per_share
}

/// Screen a REIT based on its financial ratios and current share price.
pub fn value_screen(
    ticker: &str,
    share_price: f64,
    ratios: &FinancialRatios,
    config: &ValueScreenConfig,
) -> ValueScreenResult {
    // --- P/FFO score ---------------------------------------------------
    let p_ffo_score = if ratios.price_to_ffo > 0.0 {
        if ratios.price_to_ffo <= config.max_price_to_ffo * 0.7 {
            1.0
        } else if ratios.price_to_ffo <= config.max_price_to_ffo {
            0.5
        } else if ratios.price_to_ffo <= config.max_price_to_ffo * 1.5 {
            -0.25
        } else {
            -1.0
        }
    } else {
        0.0
    };

    // --- NAV discount score -------------------------------------------
    let discount = nav_discount(share_price, ratios.nav_per_share);
    let nav_discount_score = if discount >= config.min_nav_discount {
        1.0
    } else if discount >= 0.0 {
        0.3
    } else {
        -0.5
    };

    // --- Dividend yield score ------------------------------------------
    let dividend_yield_score = if ratios.dividend_yield >= config.min_dividend_yield * 1.5 {
        1.0
    } else if ratios.dividend_yield >= config.min_dividend_yield {
        0.5
    } else if ratios.dividend_yield > 0.0 {
        -0.25
    } else {
        -1.0
    };

    // --- Leverage score ------------------------------------------------
    let leverage_score =
        if ratios.debt_to_ebitda <= config.max_debt_to_ebitda * 0.6
            && ratios.interest_coverage >= config.min_interest_coverage * 1.5
        {
            1.0
        } else if ratios.debt_to_ebitda <= config.max_debt_to_ebitda
            && ratios.interest_coverage >= config.min_interest_coverage
        {
            0.5
        } else if ratios.debt_to_ebitda <= config.max_debt_to_ebitda * 1.5 {
            -0.25
        } else {
            -1.0
        };

    // --- Weighted composite -------------------------------------------
    let overall_value_score = p_ffo_score * 0.30
        + nav_discount_score * 0.25
        + dividend_yield_score * 0.25
        + leverage_score * 0.20;

    // --- Rationale -----------------------------------------------------
    let mut parts: Vec<String> = Vec::new();
    if p_ffo_score > 0.0 {
        parts.push("attractive P/FFO".into());
    } else if p_ffo_score < 0.0 {
        parts.push("elevated P/FFO".into());
    }
    if nav_discount_score > 0.0 {
        parts.push("trading below NAV".into());
    }
    if dividend_yield_score > 0.0 {
        parts.push("solid yield".into());
    }
    if leverage_score < 0.0 {
        parts.push("high leverage".into());
    }
    let rationale = if parts.is_empty() {
        "no strong value signals".into()
    } else {
        parts.join("; ")
    };

    ValueScreenResult {
        ticker: ticker.into(),
        passes: overall_value_score >= 0.25,
        p_ffo_score,
        nav_discount_score,
        dividend_yield_score,
        leverage_score,
        overall_value_score,
        rationale,
    }
}

// ---------------------------------------------------------------------------
// Momentum Signals
// ---------------------------------------------------------------------------

/// Result of momentum analysis for a REIT.
#[derive(Debug, Clone)]
pub struct MomentumResult {
    pub ticker: String,
    /// Price trend score in [-1, 1].
    pub trend_score: f64,
    /// Volume change score in [-1, 1].
    pub volume_score: f64,
    /// Combined momentum signal.
    pub momentum_signal: Signal,
    /// Human-readable rationale.
    pub rationale: String,
}

/// Compute a normalised price-trend score from a price series.
///
/// Returns a value in [-1, 1] where 1 is a strong uptrend and -1 is a strong
/// downtrend.  Uses a weighted combination of the full-period return and a
/// recent-acceleration component.
pub fn price_trend_score(prices: &[f64]) -> f64 {
    if prices.len() < 2 {
        return 0.0;
    }
    let first = prices[0];
    let last = prices[prices.len() - 1];
    if first.abs() < 1e-9 {
        return 0.0;
    }
    let pct_change = (last - first) / first;
    // ±20 % maps to ±1.0
    let base_trend = (pct_change / 0.20).clamp(-1.0, 1.0);

    // Recent acceleration: compare second half to first half.
    let mid = prices.len() / 2;
    if mid >= 1 && prices.len() - mid >= 2 {
        let mid_price = prices[mid];
        let recent_trend = if mid_price.abs() > 1e-9 {
            (last - mid_price) / mid_price
        } else {
            0.0
        };
        let recent_score = (recent_trend / 0.10).clamp(-1.0, 1.0);
        base_trend * 0.6 + recent_score * 0.4
    } else {
        base_trend
    }
}

/// Compute a normalised volume-change score.
///
/// Compares the average volume of the most recent third of the series to the
/// older portion.  Returns [-1, 1].
pub fn volume_change_score(volumes: &[f64]) -> f64 {
    if volumes.len() < 2 {
        return 0.0;
    }
    let recent_n = volumes.len() / 3;
    let older_n = volumes.len() - recent_n;
    if older_n == 0 || recent_n == 0 {
        return 0.0;
    }
    let recent_avg: f64 = volumes[volumes.len() - recent_n..].iter().sum::<f64>() / recent_n as f64;
    let older_avg: f64 = volumes[..older_n].iter().sum::<f64>() / older_n as f64;
    if older_avg.abs() < 1e-9 {
        return 0.0;
    }
    let ratio = (recent_avg - older_avg) / older_avg;
    (ratio / 0.50).clamp(-1.0, 1.0)
}

/// Analyse momentum for a REIT given price and volume history.
pub fn momentum_analysis(ticker: &str, prices: &[f64], volumes: &[f64]) -> MomentumResult {
    let trend_score = price_trend_score(prices);
    let volume_score = volume_change_score(volumes);

    let combined = trend_score * 0.7 + volume_score * 0.3;

    let momentum_signal = if combined >= 0.3 {
        Signal::Buy
    } else if combined <= -0.3 {
        Signal::Sell
    } else {
        Signal::Hold
    };

    let mut parts: Vec<String> = Vec::new();
    if trend_score > 0.3 {
        parts.push("uptrend".into());
    } else if trend_score < -0.3 {
        parts.push("downtrend".into());
    }
    if volume_score > 0.3 {
        parts.push("rising volume".into());
    } else if volume_score < -0.3 {
        parts.push("declining volume".into());
    }
    let rationale = if parts.is_empty() {
        "neutral momentum".into()
    } else {
        parts.join(", ")
    };

    MomentumResult {
        ticker: ticker.into(),
        trend_score,
        volume_score,
        momentum_signal,
        rationale,
    }
}

// ---------------------------------------------------------------------------
// Sector Rotation Detection
// ---------------------------------------------------------------------------

/// Advice for a single sector based on relative performance.
#[derive(Debug, Clone)]
pub struct SectorRotationAdvice {
    pub sector: REITSector,
    pub recommendation: Signal,
    /// Suggested portfolio-weight adjustment (e.g. +0.10 = increase by 10 pp).
    pub weight_change: f64,
    pub rationale: String,
}

/// Detect sector-rotation signals by comparing each sector's recent return to
/// the cross-sector median.
///
/// Sectors outperforming the median by more than `threshold` receive a **Buy**
/// (increase allocation); those underperforming receive a **Sell** (decrease).
pub fn detect_sector_rotation(
    sector_returns: &HashMap<REITSector, f64>,
    current_weights: &HashMap<REITSector, f64>,
) -> Vec<SectorRotationAdvice> {
    let mut returns: Vec<f64> = sector_returns.values().copied().collect();
    returns.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    let median = if returns.len() >= 2 {
        let mid = returns.len() / 2;
        if returns.len() % 2 == 0 {
            (returns[mid - 1] + returns[mid]) / 2.0
        } else {
            returns[mid]
        }
    } else if !returns.is_empty() {
        returns[0]
    } else {
        0.0
    };

    let threshold = 0.02; // 2 percentage points

    let mut advice = Vec::new();
    for (sector, &ret) in sector_returns {
        let diff = ret - median;
        let current_weight = current_weights.get(sector).copied().unwrap_or(0.0);

        let (recommendation, weight_change, rationale) = if diff > threshold {
            let wc = (diff * 2.0).min(0.15);
            (
                Signal::Buy,
                wc,
                format!(
                    "{} outperforming median by {:.1}%, consider increasing allocation",
                    sector.label(),
                    diff * 100.0
                ),
            )
        } else if diff < -threshold {
            let wc = (diff * 2.0).max(-0.15);
            (
                Signal::Sell,
                wc,
                format!(
                    "{} underperforming median by {:.1}%, consider reducing allocation",
                    sector.label(),
                    (-diff) * 100.0
                ),
            )
        } else {
            (
                Signal::Hold,
                0.0,
                format!(
                    "{} near median performance, maintain allocation of {:.1}%",
                    sector.label(),
                    current_weight * 100.0
                ),
            )
        };

        advice.push(SectorRotationAdvice {
            sector: *sector,
            recommendation,
            weight_change,
            rationale,
        });
    }

    advice
}

// ---------------------------------------------------------------------------
// Composite Scoring
// ---------------------------------------------------------------------------

/// Configuration for the composite signal generator.
#[derive(Debug, Clone)]
pub struct CompositeConfig {
    /// Weight of value score (default 0.40).
    pub value_weight: f64,
    /// Weight of momentum score (default 0.30).
    pub momentum_weight: f64,
    /// Weight of health-check score (default 0.30).
    pub health_weight: f64,
    /// Composite score ≥ this triggers Buy (default 0.3).
    pub buy_threshold: f64,
    /// Composite score ≤ this triggers Sell (default -0.3).
    pub sell_threshold: f64,
}

impl Default for CompositeConfig {
    fn default() -> Self {
        Self {
            value_weight: 0.40,
            momentum_weight: 0.30,
            health_weight: 0.30,
            buy_threshold: 0.3,
            sell_threshold: -0.3,
        }
    }
}

/// Generate a composite [`SignalResult`] by blending value, momentum, and
/// fundamental-health scores.
///
/// `value_score` is typically the `overall_value_score` from [`value_screen`].
/// `momentum_score` is optional (pass `None` when no price history is available).
pub fn composite_signal(
    reit: &REIT,
    value_score: f64,
    momentum_score: Option<f64>,
    config: &CompositeConfig,
) -> SignalResult {
    // Normalise health_check from [0, 100] → [-1, 1].
    let health_score = reit
        .ratios
        .as_ref()
        .map(|r| health_check_score(r) / 50.0 - 1.0)
        .unwrap_or(0.0);

    let mom = momentum_score.unwrap_or(0.0);
    let composite = config.value_weight * value_score
        + config.momentum_weight * mom
        + config.health_weight * health_score;

    let signal = if composite >= config.buy_threshold {
        Signal::Buy
    } else if composite <= config.sell_threshold {
        Signal::Sell
    } else {
        Signal::Hold
    };

    let confidence = (composite.abs()).clamp(0.0, 1.0);

    let mut parts: Vec<String> = Vec::new();
    if value_score > 0.3 {
        parts.push("strong value".into());
    } else if value_score < -0.3 {
        parts.push("poor value".into());
    }
    if let Some(m) = momentum_score {
        if m > 0.3 {
            parts.push("positive momentum".into());
        } else if m < -0.3 {
            parts.push("negative momentum".into());
        }
    }
    if health_score > 0.3 {
        parts.push("healthy fundamentals".into());
    } else if health_score < -0.3 {
        parts.push("weak fundamentals".into());
    }
    let rationale = if parts.is_empty() {
        "mixed signals".into()
    } else {
        parts.join("; ")
    };

    SignalResult {
        ticker: reit.ticker.clone(),
        signal,
        confidence,
        composite_score: composite,
        rationale,
        generated_at: Utc::now(),
    }
}

// ===========================================================================
// Tests
// ===========================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use chrono::Utc;

    // ---- helpers -------------------------------------------------------

    fn make_ratios(
        p_ffo: f64,
        nav_ps: f64,
        div_yield: f64,
        d_ebitda: f64,
        ic: f64,
    ) -> FinancialRatios {
        FinancialRatios {
            computed_at: Utc::now(),
            ffo: 500.0,
            ffo_per_share: 5.0,
            affo_per_share: 4.5,
            nav_per_share: nav_ps,
            price_to_ffo: p_ffo,
            dividend_yield: div_yield,
            debt_to_ebitda: d_ebitda,
            interest_coverage: ic,
            cap_rate: 0.055,
            same_store_noi_growth: 0.03,
            current_ratio: 2.0,
            debt_to_equity: 0.8,
            return_on_equity: 0.08,
            operating_margin: 0.45,
        }
    }

    fn make_reit(ticker: &str, share_price: f64, ratios: Option<FinancialRatios>) -> REIT {
        REIT {
            ticker: ticker.into(),
            name: format!("{} Corp", ticker),
            cik: "0000000000".into(),
            sector: REITSector::Residential,
            inception_date: None,
            market_cap: share_price * 1_000_000_000.0,
            share_price,
            shares_outstanding: 1_000_000_000.0,
            balance_sheet: None,
            income_statement: None,
            ratios,
        }
    }

    // ---- nav_discount --------------------------------------------------

    #[test]
    fn test_nav_discount_positive() {
        // share = 70, NAV = 100 → discount = 30 %
        let d = nav_discount(70.0, 100.0);
        assert!((d - 0.30).abs() < 1e-9);
    }

    #[test]
    fn test_nav_discount_negative() {
        // share = 120, NAV = 100 → discount = -20 %
        let d = nav_discount(120.0, 100.0);
        assert!((d - (-0.20)).abs() < 1e-9);
    }

    #[test]
    fn test_nav_discount_zero_nav() {
        let d = nav_discount(50.0, 0.0);
        assert!((d - 0.0).abs() < 1e-9);
    }

    // ---- value_screen --------------------------------------------------

    #[test]
    fn test_value_screen_cheap_reit() {
        let ratios = make_ratios(10.0, 100.0, 0.05, 4.0, 4.5);
        let result = value_screen("CHEAP", 75.0, &ratios, &ValueScreenConfig::default());
        assert!(result.passes, "cheap REIT should pass value screen");
        assert!(result.overall_value_score > 0.0);
        assert!(result.p_ffo_score > 0.0, "P/FFO of 10 should score positively");
        assert!(result.nav_discount_score > 0.0, "25 % NAV discount should score positively");
        assert!(result.dividend_yield_score > 0.0);
    }

    #[test]
    fn test_value_screen_expensive_reit() {
        let ratios = make_ratios(22.0, 60.0, 0.01, 9.0, 1.2);
        let result = value_screen("EXPNS", 80.0, &ratios, &ValueScreenConfig::default());
        assert!(!result.passes, "expensive REIT should fail value screen");
        assert!(result.overall_value_score < 0.0);
    }

    // ---- momentum ------------------------------------------------------

    #[test]
    fn test_price_trend_score_uptrend() {
        let prices: Vec<f64> = (50..60).map(|i| i as f64).collect(); // 50 … 59
        let score = price_trend_score(&prices);
        assert!(score > 0.5, "upward price series should have positive trend");
    }

    #[test]
    fn test_price_trend_score_downtrend() {
        let prices: Vec<f64> = (50..60).rev().map(|i| i as f64).collect(); // 59 … 50
        let score = price_trend_score(&prices);
        assert!(score < -0.5, "downward price series should have negative trend");
    }

    #[test]
    fn test_price_trend_score_flat() {
        let prices = vec![100.0; 10];
        let score = price_trend_score(&prices);
        assert!((score - 0.0).abs() < 1e-9, "flat prices should yield zero trend");
    }

    #[test]
    fn test_momentum_analysis_uptrend() {
        let prices: Vec<f64> = (50..60).map(|i| i as f64).collect();
        let volumes: Vec<f64> = (100..110).map(|i| i as f64).collect();
        let result = momentum_analysis("UP", &prices, &volumes);
        assert_eq!(result.momentum_signal, Signal::Buy);
    }

    #[test]
    fn test_momentum_analysis_downtrend() {
        let prices: Vec<f64> = (50..60).rev().map(|i| i as f64).collect();
        let volumes: Vec<f64> = (200..210).map(|i| i as f64).collect();
        let result = momentum_analysis("DOWN", &prices, &volumes);
        assert_eq!(result.momentum_signal, Signal::Sell);
    }

    // ---- sector rotation -----------------------------------------------

    #[test]
    fn test_detect_sector_rotation_outperforming() {
        let mut returns: HashMap<REITSector, f64> = HashMap::new();
        returns.insert(REITSector::DataCenter, 0.08);
        returns.insert(REITSector::Residential, 0.02);
        returns.insert(REITSector::Office, -0.03);

        let mut weights: HashMap<REITSector, f64> = HashMap::new();
        weights.insert(REITSector::DataCenter, 0.20);
        weights.insert(REITSector::Residential, 0.40);
        weights.insert(REITSector::Office, 0.10);

        let advice = detect_sector_rotation(&returns, &weights);
        let dc = advice.iter().find(|a| a.sector == REITSector::DataCenter).unwrap();
        assert_eq!(dc.recommendation, Signal::Buy);
        assert!(dc.weight_change > 0.0);

        let office = advice.iter().find(|a| a.sector == REITSector::Office).unwrap();
        assert_eq!(office.recommendation, Signal::Sell);
    }

    #[test]
    fn test_detect_sector_rotation_balanced() {
        let mut returns: HashMap<REITSector, f64> = HashMap::new();
        returns.insert(REITSector::Industrial, 0.031);
        returns.insert(REITSector::Retail, 0.029);

        let weights: HashMap<REITSector, f64> = HashMap::new();
        let advice = detect_sector_rotation(&returns, &weights);

        for a in &advice {
            assert_eq!(a.recommendation, Signal::Hold, "near-median sectors should be Hold");
        }
    }

    // ---- composite -----------------------------------------------------

    #[test]
    fn test_composite_signal_buy() {
        let ratios = make_ratios(10.0, 100.0, 0.05, 3.0, 5.0);
        let reit = make_reit("GREAT", 75.0, Some(ratios));
        let result = composite_signal(&reit, 0.8, Some(0.6), &CompositeConfig::default());
        assert_eq!(result.signal, Signal::Buy);
        assert!(result.confidence > 0.3);
    }

    #[test]
    fn test_composite_signal_sell() {
        let ratios = make_ratios(25.0, 40.0, 0.01, 9.0, 1.0);
        let reit = make_reit("BAD", 100.0, Some(ratios));
        let result = composite_signal(&reit, -0.8, Some(-0.6), &CompositeConfig::default());
        assert_eq!(result.signal, Signal::Sell);
    }

    #[test]
    fn test_composite_signal_hold() {
        let ratios = make_ratios(15.0, 60.0, 0.02, 5.5, 2.5);
        let reit = make_reit("MEH", 80.0, Some(ratios));
        let result = composite_signal(&reit, 0.0, Some(0.0), &CompositeConfig::default());
        assert_eq!(result.signal, Signal::Hold);
    }
}
