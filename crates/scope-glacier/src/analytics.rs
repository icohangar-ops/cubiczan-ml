//! Energy analytics: spot price analysis, spreads, volatility, supply/demand scoring, forward curves.

use crate::timeseries::{self, log_returns, mean, std_dev};
use crate::types::*;
use chrono::{Datelike, Utc};

/// Result of spot price analysis for a commodity.
#[derive(Debug, Clone)]
pub struct SpotPriceAnalysis {
    pub commodity: EnergyCommodity,
    pub current_price: f64,
    pub mean_price: f64,
    pub median_price: f64,
    pub std_deviation: f64,
    pub min_price: f64,
    pub max_price: f64,
    pub cv: f64,
    pub trend_direction: TrendDirection,
    pub z_score: f64,
    pub percentile_rank: f64,
    pub n_points: usize,
}

/// Trend direction enum.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TrendDirection {
    Rising,
    Falling,
    Flat,
    Volatile,
}

impl std::fmt::Display for TrendDirection {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            TrendDirection::Rising => write!(f, "RISING"),
            TrendDirection::Falling => write!(f, "FALLING"),
            TrendDirection::Flat => write!(f, "FLAT"),
            TrendDirection::Volatile => write!(f, "VOLATILE"),
        }
    }
}

/// Performs comprehensive spot price analysis on a series of price points.
pub fn analyze_spot_prices(prices: &[f64], commodity: EnergyCommodity) -> Result<SpotPriceAnalysis> {
    if prices.is_empty() {
        return Err(GlacierError::InsufficientData {
            required: 1,
            actual: 0,
        });
    }

    let current_price = prices[prices.len() - 1];
    let mean_price = mean(prices);
    let sd = std_dev(prices);
    let min_price = prices.iter().cloned().fold(f64::INFINITY, f64::min);
    let max_price = prices.iter().cloned().fold(f64::NEG_INFINITY, f64::max);

    let mut sorted = prices.to_vec();
    let median_price = timeseries::median(&mut sorted);

    let cv = if mean_price.abs() > f64::EPSILON {
        sd / mean_price.abs()
    } else {
        f64::INFINITY
    };

    let z_score = if sd > f64::EPSILON {
        (current_price - mean_price) / sd
    } else {
        0.0
    };

    // Percentile rank of current price
    let percentile_rank = sorted
        .iter()
        .filter(|&&p| p <= current_price)
        .count() as f64
        / sorted.len() as f64
        * 100.0;

    // Determine trend direction from recent changes
    let trend_direction = determine_trend(prices);

    Ok(SpotPriceAnalysis {
        commodity,
        current_price,
        mean_price,
        median_price,
        std_deviation: sd,
        min_price,
        max_price,
        cv,
        trend_direction,
        z_score,
        percentile_rank,
        n_points: prices.len(),
    })
}

/// Determines the trend direction from recent price data.
fn determine_trend(prices: &[f64]) -> TrendDirection {
    if prices.len() < 5 {
        return TrendDirection::Flat;
    }

    let lookback = prices.len().min(20);
    let recent = &prices[prices.len() - lookback..];
    let recent_diffs = timeseries::diff(recent);
    let positive_count = recent_diffs.iter().filter(|d| **d > 0.0).count();
    let _negative_count = recent_diffs.iter().filter(|d| **d < 0.0).count();
    let ratio = positive_count as f64 / recent_diffs.len() as f64;

    let cv = timeseries::coefficient_of_variation(recent);

    if cv > 0.15 {
        TrendDirection::Volatile
    } else if ratio > 0.65 {
        TrendDirection::Rising
    } else if ratio < 0.35 {
        TrendDirection::Falling
    } else {
        TrendDirection::Flat
    }
}

/// Computes the spread between two price series (aligned by index).
pub fn compute_spread(series_a: &[f64], series_b: &[f64]) -> Result<Vec<f64>> {
    let len = series_a.len().min(series_b.len());
    if len == 0 {
        return Err(GlacierError::InsufficientData {
            required: 1,
            actual: 0,
        });
    }

    Ok(series_a[..len]
        .iter()
        .zip(series_b[..len].iter())
        .map(|(a, b)| a - b)
        .collect())
}

/// Computes the ratio spread between two series.
pub fn compute_ratio_spread(series_a: &[f64], series_b: &[f64]) -> Result<Vec<f64>> {
    let len = series_a.len().min(series_b.len());
    if len == 0 {
        return Err(GlacierError::InsufficientData {
            required: 1,
            actual: 0,
        });
    }

    Ok(series_a[..len]
        .iter()
        .zip(series_b[..len].iter())
        .map(|(a, b)| {
            if b.abs() > f64::EPSILON {
                a / b - 1.0
            } else {
                0.0
            }
        })
        .collect())
}

/// Volatility surface point: (tenor, strike_proxy, implied_vol).
#[derive(Debug, Clone)]
pub struct VolatilityPoint {
    pub tenor_days: u32,
    pub moneyness: f64,
    pub volatility: f64,
}

/// Computes a simple volatility surface from price data across multiple windows.
pub fn compute_volatility_surface(
    prices: &[f64],
    windows: &[usize],
) -> Result<Vec<VolatilityPoint>> {
    if prices.len() < 3 {
        return Err(GlacierError::InsufficientData {
            required: 3,
            actual: prices.len(),
        });
    }

    let log_rets = log_returns(prices);
    if log_rets.is_empty() {
        return Ok(vec![]);
    }

    let mut surface = Vec::new();
    let current = prices[prices.len() - 1];
    let hist_mean = mean(prices);

    for &window in windows {
        if log_rets.len() < window {
            continue;
        }

        let recent_rets = &log_rets[log_rets.len() - window..];
        let vol = std_dev(recent_rets) * (252.0_f64).sqrt(); // Annualized

        let moneyness = if hist_mean.abs() > f64::EPSILON {
            current / hist_mean
        } else {
            1.0
        };

        surface.push(VolatilityPoint {
            tenor_days: window as u32,
            moneyness,
            volatility: vol,
        });
    }

    Ok(surface)
}

/// Computes the volatility term structure (volatility at different tenors).
pub fn volatility_term_structure(
    prices: &[f64],
    tenors: &[u32],
) -> Result<Vec<(u32, f64)>> {
    let log_rets = log_returns(prices);
    if log_rets.is_empty() {
        return Ok(vec![]);
    }

    let mut result = Vec::new();
    for &tenor in tenors {
        let window = tenor as usize;
        if log_rets.len() >= window {
            let recent = &log_rets[log_rets.len() - window..];
            let annualized_vol = std_dev(recent) * (252.0_f64).sqrt();
            result.push((tenor, annualized_vol));
        }
    }

    Ok(result)
}

/// Supply/demand imbalance score from -1.0 (severe deficit) to +1.0 (severe surplus).
#[derive(Debug, Clone)]
pub struct ImbalanceScore {
    pub commodity: EnergyCommodity,
    pub current_imbalance: f64,
    pub trend: TrendDirection,
    pub severity: ImbalanceSeverity,
    pub inventory_months: Option<f64>,
    pub recommendation: String,
}

/// Severity of supply/demand imbalance.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ImbalanceSeverity {
    SevereDeficit,
    ModerateDeficit,
    Balanced,
    ModerateSurplus,
    SevereSurplus,
}

/// Computes supply/demand imbalance scoring from a series of records.
pub fn score_imbalance(records: &[SupplyDemandRecord]) -> Result<ImbalanceScore> {
    if records.is_empty() {
        return Err(GlacierError::InsufficientData {
            required: 1,
            actual: 0,
        });
    }

    let commodity = records[records.len() - 1].commodity;

    // Compute average recent imbalance
    let recent_count = records.len().min(30);
    let recent = &records[records.len() - recent_count..];

    let mut imbalances = Vec::new();
    for rec in recent {
        if let Some(imb) = rec.imbalance() {
            imbalances.push(imb);
        }
    }

    if imbalances.is_empty() {
        return Err(GlacierError::InvalidInput(
            "No complete supply/demand records".into(),
        ));
    }

    let current = imbalances[imbalances.len() - 1];
    let _avg = mean(&imbalances);

    // Normalize to -1..1 range
    let score = (current / (1.0 + current.abs())).clamp(-1.0, 1.0);

    let severity = match score {
        s if s < -0.5 => ImbalanceSeverity::SevereDeficit,
        s if s < -0.15 => ImbalanceSeverity::ModerateDeficit,
        s if s <= 0.15 => ImbalanceSeverity::Balanced,
        s if s <= 0.5 => ImbalanceSeverity::ModerateSurplus,
        _ => ImbalanceSeverity::SevereSurplus,
    };

    let trend = if imbalances.len() >= 3 {
        let diffs = timeseries::diff(&imbalances);
        let positive = diffs.iter().filter(|d| **d > 0.0).count();
        let ratio = positive as f64 / diffs.len() as f64;
        if ratio > 0.6 {
            TrendDirection::Rising
        } else if ratio < 0.4 {
            TrendDirection::Falling
        } else {
            TrendDirection::Flat
        }
    } else {
        TrendDirection::Flat
    };

    let last = records.last().unwrap();
    let inventory_months = match (last.inventory_level, last.consumption_rate) {
        (Some(inv), Some(cons)) if cons > f64::EPSILON => Some(inv / cons),
        _ => None,
    };

    let recommendation = match severity {
        ImbalanceSeverity::SevereDeficit => "Strong bullish: supply crunch likely to drive prices higher".into(),
        ImbalanceSeverity::ModerateDeficit => "Mildly bullish: tightening market conditions".into(),
        ImbalanceSeverity::Balanced => "Neutral: market in rough equilibrium".into(),
        ImbalanceSeverity::ModerateSurplus => "Mildly bearish: ample supply pressuring prices".into(),
        ImbalanceSeverity::SevereSurplus => "Strong bearish: oversupply likely to depress prices".into(),
    };

    Ok(ImbalanceScore {
        commodity,
        current_imbalance: score,
        trend,
        severity,
        inventory_months,
        recommendation,
    })
}

/// Forward curve estimation point.
#[derive(Debug, Clone)]
pub struct ForwardCurvePoint {
    pub months_forward: u32,
    pub price: f64,
    pub contango: bool,
}

/// Estimates a simple forward curve from spot prices and seasonal patterns.
/// Uses linear interpolation of the current spot with seasonal adjustments.
pub fn estimate_forward_curve(
    spot_price: f64,
    seasonal_adjustments: &[f64], // Monthly adjustments indexed 0=Jan
    months: u32,
) -> Result<Vec<ForwardCurvePoint>> {
    if seasonal_adjustments.len() != 12 {
        return Err(GlacierError::InvalidInput(
            "Seasonal adjustments must have 12 values (one per month)".into(),
        ));
    }

    let now = Utc::now();
    let mut curve = Vec::with_capacity(months as usize);

    for m in 1..=months {
        let future_month = (now.month() as u32 - 1 + m) % 12;
        let adj = seasonal_adjustments[future_month as usize];
        let forward_price = spot_price * (1.0 + adj);

        curve.push(ForwardCurvePoint {
            months_forward: m,
            price: forward_price,
            contango: forward_price > spot_price,
        });
    }

    Ok(curve)
}

/// Generates market signals from spot price analysis.
pub fn generate_price_signals(analysis: &SpotPriceAnalysis) -> Vec<MarketSignal> {
    let mut signals = Vec::new();
    let now = Utc::now();

    // Z-score based signal
    if analysis.z_score > 2.0 {
        signals.push(MarketSignal::bullish(
            now,
            analysis.commodity,
            0.6,
            format!(
                "Price {} is {:.1} std dev above mean — potential overbought reversal",
                analysis.current_price, analysis.z_score
            ),
        ));
    } else if analysis.z_score < -2.0 {
        signals.push(MarketSignal::bearish(
            now,
            analysis.commodity,
            0.6,
            format!(
                "Price {} is {:.1} std dev below mean — potential oversold bounce",
                analysis.current_price, analysis.z_score.abs()
            ),
        ));
    }

    // Trend signal
    match analysis.trend_direction {
        TrendDirection::Rising => {
            signals.push(MarketSignal::bullish(
                now,
                analysis.commodity,
                0.4,
                "Sustained upward trend detected over recent window",
            ));
        }
        TrendDirection::Falling => {
            signals.push(MarketSignal::bearish(
                now,
                analysis.commodity,
                0.4,
                "Sustained downward trend detected over recent window",
            ));
        }
        _ => {}
    }

    if signals.is_empty() {
        signals.push(MarketSignal::neutral(
            now,
            analysis.commodity,
            "No significant signals detected",
        ));
    }

    signals
}

/// Computes the rolling Sharpe ratio of a returns series.
pub fn rolling_sharpe(returns: &[f64], window: usize, risk_free_rate: f64) -> Result<Vec<f64>> {
    if returns.len() < window {
        return Err(GlacierError::InsufficientData {
            required: window,
            actual: returns.len(),
        });
    }

    let annualization = 252.0_f64.sqrt();
    let daily_rf = risk_free_rate / 252.0;
    let mut result = Vec::with_capacity(returns.len() - window + 1);

    for i in 0..=(returns.len() - window) {
        let w = &returns[i..i + window];
        let excess_mean = mean(w) - daily_rf;
        let vol = std_dev(w);
        let sharpe = if vol > f64::EPSILON {
            (excess_mean / vol) * annualization
        } else {
            0.0
        };
        result.push(sharpe);
    }

    Ok(result)
}

/// Computes maximum drawdown from a price series.
pub fn max_drawdown(prices: &[f64]) -> (f64, usize, usize) {
    if prices.is_empty() {
        return (0.0, 0, 0);
    }

    let mut peak = prices[0];
    let mut max_dd = 0.0_f64;
    let mut dd_start = 0;
    let mut dd_end = 0;
    let mut current_start = 0;

    for (i, &price) in prices.iter().enumerate() {
        if price > peak {
            peak = price;
            current_start = i;
        }
        let dd = (peak - price) / peak;
        if dd > max_dd {
            max_dd = dd;
            dd_start = current_start;
            dd_end = i;
        }
    }

    (max_dd, dd_start, dd_end)
}

/// Computes the Herfindahl index for market concentration.
pub fn herfindahl_index(shares: &[f64]) -> f64 {
    let total: f64 = shares.iter().sum();
    if total.abs() < f64::EPSILON {
        return 0.0;
    }
    shares.iter().map(|s| (s / total).powi(2)).sum()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_analyze_spot_prices() {
        let prices: Vec<f64> = (0..50).map(|i| 70.0 + (i as f64) * 0.5).collect();
        let analysis = analyze_spot_prices(&prices, EnergyCommodity::CrudeOil).unwrap();
        assert!((analysis.current_price - 94.5).abs() < 1e-10);
        assert!(analysis.std_deviation > 0.0);
        assert!(analysis.min_price < analysis.max_price);
        assert_eq!(analysis.n_points, 50);
    }

    #[test]
    fn test_analyze_spot_prices_empty() {
        let result = analyze_spot_prices(&[], EnergyCommodity::CrudeOil);
        assert!(result.is_err());
    }

    #[test]
    fn test_determine_trend_rising() {
        let prices: Vec<f64> = (0..30).map(|i| 10.0 + i as f64 * 0.5).collect();
        let trend = determine_trend(&prices);
        assert_eq!(trend, TrendDirection::Rising);
    }

    #[test]
    fn test_determine_trend_falling() {
        let prices: Vec<f64> = (0..30).map(|i| 100.0 - i as f64 * 0.5).collect();
        let trend = determine_trend(&prices);
        assert_eq!(trend, TrendDirection::Falling);
    }

    #[test]
    fn test_compute_spread() {
        let a = vec![100.0, 105.0, 110.0];
        let b = vec![95.0, 100.0, 108.0];
        let spread = compute_spread(&a, &b).unwrap();
        assert_eq!(spread, vec![5.0, 5.0, 2.0]);
    }

    #[test]
    fn test_compute_ratio_spread() {
        let a = vec![110.0, 105.0];
        let b = vec![100.0, 100.0];
        let spread = compute_ratio_spread(&a, &b).unwrap();
        assert!((spread[0] - 0.1).abs() < 1e-10);
    }

    #[test]
    fn test_volatility_surface() {
        let prices: Vec<f64> = (0..100)
            .map(|i| 100.0 + (i as f64) * 0.1 + (i as f64 * 0.2).sin() * 2.0)
            .collect();
        let surface = compute_volatility_surface(&prices, &[5, 10, 20, 50]).unwrap();
        assert!(!surface.is_empty());
        // Longer window volatility should generally be different from shorter
        assert_ne!(surface[0].volatility, surface[surface.len() - 1].volatility);
    }

    #[test]
    fn test_volatility_surface_insufficient() {
        let prices = vec![1.0, 2.0];
        let result = compute_volatility_surface(&prices, &[5]);
        assert!(result.is_err());
    }

    #[test]
    fn test_volatility_term_structure() {
        let prices: Vec<f64> = (0..100).map(|i| 100.0 + (i as f64 * 0.5).sin()).collect();
        let ts = volatility_term_structure(&prices, &[5, 10, 20]).unwrap();
        assert_eq!(ts.len(), 3);
        for (_, vol) in &ts {
            assert!(*vol >= 0.0);
        }
    }

    #[test]
    fn test_score_imbalance_balanced() {
        let now = Utc::now();
        let records: Vec<SupplyDemandRecord> = (0..10)
            .map(|i| {
                let mut rec = SupplyDemandRecord::new(now - chrono::Duration::days(i), EnergyCommodity::NaturalGas);
                rec.supply_mmbtu = Some(100.0);
                rec.demand_mmbtu = Some(100.0);
                rec.inventory_level = Some(500.0);
                rec.consumption_rate = Some(50.0);
                rec
            })
            .collect();
        let score = score_imbalance(&records).unwrap();
        assert_eq!(score.severity, ImbalanceSeverity::Balanced);
        assert!((score.inventory_months.unwrap() - 10.0).abs() < 1e-10);
    }

    #[test]
    fn test_score_imbalance_surplus() {
        let now = Utc::now();
        let records: Vec<SupplyDemandRecord> = (0..10)
            .map(|i| {
                let mut rec = SupplyDemandRecord::new(now - chrono::Duration::days(i), EnergyCommodity::NaturalGas);
                rec.supply_mmbtu = Some(150.0);
                rec.demand_mmbtu = Some(100.0);
                rec
            })
            .collect();
        let score = score_imbalance(&records).unwrap();
        assert!(matches!(
            score.severity,
            ImbalanceSeverity::ModerateSurplus | ImbalanceSeverity::SevereSurplus
        ));
    }

    #[test]
    fn test_score_imbalance_empty() {
        let result = score_imbalance(&[]);
        assert!(result.is_err());
    }

    #[test]
    fn test_estimate_forward_curve() {
        // Seasonal adjustments: higher in winter (months 11, 0, 1), lower in spring/fall
        let adjustments = vec![0.05, 0.03, 0.0, -0.02, -0.03, -0.04, -0.03, -0.02, -0.01, 0.0, 0.02, 0.04];
        let curve = estimate_forward_curve(3.50, &adjustments, 12).unwrap();
        assert_eq!(curve.len(), 12);
        for pt in &curve {
            assert!(pt.price > 0.0);
        }
    }

    #[test]
    fn test_estimate_forward_curve_invalid_adjustments() {
        let result = estimate_forward_curve(3.50, &[0.0; 11], 12);
        assert!(result.is_err());
    }

    #[test]
    fn test_generate_price_signals_extreme_zscore() {
        // Create analysis with extreme z-score
        let prices: Vec<f64> = (0..20).map(|_| 70.0).collect();
        let mut extreme_prices = prices.clone();
        extreme_prices.push(100.0); // Extreme outlier

        let analysis = analyze_spot_prices(&extreme_prices, EnergyCommodity::CrudeOil).unwrap();
        let signals = generate_price_signals(&analysis);
        assert!(!signals.is_empty());
    }

    #[test]
    fn test_rolling_sharpe() {
        let returns: Vec<f64> = (0..100).map(|i| 0.001 + (i as f64 * 0.01).sin() * 0.01).collect();
        let sharpe = rolling_sharpe(&returns, 20, 0.02).unwrap();
        assert!(!sharpe.is_empty());
    }

    #[test]
    fn test_max_drawdown() {
        let prices = vec![100.0, 110.0, 105.0, 95.0, 90.0, 100.0, 120.0];
        let (dd, start, end) = max_drawdown(&prices);
        assert!(dd > 0.0);
        assert!(start < end);
    }

    #[test]
    fn test_max_drawdown_empty() {
        let (dd, _, _) = max_drawdown(&[]);
        assert!((dd - 0.0).abs() < 1e-10);
    }

    #[test]
    fn test_herfindahl_index() {
        let equal_shares = vec![25.0, 25.0, 25.0, 25.0];
        let hhi = herfindahl_index(&equal_shares);
        assert!((hhi - 0.25).abs() < 1e-10); // 4 * 0.25^2 = 0.25

        let concentrated = vec![90.0, 5.0, 3.0, 2.0];
        let hhi_c = herfindahl_index(&concentrated);
        assert!(hhi_c > hhi);
    }

    #[test]
    fn test_herfindahl_index_empty() {
        let hhi = herfindahl_index(&[]);
        assert!((hhi - 0.0).abs() < 1e-10);
    }
}
