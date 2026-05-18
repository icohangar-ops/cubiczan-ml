// scope-vantage/src/commodity.rs — Commodity price integration

use crate::types::CommodityCode;
use ndarray::{Array1, Array2};

/// A single commodity price observation.
#[derive(Debug, Clone)]
pub struct PriceObservation {
    pub commodity_code: String,
    pub date: u32,          // days since epoch (simplified)
    pub price: f64,
    pub volume: f64,
}

impl PriceObservation {
    pub fn new(commodity_code: &str, date: u32, price: f64, volume: f64) -> Self {
        Self {
            commodity_code: commodity_code.to_string(),
            date,
            price,
            volume,
        }
    }
}

/// Normalized price feed: sorted observations for a single commodity.
#[derive(Debug, Clone)]
pub struct PriceFeed {
    pub commodity_code: String,
    pub observations: Vec<PriceObservation>,
}

impl PriceFeed {
    pub fn new(commodity_code: &str) -> Self {
        Self {
            commodity_code: commodity_code.to_string(),
            observations: Vec::new(),
        }
    }

    /// Add an observation, maintaining chronological order.
    pub fn add(&mut self, obs: PriceObservation) {
        if let Some(last) = self.observations.last() {
            if obs.date < last.date {
                // Insert in sorted position
                let pos = self
                    .observations
                    .iter()
                    .position(|o| o.date > obs.date)
                    .unwrap_or(self.observations.len());
                self.observations.insert(pos, obs);
                return;
            }
        }
        self.observations.push(obs);
    }

    /// Extract price array (f64).
    pub fn prices(&self) -> Array1<f64> {
        Array1::from_iter(self.observations.iter().map(|o| o.price))
    }

    /// Compute simple returns.
    pub fn simple_returns(&self) -> Array1<f64> {
        let prices = self.prices();
        if prices.len() < 2 {
            return Array1::zeros(0);
        }
        let mut ret = Array1::zeros(prices.len() - 1);
        for i in 0..ret.len() {
            if prices[i] != 0.0 {
                ret[i] = (prices[i + 1] - prices[i]) / prices[i].abs();
            }
        }
        ret
    }

    /// Compute log returns.
    pub fn log_returns(&self) -> Array1<f64> {
        let prices = self.prices();
        if prices.len() < 2 {
            return Array1::zeros(0);
        }
        let mut ret = Array1::zeros(prices.len() - 1);
        for i in 0..ret.len() {
            if prices[i] > 0.0 && prices[i + 1] > 0.0 {
                ret[i] = (prices[i + 1] / prices[i]).ln();
            }
        }
        ret
    }

    /// Mean price.
    pub fn mean_price(&self) -> f64 {
        if self.observations.is_empty() {
            return 0.0;
        }
        self.prices().mean().unwrap_or(0.0)
    }

    /// Price standard deviation (volatility).
    pub fn price_std(&self) -> f64 {
        if self.observations.len() < 2 {
            return 0.0;
        }
        let prices = self.prices();
        let m = prices.mean().unwrap_or(0.0);
        let variance = prices.iter().map(|p| (p - m).powi(2)).sum::<f64>() / (prices.len() - 1) as f64;
        variance.sqrt()
    }

    /// Coefficient of variation (std / mean).
    pub fn cv(&self) -> f64 {
        let m = self.mean_price();
        if m == 0.0 {
            return 0.0;
        }
        self.price_std() / m.abs()
    }

    /// Detect seasonal pattern via autocorrelation at lag 12 (monthly data).
    /// Returns correlation coefficient in [-1, 1].
    pub fn seasonal_autocorr_lag12(&self) -> f64 {
        let prices = self.prices();
        if prices.len() < 13 {
            return 0.0;
        }
        let lag = 12;
        let n = prices.len() - lag;
        let m = prices.slice(ndarray::s![..n]).mean().unwrap_or(0.0);
        let mut num = 0.0;
        let mut den = 0.0;
        for i in 0..n {
            let diff0 = prices[i] - m;
            num += diff0 * (prices[i + lag] - m);
            den += diff0 * diff0;
        }
        if den == 0.0 {
            return 0.0;
        }
        num / den
    }
}

/// Commodity category based on HS2 chapter.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum CommodityCategory {
    Agriculture,
    Minerals,
    Energy,
    Manufactures,
    Chemicals,
    Unknown,
}

/// Map HS2 chapter to a category.
pub fn categorize_hs2(chapter: &str) -> CommodityCategory {
    let ch: u32 = chapter.trim().parse().unwrap_or(0);
    match ch {
        1..=24 => CommodityCategory::Agriculture,
        25..=26 => CommodityCategory::Minerals,
        27 => CommodityCategory::Energy,
        28..=38 => CommodityCategory::Chemicals,
        72..=83 => CommodityCategory::Minerals,
        84..=85 => CommodityCategory::Manufactures,
        86..=89 => CommodityCategory::Manufactures,
        90..=97 => CommodityCategory::Manufactures,
        71 => CommodityCategory::Minerals,
        _ => CommodityCategory::Unknown,
    }
}

/// Categorize a full CommodityCode.
pub fn categorize_commodity(code: &CommodityCode) -> CommodityCategory {
    categorize_hs2(code.chapter())
}

/// Build a price correlation matrix from multiple feeds.
/// Feeds are aligned by observation index (shortest length wins).
pub fn price_correlation_matrix(feeds: &[&PriceFeed]) -> Array2<f64> {
    let n = feeds.len();
    if n == 0 {
        return Array2::zeros((0, 0));
    }
    if n == 1 {
        return Array2::eye(1);
    }

    // Extract returns; filter out zero-length
    let returns: Vec<Array1<f64>> = feeds.iter().map(|f| f.simple_returns()).collect();
    let min_len = returns.iter().map(|r| r.len()).min().unwrap_or(0);
    if min_len < 2 {
        return Array2::eye(n);
    }

    let mut corr = Array2::zeros((n, n));
    for i in 0..n {
        for j in 0..n {
            let ri = &returns[i].slice(ndarray::s![..min_len]);
            let rj = &returns[j].slice(ndarray::s![..min_len]);
            let mi = ri.mean().unwrap_or(0.0);
            let mj = rj.mean().unwrap_or(0.0);
            let mut num = 0.0;
            let mut di = 0.0;
            let mut dj = 0.0;
            for k in 0..min_len {
                let a = ri[k] - mi;
                let b = rj[k] - mj;
                num += a * b;
                di += a * a;
                dj += b * b;
            }
            let den = (di * dj).sqrt();
            corr[[i, j]] = if den > 0.0 { num / den } else { 0.0 };
        }
    }
    corr
}

/// Supply/demand estimation via linear regression on price vs. volume.
/// Returns (slope, intercept) where demand_curve ≈ intercept + slope * price.
pub fn supply_demand_estimate(prices: &[f64], volumes: &[f64]) -> (f64, f64) {
    if prices.len() != volumes.len() || prices.len() < 2 {
        return (0.0, 0.0);
    }
    let n = prices.len() as f64;
    let sum_x: f64 = prices.iter().sum();
    let sum_y: f64 = volumes.iter().sum();
    let sum_xy: f64 = prices.iter().zip(volumes.iter()).map(|(x, y)| x * y).sum();
    let sum_xx: f64 = prices.iter().map(|x| x * x).sum();
    let denom = n * sum_xx - sum_x * sum_x;
    if denom == 0.0 {
        return (0.0, 0.0);
    }
    let slope = (n * sum_xy - sum_x * sum_y) / denom;
    let intercept = (sum_y - slope * sum_x) / n;
    (slope, intercept)
}

// ─── Tests ──────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn price_feed_add_sorted() {
        let mut feed = PriceFeed::new("270900");
        feed.add(PriceObservation::new("270900", 1, 80.0, 1000.0));
        feed.add(PriceObservation::new("270900", 2, 82.0, 1100.0));
        assert_eq!(feed.observations.len(), 2);
        assert!((feed.observations[1].price - 82.0).abs() < 1e-9);
    }

    #[test]
    fn price_feed_add_out_of_order() {
        let mut feed = PriceFeed::new("270900");
        feed.add(PriceObservation::new("270900", 5, 85.0, 1000.0));
        feed.add(PriceObservation::new("270900", 3, 80.0, 900.0));
        assert_eq!(feed.observations[0].date, 3);
        assert_eq!(feed.observations[1].date, 5);
    }

    #[test]
    fn price_feed_simple_returns() {
        let mut feed = PriceFeed::new("270900");
        feed.add(PriceObservation::new("270900", 1, 100.0, 100.0));
        feed.add(PriceObservation::new("270900", 2, 110.0, 100.0));
        let ret = feed.simple_returns();
        assert_eq!(ret.len(), 1);
        assert!((ret[0] - 0.1).abs() < 1e-9);
    }

    #[test]
    fn price_feed_log_returns() {
        let mut feed = PriceFeed::new("270900");
        feed.add(PriceObservation::new("270900", 1, 100.0, 100.0));
        feed.add(PriceObservation::new("270900", 2, 110.0, 100.0));
        let ret = feed.log_returns();
        assert_eq!(ret.len(), 1);
        assert!((ret[0] - (1.1_f64).ln()).abs() < 1e-9);
    }

    #[test]
    fn price_feed_mean() {
        let mut feed = PriceFeed::new("270900");
        feed.add(PriceObservation::new("270900", 1, 100.0, 100.0));
        feed.add(PriceObservation::new("270900", 2, 200.0, 100.0));
        assert!((feed.mean_price() - 150.0).abs() < 1e-9);
    }

    #[test]
    fn price_feed_std() {
        let mut feed = PriceFeed::new("270900");
        feed.add(PriceObservation::new("270900", 1, 100.0, 100.0));
        feed.add(PriceObservation::new("270900", 2, 200.0, 100.0));
        feed.add(PriceObservation::new("270900", 3, 300.0, 100.0));
        let std = feed.price_std();
        // std = sqrt((100^2 + 0^2 + 100^2)/2) = sqrt(10000) = 100
        assert!((std - 100.0).abs() < 1e-6);
    }

    #[test]
    fn price_feed_cv() {
        let mut feed = PriceFeed::new("270900");
        feed.add(PriceObservation::new("270900", 1, 100.0, 100.0));
        feed.add(PriceObservation::new("270900", 2, 200.0, 100.0));
        let cv = feed.cv();
        // sample std with n=2: sqrt(((100-150)^2 + (200-150)^2)/1) = sqrt(5000) ≈ 70.71
        // CV = 70.71 / 150 ≈ 0.4714
        assert!((cv - 0.4714).abs() < 0.01);
    }

    #[test]
    fn price_feed_empty() {
        let feed = PriceFeed::new("empty");
        assert_eq!(feed.mean_price(), 0.0);
        assert_eq!(feed.price_std(), 0.0);
    }

    #[test]
    fn categorize_hs2_agriculture() {
        assert_eq!(categorize_hs2("01"), CommodityCategory::Agriculture);
        assert_eq!(categorize_hs2("10"), CommodityCategory::Agriculture);
        assert_eq!(categorize_hs2("24"), CommodityCategory::Agriculture);
    }

    #[test]
    fn categorize_hs2_minerals() {
        assert_eq!(categorize_hs2("25"), CommodityCategory::Minerals);
        assert_eq!(categorize_hs2("26"), CommodityCategory::Minerals);
    }

    #[test]
    fn categorize_hs2_energy() {
        assert_eq!(categorize_hs2("27"), CommodityCategory::Energy);
    }

    #[test]
    fn categorize_hs2_chemicals() {
        assert_eq!(categorize_hs2("28"), CommodityCategory::Chemicals);
        assert_eq!(categorize_hs2("38"), CommodityCategory::Chemicals);
    }

    #[test]
    fn categorize_hs2_manufactures() {
        assert_eq!(categorize_hs2("84"), CommodityCategory::Manufactures);
        assert_eq!(categorize_hs2("87"), CommodityCategory::Manufactures);
    }

    #[test]
    fn categorize_hs2_unknown() {
        assert_eq!(categorize_hs2("99"), CommodityCategory::Unknown);
    }

    #[test]
    fn categorize_commodity_code() {
        let code = CommodityCode::new("270900").unwrap();
        assert_eq!(categorize_commodity(&code), CommodityCategory::Energy);
    }

    #[test]
    fn correlation_matrix_identity() {
        let mut feed = PriceFeed::new("270900");
        for i in 0..20 {
            feed.add(PriceObservation::new("270900", i, 50.0 + i as f64, 100.0));
        }
        let matrix = price_correlation_matrix(&[&feed]);
        assert!((matrix[[0, 0]] - 1.0).abs() < 1e-9);
    }

    #[test]
    fn correlation_matrix_two_feeds() {
        let mut feed1 = PriceFeed::new("270900");
        let mut feed2 = PriceFeed::new("870323");
        for i in 0..20 {
            let p1 = 50.0 + i as f64;
            let p2 = 100.0 + 2.0 * i as f64; // perfectly correlated
            feed1.add(PriceObservation::new("270900", i, p1, 100.0));
            feed2.add(PriceObservation::new("870323", i, p2, 100.0));
        }
        let matrix = price_correlation_matrix(&[&feed1, &feed2]);
        assert!((matrix[[0, 1]] - 1.0).abs() < 1e-6);
    }

    #[test]
    fn supply_demand_estimate_basic() {
        let prices = vec![100.0, 200.0, 300.0];
        let volumes = vec![1000.0, 800.0, 600.0];
        let (slope, intercept) = supply_demand_estimate(&prices, &volumes);
        assert!(slope < 0.0); // demand should slope down
        assert!(intercept > 0.0);
    }

    #[test]
    fn supply_demand_estimate_insufficient_data() {
        let (slope, intercept) = supply_demand_estimate(&[100.0], &[1000.0]);
        assert_eq!(slope, 0.0);
        assert_eq!(intercept, 0.0);
    }

    #[test]
    fn supply_demand_mismatched_lengths() {
        let (slope, intercept) = supply_demand_estimate(&[100.0, 200.0], &[1000.0]);
        assert_eq!(slope, 0.0);
        assert_eq!(intercept, 0.0);
    }

    #[test]
    fn seasonal_autocorr_lag12() {
        let mut feed = PriceFeed::new("270900");
        // Create a repeating pattern with period 12
        for cycle in 0..3 {
            for month in 0..12 {
                let price = 50.0 + (month as f64) * 5.0;
                feed.add(PriceObservation::new("270900", cycle * 12 + month, price, 100.0));
            }
        }
        let ac = feed.seasonal_autocorr_lag12();
        // Perfectly periodic => autocorr near 1
        assert!(ac > 0.9);
    }
}
