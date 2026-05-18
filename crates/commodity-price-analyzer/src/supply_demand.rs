//! # Supply/Demand Modeling
//!
//! Provides supply and demand analysis for commodity prices including
//! inventory tracking, production trend analysis, consumption demand proxy,
//! geopolitical risk scoring, and supply-demand balance scoring.

use crate::types::{CommodityType, PricePoint, SupplyDemandFactor};

// ---------------------------------------------------------------------------
// Supply-Demand Model
// ---------------------------------------------------------------------------

/// Model for analyzing supply and demand dynamics in commodity markets.
pub struct SupplyDemandModel {
    /// Window size for rolling calculations.
    pub rolling_window: usize,
    /// Threshold for deficit detection (fractional change).
    pub deficit_threshold: f64,
    /// Threshold for surplus detection (fractional change).
    pub surplus_threshold: f64,
}

impl Default for SupplyDemandModel {
    fn default() -> Self {
        SupplyDemandModel {
            rolling_window: 20,
            deficit_threshold: -0.03,
            surplus_threshold: 0.03,
        }
    }
}

impl SupplyDemandModel {
    /// Create a new supply-demand model with default settings.
    pub fn new() -> Self {
        Self::default()
    }

    // -----------------------------------------------------------------------
    // Inventory Tracking (proxy via price-volume relationship)
    // -----------------------------------------------------------------------

    /// Estimate inventory changes using price-volume relationship.
    /// Rising prices + falling volume suggest inventory drawdown (deficit).
    /// Falling prices + rising volume suggest inventory buildup (surplus).
    ///
    /// Returns a `SupplyDemandFactor` for inventory.
    pub fn inventory_analysis(&self, _commodity: CommodityType, prices: &[PricePoint]) -> SupplyDemandFactor {
        if prices.len() < 2 {
            return SupplyDemandFactor {
                factor_type: "inventory".into(),
                impact_score: 0.0,
                description: "Insufficient data for inventory analysis".into(),
                source: "price_volume_proxy".into(),
            };
        }

        let n = prices.len();
        let window = self.rolling_window.min(n);

        // Recent price change
        let price_change = (prices[n - 1].close - prices[n - window].close) / prices[n - window].close;

        // Recent volume change
        let recent_vol: f64 = prices[n - window..].iter().map(|p| p.volume).sum::<f64>();
        let prior_vol: f64 = if n >= window * 2 {
            prices[n - window * 2..n - window].iter().map(|p| p.volume).sum::<f64>()
        } else {
            recent_vol
        };

        let vol_change = if prior_vol.abs() < 1e-15 {
            0.0
        } else {
            (recent_vol - prior_vol) / prior_vol
        };

        // Deficit: price up + volume down → bullish (inventory drawdown)
        // Surplus: price down + volume up → bearish (inventory buildup)
        let impact = if price_change > 0.0 && vol_change < 0.0 {
            // Deficit scenario — bullish
            (price_change.abs() * 0.6 + vol_change.abs() * 0.4).min(1.0)
        } else if price_change < 0.0 && vol_change > 0.0 {
            // Surplus scenario — bearish
            -(price_change.abs() * 0.6 + vol_change.abs() * 0.4).min(1.0)
        } else {
            price_change * 2.0 // slight directional bias
        };

        let description = if impact > self.surplus_threshold {
            "Inventory deficit detected: rising prices with declining volume suggests tightening supply".into()
        } else if impact < self.deficit_threshold {
            "Inventory surplus detected: falling prices with rising volume suggests oversupply".into()
        } else {
            "Inventory levels appear balanced based on price-volume dynamics".into()
        };

        SupplyDemandFactor {
            factor_type: "inventory".into(),
            impact_score: impact.clamp(-1.0, 1.0),
            description,
            source: "price_volume_proxy".into(),
        }
    }

    // -----------------------------------------------------------------------
    // Production Rate Trend Analysis
    // -----------------------------------------------------------------------

    /// Estimate production rate trend from price momentum.
    /// Strong upward price momentum may indicate production constraints.
    /// Returns a `SupplyDemandFactor` for production.
    pub fn production_trend(&self, _commodity: CommodityType, prices: &[PricePoint]) -> SupplyDemandFactor {
        if prices.len() < self.rolling_window {
            return SupplyDemandFactor {
                factor_type: "production".into(),
                impact_score: 0.0,
                description: "Insufficient data for production trend analysis".into(),
                source: "price_momentum_proxy".into(),
            };
        }

        let n = prices.len();
        let window = self.rolling_window;

        // Compute rolling price momentum
        let recent_avg: f64 = prices[n - window..].iter().map(|p| p.close).sum::<f64>() / window as f64;
        let prior_avg: f64 = if n >= window * 2 {
            prices[n - window * 2..n - window].iter().map(|p| p.close).sum::<f64>() / window as f64
        } else {
            recent_avg
        };

        let momentum = if prior_avg.abs() < 1e-15 {
            0.0
        } else {
            (recent_avg - prior_avg) / prior_avg
        };

        // Strong positive momentum → possible supply constraint (bullish)
        // Strong negative momentum → possible demand destruction (bearish)
        let impact = (momentum * 5.0).clamp(-1.0, 1.0);

        let description = if impact > 0.3 {
            "Strong upward price momentum suggests potential production constraints".into()
        } else if impact < -0.3 {
            "Downward price momentum suggests possible demand destruction or oversupply".into()
        } else {
            "Price momentum is within normal range, no significant production signal".into()
        };

        SupplyDemandFactor {
            factor_type: "production".into(),
            impact_score: impact,
            description,
            source: "price_momentum_proxy".into(),
        }
    }

    // -----------------------------------------------------------------------
    // Consumption Demand Proxy
    // -----------------------------------------------------------------------

    /// Proxy for consumption demand from price momentum and volume trends.
    /// Rising prices + rising volume suggests strong demand.
    /// Returns a `SupplyDemandFactor` for consumption.
    pub fn consumption_demand(&self, _commodity: CommodityType, prices: &[PricePoint]) -> SupplyDemandFactor {
        if prices.len() < self.rolling_window * 2 {
            return SupplyDemandFactor {
                factor_type: "consumption".into(),
                impact_score: 0.0,
                description: "Insufficient data for consumption demand analysis".into(),
                source: "price_volume_proxy".into(),
            };
        }

        let n = prices.len();
        let window = self.rolling_window;

        let recent_price = prices[n - 1].close;
        let prior_price = prices[n - window].close;
        let price_change = (recent_price - prior_price) / prior_price;

        let recent_vol: f64 = prices[n - window..].iter().map(|p| p.volume).sum::<f64>();
        let prior_vol: f64 = prices[n - window * 2..n - window].iter().map(|p| p.volume).sum::<f64>();
        let vol_change = if prior_vol.abs() < 1e-15 { 0.0 } else { (recent_vol - prior_vol) / prior_vol };

        // Strong demand: price up + volume up
        // Weak demand: price down + volume down
        let impact = if price_change > 0.0 && vol_change > 0.0 {
            (price_change + vol_change).min(1.0) // bullish
        } else if price_change < 0.0 && vol_change < 0.0 {
            (price_change + vol_change).max(-1.0) // bearish
        } else {
            (price_change * 2.0).clamp(-1.0, 1.0)
        };

        let description = if impact > 0.2 {
            "Strong consumption demand indicated by rising prices and volume".into()
        } else if impact < -0.2 {
            "Weak consumption demand indicated by falling prices and volume".into()
        } else {
            "Consumption demand appears stable based on price-volume dynamics".into()
        };

        SupplyDemandFactor {
            factor_type: "consumption".into(),
            impact_score: impact,
            description,
            source: "price_volume_proxy".into(),
        }
    }

    // -----------------------------------------------------------------------
    // Geopolitical Risk Factor
    // -----------------------------------------------------------------------

    /// Estimate geopolitical risk from volatility spikes.
    /// Sudden increases in price volatility may indicate geopolitical events.
    /// Returns a `SupplyDemandFactor` for geopolitical risk.
    pub fn geopolitical_risk(&self, _commodity: CommodityType, prices: &[PricePoint]) -> SupplyDemandFactor {
        if prices.len() < self.rolling_window * 2 {
            return SupplyDemandFactor {
                factor_type: "geopolitical".into(),
                impact_score: 0.0,
                description: "Insufficient data for geopolitical risk assessment".into(),
                source: "volatility_proxy".into(),
            };
        }

        let n = prices.len();
        let window = self.rolling_window;

        // Recent volatility (std dev of returns)
        let recent_returns: Vec<f64> = (n - window..n)
            .map(|i| {
                if prices[i - 1].close.abs() < 1e-15 { 0.0 }
                else { (prices[i].close - prices[i - 1].close) / prices[i - 1].close }
            })
            .collect();

        let prior_returns: Vec<f64> = (n - window * 2..n - window)
            .map(|i| {
                if prices[i - 1].close.abs() < 1e-15 { 0.0 }
                else { (prices[i].close - prices[i - 1].close) / prices[i - 1].close }
            })
            .collect();

        let recent_vol = std_dev(&recent_returns);
        let prior_vol = std_dev(&prior_returns);

        // Volatility spike ratio
        let vol_ratio = if prior_vol.abs() < 1e-15 {
            if recent_vol.abs() > 1e-15 { 2.0 } else { 1.0 }
        } else {
            recent_vol / prior_vol
        };

        // High volatility spike → elevated geopolitical risk (bearish uncertainty)
        let impact = if vol_ratio > 1.5 {
            // Significant volatility spike → risk factor (slightly bearish for uncertainty)
            -((vol_ratio - 1.0) * 0.5).clamp(0.0, 1.0)
        } else {
            0.0
        };

        let description = if impact < -0.1 {
            format!(
                "Elevated geopolitical risk detected: recent volatility is {:.1}x the baseline ({:.1}x threshold)",
                vol_ratio, 1.5
            )
        } else {
            "Geopolitical risk appears normal based on volatility levels".into()
        };

        SupplyDemandFactor {
            factor_type: "geopolitical".into(),
            impact_score: impact,
            description,
            source: "volatility_proxy".into(),
        }
    }

    // -----------------------------------------------------------------------
    // Supply-Demand Balance Score
    // -----------------------------------------------------------------------

    /// Compute a composite supply-demand balance score in [-1, 1].
    /// Positive = demand exceeds supply (bullish), Negative = supply exceeds demand (bearish).
    pub fn balance_score(&self, commodity: CommodityType, prices: &[PricePoint]) -> f64 {
        let inventory = self.inventory_analysis(commodity, prices);
        let production = self.production_trend(commodity, prices);
        let consumption = self.consumption_demand(commodity, prices);
        let geo = self.geopolitical_risk(commodity, prices);

        // Weighted combination
        let score = inventory.impact_score * 0.30
            + production.impact_score * 0.25
            + consumption.impact_score * 0.25
            + geo.impact_score * 0.20;

        score.clamp(-1.0, 1.0)
    }

    /// Run all supply-demand analyses and return the full factor list.
    pub fn full_analysis(&self, commodity: CommodityType, prices: &[PricePoint]) -> Vec<SupplyDemandFactor> {
        vec![
            self.inventory_analysis(commodity, prices),
            self.production_trend(commodity, prices),
            self.consumption_demand(commodity, prices),
            self.geopolitical_risk(commodity, prices),
        ]
    }
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

/// Compute sample standard deviation.
fn std_dev(data: &[f64]) -> f64 {
    if data.len() < 2 {
        return 0.0;
    }
    let mean = data.iter().sum::<f64>() / data.len() as f64;
    let variance = data.iter().map(|x| (x - mean).powi(2)).sum::<f64>() / (data.len() as f64 - 1.0);
    variance.sqrt()
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use chrono::{Duration, Utc};

    fn make_rising_prices(n: usize) -> Vec<PricePoint> {
        let now = Utc::now();
        (0..n)
            .map(|i| {
                let price = 100.0 + i as f64 * 1.0;
                // Rising prices with declining volume (inventory drawdown pattern)
                let vol = 20000.0 - i as f64 * 100.0;
                PricePoint::new(
                    now - Duration::days((n - 1 - i) as i64),
                    price - 0.5,
                    price + 1.0,
                    price - 1.0,
                    price,
                    vol.max(1000.0),
                )
            })
            .collect()
    }

    fn make_falling_prices(n: usize) -> Vec<PricePoint> {
        let now = Utc::now();
        (0..n)
            .map(|i| {
                let price = 200.0 - i as f64 * 1.0;
                // Falling prices with rising volume (surplus pattern)
                let vol = 5000.0 + i as f64 * 200.0;
                PricePoint::new(
                    now - Duration::days((n - 1 - i) as i64),
                    price + 0.5,
                    price + 1.0,
                    price - 1.0,
                    price,
                    vol,
                )
            })
            .collect()
    }

    fn make_flat_prices(n: usize) -> Vec<PricePoint> {
        let now = Utc::now();
        (0..n)
            .map(|i| {
                PricePoint::new(
                    now - Duration::days((n - 1 - i) as i64),
                    100.0, 100.5, 99.5, 100.0, 10000.0,
                )
            })
            .collect()
    }

    fn make_volatile_prices(n: usize) -> Vec<PricePoint> {
        let now = Utc::now();
        (0..n)
            .map(|i| {
                // First half calm, second half volatile
                let price = if i < n / 2 {
                    100.0 + (i as f64 * 0.01).sin() * 0.5
                } else {
                    100.0 + (i as f64 * 0.5).sin() * 10.0
                };
                PricePoint::new(
                    now - Duration::days((n - 1 - i) as i64),
                    price - 1.0,
                    price + 2.0,
                    price - 2.0,
                    price,
                    10000.0,
                )
            })
            .collect()
    }

    #[test]
    fn test_inventory_analysis_deficit() {
        let model = SupplyDemandModel::new();
        let prices = make_rising_prices(50);
        let factor = model.inventory_analysis(CommodityType::Gold, &prices);
        assert_eq!(factor.factor_type, "inventory");
        // Rising prices + falling volume → deficit → positive impact
        assert!(factor.impact_score > 0.0);
    }

    #[test]
    fn test_inventory_analysis_surplus() {
        let model = SupplyDemandModel::new();
        let prices = make_falling_prices(50);
        let factor = model.inventory_analysis(CommodityType::Gold, &prices);
        // Falling prices + rising volume → surplus → negative impact
        assert!(factor.impact_score < 0.0);
    }

    #[test]
    fn test_inventory_analysis_insufficient_data() {
        let model = SupplyDemandModel::new();
        let prices = vec![PricePoint::new(Utc::now(), 100.0, 101.0, 99.0, 100.0, 1000.0)];
        let factor = model.inventory_analysis(CommodityType::Gold, &prices);
        assert!((factor.impact_score - 0.0).abs() < 1e-10);
    }

    #[test]
    fn test_production_trend_bullish() {
        let model = SupplyDemandModel::new();
        let prices = make_rising_prices(50);
        let factor = model.production_trend(CommodityType::Copper, &prices);
        assert_eq!(factor.factor_type, "production");
        assert!(factor.impact_score > 0.0);
    }

    #[test]
    fn test_production_trend_bearish() {
        let model = SupplyDemandModel::new();
        let prices = make_falling_prices(50);
        let factor = model.production_trend(CommodityType::Copper, &prices);
        assert!(factor.impact_score < 0.0);
    }

    #[test]
    fn test_production_trend_insufficient_data() {
        let model = SupplyDemandModel::new();
        let prices = make_flat_prices(5);
        let factor = model.production_trend(CommodityType::Copper, &prices);
        assert!((factor.impact_score - 0.0).abs() < 1e-10);
    }

    #[test]
    fn test_consumption_demand_strong() {
        let model = SupplyDemandModel::new();
        // Rising prices + rising volume → strong demand
        let now = Utc::now();
        let prices: Vec<PricePoint> = (0..50)
            .map(|i| {
                let price = 100.0 + i as f64 * 1.0;
                let vol = 5000.0 + i as f64 * 200.0;
                PricePoint::new(now - Duration::days((49 - i) as i64), price - 0.5, price + 1.0, price - 1.0, price, vol)
            })
            .collect();
        let factor = model.consumption_demand(CommodityType::Gold, &prices);
        assert_eq!(factor.factor_type, "consumption");
        assert!(factor.impact_score > 0.0);
    }

    #[test]
    fn test_consumption_demand_weak() {
        let model = SupplyDemandModel::new();
        let prices = make_falling_prices(50);
        let factor = model.consumption_demand(CommodityType::Gold, &prices);
        assert!(factor.impact_score < 0.0);
    }

    #[test]
    fn test_geopolitical_risk_normal() {
        let model = SupplyDemandModel::new();
        let prices = make_flat_prices(60);
        let factor = model.geopolitical_risk(CommodityType::Gold, &prices);
        assert_eq!(factor.factor_type, "geopolitical");
        // Flat prices → low risk
        assert!((factor.impact_score - 0.0).abs() < 1e-10);
    }

    #[test]
    fn test_geopolitical_risk_elevated() {
        let model = SupplyDemandModel::new();
        // First 40 points perfectly flat, last 20 very volatile
        let now = Utc::now();
        let prices: Vec<PricePoint> = (0..60)
            .map(|i| {
                let price = if i < 40 {
                    100.0 // perfectly flat
                } else {
                    100.0 + 20.0 * ((i - 40) as f64 * 0.8).sin() // very volatile
                };
                PricePoint::new(
                    now - Duration::days((59 - i) as i64),
                    price - 1.0,
                    price + 2.0,
                    price - 2.0,
                    price,
                    10000.0,
                )
            })
            .collect();
        let factor = model.geopolitical_risk(CommodityType::Gold, &prices);
        // Volatility spike in second half → elevated risk
        assert!(factor.impact_score < 0.0);
    }

    #[test]
    fn test_geopolitical_risk_insufficient_data() {
        let model = SupplyDemandModel::new();
        let prices = make_flat_prices(10);
        let factor = model.geopolitical_risk(CommodityType::Gold, &prices);
        assert!((factor.impact_score - 0.0).abs() < 1e-10);
    }

    #[test]
    fn test_balance_score_range() {
        let model = SupplyDemandModel::new();
        let prices = make_rising_prices(60);
        let score = model.balance_score(CommodityType::Gold, &prices);
        assert!(score >= -1.0 && score <= 1.0);
    }

    #[test]
    fn test_balance_score_bullish() {
        let model = SupplyDemandModel::new();
        let prices = make_rising_prices(60);
        let score = model.balance_score(CommodityType::Gold, &prices);
        assert!(score > 0.0);
    }

    #[test]
    fn test_full_analysis_count() {
        let model = SupplyDemandModel::new();
        let prices = make_rising_prices(60);
        let factors = model.full_analysis(CommodityType::Gold, &prices);
        assert_eq!(factors.len(), 4);
        let types: Vec<&str> = factors.iter().map(|f| f.factor_type.as_str()).collect();
        assert!(types.contains(&"inventory"));
        assert!(types.contains(&"production"));
        assert!(types.contains(&"consumption"));
        assert!(types.contains(&"geopolitical"));
    }

    #[test]
    fn test_supply_demand_model_default() {
        let model = SupplyDemandModel::default();
        assert_eq!(model.rolling_window, 20);
        assert!((model.deficit_threshold - (-0.03)).abs() < 1e-10);
        assert!((model.surplus_threshold - 0.03).abs() < 1e-10);
    }

    #[test]
    fn test_impact_score_clamped() {
        let model = SupplyDemandModel::new();
        // All factors should have impact in [-1, 1]
        let prices = make_volatile_prices(60);
        let factors = model.full_analysis(CommodityType::Gold, &prices);
        for f in &factors {
            assert!(f.impact_score >= -1.0 && f.impact_score <= 1.0);
        }
    }
}
