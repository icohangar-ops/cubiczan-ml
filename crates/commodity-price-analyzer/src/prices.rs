//! # Price Data Management
//!
//! Provides storage, retrieval, and computation on commodity price data.
//! Supports adding price points, range queries, resampling, return calculations,
//! VWAP computation, and mock data generation for testing.

use chrono::{Datelike, DateTime, Duration, Utc};
use std::collections::HashMap;

use crate::types::{CommodityType, PricePoint, ResamplePeriod};

// ---------------------------------------------------------------------------
// Price Database
// ---------------------------------------------------------------------------

/// In-memory database of commodity price data, keyed by commodity type.
pub struct PriceDatabase {
    data: HashMap<CommodityType, Vec<PricePoint>>,
}

impl PriceDatabase {
    /// Create a new empty price database.
    pub fn new() -> Self {
        PriceDatabase {
            data: HashMap::new(),
        }
    }

    /// Add a single price point for a commodity. Data is kept sorted by timestamp.
    pub fn add_price_point(&mut self, commodity: CommodityType, point: PricePoint) {
        let prices = self.data.entry(commodity).or_insert_with(Vec::new);
        // Insert in sorted order by timestamp
        let pos = prices
            .iter()
            .position(|p| p.timestamp >= point.timestamp)
            .unwrap_or(prices.len());
        prices.insert(pos, point);
    }

    /// Add multiple price points at once.
    pub fn add_price_points(&mut self, commodity: CommodityType, points: Vec<PricePoint>) {
        for point in points {
            self.add_price_point(commodity, point);
        }
    }

    /// Get all prices for a commodity.
    pub fn get_prices(&self, commodity: CommodityType) -> Option<&Vec<PricePoint>> {
        self.data.get(&commodity)
    }

    /// Get the latest price point for a commodity.
    pub fn get_latest(&self, commodity: CommodityType) -> Option<&PricePoint> {
        self.data.get(&commodity).and_then(|prices| prices.last())
    }

    /// Get prices within a time range [start, end].
    pub fn get_price_range(
        &self,
        commodity: CommodityType,
        start: DateTime<Utc>,
        end: DateTime<Utc>,
    ) -> Vec<&PricePoint> {
        self.data
            .get(&commodity)
            .map(|prices| {
                prices
                    .iter()
                    .filter(|p| p.timestamp >= start && p.timestamp <= end)
                    .collect()
            })
            .unwrap_or_default()
    }

    /// Resample price data to a coarser time period.
    pub fn resample(
        &self,
        commodity: CommodityType,
        period: ResamplePeriod,
    ) -> Result<Vec<PricePoint>, String> {
        let prices = self
            .data
            .get(&commodity)
            .ok_or_else(|| format!("No data for {:?}", commodity))?;

        if prices.is_empty() {
            return Ok(Vec::new());
        }

        let group_fn = |ts: DateTime<Utc>| -> DateTime<Utc> {
            match period {
                ResamplePeriod::Daily => ts.date_naive().and_hms_opt(0, 0, 0).unwrap().and_utc(),
                ResamplePeriod::Weekly => {
                    let weekday = ts.weekday().num_days_from_monday() as i64;
                    ts.date_naive()
                        .and_hms_opt(0, 0, 0)
                        .unwrap()
                        .and_utc()
                        - Duration::days(weekday)
                }
                ResamplePeriod::Monthly => {
                    ts.date_naive()
                        .with_day(1)
                        .unwrap()
                        .and_hms_opt(0, 0, 0)
                        .unwrap()
                        .and_utc()
                }
            }
        };

        let mut result: Vec<PricePoint> = Vec::new();
        let mut current_group = group_fn(prices[0].timestamp);
        let mut group_open = prices[0].open;
        let mut group_high = prices[0].high;
        let mut group_low = prices[0].low;
        let mut group_close = prices[0].close;
        let mut group_volume = prices[0].volume;

        for point in prices.iter().skip(1) {
            let point_group = group_fn(point.timestamp);
            if point_group == current_group {
                group_high = group_high.max(point.high);
                group_low = group_low.min(point.low);
                group_close = point.close;
                group_volume += point.volume;
            } else {
                result.push(PricePoint::new(
                    current_group,
                    group_open,
                    group_high,
                    group_low,
                    group_close,
                    group_volume,
                ));
                current_group = point_group;
                group_open = point.open;
                group_high = point.high;
                group_low = point.low;
                group_close = point.close;
                group_volume = point.volume;
            }
        }
        result.push(PricePoint::new(
            current_group,
            group_open,
            group_high,
            group_low,
            group_close,
            group_volume,
        ));

        Ok(result)
    }

    /// Compute log returns from close prices.
    pub fn log_returns(&self, commodity: CommodityType) -> Result<Vec<f64>, String> {
        let prices = self
            .data
            .get(&commodity)
            .ok_or_else(|| format!("No data for {:?}", commodity))?;
        if prices.len() < 2 {
            return Err("Need at least 2 price points".into());
        }
        let mut returns = Vec::with_capacity(prices.len() - 1);
        for i in 1..prices.len() {
            if prices[i - 1].close.abs() < 1e-15 {
                returns.push(0.0);
            } else {
                returns.push((prices[i].close / prices[i - 1].close).ln());
            }
        }
        Ok(returns)
    }

    /// Compute simple returns from close prices.
    pub fn simple_returns(&self, commodity: CommodityType) -> Result<Vec<f64>, String> {
        let prices = self
            .data
            .get(&commodity)
            .ok_or_else(|| format!("No data for {:?}", commodity))?;
        if prices.len() < 2 {
            return Err("Need at least 2 price points".into());
        }
        let mut returns = Vec::with_capacity(prices.len() - 1);
        for i in 1..prices.len() {
            if prices[i - 1].close.abs() < 1e-15 {
                returns.push(0.0);
            } else {
                returns.push((prices[i].close - prices[i - 1].close) / prices[i - 1].close);
            }
        }
        Ok(returns)
    }

    /// Compute volume-weighted average price (VWAP) over a range of data.
    pub fn vwap(&self, commodity: CommodityType) -> Result<f64, String> {
        let prices = self
            .data
            .get(&commodity)
            .ok_or_else(|| format!("No data for {:?}", commodity))?;
        if prices.is_empty() {
            return Err("No price data".into());
        }

        let total_typical_vol: f64 = prices
            .iter()
            .map(|p| p.typical_price() * p.volume)
            .sum();
        let total_volume: f64 = prices.iter().map(|p| p.volume).sum();

        if total_volume.abs() < 1e-15 {
            return Err("Total volume is zero".into());
        }

        Ok(total_typical_vol / total_volume)
    }

    /// Get the number of price points for a commodity.
    pub fn count(&self, commodity: CommodityType) -> usize {
        self.data.get(&commodity).map(|p| p.len()).unwrap_or(0)
    }

    /// Get the list of commodities that have data.
    pub fn commodities(&self) -> Vec<CommodityType> {
        let mut list: Vec<CommodityType> = self.data.keys().copied().collect();
        list.sort();
        list
    }

    /// Generate mock price data for a commodity.
    pub fn generate_mock_data(
        commodity: CommodityType,
        days: usize,
        start_price: Option<f64>,
    ) -> Vec<PricePoint> {
        let (lo, hi) = commodity.typical_price_range();
        let base = start_price.unwrap_or((lo + hi) / 2.0);
        let daily_vol = (hi - lo) / base * 0.02; // daily volatility estimate

        let mut prices = Vec::with_capacity(days);
        let now = Utc::now();
        let mut price = base;

        for i in 0..days {
            let ts = now - Duration::days((days - 1 - i) as i64);
            // Simple random walk (deterministic-ish with a sine wave + noise)
            let noise = (i as f64 * 0.13).sin() * daily_vol * price * 0.5
                + (i as f64 * 0.07).cos() * daily_vol * price * 0.3
                + (i as f64 * 0.31).sin() * daily_vol * price * 0.2;
            price += noise;
            price = price.max(lo * 0.8).min(hi * 1.2);

            let spread = price * daily_vol * 0.3;
            let high = price + spread.abs();
            let low = price - spread.abs();
            let open = (price + (high - low) * 0.3).min(high).max(low);
            let close = price;
            let volume = 10000.0 + (i as f64 * 137.0).sin().abs() * 50000.0;

            prices.push(PricePoint::new(ts, open, high, low, close, volume));
        }

        prices
    }

    /// Populate the database with mock data for all commodity types.
    pub fn populate_all_mock(&mut self, days: usize) {
        for commodity in CommodityType::all() {
            let mock = Self::generate_mock_data(commodity, days, None);
            self.add_price_points(commodity, mock);
        }
    }
}

impl Default for PriceDatabase {
    fn default() -> Self {
        Self::new()
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    fn make_simple_prices(commodity: CommodityType, count: usize) -> Vec<PricePoint> {
        let now = Utc::now();
        (0..count)
            .map(|i| {
                let price = 100.0 + i as f64;
                PricePoint::new(
                    now - Duration::days((count - 1 - i) as i64),
                    price - 1.0,
                    price + 2.0,
                    price - 2.0,
                    price,
                    1000.0 + i as f64 * 100.0,
                )
            })
            .collect()
    }

    #[test]
    fn test_add_and_get_prices() {
        let mut db = PriceDatabase::new();
        let prices = make_simple_prices(CommodityType::Gold, 5);
        db.add_price_points(CommodityType::Gold, prices);
        assert_eq!(db.count(CommodityType::Gold), 5);
        assert!(db.get_prices(CommodityType::Gold).is_some());
        assert!(db.get_prices(CommodityType::Silver).is_none());
    }

    #[test]
    fn test_get_latest() {
        let mut db = PriceDatabase::new();
        let prices = make_simple_prices(CommodityType::Gold, 5);
        db.add_price_points(CommodityType::Gold, prices);
        let latest = db.get_latest(CommodityType::Gold).unwrap();
        assert!((latest.close - 104.0).abs() < 1e-10);
    }

    #[test]
    fn test_get_latest_empty() {
        let db = PriceDatabase::new();
        assert!(db.get_latest(CommodityType::Gold).is_none());
    }

    #[test]
    fn test_get_price_range() {
        let mut db = PriceDatabase::new();
        let prices = make_simple_prices(CommodityType::Gold, 10);
        // Use the actual timestamps from prices to avoid Utc::now() race
        let start = prices[0].timestamp;
        let end = prices[4].timestamp;
        db.add_price_points(CommodityType::Gold, prices);

        let range = db.get_price_range(CommodityType::Gold, start, end);
        assert_eq!(range.len(), 5);
    }

    #[test]
    fn test_resample_weekly() {
        let mut db = PriceDatabase::new();
        let mock = PriceDatabase::generate_mock_data(CommodityType::Gold, 30, Some(2000.0));
        db.add_price_points(CommodityType::Gold, mock);
        let weekly = db.resample(CommodityType::Gold, ResamplePeriod::Weekly).unwrap();
        assert!(weekly.len() >= 4);
        assert!(weekly.len() <= 7); // 30 days ≈ 4-5 weeks
    }

    #[test]
    fn test_resample_monthly() {
        let mut db = PriceDatabase::new();
        let mock = PriceDatabase::generate_mock_data(CommodityType::Silver, 90, Some(25.0));
        db.add_price_points(CommodityType::Silver, mock);
        let monthly = db.resample(CommodityType::Silver, ResamplePeriod::Monthly).unwrap();
        assert!(monthly.len() >= 2);
        assert!(monthly.len() <= 4); // 90 days ≈ 3 months
    }

    #[test]
    fn test_log_returns() {
        let mut db = PriceDatabase::new();
        let prices = make_simple_prices(CommodityType::Gold, 5);
        db.add_price_points(CommodityType::Gold, prices);
        let log_ret = db.log_returns(CommodityType::Gold).unwrap();
        assert_eq!(log_ret.len(), 4);
        // First return: ln(101/100) ≈ 0.00995
        assert!((log_ret[0] - (101.0 / 100.0_f64).ln()).abs() < 1e-10);
    }

    #[test]
    fn test_simple_returns() {
        let mut db = PriceDatabase::new();
        let prices = make_simple_prices(CommodityType::Gold, 5);
        db.add_price_points(CommodityType::Gold, prices);
        let sim_ret = db.simple_returns(CommodityType::Gold).unwrap();
        assert_eq!(sim_ret.len(), 4);
        // First return: (101-100)/100 = 0.01
        assert!((sim_ret[0] - 0.01).abs() < 1e-10);
    }

    #[test]
    fn test_returns_insufficient_data() {
        let mut db = PriceDatabase::new();
        db.add_price_point(CommodityType::Gold, PricePoint::new(Utc::now(), 100.0, 101.0, 99.0, 100.0, 500.0));
        let result = db.log_returns(CommodityType::Gold);
        assert!(result.is_err());
    }

    #[test]
    fn test_vwap() {
        let mut db = PriceDatabase::new();
        let mock = PriceDatabase::generate_mock_data(CommodityType::Gold, 20, Some(2000.0));
        db.add_price_points(CommodityType::Gold, mock);
        let vwap = db.vwap(CommodityType::Gold).unwrap();
        assert!(vwap > 0.0);
    }

    #[test]
    fn test_vwap_empty() {
        let db = PriceDatabase::new();
        let result = db.vwap(CommodityType::Gold);
        assert!(result.is_err());
    }

    #[test]
    fn test_mock_data_generation() {
        let mock = PriceDatabase::generate_mock_data(CommodityType::Gold, 100, None);
        assert_eq!(mock.len(), 100);
        for pp in &mock {
            assert!(pp.is_valid());
        }
    }

    #[test]
    fn test_mock_data_with_start_price() {
        let mock = PriceDatabase::generate_mock_data(CommodityType::Gold, 50, Some(1800.0));
        assert!((mock[0].close - 1800.0).abs() < 50.0); // close to start price
    }

    #[test]
    fn test_populate_all_mock() {
        let mut db = PriceDatabase::new();
        db.populate_all_mock(100);
        assert_eq!(db.commodities().len(), 12);
        for ct in CommodityType::all() {
            assert!(db.count(ct) == 100);
            assert!(db.get_latest(ct).is_some());
        }
    }

    #[test]
    fn test_add_price_point_sorted_insert() {
        let mut db = PriceDatabase::new();
        let now = Utc::now();
        let p1 = PricePoint::new(now, 100.0, 101.0, 99.0, 100.0, 500.0);
        let p2 = PricePoint::new(now - Duration::days(1), 99.0, 100.0, 98.0, 99.0, 400.0);
        let p3 = PricePoint::new(now - Duration::days(2), 98.0, 99.0, 97.0, 98.0, 300.0);
        // Insert out of order
        db.add_price_point(CommodityType::Gold, p1);
        db.add_price_point(CommodityType::Gold, p3);
        db.add_price_point(CommodityType::Gold, p2);
        let prices = db.get_prices(CommodityType::Gold).unwrap();
        assert_eq!(prices.len(), 3);
        // Should be sorted by timestamp: p3, p2, p1
        assert!(prices[0].timestamp < prices[1].timestamp);
        assert!(prices[1].timestamp < prices[2].timestamp);
    }
}
