//! In-memory time-series storage with sliding windows, OHLCV aggregation,
//! and pattern fingerprinting for graph queries.

use crate::types::{Chain, OhlcvCandle, TimeSeriesPoint};
use chrono::{DateTime, Duration, Utc};
use std::collections::HashMap;

// ---------------------------------------------------------------------------
// TimeSeriesStore
// ---------------------------------------------------------------------------

/// An in-memory time-series database keyed by (chain, series_name).
pub struct TimeSeriesStore {
    /// Maximum number of points per series before oldest entries are evicted.
    window_size: usize,
    /// The stored series: key → ordered list of points.
    series: HashMap<StoreKey, Vec<TimeSeriesPoint>>,
}

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
struct StoreKey {
    chain: Chain,
    name: String,
}

impl TimeSeriesStore {
    /// Create a new store with the given sliding-window capacity.
    pub fn new(window_size: usize) -> Self {
        Self {
            window_size,
            series: HashMap::new(),
        }
    }

    /// Insert a data point. If the series exceeds `window_size`, oldest entries are removed.
    pub fn insert(&mut self, chain: Chain, name: &str, point: TimeSeriesPoint) {
        let key = StoreKey {
            chain,
            name: name.to_string(),
        };
        let buf = self.series.entry(key).or_default();
        buf.push(point);
        if buf.len() > self.window_size {
            let drain = buf.len() - self.window_size;
            buf.drain(0..drain);
        }
    }

    /// Retrieve all points for a given chain + series name.
    pub fn get(&self, chain: Chain, name: &str) -> &[TimeSeriesPoint] {
        let key = StoreKey {
            chain,
            name: name.to_string(),
        };
        self.series.get(&key).map(|v| v.as_slice()).unwrap_or(&[])
    }

    /// Return only the values from a series as a `Vec<f64>`.
    pub fn values(&self, chain: Chain, name: &str) -> Vec<f64> {
        self.get(chain, name).iter().map(|p| p.value).collect()
    }

    /// Aggregate a series into OHLCV candles of `interval_seconds` length.
    pub fn ohlcv(
        &self,
        chain: Chain,
        name: &str,
        interval_seconds: i64,
    ) -> Vec<OhlcvCandle> {
        let points = self.get(chain, name);
        if points.is_empty() {
            return Vec::new();
        }

        // Bucket by interval
        let start_ts = points[0].timestamp;
        let interval = Duration::seconds(interval_seconds);

        // Collect into time buckets
        let mut buckets: Vec<Vec<&TimeSeriesPoint>> = Vec::new();
        let mut current_start = start_ts;
        let mut current: Vec<&TimeSeriesPoint> = Vec::new();

        for pt in points {
            while pt.timestamp >= current_start + interval {
                buckets.push(std::mem::take(&mut current));
                current_start += interval;
            }
            current.push(pt);
        }
        if !current.is_empty() {
            buckets.push(current);
        }

        buckets
            .into_iter()
            .filter_map(|bucket| {
                if bucket.is_empty() {
                    return None;
                }
                let open = bucket.first()?.value;
                let close = bucket.last()?.value;
                let high = bucket.iter().map(|p| p.value).fold(f64::NEG_INFINITY, f64::max);
                let low = bucket.iter().map(|p| p.value).fold(f64::INFINITY, f64::min);
                let volume: f64 = bucket.iter().map(|p| p.volume).sum();
                Some(OhlcvCandle {
                    open,
                    high,
                    low,
                    close,
                    volume,
                    timestamp_start: bucket[0].timestamp,
                    timestamp_end: bucket[bucket.len() - 1].timestamp,
                })
            })
            .collect()
    }

    /// Return the number of stored series.
    pub fn series_count(&self) -> usize {
        self.series.len()
    }

    /// Return the total number of points across all series.
    pub fn total_points(&self) -> usize {
        self.series.values().map(|v| v.len()).sum()
    }

    /// Clear all data.
    pub fn clear(&mut self) {
        self.series.clear();
    }

    /// Return the earliest timestamp in a series, if any.
    pub fn earliest(&self, chain: Chain, name: &str) -> Option<DateTime<Utc>> {
        self.get(chain, name).first().map(|p| p.timestamp)
    }

    /// Return the latest timestamp in a series, if any.
    pub fn latest(&self, chain: Chain, name: &str) -> Option<DateTime<Utc>> {
        self.get(chain, name).last().map(|p| p.timestamp)
    }

    /// Trim a series to only points after `after`.
    pub fn trim_before(&mut self, chain: Chain, name: &str, after: DateTime<Utc>) {
        let key = StoreKey {
            chain,
            name: name.to_string(),
        };
        if let Some(buf) = self.series.get_mut(&key) {
            let cut = buf.iter().position(|p| p.timestamp >= after).unwrap_or(buf.len());
            buf.drain(0..cut);
        }
    }
}

// ---------------------------------------------------------------------------
// PatternStore
// ---------------------------------------------------------------------------

/// Stores anomaly patterns as simple fingerprint → metadata for graph-like queries.
pub struct PatternStore {
    patterns: HashMap<String, PatternRecord>,
}

#[derive(Debug, Clone)]
pub struct PatternRecord {
    pub fingerprint: String,
    pub chain: Chain,
    pub anomaly_value: f64,
    pub count: u64,
    pub first_seen: DateTime<Utc>,
    pub last_seen: DateTime<Utc>,
}

impl PatternStore {
    pub fn new() -> Self {
        Self {
            patterns: HashMap::new(),
        }
    }

    /// Record a pattern. If it already exists, increment the count and update last_seen.
    pub fn record(
        &mut self,
        fingerprint: &str,
        chain: Chain,
        anomaly_value: f64,
        now: DateTime<Utc>,
    ) {
        let entry = self.patterns.entry(fingerprint.to_string());
        entry
            .and_modify(|r| {
                r.count += 1;
                r.last_seen = now;
            })
            .or_insert_with(|| PatternRecord {
                fingerprint: fingerprint.to_string(),
                chain,
                anomaly_value,
                count: 1,
                first_seen: now,
                last_seen: now,
            });
    }

    /// Look up a pattern by fingerprint.
    pub fn get(&self, fingerprint: &str) -> Option<&PatternRecord> {
        self.patterns.get(fingerprint)
    }

    /// Return all patterns for a given chain.
    pub fn by_chain(&self, chain: Chain) -> Vec<&PatternRecord> {
        self.patterns
            .values()
            .filter(|r| r.chain == chain)
            .collect()
    }

    /// Return the total number of stored patterns.
    pub fn len(&self) -> usize {
        self.patterns.len()
    }

    /// Return true if no patterns stored.
    pub fn is_empty(&self) -> bool {
        self.patterns.is_empty()
    }

    /// Return patterns seen more than `threshold` times.
    pub fn frequent_patterns(&self, threshold: u64) -> Vec<&PatternRecord> {
        self.patterns
            .values()
            .filter(|r| r.count > threshold)
            .collect()
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    fn ts(seconds: i64) -> DateTime<Utc> {
        Utc::now() + Duration::seconds(seconds)
    }

    fn ts_fixed(seconds: i64) -> DateTime<Utc> {
        DateTime::parse_from_rfc3339("2025-01-01T00:00:00Z")
            .unwrap()
            .with_timezone(&Utc)
            + Duration::seconds(seconds)
    }

    fn point(seconds: i64, value: f64, volume: f64) -> TimeSeriesPoint {
        TimeSeriesPoint {
            timestamp: ts(seconds),
            value,
            volume,
        }
    }

    #[test]
    fn test_store_insert_and_get() {
        let mut store = TimeSeriesStore::new(100);
        store.insert(
            Chain::Ethereum,
            "tx_value",
            point(0, 10.0, 1.0),
        );
        store.insert(
            Chain::Ethereum,
            "tx_value",
            point(1, 20.0, 2.0),
        );
        let pts = store.get(Chain::Ethereum, "tx_value");
        assert_eq!(pts.len(), 2);
        assert!((pts[1].value - 20.0).abs() < f64::EPSILON);
    }

    #[test]
    fn test_store_sliding_window_eviction() {
        let mut store = TimeSeriesStore::new(3);
        for i in 0..5 {
            store.insert(Chain::Ethereum, "x", point(i as i64, i as f64, 0.0));
        }
        let pts = store.get(Chain::Ethereum, "x");
        assert_eq!(pts.len(), 3);
        // Oldest (0, 1) should be evicted; (2, 3, 4) remain
        assert!((pts[0].value - 2.0).abs() < f64::EPSILON);
        assert!((pts[2].value - 4.0).abs() < f64::EPSILON);
    }

    #[test]
    fn test_store_values() {
        let mut store = TimeSeriesStore::new(100);
        store.insert(Chain::Ethereum, "v", point(0, 5.0, 0.0));
        store.insert(Chain::Ethereum, "v", point(1, 15.0, 0.0));
        let vals = store.values(Chain::Ethereum, "v");
        assert_eq!(vals, vec![5.0, 15.0]);
    }

    #[test]
    fn test_store_empty_series() {
        let store = TimeSeriesStore::new(100);
        assert!(store.get(Chain::Ethereum, "nonexistent").is_empty());
        assert!(store.values(Chain::Ethereum, "nonexistent").is_empty());
    }

    #[test]
    fn test_ohlcv_single_point() {
        let mut store = TimeSeriesStore::new(100);
        store.insert(Chain::Ethereum, "price", point(0, 100.0, 50.0));
        let candles = store.ohlcv(Chain::Ethereum, "price", 60);
        assert_eq!(candles.len(), 1);
        let c = &candles[0];
        assert!((c.open - 100.0).abs() < f64::EPSILON);
        assert!((c.high - 100.0).abs() < f64::EPSILON);
        assert!((c.low - 100.0).abs() < f64::EPSILON);
        assert!((c.close - 100.0).abs() < f64::EPSILON);
        assert!((c.volume - 50.0).abs() < f64::EPSILON);
    }

    #[test]
    fn test_ohlcv_multiple_intervals() {
        let mut store = TimeSeriesStore::new(100);
        store.insert(Chain::Ethereum, "price", point(0, 10.0, 1.0));
        store.insert(Chain::Ethereum, "price", point(30, 20.0, 2.0));
        store.insert(Chain::Ethereum, "price", point(61, 30.0, 3.0));
        let candles = store.ohlcv(Chain::Ethereum, "price", 60);
        assert_eq!(candles.len(), 2);
        assert!((candles[0].close - 20.0).abs() < f64::EPSILON);
        assert!((candles[1].open - 30.0).abs() < f64::EPSILON);
        assert!((candles[0].volume - 3.0).abs() < f64::EPSILON);
    }

    #[test]
    fn test_earliest_latest() {
        let mut store = TimeSeriesStore::new(100);
        assert!(store.earliest(Chain::Ethereum, "x").is_none());
        store.insert(Chain::Ethereum, "x", TimeSeriesPoint { timestamp: ts_fixed(10), value: 1.0, volume: 0.0 });
        store.insert(Chain::Ethereum, "x", TimeSeriesPoint { timestamp: ts_fixed(20), value: 2.0, volume: 0.0 });
        let earliest = store.earliest(Chain::Ethereum, "x").unwrap();
        let latest = store.latest(Chain::Ethereum, "x").unwrap();
        assert_eq!(earliest, ts_fixed(10));
        assert_eq!(latest, ts_fixed(20));
    }

    #[test]
    fn test_trim_before() {
        let mut store = TimeSeriesStore::new(100);
        store.insert(Chain::Ethereum, "x", TimeSeriesPoint { timestamp: ts_fixed(0), value: 1.0, volume: 0.0 });
        store.insert(Chain::Ethereum, "x", TimeSeriesPoint { timestamp: ts_fixed(10), value: 2.0, volume: 0.0 });
        store.insert(Chain::Ethereum, "x", TimeSeriesPoint { timestamp: ts_fixed(20), value: 3.0, volume: 0.0 });
        store.trim_before(Chain::Ethereum, "x", ts_fixed(10));
        let pts = store.get(Chain::Ethereum, "x");
        assert_eq!(pts.len(), 2);
        assert!((pts[0].value - 2.0).abs() < f64::EPSILON);
    }

    #[test]
    fn test_series_count_and_total() {
        let mut store = TimeSeriesStore::new(100);
        store.insert(Chain::Ethereum, "a", point(0, 1.0, 0.0));
        store.insert(Chain::Solana, "b", point(0, 1.0, 0.0));
        store.insert(Chain::Solana, "b", point(1, 1.0, 0.0));
        assert_eq!(store.series_count(), 2);
        assert_eq!(store.total_points(), 3);
    }

    #[test]
    fn test_pattern_store_record_and_get() {
        let mut ps = PatternStore::new();
        let now = Utc::now();
        ps.record("fp1", Chain::Ethereum, 0.9, now);
        let r = ps.get("fp1").unwrap();
        assert_eq!(r.count, 1);
        assert_eq!(r.chain, Chain::Ethereum);
    }

    #[test]
    fn test_pattern_store_increment() {
        let mut ps = PatternStore::new();
        let now = Utc::now();
        ps.record("fp2", Chain::Solana, 0.8, now);
        ps.record("fp2", Chain::Solana, 0.8, now + Duration::seconds(1));
        let r = ps.get("fp2").unwrap();
        assert_eq!(r.count, 2);
    }

    #[test]
    fn test_pattern_store_by_chain() {
        let mut ps = PatternStore::new();
        let now = Utc::now();
        ps.record("a", Chain::Ethereum, 0.5, now);
        ps.record("b", Chain::Solana, 0.6, now);
        let eth = ps.by_chain(Chain::Ethereum);
        assert_eq!(eth.len(), 1);
        assert_eq!(eth[0].fingerprint, "a");
    }

    #[test]
    fn test_pattern_store_frequent() {
        let mut ps = PatternStore::new();
        let now = Utc::now();
        for _ in 0..5 {
            ps.record("freq", Chain::Ethereum, 0.7, now);
        }
        ps.record("rare", Chain::Ethereum, 0.3, now);
        let freq = ps.frequent_patterns(3);
        assert_eq!(freq.len(), 1);
        assert_eq!(freq[0].fingerprint, "freq");
    }

    #[test]
    fn test_pattern_store_len_empty() {
        let ps = PatternStore::new();
        assert!(ps.is_empty());
        assert_eq!(ps.len(), 0);
    }

    #[test]
    fn test_store_clear() {
        let mut store = TimeSeriesStore::new(100);
        store.insert(Chain::Ethereum, "x", point(0, 1.0, 0.0));
        store.clear();
        assert_eq!(store.total_points(), 0);
        assert_eq!(store.series_count(), 0);
    }
}
