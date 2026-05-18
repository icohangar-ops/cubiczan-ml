//! End-to-end pipeline: ingest → detect → store → alert.
//!
//! This module ties together all the subsystems into a single `Pipeline` struct
//! that accepts raw transaction data, normalises it, runs anomaly detection,
//! persists results to the time-series store, and dispatches alerts.

use crate::alert::{AlertConfig, AlertRouter};
use crate::detect::AnomalyDetector;
use crate::ingest::{deduplicate_transactions, aggregate_to_blocks, RawTransaction, TransactionParser};
use crate::store::{PatternStore, TimeSeriesStore};
use crate::types::{
    Alert, AlertLevel, AnomalyScore, AnomalyType, Chain, DetectionConfig, TimeSeriesPoint,
    Transaction,
};
use chrono::{DateTime, Duration, Utc};

// ---------------------------------------------------------------------------
// Pipeline
// ---------------------------------------------------------------------------

/// The top-level anomaly detection pipeline.
///
/// ```text
///   Raw JSON  ──► Ingest  ──► Detect  ──► Store  ──► Alert
/// ```
pub struct Pipeline {
    /// Detection engine.
    detector: AnomalyDetector,
    /// Time-series store.
    store: TimeSeriesStore,
    /// Pattern store for graph-like queries.
    pattern_store: PatternStore,
    /// Alert router.
    alert_router: AlertRouter,
    /// Configuration (shared by all components).
    config: DetectionConfig,
}

/// Summary produced by a single pipeline run.
#[derive(Debug, Clone)]
pub struct PipelineRunSummary {
    pub transactions_processed: usize,
    pub duplicates_removed: usize,
    pub blocks_aggregated: usize,
    pub anomalies_detected: usize,
    pub alerts_dispatched: usize,
    pub run_timestamp: DateTime<Utc>,
}

impl Pipeline {
    /// Create a new pipeline with the given detection + alert configuration.
    pub fn new(detection_config: DetectionConfig, alert_config: AlertConfig) -> Self {
        let store_window = detection_config.sliding_window_size;
        Self {
            detector: AnomalyDetector::new(detection_config.clone()),
            store: TimeSeriesStore::new(store_window),
            pattern_store: PatternStore::new(),
            alert_router: AlertRouter::new(alert_config),
            config: detection_config,
        }
    }

    /// Convenience: create with default config for both detection and alerts.
    pub fn with_defaults() -> Self {
        Self::new(DetectionConfig::default(), AlertConfig::default())
    }

    // -- Public API ---------------------------------------------------------

    /// Process a batch of raw transactions end-to-end.
    ///
    /// 1. Parse & normalise raw data
    /// 2. Deduplicate
    /// 3. Aggregate to blocks
    /// 4. Run anomaly detection on each transaction
    /// 5. Store time-series points
    /// 6. Dispatch alerts
    pub fn process_batch(&mut self, raw_txs: Vec<RawTransaction>) -> PipelineRunSummary {
        let now = Utc::now();

        // 1. Parse
        let (txs, _parse_errors) = TransactionParser::parse_batch(raw_txs);

        // 2. Dedup
        let pre_dedup = txs.len();
        let txs = deduplicate_transactions(txs);
        let duplicates_removed = pre_dedup - txs.len();

        // 3. Aggregate to blocks
        let blocks = aggregate_to_blocks(&txs);

        // 4 + 5 + 6: Per-transaction processing
        let mut anomalies_detected = 0usize;
        let mut alerts_dispatched = 0usize;

        for tx in &txs {
            // Feed detector history
            self.detector.feed_transaction(tx);

            // Store time-series
            let ts_point = TimeSeriesPoint {
                timestamp: tx.timestamp,
                value: tx.value,
                volume: tx.gas_used,
            };
            self.store
                .insert(tx.chain, "tx_value", ts_point.clone());
            self.store
                .insert(tx.chain, "gas_price", TimeSeriesPoint {
                    timestamp: tx.timestamp,
                    value: tx.gas_price,
                    volume: tx.gas_used,
                });

            // Run detection
            let results = self.detector.detect(tx);
            anomalies_detected += results.len();

            // Dispatch alerts
            for result in &results {
                if let Some(alert) =
                    self.alert_router.dispatch(tx.chain, result.clone(), now)
                {
                    alerts_dispatched += 1;

                    // Store pattern
                    self.pattern_store.record(
                        &alert.fingerprint,
                        tx.chain,
                        result.score.value(),
                        now,
                    );
                }
            }
        }

        // Also feed block-level volume info
        for block in &blocks {
            self.detector
                .feed_block_tx_count(block.chain, block.tx_count as f64);

            // Check volume spike
            if let Some(vol_result) = self
                .detector
                .detect_volume_spike(block.chain, block.tx_count as f64)
            {
                if let Some(_alert) = self
                    .alert_router
                    .dispatch(block.chain, vol_result, now)
                {
                    alerts_dispatched += 1;
                    anomalies_detected += 1;
                }
            }
        }

        PipelineRunSummary {
            transactions_processed: txs.len(),
            duplicates_removed,
            blocks_aggregated: blocks.len(),
            anomalies_detected,
            alerts_dispatched,
            run_timestamp: now,
        }
    }

    /// Process a single already-normalised transaction.
    pub fn process_transaction(&mut self, tx: &Transaction) -> Vec<Alert> {
        let now = Utc::now();

        // Feed detector
        self.detector.feed_transaction(tx);

        // Store time-series
        self.store.insert(
            tx.chain,
            "tx_value",
            TimeSeriesPoint {
                timestamp: tx.timestamp,
                value: tx.value,
                volume: tx.gas_used,
            },
        );

        // Detect
        let results = self.detector.detect(tx);

        // Dispatch alerts
        let mut alerts = Vec::new();
        for result in &results {
            if let Some(alert) = self.alert_router.dispatch(tx.chain, result.clone(), now) {
                self.pattern_store.record(
                    &alert.fingerprint,
                    tx.chain,
                    result.score.value(),
                    now,
                );
                alerts.push(alert);
            }
        }
        alerts
    }

    /// Warm up the detector with historical time-series data.
    pub fn warm_up(&mut self, chain: Chain, points: Vec<TimeSeriesPoint>) {
        for pt in &points {
            self.detector.feed(chain, pt.clone());
            self.store.insert(chain, "tx_value", pt.clone());
        }
    }

    /// Accessor: underlying time-series store.
    pub fn store(&self) -> &TimeSeriesStore {
        &self.store
    }

    /// Accessor: underlying alert router.
    pub fn alert_router(&self) -> &AlertRouter {
        &self.alert_router
    }

    /// Accessor: underlying pattern store.
    pub fn pattern_store(&self) -> &PatternStore {
        &self.pattern_store
    }

    /// Accessor: underlying detector.
    pub fn detector(&self) -> &AnomalyDetector {
        &self.detector
    }

    /// Run a cross-chain distribution shift analysis.
    pub fn cross_chain_analysis(
        &mut self,
        chain_a: Chain,
        chain_b: Chain,
    ) -> Option<Alert> {
        let result = self.detector.detect_distribution_shift(chain_a, chain_b)?;
        let now = Utc::now();
        let result_clone = result.clone();
        self.alert_router
            .dispatch(chain_a, result, now)
            .or_else(|| self.alert_router.dispatch(chain_b, result_clone, now))
    }

    /// Reset the entire pipeline state.
    pub fn reset(&mut self) {
        self.detector = AnomalyDetector::new(self.config.clone());
        self.store.clear();
        self.pattern_store = PatternStore::new();
        self.alert_router.clear();
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ingest::RawTransaction;

    fn make_raw_tx(chain: &str, value: f64) -> RawTransaction {
        RawTransaction {
            hash: Some(format!("0x{:016x}", rand::random::<u64>())),
            chain: Some(chain.to_string()),
            block_number: Some(1),
            from: Some("0xAAAA".into()),
            to: Some("0xBBBB".into()),
            value: Some(value),
            gas_used: Some(21000.0),
            gas_price: Some(30.0),
            timestamp: None,
            token_symbol: None,
            memo: None,
        }
    }

    fn make_normal_batch(n: usize) -> Vec<RawTransaction> {
        (0..n)
            .map(|i| make_raw_tx("ethereum", 50.0 + (i as f64) * 0.1))
            .collect()
    }

    fn make_anomalous_batch() -> Vec<RawTransaction> {
        vec![
            make_raw_tx("ethereum", 10000.0), // extreme value
            make_raw_tx("ethereum", 20000.0),
        ]
    }

    fn low_min_data_points_config() -> DetectionConfig {
        let mut cfg = DetectionConfig::default();
        cfg.min_data_points = 5;
        cfg.zscore_threshold = 2.0;
        cfg.sliding_window_size = 500;
        cfg.moving_avg_window = 10;
        cfg
    }

    #[test]
    fn test_pipeline_construction() {
        let _p = Pipeline::with_defaults();
        let _p2 = Pipeline::new(DetectionConfig::default(), AlertConfig::default());
    }

    #[test]
    fn test_process_batch_normal() {
        let cfg = low_min_data_points_config();
        let mut alert_cfg = AlertConfig::default();
        alert_cfg.min_level = AlertLevel::High; // suppress low alerts

        let mut pipeline = Pipeline::new(cfg, alert_cfg);
        let summary = pipeline.process_batch(make_normal_batch(50));

        assert_eq!(summary.transactions_processed, 50);
        assert_eq!(summary.duplicates_removed, 0);
        assert!(summary.alerts_dispatched == 0 || summary.alerts_dispatched <= 1);
    }

    #[test]
    fn test_process_batch_with_anomalies() {
        let mut pipeline = Pipeline::new(
            low_min_data_points_config(),
            AlertConfig::default(),
        );

        // Warm up with normal data first
        pipeline.process_batch(make_normal_batch(50));

        // Now feed anomalous data
        let summary = pipeline.process_batch(make_anomalous_batch());
        assert_eq!(summary.transactions_processed, 2);
        assert!(summary.anomalies_detected > 0, "Expected anomalies");
        assert!(summary.alerts_dispatched > 0, "Expected alerts");
    }

    #[test]
    fn test_process_batch_deduplication() {
        let mut pipeline = Pipeline::new(
            low_min_data_points_config(),
            AlertConfig::default(),
        );

        // Create duplicate transactions (same hash)
        let raw = RawTransaction {
            hash: Some("0xDEDUP001".into()),
            chain: Some("ethereum".into()),
            block_number: Some(1),
            from: Some("0xA".into()),
            to: Some("0xB".into()),
            value: Some(100.0),
            gas_used: Some(21000.0),
            gas_price: Some(30.0),
            timestamp: None,
            token_symbol: None,
            memo: None,
        };
        let summary = pipeline.process_batch(vec![raw.clone(), raw]);
        assert_eq!(summary.duplicates_removed, 1);
        assert_eq!(summary.transactions_processed, 1);
    }

    #[test]
    fn test_process_single_transaction() {
        let mut pipeline = Pipeline::new(
            low_min_data_points_config(),
            AlertConfig::default(),
        );

        // Warm up
        pipeline.warm_up(
            Chain::Ethereum,
            (0..50)
                .map(|i| TimeSeriesPoint {
                    timestamp: Utc::now(),
                    value: 50.0 + (i as f64) * 0.1,
                    volume: 10.0,
                })
                .collect(),
        );

        let tx = Transaction {
            hash: "0xSINGLE001".into(),
            chain: Chain::Ethereum,
            block_number: 100,
            from_address: "0xA".into(),
            to_address: "0xB".into(),
            value: 5000.0,
            gas_used: 21000.0,
            gas_price: 30.0,
            timestamp: Utc::now(),
            token_symbol: None,
            memo: None,
        };

        let alerts = pipeline.process_transaction(&tx);
        assert!(
            !alerts.is_empty(),
            "Expected alerts for anomalous single transaction"
        );
    }

    #[test]
    fn test_warm_up() {
        let mut pipeline = Pipeline::new(
            low_min_data_points_config(),
            AlertConfig::default(),
        );
        let points: Vec<TimeSeriesPoint> = (0..100)
            .map(|i| TimeSeriesPoint {
                timestamp: Utc::now(),
                value: i as f64,
                volume: 1.0,
            })
            .collect();
        pipeline.warm_up(Chain::Solana, points);
        let vals = pipeline.store().values(Chain::Solana, "tx_value");
        assert_eq!(vals.len(), 100);
    }

    #[test]
    fn test_cross_chain_analysis() {
        let mut pipeline = Pipeline::new(
            low_min_data_points_config(),
            AlertConfig::default(),
        );

        // Warm up two chains with very different distributions
        let eth_points: Vec<TimeSeriesPoint> = (0..50)
            .map(|_| TimeSeriesPoint {
                timestamp: Utc::now(),
                value: 10.0,
                volume: 1.0,
            })
            .collect();
        let sol_points: Vec<TimeSeriesPoint> = (0..50)
            .map(|_| TimeSeriesPoint {
                timestamp: Utc::now(),
                value: 1000.0,
                volume: 1.0,
            })
            .collect();

        pipeline.warm_up(Chain::Ethereum, eth_points);
        pipeline.warm_up(Chain::Solana, sol_points);

        let result = pipeline.cross_chain_analysis(Chain::Ethereum, Chain::Solana);
        assert!(result.is_some(), "Expected cross-chain distribution shift alert");
    }

    #[test]
    fn test_reset() {
        let mut pipeline = Pipeline::with_defaults();
        pipeline.process_batch(make_normal_batch(5));
        pipeline.reset();
        assert_eq!(pipeline.store().total_points(), 0);
        assert_eq!(pipeline.alert_router().unique_alert_count(), 0);
    }

    #[test]
    fn test_pipeline_run_summary() {
        let mut pipeline = Pipeline::new(
            low_min_data_points_config(),
            AlertConfig::default(),
        );
        let summary = pipeline.process_batch(make_normal_batch(10));
        assert_eq!(summary.transactions_processed, 10);
        assert_eq!(summary.blocks_aggregated, 1); // all same block
        assert!(summary.run_timestamp <= Utc::now());
    }

    #[test]
    fn test_store_ohlcv_via_pipeline() {
        let mut pipeline = Pipeline::with_defaults();
        for i in 0..20 {
            pipeline.warm_up(
                Chain::Ethereum,
                vec![TimeSeriesPoint {
                    timestamp: Utc::now() + Duration::seconds(i * 60),
                    value: 100.0 + i as f64,
                    volume: 10.0,
                }],
            );
        }
        let candles = pipeline.store().ohlcv(Chain::Ethereum, "tx_value", 300);
        // 20 points at 60s intervals with 300s candle = ~4 candles
        assert!(!candles.is_empty());
        assert!(candles.len() <= 5);
        for c in &candles {
            assert!(c.high >= c.low);
            assert!(c.volume > 0.0);
        }
    }
}
