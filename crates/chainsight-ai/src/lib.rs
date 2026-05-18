//! # chainsight-ai
//!
//! An AI-powered on-chain anomaly detection engine for DeFi ecosystems.
//!
//! The engine processes blockchain transactions, detects anomalies using
//! statistical methods (Z-score, KS test, chi-squared), stores time-series
//! patterns, and dispatches multi-channel alerts.
//!
//! ## Pipeline
//!
//! ```text
//!   Raw JSON  ──► Ingest  ──► Detect  ──► Store  ──► Alert
//! ```
//!
//! ## Example
//!
//! ```
//! use chainsight_ai::pipeline::Pipeline;
//! use chainsight_ai::types::{DetectionConfig, Chain};
//! use chainsight_ai::alert::AlertConfig;
//!
//! let mut pipeline = Pipeline::with_defaults();
//! // Process transactions → detect anomalies → dispatch alerts
//! ```

pub mod alert;
pub mod detect;
pub mod ingest;
pub mod pipeline;
pub mod store;
pub mod types;

// ---------------------------------------------------------------------------
// Integration tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod integration_tests {
    use crate::alert::{AlertConfig, AlertChannel, AlertLevel};
    use crate::detect::AnomalyDetector;
    use crate::ingest::{RawTransaction, TransactionParser, deduplicate_transactions, aggregate_to_blocks, fingerprint_transaction};
    use crate::pipeline::Pipeline;
    use crate::store::{PatternStore, TimeSeriesStore};
    use crate::types::{
        Alert, AnomalyScore, AnomalyType, Chain, DetectionConfig, TimeSeriesPoint, Transaction,
    };
    use chrono::{Duration, Utc};

    // -- helpers ------------------------------------------------------------

    fn eth_raw(value: f64, hash: &str) -> RawTransaction {
        RawTransaction {
            hash: Some(hash.to_string()),
            chain: Some("ethereum".to_string()),
            block_number: Some(1),
            from: Some("0xA".into()),
            to: Some("0xB".into()),
            value: Some(value),
            gas_used: Some(21000.0),
            gas_price: Some(30.0),
            timestamp: None,
            token_symbol: None,
            memo: None,
        }
    }

    fn low_threshold_config() -> DetectionConfig {
        let mut c = DetectionConfig::default();
        c.min_data_points = 5;
        c.zscore_threshold = 2.0;
        c.sliding_window_size = 500;
        c.moving_avg_window = 10;
        c
    }

    // -- integration: full pipeline with anomaly injection -------------------

    #[test]
    fn test_full_pipeline_detects_spoof_transaction() {
        let mut pipeline = Pipeline::new(
            low_threshold_config(),
            AlertConfig::default(),
        );

        // Warm up with 50 normal transactions
        let normal: Vec<RawTransaction> = (0..50)
            .map(|i| {
                eth_raw(
                    50.0 + (i as f64) * 0.1,
                    &format!("0xNORM{:04x}", i),
                )
            })
            .collect();
        let summary_warmup = pipeline.process_batch(normal);
        assert_eq!(summary_warmup.transactions_processed, 50);
        assert_eq!(summary_warmup.duplicates_removed, 0);

        // Inject a spoof transaction with extreme value
        let spoof = vec![eth_raw(50000.0, "0xSPOOF")];
        let summary_spoof = pipeline.process_batch(spoof);

        assert_eq!(summary_spoof.transactions_processed, 1);
        assert!(
            summary_spoof.anomalies_detected > 0,
            "Spoof tx should trigger anomaly detection"
        );
        assert!(
            summary_spoof.alerts_dispatched > 0,
            "Spoof tx should dispatch at least one alert"
        );
    }

    #[test]
    fn test_full_pipeline_cross_chain_drift() {
        let mut pipeline = Pipeline::new(
            low_threshold_config(),
            AlertConfig::default(),
        );

        // Create two very different chains
        let eth_txs: Vec<RawTransaction> = (0..60)
            .map(|i| RawTransaction {
                hash: Some(format!("0xETH{:04x}", i)),
                chain: Some("ethereum".into()),
                block_number: Some(i),
                from: Some("0xA".into()),
                to: Some("0xB".into()),
                value: Some(10.0),
                gas_used: Some(21000.0),
                gas_price: Some(30.0),
                timestamp: None,
                token_symbol: None,
                memo: None,
            })
            .collect();

        let sol_txs: Vec<RawTransaction> = (0..60)
            .map(|i| RawTransaction {
                hash: Some(format!("0xSOL{:04x}", i)),
                chain: Some("solana".into()),
                block_number: Some(i),
                from: Some("0xC".into()),
                to: Some("0xD".into()),
                value: Some(5000.0),
                gas_used: Some(5000.0),
                gas_price: Some(1.0),
                timestamp: None,
                token_symbol: None,
                memo: None,
            })
            .collect();

        pipeline.process_batch(eth_txs);
        pipeline.process_batch(sol_txs);

        let alert = pipeline.cross_chain_analysis(Chain::Ethereum, Chain::Solana);
        assert!(
            alert.is_some(),
            "Cross-chain distribution shift should be detected"
        );
    }

    #[test]
    fn test_full_pipeline_deduplication_preserves_unique() {
        let mut pipeline = Pipeline::new(
            low_threshold_config(),
            AlertConfig::default(),
        );

        let batch = vec![
            eth_raw(100.0, "0xDUP1"),
            eth_raw(100.0, "0xDUP1"), // duplicate
            eth_raw(200.0, "0xDUP2"),
            eth_raw(200.0, "0xDUP2"), // duplicate
            eth_raw(300.0, "0xDUP3"),
        ];

        let summary = pipeline.process_batch(batch);
        assert_eq!(summary.transactions_processed, 3);
        assert_eq!(summary.duplicates_removed, 2);
    }

    #[test]
    fn test_full_pipeline_alert_channels_fire() {
        let mut alert_cfg = AlertConfig::default();
        alert_cfg.min_level = AlertLevel::Low;

        let mut pipeline = Pipeline::new(low_threshold_config(), alert_cfg);

        // Warm up
        let normal: Vec<RawTransaction> = (0..30)
            .map(|i| eth_raw(50.0 + (i as f64) * 0.1, &format!("0xW{:04x}", i)))
            .collect();
        pipeline.process_batch(normal);

        // Anomalous
        let anomalous = vec![eth_raw(100000.0, "0xANOM")];
        pipeline.process_batch(anomalous);

        let dispatched = pipeline.alert_router().dispatched();
        assert!(
            !dispatched.is_empty(),
            "At least one channel should receive the alert"
        );
    }

    #[test]
    fn test_pattern_store_persists_across_runs() {
        let mut pipeline = Pipeline::new(
            low_threshold_config(),
            AlertConfig::default(),
        );

        // Warm up + trigger
        let normal: Vec<RawTransaction> = (0..30)
            .map(|i| eth_raw(50.0 + (i as f64) * 0.1, &format!("0xP{:04x}", i)))
            .collect();
        pipeline.process_batch(normal);
        let anomalous = vec![eth_raw(50000.0, "0xPAT")];
        pipeline.process_batch(anomalous);

        // Pattern store should have entries
        let ps = pipeline.pattern_store();
        assert!(
            ps.len() > 0,
            "Pattern store should have at least one recorded pattern"
        );
    }

    #[test]
    fn test_full_pipeline_ohlcv_after_processing() {
        let mut pipeline = Pipeline::new(
            low_threshold_config(),
            AlertConfig::default(),
        );

        for i in 0..20 {
            let batch = vec![eth_raw(
                100.0 + (i as f64),
                &format!("0xOHLCV{:04x}", i),
            )];
            pipeline.process_batch(batch);
        }

        let candles = pipeline.store().ohlcv(Chain::Ethereum, "tx_value", 60);
        assert!(
            !candles.is_empty(),
            "OHLCV candles should be produced from stored data"
        );
        for c in &candles {
            assert!(c.high >= c.low);
        }
    }

    #[test]
    fn test_pipeline_reset_clears_all_state() {
        let mut pipeline = Pipeline::with_defaults();
        let batch: Vec<RawTransaction> = (0..10)
            .map(|i| eth_raw(i as f64 * 10.0, &format!("0xR{:04x}", i)))
            .collect();
        pipeline.process_batch(batch);

        pipeline.reset();

        assert_eq!(pipeline.store().total_points(), 0);
        assert_eq!(pipeline.alert_router().unique_alert_count(), 0);
        assert!(pipeline.pattern_store().is_empty());
    }
}
