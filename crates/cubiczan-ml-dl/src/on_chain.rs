//! # On-Chain Machine Learning
//!
//! ML components for blockchain transaction analysis, anomaly detection,
//! MEV pattern recognition, and smart contract risk scoring.

use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use tracing::debug;

/// Encoded transaction for ML processing.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EncodedTransaction {
    pub tx_hash: String,
    pub features: Vec<f64>,
    pub labels: Vec<f64>,
    pub timestamp: u64,
    pub block_number: u64,
}

/// Transaction encoder: converts raw on-chain data into ML feature vectors.
pub struct TransactionEncoder {
    pub feature_names: Vec<String>,
    pub normalization_params: HashMap<String, (f64, f64)>, // (mean, std)
}

impl TransactionEncoder {
    pub fn new(feature_names: Vec<String>) -> Self {
        Self {
            feature_names,
            normalization_params: HashMap::new(),
        }
    }

    /// Encode a transaction into a fixed-length feature vector.
    pub fn encode(&self, tx: &RawTransaction) -> EncodedTransaction {
        let mut features = Vec::with_capacity(self.feature_names.len());

        for name in &self.feature_names {
            let val = match name.as_str() {
                "value" => tx.value as f64,
                "gas_price" => tx.gas_price as f64,
                "gas_used" => tx.gas_used as f64,
                "input_size" => tx.input_data.len() as f64,
                "is_contract_call" => if tx.to_address.is_some() && !tx.input_data.is_empty() { 1.0 } else { 0.0 },
                "log_value" => (tx.value as f64 + 1.0).ln(),
                "gas_efficiency" => if tx.gas_used > 0 { tx.value as f64 / tx.gas_used as f64 } else { 0.0 },
                _ => 0.0,
            };
            features.push(val);
        }

        EncodedTransaction {
            tx_hash: tx.hash.clone(),
            features,
            labels: vec![],
            timestamp: tx.timestamp,
            block_number: tx.block_number,
        }
    }

    /// Encode a batch of transactions.
    pub fn encode_batch(&self, txs: &[RawTransaction]) -> Vec<EncodedTransaction> {
        txs.iter().map(|tx| self.encode(tx)).collect()
    }

    /// Fit normalization parameters from encoded transactions.
    pub fn fit_normalization(&mut self, encoded: &[EncodedTransaction]) {
        let n = encoded.len().max(1) as f64;
        for (i, name) in self.feature_names.iter().enumerate() {
            let mean: f64 = encoded.iter().map(|e| e.features.get(i).copied().unwrap_or(0.0)).sum::<f64>() / n;
            let var: f64 = encoded.iter()
                .map(|e| { let v = e.features.get(i).copied().unwrap_or(0.0); (v - mean).powi(2) })
                .sum::<f64>() / n;
            self.normalization_params.insert(name.clone(), (mean, var.sqrt().max(1e-8)));
        }
    }

    /// Normalize an encoded transaction's features.
    pub fn normalize(&self, encoded: &mut EncodedTransaction) {
        for (i, name) in self.feature_names.iter().enumerate() {
            if let Some(&(mean, std)) = self.normalization_params.get(name) {
                if let Some(v) = encoded.features.get_mut(i) {
                    *v = (*v - mean) / std;
                }
            }
        }
    }
}

/// Raw transaction from on-chain data.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RawTransaction {
    pub hash: String,
    pub from_address: String,
    pub to_address: Option<String>,
    pub value: u64,
    pub gas_price: u64,
    pub gas_used: u64,
    pub input_data: Vec<u8>,
    pub timestamp: u64,
    pub block_number: u64,
}

/// On-chain anomaly detection result.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OnChainAnomalyResult {
    pub tx_hash: String,
    pub anomaly_score: f64,
    pub is_anomaly: bool,
    pub anomaly_type: AnomalyType,
    pub details: String,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum AnomalyType {
    HighValue,
    GasManipulation,
    FlashLoan,
    UnusualPattern,
    DustAttack,
    Normal,
}

/// On-chain anomaly detector using statistical and rule-based methods.
pub struct OnChainAnomalyDetector {
    value_threshold: f64,     // in log space
    gas_price_threshold: f64, // standard deviations above mean
    flash_loan_threshold: f64, // minimum value for flash loan detection
}

impl OnChainAnomalyDetector {
    pub fn new() -> Self {
        Self {
            value_threshold: 20.0,    // log(value) > 20 is suspicious
            gas_price_threshold: 3.0, // > 3 std devs
            flash_loan_threshold: 1e18, // 1 ETH equivalent
        }
    }

    /// Detect anomalies in encoded transactions.
    pub fn detect(&self, encoded: &[EncodedTransaction]) -> Vec<OnChainAnomalyResult> {
        let mut results = Vec::new();
        for e in encoded {
            let value_idx = 0;
            let gas_idx = 1;
            let val = e.features.get(value_idx).copied().unwrap_or(0.0);
            let log_val = e.features.get(5).copied().unwrap_or(0.0);
            let gas = e.features.get(gas_idx).copied().unwrap_or(0.0);

            let (anomaly_type, score, details) = if log_val > self.value_threshold {
                (AnomalyType::HighValue, log_val / self.value_threshold,
                 format!("High value transaction: log_value={:.2}", log_val))
            } else if val > self.flash_loan_threshold {
                (AnomalyType::FlashLoan, val / self.flash_loan_threshold,
                 format!("Possible flash loan: value={:.2e}", val))
            } else if gas > self.gas_price_threshold {
                (AnomalyType::GasManipulation, gas / self.gas_price_threshold,
                 format!("Unusual gas: gas_price={:.2}", gas))
            } else {
                (AnomalyType::Normal, 0.0, String::new())
            };

            results.push(OnChainAnomalyResult {
                tx_hash: e.tx_hash.clone(),
                anomaly_score: score,
                is_anomaly: anomaly_type != AnomalyType::Normal,
                anomaly_type,
                details,
            });
        }
        results
    }
}

/// MEV (Maximal Extractable Value) pattern.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MevPattern {
    pub pattern_type: MevType,
    pub confidence: f64,
    pub addresses_involved: Vec<String>,
    pub estimated_profit: f64,
    pub description: String,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum MevType {
    Sandwich,
    FrontRun,
    BackRun,
    Arbitrage,
    Liquidation,
    Unknown,
}

/// Pattern recognizer for DeFi MEV and suspicious activity.
pub struct PatternRecognizer {
    min_profit_threshold: f64,
    time_window_ms: u64,
}

impl PatternRecognizer {
    pub fn new() -> Self {
        Self { min_profit_threshold: 0.01, time_window_ms: 5000 }
    }

    /// Analyze a sequence of transactions for MEV patterns.
    pub fn analyze(&self, txs: &[RawTransaction]) -> Vec<MevPattern> {
        let mut patterns = Vec::new();
        let encoded: Vec<_> = txs.iter().map(|tx| (tx.value as f64, tx.timestamp)).collect();

        // Detect potential sandwich attacks (three related txs in quick succession)
        for i in 0..encoded.len().saturating_sub(2) {
            let (v1, t1) = encoded[i];
            let (v2, t2) = encoded[i + 1];
            let (v3, t3) = encoded[i + 2];

            if t2 > t1 && t3 > t2 && t3 - t1 < self.time_window_ms {
                // Large tx sandwiched between two smaller ones
                if v2 > v1 * 10.0 && v2 > v3 * 10.0 {
                    patterns.push(MevPattern {
                        pattern_type: MevType::Sandwich,
                        confidence: 0.6,
                        addresses_involved: vec![
                            txs[i].from_address.clone(),
                            txs[i + 1].from_address.clone(),
                            txs[i + 2].from_address.clone(),
                        ],
                        estimated_profit: (v1 + v3) * 0.003,
                        description: "Potential sandwich attack detected".to_string(),
                    });
                }
            }
        }

        patterns
    }
}

/// Token flow analysis result.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TokenFlowReport {
    pub token_address: String,
    pub total_volume: f64,
    pub unique_senders: usize,
    pub unique_receivers: usize,
    pub top_senders: Vec<(String, f64)>,
    pub concentration_index: f64, // Herfindahl index
    pub is_suspicious: bool,
}

/// Flow analyzer for token transfer patterns.
pub struct FlowAnalyzer;

impl FlowAnalyzer {
    /// Analyze token flows from a list of transfers.
    pub fn analyze(
        token_address: &str,
        transfers: &[TokenTransfer],
    ) -> TokenFlowReport {
        let mut senders: HashMap<String, f64> = HashMap::new();
        let mut receivers: HashMap<String, f64> = HashMap::new();
        let mut total_volume = 0.0;

        for t in transfers {
            *senders.entry(t.from.clone()).or_insert(0.0) += t.amount;
            *receivers.entry(t.to.clone()).or_insert(0.0) += t.amount;
            total_volume += t.amount;
        }

        let unique_senders_count = senders.len();
        let unique_receivers_count = receivers.len();
        let mut top_senders: Vec<(String, f64)> = senders.into_iter().collect();
        top_senders.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());

        // Herfindahl concentration index
        let hhi: f64 = if total_volume > 0.0 {
            top_senders.iter().map(|(_, amt)| (amt / total_volume).powi(2)).sum()
        } else { 0.0 };

        TokenFlowReport {
            token_address: token_address.to_string(),
            total_volume,
            unique_senders: unique_senders_count,
            unique_receivers: unique_receivers_count,
            top_senders: top_senders.into_iter().take(10).collect(),
            concentration_index: hhi,
            is_suspicious: hhi > 0.5, // highly concentrated
        }
    }
}

/// Token transfer record.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TokenTransfer {
    pub from: String,
    pub to: String,
    pub amount: f64,
    pub timestamp: u64,
}

/// Smart contract risk assessment.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RiskAssessment {
    pub contract_address: String,
    pub overall_score: f64, // 0.0 (safe) to 1.0 (risky)
    pub risk_factors: Vec<RiskFactor>,
    pub recommendation: RiskRecommendation,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RiskFactor {
    pub name: String,
    pub score: f64,
    pub weight: f64,
    pub description: String,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum RiskRecommendation {
    Safe,
    LowRisk,
    MediumRisk,
    HighRisk,
    Critical,
}

/// Smart contract risk scorer.
pub struct ContractRiskScorer {
    factor_weights: Vec<(String, f64)>,
}

impl ContractRiskScorer {
    pub fn new() -> Self {
        Self {
            factor_weights: vec![
                ("code_complexity".to_string(), 0.2),
                ("external_calls".to_string(), 0.25),
                ("ownership_centralization".to_string(), 0.15),
                ("tx_volume_anomaly".to_string(), 0.2),
                ("age_factor".to_string(), 0.1),
                ("verified_source".to_string(), 0.1),
            ],
        }
    }

    /// Score a contract based on risk factors.
    pub fn score(&self, factors: &[(String, f64)]) -> RiskAssessment {
        let factor_map: HashMap<String, f64> = factors.iter().cloned().collect();

        let risk_factors: Vec<RiskFactor> = self
            .factor_weights
            .iter()
            .map(|(name, weight)| {
                let score = factor_map.get(name).copied().unwrap_or(0.5);
                RiskFactor {
                    name: name.clone(),
                    score,
                    weight: *weight,
                    description: format!("{}: {:.2} (weight: {:.0}%)", name, score, weight * 100.0),
                }
            })
            .collect();

        let overall: f64 = risk_factors.iter().map(|f| f.score * f.weight).sum();
        let recommendation = if overall < 0.2 { RiskRecommendation::Safe }
            else if overall < 0.4 { RiskRecommendation::LowRisk }
            else if overall < 0.6 { RiskRecommendation::MediumRisk }
            else if overall < 0.8 { RiskRecommendation::HighRisk }
            else { RiskRecommendation::Critical };

        RiskAssessment {
            contract_address: String::new(),
            overall_score: overall,
            risk_factors,
            recommendation,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn sample_tx(value: u64, gas: u64) -> RawTransaction {
        RawTransaction {
            hash: format!("0x{}", value),
            from_address: "0xABC".to_string(),
            to_address: Some("0xDEF".to_string()),
            value,
            gas_price: gas,
            gas_used: 21000,
            input_data: vec![],
            timestamp: 1000,
            block_number: 1,
        }
    }

    #[test]
    fn test_encode_transaction() {
        let encoder = TransactionEncoder::new(vec![
            "value".into(), "gas_price".into(), "gas_used".into(),
            "input_size".into(), "is_contract_call".into(), "log_value".into(), "gas_efficiency".into(),
        ]);
        let tx = sample_tx(1_000_000, 50_000);
        let encoded = encoder.encode(&tx);
        assert_eq!(encoded.features.len(), 7);
        assert!(encoded.features[0] > 0.0);
    }

    #[test]
    fn test_anomaly_detector() {
        let detector = OnChainAnomalyDetector::new();
        let encoder = TransactionEncoder::new(vec![
            "value".into(), "gas_price".into(), "gas_used".into(),
            "input_size".into(), "is_contract_call".into(), "log_value".into(), "gas_efficiency".into(),
        ]);

        let txs = vec![
            sample_tx(1_000, 50_000),
            sample_tx(u64::MAX, 50_000), // very high value
            sample_tx(5_000, 50_000),
        ];
        let encoded = encoder.encode_batch(&txs);
        let results = detector.detect(&encoded);
        let anomalies: Vec<_> = results.iter().filter(|r| r.is_anomaly).collect();
        assert!(!anomalies.is_empty());
    }

    #[test]
    fn test_flow_analyzer() {
        let transfers = vec![
            TokenTransfer { from: "A".into(), to: "B".into(), amount: 100.0, timestamp: 1 },
            TokenTransfer { from: "A".into(), to: "C".into(), amount: 50.0, timestamp: 2 },
            TokenTransfer { from: "B".into(), to: "D".into(), amount: 30.0, timestamp: 3 },
        ];
        let report = FlowAnalyzer::analyze("0xTOKEN", &transfers);
        assert_eq!(report.unique_senders, 2);
        assert_eq!(report.total_volume, 180.0);
    }

    #[test]
    fn test_risk_scorer() {
        let scorer = ContractRiskScorer::new();
        let factors = vec![
            ("code_complexity".into(), 0.8),
            ("external_calls".into(), 0.6),
        ];
        let assessment = scorer.score(&factors);
        assert!(assessment.overall_score > 0.0);
    }
}
