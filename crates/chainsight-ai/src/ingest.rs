//! Transaction stream parser – normalisation, batch processing, deduplication.

use crate::types::{Block, Chain, Transaction};
use chrono::{DateTime, Utc};
use sha3::{Digest, Keccak256};
use std::collections::{HashMap, HashSet};

// ---------------------------------------------------------------------------
// IngestError
// ---------------------------------------------------------------------------

/// Errors that can occur during ingestion.
#[derive(Debug, thiserror::Error)]
pub enum IngestError {
    #[error("missing required field: {0}")]
    MissingField(String),
    #[error("invalid chain identifier: {0}")]
    InvalidChain(String),
    #[error("parse error: {0}")]
    ParseError(String),
    #[error("duplicate transaction: {0}")]
    DuplicateTransaction(String),
}

// ---------------------------------------------------------------------------
// RawTransaction
// ---------------------------------------------------------------------------

/// A minimally-validated incoming transaction before full normalisation.
#[derive(Debug, Clone, serde::Deserialize)]
pub struct RawTransaction {
    pub hash: Option<String>,
    pub chain: Option<String>,
    pub block_number: Option<u64>,
    pub from: Option<String>,
    pub to: Option<String>,
    pub value: Option<f64>,
    pub gas_used: Option<f64>,
    pub gas_price: Option<f64>,
    pub timestamp: Option<DateTime<Utc>>,
    pub token_symbol: Option<String>,
    pub memo: Option<String>,
}

// ---------------------------------------------------------------------------
// TransactionParser
// ---------------------------------------------------------------------------

/// Stateless transaction parser responsible for normalisation.
pub struct TransactionParser;

impl TransactionParser {
    /// Normalise a `RawTransaction` into a fully-validated `Transaction`.
    pub fn parse(raw: RawTransaction) -> Result<Transaction, IngestError> {
        let hash = raw.hash.ok_or_else(|| IngestError::MissingField("hash".into()))?;
        if hash.is_empty() {
            return Err(IngestError::MissingField("hash".into()));
        }

        let chain_str = raw.chain.ok_or_else(|| IngestError::MissingField("chain".into()))?;
        let chain = Chain::from_chain_id(&chain_str)
            .ok_or_else(|| IngestError::InvalidChain(chain_str))?;

        let block_number = raw
            .block_number
            .ok_or_else(|| IngestError::MissingField("block_number".into()))?;

        let from_address = raw
            .from
            .ok_or_else(|| IngestError::MissingField("from".into()))?;
        let to_address = raw
            .to
            .ok_or_else(|| IngestError::MissingField("to".into()))?;

        let value = raw.value.unwrap_or(0.0);
        let gas_used = raw.gas_used.unwrap_or(0.0);
        let gas_price = raw.gas_price.unwrap_or(0.0);

        let timestamp = raw.timestamp.unwrap_or_else(Utc::now);

        Ok(Transaction {
            hash,
            chain,
            block_number,
            from_address,
            to_address,
            value,
            gas_used,
            gas_price,
            timestamp,
            token_symbol: raw.token_symbol,
            memo: raw.memo,
        })
    }

    /// Parse a batch of raw transactions, returning successes and a list of failures.
    pub fn parse_batch(raws: Vec<RawTransaction>) -> (Vec<Transaction>, Vec<IngestError>) {
        let mut ok = Vec::with_capacity(raws.len());
        let mut errs = Vec::new();
        for r in raws {
            match Self::parse(r) {
                Ok(tx) => ok.push(tx),
                Err(e) => errs.push(e),
            }
        }
        (ok, errs)
    }
}

// ---------------------------------------------------------------------------
// Deduplication
// ---------------------------------------------------------------------------

/// Deduplicate transactions by hash.
pub fn deduplicate_transactions(txs: Vec<Transaction>) -> Vec<Transaction> {
    let mut seen = HashSet::with_capacity(txs.len());
    txs.into_iter()
        .filter(|tx| seen.insert(tx.hash.clone()))
        .collect()
}

// ---------------------------------------------------------------------------
// Block aggregation
// ---------------------------------------------------------------------------

/// Aggregate a list of transactions into block-level summaries.
pub fn aggregate_to_blocks(txs: &[Transaction]) -> Vec<Block> {
    let mut map: HashMap<(Chain, u64), Vec<&Transaction>> = HashMap::new();
    for tx in txs {
        map.entry((tx.chain, tx.block_number))
            .or_default()
            .push(tx);
    }
    let mut blocks: Vec<Block> = map
        .into_iter()
        .map(|((chain, number), group)| {
            let total_value: f64 = group.iter().map(|t| t.value).sum();
            let gas_total: f64 = group.iter().map(|t| t.gas_used).sum();
            let ts = group
                .iter()
                .map(|t| t.timestamp)
                .min()
                .unwrap_or_else(Utc::now);
            Block {
                number,
                chain,
                timestamp: ts,
                tx_count: group.len(),
                total_value,
                gas_used_total: gas_total,
            }
        })
        .collect();
    blocks.sort_by_key(|b| (b.chain as u8, b.number));
    blocks
}

// ---------------------------------------------------------------------------
// Fingerprinting
// ---------------------------------------------------------------------------

/// Produce a content fingerprint for deduplication purposes.
pub fn fingerprint_transaction(tx: &Transaction) -> String {
    let mut hasher = Keccak256::new();
    hasher.update(tx.hash.as_bytes());
    hasher.update(tx.chain.chain_id().as_bytes());
    hasher.update(tx.block_number.to_le_bytes());
    hasher.update(tx.value.to_le_bytes());
    let result = hasher.finalize();
    hex::encode(result)
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    fn make_raw(chain: &str) -> RawTransaction {
        RawTransaction {
            hash: Some("0xabc123".into()),
            chain: Some(chain.into()),
            block_number: Some(1),
            from: Some("0xAAAA".into()),
            to: Some("0xBBBB".into()),
            value: Some(100.0),
            gas_used: Some(21000.0),
            gas_price: Some(30.0),
            timestamp: None,
            token_symbol: None,
            memo: None,
        }
    }

    #[test]
    fn test_parse_valid() {
        let tx = TransactionParser::parse(make_raw("ethereum")).unwrap();
        assert_eq!(tx.hash, "0xabc123");
        assert_eq!(tx.chain, Chain::Ethereum);
        assert!((tx.value - 100.0).abs() < f64::EPSILON);
    }

    #[test]
    fn test_parse_invalid_chain() {
        let err = TransactionParser::parse(make_raw("unknown_chain"));
        assert!(matches!(err, Err(IngestError::InvalidChain(_))));
    }

    #[test]
    fn test_parse_missing_field() {
        let mut raw = make_raw("ethereum");
        raw.hash = None;
        let err = TransactionParser::parse(raw);
        assert!(matches!(err, Err(IngestError::MissingField(_))));
    }

    #[test]
    fn test_parse_batch_mixed() {
        let r1 = make_raw("ethereum");
        let mut r2 = make_raw("solana");
        r2.hash = None;
        let (ok, errs) = TransactionParser::parse_batch(vec![r1, r2]);
        assert_eq!(ok.len(), 1);
        assert_eq!(errs.len(), 1);
    }

    #[test]
    fn test_deduplication() {
        let tx = TransactionParser::parse(make_raw("ethereum")).unwrap();
        let tx2 = tx.clone();
        let tx3 = {
            let mut t = tx.clone();
            t.hash = "0xdef456".into();
            t
        };
        let deduped = deduplicate_transactions(vec![tx, tx2, tx3]);
        assert_eq!(deduped.len(), 2);
    }

    #[test]
    fn test_aggregate_to_blocks() {
        let tx1 = TransactionParser::parse(make_raw("ethereum")).unwrap();
        let mut tx2 = tx1.clone();
        tx2.value = 200.0;
        let blocks = aggregate_to_blocks(&[tx1, tx2]);
        assert_eq!(blocks.len(), 1);
        assert_eq!(blocks[0].tx_count, 2);
        assert!((blocks[0].total_value - 300.0).abs() < f64::EPSILON);
    }

    #[test]
    fn test_fingerprint_deterministic() {
        let tx = TransactionParser::parse(make_raw("ethereum")).unwrap();
        let fp1 = fingerprint_transaction(&tx);
        let fp2 = fingerprint_transaction(&tx);
        assert_eq!(fp1, fp2);
        assert!(!fp1.is_empty());
    }

    #[test]
    fn test_fingerprint_different_txs() {
        let tx1 = TransactionParser::parse(make_raw("ethereum")).unwrap();
        let mut tx2 = tx1.clone();
        tx2.value = 999.0;
        let fp1 = fingerprint_transaction(&tx1);
        let fp2 = fingerprint_transaction(&tx2);
        assert_ne!(fp1, fp2);
    }

    #[test]
    fn test_empty_value_defaults() {
        let mut raw = make_raw("ethereum");
        raw.value = None;
        raw.gas_used = None;
        let tx = TransactionParser::parse(raw).unwrap();
        assert!((tx.value - 0.0).abs() < f64::EPSILON);
        assert!((tx.gas_used - 0.0).abs() < f64::EPSILON);
    }
}
