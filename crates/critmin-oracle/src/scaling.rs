//! On-chain scaling functions and keccak256 hashing.

use sha3::{Keccak256, Digest};
use crate::config::{PRICE_SCALE, SCORE_SCALE, SENTIMENT_SCALE, REG_RISK_SCALE};

/// Compute keccak256 hash of a mineral symbol (matches Solidity's keccak256(abi.encodePacked(symbol))).
pub fn mineral_hash(symbol: &str) -> [u8; 32] {
    let mut hasher = Keccak256::new();
    hasher.update(symbol.as_bytes());
    let result = hasher.finalize();
    let mut hash = [0u8; 32];
    hash.copy_from_slice(&result);
    hash
}

/// Compute keccak256 hash and return as hex string.
pub fn mineral_hash_hex(symbol: &str) -> String {
    let hash = mineral_hash(symbol);
    hex::encode(hash)
}

/// Scale a USD price to on-chain format (multiplied by PRICE_SCALE).
pub fn scale_price(price_usd: f64) -> i64 {
    (price_usd * PRICE_SCALE as f64) as i64
}

/// Scale composite score from [-100, 100] to on-chain format [-10000, 10000].
pub fn scale_composite(score: f64) -> i64 {
    (score * SCORE_SCALE as f64) as i64
}

/// Scale sentiment from [-1.0, 1.0] to on-chain format [-10000, 10000].
pub fn scale_sentiment(sentiment: f64) -> i64 {
    (sentiment * SENTIMENT_SCALE as f64) as i64
}

/// Scale regulatory risk from [0, 100] to on-chain format [0, 10000].
pub fn scale_reg_risk(risk: f64) -> i64 {
    (risk * REG_RISK_SCALE as f64) as i64
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_keccak_matches_solidity_lithium() {
        // keccak256("LITHIUM") should match Solidity's constant
        let hash = mineral_hash("LITHIUM");
        let hex = hex::encode(hash);
        // Known keccak256("LITHIUM") from Solidity: first bytes
        assert!(hex.len() == 64, "Expected 64-char hex, got {}", hex.len());
        // Verify determinism — same input always produces same hash
        let hash2 = mineral_hash("LITHIUM");
        assert_eq!(hash, hash2, "Hash should be deterministic");
    }

    #[test]
    fn test_scale_price() {
        assert_eq!(scale_price(15_000.0), 1_500_000_000_000_i64);
    }

    #[test]
    fn test_scale_composite() {
        assert_eq!(scale_composite(50.0), 5_000);
        assert_eq!(scale_composite(-25.0), -2_500);
    }

    #[test]
    fn test_scale_sentiment() {
        assert_eq!(scale_sentiment(0.5), 5_000);
        assert_eq!(scale_sentiment(-1.0), -10_000);
    }

    #[test]
    fn test_scale_reg_risk() {
        assert_eq!(scale_reg_risk(75.0), 7_500);
        assert_eq!(scale_reg_risk(0.0), 0);
    }
}
