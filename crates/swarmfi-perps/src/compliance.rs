//! Compliance Screen — Regulatory risk scoring for DeFi perpetuals.
//!
//! Evaluates market conditions, token characteristics, and exchange-specific
//! risk factors to produce a composite compliance risk score. Designed for
//! pre-trade screening and portfolio compliance monitoring.
//!
//! # Risk Dimensions
//!
//! - **Market Manipulation Risk**: Volume concentration, wash-trading indicators
//! - **Regulatory Risk**: Jurisdiction exposure, token classification likelihood
//! - **Counterparty Risk**: Exchange solvency indicators, withdrawal health
//! - **Sanctions Risk**: Address/entity screening (simplified)
//! - **Token Risk**: Contract audit status, governance decentralization

use crate::types::*;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

/// Overall compliance risk level.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize, Default)]
pub enum ComplianceRiskLevel {
    #[default]
    Low,
    Medium,
    High,
    Critical,
}

impl ComplianceRiskLevel {
    pub fn as_str(&self) -> &'static str {
        match self {
            ComplianceRiskLevel::Low => "LOW",
            ComplianceRiskLevel::Medium => "MEDIUM",
            ComplianceRiskLevel::High => "HIGH",
            ComplianceRiskLevel::Critical => "CRITICAL",
        }
    }

    pub fn from_score(score: f64) -> Self {
        if score < 30.0 {
            ComplianceRiskLevel::Low
        } else if score < 60.0 {
            ComplianceRiskLevel::Medium
        } else if score < 80.0 {
            ComplianceRiskLevel::High
        } else {
            ComplianceRiskLevel::Critical
        }
    }

    /// Color code for UI display.
    pub fn color_hex(&self) -> &'static str {
        match self {
            ComplianceRiskLevel::Low => "#27AE60",
            ComplianceRiskLevel::Medium => "#F39C12",
            ComplianceRiskLevel::High => "#E67E22",
            ComplianceRiskLevel::Critical => "#E74C3C",
        }
    }
}

/// Jurisdiction classification for regulatory assessment.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum Jurisdiction {
    /// United States — SEC, CFTC jurisdiction (high scrutiny for perps).
    UnitedStates,
    /// European Union — MiCA framework.
    EuropeanUnion,
    /// United Kingdom — FCA oversight.
    UnitedKingdom,
    /// Singapore — MAS regulated.
    Singapore,
    /// Hong Kong — SFC regulated.
    HongKong,
    /// Cayman Islands — Offshore, light regulation.
    CaymanIslands,
    /// Unregulated / unknown jurisdiction.
    Unregulated,
}

impl Jurisdiction {
    /// Base regulatory risk weight (0.0–1.0) for this jurisdiction.
    pub fn regulatory_weight(&self) -> f64 {
        match self {
            Jurisdiction::UnitedStates => 0.9,
            Jurisdiction::EuropeanUnion => 0.5,
            Jurisdiction::UnitedKingdom => 0.6,
            Jurisdiction::Singapore => 0.4,
            Jurisdiction::HongKong => 0.5,
            Jurisdiction::CaymanIslands => 0.3,
            Jurisdiction::Unregulated => 1.0,
        }
    }
}

/// Individual risk factor score.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RiskFactor {
    pub name: String,
    pub dimension: RiskDimension,
    /// Score from 0 (no risk) to 100 (maximum risk).
    pub score: f64,
    /// Weight in the composite score (0.0–1.0).
    pub weight: f64,
    /// Human-readable description of the finding.
    pub finding: String,
    /// Recommended action.
    pub recommendation: String,
}

/// Risk dimension categories.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum RiskDimension {
    MarketManipulation,
    Regulatory,
    Counterparty,
    Sanctions,
    TokenRisk,
    Liquidity,
}

impl RiskDimension {
    pub fn as_str(&self) -> &'static str {
        match self {
            RiskDimension::MarketManipulation => "MARKET_MANIPULATION",
            RiskDimension::Regulatory => "REGULATORY",
            RiskDimension::Counterparty => "COUNTERPARTY",
            RiskDimension::Sanctions => "SANCTIONS",
            RiskDimension::TokenRisk => "TOKEN_RISK",
            RiskDimension::Liquidity => "LIQUIDITY",
        }
    }
}

/// Market data inputs for compliance screening.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ComplianceMarketData {
    pub market: String,
    pub exchange: String,
    /// 24h volume in USD.
    pub volume_24h_usd: f64,
    /// Open interest in USD.
    pub open_interest_usd: f64,
    /// Bid-ask spread in basis points.
    pub spread_bps: f64,
    /// Number of unique traders in last 24h.
    pub unique_traders_24h: u32,
    /// Top trader volume concentration (0.0–1.0).
    pub top_trader_concentration: f64,
    /// Exchange jurisdiction.
    pub exchange_jurisdiction: Jurisdiction,
    /// Whether the exchange has a known compliance license.
    pub has_license: bool,
    /// Recent withdrawal queue depth (USD).
    pub withdrawal_queue_usd: f64,
    /// Whether the underlying token is a security (heuristic score 0–1).
    pub security_heuristic: f64,
    /// Token audit status (0 = none, 1 = partial, 2 = full).
    pub audit_status: u8,
    /// Governance decentralization score (0 = fully centralized, 1 = fully decentralized).
    pub governance_decentralization: f64,
}

impl Default for ComplianceMarketData {
    fn default() -> Self {
        Self {
            market: String::new(),
            exchange: "DYDX".into(),
            volume_24h_usd: 0.0,
            open_interest_usd: 0.0,
            spread_bps: 0.0,
            unique_traders_24h: 0,
            top_trader_concentration: 0.0,
            exchange_jurisdiction: Jurisdiction::Unregulated,
            has_license: false,
            withdrawal_queue_usd: 0.0,
            security_heuristic: 0.0,
            audit_status: 0,
            governance_decentralization: 0.0,
        }
    }
}

/// The complete compliance screening report.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ComplianceReport {
    pub market: String,
    pub exchange: String,
    /// Composite risk score (0–100).
    pub composite_score: f64,
    /// Overall risk level.
    pub risk_level: ComplianceRiskLevel,
    /// Individual risk factor scores.
    pub factors: Vec<RiskFactor>,
    /// Whether the trade is permitted under the current risk appetite.
    pub trade_permitted: bool,
    /// Minimum risk level that would block the trade.
    pub blocking_threshold: ComplianceRiskLevel,
    /// Timestamp of the assessment.
    pub timestamp_ms: i64,
    /// Recommended holding period limit (days, 0 = no limit).
    pub max_holding_days: u32,
}

/// Configuration for the compliance screener.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ComplianceConfig {
    /// Risk level at or above which trades are blocked.
    pub blocking_level: ComplianceRiskLevel,
    /// Whether to require exchange licensing.
    pub require_license: bool,
    /// Maximum acceptable security heuristic (0–1).
    pub max_security_heuristic: f64,
    /// Maximum acceptable top-trader concentration (0–1).
    pub max_trader_concentration: f64,
    /// Minimum required unique traders in 24h.
    pub min_unique_traders: u32,
    /// Maximum spread in bps.
    pub max_spread_bps: f64,
}

impl Default for ComplianceConfig {
    fn default() -> Self {
        Self {
            blocking_level: ComplianceRiskLevel::High,
            require_license: false,
            max_security_heuristic: 0.8,
            max_trader_concentration: 0.5,
            min_unique_traders: 50,
            max_spread_bps: 50.0,
        }
    }
}

/// The compliance screening engine.
pub struct ComplianceScreener {
    config: ComplianceConfig,
}

impl Default for ComplianceScreener {
    fn default() -> Self {
        Self::new()
    }
}

impl ComplianceScreener {
    pub fn new() -> Self {
        Self {
            config: ComplianceConfig::default(),
        }
    }

    pub fn with_config(config: ComplianceConfig) -> Self {
        Self { config }
    }

    /// Run a full compliance screen on the given market data.
    pub fn screen(&self, data: &ComplianceMarketData) -> ComplianceReport {
        let mut factors = Vec::new();

        // 1. Market Manipulation Risk
        let manipulation = self.assess_manipulation_risk(data);
        factors.push(manipulation);

        // 2. Regulatory Risk
        let regulatory = self.assess_regulatory_risk(data);
        factors.push(regulatory);

        // 3. Counterparty Risk
        let counterparty = self.assess_counterparty_risk(data);
        factors.push(counterparty);

        // 4. Sanctions Risk (simplified — in production would use OFAC/EU lists)
        let sanctions = self.assess_sanctions_risk(data);
        factors.push(sanctions);

        // 5. Token Risk
        let token = self.assess_token_risk(data);
        factors.push(token);

        // 6. Liquidity Risk
        let liquidity = self.assess_liquidity_risk(data);
        factors.push(liquidity);

        // Compute composite score
        let total_weight: f64 = factors.iter().map(|f| f.weight).sum();
        let composite = if total_weight > 0.0 {
            factors.iter().map(|f| f.score * f.weight).sum::<f64>() / total_weight
        } else {
            0.0
        };

        let risk_level = ComplianceRiskLevel::from_score(composite);
        let trade_permitted = risk_level < self.config.blocking_level;

        // Recommended holding period
        let max_holding_days = match risk_level {
            ComplianceRiskLevel::Low => 0,     // No limit
            ComplianceRiskLevel::Medium => 30,
            ComplianceRiskLevel::High => 7,    // Would be blocked
            ComplianceRiskLevel::Critical => 0, // Would be blocked
        };

        ComplianceReport {
            market: data.market.clone(),
            exchange: data.exchange.clone(),
            composite_score: composite,
            risk_level,
            factors,
            trade_permitted,
            blocking_threshold: self.config.blocking_level,
            timestamp_ms: chrono::Utc::now().timestamp_millis(),
            max_holding_days,
        }
    }

    /// Assess market manipulation risk based on concentration and liquidity metrics.
    fn assess_manipulation_risk(&self, data: &ComplianceMarketData) -> RiskFactor {
        let mut score: f64 = 0.0;
        let mut findings = Vec::new();

        // High trader concentration → manipulation risk
        if data.top_trader_concentration > 0.3 {
            score += 30.0 * (data.top_trader_concentration - 0.3) / 0.7;
            findings.push(format!(
                "Top trader concentration at {:.1}% (threshold: 30%)",
                data.top_trader_concentration * 100.0
            ));
        }

        // Low unique trader count
        if data.unique_traders_24h < self.config.min_unique_traders {
            let deficit = 1.0 - (data.unique_traders_24h as f64 / self.config.min_unique_traders as f64);
            score += 25.0 * deficit;
            findings.push(format!(
                "Low unique trader count: {} (minimum: {})",
                data.unique_traders_24h, self.config.min_unique_traders
            ));
        }

        // Wide spread
        if data.spread_bps > self.config.max_spread_bps {
            score += 20.0;
            findings.push(format!("Wide spread: {:.1} bps (max: {:.1})", data.spread_bps, self.config.max_spread_bps));
        }

        // Volume/OI ratio — very low volume relative to OI
        if data.open_interest_usd > 0.0 {
            let vol_oi_ratio = data.volume_24h_usd / data.open_interest_usd;
            if vol_oi_ratio < 0.1 {
                score += 25.0;
                findings.push("Very low volume relative to open interest".to_string());
            }
        }

        score = score.min(100.0);
        let recommendation = if score > 60.0 {
            "AVOID: High manipulation risk detected".to_string()
        } else if score > 30.0 {
            "CAUTION: Elevated manipulation risk".to_string()
        } else {
            "OK: Manipulation risk within acceptable range".to_string()
        };

        RiskFactor {
            name: "Market Manipulation".into(),
            dimension: RiskDimension::MarketManipulation,
            score,
            weight: 0.25,
            finding: if findings.is_empty() {
                "No manipulation indicators detected".to_string()
            } else {
                findings.join("; ")
            },
            recommendation,
        }
    }

    /// Assess regulatory risk based on jurisdiction and token classification.
    fn assess_regulatory_risk(&self, data: &ComplianceMarketData) -> RiskFactor {
        let jurisdiction_score = data.exchange_jurisdiction.regulatory_weight() * 100.0;
        let security_score = data.security_heuristic * 100.0;
        let mut score = (jurisdiction_score * 0.6 + security_score * 0.4).min(100.0);

        let mut findings = vec![
            format!("Exchange jurisdiction: {:?}", data.exchange_jurisdiction),
            format!("Security heuristic: {:.1}%", data.security_heuristic * 100.0),
        ];

        if !data.has_license && self.config.require_license {
            score = score.max(70.0);
            findings.push("Exchange lacks regulatory license (required)".to_string());
        }

        let recommendation = if score > 60.0 {
            "REVIEW: Potential regulatory exposure".to_string()
        } else {
            "OK: Regulatory risk acceptable".to_string()
        };

        RiskFactor {
            name: "Regulatory".into(),
            dimension: RiskDimension::Regulatory,
            score,
            weight: 0.20,
            finding: findings.join("; "),
            recommendation,
        }
    }

    /// Assess counterparty (exchange) risk.
    fn assess_counterparty_risk(&self, data: &ComplianceMarketData) -> RiskFactor {
        let mut score: f64 = 10.0; // Base counterparty risk
        let mut findings = Vec::new();

        // Large withdrawal queue → solvency concern
        if data.withdrawal_queue_usd > 1_000_000_000.0 {
            score += 40.0;
            findings.push("Large withdrawal queue detected (> $1B)".to_string());
        } else if data.withdrawal_queue_usd > 100_000_000.0 {
            score += 20.0;
            findings.push(format!("Withdrawal queue: ${:.0}M", data.withdrawal_queue_usd / 1_000_000.0));
        }

        // Unregulated jurisdiction adds risk
        if data.exchange_jurisdiction == Jurisdiction::Unregulated {
            score += 30.0;
            findings.push("Exchange operates from unregulated jurisdiction".to_string());
        }

        score = score.min(100.0);

        let recommendation = if score > 50.0 {
            "CAUTION: Elevated counterparty risk".to_string()
        } else {
            "OK: Counterparty risk acceptable".to_string()
        };

        RiskFactor {
            name: "Counterparty".into(),
            dimension: RiskDimension::Counterparty,
            score,
            weight: 0.20,
            finding: if findings.is_empty() {
                "No counterparty risk indicators".to_string()
            } else {
                findings.join("; ")
            },
            recommendation,
        }
    }

    /// Assess sanctions risk (simplified placeholder).
    fn assess_sanctions_risk(&self, data: &ComplianceMarketData) -> RiskFactor {
        // In production, this would screen against OFAC SDN, EU sanctions lists, etc.
        let score = if data.exchange_jurisdiction == Jurisdiction::Unregulated {
            20.0
        } else {
            5.0
        };

        RiskFactor {
            name: "Sanctions".into(),
            dimension: RiskDimension::Sanctions,
            score,
            weight: 0.10,
            finding: "No sanctions matches found (basic screening)".to_string(),
            recommendation: "OK: No sanctions flags".to_string(),
        }
    }

    /// Assess token-level risk (audit, governance).
    fn assess_token_risk(&self, data: &ComplianceMarketData) -> RiskFactor {
        let mut score: f64 = 0.0;
        let mut findings = Vec::new();

        // Audit status
        match data.audit_status {
            0 => {
                score += 30.0;
                findings.push("No known security audit".to_string());
            }
            1 => {
                score += 15.0;
                findings.push("Partial audit only".to_string());
            }
            _ => {
                findings.push("Full security audit completed".to_string());
            }
        }

        // Governance decentralization (low = more risk)
        if data.governance_decentralization < 0.3 {
            score += 25.0;
            findings.push("Highly centralized governance".to_string());
        } else if data.governance_decentralization < 0.6 {
            score += 10.0;
            findings.push("Partially centralized governance".to_string());
        }

        score = score.min(100.0);

        let recommendation = if score > 40.0 {
            "CAUTION: Token risk factors detected".to_string()
        } else {
            "OK: Token risk acceptable".to_string()
        };

        RiskFactor {
            name: "Token Risk".into(),
            dimension: RiskDimension::TokenRisk,
            score,
            weight: 0.15,
            finding: findings.join("; "),
            recommendation,
        }
    }

    /// Assess liquidity risk.
    fn assess_liquidity_risk(&self, data: &ComplianceMarketData) -> RiskFactor {
        let mut score: f64 = 0.0;
        let mut findings = Vec::new();

        // Low volume
        if data.volume_24h_usd < 1_000_000.0 {
            score += 40.0;
            findings.push("Very low 24h volume (< $1M)".to_string());
        } else if data.volume_24h_usd < 10_000_000.0 {
            score += 20.0;
            findings.push("Low 24h volume (< $10M)".to_string());
        }

        // Wide spread
        if data.spread_bps > 20.0 {
            score += 30.0;
            findings.push(format!("Wide spread: {:.1} bps", data.spread_bps));
        }

        // Low OI
        if data.open_interest_usd < 5_000_000.0 {
            score += 20.0;
            findings.push("Low open interest (< $5M)".to_string());
        }

        score = score.min(100.0);

        let recommendation = if score > 50.0 {
            "AVOID: Insufficient liquidity".to_string()
        } else if score > 25.0 {
            "CAUTION: Reduced liquidity".to_string()
        } else {
            "OK: Liquidity is healthy".to_string()
        };

        RiskFactor {
            name: "Liquidity".into(),
            dimension: RiskDimension::Liquidity,
            score,
            weight: 0.10,
            finding: if findings.is_empty() {
                "Healthy liquidity metrics".to_string()
            } else {
                findings.join("; ")
            },
            recommendation,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn healthy_market_data() -> ComplianceMarketData {
        ComplianceMarketData {
            market: "BTC-USD".into(),
            exchange: "DYDX".into(),
            volume_24h_usd: 1_000_000_000.0,
            open_interest_usd: 500_000_000.0,
            spread_bps: 1.5,
            unique_traders_24h: 5000,
            top_trader_concentration: 0.1,
            exchange_jurisdiction: Jurisdiction::CaymanIslands,
            has_license: false,
            withdrawal_queue_usd: 50_000_000.0,
            security_heuristic: 0.1,
            audit_status: 2,
            governance_decentralization: 0.7,
        }
    }

    fn risky_market_data() -> ComplianceMarketData {
        ComplianceMarketData {
            market: "SHITCOIN-USD".into(),
            exchange: "SHADYDEX".into(),
            volume_24h_usd: 500_000.0,
            open_interest_usd: 2_000_000.0,
            spread_bps: 80.0,
            unique_traders_24h: 10,
            top_trader_concentration: 0.9,
            exchange_jurisdiction: Jurisdiction::Unregulated,
            has_license: false,
            withdrawal_queue_usd: 2_000_000_000.0,
            security_heuristic: 0.95,
            audit_status: 0,
            governance_decentralization: 0.05,
        }
    }

    #[test]
    fn test_healthy_market_passes() {
        let screener = ComplianceScreener::new();
        let report = screener.screen(&healthy_market_data());
        assert_eq!(report.market, "BTC-USD");
        assert!(report.composite_score < 30.0, "Score should be low for healthy market");
        assert_eq!(report.risk_level, ComplianceRiskLevel::Low);
        assert!(report.trade_permitted);
    }

    #[test]
    fn test_risky_market_blocked() {
        let screener = ComplianceScreener::new();
        let report = screener.screen(&risky_market_data());
        assert!(report.composite_score > 50.0, "Score should be high for risky market");
        assert!(!report.trade_permitted || report.risk_level == ComplianceRiskLevel::High);
    }

    #[test]
    fn test_risk_level_from_score() {
        assert_eq!(ComplianceRiskLevel::from_score(10.0), ComplianceRiskLevel::Low);
        assert_eq!(ComplianceRiskLevel::from_score(40.0), ComplianceRiskLevel::Medium);
        assert_eq!(ComplianceRiskLevel::from_score(65.0), ComplianceRiskLevel::High);
        assert_eq!(ComplianceRiskLevel::from_score(90.0), ComplianceRiskLevel::Critical);
    }

    #[test]
    fn test_jurisdiction_regulatory_weight() {
        assert!(Jurisdiction::UnitedStates.regulatory_weight() > Jurisdiction::Singapore.regulatory_weight());
        assert!(Jurisdiction::Unregulated.regulatory_weight() > Jurisdiction::EuropeanUnion.regulatory_weight());
    }

    #[test]
    fn test_compliance_report_has_all_factors() {
        let screener = ComplianceScreener::new();
        let report = screener.screen(&healthy_market_data());
        assert_eq!(report.factors.len(), 6);
        let dimensions: Vec<_> = report.factors.iter().map(|f| f.dimension).collect();
        assert!(dimensions.contains(&RiskDimension::MarketManipulation));
        assert!(dimensions.contains(&RiskDimension::Regulatory));
        assert!(dimensions.contains(&RiskDimension::Counterparty));
        assert!(dimensions.contains(&RiskDimension::Sanctions));
        assert!(dimensions.contains(&RiskDimension::TokenRisk));
        assert!(dimensions.contains(&RiskDimension::Liquidity));
    }

    #[test]
    fn test_compliance_report_serde() {
        let screener = ComplianceScreener::new();
        let report = screener.screen(&healthy_market_data());
        let json = serde_json::to_string(&report).unwrap();
        let restored: ComplianceReport = serde_json::from_str(&json).unwrap();
        assert_eq!(restored.market, "BTC-USD");
        assert_eq!(restored.risk_level, report.risk_level);
    }

    #[test]
    fn test_risk_factor_serde() {
        let factor = RiskFactor {
            name: "Test".into(),
            dimension: RiskDimension::MarketManipulation,
            score: 45.5,
            weight: 0.25,
            finding: "Test finding".into(),
            recommendation: "Test recommendation".into(),
        };
        let json = serde_json::to_string(&factor).unwrap();
        assert!(json.contains("MarketManipulation"));
    }

    #[test]
    fn test_manipulation_risk_low_volume() {
        let screener = ComplianceScreener::new();
        let data = healthy_market_data();
        let factor = screener.assess_manipulation_risk(&data);
        assert!(factor.score < 20.0, "Healthy market should have low manipulation score");
    }

    #[test]
    fn test_manipulation_risk_high_concentration() {
        let screener = ComplianceScreener::new();
        let data = risky_market_data();
        let factor = screener.assess_manipulation_risk(&data);
        assert!(factor.score > 30.0, "High concentration should elevate manipulation score");
    }

    #[test]
    fn test_config_blocking_level() {
        let config = ComplianceConfig {
            blocking_level: ComplianceRiskLevel::Critical,
            ..Default::default()
        };
        let screener = ComplianceScreener::with_config(config);
        let report = screener.screen(&risky_market_data());
        // Even with risky data, should be permitted because only Critical is blocked
        assert!(report.trade_permitted);
    }

    #[test]
    fn test_risk_level_color_hex() {
        assert_eq!(ComplianceRiskLevel::Low.color_hex(), "#27AE60");
        assert_eq!(ComplianceRiskLevel::Critical.color_hex(), "#E74C3C");
    }

    #[test]
    fn test_compliance_market_data_default() {
        let data = ComplianceMarketData::default();
        assert_eq!(data.exchange, "DYDX");
        assert_eq!(data.exchange_jurisdiction, Jurisdiction::Unregulated);
    }
}
