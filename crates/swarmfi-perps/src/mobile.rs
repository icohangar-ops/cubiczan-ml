//! Mobile API — REST API companion for React Native / mobile apps.
//!
//! Provides a structured API layer that serializes swarm analysis results,
//! consensus data, arbitrage opportunities, vault analytics, and compliance
//! reports into mobile-friendly JSON responses. Designed to be served via
//! an HTTP server (axum/actix) or consumed by a React Native app via fetch.
//!
//! # Architecture
//!
//! ```text
//! React Native App
//!       │
//!       ▼ GET /api/v1/consensus/BTC-USD
//! ┌──────────────────┐
//! │ MobileApiServer  │ (axum or actix, not included here)
//! └────────┬─────────┘
//!          │
//!          ▼
//! ┌──────────────────┐
//! │ MobileResponse   │ ← serialize into mobile-friendly format
//! │   .consensus()   │
//! │   .arbitrage()   │
//! │   .vault()       │
//! │   .compliance()  │
//! └──────────────────┘
//! ```

use crate::types::*;
use crate::alerts::{Alert, AlertPlatform};
use crate::arbitrage::CrossExchangeComparison;
use crate::backtest::BacktestReport;
use crate::compliance::ComplianceReport;
use crate::vault::VaultReport;
use serde::{Deserialize, Serialize};

/// API version prefix.
pub const API_VERSION: &str = "v1";

/// Standard API response envelope.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ApiResponse<T> {
    pub status: ApiStatus,
    pub data: Option<T>,
    pub error: Option<ApiError>,
    pub timestamp_ms: i64,
}

/// API response status.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum ApiStatus {
    Ok,
    Error,
}

/// API error details.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ApiError {
    pub code: String,
    pub message: String,
}

impl<T> ApiResponse<T> {
    /// Create a success response.
    pub fn ok(data: T) -> Self {
        Self {
            status: ApiStatus::Ok,
            data: Some(data),
            error: None,
            timestamp_ms: chrono::Utc::now().timestamp_millis(),
        }
    }

    /// Create an error response.
    pub fn error(code: &str, message: &str) -> Self {
        Self {
            status: ApiStatus::Error,
            data: None,
            error: Some(ApiError {
                code: code.to_string(),
                message: message.to_string(),
            }),
            timestamp_ms: chrono::Utc::now().timestamp_millis(),
        }
    }
}

/// Mobile-friendly consensus data response.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MobileConsensusResponse {
    pub market: String,
    pub signal: String,
    pub confidence: f64,
    pub agent_votes: Vec<MobileAgentVote>,
    pub volatility_regime: String,
    pub liquidation_risk: String,
    pub timestamp_ms: i64,
}

/// Mobile-friendly agent vote.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MobileAgentVote {
    pub agent: String,
    pub signal: String,
    pub confidence: f64,
}

/// Mobile-friendly arbitrage response.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MobileArbitrageResponse {
    pub base_asset: String,
    pub price_opportunities: Vec<MobileArbOpportunity>,
    pub funding_opportunities: Vec<MobileFundingArb>,
    pub consensus_signal: String,
    pub consensus_confidence: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MobileArbOpportunity {
    pub buy_exchange: String,
    pub sell_exchange: String,
    pub spread_pct: f64,
    pub profit_after_fees_bps: f64,
    pub strength: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MobileFundingArb {
    pub short_exchange: String,
    pub long_exchange: String,
    pub combined_yield_pct: f64,
    pub convergence_risk: String,
}

/// Mobile-friendly vault response.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MobileVaultResponse {
    pub vault_id: String,
    pub current_tvl: Option<f64>,
    pub yield_30d_apy: Option<f64>,
    pub is_outperforming: Option<bool>,
    pub max_drawdown_pct: f64,
    pub sharpe_ratio: f64,
    pub sortino_ratio: f64,
    pub current_drawdown_pct: f64,
}

/// Mobile-friendly compliance response.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MobileComplianceResponse {
    pub market: String,
    pub composite_score: f64,
    pub risk_level: String,
    pub trade_permitted: bool,
    pub factors: Vec<MobileRiskFactor>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MobileRiskFactor {
    pub name: String,
    pub score: f64,
    pub finding: String,
    pub recommendation: String,
}

/// Mobile-friendly alert response.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MobileAlertResponse {
    pub alerts: Vec<MobileAlert>,
    pub total: u32,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MobileAlert {
    pub id: String,
    pub severity: String,
    pub market: String,
    pub signal: String,
    pub confidence: f64,
    pub reason: String,
    pub platforms: Vec<String>,
    pub timestamp_ms: i64,
}

/// Mobile-friendly backtest response.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MobileBacktestResponse {
    pub market: String,
    pub total_trades: u32,
    pub win_rate: f64,
    pub total_return_pct: f64,
    pub sharpe_ratio: f64,
    pub max_drawdown_pct: f64,
    pub profit_factor: f64,
    pub avg_bars_held: f64,
}

/// Dashboard summary combining multiple data sources.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MobileDashboard {
    pub consensus: Vec<MobileConsensusResponse>,
    pub arbitrage: Vec<MobileArbitrageResponse>,
    pub vault: Option<MobileVaultResponse>,
    pub recent_alerts: Vec<MobileAlert>,
    pub timestamp_ms: i64,
}

/// Convert a consensus result to mobile format.
pub fn consensus_to_mobile(result: &ConsensusResult) -> MobileConsensusResponse {
    MobileConsensusResponse {
        market: result.market.clone(),
        signal: result.signal.as_str().to_string(),
        confidence: result.confidence,
        agent_votes: result
            .agent_votes
            .iter()
            .map(|v| MobileAgentVote {
                agent: v.agent_type.clone(),
                signal: v.signal.as_str().to_string(),
                confidence: v.confidence,
            })
            .collect(),
        volatility_regime: format!("{:?}", result.stigmergy_board.volatility_regime),
        liquidation_risk: format!("{:?}", result.stigmergy_board.liquidation_risk_level),
        timestamp_ms: result.timestamp,
    }
}

/// Convert a cross-exchange comparison to mobile format.
pub fn arbitrage_to_mobile(comparison: &CrossExchangeComparison) -> MobileArbitrageResponse {
    MobileArbitrageResponse {
        base_asset: comparison.base_asset.clone(),
        price_opportunities: comparison
            .price_opportunities
            .iter()
            .map(|o| MobileArbOpportunity {
                buy_exchange: o.buy_exchange.as_str().to_string(),
                sell_exchange: o.sell_exchange.as_str().to_string(),
                spread_pct: o.spread_pct,
                profit_after_fees_bps: o.profit_after_fees_bps,
                strength: format!("{:?}", o.strength),
            })
            .collect(),
        funding_opportunities: comparison
            .funding_opportunities
            .iter()
            .map(|f| MobileFundingArb {
                short_exchange: f.short_exchange.as_str().to_string(),
                long_exchange: f.long_exchange.as_str().to_string(),
                combined_yield_pct: f.combined_yield_pct,
                convergence_risk: format!("{:?}", f.convergence_risk),
            })
            .collect(),
        consensus_signal: comparison.consensus_signal.as_str().to_string(),
        consensus_confidence: comparison.consensus_confidence,
    }
}

/// Convert a vault report to mobile format.
pub fn vault_to_mobile(report: &VaultReport) -> MobileVaultResponse {
    MobileVaultResponse {
        vault_id: report.vault_id.clone(),
        current_tvl: report.current_snapshot.as_ref().map(|s| s.tvl_usd),
        yield_30d_apy: report.current_snapshot.as_ref().map(|s| s.yield_30d_apy),
        is_outperforming: report.yield_comparison.as_ref().map(|y| y.is_outperforming),
        max_drawdown_pct: report.drawdown_analysis.max_drawdown_pct,
        sharpe_ratio: report.risk_metrics.sharpe_ratio,
        sortino_ratio: report.risk_metrics.sortino_ratio,
        current_drawdown_pct: report.drawdown_analysis.current_drawdown_pct,
    }
}

/// Convert a compliance report to mobile format.
pub fn compliance_to_mobile(report: &ComplianceReport) -> MobileComplianceResponse {
    MobileComplianceResponse {
        market: report.market.clone(),
        composite_score: report.composite_score,
        risk_level: report.risk_level.as_str().to_string(),
        trade_permitted: report.trade_permitted,
        factors: report
            .factors
            .iter()
            .map(|f| MobileRiskFactor {
                name: f.name.clone(),
                score: f.score,
                finding: f.finding.clone(),
                recommendation: f.recommendation.clone(),
            })
            .collect(),
    }
}

/// Convert alerts to mobile format.
pub fn alerts_to_mobile(alerts: &[Alert]) -> MobileAlertResponse {
    MobileAlertResponse {
        total: alerts.len() as u32,
        alerts: alerts
            .iter()
            .map(|a| MobileAlert {
                id: a.id.clone(),
                severity: a.severity.as_str().to_string(),
                market: a.market.clone(),
                signal: a.signal.as_str().to_string(),
                confidence: a.confidence,
                reason: format!("{:?}", a.reason),
                platforms: a.platforms.iter().map(|p| p.as_str().to_string()).collect(),
                timestamp_ms: a.timestamp_ms,
            })
            .collect(),
    }
}

/// Convert a backtest report to mobile format.
pub fn backtest_to_mobile(report: &BacktestReport) -> MobileBacktestResponse {
    MobileBacktestResponse {
        market: report.market.clone(),
        total_trades: report.metrics.total_trades,
        win_rate: report.metrics.win_rate * 100.0,
        total_return_pct: report.metrics.total_return_pct,
        sharpe_ratio: report.metrics.sharpe_ratio,
        max_drawdown_pct: report.metrics.max_drawdown_pct,
        profit_factor: if report.metrics.profit_factor.is_finite() {
            report.metrics.profit_factor
        } else {
            0.0
        },
        avg_bars_held: report.metrics.avg_bars_held,
    }
}

/// Build a dashboard response from multiple data sources.
pub fn build_dashboard(
    consensus_results: &[ConsensusResult],
    arbitrage_comparisons: &[CrossExchangeComparison],
    vault_report: Option<&VaultReport>,
    recent_alerts: &[Alert],
) -> MobileDashboard {
    MobileDashboard {
        consensus: consensus_results.iter().map(consensus_to_mobile).collect(),
        arbitrage: arbitrage_comparisons.iter().map(arbitrage_to_mobile).collect(),
        vault: vault_report.map(vault_to_mobile),
        recent_alerts: alerts_to_mobile(recent_alerts).alerts,
        timestamp_ms: chrono::Utc::now().timestamp_millis(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn sample_consensus() -> ConsensusResult {
        let mut board = StigmergyBoard::default();
        board.volatility_regime = VolatilityRegime::Normal;
        board.liquidation_risk_level = RiskLevel::Low;
        ConsensusResult {
            market: "BTC-USD".into(),
            signal: Signal::Long,
            confidence: 72.0,
            agent_votes: vec![
                AgentVote {
                    agent_type: "FundingAgent".into(),
                    signal: Signal::Short,
                    confidence: 65.0,
                    reasoning: "High funding".into(),
                },
                AgentVote {
                    agent_type: "MomentumAgent".into(),
                    signal: Signal::Long,
                    confidence: 80.0,
                    reasoning: "Uptrend".into(),
                },
            ],
            timestamp: chrono::Utc::now().timestamp_millis(),
            stigmergy_board: board,
        }
    }

    #[test]
    fn test_api_response_ok() {
        let resp: ApiResponse<String> = ApiResponse::ok("hello".to_string());
        assert_eq!(resp.status, ApiStatus::Ok);
        assert_eq!(resp.data, Some("hello".to_string()));
        assert!(resp.error.is_none());
    }

    #[test]
    fn test_api_response_error() {
        let resp: ApiResponse<String> = ApiResponse::error("NOT_FOUND", "Market not found");
        assert_eq!(resp.status, ApiStatus::Error);
        assert!(resp.data.is_none());
        assert_eq!(resp.error.unwrap().code, "NOT_FOUND");
    }

    #[test]
    fn test_consensus_to_mobile() {
        let result = sample_consensus();
        let mobile = consensus_to_mobile(&result);
        assert_eq!(mobile.market, "BTC-USD");
        assert_eq!(mobile.signal, "LONG");
        assert_eq!(mobile.confidence, 72.0);
        assert_eq!(mobile.agent_votes.len(), 2);
    }

    #[test]
    fn test_vault_to_mobile_with_snapshot() {
        let report = VaultReport {
            vault_id: "mega-1".into(),
            current_snapshot: Some(crate::vault::VaultSnapshot {
                vault_id: "mega-1".into(),
                tvl_usd: 50_000_000.0,
                nav_per_share: 1.1,
                cumulative_pnl: 5_000_000.0,
                pnl_delta: 50_000.0,
                yield_24h_pct: 0.1,
                yield_7d_apy: 30.0,
                yield_30d_apy: 25.0,
                yield_inception_apy: 20.0,
                open_positions: 12,
                commission_earned: 250_000.0,
                timestamp_ms: 0,
            }),
            composition: None,
            yield_comparison: Some(crate::vault::YieldComparison {
                vault_id: "mega-1".into(),
                vault_30d_apy: 25.0,
                vault_7d_apy: 30.0,
                btc_benchmark_apy: 15.0,
                eth_benchmark_apy: 10.0,
                risk_free_apy: 5.0,
                excess_return_30d: 20.0,
                sharpe_like: 2.0,
                sortino_like: 3.0,
                is_outperforming: true,
            }),
            nav_history: vec![],
            drawdown_analysis: crate::vault::DrawdownAnalysis::zero(),
            risk_metrics: crate::vault::VaultRiskMetrics::zero(),
        };
        let mobile = vault_to_mobile(&report);
        assert_eq!(mobile.vault_id, "mega-1");
        assert_eq!(mobile.current_tvl, Some(50_000_000.0));
        assert_eq!(mobile.yield_30d_apy, Some(25.0));
        assert!(mobile.is_outperforming.unwrap());
    }

    #[test]
    fn test_vault_to_mobile_no_snapshot() {
        let report = VaultReport {
            vault_id: "test".into(),
            current_snapshot: None,
            composition: None,
            yield_comparison: None,
            nav_history: vec![],
            drawdown_analysis: crate::vault::DrawdownAnalysis::zero(),
            risk_metrics: crate::vault::VaultRiskMetrics::zero(),
        };
        let mobile = vault_to_mobile(&report);
        assert_eq!(mobile.current_tvl, None);
    }

    #[test]
    fn test_compliance_to_mobile() {
        use crate::compliance::{ComplianceRiskLevel, ComplianceReport, RiskFactor, RiskDimension};
        let report = ComplianceReport {
            market: "BTC-USD".into(),
            exchange: "DYDX".into(),
            composite_score: 15.0,
            risk_level: ComplianceRiskLevel::Low,
            factors: vec![crate::compliance::RiskFactor {
                name: "Test".into(),
                dimension: crate::compliance::RiskDimension::MarketManipulation,
                score: 10.0,
                weight: 0.25,
                finding: "OK".into(),
                recommendation: "OK".into(),
            }],
            trade_permitted: true,
            blocking_threshold: ComplianceRiskLevel::High,
            timestamp_ms: 0,
            max_holding_days: 0,
        };
        let mobile = compliance_to_mobile(&report);
        assert_eq!(mobile.composite_score, 15.0);
        assert_eq!(mobile.risk_level, "LOW");
        assert!(mobile.trade_permitted);
    }

    #[test]
    fn test_alerts_to_mobile() {
        let alert = Alert::new(
            crate::alerts::AlertSeverity::Warning,
            crate::alerts::AlertReason::HighConfidence { value: 80.0, threshold: 75.0 },
            "BTC-USD",
            Signal::Long,
            80.0,
        );
        let response = alerts_to_mobile(&[alert]);
        assert_eq!(response.total, 1);
        assert_eq!(response.alerts[0].severity, "WARNING");
        assert_eq!(response.alerts[0].market, "BTC-USD");
    }

    #[test]
    fn test_backtest_to_mobile() {
        let report = BacktestReport {
            market: "BTC-USD".into(),
            total_candles: 100,
            metrics: crate::backtest::PerformanceMetrics {
                total_return_pct: 15.5,
                annualized_return_pct: 15.5,
                max_drawdown_pct: 3.2,
                sharpe_ratio: 1.8,
                sortino_ratio: 2.3,
                win_rate: 0.65,
                total_trades: 20,
                winning_trades: 13,
                losing_trades: 7,
                avg_win_pct: 2.1,
                avg_loss_pct: -1.0,
                profit_factor: 3.9,
                avg_bars_held: 5.5,
                long_return_pct: 12.0,
                short_return_pct: 3.5,
            },
            equity_curve: vec![],
            trades: vec![],
            signal_breakdown: vec![],
        };
        let mobile = backtest_to_mobile(&report);
        assert_eq!(mobile.market, "BTC-USD");
        assert_eq!(mobile.win_rate, 65.0);
        assert_eq!(mobile.total_return_pct, 15.5);
        assert_eq!(mobile.sharpe_ratio, 1.8);
    }

    #[test]
    fn test_api_version() {
        assert_eq!(API_VERSION, "v1");
    }

    #[test]
    fn test_api_response_serde() {
        let resp: ApiResponse<String> = ApiResponse::ok("data".to_string());
        let json = serde_json::to_string(&resp).unwrap();
        assert!(json.contains("\"status\":\"Ok\""));
    }

    #[test]
    fn test_build_dashboard() {
        let consensus = vec![sample_consensus()];
        let alerts = vec![];
        let dashboard = build_dashboard(&consensus, &[], None, &alerts);
        assert_eq!(dashboard.consensus.len(), 1);
        assert_eq!(dashboard.arbitrage.len(), 0);
        assert!(dashboard.vault.is_none());
    }

    #[test]
    fn test_mobile_consensus_serde() {
        let result = sample_consensus();
        let mobile = consensus_to_mobile(&result);
        let json = serde_json::to_string(&mobile).unwrap();
        let restored: MobileConsensusResponse = serde_json::from_str(&json).unwrap();
        assert_eq!(restored.market, "BTC-USD");
        assert_eq!(restored.signal, "LONG");
    }
}
