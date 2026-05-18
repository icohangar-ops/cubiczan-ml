//! Adversarial weighted voting consensus mechanism.
//!
//! Implements weighted scoring per agent type, adversarial confidence reduction
//! when signals are evenly split, and stigmergy board state management.

use crate::math::clamp;
use crate::types::*;

use std::collections::HashMap;

/// Agent weights reflecting reliability and information quality in perp markets.
pub fn agent_weights() -> HashMap<&'static str, f64> {
    let mut w = HashMap::new();
    w.insert("FundingAgent", 1.3);
    w.insert("MomentumAgent", 1.1);
    w.insert("VolatilityAgent", 0.8);
    w.insert("VolumeAgent", 1.2);
    w.insert("OrderbookAgent", 1.0);
    w.insert("LiquidationAgent", 1.4);
    w.insert("MeanReversionAgent", 0.9);
    w.insert("TrendAgent", 1.1);
    w.insert("SentimentAgent", 1.0);
    w
}

/// Weight configuration with description.
pub fn agent_weight_descriptions() -> Vec<AgentWeight> {
    vec![
        AgentWeight { agent_type: "FundingAgent".into(), weight: 1.3, description: "Funding rates are the strongest signal for perp market positioning".into() },
        AgentWeight { agent_type: "MomentumAgent".into(), weight: 1.1, description: "Price momentum provides reliable short-term direction".into() },
        AgentWeight { agent_type: "VolatilityAgent".into(), weight: 0.8, description: "Volatility is more of a filter than a directional signal".into() },
        AgentWeight { agent_type: "VolumeAgent".into(), weight: 1.2, description: "Volume confirms or denies the strength of moves".into() },
        AgentWeight { agent_type: "OrderbookAgent".into(), weight: 1.0, description: "Orderbook shows immediate supply/demand but can be spoofed".into() },
        AgentWeight { agent_type: "LiquidationAgent".into(), weight: 1.4, description: "Liquidation cascades create some of the strongest perp signals".into() },
        AgentWeight { agent_type: "MeanReversionAgent".into(), weight: 0.9, description: "Mean reversion works well in ranging markets".into() },
        AgentWeight { agent_type: "TrendAgent".into(), weight: 1.1, description: "Multi-timeframe trend alignment is a strong confirmation signal".into() },
        AgentWeight { agent_type: "SentimentAgent".into(), weight: 1.0, description: "Meta-agent synthesizes other agents' signals".into() },
    ]
}

/// Result of the consensus computation (without market context).
pub struct ConsensusOutput {
    pub signal: Signal,
    pub confidence: f64,
    pub stigmergy_board: StigmergyBoard,
}

/// Compute weighted consensus from a collection of agent votes.
///
/// Algorithm:
///  1. Weight each agent vote by its configured weight
///  2. Sum LONG, SHORT, NEUTRAL weights
///  3. Determine signal by weighted majority
///  4. Calculate confidence = (winning - losing) / total * 100
///  5. If adversarial split (evenly divided), reduce confidence
///  6. Update stigmergy board with results
pub fn compute_consensus(votes: &[AgentVote], previous_board: Option<&StigmergyBoard>) -> ConsensusOutput {
    let weights = agent_weights();
    let default_weight = 1.0;

    let mut long_weight = 0.0;
    let mut short_weight = 0.0;
    let mut neutral_weight = 0.0;
    let mut long_count = 0u32;
    let mut short_count = 0u32;
    let mut neutral_count = 0u32;
    let mut total_confidence = 0.0;

    let mut last_signals: HashMap<String, String> = HashMap::new();

    for vote in votes {
        let w = weights.get(vote.agent_type.as_str()).copied().unwrap_or(default_weight);
        let confidence_scaled_weight = w * (vote.confidence / 100.0);

        last_signals.insert(vote.agent_type.clone(), vote.signal.as_str().to_string());
        total_confidence += vote.confidence;

        match vote.signal {
            Signal::Long => { long_weight += confidence_scaled_weight; long_count += 1; }
            Signal::Short => { short_weight += confidence_scaled_weight; short_count += 1; }
            Signal::Neutral => { neutral_weight += confidence_scaled_weight; neutral_count += 1; }
        }
    }

    let total_weight = long_weight + short_weight + neutral_weight;

    // Determine winning signal
    let signal = if long_weight > short_weight && long_weight > neutral_weight {
        Signal::Long
    } else if short_weight > long_weight && short_weight > neutral_weight {
        Signal::Short
    } else {
        Signal::Neutral
    };

    // Calculate confidence
    let confidence = if signal == Signal::Neutral {
        if total_weight > 0.0 { (neutral_weight / total_weight) * 60.0 } else { 20.0 }
    } else {
        let winning_weight = if signal == Signal::Long { long_weight } else { short_weight };
        let losing_weight = if signal == Signal::Long { short_weight } else { long_weight };
        if total_weight > 0.0 {
            ((winning_weight - losing_weight) / total_weight) * 100.0
        } else {
            20.0
        }
    };

    // Adversarial check — penalize when signals are evenly split
    let non_neutral_agents = long_count + short_count;
    let balance_ratio = if non_neutral_agents > 0 {
        (long_count as i32 - short_count as i32).abs() as f64 / non_neutral_agents as f64
    } else {
        0.0
    };

    let confidence = if balance_ratio < 0.2 && non_neutral_agents >= 4 {
        confidence * 0.5  // Highly adversarial — halve confidence
    } else if balance_ratio < 0.35 && non_neutral_agents >= 3 {
        confidence * 0.7  // Somewhat adversarial
    } else {
        confidence
    };

    // Confidence bounds [10, 90]
    let confidence = clamp(confidence, 10.0, 90.0);

    // Volatility & liquidation risk assessment
    let mut volatility_regime = VolatilityRegime::Normal;
    let mut liquidation_risk_level = RiskLevel::Low;

    if let Some(vol_agent) = votes.iter().find(|v| v.agent_type == "VolatilityAgent") {
        volatility_regime = if vol_agent.confidence > 65.0 && vol_agent.signal == Signal::Neutral {
            VolatilityRegime::High
        } else if vol_agent.confidence > 50.0 {
            VolatilityRegime::Normal
        } else {
            VolatilityRegime::Low
        };
    }

    if let Some(liq_agent) = votes.iter().find(|v| v.agent_type == "LiquidationAgent") {
        liquidation_risk_level = if liq_agent.confidence > 65.0 {
            RiskLevel::High
        } else if liq_agent.confidence > 45.0 {
            RiskLevel::Medium
        } else {
            RiskLevel::Low
        };
    }

    // Build stigmergy board
    let mut stigmergy_board = StigmergyBoard {
        last_signals,
        signal_counts: SignalCounts {
            long: long_count,
            short: short_count,
            neutral: neutral_count,
        },
        average_confidence: if !votes.is_empty() { total_confidence / votes.len() as f64 } else { 0.0 },
        last_updated: chrono::Utc::now().timestamp_millis(),
        liquidation_risk_level,
        volatility_regime,
        previous_signals: None,
    };

    // Carry forward previous signals if available
    if let Some(prev) = previous_board {
        stigmergy_board.previous_signals = Some(prev.last_signals.clone());
    }

    ConsensusOutput { signal, confidence, stigmergy_board }
}

/// Full consensus pipeline — runs consensus and wraps result with market context.
pub fn run_consensus(
    votes: Vec<AgentVote>,
    market: &str,
    previous_board: Option<&StigmergyBoard>,
) -> ConsensusResult {
    let output = compute_consensus(&votes, previous_board);

    ConsensusResult {
        market: market.to_string(),
        signal: output.signal,
        confidence: output.confidence,
        agent_votes: votes,
        timestamp: chrono::Utc::now().timestamp_millis(),
        stigmergy_board: output.stigmergy_board,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_votes(signals: &[Signal]) -> Vec<AgentVote> {
        crate::agents::CORE_AGENTS.iter().enumerate().map(|(i, name)| AgentVote {
            agent_type: name.to_string(),
            signal: signals.get(i).copied().unwrap_or(Signal::Neutral),
            confidence: 60.0,
            reasoning: "test".into(),
        }).collect()
    }

    #[test]
    fn test_unanimous_long() {
        let signals = [Signal::Long; 8];
        let votes = make_votes(&signals);
        let result = compute_consensus(&votes, None);
        assert_eq!(result.signal, Signal::Long);
        assert!(result.confidence > 50.0);
    }

    #[test]
    fn test_unanimous_short() {
        let signals = [Signal::Short; 8];
        let votes = make_votes(&signals);
        let result = compute_consensus(&votes, None);
        assert_eq!(result.signal, Signal::Short);
        assert!(result.confidence > 50.0);
    }

    #[test]
    fn test_even_split_adversarial() {
        let signals = [
            Signal::Long, Signal::Long, Signal::Long, Signal::Long,
            Signal::Short, Signal::Short, Signal::Short, Signal::Short,
        ];
        let votes = make_votes(&signals);
        let result = compute_consensus(&votes, None);
        // Even split → confidence should be heavily penalized
        assert!(result.confidence < 40.0, "Expected adversarial penalty, got confidence={}", result.confidence);
    }

    #[test]
    fn test_mostly_neutral() {
        let signals = [
            Signal::Neutral, Signal::Neutral, Signal::Neutral, Signal::Neutral,
            Signal::Neutral, Signal::Long, Signal::Short, Signal::Neutral,
        ];
        let votes = make_votes(&signals);
        let result = compute_consensus(&votes, None);
        assert_eq!(result.signal, Signal::Neutral);
    }

    #[test]
    fn test_confidence_clamped() {
        let votes = vec![
            AgentVote { agent_type: "LiquidationAgent".into(), signal: Signal::Long, confidence: 100.0, reasoning: "".into() },
            AgentVote { agent_type: "FundingAgent".into(), signal: Signal::Long, confidence: 100.0, reasoning: "".into() },
            AgentVote { agent_type: "VolumeAgent".into(), signal: Signal::Long, confidence: 100.0, reasoning: "".into() },
        ];
        let result = compute_consensus(&votes, None);
        assert!(result.confidence <= 90.0);
        assert!(result.confidence >= 10.0);
    }

    #[test]
    fn test_stigmergy_board_counts() {
        let signals = [
            Signal::Long, Signal::Long, Signal::Short, Signal::Neutral,
            Signal::Long, Signal::Short, Signal::Short, Signal::Neutral,
        ];
        let votes = make_votes(&signals);
        let result = compute_consensus(&votes, None);
        assert_eq!(result.stigmergy_board.signal_counts.long, 3);
        assert_eq!(result.stigmergy_board.signal_counts.short, 3);
        assert_eq!(result.stigmergy_board.signal_counts.neutral, 2);
    }

    #[test]
    fn test_volatility_risk_assessment() {
        let votes = vec![
            AgentVote { agent_type: "VolatilityAgent".into(), signal: Signal::Neutral, confidence: 70.0, reasoning: "".into() },
            AgentVote { agent_type: "LiquidationAgent".into(), signal: Signal::Short, confidence: 75.0, reasoning: "".into() },
        ];
        let result = compute_consensus(&votes, None);
        assert_eq!(result.stigmergy_board.volatility_regime, VolatilityRegime::High);
        assert_eq!(result.stigmergy_board.liquidation_risk_level, RiskLevel::High);
    }

    #[test]
    fn test_previous_board_carry_forward() {
        let mut prev = StigmergyBoard::default();
        prev.last_signals.insert("FundingAgent".into(), "SHORT".into());
        prev.last_signals.insert("MomentumAgent".into(), "LONG".into());

        let signals = [Signal::Long; 8];
        let votes = make_votes(&signals);
        let result = compute_consensus(&votes, Some(&prev));
        assert!(result.stigmergy_board.previous_signals.is_some());
        let prev_signals = result.stigmergy_board.previous_signals.unwrap();
        assert_eq!(prev_signals.get("FundingAgent").unwrap(), "SHORT");
    }

    #[test]
    fn test_run_consensus_full() {
        let votes = make_votes(&[Signal::Long; 8]);
        let result = run_consensus(votes, "BTC-USD", None);
        assert_eq!(result.market, "BTC-USD");
        assert_eq!(result.signal, Signal::Long);
        assert!(result.timestamp > 0);
        assert_eq!(result.agent_votes.len(), 8);
    }

    #[test]
    fn test_agent_weights_values() {
        let w = agent_weights();
        assert_eq!(w.get("LiquidationAgent"), Some(&1.4));
        assert_eq!(w.get("FundingAgent"), Some(&1.3));
        assert_eq!(w.get("VolatilityAgent"), Some(&0.8));
        assert_eq!(w.len(), 9);
    }
}
