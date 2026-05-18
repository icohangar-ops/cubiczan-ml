use crate::predictions::LineComparison;
use crate::types::{BettingLine, GamePrediction};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

/// Betting analysis engine for value detection and bankroll management.
#[derive(Debug, Clone)]
pub struct BettingAnalyzer {
    bankroll: f64,
    kelly_fraction: f64,
    min_edge_threshold: f64,
    bet_history: Vec<BetRecord>,
}

/// Historical bet record.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BetRecord {
    pub bet_id: String,
    pub game_id: String,
    pub bet_type: BetType,
    pub side: String,      // "home", "away", "over", "under"
    pub line: f64,
    pub odds: f64,
    pub wager: f64,
    pub result: BetResult,
    pub payout: f64,
    pub model_prediction: f64,
    pub edge: f64,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum BetType {
    Spread,
    Moneyline,
    OverUnder,
    Parlay,
    Prop,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum BetResult {
    Win,
    Loss,
    Push,
    Pending,
}

impl BettingAnalyzer {
    pub fn new(bankroll: f64) -> Self {
        Self {
            bankroll: bankroll.max(0.0),
            kelly_fraction: 0.25, // Quarter Kelly (conservative)
            min_edge_threshold: 2.0,
            bet_history: Vec::new(),
        }
    }

    pub fn with_kelly(fraction: f64, bankroll: f64) -> Self {
        Self {
            bankroll: bankroll.max(0.0),
            kelly_fraction: fraction.clamp(0.01, 0.5),
            min_edge_threshold: 2.0,
            bet_history: Vec::new(),
        }
    }

    pub fn with_min_edge(edge: f64, bankroll: f64) -> Self {
        Self {
            bankroll: bankroll.max(0.0),
            kelly_fraction: 0.25,
            min_edge_threshold: edge,
            bet_history: Vec::new(),
        }
    }

    // ─── Kelly Criterion ───────────────────────────────────────

    /// Full Kelly criterion: f* = (bp - q) / b
    /// where b = decimal odds - 1, p = win probability, q = 1 - p
    pub fn kelly_criterion(win_prob: f64, decimal_odds: f64) -> f64 {
        let b = decimal_odds - 1.0;
        if b < 1e-9 {
            return 0.0;
        }
        let p = win_prob.clamp(0.0, 1.0);
        let q = 1.0 - p;
        let kelly = (b * p - q) / b;
        kelly.max(0.0)
    }

    /// Fractional Kelly based on configured fraction.
    pub fn fractional_kelly(&self, win_prob: f64, decimal_odds: f64) -> f64 {
        let full = Self::kelly_criterion(win_prob, decimal_odds);
        full * self.kelly_fraction
    }

    /// Calculate recommended bet size in dollars.
    pub fn recommended_wager(&self, win_prob: f64, decimal_odds: f64) -> f64 {
        let frac = self.fractional_kelly(win_prob, decimal_odds);
        (self.bankroll * frac).round2()
    }

    // ─── Value Detection ───────────────────────────────────────

    /// Detect value in spread betting.
    pub fn spread_value(&self, prediction: &GamePrediction, line: &BettingLine) -> ValueAssessment {
        let edge = prediction.predicted_spread - line.spread;
        let has_value = edge.abs() >= self.min_edge_threshold;

        // edge < 0: model spread more negative than line → home more favored → BetHome
        // edge > 0: model spread more positive than line → away more favored → BetAway
        let recommendation = if !has_value {
            BetRecommendation::NoBet
        } else if edge < 0.0 {
            BetRecommendation::BetHome
        } else {
            BetRecommendation::BetAway
        };

        ValueAssessment {
            bet_type: BetType::Spread,
            edge,
            has_value,
            recommendation,
            confidence: prediction.confidence,
            model_line: prediction.predicted_spread,
            market_line: line.spread,
        }
    }

    /// Detect value in over/under betting.
    pub fn total_value(&self, prediction: &GamePrediction, line: &BettingLine) -> ValueAssessment {
        let edge = prediction.predicted_total - line.over_under;
        let has_value = edge.abs() >= self.min_edge_threshold;

        let recommendation = if !has_value {
            BetRecommendation::NoBet
        } else if edge > 0.0 {
            BetRecommendation::BetOver
        } else {
            BetRecommendation::BetUnder
        };

        ValueAssessment {
            bet_type: BetType::OverUnder,
            edge,
            has_value,
            recommendation,
            confidence: prediction.confidence,
            model_line: prediction.predicted_total,
            market_line: line.over_under,
        }
    }

    /// Detect value in moneyline betting.
    pub fn moneyline_value(&self, win_prob: f64, line: &BettingLine) -> ValueAssessment {
        let implied_home = BettingLine::american_to_implied(line.moneyline_home);
        let implied_away = BettingLine::american_to_implied(line.moneyline_away);

        // Remove vig for fair comparison
        let vig = implied_home + implied_away - 1.0;
        let fair_home = implied_home / (1.0 + vig);

        let edge_home = (win_prob - fair_home) * 100.0;
        let edge_away = -edge_home;

        let (recommendation, edge) = if edge_home.abs() >= edge_away.abs() {
            if edge_home > 5.0 {
                (BetRecommendation::BetHome, edge_home)
            } else {
                (BetRecommendation::NoBet, edge_home)
            }
        } else if edge_away > 5.0 {
            (BetRecommendation::BetAway, edge_away)
        } else {
            (BetRecommendation::NoBet, edge_away)
        };

        ValueAssessment {
            bet_type: BetType::Moneyline,
            edge,
            has_value: edge.abs() >= 5.0,
            recommendation,
            confidence: edge.abs() / 100.0,
            model_line: win_prob * 100.0,
            market_line: fair_home * 100.0,
        }
    }

    // ─── Line Movement ─────────────────────────────────────────

    /// Track line movement and categorize it.
    pub fn analyze_line_movement(opening: &BettingLine, closing: &BettingLine) -> LineMovement {
        let spread_move = closing.spread - opening.spread;
        let total_move = closing.over_under - opening.over_under;

        let spread_direction = if spread_move.abs() < 0.5 {
            LineDirection::Stable
        } else if spread_move > 0.0 {
            LineDirection::MovedUp
        } else {
            LineDirection::MovedDown
        };

        let total_direction = if total_move.abs() < 0.5 {
            LineDirection::Stable
        } else if total_move > 0.0 {
            LineDirection::MovedUp
        } else {
            LineDirection::MovedDown
        };

        let significance = if spread_move.abs() >= 3.0 || total_move.abs() >= 5.0 {
            MovementSignificance::High
        } else if spread_move.abs() >= 1.5 || total_move.abs() >= 2.5 {
            MovementSignificance::Moderate
        } else {
            MovementSignificance::Low
        };

        LineMovement {
            spread_change: spread_move,
            total_change: total_move,
            spread_direction,
            total_direction,
            significance,
        }
    }

    // ─── Bet Tracking ──────────────────────────────────────────

    /// Place a bet and record it.
    pub fn place_bet(
        &mut self,
        game_id: &str,
        bet_type: BetType,
        side: &str,
        line: f64,
        odds: f64,
        wager: f64,
        model_prediction: f64,
    ) -> &BetRecord {
        let edge = (model_prediction - line).abs();
        let record = BetRecord {
            bet_id: format!("bet_{}", self.bet_history.len()),
            game_id: game_id.to_string(),
            bet_type,
            side: side.to_string(),
            line,
            odds,
            wager,
            result: BetResult::Pending,
            payout: 0.0,
            model_prediction,
            edge,
        };
        self.bet_history.push(record);
        self.bet_history.last().unwrap()
    }

    /// Record a result for a bet.
    pub fn settle_bet(&mut self, bet_id: &str, result: BetResult) -> Option<&BetRecord> {
        if let Some(bet) = self.bet_history.iter_mut().find(|b| b.bet_id == bet_id) {
            bet.result = result;
            bet.payout = match result {
                BetResult::Win => {
                    if bet.odds < 0.0 {
                        bet.wager * (1.0 + 100.0 / bet.odds.abs())
                    } else {
                        bet.wager * (1.0 + bet.odds / 100.0)
                    }
                }
                BetResult::Loss => 0.0,
                BetResult::Push => bet.wager,
                BetResult::Pending => 0.0,
            };
            match result {
                BetResult::Win => self.bankroll += bet.payout - bet.wager,
                BetResult::Loss => self.bankroll -= bet.wager,
                _ => {}
            }
            return self.bet_history.iter().find(|b| b.bet_id == bet_id);
        }
        None
    }

    // ─── Performance Metrics ───────────────────────────────────

    /// Calculate overall betting performance.
    pub fn performance_summary(&self) -> BettingPerformance {
        let settled: Vec<&BetRecord> = self
            .bet_history
            .iter()
            .filter(|b| b.result != BetResult::Pending)
            .collect();

        let wins = settled.iter().filter(|b| b.result == BetResult::Win).count();
        let losses = settled.iter().filter(|b| b.result == BetResult::Loss).count();
        let pushes = settled.iter().filter(|b| b.result == BetResult::Push).count();
        let total = wins + losses;

        let win_rate = if total > 0 { wins as f64 / total as f64 } else { 0.0 };

        let total_wagered: f64 = settled.iter().map(|b| b.wager).sum();
        let total_payout: f64 = settled.iter().map(|b| b.payout).sum();
        let profit = total_payout - total_wagered;
        let roi = if total_wagered > 0.0 {
            profit / total_wagered * 100.0
        } else {
            0.0
        };

        let avg_edge: f64 = if !settled.is_empty() {
            settled.iter().map(|b| b.edge).sum::<f64>() / settled.len() as f64
        } else {
            0.0
        };

        // Max drawdown
        let mut peak = self.bankroll;
        let mut max_dd = 0.0;
        for bet in &self.bet_history {
            match bet.result {
                BetResult::Win => peak += bet.payout - bet.wager,
                BetResult::Loss => peak -= bet.wager,
                _ => {}
            }
            let dd = self.bankroll - peak;
            if dd < max_dd {
                max_dd = dd;
            }
        }

        BettingPerformance {
            total_bets: settled.len(),
            wins,
            losses,
            pushes,
            win_rate,
            total_wagered,
            total_payout,
            profit,
            roi,
            avg_edge,
            current_bankroll: self.bankroll,
            max_drawdown: max_dd,
        }
    }

    /// Get all bet history.
    pub fn bet_history(&self) -> &[BetRecord] {
        &self.bet_history
    }

    /// Current bankroll.
    pub fn bankroll(&self) -> f64 {
        self.bankroll
    }
}

/// Value assessment for a betting opportunity.
#[derive(Debug, Clone)]
pub struct ValueAssessment {
    pub bet_type: BetType,
    pub edge: f64,
    pub has_value: bool,
    pub recommendation: BetRecommendation,
    pub confidence: f64,
    pub model_line: f64,
    pub market_line: f64,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BetRecommendation {
    BetHome,
    BetAway,
    BetOver,
    BetUnder,
    NoBet,
}

/// Line movement analysis.
#[derive(Debug, Clone)]
pub struct LineMovement {
    pub spread_change: f64,
    pub total_change: f64,
    pub spread_direction: LineDirection,
    pub total_direction: LineDirection,
    pub significance: MovementSignificance,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum LineDirection {
    MovedUp,
    MovedDown,
    Stable,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MovementSignificance {
    Low,
    Moderate,
    High,
}

/// Overall betting performance summary.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BettingPerformance {
    pub total_bets: usize,
    pub wins: usize,
    pub losses: usize,
    pub pushes: usize,
    pub win_rate: f64,
    pub total_wagered: f64,
    pub total_payout: f64,
    pub profit: f64,
    pub roi: f64,
    pub avg_edge: f64,
    pub current_bankroll: f64,
    pub max_drawdown: f64,
}

trait RoundTo2 {
    fn round2(self) -> f64;
}

impl RoundTo2 for f64 {
    fn round2(self) -> f64 {
        (self * 100.0).round() / 100.0
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_prediction(spread: f64, total: f64) -> GamePrediction {
        GamePrediction {
            game_id: "g1".into(),
            home_team: "LAL".into(),
            away_team: "BOS".into(),
            predicted_winner: "LAL".into(),
            confidence: 0.7,
            predicted_spread: spread,
            predicted_total: total,
            reasoning: "test".into(),
        }
    }

    fn make_line(spread: f64, total: f64) -> BettingLine {
        BettingLine {
            game_id: "g1".into(),
            spread,
            over_under: total,
            moneyline_home: -150.0,
            moneyline_away: 130.0,
            implied_prob_home: 0.6,
            implied_prob_away: 0.435,
        }
    }

    #[test]
    fn test_kelly_criterion_positive() {
        let k = BettingAnalyzer::kelly_criterion(0.6, 2.0); // 60% win, even money decimal
        // f* = (1*0.6 - 0.4) / 1 = 0.2
        assert!((k - 0.2).abs() < 1e-9);
    }

    #[test]
    fn test_kelly_criterion_negative() {
        let k = BettingAnalyzer::kelly_criterion(0.4, 2.0);
        // f* = (1*0.4 - 0.6) / 1 = -0.2 -> clamped to 0
        assert_eq!(k, 0.0);
    }

    #[test]
    fn test_kelly_criterion_even_odds() {
        let k = BettingAnalyzer::kelly_criterion(0.5, 2.0);
        // f* = (1*0.5 - 0.5) / 1 = 0
        assert_eq!(k, 0.0);
    }

    #[test]
    fn test_fractional_kelly() {
        let ba = BettingAnalyzer::with_kelly(0.25, 1000.0);
        let frac = ba.fractional_kelly(0.6, 2.0);
        // full = 0.2, quarter = 0.05
        assert!((frac - 0.05).abs() < 1e-9);
    }

    #[test]
    fn test_recommended_wager() {
        let ba = BettingAnalyzer::with_kelly(0.25, 1000.0);
        let wager = ba.recommended_wager(0.6, 2.0);
        // 1000 * 0.05 = 50
        assert_eq!(wager, 50.0);
    }

    #[test]
    fn test_recommended_wager_zero_bankroll() {
        let ba = BettingAnalyzer::with_kelly(0.25, 0.0);
        let wager = ba.recommended_wager(0.6, 2.0);
        assert_eq!(wager, 0.0);
    }

    #[test]
    fn test_spread_value_detected() {
        let ba = BettingAnalyzer::new(1000.0);
        let pred = make_prediction(-8.0, 215.0);
        let line = make_line(-3.0, 215.0); // 5-point edge
        let val = ba.spread_value(&pred, &line);
        assert!(val.has_value);
        assert_eq!(val.recommendation, BetRecommendation::BetHome);
    }

    #[test]
    fn test_spread_value_away() {
        let ba = BettingAnalyzer::new(1000.0);
        let pred = make_prediction(2.0, 215.0);
        let line = make_line(-5.0, 215.0); // Model says 2, line says -5, away has value
        let val = ba.spread_value(&pred, &line);
        assert!(val.has_value);
        assert_eq!(val.recommendation, BetRecommendation::BetAway);
    }

    #[test]
    fn test_spread_no_value() {
        let ba = BettingAnalyzer::new(1000.0);
        let pred = make_prediction(-5.0, 215.0);
        let line = make_line(-5.0, 215.0);
        let val = ba.spread_value(&pred, &line);
        assert!(!val.has_value);
        assert_eq!(val.recommendation, BetRecommendation::NoBet);
    }

    #[test]
    fn test_total_value_over() {
        let ba = BettingAnalyzer::new(1000.0);
        let pred = make_prediction(-3.0, 225.0);
        let line = make_line(-3.0, 218.0);
        let val = ba.total_value(&pred, &line);
        assert!(val.has_value);
        assert_eq!(val.recommendation, BetRecommendation::BetOver);
    }

    #[test]
    fn test_total_value_under() {
        let ba = BettingAnalyzer::new(1000.0);
        let pred = make_prediction(-3.0, 210.0);
        let line = make_line(-3.0, 218.0);
        let val = ba.total_value(&pred, &line);
        assert!(val.has_value);
        assert_eq!(val.recommendation, BetRecommendation::BetUnder);
    }

    #[test]
    fn test_moneyline_value() {
        let ba = BettingAnalyzer::new(1000.0);
        let line = make_line(-3.0, 215.0);
        let val = ba.moneyline_value(0.7, &line);
        // 70% model prob vs 60% implied (minus vig)
        // Some edge should exist
        assert!(val.edge > 0.0);
    }

    #[test]
    fn test_line_movement_stable() {
        let opening = make_line(-3.0, 215.0);
        let closing = make_line(-3.0, 215.0);
        let mv = BettingAnalyzer::analyze_line_movement(&opening, &closing);
        assert_eq!(mv.spread_direction, LineDirection::Stable);
        assert_eq!(mv.total_direction, LineDirection::Stable);
        assert_eq!(mv.significance, MovementSignificance::Low);
    }

    #[test]
    fn test_line_movement_significant() {
        let opening = make_line(-3.0, 215.0);
        let closing = make_line(-7.0, 222.0);
        let mv = BettingAnalyzer::analyze_line_movement(&opening, &closing);
        assert_eq!(mv.spread_direction, LineDirection::MovedDown);
        assert_eq!(mv.total_direction, LineDirection::MovedUp);
        assert_eq!(mv.significance, MovementSignificance::High);
    }

    #[test]
    fn test_place_bet() {
        let mut ba = BettingAnalyzer::new(1000.0);
        let bet = ba.place_bet("g1", BetType::Spread, "home", -3.0, -110.0, 50.0, -5.0);
        assert_eq!(bet.game_id, "g1");
        assert_eq!(bet.result, BetResult::Pending);
        assert_eq!(ba.bet_history().len(), 1);
    }

    #[test]
    fn test_settle_bet_win() {
        let mut ba = BettingAnalyzer::new(1000.0);
        ba.place_bet("g1", BetType::Spread, "home", -3.0, -110.0, 50.0, -5.0);
        ba.settle_bet("bet_0", BetResult::Win);
        let perf = ba.performance_summary();
        assert_eq!(perf.wins, 1);
        assert!(perf.profit > 0.0);
        assert!(ba.bankroll() > 1000.0);
    }

    #[test]
    fn test_settle_bet_loss() {
        let mut ba = BettingAnalyzer::new(1000.0);
        ba.place_bet("g1", BetType::Spread, "home", -3.0, -110.0, 50.0, -5.0);
        ba.settle_bet("bet_0", BetResult::Loss);
        let perf = ba.performance_summary();
        assert_eq!(perf.losses, 1);
        assert!(perf.profit < 0.0);
        assert!(ba.bankroll() < 1000.0);
    }

    #[test]
    fn test_settle_bet_push() {
        let mut ba = BettingAnalyzer::new(1000.0);
        ba.place_bet("g1", BetType::Spread, "home", -3.0, -110.0, 50.0, -5.0);
        ba.settle_bet("bet_0", BetResult::Push);
        let perf = ba.performance_summary();
        assert_eq!(perf.pushes, 1);
        assert_eq!(ba.bankroll(), 1000.0); // No change
    }

    #[test]
    fn test_performance_summary_empty() {
        let ba = BettingAnalyzer::new(1000.0);
        let perf = ba.performance_summary();
        assert_eq!(perf.total_bets, 0);
        assert_eq!(perf.win_rate, 0.0);
        assert_eq!(perf.roi, 0.0);
    }

    #[test]
    fn test_performance_roi() {
        let mut ba = BettingAnalyzer::new(1000.0);
        ba.place_bet("g1", BetType::Spread, "home", -3.0, -110.0, 100.0, -5.0);
        ba.settle_bet("bet_0", BetResult::Win);
        let perf = ba.performance_summary();
        assert!(perf.roi > 0.0);
    }

    #[test]
    fn test_with_min_edge() {
        let ba = BettingAnalyzer::with_min_edge(5.0, 1000.0);
        assert_eq!(ba.min_edge_threshold, 5.0);
    }

    #[test]
    fn test_kelly_fraction_clamped() {
        let ba = BettingAnalyzer::with_kelly(0.8, 1000.0);
        assert!((ba.kelly_fraction - 0.5).abs() < 1e-9);
    }
}
