use crate::betting::{BettingAnalyzer, BettingPerformance};
use crate::predictions::PredictionEngine;
use crate::stats::StatsEngine;
use crate::types::{
    GamePrediction, Matchup, PlayerProjection, PlayerStats, Sport, TeamStats,
};
use std::collections::HashMap;

/// Daily analysis report.
#[derive(Debug, Clone)]
pub struct DailyReport {
    pub date: String,
    pub game_predictions: Vec<GamePrediction>,
    pub player_projections: Vec<PlayerProjection>,
    pub best_values: Vec<String>,
    pub summary: String,
}

/// Top plays recommendation.
#[derive(Debug, Clone)]
pub struct TopPlay {
    pub game_id: String,
    pub recommendation: String,
    pub confidence: f64,
    pub edge: f64,
}

/// Orchestration pipeline for full sports analysis workflow.
#[derive(Debug)]
pub struct CourtvisionPipeline {
    stats_engine: StatsEngine,
    prediction_engine: PredictionEngine,
    betting_analyzer: BettingAnalyzer,
}

impl CourtvisionPipeline {
    pub fn new(bankroll: f64) -> Self {
        Self {
            stats_engine: StatsEngine::new(),
            prediction_engine: PredictionEngine::new(),
            betting_analyzer: BettingAnalyzer::new(bankroll),
        }
    }

    /// Run full prediction pipeline for a set of matchups.
    pub fn predict_games(
        &self,
        matchups: &[Matchup],
        teams: &HashMap<String, TeamStats>,
    ) -> Vec<GamePrediction> {
        matchups
            .iter()
            .filter_map(|m| {
                let home = teams.get(&m.home_team)?;
                let away = teams.get(&m.away_team)?;
                Some(self.prediction_engine.predict_game(m, home, away))
            })
            .collect()
    }

    /// Project players for a given opponent.
    pub fn project_players(
        &self,
        players: &[PlayerStats],
        opponent_defense: f64,
    ) -> Vec<PlayerProjection> {
        self.prediction_engine.project_players(players, opponent_defense)
    }

    /// Generate a daily analysis report.
    pub fn generate_daily_report(
        &self,
        date: &str,
        matchups: &[Matchup],
        teams: &HashMap<String, TeamStats>,
        players: &[PlayerStats],
    ) -> DailyReport {
        let predictions = self.predict_games(matchups, teams);

        let player_projections: Vec<PlayerProjection> = players
            .iter()
            .map(|p| self.prediction_engine.project_player(p, 110.0))
            .collect();

        let best_values: Vec<String> = predictions
            .iter()
            .filter(|p| p.confidence > 0.6)
            .map(|p| format!("{}: {} (conf: {:.0}%)", p.game_id, p.predicted_winner, p.confidence * 100.0))
            .collect();

        let high_conf_count = predictions.iter().filter(|p| p.confidence > 0.6).count();
        let summary = format!(
            "Analyzed {} games. {} high-confidence plays found. {} player projections generated.",
            predictions.len(),
            high_conf_count,
            player_projections.len()
        );

        DailyReport {
            date: date.to_string(),
            game_predictions: predictions,
            player_projections,
            best_values,
            summary,
        }
    }

    /// Get top plays from predictions.
    pub fn top_plays(&self, predictions: &[GamePrediction], max: usize) -> Vec<TopPlay> {
        let mut sorted: Vec<&GamePrediction> = predictions.iter().collect();
        sorted.sort_by(|a, b| b.confidence.partial_cmp(&a.confidence).unwrap());
        sorted
            .iter()
            .take(max)
            .map(|p| TopPlay {
                game_id: p.game_id.clone(),
                recommendation: format!("{} {} by {:.1}", p.predicted_winner, if p.predicted_spread > 0.0 { "+" } else { "" }, p.predicted_spread),
                confidence: p.confidence,
                edge: p.predicted_spread.abs(),
            })
            .collect()
    }

    /// Full pipeline: predict + evaluate betting opportunities.
    pub fn full_analysis(
        &mut self,
        matchups: &[Matchup],
        teams: &HashMap<String, TeamStats>,
        players: &[PlayerStats],
        lines: &HashMap<String, crate::types::BettingLine>,
    ) -> (DailyReport, Vec<TopPlay>) {
        let predictions = self.predict_games(matchups, teams);
        let player_projections: Vec<PlayerProjection> = players
            .iter()
            .map(|p| self.prediction_engine.project_player(p, 110.0))
            .collect();

        let best_values: Vec<String> = predictions
            .iter()
            .filter(|p| p.confidence > 0.6)
            .map(|p| format!("{}: {} (conf: {:.0}%)", p.game_id, p.predicted_winner, p.confidence * 100.0))
            .collect();

        let high_conf_count = predictions.iter().filter(|p| p.confidence > 0.6).count();
        let summary = format!(
            "Analyzed {} games. {} high-confidence plays. {} player projections.",
            predictions.len(),
            high_conf_count,
            player_projections.len()
        );

        let report = DailyReport {
            date: String::new(),
            game_predictions: predictions.clone(),
            player_projections,
            best_values: best_values.clone(),
            summary,
        };

        let plays = self.top_plays(&predictions, 5);

        // Auto-detect value bets
        for pred in &predictions {
            if let Some(line) = lines.get(&pred.game_id) {
                let val = self.betting_analyzer.spread_value(pred, line);
                if val.has_value {
                    // Could auto-place bets here
                }
            }
        }

        (report, plays)
    }

    /// Access the stats engine.
    pub fn stats_engine(&self) -> &StatsEngine {
        &self.stats_engine
    }

    /// Access the prediction engine.
    pub fn prediction_engine(&self) -> &PredictionEngine {
        &self.prediction_engine
    }

    /// Access the betting analyzer.
    pub fn betting_analyzer(&self) -> &BettingAnalyzer {
        &self.betting_analyzer
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::types::BettingLine;

    fn make_team(id: &str, wins: u32, losses: u32, pf: f64, pa: f64) -> TeamStats {
        TeamStats {
            team_id: id.to_string(),
            name: id.to_string(),
            sport: Sport::NBA,
            season: "2024".into(),
            wins,
            losses,
            win_pct: wins as f64 / (wins + losses) as f64,
            points_for: pf,
            points_against: pa,
            home_record: (wins * 2 / 3, losses / 3),
            away_record: (wins / 3, losses * 2 / 3),
            streak: (wins as i32 - losses as i32).signum() * 3,
        }
    }

    fn make_matchup(home: &str, away: &str) -> Matchup {
        Matchup::new(home, away)
    }

    #[test]
    fn test_pipeline_predict_games() {
        let pipeline = CourtvisionPipeline::new(1000.0);
        let mut teams = HashMap::new();
        teams.insert("LAL".into(), make_team("LAL", 50, 32, 8500.0, 8200.0));
        teams.insert("BOS".into(), make_team("BOS", 55, 27, 9000.0, 8400.0));

        let matchups = vec![make_matchup("LAL", "BOS")];
        let preds = pipeline.predict_games(&matchups, &teams);
        assert_eq!(preds.len(), 1);
        assert!(!preds[0].predicted_winner.is_empty());
    }

    #[test]
    fn test_pipeline_missing_team() {
        let pipeline = CourtvisionPipeline::new(1000.0);
        let teams = HashMap::new();
        let matchups = vec![make_matchup("LAL", "BOS")];
        let preds = pipeline.predict_games(&matchups, &teams);
        assert_eq!(preds.len(), 0);
    }

    #[test]
    fn test_pipeline_project_players() {
        let pipeline = CourtvisionPipeline::new(1000.0);
        let players = vec![PlayerStats::new("p1", "Star", Sport::NBA, "2024")];
        let projs = pipeline.project_players(&players, 105.0);
        assert_eq!(projs.len(), 1);
    }

    #[test]
    fn test_pipeline_daily_report() {
        let pipeline = CourtvisionPipeline::new(1000.0);
        let mut teams = HashMap::new();
        teams.insert("LAL".into(), make_team("LAL", 50, 32, 8500.0, 8200.0));
        teams.insert("BOS".into(), make_team("BOS", 55, 27, 9000.0, 8400.0));

        let report = pipeline.generate_daily_report(
            "2024-01-15",
            &[make_matchup("LAL", "BOS")],
            &teams,
            &[],
        );
        assert_eq!(report.date, "2024-01-15");
        assert_eq!(report.game_predictions.len(), 1);
        assert!(!report.summary.is_empty());
    }

    #[test]
    fn test_pipeline_top_plays() {
        let pipeline = CourtvisionPipeline::new(1000.0);
        let preds = vec![
            GamePrediction {
                game_id: "g1".into(),
                home_team: "LAL".into(),
                away_team: "BOS".into(),
                predicted_winner: "LAL".into(),
                confidence: 0.8,
                predicted_spread: -5.0,
                predicted_total: 215.0,
                reasoning: "test".into(),
            },
            GamePrediction {
                game_id: "g2".into(),
                home_team: "GSW".into(),
                away_team: "SAS".into(),
                predicted_winner: "GSW".into(),
                confidence: 0.6,
                predicted_spread: -3.0,
                predicted_total: 220.0,
                reasoning: "test".into(),
            },
        ];
        let plays = pipeline.top_plays(&preds, 1);
        assert_eq!(plays.len(), 1);
        assert_eq!(plays[0].game_id, "g1"); // Higher confidence
        assert!((plays[0].confidence - 0.8).abs() < 1e-9);
    }

    #[test]
    fn test_pipeline_full_analysis() {
        let mut pipeline = CourtvisionPipeline::new(1000.0);
        let mut teams = HashMap::new();
        teams.insert("LAL".into(), make_team("LAL", 50, 32, 8500.0, 8200.0));
        teams.insert("BOS".into(), make_team("BOS", 55, 27, 9000.0, 8400.0));

        let (report, plays) = pipeline.full_analysis(
            &[make_matchup("LAL", "BOS")],
            &teams,
            &[],
            &HashMap::new(),
        );
        assert_eq!(report.game_predictions.len(), 1);
        assert!(!report.summary.is_empty());
    }

    #[test]
    fn test_pipeline_accessors() {
        let pipeline = CourtvisionPipeline::new(1000.0);
        let _ = pipeline.stats_engine();
        let _ = pipeline.prediction_engine();
        let _ = pipeline.betting_analyzer();
    }

    #[test]
    fn test_pipeline_daily_report_with_players() {
        let pipeline = CourtvisionPipeline::new(1000.0);
        let mut teams = HashMap::new();
        teams.insert("LAL".into(), make_team("LAL", 50, 32, 8500.0, 8200.0));
        teams.insert("BOS".into(), make_team("BOS", 55, 27, 9000.0, 8400.0));

        let players = vec![PlayerStats::new("p1", "Star", Sport::NBA, "2024")];
        let report = pipeline.generate_daily_report(
            "2024-01-15",
            &[make_matchup("LAL", "BOS")],
            &teams,
            &players,
        );
        assert_eq!(report.player_projections.len(), 1);
    }

    #[test]
    fn test_pipeline_multiple_games() {
        let pipeline = CourtvisionPipeline::new(1000.0);
        let mut teams = HashMap::new();
        teams.insert("LAL".into(), make_team("LAL", 50, 32, 8500.0, 8200.0));
        teams.insert("BOS".into(), make_team("BOS", 55, 27, 9000.0, 8400.0));
        teams.insert("GSW".into(), make_team("GSW", 45, 37, 8800.0, 8500.0));
        teams.insert("SAS".into(), make_team("SAS", 30, 52, 8100.0, 8700.0));

        let matchups = vec![
            make_matchup("LAL", "BOS"),
            make_matchup("GSW", "SAS"),
        ];
        let preds = pipeline.predict_games(&matchups, &teams);
        assert_eq!(preds.len(), 2);
    }

    #[test]
    fn test_top_plays_max_limit() {
        let pipeline = CourtvisionPipeline::new(1000.0);
        let preds = vec![
            GamePrediction {
                game_id: "g1".into(), home_team: "A".into(), away_team: "B".into(),
                predicted_winner: "A".into(), confidence: 0.8, predicted_spread: -5.0,
                predicted_total: 215.0, reasoning: "test".into(),
            },
            GamePrediction {
                game_id: "g2".into(), home_team: "C".into(), away_team: "D".into(),
                predicted_winner: "C".into(), confidence: 0.7, predicted_spread: -3.0,
                predicted_total: 220.0, reasoning: "test".into(),
            },
            GamePrediction {
                game_id: "g3".into(), home_team: "E".into(), away_team: "F".into(),
                predicted_winner: "E".into(), confidence: 0.6, predicted_spread: -2.0,
                predicted_total: 210.0, reasoning: "test".into(),
            },
        ];
        let plays = pipeline.top_plays(&preds, 2);
        assert_eq!(plays.len(), 2);
    }
}
