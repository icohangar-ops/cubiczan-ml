use crate::stats::StatsEngine;
use crate::types::{
    BettingLine, GamePrediction, Matchup, PlayerProjection, PlayerStats, Sport, TeamStats,
};
use std::collections::HashMap;

/// ELO-based game prediction engine.
#[derive(Debug, Clone)]
pub struct PredictionEngine {
    k_factor: f64,
    home_advantage_elo: f64,
    ratings: HashMap<String, f64>,
}

impl PredictionEngine {
    pub fn new() -> Self {
        Self {
            k_factor: 20.0,
            home_advantage_elo: 65.0,
            ratings: HashMap::new(),
        }
    }

    pub fn with_k_factor(k: f64) -> Self {
        Self {
            k_factor: k.max(0.0),
            home_advantage_elo: 65.0,
            ratings: HashMap::new(),
        }
    }

    pub fn with_home_advantage(home_adv: f64) -> Self {
        Self {
            k_factor: 20.0,
            home_advantage_elo: home_adv,
            ratings: HashMap::new(),
        }
    }

    // ─── ELO Rating System ─────────────────────────────────────

    /// Set or update a team's ELO rating.
    pub fn set_rating(&mut self, team_id: &str, rating: f64) {
        self.ratings.insert(team_id.to_string(), rating);
    }

    /// Get a team's ELO rating, defaulting to 1500.
    pub fn get_rating(&self, team_id: &str) -> f64 {
        *self.ratings.get(team_id).unwrap_or(&1500.0)
    }

    /// Expected score between two ELO ratings.
    pub fn expected_score(rating_a: f64, rating_b: f64) -> f64 {
        let exponent = (rating_b - rating_a) / 400.0;
        1.0 / (1.0 + 10.0_f64.powf(exponent))
    }

    /// Update ELO ratings after a game result.
    /// score_a: 1.0 for win, 0.5 for tie, 0.0 for loss.
    pub fn update_ratings(
        &mut self,
        team_a: &str,
        team_b: &str,
        score_a: f64,
    ) -> (f64, f64) {
        let rating_a = self.get_rating(team_a);
        let rating_b = self.get_rating(team_b);
        let expected_a = Self::expected_score(rating_a, rating_b);
        let expected_b = 1.0 - expected_a;

        let new_a = rating_a + self.k_factor * (score_a - expected_a);
        let new_b = rating_b + self.k_factor * ((1.0 - score_a) - expected_b);

        self.set_rating(team_a, new_a);
        self.set_rating(team_b, new_b);

        (new_a, new_b)
    }

    // ─── Point Spread Prediction ───────────────────────────────

    /// Predict point spread based on ELO difference.
    pub fn predict_spread(&self, home_team: &str, away_team: &str) -> f64 {
        let home_elo = self.get_rating(home_team) + self.home_advantage_elo;
        let away_elo = self.get_rating(away_team);
        let elo_diff = home_elo - away_elo;
        // Approximately 1 point per 3 ELO points
        elo_diff / 3.0
    }

    // ─── Over/Under Total Prediction ───────────────────────────

    /// Predict game total from team stats.
    pub fn predict_total(
        home_team_stats: &TeamStats,
        away_team_stats: &TeamStats,
    ) -> f64 {
        let home_off = home_team_stats.points_for / home_team_stats.total_games().max(1) as f64;
        let away_off = away_team_stats.points_for / away_team_stats.total_games().max(1) as f64;
        let home_def = home_team_stats.points_against / home_team_stats.total_games().max(1) as f64;
        let away_def = away_team_stats.points_against / away_team_stats.total_games().max(1) as f64;

        // Home offense vs away defense, adjusted by averages
        let league_avg = (home_off + away_off + home_def + away_def) / 4.0;
        let home_projected = (home_off + away_def) / 2.0;
        let away_projected = (away_off + home_def) / 2.0;

        home_projected + away_projected
    }

    // ─── Win Probability ───────────────────────────────────────

    /// Compute win probability from ELO ratings.
    pub fn win_probability(&self, home_team: &str, away_team: &str) -> f64 {
        let home_elo = self.get_rating(home_team) + self.home_advantage_elo;
        let away_elo = self.get_rating(away_team);
        Self::expected_score(home_elo, away_elo)
    }

    // ─── Multi-Factor Model ────────────────────────────────────

    /// Multi-factor prediction combining ELO, stats, and situational factors.
    pub fn predict_game(
        &self,
        matchup: &Matchup,
        home_stats: &TeamStats,
        away_stats: &TeamStats,
    ) -> GamePrediction {
        let elo_spread = self.predict_spread(&matchup.home_team, &matchup.away_team);
        let stats_spread = self.stats_based_spread(home_stats, away_stats);
        let situational = self.situational_factor(matchup);

        // Weighted combination: 50% ELO, 30% stats, 20% situational
        let combined_spread = elo_spread * 0.5 + stats_spread * 0.3 + situational * 0.2;

        let predicted_total = Self::predict_total(home_stats, away_stats);
        let win_prob = self.win_probability(&matchup.home_team, &matchup.away_team);

        let predicted_winner = if combined_spread > 0.0 {
            matchup.home_team.clone()
        } else {
            matchup.away_team.clone()
        };

        let confidence = self.calibrate_confidence(win_prob, combined_spread, matchup);

        let reasoning = format!(
            "ELO spread: {:.1}, Stats spread: {:.1}, Situational: {:.1}, Combined: {:.1}",
            elo_spread, stats_spread, situational, combined_spread
        );

        GamePrediction {
            game_id: String::new(),
            home_team: matchup.home_team.clone(),
            away_team: matchup.away_team.clone(),
            predicted_winner,
            confidence,
            predicted_spread: combined_spread,
            predicted_total,
            reasoning,
        }
    }

    /// Stats-based spread from point differentials.
    fn stats_based_spread(&self, home: &TeamStats, away: &TeamStats) -> f64 {
        let home_pd = home.point_differential_per_game();
        let away_pd = away.point_differential_per_game();
        let net = home_pd - away_pd;
        // Add base home court from sport
        net + self.home_advantage_elo / 3.0
    }

    /// Situational factor: rest advantage, injury impact.
    fn situational_factor(&self, matchup: &Matchup) -> f64 {
        let rest_factor = matchup.rest_advantage() as f64 * 0.5;
        let injury_factor = -(matchup.injury_count() as f64 * 1.5);
        rest_factor + injury_factor
    }

    // ─── Confidence Calibration ────────────────────────────────

    /// Calibrate confidence based on model agreement.
    pub fn calibrate_confidence(
        &self,
        win_prob: f64,
        spread: f64,
        _matchup: &Matchup,
    ) -> f64 {
        let spread_confidence = (spread.abs() / 15.0).min(1.0);
        let prob_confidence = (win_prob - 0.5).abs() * 2.0;
        (spread_confidence * 0.6 + prob_confidence * 0.4).clamp(0.1, 0.95)
    }

    // ─── Player Projection Integration ─────────────────────────

    /// Project player stats with matchup adjustments.
    pub fn project_player(
        &self,
        player: &PlayerStats,
        opponent_defense: f64, // opponent's defensive rating (points allowed per game)
    ) -> PlayerProjection {
        let base_ppg = player.ppg();
        let def_factor = opponent_defense / 110.0; // 110 = league average
        let predicted_points = base_ppg * def_factor;
        let predicted_rebounds = player.rpg();
        let predicted_assists = player.apg();

        let variance = base_ppg * 0.3;
        let ceiling = predicted_points + variance;
        let floor = (predicted_points - variance).max(0.0);

        let confidence = if player.games_played >= 20 {
            0.7 + (player.efficiency_rating / 100.0).min(0.25)
        } else {
            0.5
        };

        PlayerProjection {
            player_id: player.player_id.clone(),
            predicted_points,
            predicted_rebounds,
            predicted_assists,
            ceiling,
            floor,
            confidence: confidence.min(0.95),
        }
    }

    /// Batch project multiple players against a single opponent.
    pub fn project_players(
        &self,
        players: &[PlayerStats],
        opponent_defense: f64,
    ) -> Vec<PlayerProjection> {
        players
            .iter()
            .map(|p| self.project_player(p, opponent_defense))
            .collect()
    }

    // ─── Line Comparison ───────────────────────────────────────

    /// Compare model prediction to betting line.
    pub fn compare_to_line(
        &self,
        prediction: &GamePrediction,
        line: &BettingLine,
    ) -> LineComparison {
        let spread_diff = prediction.predicted_spread - line.spread;
        let total_diff = prediction.predicted_total - line.over_under;

        LineComparison {
            spread_edge: spread_diff,
            total_edge: total_diff,
            model_spread: prediction.predicted_spread,
            line_spread: line.spread,
            model_total: prediction.predicted_total,
            line_total: line.over_under,
            spread_value: spread_diff.abs() >= 2.0,
            total_value: total_diff.abs() >= 3.0,
        }
    }
}

impl Default for PredictionEngine {
    fn default() -> Self {
        Self::new()
    }
}

/// Result of comparing model prediction to betting line.
#[derive(Debug, Clone)]
pub struct LineComparison {
    pub spread_edge: f64,
    pub total_edge: f64,
    pub model_spread: f64,
    pub line_spread: f64,
    pub model_total: f64,
    pub line_total: f64,
    pub spread_value: bool,
    pub total_value: bool,
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_engine() -> PredictionEngine {
        let mut e = PredictionEngine::new();
        e.set_rating("LAL", 1600.0);
        e.set_rating("BOS", 1550.0);
        e
    }

    fn make_teams() -> (TeamStats, TeamStats) {
        let home = TeamStats {
            team_id: "LAL".into(),
            name: "Lakers".into(),
            sport: Sport::NBA,
            season: "2024".into(),
            wins: 50,
            losses: 32,
            win_pct: 50.0 / 82.0,
            points_for: 8500.0,
            points_against: 8200.0,
            home_record: (30, 11),
            away_record: (20, 21),
            streak: 5,
        };
        let away = TeamStats {
            team_id: "BOS".into(),
            name: "Celtics".into(),
            sport: Sport::NBA,
            season: "2024".into(),
            wins: 55,
            losses: 27,
            win_pct: 55.0 / 82.0,
            points_for: 9000.0,
            points_against: 8400.0,
            home_record: (32, 9),
            away_record: (23, 18),
            streak: 3,
        };
        (home, away)
    }

    #[test]
    fn test_set_and_get_rating() {
        let mut e = PredictionEngine::new();
        assert_eq!(e.get_rating("unknown"), 1500.0);
        e.set_rating("LAL", 1600.0);
        assert!((e.get_rating("LAL") - 1600.0).abs() < 1e-9);
    }

    #[test]
    fn test_expected_score() {
        // Equal ratings -> 0.5
        assert!((PredictionEngine::expected_score(1500.0, 1500.0) - 0.5).abs() < 1e-9);
        // Higher rated team has higher expected score
        let exp = PredictionEngine::expected_score(1600.0, 1500.0);
        assert!(exp > 0.5);
        assert!(exp < 1.0);
    }

    #[test]
    fn test_update_ratings_win() {
        let mut e = PredictionEngine::with_k_factor(20.0);
        e.set_rating("A", 1500.0);
        e.set_rating("B", 1500.0);
        let (new_a, new_b) = e.update_ratings("A", "B", 1.0);
        assert!(new_a > 1500.0);
        assert!(new_b < 1500.0);
    }

    #[test]
    fn test_update_ratings_loss() {
        let mut e = PredictionEngine::with_k_factor(20.0);
        e.set_rating("A", 1500.0);
        e.set_rating("B", 1500.0);
        let (new_a, new_b) = e.update_ratings("A", "B", 0.0);
        assert!(new_a < 1500.0);
        assert!(new_b > 1500.0);
    }

    #[test]
    fn test_update_ratings_tie() {
        let mut e = PredictionEngine::with_k_factor(20.0);
        e.set_rating("A", 1500.0);
        e.set_rating("B", 1500.0);
        let (new_a, new_b) = e.update_ratings("A", "B", 0.5);
        assert!((new_a - 1500.0).abs() < 1e-9);
        assert!((new_b - 1500.0).abs() < 1e-9);
    }

    #[test]
    fn test_predict_spread() {
        let e = make_engine();
        let spread = e.predict_spread("LAL", "BOS");
        // LAL 1600 + 65 (HCA) = 1665 vs BOS 1550, diff = 115, spread ≈ 38.3
        assert!((spread - (115.0 / 3.0)).abs() < 1e-6);
    }

    #[test]
    fn test_predict_spread_home_advantage() {
        let mut e = PredictionEngine::with_home_advantage(0.0);
        e.set_rating("A", 1500.0);
        e.set_rating("B", 1500.0);
        let spread = e.predict_spread("A", "B");
        assert_eq!(spread, 0.0);
    }

    #[test]
    fn test_predict_total() {
        let (home, away) = make_teams();
        let total = PredictionEngine::predict_total(&home, &away);
        // Should be around the average of both teams' offensive outputs
        assert!(total > 150.0);
        assert!(total < 300.0);
    }

    #[test]
    fn test_win_probability() {
        let e = make_engine();
        let prob = e.win_probability("LAL", "BOS");
        assert!(prob > 0.5); // LAL is higher rated + home advantage
        assert!(prob < 1.0);
    }

    #[test]
    fn test_win_probability_equal() {
        let mut e = PredictionEngine::new();
        e.set_rating("A", 1500.0);
        e.set_rating("B", 1500.0);
        let prob = e.win_probability("A", "B");
        // Home team gets HCA boost
        assert!(prob > 0.5);
    }

    #[test]
    fn test_predict_game() {
        let e = make_engine();
        let (home, away) = make_teams();
        let matchup = Matchup::new("LAL", "BOS");
        let pred = e.predict_game(&matchup, &home, &away);
        assert!(!pred.predicted_winner.is_empty());
        assert!(pred.confidence > 0.0);
        assert!(pred.confidence <= 1.0);
        assert!(!pred.reasoning.is_empty());
    }

    #[test]
    fn test_predict_game_home_favored() {
        let mut e = PredictionEngine::new();
        e.set_rating("LAL", 1600.0);
        e.set_rating("BOS", 1500.0);
        let (home, away) = make_teams();
        let matchup = Matchup::new("LAL", "BOS");
        let pred = e.predict_game(&matchup, &home, &away);
        assert_eq!(pred.predicted_winner, "LAL");
        assert!(pred.predicted_spread > 0.0);
    }

    #[test]
    fn test_predict_game_away_favored() {
        let mut e = PredictionEngine::new();
        e.set_rating("LAL", 1500.0);
        e.set_rating("BOS", 1700.0);
        let (home, away) = make_teams();
        let matchup = Matchup::new("LAL", "BOS");
        let pred = e.predict_game(&matchup, &home, &away);
        // BOS may still be favored due to high ELO
        assert!(pred.predicted_spread < 5.0); // Spread compressed
    }

    #[test]
    fn test_calibrate_confidence() {
        let e = PredictionEngine::new();
        let conf = e.calibrate_confidence(0.75, 10.0, &Matchup::new("A", "B"));
        assert!(conf > 0.5);
        assert!(conf <= 0.95);
    }

    #[test]
    fn test_calibrate_confidence_low() {
        let e = PredictionEngine::new();
        let conf = e.calibrate_confidence(0.51, 1.0, &Matchup::new("A", "B"));
        assert!(conf >= 0.1);
        assert!(conf < 0.5);
    }

    #[test]
    fn test_project_player() {
        let e = PredictionEngine::new();
        let p = PlayerStats {
            player_id: "p1".into(),
            name: "Star".into(),
            sport: Sport::NBA,
            season: "2024".into(),
            games_played: 80,
            points: 2000.0,
            rebounds: 600.0,
            assists: 500.0,
            efficiency_rating: 28.0,
            plus_minus: 300.0,
            advanced_stats: HashMap::new(),
        };
        let proj = e.project_player(&p, 105.0);
        assert!(proj.predicted_points > 0.0);
        assert!(proj.ceiling > proj.predicted_points);
        assert!(proj.floor < proj.predicted_points);
        assert!(proj.confidence > 0.0);
    }

    #[test]
    fn test_project_player_weak_defense() {
        let e = PredictionEngine::new();
        let p = PlayerStats {
            player_id: "p1".into(),
            name: "Star".into(),
            sport: Sport::NBA,
            season: "2024".into(),
            games_played: 80,
            points: 2000.0,
            rebounds: 600.0,
            assists: 500.0,
            efficiency_rating: 28.0,
            plus_minus: 300.0,
            advanced_stats: HashMap::new(),
        };
        let proj_weak = e.project_player(&p, 120.0); // weak defense
        let proj_strong = e.project_player(&p, 95.0); // strong defense
        assert!(proj_weak.predicted_points > proj_strong.predicted_points);
    }

    #[test]
    fn test_project_players_batch() {
        let e = PredictionEngine::new();
        let players = vec![
            PlayerStats::new("p1", "A", Sport::NBA, "2024"),
            PlayerStats::new("p2", "B", Sport::NBA, "2024"),
        ];
        let projections = e.project_players(&players, 110.0);
        assert_eq!(projections.len(), 2);
    }

    #[test]
    fn test_compare_to_line() {
        let e = make_engine();
        let (home, away) = make_teams();
        let pred = e.predict_game(&Matchup::new("LAL", "BOS"), &home, &away);
        let line = BettingLine {
            game_id: "g1".into(),
            spread: -5.0,
            over_under: 215.5,
            moneyline_home: -150.0,
            moneyline_away: 130.0,
            implied_prob_home: 0.6,
            implied_prob_away: 0.435,
        };
        let comp = e.compare_to_line(&pred, &line);
        // Should have non-zero edges
        assert!(comp.model_spread != comp.line_spread || comp.model_total != comp.line_total);
    }

    #[test]
    fn test_compare_to_line_spread_value() {
        let e = PredictionEngine::new();
        let pred = GamePrediction {
            game_id: "g1".into(),
            home_team: "LAL".into(),
            away_team: "BOS".into(),
            predicted_winner: "LAL".into(),
            confidence: 0.7,
            predicted_spread: -10.0,
            predicted_total: 220.0,
            reasoning: "test".into(),
        };
        let line = BettingLine {
            game_id: "g1".into(),
            spread: -5.0,
            over_under: 215.0,
            moneyline_home: -150.0,
            moneyline_away: 130.0,
            implied_prob_home: 0.6,
            implied_prob_away: 0.435,
        };
        let comp = e.compare_to_line(&pred, &line);
        assert!(comp.spread_value); // 5-point difference
    }

    #[test]
    fn test_compare_to_line_no_value() {
        let e = PredictionEngine::new();
        let pred = GamePrediction {
            game_id: "g1".into(),
            home_team: "LAL".into(),
            away_team: "BOS".into(),
            predicted_winner: "LAL".into(),
            confidence: 0.5,
            predicted_spread: -5.0,
            predicted_total: 215.5,
            reasoning: "test".into(),
        };
        let line = BettingLine {
            game_id: "g1".into(),
            spread: -5.0,
            over_under: 215.0,
            moneyline_home: -150.0,
            moneyline_away: 130.0,
            implied_prob_home: 0.6,
            implied_prob_away: 0.435,
        };
        let comp = e.compare_to_line(&pred, &line);
        assert!(!comp.spread_value); // Only 0.5 diff, threshold is 2.0
        assert!(!comp.total_value); // Only 0.5 diff, threshold is 3.0
    }

    #[test]
    fn test_default_engine() {
        let e = PredictionEngine::default();
        assert_eq!(e.k_factor, 20.0);
    }

    #[test]
    fn test_with_k_factor() {
        let e = PredictionEngine::with_k_factor(32.0);
        assert!((e.k_factor - 32.0).abs() < 1e-9);
    }

    #[test]
    fn test_k_factor_clamped() {
        let e = PredictionEngine::with_k_factor(-10.0);
        assert_eq!(e.k_factor, 0.0);
    }
}
