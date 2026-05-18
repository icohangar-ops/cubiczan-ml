use crate::types::{PlayerStats, Sport, TeamStats};
use std::collections::HashMap;

/// Statistics computation engine for sports analytics.
#[derive(Debug, Clone)]
pub struct StatsEngine {
    league_avg_pace: f64,
}

impl StatsEngine {
    pub fn new() -> Self {
        Self {
            league_avg_pace: 100.0,
        }
    }

    pub fn with_pace(pace: f64) -> Self {
        Self {
            league_avg_pace: pace.max(0.0),
        }
    }

    // ─── Per-game Averages ─────────────────────────────────────

    /// Points per game.
    pub fn ppg(stats: &PlayerStats) -> f64 {
        stats.ppg()
    }

    /// Rebounds per game.
    pub fn rpg(stats: &PlayerStats) -> f64 {
        stats.rpg()
    }

    /// Assists per game.
    pub fn apg(stats: &PlayerStats) -> f64 {
        stats.apg()
    }

    /// Per-game value for a named advanced stat.
    pub fn advanced_per_game(stats: &PlayerStats, key: &str) -> f64 {
        if stats.games_played == 0 {
            return 0.0;
        }
        stats
            .advanced_stats
            .get(key)
            .copied()
            .unwrap_or(0.0)
            / stats.games_played as f64
    }

    // ─── Advanced Metrics ──────────────────────────────────────

    /// PER-like efficiency rating.
    /// Simplified: (ppg + rpg + apg + spg + bpg - turnovers) / minutes_proxy
    pub fn per_like(stats: &PlayerStats) -> f64 {
        if stats.games_played == 0 {
            return 0.0;
        }
        let ppg = stats.ppg();
        let rpg = stats.rpg();
        let apg = stats.apg();
        let stl = stats.advanced_stats.get("steals").copied().unwrap_or(0.0) / stats.games_played as f64;
        let blk = stats.advanced_stats.get("blocks").copied().unwrap_or(0.0) / stats.games_played as f64;
        let tov = stats.advanced_stats.get("turnovers").copied().unwrap_or(0.0) / stats.games_played as f64;
        let mins = stats.advanced_stats.get("minutes").copied().unwrap_or(0.0) / stats.games_played as f64;
        if mins < 1.0 {
            return 0.0;
        }
        (ppg + rpg + apg + stl + blk - tov) * (36.0 / mins) * 15.0
    }

    /// True shooting percentage.
    /// TS% = Points / (2 * (FGA + 0.44 * FTA))
    pub fn true_shooting_pct(stats: &PlayerStats) -> f64 {
        let fga = stats.advanced_stats.get("field_goal_attempts").copied().unwrap_or(0.0);
        let fta = stats.advanced_stats.get("free_throw_attempts").copied().unwrap_or(0.0);
        let ts_attempts = 2.0 * (fga + 0.44 * fta);
        if ts_attempts < 1e-9 {
            return 0.0;
        }
        stats.points / ts_attempts
    }

    /// Usage rate estimate.
    /// USG% ≈ (FGA + 0.44*FTA + TOV) / (2 * team_possessions_proxy * games)
    pub fn usage_rate(stats: &PlayerStats, team_possessions: f64) -> f64 {
        if stats.games_played == 0 || team_possessions < 1e-9 {
            return 0.0;
        }
        let fga = stats.advanced_stats.get("field_goal_attempts").copied().unwrap_or(0.0);
        let fta = stats.advanced_stats.get("free_throw_attempts").copied().unwrap_or(0.0);
        let tov = stats.advanced_stats.get("turnovers").copied().unwrap_or(0.0);
        let possessions = team_possessions * stats.games_played as f64;
        ((fga + 0.44 * fta + tov) / (2.0 * possessions)) * 100.0
    }

    /// Net rating = (offensive_rating - defensive_rating) per 100 possessions.
    pub fn net_rating(off_rating: f64, def_rating: f64) -> f64 {
        off_rating - def_rating
    }

    // ─── Pace Adjustment ───────────────────────────────────────

    /// Adjust a per-game stat for pace differences.
    pub fn pace_adjust(&self, stat_value: f64, player_pace: f64) -> f64 {
        if self.league_avg_pace < 1e-9 || player_pace < 1e-9 {
            return stat_value;
        }
        stat_value * (self.league_avg_pace / player_pace)
    }

    /// Get the league average pace.
    pub fn league_pace(&self) -> f64 {
        self.league_avg_pace
    }

    // ─── Strength of Schedule ──────────────────────────────────

    /// Compute strength of schedule from opponent win percentages.
    /// SOS = (2 * opponents' winning pct + opponents' opponents' winning pct) / 3
    pub fn strength_of_schedule(
        opponents_win_pct: f64,
        opponents_opponents_win_pct: f64,
    ) -> f64 {
        (2.0 * opponents_win_pct + opponents_opponents_win_pct) / 3.0
    }

    /// Batch SOS computation from a list of opponent win pcts.
    pub fn batch_sos(opponents_wpcts: &[f64], opp_opp_wpcts: &[f64]) -> f64 {
        if opponents_wpcts.is_empty() {
            return 0.5;
        }
        let avg_opp: f64 = opponents_wpcts.iter().sum::<f64>() / opponents_wpcts.len() as f64;
        let avg_opp_opp: f64 = if opp_opp_wpcts.is_empty() {
            avg_opp
        } else {
            opp_opp_wpcts.iter().sum::<f64>() / opp_opp_wpcts.len() as f64
        };
        Self::strength_of_schedule(avg_opp, avg_opp_opp)
    }

    // ─── Home/Away Splits ──────────────────────────────────────

    /// Compute home vs away point differential gap.
    pub fn home_away_split_gap(team: &TeamStats) -> f64 {
        let home_games = team.home_record.0 + team.home_record.1;
        let away_games = team.away_record.0 + team.away_record.1;
        if home_games == 0 || away_games == 0 {
            return 0.0;
        }
        team.home_win_pct() - team.away_win_pct()
    }

    /// Normalize point differential for home court advantage.
    pub fn home_court_adjustment(raw_spread: f64, sport: Sport) -> f64 {
        let hca = match sport {
            Sport::NBA => 2.5,
            Sport::NFL => 3.0,
            Sport::NHL => 0.3,
            Sport::MLB => 0.1,
            Sport::Soccer => 0.5,
            Sport::Tennis => 0.0,
            Sport::MMA => 0.0,
            Sport::Cricket => 0.2,
        };
        raw_spread - hca
    }

    // ─── Streak Analysis ───────────────────────────────────────

    /// Classify a streak as hot, cold, or neutral.
    pub fn streak_category(streak: i32) -> &'static str {
        if streak >= 5 {
            "very_hot"
        } else if streak >= 3 {
            "hot"
        } else if streak >= 1 {
            "warm"
        } else if streak == 0 {
            "neutral"
        } else if streak >= -2 {
            "cool"
        } else if streak >= -4 {
            "cold"
        } else {
            "very_cold"
        }
    }

    /// Compute expected win rate regression toward the mean.
    pub fn streak_regressed_win_pct(current_pct: f64, streak: i32, sample_size: u32) -> f64 {
        if sample_size == 0 {
            return 0.5;
        }
        let streak_weight = (streak.abs() as f64).min(10.0) / 10.0;
        let regression_weight = 1.0 - streak_weight;
        let regression_target = 0.5; // mean regression
        current_pct * streak_weight + regression_target * regression_weight
    }

    // ─── Composite Scoring ─────────────────────────────────────

    /// Compute a composite score from multiple weighted metrics.
    pub fn composite_score(metrics: &HashMap<String, f64>, weights: &HashMap<String, f64>) -> f64 {
        let mut total = 0.0;
        let mut weight_sum = 0.0;
        for (key, weight) in weights {
            if let Some(value) = metrics.get(key) {
                total += value * weight;
                weight_sum += weight;
            }
        }
        if weight_sum < 1e-9 {
            return 0.0;
        }
        total / weight_sum
    }
}

impl Default for StatsEngine {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn sample_player() -> PlayerStats {
        let mut adv = HashMap::new();
        adv.insert("steals".to_string(), 100.0);
        adv.insert("blocks".to_string(), 50.0);
        adv.insert("turnovers".to_string(), 120.0);
        adv.insert("minutes".to_string(), 2400.0);
        adv.insert("field_goal_attempts".to_string(), 1200.0);
        adv.insert("free_throw_attempts".to_string(), 300.0);
        PlayerStats {
            player_id: "p1".into(),
            name: "Test Player".into(),
            sport: Sport::NBA,
            season: "2024".into(),
            games_played: 80,
            points: 1800.0,
            rebounds: 600.0,
            assists: 400.0,
            efficiency_rating: 22.5,
            plus_minus: 200.0,
            advanced_stats: adv,
        }
    }

    fn zero_player() -> PlayerStats {
        PlayerStats::new("p0", "Empty", Sport::NBA, "2024")
    }

    #[test]
    fn test_ppg() {
        let p = sample_player();
        assert!((StatsEngine::ppg(&p) - 22.5).abs() < 1e-9);
    }

    #[test]
    fn test_rpg() {
        let p = sample_player();
        assert!((StatsEngine::rpg(&p) - 7.5).abs() < 1e-9);
    }

    #[test]
    fn test_apg() {
        let p = sample_player();
        assert!((StatsEngine::apg(&p) - 5.0).abs() < 1e-9);
    }

    #[test]
    fn test_advanced_per_game() {
        let p = sample_player();
        let stl_pg = StatsEngine::advanced_per_game(&p, "steals");
        assert!((stl_pg - 1.25).abs() < 1e-9);
    }

    #[test]
    fn test_advanced_per_game_missing_key() {
        let p = sample_player();
        let val = StatsEngine::advanced_per_game(&p, "nonexistent");
        assert!((val - 0.0).abs() < 1e-9);
    }

    #[test]
    fn test_advanced_per_game_zero_games() {
        let p = zero_player();
        assert_eq!(StatsEngine::advanced_per_game(&p, "steals"), 0.0);
    }

    #[test]
    fn test_per_like() {
        let p = sample_player();
        let per = StatsEngine::per_like(&p);
        assert!(per > 0.0);
        // per should be positive since ppg+rpg+apg > turnovers
        assert!(per > 10.0);
    }

    #[test]
    fn test_per_like_zero_games() {
        let p = zero_player();
        assert_eq!(StatsEngine::per_like(&p), 0.0);
    }

    #[test]
    fn test_true_shooting_pct() {
        let p = sample_player();
        let ts = StatsEngine::true_shooting_pct(&p);
        assert!(ts > 0.0);
        assert!(ts < 1.0); // should be a percentage
        // 1800 / (2 * (1200 + 0.44 * 300)) = 1800 / 2664 ≈ 0.6757
        assert!((ts - 0.6757).abs() < 0.01);
    }

    #[test]
    fn test_true_shooting_zero_attempts() {
        let p = zero_player();
        assert_eq!(StatsEngine::true_shooting_pct(&p), 0.0);
    }

    #[test]
    fn test_usage_rate() {
        let p = sample_player();
        let usage = StatsEngine::usage_rate(&p, 80.0); // ~80 possessions per game
        assert!(usage > 0.0);
        assert!(usage < 100.0);
    }

    #[test]
    fn test_usage_rate_zero_games() {
        let p = zero_player();
        assert_eq!(StatsEngine::usage_rate(&p, 80.0), 0.0);
    }

    #[test]
    fn test_net_rating() {
        assert!((StatsEngine::net_rating(115.0, 110.0) - 5.0).abs() < 1e-9);
        assert!((StatsEngine::net_rating(100.0, 105.0) - (-5.0)).abs() < 1e-9);
    }

    #[test]
    fn test_pace_adjust() {
        let engine = StatsEngine::with_pace(100.0);
        let adjusted = engine.pace_adjust(20.0, 105.0);
        // 20 * (100/105) ≈ 19.05
        assert!((adjusted - 19.0476).abs() < 0.01);
    }

    #[test]
    fn test_pace_adjust_zero_league_pace() {
        let engine = StatsEngine::with_pace(0.0);
        assert_eq!(engine.pace_adjust(20.0, 105.0), 20.0);
    }

    #[test]
    fn test_strength_of_schedule() {
        let sos = StatsEngine::strength_of_schedule(0.55, 0.52);
        // (2*0.55 + 0.52) / 3 = 1.62 / 3 = 0.54
        assert!((sos - 0.54).abs() < 1e-9);
    }

    #[test]
    fn test_batch_sos() {
        let sos = StatsEngine::batch_sos(&[0.6, 0.5, 0.55], &[0.52, 0.51, 0.53]);
        // avg_opp = 0.55, avg_opp_opp = 0.52
        // (2*0.55 + 0.52) / 3 = 0.54
        assert!((sos - 0.54).abs() < 1e-9);
    }

    #[test]
    fn test_batch_sos_empty() {
        assert_eq!(StatsEngine::batch_sos(&[], &[]), 0.5);
    }

    #[test]
    fn test_home_away_split_gap() {
        let team = TeamStats {
            team_id: "t1".into(),
            name: "Team".into(),
            sport: Sport::NBA,
            season: "2024".into(),
            wins: 41,
            losses: 41,
            win_pct: 0.5,
            points_for: 8000.0,
            points_against: 8000.0,
            home_record: (30, 11),
            away_record: (11, 30),
            streak: 0,
        };
        let gap = StatsEngine::home_away_split_gap(&team);
        assert!(gap > 0.0);
    }

    #[test]
    fn test_home_away_split_zero_games() {
        let team = TeamStats::new("t1", "Team", Sport::NBA, "2024");
        assert_eq!(StatsEngine::home_away_split_gap(&team), 0.0);
    }

    #[test]
    fn test_home_court_adjustment() {
        let adjusted = StatsEngine::home_court_adjustment(5.0, Sport::NBA);
        assert!((adjusted - 2.5).abs() < 1e-9);
    }

    #[test]
    fn test_home_court_adjustment_nfl() {
        let adjusted = StatsEngine::home_court_adjustment(7.0, Sport::NFL);
        assert!((adjusted - 4.0).abs() < 1e-9);
    }

    #[test]
    fn test_home_court_adjustment_neutral_sport() {
        let adjusted = StatsEngine::home_court_adjustment(3.0, Sport::Tennis);
        assert!((adjusted - 3.0).abs() < 1e-9);
    }

    #[test]
    fn test_streak_categories() {
        assert_eq!(StatsEngine::streak_category(6), "very_hot");
        assert_eq!(StatsEngine::streak_category(4), "hot");
        assert_eq!(StatsEngine::streak_category(2), "warm");
        assert_eq!(StatsEngine::streak_category(0), "neutral");
        assert_eq!(StatsEngine::streak_category(-1), "cool");
        assert_eq!(StatsEngine::streak_category(-3), "cold");
        assert_eq!(StatsEngine::streak_category(-6), "very_cold");
    }

    #[test]
    fn test_streak_regressed_win_pct() {
        let regressed = StatsEngine::streak_regressed_win_pct(0.65, 5, 82);
        assert!(regressed > 0.5); // pulled up by streak
        assert!(regressed < 0.65);
    }

    #[test]
    fn test_streak_regressed_zero_sample() {
        assert_eq!(StatsEngine::streak_regressed_win_pct(0.6, 3, 0), 0.5);
    }

    #[test]
    fn test_composite_score() {
        let mut metrics = HashMap::new();
        metrics.insert("a".to_string(), 10.0);
        metrics.insert("b".to_string(), 20.0);
        let mut weights = HashMap::new();
        weights.insert("a".to_string(), 1.0);
        weights.insert("b".to_string(), 2.0);
        let score = StatsEngine::composite_score(&metrics, &weights);
        // (10*1 + 20*2) / (1+2) = 50/3 ≈ 16.667
        assert!((score - 16.6667).abs() < 0.01);
    }

    #[test]
    fn test_composite_score_empty() {
        let score = StatsEngine::composite_score(&HashMap::new(), &HashMap::new());
        assert_eq!(score, 0.0);
    }

    #[test]
    fn test_default_engine() {
        let engine = StatsEngine::default();
        assert!((engine.league_pace() - 100.0).abs() < 1e-9);
    }

    #[test]
    fn test_league_pace() {
        let engine = StatsEngine::with_pace(98.5);
        assert!((engine.league_pace() - 98.5).abs() < 1e-9);
    }
}
