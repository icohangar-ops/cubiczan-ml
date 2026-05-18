use serde::{Deserialize, Serialize};
use std::collections::HashMap;

/// Supported sports for analysis.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum Sport {
    NBA,
    NFL,
    MLB,
    NHL,
    Soccer,
    Tennis,
    MMA,
    Cricket,
}

impl std::fmt::Display for Sport {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Sport::NBA => write!(f, "NBA"),
            Sport::NFL => write!(f, "NFL"),
            Sport::MLB => write!(f, "MLB"),
            Sport::NHL => write!(f, "NHL"),
            Sport::Soccer => write!(f, "Soccer"),
            Sport::Tennis => write!(f, "Tennis"),
            Sport::MMA => write!(f, "MMA"),
            Sport::Cricket => write!(f, "Cricket"),
        }
    }
}

/// Per-player statistical profile for a given season.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PlayerStats {
    pub player_id: String,
    pub name: String,
    pub sport: Sport,
    pub season: String,
    pub games_played: u32,
    pub points: f64,
    pub rebounds: f64,
    pub assists: f64,
    pub efficiency_rating: f64,
    pub plus_minus: f64,
    pub advanced_stats: HashMap<String, f64>,
}

impl PlayerStats {
    /// Create a minimal PlayerStats with required fields.
    pub fn new(player_id: &str, name: &str, sport: Sport, season: &str) -> Self {
        Self {
            player_id: player_id.to_string(),
            name: name.to_string(),
            sport,
            season: season.to_string(),
            games_played: 0,
            points: 0.0,
            rebounds: 0.0,
            assists: 0.0,
            efficiency_rating: 0.0,
            plus_minus: 0.0,
            advanced_stats: HashMap::new(),
        }
    }

    /// Per-game average for any stat field.
    pub fn per_game(&self, total: f64) -> f64 {
        if self.games_played == 0 {
            return 0.0;
        }
        total / self.games_played as f64
    }

    /// Points per game.
    pub fn ppg(&self) -> f64 {
        self.per_game(self.points)
    }

    /// Rebounds per game.
    pub fn rpg(&self) -> f64 {
        self.per_game(self.rebounds)
    }

    /// Assists per game.
    pub fn apg(&self) -> f64 {
        self.per_game(self.assists)
    }
}

/// Per-team statistical profile for a given season.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TeamStats {
    pub team_id: String,
    pub name: String,
    pub sport: Sport,
    pub season: String,
    pub wins: u32,
    pub losses: u32,
    pub win_pct: f64,
    pub points_for: f64,
    pub points_against: f64,
    pub home_record: (u32, u32), // (wins, losses)
    pub away_record: (u32, u32), // (wins, losses)
    pub streak: i32, // positive = win streak, negative = loss streak
}

impl TeamStats {
    pub fn new(team_id: &str, name: &str, sport: Sport, season: &str) -> Self {
        Self {
            team_id: team_id.to_string(),
            name: name.to_string(),
            sport,
            season: season.to_string(),
            wins: 0,
            losses: 0,
            win_pct: 0.0,
            points_for: 0.0,
            points_against: 0.0,
            home_record: (0, 0),
            away_record: (0, 0),
            streak: 0,
        }
    }

    pub fn total_games(&self) -> u32 {
        self.wins + self.losses
    }

    pub fn point_differential(&self) -> f64 {
        self.points_for - self.points_against
    }

    pub fn point_differential_per_game(&self) -> f64 {
        let total = self.total_games();
        if total == 0 {
            return 0.0;
        }
        self.point_differential() / total as f64
    }

    pub fn home_win_pct(&self) -> f64 {
        let (w, l) = self.home_record;
        let total = w + l;
        if total == 0 {
            return 0.0;
        }
        w as f64 / total as f64
    }

    pub fn away_win_pct(&self) -> f64 {
        let (w, l) = self.away_record;
        let total = w + l;
        if total == 0 {
            return 0.0;
        }
        w as f64 / total as f64
    }
}

/// A single event within a game (shot made, foul, turnover, etc.).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GameEvent {
    pub game_id: String,
    pub timestamp: f64,
    pub event_type: String,
    pub player_id: String,
    pub team_id: String,
    pub value: f64,
    pub quarter_period: u32,
}

/// Prediction output for a game matchup.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GamePrediction {
    pub game_id: String,
    pub home_team: String,
    pub away_team: String,
    pub predicted_winner: String,
    pub confidence: f64,
    pub predicted_spread: f64,
    pub predicted_total: f64,
    pub reasoning: String,
}

impl GamePrediction {
    pub fn new(game_id: &str, home_team: &str, away_team: &str) -> Self {
        Self {
            game_id: game_id.to_string(),
            home_team: home_team.to_string(),
            away_team: away_team.to_string(),
            predicted_winner: String::new(),
            confidence: 0.0,
            predicted_spread: 0.0,
            predicted_total: 0.0,
            reasoning: String::new(),
        }
    }
}

/// Projected player performance for a game.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PlayerProjection {
    pub player_id: String,
    pub predicted_points: f64,
    pub predicted_rebounds: f64,
    pub predicted_assists: f64,
    pub ceiling: f64,
    pub floor: f64,
    pub confidence: f64,
}

impl PlayerProjection {
    pub fn new(player_id: &str) -> Self {
        Self {
            player_id: player_id.to_string(),
            predicted_points: 0.0,
            predicted_rebounds: 0.0,
            predicted_assists: 0.0,
            ceiling: 0.0,
            floor: 0.0,
            confidence: 0.0,
        }
    }

    /// Expected fantasy-style score (simplified: pts + reb + ast).
    pub fn fantasy_score(&self) -> f64 {
        self.predicted_points + self.predicted_rebounds + self.predicted_assists
    }
}

/// Current betting line for a game.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BettingLine {
    pub game_id: String,
    pub spread: f64,
    pub over_under: f64,
    pub moneyline_home: f64,
    pub moneyline_away: f64,
    pub implied_prob_home: f64,
    pub implied_prob_away: f64,
}

impl BettingLine {
    pub fn new(game_id: &str) -> Self {
        Self {
            game_id: game_id.to_string(),
            spread: 0.0,
            over_under: 0.0,
            moneyline_home: -110.0,
            moneyline_away: -110.0,
            implied_prob_home: 0.5,
            implied_prob_away: 0.5,
        }
    }

    /// Convert American odds to implied probability.
    pub fn american_to_implied(odds: f64) -> f64 {
        if odds > 0.0 {
            100.0 / (odds + 100.0)
        } else if odds < 0.0 {
            -odds / (-odds + 100.0)
        } else {
            0.5
        }
    }

    /// Calculate the vigorish (overround) as a decimal.
    pub fn vigorish(&self) -> f64 {
        self.implied_prob_home + self.implied_prob_away - 1.0
    }

    /// True implied probability removing vig.
    pub fn true_prob_home(&self) -> f64 {
        let vig = self.vigorish();
        if (self.implied_prob_home + self.implied_prob_home).abs() < f64::EPSILON {
            return 0.5;
        }
        self.implied_prob_home / (1.0 + vig)
    }
}

/// Matchup context including venue, injuries, rest.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Matchup {
    pub home_team: String,
    pub away_team: String,
    pub venue: String,
    pub date: String,
    pub injuries: Vec<String>,
    pub rest_days_home: u32,
    pub rest_days_away: u32,
}

impl Matchup {
    pub fn new(home_team: &str, away_team: &str) -> Self {
        Self {
            home_team: home_team.to_string(),
            away_team: away_team.to_string(),
            venue: String::new(),
            date: String::new(),
            injuries: Vec::new(),
            rest_days_home: 3,
            rest_days_away: 3,
        }
    }

    /// Rest advantage (positive = home team more rested).
    pub fn rest_advantage(&self) -> i32 {
        self.rest_days_home as i32 - self.rest_days_away as i32
    }

    /// Count of key injuries (non-empty entries).
    pub fn injury_count(&self) -> usize {
        self.injuries.iter().filter(|s| !s.is_empty()).count()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_sport_display() {
        assert_eq!(Sport::NBA.to_string(), "NBA");
        assert_eq!(Sport::NFL.to_string(), "NFL");
        assert_eq!(Sport::Soccer.to_string(), "Soccer");
    }

    #[test]
    fn test_player_stats_new() {
        let p = PlayerStats::new("p1", "LeBron", Sport::NBA, "2024");
        assert_eq!(p.player_id, "p1");
        assert_eq!(p.name, "LeBron");
        assert_eq!(p.sport, Sport::NBA);
        assert_eq!(p.season, "2024");
        assert_eq!(p.games_played, 0);
    }

    #[test]
    fn test_player_stats_per_game() {
        let p = PlayerStats {
            player_id: "p1".into(),
            name: "Test".into(),
            sport: Sport::NBA,
            season: "2024".into(),
            games_played: 10,
            points: 200.0,
            rebounds: 50.0,
            assists: 80.0,
            efficiency_rating: 25.0,
            plus_minus: 30.0,
            advanced_stats: HashMap::new(),
        };
        assert!((p.ppg() - 20.0).abs() < 1e-9);
        assert!((p.rpg() - 5.0).abs() < 1e-9);
        assert!((p.apg() - 8.0).abs() < 1e-9);
    }

    #[test]
    fn test_player_stats_zero_games() {
        let p = PlayerStats::new("p1", "Test", Sport::NBA, "2024");
        assert_eq!(p.ppg(), 0.0);
        assert_eq!(p.rpg(), 0.0);
        assert_eq!(p.apg(), 0.0);
    }

    #[test]
    fn test_team_stats_new() {
        let t = TeamStats::new("t1", "Lakers", Sport::NBA, "2024");
        assert_eq!(t.team_id, "t1");
        assert_eq!(t.wins, 0);
        assert_eq!(t.streak, 0);
    }

    #[test]
    fn test_team_stats_point_differential() {
        let t = TeamStats {
            team_id: "t1".into(),
            name: "Lakers".into(),
            sport: Sport::NBA,
            season: "2024".into(),
            wins: 50,
            losses: 32,
            win_pct: 0.610,
            points_for: 8500.0,
            points_against: 8200.0,
            home_record: (30, 11),
            away_record: (20, 21),
            streak: 5,
        };
        assert!((t.point_differential() - 300.0).abs() < 1e-9);
        let pd_per = t.point_differential_per_game();
        assert!((pd_per - (300.0 / 82.0)).abs() < 1e-6);
    }

    #[test]
    fn test_team_stats_home_away_pct() {
        let t = TeamStats {
            team_id: "t1".into(),
            name: "Lakers".into(),
            sport: Sport::NBA,
            season: "2024".into(),
            wins: 50,
            losses: 32,
            win_pct: 0.610,
            points_for: 8500.0,
            points_against: 8200.0,
            home_record: (30, 11),
            away_record: (20, 21),
            streak: 5,
        };
        assert!((t.home_win_pct() - (30.0 / 41.0)).abs() < 1e-9);
        assert!((t.away_win_pct() - (20.0 / 41.0)).abs() < 1e-9);
    }

    #[test]
    fn test_team_stats_zero_record_pct() {
        let t = TeamStats::new("t1", "Team", Sport::NBA, "2024");
        assert_eq!(t.home_win_pct(), 0.0);
        assert_eq!(t.away_win_pct(), 0.0);
        assert_eq!(t.point_differential_per_game(), 0.0);
    }

    #[test]
    fn test_betting_line_implied_prob() {
        assert!((BettingLine::american_to_implied(-110.0) - 0.5238).abs() < 0.001);
        assert!((BettingLine::american_to_implied(110.0) - 0.4762).abs() < 0.001);
        assert!((BettingLine::american_to_implied(0.0) - 0.5).abs() < 1e-9);
        assert!((BettingLine::american_to_implied(-200.0) - (200.0 / 300.0)).abs() < 1e-9);
    }

    #[test]
    fn test_betting_line_vigorish() {
        let bl = BettingLine {
            game_id: "g1".into(),
            spread: -3.0,
            over_under: 215.5,
            moneyline_home: -150.0,
            moneyline_away: 130.0,
            implied_prob_home: 0.6,
            implied_prob_away: 0.435,
        };
        let vig = bl.vigorish();
        assert!(vig > 0.0);
    }

    #[test]
    fn test_matchup_rest_advantage() {
        let m = Matchup {
            home_team: "LAL".into(),
            away_team: "BOS".into(),
            venue: "Crypto.com".into(),
            date: "2024-01-15".into(),
            injuries: vec![],
            rest_days_home: 5,
            rest_days_away: 2,
        };
        assert_eq!(m.rest_advantage(), 3);
        assert_eq!(m.injury_count(), 0);
    }

    #[test]
    fn test_matchup_injury_count() {
        let m = Matchup {
            home_team: "LAL".into(),
            away_team: "BOS".into(),
            venue: "".into(),
            date: "".into(),
            injuries: vec!["Anthony Davis - questionable".into(), "".into()],
            rest_days_home: 3,
            rest_days_away: 3,
        };
        assert_eq!(m.injury_count(), 1);
    }

    #[test]
    fn test_game_prediction_new() {
        let gp = GamePrediction::new("g1", "LAL", "BOS");
        assert_eq!(gp.game_id, "g1");
        assert_eq!(gp.confidence, 0.0);
        assert_eq!(gp.predicted_spread, 0.0);
    }

    #[test]
    fn test_player_projection_fantasy_score() {
        let pp = PlayerProjection {
            player_id: "p1".into(),
            predicted_points: 25.0,
            predicted_rebounds: 7.0,
            predicted_assists: 10.0,
            ceiling: 50.0,
            floor: 20.0,
            confidence: 0.85,
        };
        assert!((pp.fantasy_score() - 42.0).abs() < 1e-9);
    }

    #[test]
    fn test_player_projection_new() {
        let pp = PlayerProjection::new("p1");
        assert_eq!(pp.player_id, "p1");
        assert_eq!(pp.fantasy_score(), 0.0);
    }

    #[test]
    fn test_serde_roundtrip_player_stats() {
        let p = PlayerStats::new("p1", "Test", Sport::NBA, "2024");
        let json = serde_json::to_string(&p).unwrap();
        let p2: PlayerStats = serde_json::from_str(&json).unwrap();
        assert_eq!(p.player_id, p2.player_id);
        assert_eq!(p.sport, p2.sport);
    }

    #[test]
    fn test_serde_roundtrip_team_stats() {
        let t = TeamStats::new("t1", "Test", Sport::NFL, "2024");
        let json = serde_json::to_string(&t).unwrap();
        let t2: TeamStats = serde_json::from_str(&json).unwrap();
        assert_eq!(t.team_id, t2.team_id);
    }

    #[test]
    fn test_serde_roundtrip_betting_line() {
        let bl = BettingLine::new("g1");
        let json = serde_json::to_string(&bl).unwrap();
        let bl2: BettingLine = serde_json::from_str(&json).unwrap();
        assert_eq!(bl.game_id, bl2.game_id);
    }
}
