use crate::types::{PlayerProjection, PlayerStats, Sport};
use std::collections::HashMap;

/// Player analysis engine for performance projection and comparison.
#[derive(Debug, Clone)]
pub struct PlayerAnalyzer {
    league_avg_ppg: f64,
    league_avg_rpg: f64,
    league_avg_apg: f64,
    regression_weight: f64,
}

impl PlayerAnalyzer {
    pub fn new() -> Self {
        Self {
            league_avg_ppg: 15.0,
            league_avg_rpg: 5.0,
            league_avg_apg: 3.5,
            regression_weight: 0.3,
        }
    }

    pub fn with_league_averages(ppg: f64, rpg: f64, apg: f64) -> Self {
        Self {
            league_avg_ppg: ppg,
            league_avg_rpg: rpg,
            league_avg_apg: apg,
            regression_weight: 0.3,
        }
    }

    pub fn with_regression_weight(w: f64) -> Self {
        Self {
            league_avg_ppg: 15.0,
            league_avg_rpg: 5.0,
            league_avg_apg: 3.5,
            regression_weight: w.clamp(0.0, 1.0),
        }
    }

    // ─── Performance Projection ────────────────────────────────

    /// Project a player's performance with ceiling, floor, and expected.
    pub fn project(&self, player: &PlayerStats) -> PlayerProjection {
        let ppg = player.ppg();
        let rpg = player.rpg();
        let apg = player.apg();

        // Apply regression to mean based on sample size
        let sample_factor = self.sample_factor(player.games_played);
        let expected_ppg = ppg * sample_factor + self.league_avg_ppg * (1.0 - sample_factor);
        let expected_rpg = rpg * sample_factor + self.league_avg_rpg * (1.0 - sample_factor);
        let expected_apg = apg * sample_factor + self.league_avg_apg * (1.0 - sample_factor);

        // Compute standard deviation proxy from advanced stats
        let variance = self.estimate_variance(player);

        PlayerProjection {
            player_id: player.player_id.clone(),
            predicted_points: expected_ppg,
            predicted_rebounds: expected_rpg,
            predicted_assists: expected_apg,
            ceiling: expected_ppg + variance,
            floor: (expected_ppg - variance).max(0.0),
            confidence: self.compute_confidence(player),
        }
    }

    /// Sample size factor for regression to mean.
    fn sample_factor(&self, games: u32) -> f64 {
        let effective_games = (games as f64 * 0.7).min(82.0); // account for rotation/bench
        effective_games / (effective_games + 20.0) // James-Stein style
    }

    /// Estimate performance variance from available data.
    fn estimate_variance(&self, player: &PlayerStats) -> f64 {
        if player.games_played == 0 {
            return self.league_avg_ppg * 0.3;
        }
        let ppg = player.ppg();
        // Higher usage -> higher variance
        let usage = player
            .advanced_stats
            .get("usage_rate")
            .copied()
            .unwrap_or(20.0);
        let base_variance = ppg * 0.2 * (usage / 20.0).max(0.5);
        base_variance
    }

    /// Confidence in projection based on sample size and consistency.
    fn compute_confidence(&self, player: &PlayerStats) -> f64 {
        if player.games_played == 0 {
            return 0.1;
        }
        let sample_conf = self.sample_factor(player.games_played);
        let efficiency_conf = if player.efficiency_rating > 0.0 {
            (player.efficiency_rating / 30.0).min(1.0) * 0.3
        } else {
            0.0
        };
        (sample_conf * 0.7 + efficiency_conf).clamp(0.1, 0.95)
    }

    // ─── Usage vs Efficiency ───────────────────────────────────

    /// Analyze usage rate vs efficiency tradeoff.
    /// Returns (usage_tier, efficiency_tier, recommendation).
    pub fn usage_efficiency_analysis(
        &self,
        player: &PlayerStats,
    ) -> UsageEfficiencyResult {
        let usage = player
            .advanced_stats
            .get("usage_rate")
            .copied()
            .unwrap_or(20.0);
        let eff = player.efficiency_rating;

        let usage_tier = match usage {
            u if u < 15.0 => "low",
            u if u < 25.0 => "moderate",
            u if u < 32.0 => "high",
            _ => "elite",
        };

        let eff_tier = match eff {
            e if e < 10.0 => "inefficient",
            e if e < 18.0 => "below_average",
            e if e < 25.0 => "average",
            e if e < 30.0 => "good",
            _ => "elite",
        };

        let recommendation = match (usage_tier, eff_tier) {
            ("low", "elite") => "Consider increasing usage — player is highly efficient with low volume",
            ("low", "good" | "average") => "Moderate usage increase could improve team output",
            ("moderate", "elite" | "good") => "Optimal zone — maintain current usage",
            ("moderate", "average" | "below_average") => "Monitor efficiency at current usage",
            ("high", "elite") => "Star usage — player carries offense efficiently",
            ("high", "good" | "average") => "High usage with acceptable efficiency",
            ("high", "below_average" | "inefficient") => "Red flag — consider reducing usage",
            ("elite", "elite") => "MVP-caliber usage-efficiency profile",
            ("elite", "good" | "average") => "Volume scorer — team dependent on this player",
            ("elite", "below_average" | "inefficient") => "Concerning — extremely high usage without efficiency",
            _ => "Standard profile",
        };

        UsageEfficiencyResult {
            usage_rate: usage,
            efficiency_rating: eff,
            usage_tier: usage_tier.to_string(),
            efficiency_tier: eff_tier.to_string(),
            recommendation: recommendation.to_string(),
        }
    }

    // ─── Matchup Adjustment ────────────────────────────────────

    /// Adjust player projection based on opponent quality.
    pub fn matchup_adjustment(
        &self,
        player: &PlayerStats,
        opponent_def_rating: f64, // 0-100 where lower is better defense
        position_difficulty: f64, // 0.0-1.0, how well defender matches position
    ) -> f64 {
        // League average defensive rating
        let avg_def = 50.0;
        let def_factor = opponent_def_rating / avg_def;
        // Higher def rating = worse defense = player performs better
        let adjustment = 1.0 + (def_factor - 1.0) * 0.3;

        // Position difficulty reduces performance
        let pos_factor = 1.0 - (position_difficulty - 0.5) * 0.2;

        adjustment * pos_factor
    }

    /// Generate matchup-adjusted projection.
    pub fn matchup_project(
        &self,
        player: &PlayerStats,
        opponent_def_rating: f64,
        position_difficulty: f64,
    ) -> PlayerProjection {
        let base = self.project(player);
        let adj = self.matchup_adjustment(player, opponent_def_rating, position_difficulty);
        PlayerProjection {
            predicted_points: base.predicted_points * adj,
            predicted_rebounds: base.predicted_rebounds * (1.0 + (adj - 1.0) * 0.5),
            predicted_assists: base.predicted_assists * (1.0 + (adj - 1.0) * 0.3),
            ceiling: base.ceiling * adj,
            floor: (base.floor * (2.0 - adj)).max(0.0),
            ..base
        }
    }

    // ─── Fatigue/Rest Modeling ─────────────────────────────────

    /// Model fatigue impact from rest days.
    pub fn fatigue_factor(rest_days: u32, back_to_back: bool, travel: bool) -> f64 {
        let mut factor: f64 = 1.0;

        // Rest days impact
        match rest_days {
            0 => factor *= 0.85,   // Playing same day (unusual)
            1 => factor *= 0.90,   // Back to back
            2 => factor *= 0.95,   // 1 day rest
            3 => factor *= 1.00,   // Normal rest
            4..=6 => factor *= 1.02, // Extra rest
            _ => factor *= 1.05,   // Extended rest (may have rust)
        }

        if back_to_back {
            factor *= 0.92;
        }

        if travel {
            factor *= 0.97;
        }

        factor.clamp(0.7, 1.1)
    }

    /// Apply fatigue-adjusted projection.
    pub fn fatigued_projection(
        &self,
        player: &PlayerStats,
        rest_days: u32,
        back_to_back: bool,
        travel: bool,
    ) -> PlayerProjection {
        let base = self.project(player);
        let fatigue = Self::fatigue_factor(rest_days, back_to_back, travel);
        PlayerProjection {
            predicted_points: base.predicted_points * fatigue,
            predicted_rebounds: base.predicted_rebounds * fatigue,
            predicted_assists: base.predicted_assists * fatigue,
            ceiling: base.ceiling * (fatigue * 1.05),
            floor: (base.floor * (fatigue * 0.95)).max(0.0),
            confidence: (base.confidence * (1.0 - (1.0 - fatigue) * 0.3)).clamp(0.1, 0.95),
            ..base
        }
    }

    // ─── Player Comparison ─────────────────────────────────────

    /// Compute similarity score between two players (0.0 to 1.0).
    pub fn similarity_score(&self, a: &PlayerStats, b: &PlayerStats) -> f64 {
        let ppg_sim = self.stat_similarity(a.ppg(), b.ppg());
        let rpg_sim = self.stat_similarity(a.rpg(), b.rpg());
        let apg_sim = self.stat_similarity(a.apg(), b.apg());
        let eff_sim = self.stat_similarity(a.efficiency_rating, b.efficiency_rating);

        (ppg_sim + rpg_sim + apg_sim + eff_sim) / 4.0
    }

    /// Similarity of two stat values on 0-1 scale.
    fn stat_similarity(&self, a: f64, b: f64) -> f64 {
        let max_val = a.max(b).max(1.0);
        let diff = (a - b).abs();
        1.0 - (diff / max_val).min(1.0)
    }

    /// Compare two players and return a detailed breakdown.
    pub fn compare_players(&self, a: &PlayerStats, b: &PlayerStats) -> PlayerComparison {
        let similarity = self.similarity_score(a, b);
        let ppg_adv = a.ppg() - b.ppg();
        let rpg_adv = a.rpg() - b.rpg();
        let apg_adv = a.apg() - b.apg();
        let eff_adv = a.efficiency_rating - b.efficiency_rating;
        let pm_adv = a.plus_minus - b.plus_minus;

        let better_overall = if (ppg_adv + rpg_adv + apg_adv + eff_adv * 0.5) > 0.0 {
            &a.name
        } else {
            &b.name
        };

        PlayerComparison {
            player_a: a.name.clone(),
            player_b: b.name.clone(),
            similarity,
            ppg_advantage: ppg_adv,
            rpg_advantage: rpg_adv,
            apg_advantage: apg_adv,
            efficiency_advantage: eff_adv,
            plus_minus_advantage: pm_adv,
            better_overall: better_overall.to_string(),
        }
    }

    // ─── Aging Curve ───────────────────────────────────────────

    /// Simple aging curve adjustment factor.
    pub fn aging_adjustment(age: u32) -> f64 {
        match age {
            0..=19 => 0.70,
            20..=21 => 0.82,
            22..=23 => 0.92,
            24..=26 => 1.00,  // Prime entry
            27..=29 => 1.00,  // Peak
            30..=31 => 0.96,
            32..=33 => 0.90,
            34..=35 => 0.83,
            36..=37 => 0.75,
            _ => 0.65,
        }
    }
}

impl Default for PlayerAnalyzer {
    fn default() -> Self {
        Self::new()
    }
}

/// Result of usage vs efficiency analysis.
#[derive(Debug, Clone)]
pub struct UsageEfficiencyResult {
    pub usage_rate: f64,
    pub efficiency_rating: f64,
    pub usage_tier: String,
    pub efficiency_tier: String,
    pub recommendation: String,
}

/// Detailed comparison between two players.
#[derive(Debug, Clone)]
pub struct PlayerComparison {
    pub player_a: String,
    pub player_b: String,
    pub similarity: f64,
    pub ppg_advantage: f64,
    pub rpg_advantage: f64,
    pub apg_advantage: f64,
    pub efficiency_advantage: f64,
    pub plus_minus_advantage: f64,
    pub better_overall: String,
}

#[cfg(test)]
mod tests {
    use super::*;

    fn star_player() -> PlayerStats {
        let mut adv = HashMap::new();
        adv.insert("usage_rate".to_string(), 30.0);
        adv.insert("minutes".to_string(), 2400.0);
        PlayerStats {
            player_id: "p1".into(),
            name: "Star".into(),
            sport: Sport::NBA,
            season: "2024".into(),
            games_played: 75,
            points: 1875.0, // 25 ppg
            rebounds: 525.0, // 7 rpg
            assists: 525.0, // 7 apg
            efficiency_rating: 27.0,
            plus_minus: 250.0,
            advanced_stats: adv,
        }
    }

    fn role_player() -> PlayerStats {
        PlayerStats {
            player_id: "p2".into(),
            name: "Role".into(),
            sport: Sport::NBA,
            season: "2024".into(),
            games_played: 60,
            points: 600.0,  // 10 ppg
            rebounds: 300.0, // 5 rpg
            assists: 180.0, // 3 apg
            efficiency_rating: 15.0,
            plus_minus: 50.0,
            advanced_stats: HashMap::new(),
        }
    }

    #[test]
    fn test_project_star() {
        let a = PlayerAnalyzer::new();
        let proj = a.project(&star_player());
        assert!(proj.predicted_points > 0.0);
        assert!(proj.ceiling > proj.predicted_points);
        assert!(proj.floor < proj.predicted_points);
        assert!(proj.floor >= 0.0);
    }

    #[test]
    fn test_project_zero_games() {
        let a = PlayerAnalyzer::new();
        let p = PlayerStats::new("p1", "Empty", Sport::NBA, "2024");
        let proj = a.project(&p);
        // Should regress to league average
        assert!((proj.predicted_points - a.league_avg_ppg).abs() < 0.01);
    }

    #[test]
    fn test_regression_to_mean() {
        let a = PlayerAnalyzer::with_league_averages(20.0, 6.0, 4.0);
        let p = PlayerStats {
            player_id: "p1".into(),
            name: "Rookie".into(),
            sport: Sport::NBA,
            season: "2024".into(),
            games_played: 5, // Small sample
            points: 150.0,   // 30 ppg - likely unsustainable
            rebounds: 30.0,  // 6 rpg
            assists: 20.0,   // 4 apg
            efficiency_rating: 25.0,
            plus_minus: 10.0,
            advanced_stats: HashMap::new(),
        };
        let proj = a.project(&p);
        // Regression should pull toward league average
        assert!(proj.predicted_points < 30.0);
        assert!(proj.predicted_points > 20.0); // But still above average
    }

    #[test]
    fn test_usage_efficiency_star() {
        let a = PlayerAnalyzer::new();
        let result = a.usage_efficiency_analysis(&star_player());
        assert_eq!(result.usage_tier, "high");
        assert_eq!(result.efficiency_tier, "good");
        assert!(!result.recommendation.is_empty());
    }

    #[test]
    fn test_usage_efficiency_role() {
        let a = PlayerAnalyzer::new();
        let result = a.usage_efficiency_analysis(&role_player());
        assert_eq!(result.usage_tier, "moderate");
        assert_eq!(result.efficiency_tier, "below_average");
    }

    #[test]
    fn test_usage_efficiency_low_usage() {
        let a = PlayerAnalyzer::new();
        let mut adv = HashMap::new();
        adv.insert("usage_rate".to_string(), 10.0);
        let p = PlayerStats {
            player_id: "p1".into(),
            name: "Bench".into(),
            sport: Sport::NBA,
            season: "2024".into(),
            games_played: 82,
            points: 492.0, // 6 ppg
            rebounds: 246.0, // 3 rpg
            assists: 82.0,  // 1 apg
            efficiency_rating: 32.0, // elite efficiency
            plus_minus: 100.0,
            advanced_stats: adv,
        };
        let result = a.usage_efficiency_analysis(&p);
        assert_eq!(result.usage_tier, "low");
        assert_eq!(result.efficiency_tier, "elite");
        assert!(result.recommendation.contains("increasing usage"));
    }

    #[test]
    fn test_matchup_adjustment_avg_defense() {
        let a = PlayerAnalyzer::new();
        let adj = a.matchup_adjustment(&star_player(), 50.0, 0.5);
        // Average defense, neutral position match -> ~1.0
        assert!((adj - 1.0).abs() < 0.01);
    }

    #[test]
    fn test_matchup_adjustment_weak_defense() {
        let a = PlayerAnalyzer::new();
        let adj_weak = a.matchup_adjustment(&star_player(), 65.0, 0.5);
        let adj_strong = a.matchup_adjustment(&star_player(), 35.0, 0.5);
        assert!(adj_weak > adj_strong); // Better stats vs weak defense
    }

    #[test]
    fn test_matchup_project() {
        let a = PlayerAnalyzer::new();
        let proj = a.matchup_project(&star_player(), 60.0, 0.8);
        assert!(proj.predicted_points > 0.0);
        assert!(proj.ceiling > proj.predicted_points);
    }

    #[test]
    fn test_fatigue_factor_normal_rest() {
        let f = PlayerAnalyzer::fatigue_factor(3, false, false);
        assert!((f - 1.0).abs() < 1e-9);
    }

    #[test]
    fn test_fatigue_factor_back_to_back() {
        let f = PlayerAnalyzer::fatigue_factor(1, true, false);
        // 1 day = 0.90, B2B = 0.92 -> 0.90 * 0.92 = 0.828
        assert!((f - 0.828).abs() < 0.01);
    }

    #[test]
    fn test_fatigue_factor_extended_rest() {
        let f = PlayerAnalyzer::fatigue_factor(7, false, false);
        assert!(f > 1.0);
    }

    #[test]
    fn test_fatigue_factor_travel() {
        let f = PlayerAnalyzer::fatigue_factor(3, false, true);
        assert!(f < 1.0);
    }

    #[test]
    fn test_fatigued_projection() {
        let a = PlayerAnalyzer::new();
        let proj = a.fatigued_projection(&star_player(), 1, true, false);
        assert!(proj.predicted_points > 0.0);
        // Fatigued should be less than base projection
        let base = a.project(&star_player());
        assert!(proj.predicted_points < base.predicted_points);
    }

    #[test]
    fn test_similarity_identical() {
        let a = PlayerAnalyzer::new();
        let p = star_player();
        let sim = a.similarity_score(&p, &p);
        assert!((sim - 1.0).abs() < 1e-9);
    }

    #[test]
    fn test_similarity_different() {
        let a = PlayerAnalyzer::new();
        let sim = a.similarity_score(&star_player(), &role_player());
        assert!(sim < 1.0);
        assert!(sim > 0.0);
    }

    #[test]
    fn test_compare_players() {
        let a = PlayerAnalyzer::new();
        let comp = a.compare_players(&star_player(), &role_player());
        assert_eq!(comp.player_a, "Star");
        assert_eq!(comp.player_b, "Role");
        assert!(comp.ppg_advantage > 0.0); // Star has higher PPG
        assert_eq!(comp.better_overall, "Star");
    }

    #[test]
    fn test_aging_curve_prime() {
        let f25 = PlayerAnalyzer::aging_adjustment(25);
        let f28 = PlayerAnalyzer::aging_adjustment(28);
        assert!((f25 - 1.0).abs() < 1e-9);
        assert!((f28 - 1.0).abs() < 1e-9);
    }

    #[test]
    fn test_aging_curve_decline() {
        let f35 = PlayerAnalyzer::aging_adjustment(35);
        let f22 = PlayerAnalyzer::aging_adjustment(22);
        assert!(f35 < f22);
        assert!(f35 < 1.0);
    }

    #[test]
    fn test_aging_curve_young() {
        let f19 = PlayerAnalyzer::aging_adjustment(19);
        assert!(f19 < 1.0);
    }

    #[test]
    fn test_default_analyzer() {
        let a = PlayerAnalyzer::default();
        assert!((a.league_avg_ppg - 15.0).abs() < 1e-9);
    }

    #[test]
    fn test_with_regression_weight() {
        let a = PlayerAnalyzer::with_regression_weight(0.5);
        assert!((a.regression_weight - 0.5).abs() < 1e-9);
    }

    #[test]
    fn test_regression_weight_clamped() {
        let a = PlayerAnalyzer::with_regression_weight(2.0);
        assert!((a.regression_weight - 1.0).abs() < 1e-9);
    }
}
