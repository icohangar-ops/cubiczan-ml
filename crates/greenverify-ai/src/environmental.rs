use crate::types::{EnvironmentalMetric, ESGRating};
use std::collections::HashMap;

/// Environmental scoring engine.
#[derive(Debug, Clone)]
pub struct EnvironmentalScorer {
    industry_benchmarks: HashMap<String, EnvironmentalBenchmark>,
}

/// Industry-specific environmental benchmarks.
#[derive(Debug, Clone)]
pub struct EnvironmentalBenchmark {
    pub avg_carbon_per_revenue: f64,
    pub avg_renewable_pct: f64,
    pub avg_water_per_revenue: f64,
    pub avg_waste_per_revenue: f64,
}

impl EnvironmentalScorer {
    pub fn new() -> Self {
        let mut benchmarks = HashMap::new();
        benchmarks.insert("technology".into(), EnvironmentalBenchmark {
            avg_carbon_per_revenue: 50.0,
            avg_renewable_pct: 40.0,
            avg_water_per_revenue: 100.0,
            avg_waste_per_revenue: 5.0,
        });
        benchmarks.insert("energy".into(), EnvironmentalBenchmark {
            avg_carbon_per_revenue: 500.0,
            avg_renewable_pct: 20.0,
            avg_water_per_revenue: 500.0,
            avg_waste_per_revenue: 20.0,
        });
        benchmarks.insert("manufacturing".into(), EnvironmentalBenchmark {
            avg_carbon_per_revenue: 200.0,
            avg_renewable_pct: 15.0,
            avg_water_per_revenue: 300.0,
            avg_waste_per_revenue: 30.0,
        });
        benchmarks.insert("finance".into(), EnvironmentalBenchmark {
            avg_carbon_per_revenue: 10.0,
            avg_renewable_pct: 50.0,
            avg_water_per_revenue: 20.0,
            avg_waste_per_revenue: 2.0,
        });
        Self { industry_benchmarks: benchmarks }
    }

    // ─── Carbon Footprint Analysis ─────────────────────────────

    /// Score carbon emissions relative to industry benchmark.
    pub fn carbon_footprint_score(
        &self,
        carbon_per_revenue: f64,
        industry: &str,
    ) -> f64 {
        let benchmark = self.industry_benchmarks.get(industry).cloned().unwrap_or(EnvironmentalBenchmark {
            avg_carbon_per_revenue: 100.0,
            avg_renewable_pct: 30.0,
            avg_water_per_revenue: 200.0,
            avg_waste_per_revenue: 10.0,
        });

        if benchmark.avg_carbon_per_revenue < 1e-9 {
            return 50.0;
        }

        let ratio = carbon_per_revenue / benchmark.avg_carbon_per_revenue;
        // Below industry avg -> higher score
        if ratio <= 0.25 {
            95.0
        } else if ratio <= 0.5 {
            85.0
        } else if ratio <= 0.75 {
            75.0
        } else if ratio <= 1.0 {
            65.0
        } else if ratio <= 1.5 {
            50.0
        } else if ratio <= 2.0 {
            35.0
        } else {
            20.0
        }
    }

    /// Per-employee carbon analysis.
    pub fn carbon_per_employee_score(carbon_per_employee: f64) -> f64 {
        match carbon_per_employee {
            e if e <= 2.0 => 95.0,
            e if e <= 5.0 => 85.0,
            e if e <= 10.0 => 70.0,
            e if e <= 20.0 => 55.0,
            e if e <= 50.0 => 40.0,
            _ => 25.0,
        }
    }

    // ─── Energy Transition Scoring ─────────────────────────────

    /// Score based on renewable energy percentage.
    pub fn energy_transition_score(
        &self,
        renewable_pct: f64,
        industry: &str,
    ) -> f64 {
        let benchmark = self.industry_benchmarks.get(industry).cloned().unwrap_or(EnvironmentalBenchmark {
            avg_carbon_per_revenue: 100.0,
            avg_renewable_pct: 30.0,
            avg_water_per_revenue: 200.0,
            avg_waste_per_revenue: 10.0,
        });

        let industry_avg = benchmark.avg_renewable_pct;
        // Score based on how far above industry average
        if renewable_pct >= 90.0 {
            95.0
        } else if renewable_pct >= 75.0 {
            85.0
        } else if renewable_pct >= 50.0 {
            75.0
        } else if renewable_pct >= industry_avg {
            65.0
        } else if renewable_pct >= industry_avg * 0.5 {
            50.0
        } else {
            30.0
        }
    }

    /// Fossil fuel vs renewable trajectory.
    pub fn energy_trajectory_score(
        current_renewable: f64,
        previous_renewable: f64,
    ) -> f64 {
        let improvement = current_renewable - previous_renewable;
        let base = match current_renewable {
            r if r >= 75.0 => 80.0,
            r if r >= 50.0 => 60.0,
            r if r >= 25.0 => 40.0,
            _ => 20.0,
        };
        let trajectory_bonus = improvement * 0.5;
        (base + trajectory_bonus).clamp(0.0, 100.0)
    }

    // ─── Water Stress Assessment ───────────────────────────────

    /// Score water usage efficiency.
    pub fn water_stress_score(
        &self,
        water_per_revenue: f64,
        industry: &str,
    ) -> f64 {
        let benchmark = self.industry_benchmarks.get(industry).cloned().unwrap_or(EnvironmentalBenchmark {
            avg_carbon_per_revenue: 100.0,
            avg_renewable_pct: 30.0,
            avg_water_per_revenue: 200.0,
            avg_waste_per_revenue: 10.0,
        });

        let ratio = water_per_revenue / benchmark.avg_water_per_revenue.max(1.0);
        if ratio <= 0.3 {
            95.0
        } else if ratio <= 0.6 {
            80.0
        } else if ratio <= 1.0 {
            65.0
        } else if ratio <= 1.5 {
            45.0
        } else {
            25.0
        }
    }

    // ─── Circular Economy Metrics ──────────────────────────────

    /// Score waste management efficiency.
    pub fn circular_economy_score(
        &self,
        waste_per_revenue: f64,
        recycling_rate: f64,
        industry: &str,
    ) -> f64 {
        let benchmark = self.industry_benchmarks.get(industry).cloned().unwrap_or(EnvironmentalBenchmark {
            avg_carbon_per_revenue: 100.0,
            avg_renewable_pct: 30.0,
            avg_water_per_revenue: 200.0,
            avg_waste_per_revenue: 10.0,
        });

        let waste_ratio = waste_per_revenue / benchmark.avg_waste_per_revenue.max(1.0);
        let waste_score = if waste_ratio <= 0.3 { 90.0 }
            else if waste_ratio <= 0.7 { 70.0 }
            else if waste_ratio <= 1.0 { 55.0 }
            else { 35.0 };

        let recycle_score = recycling_rate * 0.3; // up to 30 bonus points
        (waste_score + recycle_score).min(100.0)
    }

    // ─── Biodiversity Impact ───────────────────────────────────

    /// Score biodiversity impact (lower = better).
    pub fn biodiversity_score(impact: f64) -> f64 {
        // impact is 0-100 where higher = worse
        (100.0 - impact).clamp(0.0, 100.0)
    }

    // ─── Science-Based Targets ─────────────────────────────────

    /// Check alignment with Science-Based Targets initiative.
    pub fn sbt_alignment_score(
        has_target: bool,
        target_year: u32,
        current_trajectory_on_track: bool,
    ) -> f64 {
        if !has_target {
            return 20.0; // No target at all
        }

        let target_score = if target_year <= 2030 {
            40.0 // Ambitious
        } else if target_year <= 2040 {
            30.0
        } else if target_year <= 2050 {
            20.0
        } else {
            10.0
        };

        let trajectory_score = if current_trajectory_on_track { 40.0 } else { 15.0 };

        target_score + trajectory_score
    }

    // ─── Composite Environmental Score ─────────────────────────

    /// Compute overall environmental score.
    pub fn composite_environmental_score(
        &self,
        metrics: &EnvironmentalMetric,
        revenue: f64,
        employees: u32,
        industry: &str,
        recycling_rate: f64,
    ) -> f64 {
        let carbon_rev = metrics.carbon_intensity_per_revenue(revenue);
        let carbon_emp = metrics.carbon_intensity_per_employee(employees);

        let carbon_score = self.carbon_footprint_score(carbon_rev, industry) * 0.30;
        let energy_score = self.energy_transition_score(metrics.renewable_pct, industry) * 0.20;
        let water_score = self.water_stress_score(
            if revenue > 0.0 { metrics.water_usage / (revenue / 1e6) } else { 999.0 },
            industry,
        ) * 0.15;
        let waste_score = self.circular_economy_score(
            if revenue > 0.0 { metrics.waste_generated / (revenue / 1e6) } else { 999.0 },
            recycling_rate,
            industry,
        ) * 0.15;
        let bio_score = Self::biodiversity_score(metrics.biodiversity_impact) * 0.10;
        let carbon_emp_score = Self::carbon_per_employee_score(carbon_emp) * 0.10;

        carbon_score + energy_score + water_score + waste_score + bio_score + carbon_emp_score
    }

    /// Generate detailed environmental scoring breakdown.
    pub fn detailed_score(
        &self,
        metrics: &EnvironmentalMetric,
        revenue: f64,
        employees: u32,
        industry: &str,
        recycling_rate: f64,
    ) -> EnvironmentalScoreBreakdown {
        let overall = self.composite_environmental_score(
            metrics, revenue, employees, industry, recycling_rate,
        );
        let rating = ESGRating::from_score(overall);

        EnvironmentalScoreBreakdown {
            overall,
            rating,
            carbon_footprint: self.carbon_footprint_score(
                metrics.carbon_intensity_per_revenue(revenue), industry,
            ),
            energy_transition: self.energy_transition_score(metrics.renewable_pct, industry),
            water_stress: self.water_stress_score(
                if revenue > 0.0 { metrics.water_usage / (revenue / 1e6) } else { 999.0 },
                industry,
            ),
            circular_economy: self.circular_economy_score(
                if revenue > 0.0 { metrics.waste_generated / (revenue / 1e6) } else { 999.0 },
                recycling_rate,
                industry,
            ),
            biodiversity: Self::biodiversity_score(metrics.biodiversity_impact),
        }
    }
}

impl Default for EnvironmentalScorer {
    fn default() -> Self {
        Self::new()
    }
}

/// Detailed environmental scoring breakdown.
#[derive(Debug, Clone)]
pub struct EnvironmentalScoreBreakdown {
    pub overall: f64,
    pub rating: ESGRating,
    pub carbon_footprint: f64,
    pub energy_transition: f64,
    pub water_stress: f64,
    pub circular_economy: f64,
    pub biodiversity: f64,
}

#[cfg(test)]
mod tests {
    use super::*;

    fn sample_metrics() -> EnvironmentalMetric {
        EnvironmentalMetric {
            carbon_emissions: 50000.0,
            energy_consumption: 200000.0,
            water_usage: 100000.0,
            waste_generated: 5000.0,
            renewable_pct: 60.0,
            biodiversity_impact: 30.0,
        }
    }

    #[test]
    fn test_carbon_footprint_above_avg() {
        let scorer = EnvironmentalScorer::new();
        let score = scorer.carbon_footprint_score(100.0, "technology");
        // technology benchmark is 50, ratio = 2.0 -> 35
        assert_eq!(score, 35.0);
    }

    #[test]
    fn test_carbon_footprint_below_avg() {
        let scorer = EnvironmentalScorer::new();
        let score = scorer.carbon_footprint_score(25.0, "technology");
        // ratio = 0.5 -> 85
        assert_eq!(score, 85.0);
    }

    #[test]
    fn test_carbon_footprint_unknown_industry() {
        let scorer = EnvironmentalScorer::new();
        let score = scorer.carbon_footprint_score(50.0, "unknown");
        // default benchmark 100, ratio = 0.5 -> 85
        assert_eq!(score, 85.0);
    }

    #[test]
    fn test_carbon_per_employee_score() {
        assert_eq!(EnvironmentalScorer::carbon_per_employee_score(1.0), 95.0);
        assert_eq!(EnvironmentalScorer::carbon_per_employee_score(15.0), 55.0);
        assert_eq!(EnvironmentalScorer::carbon_per_employee_score(100.0), 25.0);
    }

    #[test]
    fn test_energy_transition_high_renewable() {
        let scorer = EnvironmentalScorer::new();
        let score = scorer.energy_transition_score(95.0, "technology");
        assert_eq!(score, 95.0);
    }

    #[test]
    fn test_energy_transition_low_renewable() {
        let scorer = EnvironmentalScorer::new();
        let score = scorer.energy_transition_score(10.0, "technology");
        // technology avg = 40, 10 < 40*0.5=20 -> 30
        assert_eq!(score, 30.0);
    }

    #[test]
    fn test_energy_trajectory_improving() {
        let score = EnvironmentalScorer::energy_trajectory_score(60.0, 40.0);
        // base = 60, bonus = 20*0.5 = 10 -> 70
        assert_eq!(score, 70.0);
    }

    #[test]
    fn test_energy_trajectory_declining() {
        let score = EnvironmentalScorer::energy_trajectory_score(30.0, 50.0);
        // base = 40, bonus = -20*0.5 = -10 -> 30
        assert_eq!(score, 30.0);
    }

    #[test]
    fn test_water_stress_score() {
        let scorer = EnvironmentalScorer::new();
        let score = scorer.water_stress_score(50.0, "technology");
        // tech benchmark = 100, ratio = 0.5 -> 80
        assert_eq!(score, 80.0);
    }

    #[test]
    fn test_circular_economy_score() {
        let scorer = EnvironmentalScorer::new();
        let score = scorer.circular_economy_score(5.0, 80.0, "technology");
        // tech benchmark = 5, ratio = 1.0 -> 55, recycle = 80*0.3 = 24 -> 79
        assert_eq!(score, 79.0);
    }

    #[test]
    fn test_circular_economy_high_waste() {
        let scorer = EnvironmentalScorer::new();
        let score = scorer.circular_economy_score(20.0, 50.0, "technology");
        // ratio = 4.0 -> 35, recycle = 15 -> 50
        assert_eq!(score, 50.0);
    }

    #[test]
    fn test_biodiversity_score() {
        assert!((EnvironmentalScorer::biodiversity_score(10.0) - 90.0).abs() < 1e-9);
        assert!((EnvironmentalScorer::biodiversity_score(50.0) - 50.0).abs() < 1e-9);
        assert!((EnvironmentalScorer::biodiversity_score(100.0) - 0.0).abs() < 1e-9);
    }

    #[test]
    fn test_sbt_alignment_no_target() {
        assert_eq!(EnvironmentalScorer::sbt_alignment_score(false, 2050, true), 20.0);
    }

    #[test]
    fn test_sbt_alignment_on_track() {
        let score = EnvironmentalScorer::sbt_alignment_score(true, 2030, true);
        // target = 40, trajectory = 40 -> 80
        assert_eq!(score, 80.0);
    }

    #[test]
    fn test_sbt_alignment_off_track() {
        let score = EnvironmentalScorer::sbt_alignment_score(true, 2050, false);
        // target = 20, trajectory = 15 -> 35
        assert_eq!(score, 35.0);
    }

    #[test]
    fn test_composite_environmental_score() {
        let scorer = EnvironmentalScorer::new();
        let m = sample_metrics();
        let score = scorer.composite_environmental_score(&m, 1e9, 5000, "technology", 70.0);
        assert!(score > 0.0);
        assert!(score <= 100.0);
    }

    #[test]
    fn test_detailed_score() {
        let scorer = EnvironmentalScorer::new();
        let m = sample_metrics();
        let breakdown = scorer.detailed_score(&m, 1e9, 5000, "technology", 70.0);
        assert!(breakdown.overall > 0.0);
        assert!(breakdown.carbon_footprint > 0.0);
        assert!(breakdown.energy_transition > 0.0);
    }

    #[test]
    fn test_default_scorer() {
        let scorer = EnvironmentalScorer::default();
        let _ = scorer.industry_benchmarks.get("technology");
    }
}
