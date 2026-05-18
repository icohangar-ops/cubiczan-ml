// ─── Integration KPI engine ──────────────────────────────────────────────────

use crate::types::{HealthStatus, KPI, KPICategory};
use std::collections::HashMap;

/// A synergy tracking entry for cost or revenue synergies.
#[derive(Debug, Clone)]
pub struct SynergyEntry {
    pub category: KPICategory,
    pub planned_annual: f64,
    pub realized_to_date: f64,
    pub run_rate: f64,
    pub percent_achieved: f64,
}

/// Integration velocity snapshot.
#[derive(Debug, Clone)]
pub struct VelocityMetrics {
    pub milestones_per_week: f64,
    pub tasks_per_week: f64,
    pub avg_days_to_complete: f64,
    pub velocity_trend: VelocityTrend,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, serde::Serialize)]
pub enum VelocityTrend {
    Accelerating,
    Stable,
    Decelerating,
}

/// Retention metrics for employees or customers.
#[derive(Debug, Clone)]
pub struct RetentionMetrics {
    pub baseline_count: usize,
    pub current_count: usize,
    pub departed: usize,
    pub retention_rate: f64, // 0.0 – 100.0
    pub status: HealthStatus,
}

/// Result of trend analysis on a KPI's history.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct TrendAnalysis {
    pub kpi_name: String,
    pub history_len: usize,
    pub mean: f64,
    pub std_dev: f64,
    pub slope: f64,          // linear regression slope
    pub direction: TrendDirection,
    pub forecast: Option<f64>, // simple linear extrapolation
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub enum TrendDirection {
    Improving,
    Flat,
    Declining,
}

/// Compute the mean of a slice of f64.
fn mean(data: &[f64]) -> f64 {
    if data.is_empty() {
        return 0.0;
    }
    data.iter().sum::<f64>() / data.len() as f64
}

/// Compute the population standard deviation of a slice of f64.
fn std_dev(data: &[f64]) -> f64 {
    if data.len() < 2 {
        return 0.0;
    }
    let m = mean(data);
    let variance = data.iter().map(|x| (x - m).powi(2)).sum::<f64>() / data.len() as f64;
    variance.sqrt()
}

/// Compute a synergy entry from planned and realized values.
pub fn compute_synergy(planned_annual: f64, realized_to_date: f64) -> SynergyEntry {
    let percent_achieved = if planned_annual.abs() > f64::EPSILON {
        (realized_to_date / planned_annual) * 100.0
    } else {
        0.0
    };
    // Assume realized_to_date spans roughly half a year for run-rate
    let run_rate = realized_to_date * 2.0;
    SynergyEntry {
        category: KPICategory::SynergyCost,
        planned_annual,
        realized_to_date,
        run_rate,
        percent_achieved,
    }
}

/// Compute revenue synergy entry.
pub fn compute_revenue_synergy(planned_annual: f64, realized_to_date: f64) -> SynergyEntry {
    let mut entry = compute_synergy(planned_annual, realized_to_date);
    entry.category = KPICategory::SynergyRevenue;
    entry
}

/// Compute integration velocity metrics from milestone completion data.
/// `completed_milestones` is a list of milestone counts completed per week.
pub fn compute_velocity(completed_milestones: &[f64], days_to_complete: &[f64]) -> VelocityMetrics {
    let milestones_per_week = mean(completed_milestones);
    let avg_days = mean(days_to_complete);

    let velocity_trend = if completed_milestones.len() < 3 {
        VelocityTrend::Stable
    } else {
        let first_half: f64 = completed_milestones[..completed_milestones.len() / 2].iter().sum();
        let second_half: f64 = completed_milestones[completed_milestones.len() / 2..]
            .iter()
            .sum();
        let first_avg = first_half / (completed_milestones.len() / 2) as f64;
        let second_avg = second_half / (completed_milestones.len() - completed_milestones.len() / 2) as f64;
        if second_avg > first_avg * 1.1 {
            VelocityTrend::Accelerating
        } else if second_avg < first_avg * 0.9 {
            VelocityTrend::Decelerating
        } else {
            VelocityTrend::Stable
        }
    };

    VelocityMetrics {
        milestones_per_week,
        tasks_per_week: milestones_per_week * 5.0, // rough estimate
        avg_days_to_complete: avg_days,
        velocity_trend,
    }
}

/// Compute employee retention metrics.
pub fn compute_employee_retention(baseline_count: usize, current_count: usize) -> RetentionMetrics {
    let departed = baseline_count.saturating_sub(current_count);
    let retention_rate = if baseline_count > 0 {
        (current_count as f64 / baseline_count as f64) * 100.0
    } else {
        100.0
    };
    let status = if retention_rate >= 90.0 {
        HealthStatus::OnTrack
    } else if retention_rate >= 80.0 {
        HealthStatus::AtRisk
    } else {
        HealthStatus::Critical
    };

    RetentionMetrics {
        baseline_count,
        current_count,
        departed,
        retention_rate,
        status,
    }
}

/// Compute customer retention metrics (same logic with different naming).
pub fn compute_customer_retention(baseline_count: usize, current_count: usize) -> RetentionMetrics {
    compute_employee_retention(baseline_count, current_count)
}

/// Perform trend analysis on a KPI's history using simple linear regression.
pub fn analyze_trend(kpi: &KPI, forecast_steps: usize) -> TrendAnalysis {
    let history = &kpi.history;
    let n = history.len();

    if n < 2 {
        return TrendAnalysis {
            kpi_name: kpi.name.clone(),
            history_len: n,
            mean: kpi.current_value,
            std_dev: 0.0,
            slope: 0.0,
            direction: TrendDirection::Flat,
            forecast: None,
        };
    }

    let m = mean(history);
    let sd = std_dev(history);

    // Simple linear regression: y = a + b*x, where x = 0..n-1
    let n_f = n as f64;
    let sum_x: f64 = (0..n).map(|i| i as f64).sum();
    let sum_y: f64 = history.iter().sum();
    let sum_xy: f64 = history
        .iter()
        .enumerate()
        .map(|(i, y)| i as f64 * y)
        .sum();
    let sum_x2: f64 = (0..n).map(|i| (i as f64) * (i as f64)).sum();

    let denominator = n_f * sum_x2 - sum_x * sum_x;
    let slope = if denominator.abs() > f64::EPSILON {
        (n_f * sum_xy - sum_x * sum_y) / denominator
    } else {
        0.0
    };

    let direction = if slope.abs() < sd * 0.05 {
        TrendDirection::Flat
    } else if slope > 0.0 {
        TrendDirection::Improving
    } else {
        TrendDirection::Declining
    };

    // Simple linear extrapolation
    let forecast = if forecast_steps > 0 {
        let intercept = (sum_y - slope * sum_x) / n_f;
        Some(intercept + slope * (n as f64 + forecast_steps as f64 - 1.0))
    } else {
        None
    };

    TrendAnalysis {
        kpi_name: kpi.name.clone(),
        history_len: n,
        mean: m,
        std_dev: sd,
        slope,
        direction,
        forecast,
    }
}

/// Compute a composite KPI health score (0 – 100) across a set of KPIs.
pub fn compute_kpi_health(kpis: &[KPI]) -> f64 {
    if kpis.is_empty() {
        return 100.0;
    }
    let total_progress: f64 = kpis.iter().map(|k| k.progress()).sum();
    (total_progress / kpis.len() as f64) * 100.0
}

/// Aggregate synergy tracking across multiple KPI categories.
pub fn aggregate_synergies(kpis: &[KPI]) -> HashMap<KPICategory, (f64, f64)> {
    let mut map = HashMap::new();
    for kpi in kpis {
        let entry = map
            .entry(kpi.category.clone())
            .or_insert((0.0_f64, 0.0_f64));
        entry.0 += kpi.current_value;
        entry.1 += kpi.target_value;
    }
    map
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_compute_synergy() {
        let s = compute_synergy(100.0, 45.0);
        assert!((s.percent_achieved - 45.0).abs() < 0.01);
        assert!((s.run_rate - 90.0).abs() < 0.01);
    }

    #[test]
    fn test_compute_revenue_synergy() {
        let s = compute_revenue_synergy(200.0, 50.0);
        assert_eq!(s.category, KPICategory::SynergyRevenue);
        assert!((s.percent_achieved - 25.0).abs() < 0.01);
    }

    #[test]
    fn test_synergy_zero_planned() {
        let s = compute_synergy(0.0, 10.0);
        assert!((s.percent_achieved - 0.0).abs() < 0.01);
    }

    #[test]
    fn test_velocity_basic() {
        let completed = vec![2.0, 3.0, 4.0, 3.0, 5.0];
        let days = vec![14.0, 10.0, 7.0, 12.0, 8.0];
        let v = compute_velocity(&completed, &days);
        assert!((v.milestones_per_week - 3.4).abs() < 0.1);
        assert!((v.avg_days_to_complete - 10.2).abs() < 0.1);
    }

    #[test]
    fn test_velocity_trend_accelerating() {
        let completed = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
        let v = compute_velocity(&completed, &[]);
        assert_eq!(v.velocity_trend, VelocityTrend::Accelerating);
    }

    #[test]
    fn test_velocity_trend_decelerating() {
        let completed = vec![6.0, 5.0, 4.0, 3.0, 2.0, 1.0];
        let v = compute_velocity(&completed, &[]);
        assert_eq!(v.velocity_trend, VelocityTrend::Decelerating);
    }

    #[test]
    fn test_velocity_empty() {
        let v = compute_velocity(&[], &[]);
        assert!((v.milestones_per_week - 0.0).abs() < 0.01);
        assert_eq!(v.velocity_trend, VelocityTrend::Stable);
    }

    #[test]
    fn test_employee_retention_on_track() {
        let r = compute_employee_retention(1000, 950);
        assert!((r.retention_rate - 95.0).abs() < 0.01);
        assert_eq!(r.status, HealthStatus::OnTrack);
        assert_eq!(r.departed, 50);
    }

    #[test]
    fn test_employee_retention_at_risk() {
        let r = compute_employee_retention(1000, 850);
        assert!((r.retention_rate - 85.0).abs() < 0.01);
        assert_eq!(r.status, HealthStatus::AtRisk);
    }

    #[test]
    fn test_employee_retention_critical() {
        let r = compute_employee_retention(1000, 750);
        assert!((r.retention_rate - 75.0).abs() < 0.01);
        assert_eq!(r.status, HealthStatus::Critical);
    }

    #[test]
    fn test_customer_retention() {
        let r = compute_customer_retention(500, 480);
        assert!((r.retention_rate - 96.0).abs() < 0.01);
    }

    #[test]
    fn test_analyze_trend_improving() {
        let mut kpi = KPI::new("Synergies", KPICategory::SynergyCost, "$M", 100.0, 0.0);
        kpi.record(10.0);
        kpi.record(25.0);
        kpi.record(40.0);
        kpi.record(60.0);
        let analysis = analyze_trend(&kpi, 1);
        assert_eq!(analysis.direction, TrendDirection::Improving);
        assert!(analysis.slope > 0.0);
        assert!(analysis.forecast.is_some());
    }

    #[test]
    fn test_analyze_trend_declining() {
        let mut kpi = KPI::new("Retention", KPICategory::EmployeeRetention, "%", 95.0, 100.0);
        kpi.record(98.0);
        kpi.record(96.0);
        kpi.record(93.0);
        kpi.record(90.0);
        let analysis = analyze_trend(&kpi, 0);
        assert_eq!(analysis.direction, TrendDirection::Declining);
        assert!(analysis.slope < 0.0);
        assert!(analysis.forecast.is_none());
    }

    #[test]
    fn test_analyze_trend_insufficient_data() {
        let kpi = KPI::new("Test", KPICategory::IntegrationVelocity, "", 100.0, 50.0);
        let analysis = analyze_trend(&kpi, 1);
        assert_eq!(analysis.direction, TrendDirection::Flat);
        assert_eq!(analysis.history_len, 1);
    }

    #[test]
    fn test_kpi_health_score() {
        let kpi1 = KPI::new("K1", KPICategory::SynergyCost, "$M", 100.0, 0.0);
        let mut kpi2 = KPI::new("K2", KPICategory::SynergyRevenue, "$M", 100.0, 0.0);
        kpi2.record(75.0); // 75% progress
        let health = compute_kpi_health(&[kpi1, kpi2]);
        // (0.0 + 0.75) / 2 * 100 = 37.5
        assert!((health - 37.5).abs() < 0.01);
    }

    #[test]
    fn test_kpi_health_empty() {
        let health = compute_kpi_health(&[]);
        assert!((health - 100.0).abs() < 0.01);
    }

    #[test]
    fn test_aggregate_synergies() {
        let mut kpi1 = KPI::new("Cost Synergy", KPICategory::SynergyCost, "$M", 50.0, 0.0);
        kpi1.record(30.0);
        let mut kpi2 = KPI::new("Revenue Synergy", KPICategory::SynergyRevenue, "$M", 80.0, 0.0);
        kpi2.record(40.0);
        let agg = aggregate_synergies(&[kpi1, kpi2]);
        assert!(agg.contains_key(&KPICategory::SynergyCost));
        assert!(agg.contains_key(&KPICategory::SynergyRevenue));
        let (current, target) = agg.get(&KPICategory::SynergyCost).unwrap();
        assert!((*current - 30.0).abs() < 0.01);
        assert!((*target - 50.0).abs() < 0.01);
    }

    #[test]
    fn test_mean_and_stddev_helpers() {
        assert!((mean(&[2.0, 4.0, 6.0]) - 4.0).abs() < 0.01);
        assert!((mean(&[]) - 0.0).abs() < 0.01);
        let sd = std_dev(&[2.0, 4.0, 6.0]);
        assert!((sd - 1.632).abs() < 0.01);
        assert!((std_dev(&[42.0]) - 0.0).abs() < 0.01);
    }
}
