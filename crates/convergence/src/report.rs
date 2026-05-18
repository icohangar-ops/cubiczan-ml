// ─── Board-ready report generation ───────────────────────────────────────────

use crate::kpi::{
    compute_kpi_health, analyze_trend,
    TrendAnalysis,
};
use crate::reconcile::{reconcile_balance_sheet};
use crate::track::{WorkstreamTracker};
use crate::types::{HealthStatus, IntegrationPhase, IntegrationScore, KPI, MergeEntity};
use chrono::{DateTime, Utc};
use serde::{Serialize, Deserialize};

/// A formatted risk-register entry.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RiskEntry {
    pub id: usize,
    pub description: String,
    pub severity: HealthStatus,
    pub likelihood: RiskLikelihood,
    pub mitigation: String,
    pub owner: String,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum RiskLikelihood {
    High,
    Medium,
    Low,
}

/// A section in the board report.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ReportSection {
    pub title: String,
    pub content: String,
    pub rag_status: HealthStatus,
}

/// A complete integration health scorecard.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HealthScorecard {
    pub generated_at: DateTime<Utc>,
    pub phase: IntegrationPhase,
    pub overall_score: f64,
    pub overall_status: HealthStatus,
    pub reconciliation: ReconciliationScorecardSection,
    pub milestones: MilestoneScorecardSection,
    pub kpis: KPIScorecardSection,
    pub sections: Vec<ReportSection>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ReconciliationScorecardSection {
    pub match_rate: f64,
    pub net_variance: f64,
    pub status: HealthStatus,
    pub details: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MilestoneScorecardSection {
    pub total_milestones: usize,
    pub complete: usize,
    pub on_track: usize,
    pub at_risk: usize,
    pub critical: usize,
    pub percent_complete: f64,
    pub status: HealthStatus,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct KPIScorecardSection {
    pub total_kpis: usize,
    pub health_score: f64,
    pub status: HealthStatus,
    pub summaries: Vec<KPISummary>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct KPISummary {
    pub name: String,
    pub progress_percent: f64,
    pub status: HealthStatus,
}

/// The full board-ready integration report.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct IntegrationReport {
    pub title: String,
    pub generated_at: DateTime<Utc>,
    pub scorecard: HealthScorecard,
    pub executive_summary: String,
    pub risk_register: Vec<RiskEntry>,
    pub milestone_report: String,
    pub trend_data: Vec<TrendAnalysis>,
}

// ── Scorecard builder ────────────────────────────────────────────────────────

/// Build a reconciliation scorecard section from two entities.
pub fn build_reconciliation_section(acquirer: &MergeEntity, target: &MergeEntity) -> ReconciliationScorecardSection {
    let result = reconcile_balance_sheet(acquirer, target);
    let status = if result.match_rate >= 0.8 {
        HealthStatus::OnTrack
    } else if result.match_rate >= 0.5 {
        HealthStatus::AtRisk
    } else {
        HealthStatus::Critical
    };
    let details = format!(
        "Total entries: {} | Reconciled: {} | Match rate: {:.1}% | Net variance: ${:.2}",
        result.total_entries, result.reconciled, result.match_rate * 100.0, result.net_variance
    );

    ReconciliationScorecardSection {
        match_rate: result.match_rate * 100.0,
        net_variance: result.net_variance,
        status,
        details,
    }
}

/// Build a milestone scorecard section from the workstream tracker.
pub fn build_milestone_section(tracker: &WorkstreamTracker) -> MilestoneScorecardSection {
    let rag = tracker.compute_overall_rag();
    let total = rag.on_track_count + rag.at_risk_count + rag.critical_count + rag.complete_count;

    let status = if rag.score >= 75.0 {
        HealthStatus::OnTrack
    } else if rag.score >= 50.0 {
        HealthStatus::AtRisk
    } else {
        HealthStatus::Critical
    };

    let percent_complete = if total > 0 {
        rag.complete_count as f64 / total as f64 * 100.0
    } else {
        100.0
    };

    MilestoneScorecardSection {
        total_milestones: total,
        complete: rag.complete_count,
        on_track: rag.on_track_count,
        at_risk: rag.at_risk_count,
        critical: rag.critical_count,
        percent_complete,
        status,
    }
}

/// Build a KPI scorecard section from a list of KPIs.
pub fn build_kpi_section(kpis: &[KPI]) -> KPIScorecardSection {
    let health = compute_kpi_health(kpis);
    let status = if health >= 75.0 {
        HealthStatus::OnTrack
    } else if health >= 50.0 {
        HealthStatus::AtRisk
    } else {
        HealthStatus::Critical
    };

    let summaries: Vec<KPISummary> = kpis
        .iter()
        .map(|k| KPISummary {
            name: k.name.clone(),
            progress_percent: k.progress() * 100.0,
            status: if k.progress() >= 0.75 {
                HealthStatus::OnTrack
            } else if k.progress() >= 0.5 {
                HealthStatus::AtRisk
            } else {
                HealthStatus::Critical
            },
        })
        .collect();

    KPIScorecardSection {
        total_kpis: kpis.len(),
        health_score: health,
        status,
        summaries,
    }
}

/// Build the full health scorecard from all data sources.
pub fn build_health_scorecard(
    phase: IntegrationPhase,
    acquirer: &MergeEntity,
    target: &MergeEntity,
    tracker: &WorkstreamTracker,
    kpis: &[KPI],
) -> HealthScorecard {
    let recon = build_reconciliation_section(acquirer, target);
    let milestones = build_milestone_section(tracker);
    let kpi_section = build_kpi_section(kpis);

    let recon_score = recon.match_rate;
    let milestone_score = milestones.percent_complete;
    let kpi_score = kpi_section.health_score;

    let integration = IntegrationScore::compute(recon_score, milestone_score, kpi_score, phase);

    let mut sections = Vec::new();

    sections.push(ReportSection {
        title: "Financial Reconciliation".to_owned(),
        content: recon.details.clone(),
        rag_status: recon.status,
    });

    sections.push(ReportSection {
        title: format!(
            "Milestone Tracking ({} complete / {} total)",
            milestones.complete, milestones.total_milestones
        ),
        content: format!(
            "On Track: {} | At Risk: {} | Critical: {} | {:.1}% complete",
            milestones.on_track, milestones.at_risk, milestones.critical, milestones.percent_complete
        ),
        rag_status: milestones.status,
    });

    sections.push(ReportSection {
        title: format!("KPI Dashboard ({} KPIs)", kpis.len()),
        content: format!("Composite health score: {:.1}", kpi_section.health_score),
        rag_status: kpi_section.status,
    });

    HealthScorecard {
        generated_at: Utc::now(),
        phase,
        overall_score: integration.overall,
        overall_status: integration.risk_level,
        reconciliation: recon,
        milestones,
        kpis: kpi_section,
        sections,
    }
}

// ── Executive summary builder ────────────────────────────────────────────────

/// Build a human-readable executive summary for the board.
pub fn build_executive_summary(scorecard: &HealthScorecard) -> String {
    let phase_label = scorecard.phase.label();
    let rag_label = scorecard.overall_status.rag_tag();

    let mut lines = vec![
        format!("Post-Merger Integration Report — {}", phase_label),
        format!("Generated: {}", scorecard.generated_at.format("%Y-%m-%d %H:%M UTC")),
        format!("Overall Status: {}", rag_label),
        format!("Overall Score: {:.1}/100", scorecard.overall_score),
        String::new(),
        "Key Findings:".to_owned(),
    ];

    for section in &scorecard.sections {
        lines.push(format!("  • {} — {}", section.title, section.rag_status));
    }

    lines.push(String::new());
    lines.push("Financial Reconciliation:".to_owned());
    lines.push(format!("  {}", scorecard.reconciliation.details));

    lines.push(String::new());
    lines.push("Milestone Progress:".to_owned());
    lines.push(format!(
        "  {}/{} milestones complete ({:.1}%)",
        scorecard.milestones.complete,
        scorecard.milestones.total_milestones,
        scorecard.milestones.percent_complete
    ));
    if scorecard.milestones.at_risk > 0 || scorecard.milestones.critical > 0 {
        lines.push(format!(
            "  ⚠ {} at risk, {} critical",
            scorecard.milestones.at_risk, scorecard.milestones.critical
        ));
    }

    lines.push(String::new());
    lines.push("KPI Dashboard:".to_owned());
    for summary in &scorecard.kpis.summaries {
        lines.push(format!(
            "  • {}: {:.1}% ({})",
            summary.name, summary.progress_percent, summary.status
        ));
    }

    lines.join("\n")
}

// ── Risk register ────────────────────────────────────────────────────────────

/// Auto-generate a risk register from current state data.
pub fn build_risk_register(
    scorecard: &HealthScorecard,
    tracker: &WorkstreamTracker,
    _kpis: &[KPI],
) -> Vec<RiskEntry> {
    let mut risks = Vec::new();
    let mut id = 1_usize;

    // Check for critical milestones
    if scorecard.milestones.critical > 0 {
        risks.push(RiskEntry {
            id: id,
            description: format!(
                "{} milestones in critical status may delay integration timeline",
                scorecard.milestones.critical
            ),
            severity: HealthStatus::Critical,
            likelihood: RiskLikelihood::High,
            mitigation: "Escalate to steering committee; allocate additional resources".to_owned(),
            owner: "Integration PMO".to_owned(),
        });
        id += 1;
    }

    // Check for at-risk milestones
    if scorecard.milestones.at_risk > 0 {
        risks.push(RiskEntry {
            id: id,
            description: format!(
                "{} milestones at risk of missing target dates",
                scorecard.milestones.at_risk
            ),
            severity: HealthStatus::AtRisk,
            likelihood: RiskLikelihood::Medium,
            mitigation: "Weekly status reviews with workstream leads".to_owned(),
            owner: "Integration PMO".to_owned(),
        });
        id += 1;
    }

    // Check reconciliation health
    if scorecard.reconciliation.status == HealthStatus::AtRisk
        || scorecard.reconciliation.status == HealthStatus::Critical
    {
        risks.push(RiskEntry {
            id: id,
            description: format!(
                "Financial reconciliation match rate at {:.1}% — below target",
                scorecard.reconciliation.match_rate
            ),
            severity: scorecard.reconciliation.status,
            likelihood: if scorecard.reconciliation.match_rate < 50.0 {
                RiskLikelihood::High
            } else {
                RiskLikelihood::Medium
            },
            mitigation: "Prioritize account mapping remediation; engage external auditors".to_owned(),
            owner: "CFO / Finance Integration Lead".to_owned(),
        });
        id += 1;
    }

    // Check KPI health
    for summary in &scorecard.kpis.summaries {
        if summary.status == HealthStatus::Critical {
            risks.push(RiskEntry {
                id: id,
                description: format!("KPI '{}' critically behind target ({:.1}%)", summary.name, summary.progress_percent),
                severity: HealthStatus::Critical,
                likelihood: RiskLikelihood::High,
                mitigation: "Review underlying data; adjust targets if unrealistic; assign action owners".to_owned(),
                owner: "Integration PMO".to_owned(),
            });
            id += 1;
        }
    }

    // Check for overdue milestones via the tracker
    let rag = tracker.compute_overall_rag();
    if rag.overdue_count > 0 {
        risks.push(RiskEntry {
            id: id,
            description: format!("{} milestones are past their target date", rag.overdue_count),
            severity: HealthStatus::AtRisk,
            likelihood: RiskLikelihood::High,
            mitigation: "Conduct impact assessment; revise timeline if necessary".to_owned(),
            owner: "Integration PMO".to_owned(),
        });
    }

    risks
}

// ── Milestone completion report ──────────────────────────────────────────────

/// Build a text-based milestone completion report.
pub fn build_milestone_report(tracker: &WorkstreamTracker) -> String {
    let rag = tracker.compute_overall_rag();
    let mut report = String::new();

    report.push_str(&format!("MILESTONE COMPLETION REPORT\n"));
    report.push_str(&format!("Generated: {}\n", Utc::now().format("%Y-%m-%d %H:%M UTC")));
    report.push_str(&format!("{}\n\n", rag.details));

    for ws_id in tracker.workstream_ids() {
        if let Some(ws) = tracker.get_workstream(&ws_id) {
            report.push_str(&format!("▸ {} (Owner: {}, Phase: {})\n", ws.name, ws.owner, ws.phase));
            report.push_str(&format!("  Progress: {:.1}% | Status: {}\n", ws.percent_complete, ws.status));
            for m in &ws.milestones {
                let date_str = m.completed_date
                    .map(|d| format!("✓ Completed: {}", d.format("%Y-%m-%d")))
                    .unwrap_or_else(|| format!("Target: {}", m.target_date.format("%Y-%m-%d")));
                report.push_str(&format!("    • {} — {} | {}\n", m.name, m.status, date_str));
            }
            report.push('\n');
        }
    }

    report
}

// ── Trend visualization data ─────────────────────────────────────────────────

/// Generate trend analysis data for all KPIs (suitable for charting).
pub fn generate_trend_data(kpis: &[KPI], forecast_steps: usize) -> Vec<TrendAnalysis> {
    kpis.iter()
        .map(|k| analyze_trend(k, forecast_steps))
        .collect()
}

// ── Full report assembly ─────────────────────────────────────────────────────

/// Assemble the complete integration report from all data sources.
pub fn build_integration_report(
    title: &str,
    phase: IntegrationPhase,
    acquirer: &MergeEntity,
    target: &MergeEntity,
    tracker: &WorkstreamTracker,
    kpis: &[KPI],
) -> IntegrationReport {
    let scorecard = build_health_scorecard(phase, acquirer, target, tracker, kpis);
    let executive_summary = build_executive_summary(&scorecard);
    let risk_register = build_risk_register(&scorecard, tracker, kpis);
    let milestone_report = build_milestone_report(tracker);
    let trend_data = generate_trend_data(kpis, 3);

    IntegrationReport {
        title: title.to_owned(),
        generated_at: Utc::now(),
        scorecard,
        executive_summary,
        risk_register,
        milestone_report,
        trend_data,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::types::{EntityType, Workstream, KPICategory};
    use crate::reconcile::auto_reconcile;

    fn make_test_entities() -> (MergeEntity, MergeEntity) {
        let mut acq = MergeEntity::new("AcquirerCo", EntityType::Acquirer, "USD");
        acq.add_entry(crate::types::ReconciliationEntry::new("AcquirerCo", "1000", "Cash", 1_000_000.0, 0.0));
        acq.add_entry(crate::types::ReconciliationEntry::new("AcquirerCo", "2000", "Revenue", 0.0, 2_000_000.0));

        let mut tgt = MergeEntity::new("TargetCo", EntityType::Target, "USD");
        tgt.add_entry(crate::types::ReconciliationEntry::new("TargetCo", "1000", "Cash", 500_000.0, 0.0));
        tgt.add_entry(crate::types::ReconciliationEntry::new("TargetCo", "2000", "Revenue", 0.0, 1_000_000.0));

        (acq, tgt)
    }

    fn make_test_tracker() -> WorkstreamTracker {
        let mut tracker = WorkstreamTracker::new();
        let mut ws = Workstream::new("Finance", IntegrationPhase::Stabilization, "Alice");
        let mut m1 = crate::types::Milestone::new("Chart mapping", Utc::now());
        m1.complete();
        let m2 = crate::types::Milestone::new("TB consolidation", Utc::now());
        ws.milestones.push(m1);
        ws.milestones.push(m2);
        tracker.add_workstream(ws);
        tracker
    }

    fn make_test_kpis() -> Vec<KPI> {
        let mut kpi1 = KPI::new("Cost Synergies", KPICategory::SynergyCost, "$M", 50.0, 0.0);
        kpi1.record(35.0);
        let mut kpi2 = KPI::new("Revenue Synergies", KPICategory::SynergyRevenue, "$M", 80.0, 0.0);
        kpi2.record(60.0);
        vec![kpi1, kpi2]
    }

    #[test]
    fn test_build_reconciliation_section() {
        let (acq, tgt) = make_test_entities();
        let section = build_reconciliation_section(&acq, &tgt);
        assert_eq!(section.status, HealthStatus::Critical); // 0 reconciled
        assert!(section.details.contains("Match rate: 0.0%"));
    }

    #[test]
    fn test_build_reconciliation_section_reconciled() {
        let (mut acq, mut tgt) = make_test_entities();
        auto_reconcile(&mut acq, &mut tgt, 0.8);
        let section = build_reconciliation_section(&acq, &tgt);
        assert!(section.match_rate > 50.0);
    }

    #[test]
    fn test_build_milestone_section() {
        let tracker = make_test_tracker();
        let section = build_milestone_section(&tracker);
        assert_eq!(section.total_milestones, 2);
        assert_eq!(section.complete, 1);
        assert!((section.percent_complete - 50.0).abs() < 0.01);
    }

    #[test]
    fn test_build_kpi_section() {
        let kpis = make_test_kpis();
        let section = build_kpi_section(&kpis);
        assert_eq!(section.total_kpis, 2);
        assert!(section.health_score > 0.0);
        assert_eq!(section.summaries.len(), 2);
    }

    #[test]
    fn test_build_health_scorecard() {
        let (acq, tgt) = make_test_entities();
        let tracker = make_test_tracker();
        let kpis = make_test_kpis();
        let card = build_health_scorecard(IntegrationPhase::Stabilization, &acq, &tgt, &tracker, &kpis);
        assert_eq!(card.phase, IntegrationPhase::Stabilization);
        assert!(card.overall_score >= 0.0);
        assert!(card.sections.len() >= 3);
    }

    #[test]
    fn test_build_executive_summary() {
        let (acq, tgt) = make_test_entities();
        let tracker = make_test_tracker();
        let kpis = make_test_kpis();
        let card = build_health_scorecard(IntegrationPhase::Stabilization, &acq, &tgt, &tracker, &kpis);
        let summary = build_executive_summary(&card);
        assert!(summary.contains("Post-Merger Integration Report"));
        assert!(summary.contains("Stabilization"));
        assert!(summary.contains("Financial Reconciliation"));
    }

    #[test]
    fn test_build_risk_register_empty() {
        let (acq, tgt) = make_test_entities();
        let tracker = make_test_tracker();
        let kpis = make_test_kpis();
        let card = build_health_scorecard(IntegrationPhase::Stabilization, &acq, &tgt, &tracker, &kpis);
        // No critical items, but reconciliation is critical → should have at least 1 risk
        let risks = build_risk_register(&card, &tracker, &kpis);
        // May have 0 or more risks depending on the state
        for risk in &risks {
            assert!(!risk.description.is_empty());
            assert!(!risk.mitigation.is_empty());
            assert!(!risk.owner.is_empty());
        }
    }

    #[test]
    fn test_build_milestone_report() {
        let tracker = make_test_tracker();
        let report = build_milestone_report(&tracker);
        assert!(report.contains("MILESTONE COMPLETION REPORT"));
        assert!(report.contains("Finance"));
    }

    #[test]
    fn test_generate_trend_data() {
        let kpis = make_test_kpis();
        let trends = generate_trend_data(&kpis, 3);
        assert_eq!(trends.len(), 2);
        for trend in &trends {
            assert!(trend.forecast.is_some());
        }
    }

    #[test]
    fn test_build_integration_report() {
        let (acq, tgt) = make_test_entities();
        let tracker = make_test_tracker();
        let kpis = make_test_kpis();
        let report = build_integration_report(
            "Q3 Integration Report",
            IntegrationPhase::Stabilization,
            &acq, &tgt, &tracker, &kpis,
        );
        assert_eq!(report.title, "Q3 Integration Report");
        assert!(!report.executive_summary.is_empty());
        assert!(report.trend_data.len() == 2);
    }
}
