// ─── Core types for post-merger integration intelligence ────────────────────

use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use uuid::Uuid;

/// Phases of a post-merger integration lifecycle.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize, Hash)]
pub enum IntegrationPhase {
    Planning,
    Day1,
    Stabilization,
    Optimization,
}

impl IntegrationPhase {
    /// Returns the ordered list of phases from earliest to latest.
    pub fn all_phases() -> &'static [IntegrationPhase] {
        &[
            IntegrationPhase::Planning,
            IntegrationPhase::Day1,
            IntegrationPhase::Stabilization,
            IntegrationPhase::Optimization,
        ]
    }

    /// Returns whether `self` is strictly before `other` in the phase sequence.
    pub fn precedes(&self, other: &IntegrationPhase) -> bool {
        *self < *other
    }

    /// Returns the next phase, if one exists.
    pub fn next(&self) -> Option<IntegrationPhase> {
        match self {
            IntegrationPhase::Planning => Some(IntegrationPhase::Day1),
            IntegrationPhase::Day1 => Some(IntegrationPhase::Stabilization),
            IntegrationPhase::Stabilization => Some(IntegrationPhase::Optimization),
            IntegrationPhase::Optimization => None,
        }
    }

    /// Returns the index of this phase in the canonical ordering.
    pub fn index(&self) -> usize {
        match self {
            IntegrationPhase::Planning => 0,
            IntegrationPhase::Day1 => 1,
            IntegrationPhase::Stabilization => 2,
            IntegrationPhase::Optimization => 3,
        }
    }

    /// Returns a human-readable label.
    pub fn label(&self) -> &'static str {
        match self {
            IntegrationPhase::Planning => "Planning",
            IntegrationPhase::Day1 => "Day 1",
            IntegrationPhase::Stabilization => "Stabilization",
            IntegrationPhase::Optimization => "Optimization",
        }
    }
}

impl std::fmt::Display for IntegrationPhase {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.label())
    }
}

/// Health status used for workstreams, milestones, and the overall integration.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, Hash)]
pub enum HealthStatus {
    OnTrack,
    AtRisk,
    Critical,
    Complete,
}

impl HealthStatus {
    /// Returns a numeric severity score (higher = worse).
    pub fn severity(&self) -> u8 {
        match self {
            HealthStatus::OnTrack => 0,
            HealthStatus::AtRisk => 1,
            HealthStatus::Critical => 2,
            HealthStatus::Complete => 0,
        }
    }

    /// Returns the short tag used in RAG reports.
    pub fn rag_tag(&self) -> &'static str {
        match self {
            HealthStatus::OnTrack => "Green",
            HealthStatus::AtRisk => "Amber",
            HealthStatus::Critical => "Red",
            HealthStatus::Complete => "Green",
        }
    }

    /// Aggregates a collection of statuses to the worst one.
    pub fn worst(all: &[HealthStatus]) -> HealthStatus {
        if all.is_empty() {
            return HealthStatus::OnTrack;
        }
        all.iter()
            .max_by_key(|s| s.severity())
            .copied()
            .unwrap_or(HealthStatus::OnTrack)
    }
}

impl std::fmt::Display for HealthStatus {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            HealthStatus::OnTrack => write!(f, "On Track"),
            HealthStatus::AtRisk => write!(f, "At Risk"),
            HealthStatus::Critical => write!(f, "Critical"),
            HealthStatus::Complete => write!(f, "Complete"),
        }
    }
}

/// A discrete unit of integration work with a phase, status, and set of milestones.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Workstream {
    pub id: Uuid,
    pub name: String,
    pub phase: IntegrationPhase,
    pub status: HealthStatus,
    pub owner: String,
    pub milestones: Vec<Milestone>,
    pub created_at: DateTime<Utc>,
    pub due_date: Option<DateTime<Utc>>,
    pub percent_complete: f64,
    pub notes: String,
}

impl Workstream {
    pub fn new(name: &str, phase: IntegrationPhase, owner: &str) -> Self {
        Self {
            id: Uuid::new_v4(),
            name: name.to_owned(),
            phase,
            status: HealthStatus::OnTrack,
            owner: owner.to_owned(),
            milestones: Vec::new(),
            created_at: Utc::now(),
            due_date: None,
            percent_complete: 0.0,
            notes: String::new(),
        }
    }

    /// Recompute `percent_complete` and `status` from child milestones.
    pub fn refresh_status(&mut self) {
        if self.milestones.is_empty() {
            self.percent_complete = 0.0;
            self.status = HealthStatus::OnTrack;
            return;
        }
        let total: f64 = self.milestones.len() as f64;
        let done: f64 = self
            .milestones
            .iter()
            .filter(|m| m.status == HealthStatus::Complete)
            .count() as f64;
        self.percent_complete = done / total * 100.0;
        let statuses: Vec<HealthStatus> = self
            .milestones
            .iter()
            .map(|m| m.status)
            .filter(|s| *s != HealthStatus::Complete)
            .collect();
        self.status = if (done - total).abs() < f64::EPSILON {
            HealthStatus::Complete
        } else {
            HealthStatus::worst(&statuses)
        };
    }

    /// Transition the workstream to a new phase. Returns an error if the
    /// transition is non-monotonic (skipping or going backwards).
    pub fn transition_phase(&mut self, target: IntegrationPhase) -> anyhow::Result<()> {
        let current_idx = self.phase.index();
        let target_idx = target.index();
        if target_idx < current_idx {
            anyhow::bail!(
                "Cannot regress from {} to {}",
                self.phase,
                target
            );
        }
        if target_idx > current_idx + 1 {
            anyhow::bail!(
                "Cannot skip phases from {} to {}",
                self.phase,
                target
            );
        }
        self.phase = target;
        Ok(())
    }
}

/// A single milestone belonging to a workstream.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Milestone {
    pub id: Uuid,
    pub name: String,
    pub status: HealthStatus,
    pub target_date: DateTime<Utc>,
    pub completed_date: Option<DateTime<Utc>>,
    pub dependencies: Vec<Uuid>, // IDs of other milestones
    pub notes: String,
}

impl Milestone {
    pub fn new(name: &str, target_date: DateTime<Utc>) -> Self {
        Self {
            id: Uuid::new_v4(),
            name: name.to_owned(),
            status: HealthStatus::OnTrack,
            target_date,
            completed_date: None,
            dependencies: Vec::new(),
            notes: String::new(),
        }
    }

    /// Mark the milestone as complete, recording the completion timestamp.
    pub fn complete(&mut self) {
        self.status = HealthStatus::Complete;
        self.completed_date = Some(Utc::now());
    }

    /// Mark the milestone as at-risk.
    pub fn mark_at_risk(&mut self) {
        if self.status != HealthStatus::Complete {
            self.status = HealthStatus::AtRisk;
        }
    }

    /// Mark the milestone as critical.
    pub fn mark_critical(&mut self) {
        if self.status != HealthStatus::Complete {
            self.status = HealthStatus::Critical;
        }
    }

    /// Check if the milestone is overdue (target date passed and not complete).
    pub fn is_overdue(&self) -> bool {
        self.status != HealthStatus::Complete && self.target_date < Utc::now()
    }
}

/// A tracked KPI for the integration.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct KPI {
    pub id: Uuid,
    pub name: String,
    pub category: KPICategory,
    pub unit: String,
    pub target_value: f64,
    pub current_value: f64,
    pub baseline_value: f64,
    pub history: Vec<f64>,
    pub measurement_date: DateTime<Utc>,
}

/// Category of a KPI metric.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize, Hash)]
pub enum KPICategory {
    SynergyCost,
    SynergyRevenue,
    EmployeeRetention,
    CustomerRetention,
    IntegrationVelocity,
    RevenueRetention,
    CostSavings,
    Custom(String),
}

impl KPI {
    pub fn new(name: &str, category: KPICategory, unit: &str, target: f64, baseline: f64) -> Self {
        Self {
            id: Uuid::new_v4(),
            name: name.to_owned(),
            category,
            unit: unit.to_owned(),
            target_value: target,
            current_value: baseline,
            baseline_value: baseline,
            history: vec![baseline],
            measurement_date: Utc::now(),
        }
    }

    /// Percentage progress toward target (0.0 – 1.0+, capped at 1.0).
    pub fn progress(&self) -> f64 {
        let denom = (self.target_value - self.baseline_value).abs();
        if denom < f64::EPSILON {
            return 1.0;
        }
        let achieved = self.current_value - self.baseline_value;
        ((achieved / denom) * self.target_value.signum()).clamp(0.0, 1.0)
    }

    /// Record a new measurement and push to history.
    pub fn record(&mut self, value: f64) {
        self.current_value = value;
        self.history.push(value);
        self.measurement_date = Utc::now();
    }
}

/// A single entry in the financial reconciliation ledger.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ReconciliationEntry {
    pub id: Uuid,
    pub entity: String,
    pub account_code: String,
    pub account_name: String,
    pub mapped_code: Option<String>,
    pub debit: f64,
    pub credit: f64,
    pub net: f64,
    pub is_reconciled: bool,
    pub notes: String,
}

impl ReconciliationEntry {
    pub fn new(entity: &str, account_code: &str, account_name: &str, debit: f64, credit: f64) -> Self {
        Self {
            id: Uuid::new_v4(),
            entity: entity.to_owned(),
            account_code: account_code.to_owned(),
            account_name: account_name.to_owned(),
            mapped_code: None,
            debit,
            credit,
            net: debit - credit,
            is_reconciled: false,
            notes: String::new(),
        }
    }

    /// Mark this entry as reconciled with a mapping.
    pub fn reconcile(&mut self, mapped_code: &str) {
        self.mapped_code = Some(mapped_code.to_owned());
        self.is_reconciled = true;
    }
}

/// Represents one of the merging entities (acquirer or target).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MergeEntity {
    pub name: String,
    pub entity_type: EntityType,
    pub currency: String,
    pub fiscal_year_end: String,
    pub entries: Vec<ReconciliationEntry>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum EntityType {
    Acquirer,
    Target,
}

impl MergeEntity {
    pub fn new(name: &str, entity_type: EntityType, currency: &str) -> Self {
        Self {
            name: name.to_owned(),
            entity_type,
            currency: currency.to_owned(),
            fiscal_year_end: "12-31".to_owned(),
            entries: Vec::new(),
        }
    }

    /// Total net balance across all entries.
    pub fn total_net(&self) -> f64 {
        self.entries.iter().map(|e| e.net).sum()
    }

    /// Count of reconciled entries.
    pub fn reconciled_count(&self) -> usize {
        self.entries.iter().filter(|e| e.is_reconciled).count()
    }

    /// Add an entry to this entity.
    pub fn add_entry(&mut self, entry: ReconciliationEntry) {
        self.entries.push(entry);
    }
}

/// Composite integration score computed from the pipeline.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct IntegrationScore {
    pub overall: f64,          // 0 – 100
    pub reconciliation: f64,   // 0 – 100
    pub milestone_progress: f64, // 0 – 100
    pub kpi_health: f64,       // 0 – 100
    pub risk_level: HealthStatus,
    pub computed_at: DateTime<Utc>,
    pub phase: IntegrationPhase,
}

impl IntegrationScore {
    /// Compute the overall score as a weighted average of sub-scores.
    pub fn compute(
        reconciliation: f64,
        milestone_progress: f64,
        kpi_health: f64,
        phase: IntegrationPhase,
    ) -> Self {
        let overall = reconciliation * 0.3 + milestone_progress * 0.4 + kpi_health * 0.3;
        let risk_level = if overall >= 75.0 {
            HealthStatus::OnTrack
        } else if overall >= 50.0 {
            HealthStatus::AtRisk
        } else {
            HealthStatus::Critical
        };
        Self {
            overall,
            reconciliation,
            milestone_progress,
            kpi_health,
            risk_level,
            computed_at: Utc::now(),
            phase,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_phase_ordering() {
        assert!(IntegrationPhase::Planning < IntegrationPhase::Day1);
        assert!(IntegrationPhase::Day1 < IntegrationPhase::Stabilization);
        assert!(IntegrationPhase::Stabilization < IntegrationPhase::Optimization);
    }

    #[test]
    fn test_phase_next() {
        assert_eq!(IntegrationPhase::Planning.next(), Some(IntegrationPhase::Day1));
        assert_eq!(IntegrationPhase::Optimization.next(), None);
    }

    #[test]
    fn test_phase_precedes() {
        assert!(IntegrationPhase::Planning.precedes(&IntegrationPhase::Day1));
        assert!(!IntegrationPhase::Optimization.precedes(&IntegrationPhase::Day1));
    }

    #[test]
    fn test_phase_all_phases() {
        let phases = IntegrationPhase::all_phases();
        assert_eq!(phases.len(), 4);
    }

    #[test]
    fn test_health_status_severity() {
        assert!(HealthStatus::Critical.severity() > HealthStatus::AtRisk.severity());
        assert!(HealthStatus::AtRisk.severity() > HealthStatus::OnTrack.severity());
        assert_eq!(HealthStatus::Complete.severity(), 0);
    }

    #[test]
    fn test_health_worst_aggregation() {
        let statuses = vec![HealthStatus::OnTrack, HealthStatus::AtRisk, HealthStatus::OnTrack];
        assert_eq!(HealthStatus::worst(&statuses), HealthStatus::AtRisk);
        let critical = vec![HealthStatus::Critical, HealthStatus::OnTrack];
        assert_eq!(HealthStatus::worst(&critical), HealthStatus::Critical);
    }

    #[test]
    fn test_workstream_creation_and_transition() {
        let mut ws = Workstream::new("IT Systems", IntegrationPhase::Planning, "Alice");
        assert_eq!(ws.phase, IntegrationPhase::Planning);
        assert!(ws.transition_phase(IntegrationPhase::Day1).is_ok());
        assert_eq!(ws.phase, IntegrationPhase::Day1);
        // Cannot skip phases
        assert!(ws.transition_phase(IntegrationPhase::Optimization).is_err());
    }

    #[test]
    fn test_workstream_cannot_regress() {
        let mut ws = Workstream::new("Finance", IntegrationPhase::Day1, "Bob");
        assert!(ws.transition_phase(IntegrationPhase::Planning).is_err());
    }

    #[test]
    fn test_workstream_refresh_status() {
        let mut ws = Workstream::new("HR", IntegrationPhase::Planning, "Carol");
        let mut m1 = Milestone::new("Hire plan", Utc::now());
        let m2 = Milestone::new("Benefits mapping", Utc::now());
        m1.complete();
        ws.milestones.push(m1);
        ws.milestones.push(m2);
        ws.refresh_status();
        assert!((ws.percent_complete - 50.0).abs() < 0.01);
        assert_eq!(ws.status, HealthStatus::OnTrack);
    }

    #[test]
    fn test_milestone_complete() {
        let mut m = Milestone::new("Go-live", Utc::now());
        assert_eq!(m.status, HealthStatus::OnTrack);
        m.complete();
        assert_eq!(m.status, HealthStatus::Complete);
        assert!(m.completed_date.is_some());
    }

    #[test]
    fn test_milestone_rag() {
        let mut m = Milestone::new("Migration", Utc::now());
        m.mark_at_risk();
        assert_eq!(m.status, HealthStatus::AtRisk);
        m.mark_critical();
        assert_eq!(m.status, HealthStatus::Critical);
        // Complete milestones can't be downgraded
        m.complete();
        m.mark_at_risk();
        assert_eq!(m.status, HealthStatus::Complete);
    }

    #[test]
    fn test_kpi_progress() {
        let mut kpi = KPI::new("Cost Synergies", KPICategory::SynergyCost, "$M", 50.0, 0.0);
        assert!((kpi.progress() - 0.0).abs() < 0.01);
        kpi.record(25.0);
        assert!((kpi.progress() - 0.5).abs() < 0.01);
        kpi.record(50.0);
        assert!((kpi.progress() - 1.0).abs() < 0.01);
    }

    #[test]
    fn test_kpi_history() {
        let mut kpi = KPI::new("Velocity", KPICategory::IntegrationVelocity, "tasks/wk", 100.0, 20.0);
        kpi.record(40.0);
        kpi.record(60.0);
        assert_eq!(kpi.history.len(), 3); // baseline + 2 records
    }

    #[test]
    fn test_reconciliation_entry() {
        let mut entry = ReconciliationEntry::new("TargetCo", "1000", "Cash", 1_000_000.0, 0.0);
        assert!(!entry.is_reconciled);
        entry.reconcile("ACQ-1000");
        assert!(entry.is_reconciled);
        assert_eq!(entry.mapped_code.as_deref(), Some("ACQ-1000"));
    }

    #[test]
    fn test_merge_entity_totals() {
        let mut entity = MergeEntity::new("Acquirer", EntityType::Acquirer, "USD");
        entity.add_entry(ReconciliationEntry::new("Acquirer", "1000", "Cash", 500.0, 100.0));
        entity.add_entry(ReconciliationEntry::new("Acquirer", "2000", "AR", 300.0, 0.0));
        assert!((entity.total_net() - 700.0).abs() < 0.01);
    }

    #[test]
    fn test_integration_score_compute() {
        let score = IntegrationScore::compute(80.0, 90.0, 70.0, IntegrationPhase::Stabilization);
        // weighted: 80*0.3 + 90*0.4 + 70*0.3 = 24+36+21 = 81
        assert!((score.overall - 81.0).abs() < 0.01);
        assert_eq!(score.risk_level, HealthStatus::OnTrack);
    }

    #[test]
    fn test_integration_score_risk_level() {
        let low = IntegrationScore::compute(30.0, 20.0, 40.0, IntegrationPhase::Planning);
        assert_eq!(low.risk_level, HealthStatus::Critical);
        let mid = IntegrationScore::compute(50.0, 55.0, 50.0, IntegrationPhase::Planning);
        assert_eq!(mid.risk_level, HealthStatus::AtRisk);
    }

    #[test]
    fn test_rag_tags() {
        assert_eq!(HealthStatus::OnTrack.rag_tag(), "Green");
        assert_eq!(HealthStatus::AtRisk.rag_tag(), "Amber");
        assert_eq!(HealthStatus::Critical.rag_tag(), "Red");
        assert_eq!(HealthStatus::Complete.rag_tag(), "Green");
    }

    #[test]
    fn test_merge_entity_reconciled_count() {
        let mut entity = MergeEntity::new("Target", EntityType::Target, "USD");
        let mut e1 = ReconciliationEntry::new("Target", "1000", "Cash", 100.0, 0.0);
        e1.reconcile("ACQ-1000");
        entity.add_entry(e1);
        entity.add_entry(ReconciliationEntry::new("Target", "2000", "AR", 200.0, 0.0));
        assert_eq!(entity.reconciled_count(), 1);
    }
}
