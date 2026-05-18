// ─── Milestone / workstream state machine & scheduling ───────────────────────

use crate::types::{HealthStatus, IntegrationPhase, Milestone, Workstream};
use chrono::{DateTime, Utc};
use std::collections::HashMap;
use uuid::Uuid;

/// A Gantt-like schedule entry for a milestone.
#[derive(Debug, Clone)]
pub struct GanttEntry {
    pub milestone_id: Uuid,
    pub milestone_name: String,
    pub workstream_name: String,
    pub start: DateTime<Utc>,
    pub end: DateTime<Utc>,
    pub status: HealthStatus,
    pub dependencies: Vec<Uuid>,
    pub is_overdue: bool,
}

/// A RAG (Red / Amber / Green) status report for a workstream or the whole integration.
#[derive(Debug, Clone)]
pub struct RAGStatus {
    pub status: HealthStatus,
    pub rag_label: &'static str,
    pub score: f64,           // 0.0 – 100.0
    pub on_track_count: usize,
    pub at_risk_count: usize,
    pub critical_count: usize,
    pub complete_count: usize,
    pub overdue_count: usize,
    pub details: String,
}

/// The result of a phase transition attempt.
#[derive(Debug)]
pub struct PhaseTransitionResult {
    pub workstream_id: Uuid,
    pub from_phase: IntegrationPhase,
    pub to_phase: IntegrationPhase,
    pub success: bool,
    pub reason: Option<String>,
}

/// Manage a collection of workstreams and enforce phase-transition rules.
#[derive(Debug, Default)]
pub struct WorkstreamTracker {
    workstreams: HashMap<Uuid, Workstream>,
}

impl WorkstreamTracker {
    pub fn new() -> Self {
        Self::default()
    }

    /// Add a workstream to the tracker.
    pub fn add_workstream(&mut self, ws: Workstream) -> Uuid {
        let id = ws.id;
        self.workstreams.insert(id, ws);
        id
    }

    /// Get a reference to a workstream by ID.
    pub fn get_workstream(&self, id: &Uuid) -> Option<&Workstream> {
        self.workstreams.get(id)
    }

    /// Get a mutable reference to a workstream by ID.
    pub fn get_workstream_mut(&mut self, id: &Uuid) -> Option<&mut Workstream> {
        self.workstreams.get_mut(id)
    }

    /// Attempt to advance a workstream's phase.
    /// Only forward transitions by exactly one phase are allowed.
    pub fn transition_phase(&mut self, id: &Uuid, target: IntegrationPhase) -> PhaseTransitionResult {
        let result;
        if let Some(ws) = self.workstreams.get_mut(id) {
            let from = ws.phase;
            match ws.transition_phase(target) {
                Ok(()) => {
                    result = PhaseTransitionResult {
                        workstream_id: *id,
                        from_phase: from,
                        to_phase: target,
                        success: true,
                        reason: None,
                    };
                }
                Err(e) => {
                    result = PhaseTransitionResult {
                        workstream_id: *id,
                        from_phase: from,
                        to_phase: target,
                        success: false,
                        reason: Some(e.to_string()),
                    };
                }
            }
        } else {
            result = PhaseTransitionResult {
                workstream_id: *id,
                from_phase: IntegrationPhase::Planning,
                to_phase: target,
                success: false,
                reason: Some("Workstream not found".to_owned()),
            };
        }
        result
    }

    /// Add a milestone to a workstream.
    pub fn add_milestone(&mut self, ws_id: &Uuid, milestone: Milestone) -> anyhow::Result<()> {
        let ws = self
            .workstreams
            .get_mut(ws_id)
            .ok_or_else(|| anyhow::anyhow!("Workstream not found"))?;
        ws.milestones.push(milestone);
        ws.refresh_status();
        Ok(())
    }

    /// Complete a milestone by ID (searching all workstreams).
    pub fn complete_milestone(&mut self, milestone_id: &Uuid) -> bool {
        for ws in self.workstreams.values_mut() {
            for m in ws.milestones.iter_mut() {
                if m.id == *milestone_id {
                    m.complete();
                    ws.refresh_status();
                    return true;
                }
            }
        }
        false
    }

    /// Check all milestones for overdue status and mark at-risk / critical as needed.
    pub fn check_overdue(&mut self) -> Vec<Uuid> {
        let mut overdue_ids = Vec::new();
        for ws in self.workstreams.values_mut() {
            for m in ws.milestones.iter_mut() {
                if m.is_overdue() {
                    overdue_ids.push(m.id);
                    if m.status == HealthStatus::OnTrack {
                        m.mark_at_risk();
                    }
                }
            }
            ws.refresh_status();
        }
        overdue_ids
    }

    /// Resolve dependencies: returns a list of milestone IDs whose dependencies
    /// are all complete (i.e., they are unblocked).
    pub fn unblocked_milestones(&self) -> Vec<Uuid> {
        let complete_set: std::collections::HashSet<Uuid> = self
            .workstreams
            .values()
            .flat_map(|ws| ws.milestones.iter())
            .filter(|m| m.status == HealthStatus::Complete)
            .map(|m| m.id)
            .collect();

        self.workstreams
            .values()
            .flat_map(|ws| ws.milestones.iter())
            .filter(|m| {
                m.status != HealthStatus::Complete
                    && m.dependencies.iter().all(|dep| complete_set.contains(dep))
            })
            .map(|m| m.id)
            .collect()
    }

    /// Build a Gantt-like schedule from all milestones across all workstreams.
    pub fn build_gantt(&self) -> Vec<GanttEntry> {
        let mut entries = Vec::new();
        for ws in self.workstreams.values() {
            for m in &ws.milestones {
                // Estimate start: latest completed dependency date, or created_at
                let start = m.completed_date.unwrap_or(m.target_date);
                entries.push(GanttEntry {
                    milestone_id: m.id,
                    milestone_name: m.name.clone(),
                    workstream_name: ws.name.clone(),
                    start,
                    end: m.target_date,
                    status: m.status,
                    dependencies: m.dependencies.clone(),
                    is_overdue: m.is_overdue(),
                });
            }
        }
        entries.sort_by_key(|e| e.start);
        entries
    }

    /// Compute the RAG status for a single workstream.
    pub fn compute_workstream_rag(&self, ws_id: &Uuid) -> Option<RAGStatus> {
        let ws = self.workstreams.get(ws_id)?;
        let mut on_track = 0;
        let mut at_risk = 0;
        let mut critical = 0;
        let mut complete = 0;
        let mut overdue = 0;

        for m in &ws.milestones {
            match m.status {
                HealthStatus::OnTrack => on_track += 1,
                HealthStatus::AtRisk => at_risk += 1,
                HealthStatus::Critical => critical += 1,
                HealthStatus::Complete => complete += 1,
            }
            if m.is_overdue() {
                overdue += 1;
            }
        }

        let total = ws.milestones.len();
        let score = if total > 0 {
            (complete as f64 * 100.0 + on_track as f64 * 75.0 + at_risk as f64 * 50.0 + critical as f64 * 25.0)
                / total as f64
        } else {
            100.0
        };

        let status = if total == 0 || complete == total {
            HealthStatus::Complete
        } else {
            HealthStatus::worst(&ws.milestones.iter().map(|m| m.status).collect::<Vec<_>>())
        };

        let rag_label = status.rag_tag();
        let details = format!(
            "{}: {} milestones ({} complete, {} on track, {} at risk, {} critical, {} overdue) — {:.1}%",
            ws.name, total, complete, on_track, at_risk, critical, overdue, ws.percent_complete
        );

        Some(RAGStatus {
            status,
            rag_label,
            score,
            on_track_count: on_track,
            at_risk_count: at_risk,
            critical_count: critical,
            complete_count: complete,
            overdue_count: overdue,
            details,
        })
    }

    /// Compute the overall RAG status across all workstreams.
    pub fn compute_overall_rag(&self) -> RAGStatus {
        let mut total_on_track = 0;
        let mut total_at_risk = 0;
        let mut total_critical = 0;
        let mut total_complete = 0;
        let mut total_overdue = 0;

        for ws in self.workstreams.values() {
            for m in &ws.milestones {
                match m.status {
                    HealthStatus::OnTrack => total_on_track += 1,
                    HealthStatus::AtRisk => total_at_risk += 1,
                    HealthStatus::Critical => total_critical += 1,
                    HealthStatus::Complete => total_complete += 1,
                }
                if m.is_overdue() {
                    total_overdue += 1;
                }
            }
        }

        let total = total_on_track + total_at_risk + total_critical + total_complete;
        let score = if total > 0 {
            (total_complete as f64 * 100.0
                + total_on_track as f64 * 75.0
                + total_at_risk as f64 * 50.0
                + total_critical as f64 * 25.0)
                / total as f64
        } else {
            100.0
        };

        let statuses: Vec<HealthStatus> = self
            .workstreams
            .values()
            .map(|ws| ws.status)
            .collect();
        let status = HealthStatus::worst(&statuses);
        let rag_label = status.rag_tag();

        let details = format!(
            "Overall: {} milestones across {} workstreams ({} complete, {} on track, {} at risk, {} critical, {} overdue) — score {:.1}",
            total, self.workstreams.len(), total_complete, total_on_track, total_at_risk, total_critical, total_overdue, score
        );

        RAGStatus {
            status,
            rag_label,
            score,
            on_track_count: total_on_track,
            at_risk_count: total_at_risk,
            critical_count: total_critical,
            complete_count: total_complete,
            overdue_count: total_overdue,
            details,
        }
    }

    /// Get all workstream IDs.
    pub fn workstream_ids(&self) -> Vec<Uuid> {
        self.workstreams.keys().copied().collect()
    }

    /// Get the number of workstreams.
    pub fn len(&self) -> usize {
        self.workstreams.len()
    }

    /// Check if the tracker has no workstreams.
    pub fn is_empty(&self) -> bool {
        self.workstreams.is_empty()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_tracker_with_data() -> (WorkstreamTracker, Uuid, Uuid) {
        let mut tracker = WorkstreamTracker::new();
        let mut ws1 = Workstream::new("Finance Integration", IntegrationPhase::Planning, "Alice");
        let mut ws2 = Workstream::new("IT Integration", IntegrationPhase::Planning, "Bob");

        let m1 = Milestone::new("Chart of accounts mapping", Utc::now());
        let m2 = Milestone::new("Trial balance consolidation", Utc::now());
        let m3 = Milestone::new("System migration", Utc::now());
        let m4 = Milestone::new("Data migration", Utc::now());

        ws1.milestones.push(m1);
        ws1.milestones.push(m2);
        ws2.milestones.push(m3);
        ws2.milestones.push(m4);

        let id1 = tracker.add_workstream(ws1);
        let id2 = tracker.add_workstream(ws2);

        (tracker, id1, id2)
    }

    #[test]
    fn test_add_and_get_workstream() {
        let mut tracker = WorkstreamTracker::new();
        let ws = Workstream::new("Test", IntegrationPhase::Planning, "TestOwner");
        let id = tracker.add_workstream(ws);
        assert!(tracker.get_workstream(&id).is_some());
        assert_eq!(tracker.len(), 1);
    }

    #[test]
    fn test_phase_transition_success() {
        let (mut tracker, id1, _) = make_tracker_with_data();
        let result = tracker.transition_phase(&id1, IntegrationPhase::Day1);
        assert!(result.success);
        assert!(result.reason.is_none());
        assert_eq!(tracker.get_workstream(&id1).unwrap().phase, IntegrationPhase::Day1);
    }

    #[test]
    fn test_phase_transition_skip_fails() {
        let (mut tracker, id1, _) = make_tracker_with_data();
        let result = tracker.transition_phase(&id1, IntegrationPhase::Stabilization);
        assert!(!result.success);
        assert!(result.reason.is_some());
    }

    #[test]
    fn test_phase_transition_nonexistent() {
        let mut tracker = WorkstreamTracker::new();
        let fake_id = Uuid::new_v4();
        let result = tracker.transition_phase(&fake_id, IntegrationPhase::Day1);
        assert!(!result.success);
    }

    #[test]
    fn test_add_milestone() {
        let (mut tracker, id1, _) = make_tracker_with_data();
        let m = Milestone::new("New milestone", Utc::now());
        assert!(tracker.add_milestone(&id1, m).is_ok());
        assert_eq!(tracker.get_workstream(&id1).unwrap().milestones.len(), 3);
    }

    #[test]
    fn test_add_milestone_nonexistent() {
        let mut tracker = WorkstreamTracker::new();
        let m = Milestone::new("New milestone", Utc::now());
        assert!(tracker.add_milestone(&Uuid::new_v4(), m).is_err());
    }

    #[test]
    fn test_complete_milestone() {
        let (mut tracker, id1, _) = make_tracker_with_data();
        let mid = tracker.get_workstream(&id1).unwrap().milestones[0].id;
        assert!(tracker.complete_milestone(&mid));
        let ws = tracker.get_workstream(&id1).unwrap();
        assert_eq!(ws.milestones[0].status, HealthStatus::Complete);
        assert!((ws.percent_complete - 50.0).abs() < 0.01);
    }

    #[test]
    fn test_check_overdue() {
        let (mut tracker, _id1, id2) = make_tracker_with_data();
        // Set a milestone's target_date to the past
        let past = Utc::now() - chrono::Duration::days(30);
        if let Some(ws) = tracker.get_workstream_mut(&id2) {
            ws.milestones[0].target_date = past;
        }
        let overdue = tracker.check_overdue();
        assert!(!overdue.is_empty());
    }

    #[test]
    fn test_unblocked_milestones() {
        let (mut tracker, id1, _) = make_tracker_with_data();
        // Complete the first milestone, the second has no deps so it's always unblocked
        let ws = tracker.get_workstream(&id1).unwrap();
        let first_mid = ws.milestones[0].id;
        tracker.complete_milestone(&first_mid);
        let unblocked = tracker.unblocked_milestones();
        // At least the second milestone of ws1 should be unblocked
        assert!(!unblocked.is_empty());
    }

    #[test]
    fn test_build_gantt() {
        let (tracker, _, _) = make_tracker_with_data();
        let gantt = tracker.build_gantt();
        assert_eq!(gantt.len(), 4);
    }

    #[test]
    fn test_compute_workstream_rag() {
        let (tracker, id1, _) = make_tracker_with_data();
        let rag = tracker.compute_workstream_rag(&id1).unwrap();
        assert_eq!(rag.on_track_count, 2);
        assert_eq!(rag.rag_label, "Green");
    }

    #[test]
    fn test_compute_overall_rag() {
        let (tracker, _, _) = make_tracker_with_data();
        let rag = tracker.compute_overall_rag();
        assert_eq!(rag.on_track_count, 4);
        assert!(rag.score > 70.0);
    }

    #[test]
    fn test_overall_rag_with_critical() {
        let mut tracker = WorkstreamTracker::new();
        let ws = Workstream::new("Critical WS", IntegrationPhase::Planning, "Owner");
        let ws_id = tracker.add_workstream(ws);
        let mut m = Milestone::new("Critical milestone", Utc::now());
        m.mark_critical();
        tracker.add_milestone(&ws_id, m).unwrap();
        // After adding a critical milestone, the workstream status should be refreshed
        let rag = tracker.compute_overall_rag();
        assert_eq!(rag.rag_label, "Red");
    }

    #[test]
    fn test_tracker_empty() {
        let tracker = WorkstreamTracker::new();
        assert!(tracker.is_empty());
        assert_eq!(tracker.len(), 0);
    }
}
