// ─── convergence — Post-merger integration intelligence platform ─────────────
//
// Governed multi-agent convergence for M&A finance integration.
// Tracks workstreams, KPIs, milestones, and integration health.

pub mod types;
pub mod reconcile;
pub mod track;
pub mod kpi;
pub mod report;
pub mod pipeline;

// Re-export commonly used types at crate root
pub use types::{
    IntegrationPhase, Workstream, Milestone, KPI, KPICategory,
    HealthStatus, ReconciliationEntry, MergeEntity, EntityType, IntegrationScore,
};
pub use pipeline::{IntegrationPipeline, PipelineConfig, PipelineOutput, create_sample_pipeline};
pub use track::{WorkstreamTracker, RAGStatus};
pub use reconcile::{ReconciliationResult, TrialBalanceComparison, AccountVariance, FuzzyMatch};

/// Crate version constant.
pub const VERSION: &str = env!("CARGO_PKG_VERSION");

#[cfg(test)]
mod integration_tests {
    use super::*;
    use crate::pipeline::create_sample_pipeline;
    use crate::track::WorkstreamTracker;
    use chrono::Utc;

    #[test]
    fn test_end_to_end_pipeline() {
        let pipeline = create_sample_pipeline();
        let output = pipeline.run().expect("Pipeline should succeed");
        assert!(!output.deal_name.is_empty());
        assert!(output.integration_score.overall > 0.0);
        assert!(!output.report.executive_summary.is_empty());
    }

    #[test]
    fn test_full_integration_report_generation() {
        let pipeline = create_sample_pipeline();
        let output = pipeline.run().unwrap();

        // Verify all sections of the report are populated
        let sc = &output.report.scorecard;
        assert!(sc.reconciliation.match_rate >= 0.0);
        assert!(sc.milestones.total_milestones > 0);
        assert!(sc.kpis.total_kpis > 0);
        assert!(sc.overall_score > 0.0);
    }

    #[test]
    fn test_reconciliation_across_pipeline() {
        let pipeline = create_sample_pipeline();
        let (recon, tb) = pipeline.run_reconcile_only().unwrap();
        assert_eq!(recon.total_entries, 12); // 6 + 6
        assert!(tb.matched_accounts >= 4); // at least the shared codes
    }

    #[test]
    fn test_workstream_tracker_integration() {
        let mut tracker = WorkstreamTracker::new();
        let ws = Workstream::new("Test WS", IntegrationPhase::Planning, "Owner");
        let id = tracker.add_workstream(ws);
        assert!(tracker.get_workstream(&id).is_some());

        // Add a milestone
        let m = Milestone::new("Test milestone", Utc::now());
        tracker.add_milestone(&id, m).unwrap();
        assert_eq!(tracker.get_workstream(&id).unwrap().milestones.len(), 1);

        // Complete it
        let mid = tracker.get_workstream(&id).unwrap().milestones[0].id;
        tracker.complete_milestone(&mid);
        assert_eq!(
            tracker.get_workstream(&id).unwrap().milestones[0].status,
            HealthStatus::Complete
        );
    }

    #[test]
    fn test_kpi_across_modules() {
        let pipeline = create_sample_pipeline();
        let output = pipeline.run().unwrap();
        // Verify KPI health was computed
        assert!(output.report.scorecard.kpis.health_score > 0.0);
        // Verify trend data was generated
        assert!(!output.report.trend_data.is_empty());
        for trend in &output.report.trend_data {
            assert!(!trend.kpi_name.is_empty());
        }
    }

    #[test]
    fn test_phase_transition_enforcement() {
        let mut tracker = WorkstreamTracker::new();
        let ws = Workstream::new("Phase Test", IntegrationPhase::Planning, "Owner");
        let id = tracker.add_workstream(ws);

        // Valid transition
        let r1 = tracker.transition_phase(&id, IntegrationPhase::Day1);
        assert!(r1.success);

        // Invalid: skip phase
        let r2 = tracker.transition_phase(&id, IntegrationPhase::Optimization);
        assert!(!r2.success);

        // Invalid: regress
        let r3 = tracker.transition_phase(&id, IntegrationPhase::Planning);
        assert!(!r3.success);

        // Valid: next phase
        let r4 = tracker.transition_phase(&id, IntegrationPhase::Stabilization);
        assert!(r4.success);
    }

    #[test]
    fn test_serialization_roundtrip() {
        let pipeline = create_sample_pipeline();
        let output = pipeline.run().unwrap();

        let json = serde_json::to_string(&output).expect("Serialize should work");
        let deserialized: PipelineOutput =
            serde_json::from_str(&json).expect("Deserialize should work");

        assert_eq!(deserialized.deal_name, output.deal_name);
        assert_eq!(deserialized.phase, output.phase);
    }

    #[test]
    fn test_version_constant() {
        assert!(!VERSION.is_empty());
    }

    #[test]
    fn test_multiple_pipeline_runs_independence() {
        // Two pipelines should produce independent results
        let output1 = create_sample_pipeline().run().unwrap();
        let output2 = create_sample_pipeline().run().unwrap();

        // They should have the same deal name but different timestamps
        assert_eq!(output1.deal_name, output2.deal_name);
        assert_ne!(output1.ran_at, output2.ran_at);
    }

    #[test]
    fn test_empty_pipeline_still_succeeds() {
        let acquirer = MergeEntity::new("Empty", EntityType::Acquirer, "USD");
        let target = MergeEntity::new("Empty", EntityType::Target, "USD");
        let tracker = WorkstreamTracker::new();
        let config = PipelineConfig::default();

        let pipeline = IntegrationPipeline::new(config, acquirer, target, tracker, vec![]);
        let output = pipeline.run().unwrap();
        assert!(!output.report.executive_summary.is_empty());
    }

    #[test]
    fn test_all_types_are_send_sync() {
        fn assert_send_sync<T: Send + Sync>() {}
        assert_send_sync::<IntegrationPhase>();
        assert_send_sync::<HealthStatus>();
        assert_send_sync::<Workstream>();
        assert_send_sync::<Milestone>();
        assert_send_sync::<KPI>();
        assert_send_sync::<ReconciliationEntry>();
        assert_send_sync::<MergeEntity>();
        assert_send_sync::<IntegrationScore>();
        assert_send_sync::<WorkstreamTracker>();
        assert_send_sync::<PipelineConfig>();
        assert_send_sync::<PipelineOutput>();
    }
}
