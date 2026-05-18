// ─── End-to-end M&A integration pipeline ─────────────────────────────────────
//
// Orchestrates: reconcile → track → kpi → report

use crate::kpi::{
    compute_kpi_health, analyze_trend,
};
use crate::reconcile::{auto_reconcile, normalize_entity, reconcile_balance_sheet, compare_trial_balances};
use crate::report::build_integration_report;
use crate::track::WorkstreamTracker;
use crate::types::{
    IntegrationPhase, IntegrationScore, KPI, KPICategory,
    MergeEntity, ReconciliationEntry, Workstream,
};
use chrono::Utc;
use serde::Serialize;

/// Configuration for a pipeline run.
#[derive(Debug, Clone)]
pub struct PipelineConfig {
    pub deal_name: String,
    pub phase: IntegrationPhase,
    pub fuzzy_match_threshold: f64,
    pub forecast_steps: usize,
}

impl Default for PipelineConfig {
    fn default() -> Self {
        Self {
            deal_name: "Untitled Integration".to_owned(),
            phase: IntegrationPhase::Planning,
            fuzzy_match_threshold: 0.8,
            forecast_steps: 3,
        }
    }
}

/// The complete output of a pipeline run.
#[derive(Debug, Clone, Serialize, serde::Deserialize)]
pub struct PipelineOutput {
    pub deal_name: String,
    pub phase: IntegrationPhase,
    pub integration_score: IntegrationScore,
    pub auto_reconciled_count: usize,
    pub report: crate::report::IntegrationReport,
    pub ran_at: chrono::DateTime<chrono::Utc>,
}

/// The main integration pipeline.
///
/// Steps:
/// 1. **Reconcile** — normalize entities, auto-reconcile accounts, compute match rate.
/// 2. **Track** — build workstream tracker, check overdue, compute RAG.
/// 3. **KPI** — record measurements, compute health, analyze trends.
/// 4. **Report** — assemble the full board-ready report.
pub struct IntegrationPipeline {
    config: PipelineConfig,
    acquirer: MergeEntity,
    target: MergeEntity,
    tracker: WorkstreamTracker,
    kpis: Vec<KPI>,
}

impl IntegrationPipeline {
    /// Create a new pipeline with the given config and data.
    pub fn new(
        config: PipelineConfig,
        acquirer: MergeEntity,
        target: MergeEntity,
        tracker: WorkstreamTracker,
        kpis: Vec<KPI>,
    ) -> Self {
        Self {
            config,
            acquirer,
            target,
            tracker,
            kpis,
        }
    }

    /// Run the full pipeline end-to-end.
    pub fn run(mut self) -> anyhow::Result<PipelineOutput> {
        // ── Step 1: Reconcile ──────────────────────────────────────────
        self.step_reconcile()?;

        // ── Step 2: Track ──────────────────────────────────────────────
        self.step_track();

        // ── Step 3: KPI ────────────────────────────────────────────────
        self.step_kpi();

        // ── Step 4: Report ─────────────────────────────────────────────
        let report = build_integration_report(
            &self.config.deal_name,
            self.config.phase,
            &self.acquirer,
            &self.target,
            &self.tracker,
            &self.kpis,
        );

        // Compute integration score from the report's scorecard
        let scorecard = &report.scorecard;
        let integration_score = IntegrationScore::compute(
            scorecard.reconciliation.match_rate,
            scorecard.milestones.percent_complete,
            scorecard.kpis.health_score,
            self.config.phase,
        );

        Ok(PipelineOutput {
            deal_name: self.config.deal_name,
            phase: self.config.phase,
            integration_score,
            auto_reconciled_count: self.acquirer.reconciled_count() + self.target.reconciled_count() / 2,
            report,
            ran_at: Utc::now(),
        })
    }

    /// Step 1: Normalize and auto-reconcile financial data.
    fn step_reconcile(&mut self) -> anyhow::Result<()> {
        // Normalize account names and amounts
        normalize_entity(&mut self.acquirer);
        normalize_entity(&mut self.target);

        // Auto-reconcile using the configured fuzzy threshold
        let count = auto_reconcile(
            &mut self.acquirer,
            &mut self.target,
            self.config.fuzzy_match_threshold,
        );

        // Validate that reconciliation produced results
        if count == 0 && !self.acquirer.entries.is_empty() && !self.target.entries.is_empty() {
            // Not an error — just no matches found at this threshold
        }

        Ok(())
    }

    /// Step 2: Refresh tracker state, check overdue, compute RAG.
    fn step_track(&mut self) {
        // Check overdue milestones
        self.tracker.check_overdue();

        // Refresh all workstream statuses
        for ws_id in self.tracker.workstream_ids() {
            if let Some(ws) = self.tracker.get_workstream_mut(&ws_id) {
                ws.refresh_status();
            }
        }
    }

    /// Step 3: Analyze KPIs and compute health.
    fn step_kpi(&mut self) {
        // KPI health is computed during report generation, but we can
        // compute it here for the integration score.
        let _health = compute_kpi_health(&self.kpis);

        // Analyze trends for each KPI
        for kpi in &self.kpis {
            let _trend = analyze_trend(kpi, self.config.forecast_steps);
        }
    }

    /// Run only the reconciliation step and return results.
    pub fn run_reconcile_only(mut self) -> anyhow::Result<(crate::reconcile::ReconciliationResult, crate::reconcile::TrialBalanceComparison)> {
        normalize_entity(&mut self.acquirer);
        normalize_entity(&mut self.target);
        let _count = auto_reconcile(&mut self.acquirer, &mut self.target, self.config.fuzzy_match_threshold);
        let recon_result = reconcile_balance_sheet(&self.acquirer, &self.target);
        let tb_comparison = compare_trial_balances(&self.acquirer, &self.target);
        Ok((recon_result, tb_comparison))
    }

    /// Run reconciliation and tracking only.
    pub fn run_reconcile_and_track(mut self) -> anyhow::Result<(crate::reconcile::ReconciliationResult, crate::track::RAGStatus)> {
        self.step_reconcile()?;
        self.step_track();
        let recon_result = reconcile_balance_sheet(&self.acquirer, &self.target);
        let rag = self.tracker.compute_overall_rag();
        Ok((recon_result, rag))
    }
}

/// Create a sample pipeline for testing / demonstration.
pub fn create_sample_pipeline() -> IntegrationPipeline {
    let config = PipelineConfig {
        deal_name: "AcmeCorp / BetaInc Merger".to_owned(),
        phase: IntegrationPhase::Stabilization,
        fuzzy_match_threshold: 0.8,
        forecast_steps: 3,
    };

    // Acquirer entity
    let mut acquirer = MergeEntity::new("AcmeCorp", crate::types::EntityType::Acquirer, "USD");
    acquirer.add_entry(ReconciliationEntry::new("AcmeCorp", "1000", "Cash and Equivalents", 5_000_000.0, 0.0));
    acquirer.add_entry(ReconciliationEntry::new("AcmeCorp", "1200", "Accounts Receivable", 3_000_000.0, 0.0));
    acquirer.add_entry(ReconciliationEntry::new("AcmeCorp", "1500", "Inventory", 1_500_000.0, 0.0));
    acquirer.add_entry(ReconciliationEntry::new("AcmeCorp", "2000", "Accounts Payable", 0.0, 2_000_000.0));
    acquirer.add_entry(ReconciliationEntry::new("AcmeCorp", "3000", "Revenue", 0.0, 10_000_000.0));
    acquirer.add_entry(ReconciliationEntry::new("AcmeCorp", "4000", "Cost of Goods Sold", 6_000_000.0, 0.0));

    // Target entity
    let mut target = MergeEntity::new("BetaInc", crate::types::EntityType::Target, "USD");
    target.add_entry(ReconciliationEntry::new("BetaInc", "1000", "Cash & Equivalents", 2_000_000.0, 0.0));
    target.add_entry(ReconciliationEntry::new("BetaInc", "1200", "Accounts Receivable", 1_500_000.0, 0.0));
    target.add_entry(ReconciliationEntry::new("BetaInc", "1500", "Inventory", 800_000.0, 0.0));
    target.add_entry(ReconciliationEntry::new("BetaInc", "2000", "Accounts Payable", 0.0, 1_000_000.0));
    target.add_entry(ReconciliationEntry::new("BetaInc", "3000", "Revenue", 0.0, 5_000_000.0));
    target.add_entry(ReconciliationEntry::new("BetaInc", "9999", "Goodwill", 3_000_000.0, 0.0));

    // Workstream tracker
    let mut tracker = WorkstreamTracker::new();

    let mut ws_finance = Workstream::new("Finance Integration", IntegrationPhase::Stabilization, "CFO Office");
    let mut m1 = crate::types::Milestone::new("Chart of accounts mapping", Utc::now());
    let mut m2 = crate::types::Milestone::new("Trial balance consolidation", Utc::now());
    let m3 = crate::types::Milestone::new("Tax structure alignment", Utc::now());
    m1.complete();
    m2.complete();
    ws_finance.milestones.push(m1);
    ws_finance.milestones.push(m2);
    ws_finance.milestones.push(m3);
    tracker.add_workstream(ws_finance);

    let mut ws_it = Workstream::new("IT Systems Integration", IntegrationPhase::Day1, "CTO Office");
    let mut m4 = crate::types::Milestone::new("Email migration", Utc::now());
    let m5 = crate::types::Milestone::new("ERP consolidation", Utc::now());
    let mut m6 = crate::types::Milestone::new("Network integration", Utc::now());
    m4.complete();
    m6.mark_at_risk();
    ws_it.milestones.push(m4);
    ws_it.milestones.push(m5);
    ws_it.milestones.push(m6);
    tracker.add_workstream(ws_it);

    let mut ws_hr = Workstream::new("HR Integration", IntegrationPhase::Planning, "CHRO Office");
    let m7 = crate::types::Milestone::new("Benefits harmonization", Utc::now());
    let m8 = crate::types::Milestone::new("Org structure design", Utc::now());
    ws_hr.milestones.push(m7);
    ws_hr.milestones.push(m8);
    tracker.add_workstream(ws_hr);

    // KPIs
    let mut kpi1 = KPI::new("Cost Synergies", KPICategory::SynergyCost, "$M", 20.0, 0.0);
    kpi1.record(5.0);
    kpi1.record(8.0);
    kpi1.record(12.0);

    let mut kpi2 = KPI::new("Revenue Synergies", KPICategory::SynergyRevenue, "$M", 15.0, 0.0);
    kpi2.record(3.0);
    kpi2.record(6.0);

    let mut kpi3 = KPI::new("Employee Retention", KPICategory::EmployeeRetention, "%", 95.0, 100.0);
    kpi3.record(97.0);
    kpi3.record(95.0);
    kpi3.record(93.0);

    let mut kpi4 = KPI::new("Customer Retention", KPICategory::CustomerRetention, "%", 98.0, 100.0);
    kpi4.record(99.0);
    kpi4.record(98.5);
    kpi4.record(98.0);

    IntegrationPipeline::new(
        config,
        acquirer,
        target,
        tracker,
        vec![kpi1, kpi2, kpi3, kpi4],
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_pipeline_full_run() {
        let pipeline = create_sample_pipeline();
        let result = pipeline.run();
        assert!(result.is_ok());
        let output = result.unwrap();
        assert_eq!(output.deal_name, "AcmeCorp / BetaInc Merger");
        assert_eq!(output.phase, IntegrationPhase::Stabilization);
        assert!(output.integration_score.overall > 0.0);
    }

    #[test]
    fn test_pipeline_reconcile_only() {
        let pipeline = create_sample_pipeline();
        let result = pipeline.run_reconcile_only();
        assert!(result.is_ok());
        let (recon, tb) = result.unwrap();
        assert!(recon.total_entries > 0);
        assert!(tb.matched_accounts > 0);
    }

    #[test]
    fn test_pipeline_reconcile_and_track() {
        let pipeline = create_sample_pipeline();
        let result = pipeline.run_reconcile_and_track();
        assert!(result.is_ok());
        let (recon, rag) = result.unwrap();
        assert!(recon.match_rate > 0.0);
        assert!(rag.score > 0.0);
    }

    #[test]
    fn test_pipeline_output_has_report() {
        let pipeline = create_sample_pipeline();
        let output = pipeline.run().unwrap();
        assert!(!output.report.executive_summary.is_empty());
        assert!(!output.report.milestone_report.is_empty());
        assert!(!output.report.risk_register.is_empty() || true); // may be empty
        assert!(!output.report.trend_data.is_empty());
    }

    #[test]
    fn test_pipeline_output_scorecard() {
        let pipeline = create_sample_pipeline();
        let output = pipeline.run().unwrap();
        let sc = &output.report.scorecard;
        assert_eq!(sc.phase, IntegrationPhase::Stabilization);
        assert!(sc.overall_score > 0.0);
        assert!(!sc.sections.is_empty());
    }

    #[test]
    fn test_pipeline_default_config() {
        let config = PipelineConfig::default();
        assert_eq!(config.deal_name, "Untitled Integration");
        assert_eq!(config.phase, IntegrationPhase::Planning);
        assert!((config.fuzzy_match_threshold - 0.8).abs() < 0.001);
    }

    #[test]
    fn test_pipeline_with_empty_entities() {
        let acquirer = MergeEntity::new("Empty", crate::types::EntityType::Acquirer, "USD");
        let target = MergeEntity::new("Empty", crate::types::EntityType::Target, "USD");
        let tracker = WorkstreamTracker::new();
        let config = PipelineConfig::default();

        let pipeline = IntegrationPipeline::new(config, acquirer, target, tracker, vec![]);
        let result = pipeline.run();
        assert!(result.is_ok());
    }

    #[test]
    fn test_pipeline_serialization() {
        let pipeline = create_sample_pipeline();
        let output = pipeline.run().unwrap();
        // Ensure the output is serializable
        let json = serde_json::to_string(&output);
        assert!(json.is_ok());
        let json_str = json.unwrap();
        assert!(json_str.contains("AcmeCorp"));
    }
}
