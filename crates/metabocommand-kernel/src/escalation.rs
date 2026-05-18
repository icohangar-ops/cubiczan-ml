use crate::types::*;

/// Known guardrail patterns that trigger approval-required mode.
const APPROVAL_GUARDRAILS: &[&str] = &[
    "revenue_threshold",
    "inventory_limit",
    "spending_cap",
    "price_change_limit",
    "volume_limit",
    "discount_cap",
];

/// Summarize a collection of escalation lanes by counting modes and evidence checks.
pub fn summarize_escalations(lanes: &[EscalationLane]) -> EscalationSummary {
    let mut summary = EscalationSummary::zero();
    for lane in lanes {
        summary.total += 1;
        match lane.mode {
            EscalationMode::Autonomous => summary.autonomous += 1,
            EscalationMode::ApprovalRequired => summary.approval_required += 1,
            EscalationMode::HumanHandoff => summary.human_handoff += 1,
        }
        summary.evidence_checks += lane.evidence.len() as u32;
    }
    summary
}

/// Classify an escalation lane based on rules:
/// - If evidence count >= 3 → Autonomous
/// - Else if guardrail matches known patterns → ApprovalRequired
/// - Else → HumanHandoff
pub fn classify_escalation(lane: &EscalationLane) -> EscalationMode {
    if lane.evidence.len() >= 3 {
        EscalationMode::Autonomous
    } else if APPROVAL_GUARDRAILS.contains(&lane.guardrail.as_str()) {
        EscalationMode::ApprovalRequired
    } else {
        EscalationMode::HumanHandoff
    }
}

/// Calculate priority score for an escalation lane.
///
/// Base: HumanHandoff=3, ApprovalRequired=2, Autonomous=1
/// Bonus: +1 for each evidence item (capped at +3)
pub fn escalation_priority(lane: &EscalationLane) -> u8 {
    let base: u8 = match lane.mode {
        EscalationMode::HumanHandoff => 3,
        EscalationMode::ApprovalRequired => 2,
        EscalationMode::Autonomous => 1,
    };
    let bonus = (lane.evidence.len() as u8).min(3);
    base + bonus
}

/// Calculate priority using classified mode instead of the lane's stored mode.
pub fn classified_priority(lane: &EscalationLane) -> u8 {
    let mode = classify_escalation(lane);
    let base: u8 = match mode {
        EscalationMode::HumanHandoff => 3,
        EscalationMode::ApprovalRequired => 2,
        EscalationMode::Autonomous => 1,
    };
    let bonus = (lane.evidence.len() as u8).min(3);
    base + bonus
}

/// Sort lanes by priority descending.
pub fn lanes_by_priority(lanes: &[EscalationLane]) -> Vec<&EscalationLane> {
    let mut sorted: Vec<&EscalationLane> = lanes.iter().collect();
    sorted.sort_by(|a, b| escalation_priority(b).cmp(&escalation_priority(a)));
    sorted
}

/// Count lanes that would be reclassified by the rules engine.
pub fn count_reclassifications(lanes: &[EscalationLane]) -> u32 {
    lanes.iter()
        .filter(|lane| classify_escalation(lane) != lane.mode)
        .count() as u32
}

/// Filter lanes to only those with a specific mode.
pub fn filter_by_mode(lanes: &[EscalationLane], mode: EscalationMode) -> Vec<&EscalationLane> {
    lanes.iter().filter(|l| l.mode == mode).collect()
}

/// Calculate the human oversight ratio: (approval_required + human_handoff) / total.
pub fn human_oversight_ratio(summary: &EscalationSummary) -> f64 {
    if summary.total == 0 {
        return 0.0;
    }
    (summary.approval_required + summary.human_handoff) as f64 / summary.total as f64
}

#[cfg(test)]
mod tests {
    use super::*;

    fn sample_lane(id: &str, mode: EscalationMode, evidence_count: usize) -> EscalationLane {
        let mut lane = EscalationLane::new(
            id,
            "test",
            mode,
            "generic",
            "owner",
        );
        lane.evidence = (0..evidence_count).map(|i| format!("evidence_{}", i)).collect();
        lane
    }

    #[test]
    fn test_summarize_empty() {
        let summary = summarize_escalations(&[]);
        assert_eq!(summary.total, 0);
    }

    #[test]
    fn test_summarize_mixed() {
        let lanes = vec![
            sample_lane("A", EscalationMode::Autonomous, 2),
            sample_lane("B", EscalationMode::ApprovalRequired, 1),
            sample_lane("C", EscalationMode::HumanHandoff, 0),
        ];
        let summary = summarize_escalations(&lanes);
        assert_eq!(summary.total, 3);
        assert_eq!(summary.autonomous, 1);
        assert_eq!(summary.approval_required, 1);
        assert_eq!(summary.human_handoff, 1);
        assert_eq!(summary.evidence_checks, 3); // 2 + 1 + 0
    }

    #[test]
    fn test_classify_sufficient_evidence() {
        let lane = sample_lane("A", EscalationMode::HumanHandoff, 4);
        assert_eq!(classify_escalation(&lane), EscalationMode::Autonomous);
    }

    #[test]
    fn test_classify_known_guardrail() {
        let lane = EscalationLane::new(
            "A", "test", EscalationMode::HumanHandoff, "revenue_threshold", "owner"
        );
        assert_eq!(classify_escalation(&lane), EscalationMode::ApprovalRequired);
    }

    #[test]
    fn test_classify_human_handoff() {
        let lane = EscalationLane::new(
            "A", "test", EscalationMode::HumanHandoff, "unknown_guardrail", "owner"
        );
        assert_eq!(classify_escalation(&lane), EscalationMode::HumanHandoff);
    }

    #[test]
    fn test_priority_human_handoff() {
        let lane = sample_lane("A", EscalationMode::HumanHandoff, 0);
        assert_eq!(escalation_priority(&lane), 3);
    }

    #[test]
    fn test_priority_approval_required() {
        let lane = sample_lane("A", EscalationMode::ApprovalRequired, 0);
        assert_eq!(escalation_priority(&lane), 2);
    }

    #[test]
    fn test_priority_autonomous() {
        let lane = sample_lane("A", EscalationMode::Autonomous, 0);
        assert_eq!(escalation_priority(&lane), 1);
    }

    #[test]
    fn test_priority_evidence_bonus() {
        let lane = sample_lane("A", EscalationMode::Autonomous, 5);
        assert_eq!(escalation_priority(&lane), 4); // 1 + 3 (capped)
    }

    #[test]
    fn test_classified_priority() {
        let lane = sample_lane("A", EscalationMode::HumanHandoff, 4);
        // classify_escalation would return Autonomous (4 evidence >= 3)
        assert_eq!(classified_priority(&lane), 4); // 1 (Autonomous) + 3 (capped)
    }

    #[test]
    fn test_lanes_by_priority_sort() {
        let lanes = vec![
            sample_lane("A", EscalationMode::Autonomous, 0),
            sample_lane("B", EscalationMode::HumanHandoff, 2),
            sample_lane("C", EscalationMode::ApprovalRequired, 1),
        ];
        let sorted = lanes_by_priority(&lanes);
        assert_eq!(sorted[0].id, "B"); // priority 3+2=5
        assert_eq!(sorted[2].id, "A"); // priority 1
    }

    #[test]
    fn test_count_reclassifications() {
        let lanes = vec![
            sample_lane("A", EscalationMode::Autonomous, 4), // Already Autonomous
            EscalationLane::new("B", "test", EscalationMode::HumanHandoff, "revenue_threshold", "owner"), // Would be ApprovalRequired
        ];
        assert_eq!(count_reclassifications(&lanes), 1);
    }

    #[test]
    fn test_filter_by_mode() {
        let lanes = vec![
            sample_lane("A", EscalationMode::Autonomous, 0),
            sample_lane("B", EscalationMode::ApprovalRequired, 0),
            sample_lane("C", EscalationMode::Autonomous, 0),
        ];
        let filtered = filter_by_mode(&lanes, EscalationMode::Autonomous);
        assert_eq!(filtered.len(), 2);
    }

    #[test]
    fn test_human_oversight_ratio() {
        let mut summary = EscalationSummary::zero();
        summary.total = 100;
        summary.approval_required = 30;
        summary.human_handoff = 20;
        let ratio = human_oversight_ratio(&summary);
        assert!((ratio - 0.5).abs() < 0.001);
    }

    #[test]
    fn test_human_oversight_ratio_zero_total() {
        let summary = EscalationSummary::zero();
        assert!((human_oversight_ratio(&summary) - 0.0).abs() < 0.001);
    }
}
