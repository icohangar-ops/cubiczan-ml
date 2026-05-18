use serde::{Deserialize, Serialize};

/// Escalation handling mode for commerce decisions.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum EscalationMode {
    Autonomous,
    ApprovalRequired,
    HumanHandoff,
}

impl std::fmt::Display for EscalationMode {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            EscalationMode::Autonomous => write!(f, "Autonomous"),
            EscalationMode::ApprovalRequired => write!(f, "ApprovalRequired"),
            EscalationMode::HumanHandoff => write!(f, "HumanHandoff"),
        }
    }
}

/// An escalation lane defining the handling mode and guardrails for a decision category.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EscalationLane {
    pub id: String,
    pub lane: String,
    pub mode: EscalationMode,
    pub guardrail: String,
    pub evidence: Vec<String>,
    pub owner: String,
}

impl EscalationLane {
    /// Create a new escalation lane.
    pub fn new(
        id: impl Into<String>,
        lane: impl Into<String>,
        mode: EscalationMode,
        guardrail: impl Into<String>,
        owner: impl Into<String>,
    ) -> Self {
        Self {
            id: id.into(),
            lane: lane.into(),
            mode,
            guardrail: guardrail.into(),
            evidence: Vec::new(),
            owner: owner.into(),
        }
    }
}

/// Summary counts of escalations by mode.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EscalationSummary {
    pub total: u32,
    pub autonomous: u32,
    pub approval_required: u32,
    pub human_handoff: u32,
    pub evidence_checks: u32,
}

impl EscalationSummary {
    /// Create a zero-initialized summary.
    pub fn zero() -> Self {
        Self {
            total: 0,
            autonomous: 0,
            approval_required: 0,
            human_handoff: 0,
            evidence_checks: 0,
        }
    }

    /// The fraction of escalations handled autonomously (0.0 to 1.0).
    pub fn autonomous_rate(&self) -> f64 {
        if self.total == 0 {
            return 0.0;
        }
        self.autonomous as f64 / self.total as f64
    }
}

/// Velocity tier for commerce action throughput classification.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum VelocityTier {
    Cold,
    Warm,
    Hot,
    Critical,
}

impl std::fmt::Display for VelocityTier {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            VelocityTier::Cold => write!(f, "Cold"),
            VelocityTier::Warm => write!(f, "Warm"),
            VelocityTier::Hot => write!(f, "Hot"),
            VelocityTier::Critical => write!(f, "Critical"),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_escalation_mode_display() {
        assert_eq!(EscalationMode::Autonomous.to_string(), "Autonomous");
        assert_eq!(EscalationMode::ApprovalRequired.to_string(), "ApprovalRequired");
        assert_eq!(EscalationMode::HumanHandoff.to_string(), "HumanHandoff");
    }

    #[test]
    fn test_velocity_tier_display() {
        assert_eq!(VelocityTier::Cold.to_string(), "Cold");
        assert_eq!(VelocityTier::Critical.to_string(), "Critical");
    }

    #[test]
    fn test_escalation_lane_new() {
        let lane = EscalationLane::new("L1", "pricing", EscalationMode::Autonomous, "<$1000", "bot");
        assert_eq!(lane.id, "L1");
        assert!(lane.evidence.is_empty());
    }

    #[test]
    fn test_escalation_summary_zero() {
        let s = EscalationSummary::zero();
        assert_eq!(s.total, 0);
        assert_eq!(s.autonomous_rate(), 0.0);
    }

    #[test]
    fn test_escalation_summary_autonomous_rate() {
        let mut s = EscalationSummary::zero();
        s.total = 100;
        s.autonomous = 75;
        assert!((s.autonomous_rate() - 0.75).abs() < 0.001);
    }
}
