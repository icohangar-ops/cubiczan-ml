//! Canonical data model for Consensus Hardening Protocol.

use serde::{Deserialize, Serialize};
use std::collections::HashMap;

// ============================================================================
// Enums
// ============================================================================

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, Default)]
#[repr(i32)]
pub enum Phase { #[default] Foundation = 0, Spec = 1, Implementation = 2 }

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum Verdict {
    #[serde(rename = "PASS")]
    PASS,
    #[serde(rename = "FAIL")]
    FAIL,
    #[serde(rename = "HALT")]
    HALT,
    #[serde(rename = "REFRAME")]
    REFRAME,
    #[serde(rename = "ITERATE")]
    ITERATE,
    #[serde(rename = "CONVERGED")]
    CONVERGED,
    #[serde(rename = "PHASE_GATE_FAIL")]
    PHASE_GATE_FAIL,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize, Default)]
pub enum SessionStatus {
    #[default]
    EXPLORING,
    PROVISIONAL,
    PROVISIONAL_LOCK,
    LOCKED,
    CONVERGED,
    UNRESOLVED,
    REQUIRES_HUMAN_VERIFICATION,
    REFRAME_REQUIRED,
    HALT,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum ValidationResult {
    CONFIRM,
    REJECT,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum ModelTier {
    #[serde(rename = "small")]
    SMALL,
    #[serde(rename = "mid")]
    MID,
    #[serde(rename = "high")]
    HIGH,
    #[serde(rename = "frontier")]
    FRONTIER,
    #[serde(rename = "unknown")]
    UNKNOWN,
}

// ============================================================================
// Structs
// ============================================================================

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DevilsAdvocateRound {
    pub phase: Phase,
    pub round_number: u32,
    pub why_direction_wrong: String,
    pub what_not_seeing: String,
    pub false_consensus_risk: String,
    #[serde(default)]
    pub structural_vulnerabilities: Vec<String>,
}

impl DevilsAdvocateRound {
    pub fn validate(&self) -> Vec<String> {
        let mut errors = Vec::new();
        if self.why_direction_wrong.is_empty() {
            errors.push("why_direction_wrong is required".into());
        }
        if self.what_not_seeing.is_empty() {
            errors.push("what_not_seeing is required".into());
        }
        if self.false_consensus_risk.is_empty() {
            errors.push("false_consensus_risk is required".into());
        }
        if self.structural_vulnerabilities.len() > 3 {
            errors.push("structural_vulnerabilities is limited to three items".into());
        }
        errors
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VCLDiagnosis {
    pub item: String,
    pub symptom_altitude: String,
    pub constraint_altitude: String,
    pub diagnosis: String,
}

impl VCLDiagnosis {
    pub fn validate(&self) -> Vec<String> {
        let mut errors = Vec::new();
        let allowed: Vec<String> = (1..=10).map(|i| format!("R{}", i)).collect();
        let symptom_ok = self.symptom_altitude.split_whitespace().next()
            .map_or(false, |s| allowed.iter().any(|a| a == s));
        if !symptom_ok {
            errors.push("symptom_altitude must start with R1-R10".into());
        }
        let constraint_ok = self.constraint_altitude.split_whitespace().next()
            .map_or(false, |s| allowed.iter().any(|a| a == s));
        if !constraint_ok {
            errors.push("constraint_altitude must start with R1-R10".into());
        }
        if self.diagnosis.is_empty() {
            errors.push("diagnosis is required".into());
        }
        errors
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StateSnapshot {
    #[serde(rename = "PHASE")]
    pub phase: Phase,
    #[serde(rename = "ROUND")]
    pub round_number: String, // stored as "X/5"
    #[serde(rename = "STATUS")]
    pub status: SessionStatus,
    #[serde(rename = "PAYLOAD_ECHO")]
    pub payload_echo: String,
    #[serde(rename = "FOUNDATION_SCORE", skip_serializing_if = "Option::is_none")]
    pub foundation_score: Option<i32>,
    #[serde(default)]
    pub locked: Vec<String>,
    #[serde(default)]
    pub provisional: Vec<String>,
    #[serde(default)]
    pub provisional_lock: Vec<String>,
    #[serde(default)]
    pub flip_active: Vec<String>,
    #[serde(default)]
    pub blind_spots_acknowledged: HashMap<String, Vec<String>>,
    #[serde(default)]
    pub structural_vulnerabilities: Vec<String>,
    #[serde(default)]
    pub third_party_pending: Vec<String>,
}

impl StateSnapshot {
    /// Create with round_number as u32; internally stored as "X/5"
    pub fn new(phase: Phase, round_number: u32, status: SessionStatus, payload_echo: &str) -> Self {
        Self {
            phase,
            round_number: format!("{}/5", round_number),
            status,
            payload_echo: payload_echo.into(),
            foundation_score: None,
            locked: Vec::new(),
            provisional: Vec::new(),
            provisional_lock: Vec::new(),
            flip_active: Vec::new(),
            blind_spots_acknowledged: HashMap::new(),
            structural_vulnerabilities: Vec::new(),
            third_party_pending: Vec::new(),
        }
    }

    /// Extract round number as u32 from "X/5" format
    pub fn round_number_u32(&self) -> u32 {
        self.round_number.split('/').next()
            .and_then(|s| s.parse().ok())
            .unwrap_or(0)
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ContextCheck {
    pub memory_tools: String,
    #[serde(default)]
    pub prior_sessions_count: u32,
    #[serde(default)]
    pub prior_lock_versions: Vec<String>,
    #[serde(default)]
    pub legacy_warning: bool,
    #[serde(default)]
    pub related_locks: Vec<String>,
    #[serde(default)]
    pub assessment: String,
    #[serde(default)]
    pub action: String,
}

impl Default for ContextCheck {
    fn default() -> Self {
        Self {
            memory_tools: "UNAVAILABLE".into(),
            prior_sessions_count: 0,
            prior_lock_versions: Vec::new(),
            legacy_warning: false,
            related_locks: Vec::new(),
            assessment: "SPARSE".into(),
            action: "PROCEED".into(),
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelParityCheck {
    pub origin: String,
    pub partner: String,
    pub delta: String,
    pub advisory: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct Dossier {
    pub core_problem: String,
    #[serde(default)]
    pub goal_state: Vec<String>,
    #[serde(default)]
    pub current_state: Vec<String>,
    #[serde(default)]
    pub prior_decisions: Vec<String>,
    #[serde(default)]
    pub constraints: Vec<String>,
    #[serde(default)]
    pub unknowns: Vec<String>,
    #[serde(default)]
    pub scope: Vec<String>,
    #[serde(default)]
    pub origin_direction: Vec<String>,
    #[serde(default)]
    pub prior_round_summary: Vec<String>,
    #[serde(default)]
    pub unknowns_carried: Vec<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub foundation_score: Option<i32>,
    #[serde(default)]
    pub structural_vulnerabilities: Vec<String>,
}

impl Dossier {
    pub fn validate(&self) -> Vec<String> {
        let mut errors = Vec::new();
        if self.core_problem.is_empty() || self.core_problem == "UNKNOWN" {
            errors.push("core_problem is required".into());
        }
        let populated = [
            &self.goal_state,
            &self.current_state,
            &self.constraints,
            &self.scope,
        ].iter().filter(|v| !v.is_empty()).count();
        if populated < 3 {
            errors.push("dossier must include at least three populated context sections".into());
        }
        errors
    }
}

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct FoundationDisclosure {
    #[serde(default)]
    pub weakest_assumptions: Vec<String>,
    #[serde(default)]
    pub invalidation_conditions: Vec<String>,
    pub key_vulnerability: String,
}

impl FoundationDisclosure {
    pub fn validate(&self) -> Vec<String> {
        let mut errors = Vec::new();
        if self.weakest_assumptions.is_empty() || self.weakest_assumptions.len() > 3 {
            errors.push("weakest_assumptions must include 1-3 items".into());
        }
        if self.invalidation_conditions.is_empty() || self.invalidation_conditions.len() > 2 {
            errors.push("invalidation_conditions must include 1-2 items".into());
        }
        if self.key_vulnerability.is_empty() {
            errors.push("key_vulnerability is required".into());
        }
        errors
    }
}

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct FoundationAttack {
    #[serde(default)]
    pub assumption_attacks: Vec<String>,
    #[serde(default)]
    pub invalidation_exploitation: Vec<String>,
    pub vulnerability_strike: String,
    pub foundation_score: i32,
    #[serde(default)]
    pub attack_summary: String,
}

impl FoundationAttack {
    pub fn validate(&self) -> Vec<String> {
        let mut errors = Vec::new();
        if self.assumption_attacks.is_empty() {
            errors.push("assumption_attacks is required".into());
        }
        if self.vulnerability_strike.is_empty() {
            errors.push("vulnerability_strike is required".into());
        }
        if !(0..=100).contains(&self.foundation_score) {
            errors.push("foundation_score must be between 0 and 100".into());
        }
        errors
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ThirdPartyValidation {
    pub validator: String,
    pub item: String,
    pub challenge: String,
    pub result: ValidationResult,
    pub rationale: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RoundRecord {
    pub decision_id: String,
    pub phase: Phase,
    pub round_number: u32,
    pub payload_id: String,
    #[serde(default)]
    pub origin_packet: String,
    #[serde(default)]
    pub partner_packet: String,
    #[serde(default)]
    pub payload_echo_confirmed: bool,
    #[serde(default)]
    pub state_snapshot: serde_json::Value,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DecisionCase {
    pub decision_id: String,
    pub title: String,
    pub domain: String,
    pub created_at: String,
    pub owner: String,
    #[serde(default)]
    pub status: SessionStatus,
    #[serde(default)]
    pub high_stakes: bool,
    #[serde(default)]
    pub current_phase: Phase,
    #[serde(default)]
    pub current_round: u32,
    #[serde(default)]
    pub origin_system: String,
    #[serde(default)]
    pub origin_model: String,
    #[serde(default)]
    pub partner_system: String,
    #[serde(default)]
    pub partner_model: String,
    pub context_check: Option<ContextCheck>,
    pub model_parity: Option<ModelParityCheck>,
    pub dossier: Option<Dossier>,
    pub foundation_score: Option<i32>,
    #[serde(default)]
    pub locked_decisions: Vec<String>,
    #[serde(default)]
    pub structural_vulnerabilities: Vec<String>,
    #[serde(default)]
    pub blind_spots: Vec<String>,
    #[serde(default)]
    pub flip_criteria: Vec<String>,
    #[serde(default)]
    pub devil_advocate_rounds: Vec<DevilsAdvocateRound>,
    #[serde(default)]
    pub vcl_diagnoses: Vec<VCLDiagnosis>,
    #[serde(default)]
    pub state_snapshots: Vec<StateSnapshot>,
    #[serde(default)]
    pub third_party_log: Vec<ThirdPartyValidation>,
    #[serde(default)]
    pub rounds: Vec<RoundRecord>,
}

impl DecisionCase {
    pub fn add_round(&mut self, record: RoundRecord) {
        self.current_phase = record.phase;
        self.current_round = record.round_number;
        self.rounds.push(record);
    }

    pub fn new(id: &str, title: &str, domain: &str, owner: &str) -> Self {
        Self {
            decision_id: id.into(),
            title: title.into(),
            domain: domain.into(),
            created_at: chrono::Utc::now().to_rfc3339(),
            owner: owner.into(),
            origin_system: "Claude".into(),
            origin_model: "UNKNOWN".into(),
            partner_system: "UNKNOWN".into(),
            partner_model: "UNKNOWN".into(),
            status: SessionStatus::EXPLORING,
            current_phase: Phase::Foundation,
            ..Default::default()
        }
    }
}

impl Default for DecisionCase {
    fn default() -> Self {
        Self {
            decision_id: String::new(),
            title: String::new(),
            domain: String::new(),
            created_at: String::new(),
            owner: String::new(),
            status: SessionStatus::EXPLORING,
            high_stakes: false,
            current_phase: Phase::Foundation,
            current_round: 0,
            origin_system: "Claude".into(),
            origin_model: "UNKNOWN".into(),
            partner_system: "UNKNOWN".into(),
            partner_model: "UNKNOWN".into(),
            context_check: None,
            model_parity: None,
            dossier: None,
            foundation_score: None,
            locked_decisions: Vec::new(),
            structural_vulnerabilities: Vec::new(),
            blind_spots: Vec::new(),
            flip_criteria: Vec::new(),
            devil_advocate_rounds: Vec::new(),
            vcl_diagnoses: Vec::new(),
            state_snapshots: Vec::new(),
            third_party_log: Vec::new(),
            rounds: Vec::new(),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_decision_case_new() {
        let case = DecisionCase::new("dc-001", "Test Decision", "finance", "alice");
        assert_eq!(case.status, SessionStatus::EXPLORING);
        assert_eq!(case.current_phase, Phase::Foundation);
    }

    #[test]
    fn test_add_round() {
        let mut case = DecisionCase::new("dc-001", "Test", "finance", "alice");
        let record = RoundRecord {
            decision_id: "dc-001".into(),
            phase: Phase::Spec,
            round_number: 1,
            payload_id: "ABC123".into(),
            origin_packet: String::new(),
            partner_packet: String::new(),
            payload_echo_confirmed: false,
            state_snapshot: serde_json::Value::Null,
        };
        case.add_round(record);
        assert_eq!(case.current_phase, Phase::Spec);
        assert_eq!(case.current_round, 1);
        assert_eq!(case.rounds.len(), 1);
    }

    #[test]
    fn test_devils_advocate_validate() {
        let da = DevilsAdvocateRound {
            phase: Phase::Foundation,
            round_number: 0,
            why_direction_wrong: "Test".into(),
            what_not_seeing: "Test".into(),
            false_consensus_risk: "Test".into(),
            structural_vulnerabilities: vec!["a".into(), "b".into()],
        };
        assert!(da.validate().is_empty());
    }

    #[test]
    fn test_devils_advocate_validate_too_many_vulns() {
        let da = DevilsAdvocateRound {
            phase: Phase::Foundation, round_number: 0,
            why_direction_wrong: "T".into(), what_not_seeing: "T".into(),
            false_consensus_risk: "T".into(),
            structural_vulnerabilities: vec!["a".into(), "b".into(), "c".into(), "d".into()],
        };
        assert_eq!(da.validate().len(), 1);
    }

    #[test]
    fn test_dossier_validate() {
        let d = Dossier {
            core_problem: "Real problem".into(),
            goal_state: vec!["goal1".into()],
            current_state: vec!["state1".into()],
            constraints: vec!["c1".into()],
            ..Default::default()
        };
        assert!(d.validate().is_empty());
    }

    #[test]
    fn test_dossier_validate_too_few_sections() {
        let d = Dossier {
            core_problem: "Problem".into(),
            goal_state: vec!["g1".into()],
            ..Default::default()
        };
        assert!(!d.validate().is_empty());
    }

    #[test]
    fn test_foundation_disclosure_validate() {
        let fd = FoundationDisclosure {
            weakest_assumptions: vec!["assumption1".into()],
            invalidation_conditions: vec!["cond1".into()],
            key_vulnerability: "vuln".into(),
        };
        assert!(fd.validate().is_empty());
    }

    #[test]
    fn test_foundation_attack_validate() {
        let fa = FoundationAttack {
            assumption_attacks: vec!["attack1".into()],
            vulnerability_strike: "strike".into(),
            foundation_score: 75,
            ..Default::default()
        };
        assert!(fa.validate().is_empty());
    }

    #[test]
    fn test_serde_round_trip_decision_case() {
        let case = DecisionCase::new("dc-123", "Investment Plan", "finance", "bob");
        let json = serde_json::to_string(&case).unwrap();
        let restored: DecisionCase = serde_json::from_str(&json).unwrap();
        assert_eq!(restored.decision_id, "dc-123");
        assert_eq!(restored.title, "Investment Plan");
    }

    #[test]
    fn test_state_snapshot_round_format() {
        let snap = StateSnapshot::new(Phase::Spec, 3, SessionStatus::PROVISIONAL_LOCK, "echo");
        assert_eq!(snap.round_number, "3/5");
        assert_eq!(snap.round_number_u32(), 3);
    }
}
