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
// Constraint Altitude Types (replaces VCL)
// ============================================================================

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum ConstraintAltitude {
    #[serde(rename = "T1_TACTICAL")]
    TACTICAL,
    #[serde(rename = "T2_STRUCTURAL")]
    STRUCTURAL,
    #[serde(rename = "T3_STRATEGIC")]
    STRATEGIC,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ConstraintDiagnosis {
    pub item: String,
    pub symptom_altitude: ConstraintAltitude,
    pub constraint_altitude: ConstraintAltitude,
    pub diagnosis: String,
}

impl ConstraintDiagnosis {
    pub fn validate(&self) -> Vec<String> {
        let mut errors = Vec::new();
        if self.diagnosis.is_empty() {
            errors.push("diagnosis is required".into());
        }
        // Rule: higher tier constrains lower
        if (self.symptom_altitude as u8) < (self.constraint_altitude as u8) {
            // symptom is lower than constraint — this is expected and valid
        }
        errors
    }
}

// ============================================================================
// ModelParityDelta
// ============================================================================

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum ModelParityDelta {
    NONE,
    MINOR,
    SIGNIFICANT,
}

// ============================================================================
// PreFlightDeclaration
// ============================================================================

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PriorBelief {
    pub item: String,
    pub position: String,
    pub confidence: u32, // 0-100
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PreFlightDeclaration {
    pub party: String, // "origin" or "partner"
    pub prior_beliefs: Vec<PriorBelief>,
    pub blind_spots: Vec<String>, // 1-3 areas
}

impl PreFlightDeclaration {
    pub fn validate(&self) -> Vec<String> {
        let mut errors = Vec::new();
        if self.prior_beliefs.is_empty() {
            errors.push("prior_beliefs must have at least one entry".into());
        }
        for b in &self.prior_beliefs {
            if b.position.is_empty() {
                errors.push(format!("{}: position is required", b.item));
            }
            if b.confidence > 100 {
                errors.push(format!("{}: confidence must be 0-100", b.item));
            }
        }
        if self.blind_spots.is_empty() || self.blind_spots.len() > 3 {
            errors.push("blind_spots must include 1-3 items".into());
        }
        errors
    }
}

// ============================================================================
// ClosureAudit
// ============================================================================

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct ClosureAudit {
    pub blind_spots_resolved: Vec<String>,
    pub blind_spots_accepted: Vec<String>,
    pub vulnerabilities_addressed: Vec<String>,
    pub vulnerabilities_accepted_risk: Vec<String>,
}

impl ClosureAudit {
    pub fn has_accepted_risks(&self) -> bool {
        !self.vulnerabilities_accepted_risk.is_empty()
    }
}

// ============================================================================
// SectionLimits
// ============================================================================

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SectionLimits {
    pub max_agreement_lines: usize,   // default 3
    pub max_winner_framing_sentences: usize, // default 4
    pub max_objections: usize,        // default 5
    pub max_objection_lines: usize,   // default 2
}

impl Default for SectionLimits {
    fn default() -> Self {
        Self {
            max_agreement_lines: 3,
            max_winner_framing_sentences: 4,
            max_objections: 5,
            max_objection_lines: 2,
        }
    }
}

// ============================================================================
// SuperServeValidation
// ============================================================================

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SuperServeValidation {
    pub sandbox_id: String,
    pub proposal_item: String,
    pub exit_code: i32,
    pub stdout: String,
    pub passed: bool,
}

impl SuperServeValidation {
    pub fn as_third_party(&self) -> ThirdPartyValidation {
        ThirdPartyValidation {
            validator: format!("superserve:{}", self.sandbox_id),
            item: self.proposal_item.clone(),
            challenge: "sandbox execution".into(),
            result: if self.passed { ValidationResult::CONFIRM } else { ValidationResult::REJECT },
            rationale: format!("exit_code={}, stdout_truncated={}", self.exit_code, &self.stdout[..self.stdout.len().min(200)]),
        }
    }
}

// ============================================================================
// GuardTrigger
// ============================================================================

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GuardTrigger {
    pub guard_name: String,
    pub item: String,
    pub reason: String,
    pub severity: String, // "WARNING" or "CRITICAL"
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
    #[serde(default)]
    pub constraint_diagnosis: HashMap<String, ConstraintDiagnosis>,
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
            constraint_diagnosis: HashMap::new(),
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
    pub delta: ModelParityDelta,
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
    pub constraint_diagnoses: Vec<ConstraintDiagnosis>,
    #[serde(default)]
    pub preflight_origin: Option<PreFlightDeclaration>,
    #[serde(default)]
    pub preflight_partner: Option<PreFlightDeclaration>,
    #[serde(default)]
    pub section_limits: SectionLimits,
    #[serde(default)]
    pub guard_triggers: Vec<GuardTrigger>,
    #[serde(default)]
    pub closure_audit: Option<ClosureAudit>,
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
            constraint_diagnoses: Vec::new(),
            preflight_origin: None,
            preflight_partner: None,
            section_limits: SectionLimits::default(),
            guard_triggers: Vec::new(),
            closure_audit: None,
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

    // --- ConstraintDiagnosis tests ---

    #[test]
    fn test_constraint_diagnosis_valid() {
        let cd = ConstraintDiagnosis {
            item: "item1".into(),
            symptom_altitude: ConstraintAltitude::TACTICAL,
            constraint_altitude: ConstraintAltitude::STRUCTURAL,
            diagnosis: "Fix the structural constraint first.".into(),
        };
        assert!(cd.validate().is_empty());
    }

    #[test]
    fn test_constraint_diagnosis_empty_diagnosis() {
        let cd = ConstraintDiagnosis {
            item: "item1".into(),
            symptom_altitude: ConstraintAltitude::TACTICAL,
            constraint_altitude: ConstraintAltitude::STRUCTURAL,
            diagnosis: String::new(),
        };
        let errors = cd.validate();
        assert!(errors.iter().any(|e| e.contains("diagnosis is required")));
    }

    #[test]
    fn test_constraint_altitude_ordering() {
        // TACTICAL < STRUCTURAL < STRATEGIC
        assert!((ConstraintAltitude::TACTICAL as u8) < (ConstraintAltitude::STRUCTURAL as u8));
        assert!((ConstraintAltitude::STRUCTURAL as u8) < (ConstraintAltitude::STRATEGIC as u8));
    }

    #[test]
    fn test_constraint_diagnosis_serde_roundtrip() {
        let cd = ConstraintDiagnosis {
            item: "item1".into(),
            symptom_altitude: ConstraintAltitude::TACTICAL,
            constraint_altitude: ConstraintAltitude::STRATEGIC,
            diagnosis: "Strategic fix needed.".into(),
        };
        let json = serde_json::to_string(&cd).unwrap();
        let restored: ConstraintDiagnosis = serde_json::from_str(&json).unwrap();
        assert_eq!(restored.symptom_altitude, ConstraintAltitude::TACTICAL);
        assert_eq!(restored.constraint_altitude, ConstraintAltitude::STRATEGIC);
    }

    // --- ModelParityDelta tests ---

    #[test]
    fn test_model_parity_delta_serde() {
        for delta in [ModelParityDelta::NONE, ModelParityDelta::MINOR, ModelParityDelta::SIGNIFICANT] {
            let json = serde_json::to_string(&delta).unwrap();
            let restored: ModelParityDelta = serde_json::from_str(&json).unwrap();
            assert_eq!(delta, restored);
        }
    }

    #[test]
    fn test_model_parity_check_with_delta() {
        let mpc = ModelParityCheck {
            origin: "claude-opus-4".into(),
            partner: "claude-haiku-3".into(),
            delta: ModelParityDelta::SIGNIFICANT,
            advisory: None,
        };
        assert_eq!(mpc.delta, ModelParityDelta::SIGNIFICANT);
    }

    // --- PreFlightDeclaration tests ---

    #[test]
    fn test_preflight_declaration_valid() {
        let pfd = PreFlightDeclaration {
            party: "origin".into(),
            prior_beliefs: vec![
                PriorBelief {
                    item: "item1".into(),
                    position: "for it".into(),
                    confidence: 80,
                },
            ],
            blind_spots: vec!["area1".into()],
        };
        assert!(pfd.validate().is_empty());
    }

    #[test]
    fn test_preflight_declaration_empty_beliefs() {
        let pfd = PreFlightDeclaration {
            party: "origin".into(),
            prior_beliefs: vec![],
            blind_spots: vec!["area1".into()],
        };
        let errors = pfd.validate();
        assert!(errors.iter().any(|e| e.contains("prior_beliefs")));
    }

    #[test]
    fn test_preflight_declaration_empty_position() {
        let pfd = PreFlightDeclaration {
            party: "origin".into(),
            prior_beliefs: vec![
                PriorBelief {
                    item: "item1".into(),
                    position: String::new(),
                    confidence: 50,
                },
            ],
            blind_spots: vec!["area1".into()],
        };
        let errors = pfd.validate();
        assert!(errors.iter().any(|e| e.contains("position is required")));
    }

    #[test]
    fn test_preflight_declaration_confidence_exceeds_100() {
        let pfd = PreFlightDeclaration {
            party: "origin".into(),
            prior_beliefs: vec![
                PriorBelief {
                    item: "item1".into(),
                    position: "for it".into(),
                    confidence: 150,
                },
            ],
            blind_spots: vec!["area1".into()],
        };
        let errors = pfd.validate();
        assert!(errors.iter().any(|e| e.contains("confidence must be 0-100")));
    }

    #[test]
    fn test_preflight_declaration_too_many_blind_spots() {
        let pfd = PreFlightDeclaration {
            party: "origin".into(),
            prior_beliefs: vec![
                PriorBelief {
                    item: "item1".into(),
                    position: "for it".into(),
                    confidence: 50,
                },
            ],
            blind_spots: vec!["a".into(), "b".into(), "c".into(), "d".into()],
        };
        let errors = pfd.validate();
        assert!(errors.iter().any(|e| e.contains("blind_spots must include 1-3")));
    }

    #[test]
    fn test_preflight_declaration_no_blind_spots() {
        let pfd = PreFlightDeclaration {
            party: "origin".into(),
            prior_beliefs: vec![
                PriorBelief {
                    item: "item1".into(),
                    position: "for it".into(),
                    confidence: 50,
                },
            ],
            blind_spots: vec![],
        };
        let errors = pfd.validate();
        assert!(errors.iter().any(|e| e.contains("blind_spots must include 1-3")));
    }

    #[test]
    fn test_preflight_declaration_multiple_beliefs_valid() {
        let pfd = PreFlightDeclaration {
            party: "partner".into(),
            prior_beliefs: vec![
                PriorBelief { item: "item1".into(), position: "pro".into(), confidence: 90 },
                PriorBelief { item: "item2".into(), position: "against".into(), confidence: 60 },
            ],
            blind_spots: vec!["x".into(), "y".into()],
        };
        assert!(pfd.validate().is_empty());
    }

    // --- ClosureAudit tests ---

    #[test]
    fn test_closure_audit_no_accepted_risks() {
        let audit = ClosureAudit::default();
        assert!(!audit.has_accepted_risks());
    }

    #[test]
    fn test_closure_audit_has_accepted_risks() {
        let mut audit = ClosureAudit::default();
        audit.vulnerabilities_accepted_risk.push("risk1".into());
        assert!(audit.has_accepted_risks());
    }

    #[test]
    fn test_closure_audit_default_empty() {
        let audit = ClosureAudit::default();
        assert!(audit.blind_spots_resolved.is_empty());
        assert!(audit.blind_spots_accepted.is_empty());
        assert!(audit.vulnerabilities_addressed.is_empty());
        assert!(audit.vulnerabilities_accepted_risk.is_empty());
    }

    // --- SectionLimits tests ---

    #[test]
    fn test_section_limits_default() {
        let limits = SectionLimits::default();
        assert_eq!(limits.max_agreement_lines, 3);
        assert_eq!(limits.max_winner_framing_sentences, 4);
        assert_eq!(limits.max_objections, 5);
        assert_eq!(limits.max_objection_lines, 2);
    }

    #[test]
    fn test_section_limits_custom() {
        let limits = SectionLimits {
            max_agreement_lines: 5,
            max_winner_framing_sentences: 6,
            max_objections: 3,
            max_objection_lines: 4,
        };
        assert_eq!(limits.max_objections, 3);
    }

    #[test]
    fn test_section_limits_serde() {
        let limits = SectionLimits::default();
        let json = serde_json::to_string(&limits).unwrap();
        let restored: SectionLimits = serde_json::from_str(&json).unwrap();
        assert_eq!(restored.max_agreement_lines, 3);
    }

    // --- SuperServeValidation tests ---

    #[test]
    fn test_superserve_passed() {
        let sv = SuperServeValidation {
            sandbox_id: "sb-001".into(),
            proposal_item: "item1".into(),
            exit_code: 0,
            stdout: "All tests passed.".into(),
            passed: true,
        };
        let tp = sv.as_third_party();
        assert_eq!(tp.validator, "superserve:sb-001");
        assert_eq!(tp.item, "item1");
        assert_eq!(tp.result, ValidationResult::CONFIRM);
    }

    #[test]
    fn test_superserve_failed() {
        let sv = SuperServeValidation {
            sandbox_id: "sb-002".into(),
            proposal_item: "item2".into(),
            exit_code: 1,
            stdout: "Test failed.".into(),
            passed: false,
        };
        let tp = sv.as_third_party();
        assert_eq!(tp.result, ValidationResult::REJECT);
        assert!(tp.rationale.contains("exit_code=1"));
    }

    #[test]
    fn test_superserve_long_stdout_truncated() {
        let long_stdout = "X".repeat(500);
        let sv = SuperServeValidation {
            sandbox_id: "sb-003".into(),
            proposal_item: "item3".into(),
            exit_code: 0,
            stdout: long_stdout,
            passed: true,
        };
        let tp = sv.as_third_party();
        // stdout_truncated should be at most 200 chars
        assert!(tp.rationale.len() < 300);
    }

    // --- GuardTrigger tests ---

    #[test]
    fn test_guard_trigger_fields() {
        let gt = GuardTrigger {
            guard_name: "ModelParityGate".into(),
            item: "decision-x".into(),
            reason: "Significant model gap".into(),
            severity: "CRITICAL".into(),
        };
        assert_eq!(gt.severity, "CRITICAL");
    }

    #[test]
    fn test_guard_trigger_serde() {
        let gt = GuardTrigger {
            guard_name: "PhaseGate".into(),
            item: "item1".into(),
            reason: "Phase 1 not locked".into(),
            severity: "WARNING".into(),
        };
        let json = serde_json::to_string(&gt).unwrap();
        let restored: GuardTrigger = serde_json::from_str(&json).unwrap();
        assert_eq!(restored.guard_name, "PhaseGate");
        assert_eq!(restored.severity, "WARNING");
    }

    // --- StateSnapshot constraint_diagnosis ---

    #[test]
    fn test_state_snapshot_constraint_diagnosis_default() {
        let snap = StateSnapshot::new(Phase::Spec, 1, SessionStatus::EXPLORING, "echo");
        assert!(snap.constraint_diagnosis.is_empty());
    }

    #[test]
    fn test_state_snapshot_with_constraint_diagnosis() {
        let mut snap = StateSnapshot::new(Phase::Spec, 1, SessionStatus::EXPLORING, "echo");
        snap.constraint_diagnosis.insert("item1".into(), ConstraintDiagnosis {
            item: "item1".into(),
            symptom_altitude: ConstraintAltitude::TACTICAL,
            constraint_altitude: ConstraintAltitude::STRUCTURAL,
            diagnosis: "Fix structural first.".into(),
        });
        assert_eq!(snap.constraint_diagnosis.len(), 1);
        assert!(snap.constraint_diagnosis.contains_key("item1"));
    }

    // --- DecisionCase new fields ---

    #[test]
    fn test_decision_case_default_new_fields() {
        let case = DecisionCase::default();
        assert!(case.constraint_diagnoses.is_empty());
        assert!(case.preflight_origin.is_none());
        assert!(case.preflight_partner.is_none());
        assert_eq!(case.section_limits.max_agreement_lines, 3);
        assert!(case.guard_triggers.is_empty());
        assert!(case.closure_audit.is_none());
    }

    #[test]
    fn test_decision_case_with_constraint_diagnoses() {
        let mut case = DecisionCase::new("dc-1", "Test", "finance", "alice");
        case.constraint_diagnoses.push(ConstraintDiagnosis {
            item: "scope1".into(),
            symptom_altitude: ConstraintAltitude::TACTICAL,
            constraint_altitude: ConstraintAltitude::STRATEGIC,
            diagnosis: "Strategic constraint.".into(),
        });
        assert_eq!(case.constraint_diagnoses.len(), 1);
    }

    #[test]
    fn test_decision_case_with_preflight() {
        let mut case = DecisionCase::new("dc-1", "Test", "finance", "alice");
        case.preflight_origin = Some(PreFlightDeclaration {
            party: "origin".into(),
            prior_beliefs: vec![PriorBelief {
                item: "i1".into(), position: "pro".into(), confidence: 80,
            }],
            blind_spots: vec!["blind1".into()],
        });
        assert!(case.preflight_origin.is_some());
    }

    #[test]
    fn test_decision_case_with_guard_triggers() {
        let mut case = DecisionCase::new("dc-1", "Test", "finance", "alice");
        case.guard_triggers.push(GuardTrigger {
            guard_name: "PhaseGate".into(),
            item: "all".into(),
            reason: "Phase 1 not locked".into(),
            severity: "WARNING".into(),
        });
        assert_eq!(case.guard_triggers.len(), 1);
    }

    #[test]
    fn test_decision_case_with_closure_audit() {
        let mut case = DecisionCase::new("dc-1", "Test", "finance", "alice");
        let mut audit = ClosureAudit::default();
        audit.vulnerabilities_accepted_risk.push("risk1".into());
        case.closure_audit = Some(audit);
        assert!(case.closure_audit.as_ref().unwrap().has_accepted_risks());
    }

    #[test]
    fn test_decision_case_serde_roundtrip_with_new_fields() {
        let mut case = DecisionCase::new("dc-123", "Investment", "finance", "bob");
        case.guard_triggers.push(GuardTrigger {
            guard_name: "ModelParityGate".into(),
            item: "all".into(),
            reason: "gap".into(),
            severity: "CRITICAL".into(),
        });
        case.section_limits.max_objections = 10;
        let json = serde_json::to_string(&case).unwrap();
        let restored: DecisionCase = serde_json::from_str(&json).unwrap();
        assert_eq!(restored.guard_triggers.len(), 1);
        assert_eq!(restored.section_limits.max_objections, 10);
    }
}
