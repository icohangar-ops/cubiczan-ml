//! Strict CHP packet, council, closure, and verification contracts.

use std::collections::{HashMap, HashSet};
use crate::models::*;
use crate::payloads::validate_payload_envelope;
use crate::rounds::MAX_ROUNDS;

pub const ORIGIN_REQUIRED_SECTIONS: &[&str] = &[
    "1. CORE_PROBLEM_STATEMENT",
    "2. PARTNER_SYSTEM_PACKET",
    "3. TRANSMISSION_CHECKLIST",
];

pub const PARTNER_REQUIRED_SECTIONS: &[&str] = &[
    "ITEM_AGREEMENTS",
    "WINNER_FRAMING",
    "SCORING_TABLE",
    "OBJECTIONS",
    "FRAMEWORKS",
    "CONVERGENCE_PLAN",
    "STATE_SNAPSHOT",
];

// ============================================================================
// ItemAgreement
// ============================================================================

#[derive(Debug, Clone, Default)]
pub struct ItemAgreement {
    pub item: String,
    pub score: i32,
    pub status: SessionStatus,
    pub disagreement: String,
    pub revision: String,
    pub flip_criteria: String,
    pub third_party_status: String,
}

impl ItemAgreement {
    pub fn validate(&self) -> Vec<String> {
        let mut errors = Vec::new();
        if !(0..=100).contains(&self.score) {
            errors.push(format!("{}: score must be 0-100", self.item));
        }
        if self.status == SessionStatus::PROVISIONAL && self.flip_criteria.is_empty() {
            errors.push(format!("{}: PROVISIONAL requires FLIP_CRITERIA", self.item));
        }
        if self.status == SessionStatus::PROVISIONAL && self.score >= 90 {
            errors.push(format!("{}: PROVISIONAL score must be below 90", self.item));
        }
        if self.status == SessionStatus::PROVISIONAL_LOCK && self.score < 90 {
            errors.push(format!("{}: PROVISIONAL_LOCK requires score >=90", self.item));
        }
        errors
    }
}

// ============================================================================
// ScoringOption
// ============================================================================

#[derive(Debug, Clone)]
pub struct ScoringOption {
    pub name: String,
    pub clarity: i32,
    pub leverage: i32,
    pub risk: i32,
    pub winner: bool,
    pub elimination_note: String, // required for non-winners
}

impl ScoringOption {
    pub fn total(&self) -> i32 {
        self.clarity + self.leverage + self.risk
    }

    pub fn validate(&self) -> Vec<String> {
        let mut errors = Vec::new();
        for (label, val) in [("clarity", self.clarity), ("leverage", self.leverage), ("risk", self.risk)] {
            if !(0..=10).contains(&val) {
                errors.push(format!("{}: {} must be 0-10", self.name, label));
            }
        }
        // Non-winners must have elimination_note
        if !self.winner && self.elimination_note.is_empty() {
            errors.push(format!("{}: elimination_note is required for non-winners", self.name));
        }
        errors
    }
}

// ============================================================================
// PartnerPacket
// ============================================================================

#[derive(Debug, Clone)]
pub struct PartnerPacket {
    pub item_agreements: Vec<ItemAgreement>,
    pub winner_framing: String,
    pub scoring_table: Vec<ScoringOption>,
    pub objections: Vec<String>,
    pub frameworks: Vec<String>,
    pub convergence_plan: Vec<String>,
    pub state_snapshot: StateSnapshot,
    pub raw_payload: String,
}

impl PartnerPacket {
    pub fn validate(&self) -> Vec<String> {
        let mut errors = Vec::new();
        if !self.raw_payload.is_empty() {
            if !validate_payload_envelope(&self.raw_payload) {
                errors.push("partner packet payload envelope is invalid".into());
            }
            errors.extend(require_ascii(&self.raw_payload));
            for section in PARTNER_REQUIRED_SECTIONS {
                if !self.raw_payload.contains(section) {
                    errors.push(format!("partner packet missing section: {}", section));
                }
            }
        }
        for a in &self.item_agreements {
            errors.extend(a.validate());
        }
        for s in &self.scoring_table {
            errors.extend(s.validate());
        }
        let winners: Vec<_> = self.scoring_table.iter().filter(|s| s.winner).collect();
        if winners.len() != 1 {
            errors.push("SCORING_TABLE requires exactly one winner".into());
        }
        if winners.len() == 1 {
            let totals: HashSet<_> = self.scoring_table.iter().map(|s| s.total()).collect();
            if totals.len() != self.scoring_table.len() {
                errors.push("SCORING_TABLE cannot contain tied total scores".into());
            }
        }
        if self.state_snapshot.round_number_u32() > MAX_ROUNDS {
            errors.push("round_number exceeds max 5".into());
        }
        if self.state_snapshot.round_number_u32() == MAX_ROUNDS {
            if self.item_agreements.iter().any(|a| a.status == SessionStatus::PROVISIONAL) {
                errors.push("Round 5 cannot return PROVISIONAL".into());
            }
            // R5 PROVISIONAL_LOCK without third-party → force UNRESOLVED
            for a in &self.item_agreements {
                if a.status == SessionStatus::PROVISIONAL_LOCK && a.third_party_status.is_empty() {
                    errors.push(format!(
                        "{}: PROVISIONAL_LOCK at Round 5 without third-party validation → must resolve to UNRESOLVED",
                        a.item
                    ));
                }
            }
        }
        errors
    }
}

// ============================================================================
// OriginPacketContract
// ============================================================================

pub struct OriginPacketContract {
    pub raw_payload: String,
}

impl OriginPacketContract {
    pub fn validate(&self) -> Vec<String> {
        let mut errors = Vec::new();
        errors.extend(require_ascii(&self.raw_payload));
        if !validate_payload_envelope(&self.raw_payload) {
            errors.push("origin payload envelope is invalid".into());
        }
        for section in ORIGIN_REQUIRED_SECTIONS {
            if !self.raw_payload.contains(section) {
                errors.push(format!("origin packet missing section: {}", section));
            }
        }
        errors
    }
}

// ============================================================================
// CouncilSpawn
// ============================================================================

pub struct CouncilSpawn {
    pub trigger_reason: String,
    pub composition: Vec<String>,
    pub attack_phase: Vec<String>,
    pub peer_review: Vec<String>,
    pub synthesized_vulnerabilities: Vec<String>,
    pub feed_back_to_round: u32,
}

impl CouncilSpawn {
    pub fn maybe_spawn(high_stakes: bool, confidence_pct: u32, current_round: u32) -> Option<Self> {
        if !high_stakes || confidence_pct >= 85 {
            return None;
        }
        Some(Self {
            trigger_reason: format!("High-stakes decision with confidence {}% below 85%.", confidence_pct),
            composition: vec![
                "Model 1 - Role: Attacker".into(),
                "Model 2 - Role: Attacker".into(),
                "Model 3 - Role: Synthesizer".into(),
            ],
            attack_phase: vec![
                "Attack foundation assumptions.".into(),
                "Attack evidence quality and missing data.".into(),
                "Attack implementation risk and false consensus.".into(),
            ],
            peer_review: vec![
                "Model 1 reviews Model 2 attack for missed constraints.".into(),
                "Model 2 reviews Model 3 synthesis for over-compression.".into(),
                "Model 3 reviews Model 1 attack for unsupported objections.".into(),
            ],
            synthesized_vulnerabilities: vec![
                "Confidence is below the high-stakes threshold.".into(),
                "At least one independent attacker is required before lock.".into(),
                "Human verification should remain active until vulnerabilities close.".into(),
            ],
            feed_back_to_round: MAX_ROUNDS.min(current_round + 1),
        })
    }
}

// ============================================================================
// ConvergenceClosure
// ============================================================================

pub struct ConvergenceClosure {
    pub status: SessionStatus,
    pub foundation_score: Option<i32>,
    pub locked_decisions: Vec<String>,
    pub blind_spots_resolved: Vec<String>,
    pub blind_spots_accepted: Vec<String>,
    pub vulnerabilities_addressed: Vec<String>,
    pub vulnerabilities_accepted_risk: Vec<String>,
    pub session_urls: HashMap<String, String>,
    pub third_party_log: Vec<serde_json::Value>,
}

impl ConvergenceClosure {
    pub fn from_case(case: &DecisionCase, origin_url: &str, partner_url: &str) -> Self {
        let mut urls = HashMap::new();
        urls.insert("Origin".into(), origin_url.into());
        urls.insert("Partner".into(), partner_url.into());
        let tp_log: Vec<serde_json::Value> = case.third_party_log
            .iter()
            .map(|v| serde_json::to_value(v).unwrap_or(serde_json::Value::Null))
            .collect();

        // Add closure audit
        let audit = case.closure_audit.clone().unwrap_or_default();

        // If audit has accepted risks, tag as REQUIRES_HUMAN_VERIFICATION
        let status = if audit.has_accepted_risks() && case.status != SessionStatus::HALT {
            SessionStatus::REQUIRES_HUMAN_VERIFICATION
        } else {
            case.status.clone()
        };

        Self {
            status,
            foundation_score: case.foundation_score,
            locked_decisions: case.locked_decisions.clone(),
            blind_spots_resolved: audit.blind_spots_resolved,
            blind_spots_accepted: audit.blind_spots_accepted,
            vulnerabilities_addressed: audit.vulnerabilities_addressed,
            vulnerabilities_accepted_risk: audit.vulnerabilities_accepted_risk,
            session_urls: urls,
            third_party_log: tp_log,
        }
    }
}

impl Default for ConvergenceClosure {
    fn default() -> Self {
        Self {
            status: SessionStatus::EXPLORING,
            foundation_score: None,
            locked_decisions: Vec::new(),
            blind_spots_resolved: Vec::new(),
            blind_spots_accepted: Vec::new(),
            vulnerabilities_addressed: Vec::new(),
            vulnerabilities_accepted_risk: Vec::new(),
            session_urls: HashMap::new(),
            third_party_log: Vec::new(),
        }
    }
}

// ============================================================================
// InterruptionRecovery
// ============================================================================

pub struct InterruptionRecovery {
    pub phase: Phase,
    pub last_section: String,
    pub decision_state: SessionStatus,
    pub partial_state: serde_json::Value,
    pub foundation_score: Option<i32>,
}

impl InterruptionRecovery {
    pub fn options(&self) -> Vec<&str> {
        vec!["A) Continue", "B) Restart Phase", "C) Next round"]
    }
}

// ============================================================================
// VerificationChecklist
// ============================================================================

pub struct VerificationChecklist {
    pub pre_session: Vec<String>,
    pub phase_0: Vec<String>,
    pub contract: Vec<String>,
    pub truth: Vec<String>,
    pub limits: Vec<String>,
    pub validation: Vec<String>,
}

impl VerificationChecklist {
    pub fn run(case: &DecisionCase, packet: &str) -> Self {
        Self {
            pre_session: vec![
                if case.context_check.is_some() {
                    "Context check executed".into()
                } else {
                    "MISSING context check".into()
                },
                if let Some(ref mp) = case.model_parity {
                    if mp.delta != ModelParityDelta::SIGNIFICANT {
                        "Model parity gate passed".into()
                    } else {
                        "MISSING model parity pass".into()
                    }
                } else {
                    "MISSING model parity pass".into()
                },
            ],
            phase_0: vec![
                if case.foundation_score.is_some() {
                    "Foundation score present".into()
                } else {
                    "MISSING foundation score".into()
                },
                if case.foundation_score.unwrap_or(0) >= 70 {
                    "Foundation >=70".into()
                } else {
                    "REFRAME required".into()
                },
                if !case.devil_advocate_rounds.is_empty() {
                    "Devil's advocate complete".into()
                } else {
                    "MISSING devil's advocate".into()
                },
            ],
            contract: vec![
                if !packet.is_empty() && validate_payload_envelope(packet) {
                    "Payload envelope valid".into()
                } else {
                    "MISSING valid payload envelope".into()
                },
                if !case.state_snapshots.is_empty() {
                    "State snapshot present".into()
                } else {
                    "MISSING state snapshot".into()
                },
                if !case.constraint_diagnoses.is_empty() {
                    "Constraint altitude tagged".into()
                } else {
                    "MISSING constraint diagnosis".into()
                },
            ],
            truth: vec![
                if let Some(ref d) = case.dossier {
                    if !d.unknowns.is_empty() {
                        "Unknowns carried".into()
                    } else {
                        "MISSING unknowns".into()
                    }
                } else {
                    "MISSING unknowns".into()
                },
                if !case.structural_vulnerabilities.is_empty() {
                    "Structural vulnerabilities carried".into()
                } else {
                    "MISSING structural vulnerabilities".into()
                },
            ],
            limits: vec![
                if case.current_round <= MAX_ROUNDS {
                    "Within max rounds".into()
                } else {
                    "ROUND_LIMIT_EXCEEDED".into()
                },
                if packet.is_empty() || require_ascii(packet).is_empty() {
                    "ASCII packet".into()
                } else {
                    "NON_ASCII packet".into()
                },
            ],
            validation: vec![
                if !case.third_party_log.is_empty() {
                    "Third-party validation present".into()
                } else {
                    "PENDING third-party validation".into()
                },
                if case.status != SessionStatus::LOCKED || !case.third_party_log.is_empty() {
                    "Locked only after validation".into()
                } else {
                    "LOCKED without validation".into()
                },
            ],
        }
    }

    pub fn failures(&self) -> Vec<&str> {
        let all: Vec<&str> = self.pre_session.iter()
            .chain(self.phase_0.iter())
            .chain(self.contract.iter())
            .chain(self.truth.iter())
            .chain(self.limits.iter())
            .chain(self.validation.iter())
            .map(|s| s.as_str())
            .collect();
        all.iter()
            .filter(|item| item.starts_with("MISSING")
                || item.starts_with("REFRAME")
                || item.starts_with("ROUND_LIMIT")
                || item.starts_with("NON_ASCII")
                || item.starts_with("PENDING")
                || item.starts_with("LOCKED"))
            .copied()
            .collect()
    }
}

// ============================================================================
// Section limit checking
// ============================================================================

pub fn check_section_limits(packet: &PartnerPacket, limits: &SectionLimits) -> Vec<String> {
    let mut errors = Vec::new();

    // Check objections count
    if packet.objections.len() > limits.max_objections {
        errors.push(format!("OBJECTIONS exceeds max {} — tighten next round", limits.max_objections));
    }

    // Check winner framing sentence count
    let framing_sentences = packet.winner_framing.split(&['.', '!', '?'][..])
        .filter(|s| !s.trim().is_empty()).count();
    if framing_sentences > limits.max_winner_framing_sentences {
        errors.push(format!("WINNER_FRAMING has {} sentences, max {} — tighten next round",
            framing_sentences, limits.max_winner_framing_sentences));
    }

    errors
}

// ============================================================================
// require_ascii
// ============================================================================

pub fn require_ascii(text: &str) -> Vec<String> {
    if text.is_ascii() {
        Vec::new()
    } else {
        vec!["payload must be ASCII only".into()]
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_require_ascii() {
        assert!(require_ascii("hello world").is_empty());
        assert!(!require_ascii("héllo").is_empty());
    }

    #[test]
    fn test_item_agreement_valid() {
        let a = ItemAgreement {
            item: "test".into(), score: 85, status: SessionStatus::PROVISIONAL,
            flip_criteria: "when X".into(), ..Default::default()
        };
        assert!(a.validate().is_empty());
    }

    #[test]
    fn test_item_agreement_missing_flip() {
        let a = ItemAgreement {
            item: "test".into(), score: 50, status: SessionStatus::PROVISIONAL,
            flip_criteria: String::new(), disagreement: String::new(), revision: String::new(), third_party_status: "N/A".into(),
        };
        assert!(!a.validate().is_empty());
    }

    #[test]
    fn test_scoring_option_valid_winner() {
        let s = ScoringOption {
            name: "A".into(), clarity: 7, leverage: 8, risk: 5, winner: true,
            elimination_note: String::new(),
        };
        assert!(s.validate().is_empty());
        assert_eq!(s.total(), 20);
    }

    #[test]
    fn test_scoring_option_valid_non_winner_with_note() {
        let s = ScoringOption {
            name: "B".into(), clarity: 5, leverage: 4, risk: 3, winner: false,
            elimination_note: "Lower clarity and leverage.".into(),
        };
        assert!(s.validate().is_empty());
    }

    #[test]
    fn test_scoring_option_non_winner_missing_elimination_note() {
        let s = ScoringOption {
            name: "B".into(), clarity: 5, leverage: 4, risk: 3, winner: false,
            elimination_note: String::new(),
        };
        let errors = s.validate();
        assert!(errors.iter().any(|e| e.contains("elimination_note is required for non-winners")));
    }

    #[test]
    fn test_scoring_option_out_of_range() {
        let s = ScoringOption {
            name: "A".into(), clarity: 11, leverage: 5, risk: 3, winner: true,
            elimination_note: String::new(),
        };
        assert!(!s.validate().is_empty());
    }

    #[test]
    fn test_council_spawn_none_when_confident() {
        assert!(CouncilSpawn::maybe_spawn(true, 90, 3).is_none());
    }

    #[test]
    fn test_council_spawn_some_when_low_confidence() {
        let cs = CouncilSpawn::maybe_spawn(true, 60, 3).unwrap();
        assert_eq!(cs.feed_back_to_round, 4);
        assert_eq!(cs.composition.len(), 3);
    }

    #[test]
    fn test_verification_checklist_failures() {
        let case = DecisionCase::default(); // everything missing
        let cl = VerificationChecklist::run(&case, "");
        let failures = cl.failures();
        assert!(!failures.is_empty());
    }

    #[test]
    fn test_verification_checklist_constraint_diagnosis_tagged() {
        let mut case = DecisionCase::default();
        case.constraint_diagnoses.push(ConstraintDiagnosis {
            item: "i1".into(),
            symptom_altitude: ConstraintAltitude::TACTICAL,
            constraint_altitude: ConstraintAltitude::STRUCTURAL,
            diagnosis: "Fix structural first.".into(),
        });
        let cl = VerificationChecklist::run(&case, "");
        assert!(cl.contract.iter().any(|c| c.contains("Constraint altitude tagged")));
        assert!(!cl.contract.iter().any(|c| c.contains("MISSING constraint")));
    }

    #[test]
    fn test_verification_checklist_missing_constraint_diagnosis() {
        let case = DecisionCase::default();
        let cl = VerificationChecklist::run(&case, "");
        assert!(cl.contract.iter().any(|c| c.contains("MISSING constraint diagnosis")));
    }

    #[test]
    fn test_r5_provisional_lock_without_third_party_error() {
        let packet = PartnerPacket {
            item_agreements: vec![ItemAgreement {
                item: "decision-x".into(),
                score: 92,
                status: SessionStatus::PROVISIONAL_LOCK,
                third_party_status: String::new(), // missing!
                ..Default::default()
            }],
            winner_framing: String::new(),
            scoring_table: vec![
                ScoringOption {
                    name: "A".into(), clarity: 8, leverage: 9, risk: 7, winner: true,
                    elimination_note: String::new(),
                },
                ScoringOption {
                    name: "B".into(), clarity: 5, leverage: 4, risk: 3, winner: false,
                    elimination_note: "Lower scores.".into(),
                },
            ],
            objections: vec![],
            frameworks: vec![],
            convergence_plan: vec![],
            state_snapshot: StateSnapshot::new(Phase::Spec, MAX_ROUNDS, SessionStatus::PROVISIONAL_LOCK, "echo"),
            raw_payload: String::new(),
        };
        let errors = packet.validate();
        assert!(errors.iter().any(|e| e.contains("PROVISIONAL_LOCK at Round 5 without third-party")));
    }

    #[test]
    fn test_r5_provisional_lock_with_third_party_ok() {
        let packet = PartnerPacket {
            item_agreements: vec![ItemAgreement {
                item: "decision-x".into(),
                score: 92,
                status: SessionStatus::PROVISIONAL_LOCK,
                third_party_status: "CONFIRMED".into(),
                ..Default::default()
            }],
            winner_framing: String::new(),
            scoring_table: vec![
                ScoringOption {
                    name: "A".into(), clarity: 8, leverage: 9, risk: 7, winner: true,
                    elimination_note: String::new(),
                },
                ScoringOption {
                    name: "B".into(), clarity: 5, leverage: 4, risk: 3, winner: false,
                    elimination_note: "Lower scores.".into(),
                },
            ],
            objections: vec![],
            frameworks: vec![],
            convergence_plan: vec![],
            state_snapshot: StateSnapshot::new(Phase::Spec, MAX_ROUNDS, SessionStatus::PROVISIONAL_LOCK, "echo"),
            raw_payload: String::new(),
        };
        let errors = packet.validate();
        assert!(!errors.iter().any(|e| e.contains("PROVISIONAL_LOCK at Round 5 without third-party")));
    }

    // --- Section limits tests ---

    #[test]
    fn test_section_limits_objections_exceeded() {
        let packet = PartnerPacket {
            item_agreements: vec![],
            winner_framing: String::new(),
            scoring_table: vec![
                ScoringOption {
                    name: "A".into(), clarity: 8, leverage: 8, risk: 8, winner: true,
                    elimination_note: String::new(),
                },
                ScoringOption {
                    name: "B".into(), clarity: 1, leverage: 1, risk: 1, winner: false,
                    elimination_note: "Low.".into(),
                },
            ],
            objections: vec!["obj1".into(), "obj2".into(), "obj3".into(), "obj4".into(), "obj5".into(), "obj6".into()],
            frameworks: vec![],
            convergence_plan: vec![],
            state_snapshot: StateSnapshot::new(Phase::Spec, 1, SessionStatus::EXPLORING, "echo"),
            raw_payload: String::new(),
        };
        let limits = SectionLimits::default(); // max_objections = 5
        let errors = check_section_limits(&packet, &limits);
        assert!(errors.iter().any(|e| e.contains("OBJECTIONS exceeds max 5")));
    }

    #[test]
    fn test_section_limits_framing_exceeded() {
        let packet = PartnerPacket {
            item_agreements: vec![],
            winner_framing: "Sentence one. Sentence two. Sentence three. Sentence four. Sentence five.".into(),
            scoring_table: vec![
                ScoringOption {
                    name: "A".into(), clarity: 8, leverage: 8, risk: 8, winner: true,
                    elimination_note: String::new(),
                },
                ScoringOption {
                    name: "B".into(), clarity: 1, leverage: 1, risk: 1, winner: false,
                    elimination_note: "Low.".into(),
                },
            ],
            objections: vec![],
            frameworks: vec![],
            convergence_plan: vec![],
            state_snapshot: StateSnapshot::new(Phase::Spec, 1, SessionStatus::EXPLORING, "echo"),
            raw_payload: String::new(),
        };
        let limits = SectionLimits::default(); // max_winner_framing_sentences = 4
        let errors = check_section_limits(&packet, &limits);
        assert!(errors.iter().any(|e| e.contains("WINNER_FRAMING has 5 sentences")));
    }

    #[test]
    fn test_section_limits_within_bounds() {
        let packet = PartnerPacket {
            item_agreements: vec![],
            winner_framing: "One sentence. Two sentences.".into(),
            scoring_table: vec![
                ScoringOption {
                    name: "A".into(), clarity: 8, leverage: 8, risk: 8, winner: true,
                    elimination_note: String::new(),
                },
                ScoringOption {
                    name: "B".into(), clarity: 1, leverage: 1, risk: 1, winner: false,
                    elimination_note: "Low.".into(),
                },
            ],
            objections: vec!["obj1".into(), "obj2".into()],
            frameworks: vec![],
            convergence_plan: vec![],
            state_snapshot: StateSnapshot::new(Phase::Spec, 1, SessionStatus::EXPLORING, "echo"),
            raw_payload: String::new(),
        };
        let limits = SectionLimits::default();
        let errors = check_section_limits(&packet, &limits);
        assert!(errors.is_empty());
    }

    // --- ConvergenceClosure tests ---

    #[test]
    fn test_convergence_closure_from_case_no_audit() {
        let case = DecisionCase::new("dc-1", "Test", "finance", "alice");
        let closure = ConvergenceClosure::from_case(&case, "url1", "url2");
        assert_eq!(closure.status, SessionStatus::EXPLORING);
        assert!(closure.blind_spots_resolved.is_empty());
    }

    #[test]
    fn test_convergence_closure_with_accepted_risks() {
        let mut case = DecisionCase::new("dc-1", "Test", "finance", "alice");
        case.status = SessionStatus::CONVERGED;
        let mut audit = ClosureAudit::default();
        audit.vulnerabilities_accepted_risk.push("risk1".into());
        case.closure_audit = Some(audit);

        let closure = ConvergenceClosure::from_case(&case, "url1", "url2");
        // Should be REQUIRES_HUMAN_VERIFICATION because of accepted risks
        assert_eq!(closure.status, SessionStatus::REQUIRES_HUMAN_VERIFICATION);
        assert!(closure.vulnerabilities_accepted_risk.contains(&"risk1".to_string()));
    }

    #[test]
    fn test_convergence_closure_halt_not_overridden() {
        let mut case = DecisionCase::new("dc-1", "Test", "finance", "alice");
        case.status = SessionStatus::HALT;
        let mut audit = ClosureAudit::default();
        audit.vulnerabilities_accepted_risk.push("risk1".into());
        case.closure_audit = Some(audit);

        let closure = ConvergenceClosure::from_case(&case, "url1", "url2");
        // HALT status should not be overridden
        assert_eq!(closure.status, SessionStatus::HALT);
    }

    #[test]
    fn test_convergence_closure_with_resolved_blind_spots() {
        let mut case = DecisionCase::new("dc-1", "Test", "finance", "alice");
        case.status = SessionStatus::CONVERGED;
        let mut audit = ClosureAudit::default();
        audit.blind_spots_resolved.push("blind1".into());
        audit.blind_spots_accepted.push("blind2".into());
        audit.vulnerabilities_addressed.push("vuln1".into());
        case.closure_audit = Some(audit);

        let closure = ConvergenceClosure::from_case(&case, "url1", "url2");
        assert!(closure.blind_spots_resolved.contains(&"blind1".to_string()));
        assert!(closure.blind_spots_accepted.contains(&"blind2".to_string()));
        assert!(closure.vulnerabilities_addressed.contains(&"vuln1".to_string()));
        // No accepted risks → status stays as-is
        assert_eq!(closure.status, SessionStatus::CONVERGED);
    }

    #[test]
    fn test_convergence_closure_default() {
        let closure = ConvergenceClosure::default();
        assert_eq!(closure.status, SessionStatus::EXPLORING);
        assert!(closure.locked_decisions.is_empty());
        assert!(closure.session_urls.is_empty());
    }

    #[test]
    fn test_convergence_closure_urls() {
        let case = DecisionCase::new("dc-1", "Test", "finance", "alice");
        let closure = ConvergenceClosure::from_case(&case, "http://origin", "http://partner");
        assert_eq!(closure.session_urls.get("Origin").unwrap(), "http://origin");
        assert_eq!(closure.session_urls.get("Partner").unwrap(), "http://partner");
    }
}
