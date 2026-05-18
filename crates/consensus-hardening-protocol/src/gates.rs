//! Session gate logic for CHP.

use std::collections::HashMap;
use crate::models::*;

#[derive(Debug, Clone)]
pub struct GateEvaluation {
    pub results: HashMap<String, String>,
    pub verdict: Verdict,
}

pub fn evaluate_r0_gate(solvable: bool, scoped: bool, valid: bool, worth_it: bool) -> GateEvaluation {
    let mut results = HashMap::new();
    results.insert("Solvable".into(), if solvable { "PASS" } else { "FATAL" }.into());
    results.insert("Scoped".into(), if scoped { "PASS" } else { "FATAL" }.into());
    results.insert("Valid".into(), if valid { "PASS" } else { "FATAL" }.into());
    results.insert("Worth_it".into(), if worth_it { "PASS" } else { "FATAL" }.into());

    let all_pass = results.values().all(|v| v == "PASS");
    GateEvaluation {
        verdict: if all_pass { Verdict::PASS } else { Verdict::HALT },
        results,
    }
}

pub fn evaluate_phase_gate(round_number: u32, phase_one_status: &SessionStatus) -> Verdict {
    if round_number <= 2 {
        return Verdict::PASS;
    }
    match phase_one_status {
        SessionStatus::PROVISIONAL_LOCK | SessionStatus::LOCKED | SessionStatus::CONVERGED => Verdict::PASS,
        _ => Verdict::PHASE_GATE_FAIL,
    }
}

// ============================================================================
// PhaseGate
// ============================================================================

pub struct PhaseGate {
    pub max_spec_rounds: u32,
}

impl PhaseGate {
    pub fn new() -> Self { Self { max_spec_rounds: 2 } }

    pub fn check(&self, round_number: u32, phase: Phase, status: &SessionStatus) -> Result<Verdict, String> {
        if phase == Phase::Foundation {
            return Ok(Verdict::PASS);
        }
        if round_number <= self.max_spec_rounds {
            return Ok(Verdict::PASS);
        }
        // Round > 2: Phase 1 must be locked
        match status {
            SessionStatus::LOCKED | SessionStatus::CONVERGED => Ok(Verdict::PASS),
            SessionStatus::PROVISIONAL_LOCK => Ok(Verdict::PASS), // can proceed, third-party is Phase 2 entry condition
            _ => Ok(Verdict::PHASE_GATE_FAIL),
        }
    }
}

impl Default for PhaseGate {
    fn default() -> Self {
        Self::new()
    }
}

// ============================================================================
// ModelParityGate
// ============================================================================

pub struct ModelParityGate;

impl ModelParityGate {
    pub fn check(parity: &ModelParityCheck) -> Result<(), String> {
        match parity.delta {
            ModelParityDelta::NONE => Ok(()),
            ModelParityDelta::MINOR => Ok(()), // advisory only, logged
            ModelParityDelta::SIGNIFICANT => Err(format!(
                "[HALT] Model parity SIGNIFICANT: origin={}, partner={}. Full generation gap detected. Session cannot proceed.",
                parity.origin, parity.partner
            )),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_r0_gate_all_pass() {
        let g = evaluate_r0_gate(true, true, true, true);
        assert_eq!(g.verdict, Verdict::PASS);
    }

    #[test]
    fn test_r0_gate_fail() {
        let g = evaluate_r0_gate(true, false, true, true);
        assert_eq!(g.verdict, Verdict::HALT);
    }

    #[test]
    fn test_phase_gate_early_rounds() {
        assert_eq!(evaluate_phase_gate(1, &SessionStatus::EXPLORING), Verdict::PASS);
        assert_eq!(evaluate_phase_gate(2, &SessionStatus::EXPLORING), Verdict::PASS);
    }

    #[test]
    fn test_phase_gate_locked() {
        assert_eq!(evaluate_phase_gate(3, &SessionStatus::LOCKED), Verdict::PASS);
    }

    #[test]
    fn test_phase_gate_fail() {
        assert_eq!(evaluate_phase_gate(3, &SessionStatus::PROVISIONAL), Verdict::PHASE_GATE_FAIL);
    }

    // --- PhaseGate tests ---

    #[test]
    fn test_phase_gate_foundation_always_pass() {
        let gate = PhaseGate::new();
        let result = gate.check(3, Phase::Foundation, &SessionStatus::EXPLORING).unwrap();
        assert_eq!(result, Verdict::PASS);
    }

    #[test]
    fn test_phase_gate_early_rounds_pass() {
        let gate = PhaseGate::new();
        let result = gate.check(1, Phase::Spec, &SessionStatus::EXPLORING).unwrap();
        assert_eq!(result, Verdict::PASS);
        let result = gate.check(2, Phase::Spec, &SessionStatus::EXPLORING).unwrap();
        assert_eq!(result, Verdict::PASS);
    }

    #[test]
    fn test_phase_gate_r3_exploring_fails() {
        let gate = PhaseGate::new();
        let result = gate.check(3, Phase::Spec, &SessionStatus::EXPLORING).unwrap();
        assert_eq!(result, Verdict::PHASE_GATE_FAIL);
    }

    #[test]
    fn test_phase_gate_r3_locked_pass() {
        let gate = PhaseGate::new();
        let result = gate.check(3, Phase::Implementation, &SessionStatus::LOCKED).unwrap();
        assert_eq!(result, Verdict::PASS);
    }

    #[test]
    fn test_phase_gate_r3_converged_pass() {
        let gate = PhaseGate::new();
        let result = gate.check(3, Phase::Implementation, &SessionStatus::CONVERGED).unwrap();
        assert_eq!(result, Verdict::PASS);
    }

    #[test]
    fn test_phase_gate_r3_provisional_lock_pass() {
        let gate = PhaseGate::new();
        let result = gate.check(3, Phase::Implementation, &SessionStatus::PROVISIONAL_LOCK).unwrap();
        assert_eq!(result, Verdict::PASS);
    }

    #[test]
    fn test_phase_gate_r4_provisional_fails() {
        let gate = PhaseGate::new();
        let result = gate.check(4, Phase::Implementation, &SessionStatus::PROVISIONAL).unwrap();
        assert_eq!(result, Verdict::PHASE_GATE_FAIL);
    }

    #[test]
    fn test_phase_gate_r5_unresolved_fails() {
        let gate = PhaseGate::new();
        let result = gate.check(5, Phase::Implementation, &SessionStatus::UNRESOLVED).unwrap();
        assert_eq!(result, Verdict::PHASE_GATE_FAIL);
    }

    #[test]
    fn test_phase_gate_default() {
        let gate = PhaseGate::default();
        assert_eq!(gate.max_spec_rounds, 2);
    }

    #[test]
    fn test_phase_gate_custom_max_spec_rounds() {
        let gate = PhaseGate { max_spec_rounds: 3 };
        // Round 3 should now pass for spec phase
        let result = gate.check(3, Phase::Spec, &SessionStatus::EXPLORING).unwrap();
        assert_eq!(result, Verdict::PASS);
    }

    // --- ModelParityGate tests ---

    #[test]
    fn test_model_parity_gate_none_pass() {
        let parity = ModelParityCheck {
            origin: "model-a".into(),
            partner: "model-a".into(),
            delta: ModelParityDelta::NONE,
            advisory: None,
        };
        assert!(ModelParityGate::check(&parity).is_ok());
    }

    #[test]
    fn test_model_parity_gate_minor_pass() {
        let parity = ModelParityCheck {
            origin: "model-a".into(),
            partner: "model-b".into(),
            delta: ModelParityDelta::MINOR,
            advisory: Some("advisory".into()),
        };
        assert!(ModelParityGate::check(&parity).is_ok());
    }

    #[test]
    fn test_model_parity_gate_significant_halt() {
        let parity = ModelParityCheck {
            origin: "claude-opus-4".into(),
            partner: "claude-haiku-3".into(),
            delta: ModelParityDelta::SIGNIFICANT,
            advisory: None,
        };
        let result = ModelParityGate::check(&parity);
        assert!(result.is_err());
        let err_msg = result.unwrap_err();
        assert!(err_msg.contains("[HALT]"));
        assert!(err_msg.contains("claude-opus-4"));
        assert!(err_msg.contains("claude-haiku-3"));
    }
}
