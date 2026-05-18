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
}
