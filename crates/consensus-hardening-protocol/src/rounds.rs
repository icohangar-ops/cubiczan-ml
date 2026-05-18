//! Round progression helpers for CHP sessions.

use crate::gates::PhaseGate;
use crate::models::*;

pub const MAX_ROUNDS: u32 = 5;

pub fn next_round(phase: Phase, round_number: u32, status: &SessionStatus) -> Result<(Phase, u32), Verdict> {
    let gate = PhaseGate::new();

    // Check max rounds
    if round_number >= MAX_ROUNDS {
        return Err(Verdict::HALT);
    }

    // Phase gate: Round > 2 requires Phase 1 lock
    let gate_result = gate.check(round_number, phase, status).unwrap_or(Verdict::PHASE_GATE_FAIL);
    if gate_result == Verdict::PHASE_GATE_FAIL {
        return Err(Verdict::PHASE_GATE_FAIL);
    }

    match phase {
        Phase::Foundation => Ok((Phase::Spec, 1)),
        Phase::Spec if round_number >= 2 => Ok((Phase::Implementation, 3)),
        _ => Ok((phase, round_number + 1)),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_foundation_to_spec() {
        assert_eq!(
            next_round(Phase::Foundation, 0, &SessionStatus::EXPLORING).unwrap(),
            (Phase::Spec, 1)
        );
    }

    #[test]
    fn test_spec_to_implementation() {
        assert_eq!(
            next_round(Phase::Spec, 2, &SessionStatus::LOCKED).unwrap(),
            (Phase::Implementation, 3)
        );
    }

    #[test]
    fn test_spec_stays_spec() {
        assert_eq!(
            next_round(Phase::Spec, 1, &SessionStatus::EXPLORING).unwrap(),
            (Phase::Spec, 2)
        );
    }

    #[test]
    fn test_implementation_advances() {
        assert_eq!(
            next_round(Phase::Implementation, 3, &SessionStatus::LOCKED).unwrap(),
            (Phase::Implementation, 4)
        );
    }

    #[test]
    fn test_max_rounds_halt() {
        assert_eq!(
            next_round(Phase::Implementation, 5, &SessionStatus::LOCKED),
            Err(Verdict::HALT)
        );
    }

    #[test]
    fn test_phase_gate_fail_r3_exploring() {
        assert_eq!(
            next_round(Phase::Spec, 3, &SessionStatus::EXPLORING),
            Err(Verdict::PHASE_GATE_FAIL)
        );
    }

    #[test]
    fn test_phase_gate_pass_r3_locked() {
        // At round 3 in Spec, phase gate passes (LOCKED), and Spec>=2 transitions to Implementation
        assert_eq!(
            next_round(Phase::Spec, 3, &SessionStatus::LOCKED).unwrap(),
            (Phase::Implementation, 3)
        );
    }

    #[test]
    fn test_phase_gate_pass_r3_provisional_lock() {
        let result = next_round(Phase::Implementation, 3, &SessionStatus::PROVISIONAL_LOCK);
        assert!(result.is_ok());
    }

    #[test]
    fn test_phase_gate_fail_r4_provisional() {
        assert_eq!(
            next_round(Phase::Implementation, 4, &SessionStatus::PROVISIONAL),
            Err(Verdict::PHASE_GATE_FAIL)
        );
    }

    #[test]
    fn test_spec_r2_locked_to_implementation() {
        assert_eq!(
            next_round(Phase::Spec, 2, &SessionStatus::LOCKED).unwrap(),
            (Phase::Implementation, 3)
        );
    }

    #[test]
    fn test_implementation_r4_converged() {
        assert_eq!(
            next_round(Phase::Implementation, 4, &SessionStatus::CONVERGED).unwrap(),
            (Phase::Implementation, 5)
        );
    }

    #[test]
    fn test_spec_r1_exploring_pass() {
        assert_eq!(
            next_round(Phase::Spec, 1, &SessionStatus::EXPLORING).unwrap(),
            (Phase::Spec, 2)
        );
    }

    #[test]
    fn test_foundation_r0_unresolved_pass() {
        // Foundation always passes regardless of status
        assert_eq!(
            next_round(Phase::Foundation, 0, &SessionStatus::UNRESOLVED).unwrap(),
            (Phase::Spec, 1)
        );
    }
}
