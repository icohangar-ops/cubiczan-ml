//! Round progression helpers for CHP sessions.

use crate::models::Phase;

pub fn next_round(phase: Phase, round_number: u32) -> (Phase, u32) {
    match phase {
        Phase::Foundation => (Phase::Spec, 1),
        Phase::Spec if round_number >= 2 => (Phase::Implementation, 3),
        _ => (phase, round_number + 1),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_foundation_to_spec() {
        assert_eq!(next_round(Phase::Foundation, 0), (Phase::Spec, 1));
    }

    #[test]
    fn test_spec_to_implementation() {
        assert_eq!(next_round(Phase::Spec, 2), (Phase::Implementation, 3));
    }

    #[test]
    fn test_spec_stays_spec() {
        assert_eq!(next_round(Phase::Spec, 1), (Phase::Spec, 2));
    }

    #[test]
    fn test_implementation_advances() {
        assert_eq!(next_round(Phase::Implementation, 3), (Phase::Implementation, 4));
    }
}
