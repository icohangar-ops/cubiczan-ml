//! Foundation-stage helpers for CHP.

use crate::models::*;

pub fn foundation_verdict(attack: &FoundationAttack) -> Verdict {
    if attack.foundation_score >= 70 {
        Verdict::PASS
    } else {
        Verdict::REFRAME
    }
}

pub fn validate_foundation_pair(disclosure: &FoundationDisclosure, attack: &FoundationAttack) -> Vec<String> {
    let mut errors = disclosure.validate();
    errors.extend(attack.validate());
    if !disclosure.weakest_assumptions.is_empty() && !attack.assumption_attacks.is_empty() {
        let min_required = 3.min(disclosure.weakest_assumptions.len());
        if attack.assumption_attacks.len() < min_required {
            errors.push("attack must address each disclosed weak assumption".into());
        }
    }
    errors
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_valid_disclosure() -> FoundationDisclosure {
        FoundationDisclosure {
            weakest_assumptions: vec!["a1".into(), "a2".into()],
            invalidation_conditions: vec!["i1".into()],
            key_vulnerability: "kv".into(),
        }
    }

    fn make_valid_attack(score: i32) -> FoundationAttack {
        FoundationAttack {
            assumption_attacks: vec!["aa1".into(), "aa2".into()],
            vulnerability_strike: "vs".into(),
            foundation_score: score,
            ..Default::default()
        }
    }

    #[test]
    fn test_foundation_verdict_pass() {
        let attack = make_valid_attack(80);
        assert_eq!(foundation_verdict(&attack), Verdict::PASS);
    }

    #[test]
    fn test_foundation_verdict_reframe() {
        let attack = make_valid_attack(50);
        assert_eq!(foundation_verdict(&attack), Verdict::REFRAME);
    }

    #[test]
    fn test_validate_foundation_pair_ok() {
        let d = make_valid_disclosure();
        let a = make_valid_attack(80);
        assert!(validate_foundation_pair(&d, &a).is_empty());
    }

    #[test]
    fn test_validate_foundation_pair_too_few_attacks() {
        let d = make_valid_disclosure();
        let mut a = make_valid_attack(80);
        a.assumption_attacks = vec!["one".into()]; // need >= 2
        let errors = validate_foundation_pair(&d, &a);
        assert!(errors.iter().any(|e| e.contains("attack must address")));
    }
}
