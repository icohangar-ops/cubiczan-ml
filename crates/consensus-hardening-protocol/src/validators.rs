//! Third-party validation helpers for CHP lock progression.

use crate::models::*;

pub fn apply_third_party_validation(
    case: &mut DecisionCase,
    validation: ThirdPartyValidation,
) -> Result<SessionStatus, String> {
    if case.status != SessionStatus::PROVISIONAL_LOCK {
        return Err("third-party validation requires PROVISIONAL_LOCK status".into());
    }
    case.third_party_log.push(validation.clone());
    match &validation.result {
        ValidationResult::CONFIRM => {
            case.status = SessionStatus::LOCKED;
            if !case.locked_decisions.contains(&validation.item) {
                case.locked_decisions.push(validation.item.clone());
            }
            Ok(SessionStatus::LOCKED)
        }
        ValidationResult::REJECT => {
            case.status = SessionStatus::EXPLORING;
            case.flip_criteria.push(format!("Validation rejected: {}", validation.item));
            Ok(SessionStatus::EXPLORING)
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_confirm_locks() {
        let mut case = DecisionCase::new("dc-1", "Test", "finance", "alice");
        case.status = SessionStatus::PROVISIONAL_LOCK;
        let v = ThirdPartyValidation {
            validator: "model-c".into(),
            item: "decision-x".into(),
            challenge: "ch".into(),
            result: ValidationResult::CONFIRM,
            rationale: "r".into(),
        };
        let status = apply_third_party_validation(&mut case, v).unwrap();
        assert_eq!(status, SessionStatus::LOCKED);
        assert!(case.locked_decisions.contains(&"decision-x".to_string()));
    }

    #[test]
    fn test_reject_exploring() {
        let mut case = DecisionCase::new("dc-1", "Test", "finance", "alice");
        case.status = SessionStatus::PROVISIONAL_LOCK;
        let v = ThirdPartyValidation {
            validator: "model-c".into(),
            item: "decision-x".into(),
            challenge: "ch".into(),
            result: ValidationResult::REJECT,
            rationale: "r".into(),
        };
        let status = apply_third_party_validation(&mut case, v).unwrap();
        assert_eq!(status, SessionStatus::EXPLORING);
        assert!(!case.locked_decisions.contains(&"decision-x".to_string()));
    }

    #[test]
    fn test_wrong_status() {
        let mut case = DecisionCase::new("dc-1", "Test", "finance", "alice");
        case.status = SessionStatus::EXPLORING;
        let v = ThirdPartyValidation {
            validator: "m".into(), item: "i".into(), challenge: "c".into(),
            result: ValidationResult::CONFIRM, rationale: "r".into(),
        };
        assert!(apply_third_party_validation(&mut case, v).is_err());
    }
}
