//! Helpers for building and validating CHP dossiers.

use crate::models::Dossier;

pub fn validate_dossier(dossier: &Dossier) -> Vec<String> {
    dossier.validate()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_valid_dossier() {
        let d = Dossier {
            core_problem: "Real problem".into(),
            goal_state: vec!["g".into()],
            current_state: vec!["c".into()],
            constraints: vec!["x".into()],
            ..Default::default()
        };
        assert!(validate_dossier(&d).is_empty());
    }

    #[test]
    fn test_invalid_dossier() {
        let d = Dossier {
            core_problem: "".into(),
            ..Default::default()
        };
        assert!(!validate_dossier(&d).is_empty());
    }
}
