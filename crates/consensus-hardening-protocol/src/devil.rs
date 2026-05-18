//! Devil's advocate helpers for CHP sessions.
use std::collections::HashMap;

use crate::models::*;

pub fn merge_structural_vulnerabilities(existing: &[String], new_items: &[String]) -> Vec<String> {
    let mut merged: Vec<String> = existing.to_vec();
    for item in new_items {
        if !merged.contains(item) {
            merged.push(item.clone());
        }
    }
    merged
}

pub fn build_phase0_devils_advocate(
    disclosure: &FoundationDisclosure,
    attack: &FoundationAttack,
) -> DevilsAdvocateRound {
    let mut vulnerabilities = vec![attack.vulnerability_strike.clone()];
    vulnerabilities.extend(attack.assumption_attacks.iter().take(2).cloned());

    DevilsAdvocateRound {
        phase: Phase::Foundation,
        round_number: 0,
        why_direction_wrong: attack.vulnerability_strike.clone(),
        what_not_seeing: disclosure
            .invalidation_conditions
            .first()
            .cloned()
            .unwrap_or_else(|| "The invalidation path is under-specified.".into()),
        false_consensus_risk: "Foundation agreement may reflect shared optimism unless the disclosed weak assumptions survive attack.".into(),
        structural_vulnerabilities: vulnerabilities.into_iter().take(3).collect(),
    }
}

pub fn build_round3_devils_advocate(case: &DecisionCase) -> DevilsAdvocateRound {
    DevilsAdvocateRound {
        phase: Phase::Implementation,
        round_number: 3,
        why_direction_wrong: "Implementation QA can drift from the locked spec if acceptance criteria are not explicit.".into(),
        what_not_seeing: "Operational handoffs, owner capacity, and evidence quality can fail below the visible decision layer.".into(),
        false_consensus_risk: "A clean spec lock can create premature confidence that implementation risk has been resolved.".into(),
        structural_vulnerabilities: case.structural_vulnerabilities.iter().take(3).cloned().collect(),
    }
}

pub fn build_constraint_diagnoses(case: &DecisionCase) -> Vec<ConstraintDiagnosis> {
    let Some(dossier) = &case.dossier else {
        return Vec::new();
    };
    let items: Vec<&String> = if !dossier.scope.is_empty() {
        dossier.scope.iter().collect()
    } else {
        vec![&case.title]
    };
    let constraint = dossier.constraints.first().cloned()
        .unwrap_or_else(|| "No governing constraint was supplied.".into());

    items.iter().take(5).map(|item| ConstraintDiagnosis {
        item: (**item).clone(),
        symptom_altitude: ConstraintAltitude::TACTICAL,
        constraint_altitude: ConstraintAltitude::STRUCTURAL,
        diagnosis: format!("Lower tactical fixes fail unless the structural constraint is handled first: {}", constraint),
    }).collect()
}

pub fn build_state_snapshot(
    case: &DecisionCase,
    payload_echo: &str,
    phase: Option<Phase>,
    round_number: Option<u32>,
    status: Option<&SessionStatus>,
) -> StateSnapshot {
    let current_status = status.unwrap_or(&case.status);
    let provisional = if *current_status == SessionStatus::PROVISIONAL {
        vec![case.title.clone()]
    } else {
        Vec::new()
    };
    let provisional_lock = if *current_status == SessionStatus::PROVISIONAL_LOCK {
        vec![case.title.clone()]
    } else {
        Vec::new()
    };
    let pending = if case.third_party_log.is_empty() {
        provisional_lock.clone()
    } else {
        Vec::new()
    };
    let mut blind_spots = HashMap::new();
    blind_spots.insert("Origin".into(), case.blind_spots.clone());
    blind_spots.insert("Partner".into(), Vec::new());

    // Build constraint diagnosis map from case diagnoses
    let constraint_diagnosis: HashMap<String, ConstraintDiagnosis> = case.constraint_diagnoses.iter()
        .map(|cd| (cd.item.clone(), cd.clone()))
        .collect();

    StateSnapshot {
        phase: phase.unwrap_or(case.current_phase),
        round_number: format!("{}/5", round_number.unwrap_or(case.current_round)),
        status: current_status.clone(),
        payload_echo: payload_echo.into(),
        foundation_score: case.foundation_score,
        locked: case.locked_decisions.clone(),
        provisional,
        provisional_lock,
        flip_active: case.flip_criteria.clone(),
        blind_spots_acknowledged: blind_spots,
        structural_vulnerabilities: case.structural_vulnerabilities.clone(),
        third_party_pending: pending,
        constraint_diagnosis,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_merge_vulnerabilities_no_duplicates() {
        let existing = vec!["a".into(), "b".into()];
        let new = vec!["b".into(), "c".into()];
        let merged = merge_structural_vulnerabilities(&existing, &new);
        assert_eq!(merged, vec!["a", "b", "c"]);
    }

    #[test]
    fn test_build_phase0_devils_advocate() {
        let d = FoundationDisclosure {
            weakest_assumptions: vec!["wa1".into()],
            invalidation_conditions: vec!["ic1".into()],
            key_vulnerability: "kv".into(),
        };
        let a = FoundationAttack {
            assumption_attacks: vec!["aa1".into(), "aa2".into()],
            vulnerability_strike: "vs".into(),
            foundation_score: 80,
            ..Default::default()
        };
        let da = build_phase0_devils_advocate(&d, &a);
        assert!(da.validate().is_empty());
        assert_eq!(da.phase, Phase::Foundation);
        assert_eq!(da.round_number, 0);
    }

    #[test]
    fn test_build_round3() {
        let case = DecisionCase {
            structural_vulnerabilities: vec!["v1".into(), "v2".into()],
            ..Default::default()
        };
        let da = build_round3_devils_advocate(&case);
        assert_eq!(da.phase, Phase::Implementation);
        assert_eq!(da.round_number, 3);
        assert_eq!(da.structural_vulnerabilities.len(), 2);
    }

    #[test]
    fn test_build_constraint_diagnoses_empty() {
        let case = DecisionCase::default();
        assert!(build_constraint_diagnoses(&case).is_empty());
    }

    #[test]
    fn test_build_constraint_diagnoses_with_dossier() {
        let mut case = DecisionCase::new("dc-1", "Test", "finance", "alice");
        case.dossier = Some(Dossier {
            scope: vec!["item1".into(), "item2".into()],
            constraints: vec!["constraint_x".into()],
            ..Default::default()
        });
        let diagnoses = build_constraint_diagnoses(&case);
        assert_eq!(diagnoses.len(), 2);
        assert_eq!(diagnoses[0].symptom_altitude, ConstraintAltitude::TACTICAL);
        assert_eq!(diagnoses[0].constraint_altitude, ConstraintAltitude::STRUCTURAL);
        assert!(diagnoses[0].diagnosis.contains("constraint_x"));
    }

    #[test]
    fn test_build_constraint_diagnoses_no_scope() {
        let mut case = DecisionCase::new("dc-1", "Test Title", "finance", "alice");
        case.dossier = Some(Dossier {
            core_problem: "Problem".into(),
            goal_state: vec!["g".into()],
            current_state: vec!["c".into()],
            constraints: vec!["con".into()],
            ..Default::default()
        });
        let diagnoses = build_constraint_diagnoses(&case);
        assert_eq!(diagnoses.len(), 1);
        assert_eq!(diagnoses[0].item, "Test Title");
    }

    #[test]
    fn test_build_constraint_diagnoses_no_constraint() {
        let mut case = DecisionCase::new("dc-1", "Test", "finance", "alice");
        case.dossier = Some(Dossier {
            scope: vec!["item1".into()],
            constraints: vec![],
            ..Default::default()
        });
        let diagnoses = build_constraint_diagnoses(&case);
        assert_eq!(diagnoses.len(), 1);
        assert!(diagnoses[0].diagnosis.contains("No governing constraint"));
    }

    #[test]
    fn test_build_state_snapshot_includes_constraint_diagnosis() {
        let mut case = DecisionCase::new("dc-1", "Test", "finance", "alice");
        case.constraint_diagnoses.push(ConstraintDiagnosis {
            item: "scope1".into(),
            symptom_altitude: ConstraintAltitude::TACTICAL,
            constraint_altitude: ConstraintAltitude::STRUCTURAL,
            diagnosis: "Fix structural first.".into(),
        });
        let snapshot = build_state_snapshot(&case, "echo", Some(Phase::Spec), Some(1), None);
        assert!(snapshot.constraint_diagnosis.contains_key("scope1"));
    }
}
