//! High-level CHP orchestration for finance decision sessions.

use crate::context::ContextEngine;
use crate::devil::*;
use crate::models::*;
use crate::foundation::*;
use crate::gates::*;
use crate::parity::*;
use crate::payloads::*;
use crate::registry::DecisionRegistry;
use crate::rounds::MAX_ROUNDS;
use crate::validators::*;

pub struct CHPReport {
    pub case: DecisionCase,
    pub foundation_disclosure: FoundationDisclosure,
    pub foundation_attack: FoundationAttack,
    pub r0_verdict: Verdict,
    pub foundation_verdict: Verdict,
    pub initial_packet: String,
}

impl CHPReport {
    pub fn render(&self) -> String {
        let mut lines = vec![
            "# CHP Session".into(),
            format!("Decision: {}", self.case.title),
            format!("Status: {}", format!("{:?}", self.case.status)),
            String::new(),
            "## Context Check".into(),
        ];
        if let Some(ref cc) = self.case.context_check {
            lines.push(format!("- memory_tools: {}", cc.memory_tools));
            lines.push(format!("- assessment: {}", cc.assessment));
            lines.push(format!("- action: {}", cc.action));
            lines.push(format!("- prior_sessions_count: {}", cc.prior_sessions_count));
        }
        lines.push(String::new());
        lines.push("## Model Parity".into());
        if let Some(ref mp) = self.case.model_parity {
            lines.push(format!("- origin: {}", mp.origin));
            lines.push(format!("- partner: {}", mp.partner));
            lines.push(format!("- delta: {:?}", mp.delta));
            if let Some(ref adv) = mp.advisory {
                lines.push(format!("- advisory: {}", adv));
            }
        }
        lines.extend(vec![
            String::new(),
            "## R0 Gate".into(),
            format!("- verdict: {:?}", self.r0_verdict),
            String::new(),
            "## Foundation".into(),
            format!("- weakest_assumptions: {}", self.foundation_disclosure.weakest_assumptions.len()),
            format!("- foundation_score: {}", self.foundation_attack.foundation_score),
            format!("- verdict: {:?}", self.foundation_verdict),
            String::new(),
            "## Initial Packet".into(),
        ]);
        if !self.initial_packet.is_empty() {
            lines.push(self.initial_packet.clone());
        }
        lines.join("\n")
    }
}

pub struct CHPOrchestrator {
    pub registry: DecisionRegistry,
    pub context: ContextEngine,
}

impl CHPOrchestrator {
    pub fn new() -> Self {
        Self {
            registry: DecisionRegistry::new(),
            context: ContextEngine::new(),
        }
    }

    pub fn with_registry(mut self, registry: DecisionRegistry) -> Self {
        self.registry = registry;
        self
    }

    pub fn with_context(mut self, context: ContextEngine) -> Self {
        self.context = context;
        self
    }

    pub fn run_initial_session(
        &mut self,
        case: &mut DecisionCase,
        disclosure: &FoundationDisclosure,
        attack: &FoundationAttack,
    ) -> Result<CHPReport, String> {
        // Context check
        let context_check = self._context_check(case);
        if context_check.action == "AUTO_POPULATE" {
            if let Some(ref mut dossier) = case.dossier {
                let mut prior = dossier.prior_decisions.clone();
                for lock in &context_check.related_locks {
                    if !prior.contains(lock) {
                        prior.push(lock.clone());
                    }
                }
                dossier.prior_decisions = prior;
            }
        }
        case.context_check = Some(context_check);

        // Model parity
        case.model_parity = Some(assess_model_parity(&case.origin_model, &case.partner_model));

        // ModelParityGate check
        if let Some(ref mp) = case.model_parity {
            if let Err(msg) = ModelParityGate::check(mp) {
                case.status = SessionStatus::HALT;
                case.guard_triggers.push(GuardTrigger {
                    guard_name: "ModelParityGate".into(),
                    item: case.title.clone(),
                    reason: msg.clone(),
                    severity: "CRITICAL".into(),
                });
            }
        }

        // R0 gate
        let scoped = case.dossier.as_ref().map_or(false, |d| !d.scope.is_empty());
        let valid = case.dossier.as_ref().map_or(false, |d| !d.current_state.is_empty());
        let worth_it = case.high_stakes || case.domain == "capital_allocation" || case.domain == "board_decision";
        let r0 = evaluate_r0_gate(true, scoped, valid, worth_it);

        // Foundation validation
        let foundation_errors = validate_foundation_pair(disclosure, attack);
        if !foundation_errors.is_empty() {
            return Err(foundation_errors.join("; "));
        }

        let f_verdict = foundation_verdict(attack);
        case.foundation_score = Some(attack.foundation_score);
        if let Some(ref mut dossier) = case.dossier {
            dossier.foundation_score = Some(attack.foundation_score);
            case.structural_vulnerabilities = dossier.structural_vulnerabilities.clone();
            case.constraint_diagnoses = build_constraint_diagnoses(case);
        }

        // Section limits (use defaults)
        case.section_limits = SectionLimits::default();

        // Status determination (HALT from guard triggers takes precedence)
        if case.status == SessionStatus::HALT {
            // Already halted by ModelParityGate or similar — don't override
        } else if case.context_check.as_ref().map_or(false, |cc| cc.action == "HALT_DUPLICATE") {
            case.status = SessionStatus::HALT;
        } else if case.model_parity.as_ref().map_or(false, |mp| mp.delta == ModelParityDelta::SIGNIFICANT) {
            case.status = SessionStatus::HALT;
        } else if r0.verdict == Verdict::HALT {
            case.status = SessionStatus::HALT;
        } else if f_verdict == Verdict::REFRAME {
            case.status = SessionStatus::REFRAME_REQUIRED;
        } else {
            case.status = SessionStatus::EXPLORING;
        }

        // Build packet if not halted
        let mut packet = String::new();
        if case.status != SessionStatus::HALT && case.status != SessionStatus::REFRAME_REQUIRED {
            let devil_round = build_phase0_devils_advocate(disclosure, attack);
            let devil_errors = devil_round.validate();
            if !devil_errors.is_empty() {
                return Err(devil_errors.join("; "));
            }
            case.devil_advocate_rounds.push(devil_round.clone());
            case.structural_vulnerabilities = merge_structural_vulnerabilities(
                &case.structural_vulnerabilities,
                &devil_round.structural_vulnerabilities,
            );

            packet = self._build_initial_packet(case, disclosure, attack, &r0.verdict, &f_verdict);
            let payload_id = extract_payload_id(&packet).unwrap_or_else(|| "UNKNOWN".into());
            case.state_snapshots.push(build_state_snapshot(
                case,
                &format!("[RX] [{}] ORIGIN_SENT", payload_id),
                Some(Phase::Foundation),
                Some(0),
                None,
            ));
        }

        if case.context_check.as_ref().map_or(true, |cc| cc.action != "HALT_DUPLICATE") {
            self.registry.add(case.clone());
        }

        Ok(CHPReport {
            case: case.clone(),
            foundation_disclosure: disclosure.clone(),
            foundation_attack: attack.clone(),
            r0_verdict: r0.verdict,
            foundation_verdict: f_verdict,
            initial_packet: packet,
        })
    }

    pub fn receive_partner_packet(
        &mut self,
        decision_id: &str,
        partner_packet: &str,
        phase: Phase,
        round_number: u32,
        payload_echo: &str,
        snapshot_status: &str,
    ) -> Result<(), String> {
        let case = self.registry.get_mut(decision_id)
            .ok_or_else(|| format!("Unknown decision_id: {}", decision_id))?;

        if !validate_payload_envelope(partner_packet) {
            return Err("partner packet failed payload envelope validation".into());
        }
        let payload_id = extract_payload_id(partner_packet)
            .ok_or_else(|| String::from("partner packet is missing a payload id"))?;

        // PayloadValidator::gate check
        if let Err(e) = PayloadValidator::gate(payload_echo, "RX", &payload_id) {
            return Err(e);
        }

        let mut incoming_status = match snapshot_status {
            "EXPLORING" => SessionStatus::EXPLORING,
            "PROVISIONAL" => SessionStatus::PROVISIONAL,
            "PROVISIONAL_LOCK" => SessionStatus::PROVISIONAL_LOCK,
            "LOCKED" => SessionStatus::LOCKED,
            "CONVERGED" => SessionStatus::CONVERGED,
            _ => SessionStatus::EXPLORING,
        };
        if round_number >= MAX_ROUNDS && incoming_status == SessionStatus::PROVISIONAL {
            incoming_status = SessionStatus::UNRESOLVED;
        }

        let phase_gate = evaluate_phase_gate(round_number, &case.status);
        if phase_gate == Verdict::PHASE_GATE_FAIL {
            case.status = SessionStatus::HALT;
            case.guard_triggers.push(GuardTrigger {
                guard_name: "PhaseGate".into(),
                item: decision_id.into(),
                reason: "cannot enter implementation before Phase 1 reaches PROVISIONAL_LOCK or LOCKED".into(),
                severity: "CRITICAL".into(),
            });
            return Err("cannot enter implementation before Phase 1 reaches PROVISIONAL_LOCK or LOCKED".into());
        }

        if phase == Phase::Implementation && round_number == 3 {
            let devil_round = build_round3_devils_advocate(case);
            let devil_errors = devil_round.validate();
            if !devil_errors.is_empty() {
                return Err(devil_errors.join("; "));
            }
            case.devil_advocate_rounds.push(devil_round.clone());
            case.structural_vulnerabilities = merge_structural_vulnerabilities(
                &case.structural_vulnerabilities,
                &devil_round.structural_vulnerabilities,
            );
        }

        let snapshot = build_state_snapshot(
            case,
            payload_echo,
            Some(phase),
            Some(round_number),
            Some(&incoming_status),
        );

        let record = RoundRecord {
            decision_id: decision_id.into(),
            phase,
            round_number,
            payload_id,
            origin_packet: String::new(),
            partner_packet: partner_packet.into(),
            payload_echo_confirmed: true,
            state_snapshot: serde_json::to_value(&snapshot).unwrap_or(serde_json::Value::Null),
        };

        case.add_round(record);
        case.status = incoming_status;
        case.state_snapshots.push(snapshot);
        Ok(())
    }

    pub fn apply_validation(
        &mut self,
        decision_id: &str,
        validation: ThirdPartyValidation,
    ) -> Result<(), String> {
        let case = self.registry.get_mut(decision_id)
            .ok_or_else(|| format!("Unknown decision_id: {}", decision_id))?;
        apply_third_party_validation(case, validation)?;
        Ok(())
    }

    fn _context_check(&self, case: &DecisionCase) -> ContextCheck {
        let query = case.dossier.as_ref()
            .map(|d| d.core_problem.as_str())
            .unwrap_or("");
        let related = if !query.is_empty() {
            self.registry.find_related(query)
        } else {
            Vec::new()
        };
        let exact: Vec<_> = related.iter().filter(|r| r.title == case.title).collect();
        let (assessment, action) = if !exact.is_empty() {
            ("DUPLICATE", "HALT_DUPLICATE")
        } else if !related.is_empty() {
            ("RELATED", "AUTO_POPULATE")
        } else {
            ("SPARSE", "PROCEED")
        };
        let prior_lock_versions = if !related.is_empty() { vec!["chp-v1".into()] } else { Vec::new() };
        ContextCheck {
            memory_tools: "AVAILABLE".into(),
            prior_sessions_count: related.len() as u32,
            prior_lock_versions,
            legacy_warning: false,
            related_locks: related.iter().filter(|r| r.status == SessionStatus::LOCKED).map(|r| r.title.clone()).collect(),
            assessment: assessment.into(),
            action: action.into(),
        }
    }

    fn _build_initial_packet(
        &self,
        case: &DecisionCase,
        disclosure: &FoundationDisclosure,
        attack: &FoundationAttack,
        r0_verdict: &Verdict,
        f_verdict: &Verdict,
    ) -> String {
        let mut body = Vec::new();
        body.push("1. CORE_PROBLEM_STATEMENT".into());
        body.push(case.dossier.as_ref().map(|d| d.core_problem.as_str()).unwrap_or("UNKNOWN").into());
        body.push(String::new());
        body.push("2. PARTNER_SYSTEM_PACKET".into());
        body.push(format!("From: {}", case.origin_system));
        body.push(format!("To: {}", case.partner_system));
        body.push("Subject: CHP - Phase 0 Round 0".into());
        body.push(String::new());
        body.push("STYLE_GUIDE:".into());
        body.push("- Tone: Calm, spec-like".into());
        body.push("- Framing: does not X unless Y".into());
        body.push("- ASCII only".into());
        body.push(String::new());
        body.push(format!("R0_GATE:"));
        body.push(format!("- verdict: {:?}", r0_verdict));
        body.push(String::new());
        body.push("FOUNDATION_DISCLOSURE:".into());
        for (i, item) in disclosure.weakest_assumptions.iter().enumerate() {
            body.push(format!("{}. {}", i + 1, item));
        }
        body.push("INVALIDATION_CONDITIONS:".into());
        for (i, item) in disclosure.invalidation_conditions.iter().enumerate() {
            body.push(format!("{}. {}", i + 1, item));
        }
        body.push(format!("KEY_VULNERABILITY: {}", disclosure.key_vulnerability));
        body.push(String::new());
        body.push("FOUNDATION_ATTACK:".into());
        body.push(format!("- score: {}", attack.foundation_score));
        body.push(format!("- verdict: {:?}", f_verdict));
        body.push(format!("- summary: {}", attack.attack_summary));
        body.push(String::new());

        let env = build_payload_envelope(&body.join("\n"), "RX", None);
        env.render()
    }
}

impl Default for CHPOrchestrator {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_valid_disclosure() -> FoundationDisclosure {
        FoundationDisclosure {
            weakest_assumptions: vec!["Market grows 5% annually".into(), "Supply chains remain stable".into()],
            invalidation_conditions: vec!["Global recession".into()],
            key_vulnerability: "Tariff escalation".into(),
        }
    }

    fn make_valid_attack(score: i32) -> FoundationAttack {
        FoundationAttack {
            assumption_attacks: vec!["Market may contract".into(), "Supply chains have fragility".into()],
            vulnerability_strike: "Tariff risk is underpriced".into(),
            foundation_score: score,
            attack_summary: "Two of three assumptions directly attacked.".into(),
            ..Default::default()
        }
    }

    fn make_test_case() -> DecisionCase {
        let mut case = DecisionCase::new("dc-001", "Expand into APAC", "capital_allocation", "CFO");
        case.high_stakes = true;
        case.dossier = Some(Dossier {
            core_problem: "Determine optimal capital allocation for APAC expansion".into(),
            goal_state: vec!["Revenue target achieved".into()],
            current_state: vec!["Current allocation is US-heavy".into()],
            constraints: vec!["Budget ceiling $50M".into()],
            scope: vec!["APAC markets".into(), "Entry strategy".into()],
            ..Default::default()
        });
        case
    }

    #[test]
    fn test_run_initial_session_success() {
        let mut orch = CHPOrchestrator::new();
        let mut case = make_test_case();
        let disclosure = make_valid_disclosure();
        let attack = make_valid_attack(85);

        let _report = orch.run_initial_session(&mut case, &disclosure, &attack).unwrap();
        assert_eq!(case.status, SessionStatus::EXPLORING);
        assert!(case.foundation_score.is_some());
        assert!(!_report.initial_packet.is_empty());
        assert!(_report.render().contains("# CHP Session"));
    }

    #[test]
    fn test_run_initial_session_halt_low_foundation() {
        let mut orch = CHPOrchestrator::new();
        let mut case = make_test_case();
        let disclosure = make_valid_disclosure();
        let attack = make_valid_attack(40);

        let _report = orch.run_initial_session(&mut case, &disclosure, &attack).unwrap();
        assert_eq!(case.status, SessionStatus::REFRAME_REQUIRED);
    }

    #[test]
    fn test_context_check_sparse() {
        let orch = CHPOrchestrator::new();
        let case = make_test_case();
        let cc = orch._context_check(&case);
        assert_eq!(cc.assessment, "SPARSE");
        assert_eq!(cc.action, "PROCEED");
    }

    #[test]
    fn test_run_initial_session_constraint_diagnoses() {
        let mut orch = CHPOrchestrator::new();
        let mut case = make_test_case();
        let disclosure = make_valid_disclosure();
        let attack = make_valid_attack(85);

        orch.run_initial_session(&mut case, &disclosure, &attack).unwrap();
        // Should have constraint diagnoses
        assert!(!case.constraint_diagnoses.is_empty());
        assert_eq!(case.constraint_diagnoses.len(), 2); // 2 scope items
    }

    #[test]
    fn test_run_initial_session_section_limits_set() {
        let mut orch = CHPOrchestrator::new();
        let mut case = make_test_case();
        let disclosure = make_valid_disclosure();
        let attack = make_valid_attack(85);

        orch.run_initial_session(&mut case, &disclosure, &attack).unwrap();
        assert_eq!(case.section_limits.max_agreement_lines, 3);
        assert_eq!(case.section_limits.max_objections, 5);
    }

    #[test]
    fn test_run_initial_session_model_parity_gate_halt() {
        let mut orch = CHPOrchestrator::new();
        let mut case = make_test_case();
        case.origin_model = "claude-opus-4".into();
        case.partner_model = "claude-3-haiku".into(); // SIGNIFICANT gap
        let disclosure = make_valid_disclosure();
        let attack = make_valid_attack(85);

        let _report = orch.run_initial_session(&mut case, &disclosure, &attack).unwrap();
        assert_eq!(case.status, SessionStatus::HALT);
        assert!(!case.guard_triggers.is_empty());
        assert!(case.guard_triggers.iter().any(|g| g.guard_name == "ModelParityGate"));
    }

    #[test]
    fn test_receive_partner_packet_phase_gate_fail_adds_trigger() {
        let mut orch = CHPOrchestrator::new();
        let mut case = make_test_case();
        let disclosure = make_valid_disclosure();
        let attack = make_valid_attack(85);
        orch.run_initial_session(&mut case, &disclosure, &attack).unwrap();

        // Try to receive at round 3 with EXPLORING status → should fail
        let packet = build_payload_envelope("body", "RX", Some("TEST01")).render();
        let result = orch.receive_partner_packet(
            "dc-001", &packet, Phase::Implementation, 3,
            "[RX] [TEST01] CONFIRMED", "EXPLORING",
        );
        assert!(result.is_err());
        let case = orch.registry.get("dc-001").unwrap();
        assert!(case.guard_triggers.iter().any(|g| g.guard_name == "PhaseGate"));
    }

    #[test]
    fn test_render_shows_model_parity_delta_enum() {
        let mut orch = CHPOrchestrator::new();
        let mut case = make_test_case();
        let disclosure = make_valid_disclosure();
        let attack = make_valid_attack(85);
        let report = orch.run_initial_session(&mut case, &disclosure, &attack).unwrap();
        let rendered = report.render();
        // Should show delta as enum debug format
        assert!(rendered.contains("delta:"));
    }
}
