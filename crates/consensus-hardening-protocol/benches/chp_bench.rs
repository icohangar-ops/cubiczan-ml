//! Tier 2 benchmarks for consensus-hardening-protocol
#![allow(clippy::all)]

use criterion::{black_box, criterion_group, criterion_main, Criterion};
use chp::*;

/// Build a fully populated DecisionCase for benchmark use.
fn make_populated_case() -> DecisionCase {
    let mut case = DecisionCase::new("dc-bench-001", "Bench Test Decision", "finance", "alice");
    case.status = SessionStatus::LOCKED;
    case.current_phase = Phase::Spec;
    case.current_round = 3;
    case.foundation_score = Some(85);
    case.locked_decisions.push("item-1".into());
    case.locked_decisions.push("item-2".into());
    case.structural_vulnerabilities.push("vuln-a".into());
    case.structural_vulnerabilities.push("vuln-b".into());
    case.blind_spots.push("blind-1".into());
    case.constraint_diagnoses.push(ConstraintDiagnosis {
        item: "scope-1".into(),
        symptom_altitude: ConstraintAltitude::TACTICAL,
        constraint_altitude: ConstraintAltitude::STRUCTURAL,
        diagnosis: "Structural constraint must be resolved first.".into(),
    });
    case.dossier = Some(Dossier {
        core_problem: "How to optimize portfolio allocation under uncertainty".into(),
        goal_state: vec!["Maximize Sharpe ratio".into()],
        current_state: vec!["Current Sharpe is 0.8".into()],
        constraints: vec!["Max drawdown 15%".into()],
        scope: vec!["scope-a".into(), "scope-b".into()],
        ..Default::default()
    });
    case
}

/// Build a valid partner packet with all 7 required sections.
fn make_valid_partner_packet() -> contracts::PartnerPacket {
    let mut case = make_populated_case();
    case.status = SessionStatus::LOCKED;
    let snapshot = build_state_snapshot(&case, "[RX] [ABC123] CONFIRMED", None, None, None);

    let raw_payload = "\
BEGIN_PAYLOAD [RX] [ABC123]
ITEM_AGREEMENTS:
item-1: score 92, LOCKED
item-2: score 88, LOCKED
WINNER_FRAMING:
Option A provides superior risk-adjusted returns.
SCORING_TABLE:
A: clarity=8 leverage=9 risk=7 winner=true
B: clarity=5 leverage=4 risk=3 winner=false elimination_note=Lower scores.
OBJECTIONS:
obj-1, obj-2
FRAMEWORKS:
framework-a
CONVERGENCE_PLAN:
Plan step 1.
STATE_SNAPSHOT:
Phase: Spec, Round: 3/5, Status: LOCKED
END_PAYLOAD [RX] [ABC123]";

    contracts::PartnerPacket {
        item_agreements: vec![
            contracts::ItemAgreement {
                item: "item-1".into(),
                score: 92,
                status: SessionStatus::LOCKED,
                third_party_status: "CONFIRMED".into(),
                ..Default::default()
            },
            contracts::ItemAgreement {
                item: "item-2".into(),
                score: 88,
                status: SessionStatus::LOCKED,
                third_party_status: "CONFIRMED".into(),
                ..Default::default()
            },
        ],
        winner_framing: "Option A provides superior risk-adjusted returns.".into(),
        scoring_table: vec![
            contracts::ScoringOption {
                name: "A".into(),
                clarity: 8,
                leverage: 9,
                risk: 7,
                winner: true,
                elimination_note: String::new(),
            },
            contracts::ScoringOption {
                name: "B".into(),
                clarity: 5,
                leverage: 4,
                risk: 3,
                winner: false,
                elimination_note: "Lower scores.".into(),
            },
        ],
        objections: vec!["obj-1".into(), "obj-2".into()],
        frameworks: vec!["framework-a".into()],
        convergence_plan: vec!["Plan step 1.".into()],
        state_snapshot: snapshot,
        raw_payload: raw_payload.into(),
    }
}

fn bench_r0_gate(c: &mut Criterion) {
    let mut group = c.benchmark_group("gates/r0_gate");
    group.bench_function("all_pass", |b| {
        b.iter(|| evaluate_r0_gate(black_box(true), black_box(true), black_box(true), black_box(true)))
    });
    group.bench_function("fail_scenario", |b| {
        b.iter(|| evaluate_r0_gate(black_box(true), black_box(false), black_box(true), black_box(true)))
    });
    group.finish();
}

fn bench_phase_gate(c: &mut Criterion) {
    let gate = PhaseGate::new();

    let mut group = c.benchmark_group("gates/evaluate_phase_gate");
    group.bench_function("early_round_1", |b| {
        b.iter(|| evaluate_phase_gate(black_box(1), black_box(&SessionStatus::EXPLORING)))
    });
    group.bench_function("early_round_2", |b| {
        b.iter(|| evaluate_phase_gate(black_box(2), black_box(&SessionStatus::PROVISIONAL)))
    });
    group.bench_function("r3_locked", |b| {
        b.iter(|| evaluate_phase_gate(black_box(3), black_box(&SessionStatus::LOCKED)))
    });
    group.bench_function("r3_fail", |b| {
        b.iter(|| evaluate_phase_gate(black_box(3), black_box(&SessionStatus::EXPLORING)))
    });
    group.finish();

    let mut group2 = c.benchmark_group("gates/PhaseGate_check");
    group2.bench_function("foundation_always_pass", |b| {
        b.iter(|| gate.check(black_box(3), Phase::Foundation, black_box(&SessionStatus::EXPLORING)))
    });
    group2.bench_function("spec_early_round_pass", |b| {
        b.iter(|| gate.check(black_box(1), Phase::Spec, black_box(&SessionStatus::EXPLORING)))
    });
    group2.bench_function("spec_r3_locked_pass", |b| {
        b.iter(|| gate.check(black_box(3), Phase::Spec, black_box(&SessionStatus::LOCKED)))
    });
    group2.bench_function("impl_r3_converged_pass", |b| {
        b.iter(|| gate.check(black_box(3), Phase::Implementation, black_box(&SessionStatus::CONVERGED)))
    });
    group2.bench_function("spec_r3_exploring_fail", |b| {
        b.iter(|| gate.check(black_box(3), Phase::Spec, black_box(&SessionStatus::EXPLORING)))
    });
    group2.bench_function("impl_r5_unresolved_fail", |b| {
        b.iter(|| gate.check(black_box(5), Phase::Implementation, black_box(&SessionStatus::UNRESOLVED)))
    });
    group2.finish();
}

fn bench_payload_validator(c: &mut Criterion) {
    let mut group = c.benchmark_group("payloads/validate_echo");
    group.bench_function("confirmed", |b| {
        b.iter(|| PayloadValidator::validate_echo(black_box("ABC123"), black_box("ABC123")))
    });
    group.bench_function("mismatch", |b| {
        b.iter(|| PayloadValidator::validate_echo(black_box("ABC123"), black_box("XYZ789")))
    });
    group.bench_function("missing_empty", |b| {
        b.iter(|| PayloadValidator::validate_echo(black_box(""), black_box("ABC123")))
    });
    group.finish();

    let mut group2 = c.benchmark_group("payloads/gate");
    group2.bench_function("pass", |b| {
        b.iter(|| PayloadValidator::gate(
            black_box("[RX] [ABC123] CONFIRMED"),
            black_box("RX"),
            black_box("ABC123"),
        ))
    });
    group2.bench_function("fail_empty", |b| {
        b.iter(|| PayloadValidator::gate(black_box(""), black_box("RX"), black_box("ABC123")))
    });
    group2.bench_function("fail_mismatch", |b| {
        b.iter(|| PayloadValidator::gate(
            black_box("[RX] [WRONG] CONFIRMED"),
            black_box("RX"),
            black_box("ABC123"),
        ))
    });
    group2.finish();
}

fn bench_payload_id(c: &mut Criterion) {
    c.bench_function("payloads/make_payload_id", |b| {
        b.iter(|| black_box(make_payload_id()))
    });
}

fn bench_validate_envelope(c: &mut Criterion) {
    let valid_envelope = build_payload_envelope("test body content here", "RX", Some("ABC123")).render();
    let invalid_envelope = "random text without proper envelope markers";

    let mut group = c.benchmark_group("payloads/validate_envelope");
    group.bench_function("valid_envelope", |b| {
        b.iter(|| validate_payload_envelope(black_box(&valid_envelope)))
    });
    group.bench_function("invalid_envelope", |b| {
        b.iter(|| validate_payload_envelope(black_box(invalid_envelope)))
    });
    group.finish();
}

fn bench_build_state_snapshot(c: &mut Criterion) {
    let case = make_populated_case();
    c.benchmark_group("devil/build_state_snapshot")
        .bench_function("from_case", |b| {
            b.iter(|| build_state_snapshot(black_box(&case), black_box("[RX] [PID] CONFIRMED"), None, None, None))
        });
}

fn bench_build_devils_advocate(c: &mut Criterion) {
    let disclosure = FoundationDisclosure {
        weakest_assumptions: vec!["assumption-1".into(), "assumption-2".into()],
        invalidation_conditions: vec!["condition-1".into()],
        key_vulnerability: "key-vuln".into(),
    };
    let attack = FoundationAttack {
        assumption_attacks: vec!["attack-1".into(), "attack-2".into()],
        vulnerability_strike: "strike-text".into(),
        foundation_score: 80,
        ..Default::default()
    };
    let case = make_populated_case();

    let mut group = c.benchmark_group("devil/build_devils_advocate");
    group.bench_function("phase0", |b| {
        b.iter(|| build_phase0_devils_advocate(black_box(&disclosure), black_box(&attack)))
    });
    group.bench_function("round3", |b| {
        b.iter(|| build_round3_devils_advocate(black_box(&case)))
    });
    group.finish();
}

fn bench_partner_packet_validate(c: &mut Criterion) {
    let packet = make_valid_partner_packet();
    c.benchmark_group("contracts/PartnerPacket")
        .bench_function("validate_full_7_sections", |b| {
            b.iter(|| packet.validate())
        });
}

fn bench_parse_partner_packet(c: &mut Criterion) {
    let raw = "\
BEGIN_PAYLOAD [RX] [A1B2C3]
ITEM_AGREEMENTS:
item1: score 90, LOCKED
item2: score 85, PROVISIONAL
WINNER_FRAMING:
The winner is A due to superior clarity.
SCORING_TABLE:
A: clarity=8 leverage=9 risk=7 winner=true
B: clarity=5 leverage=4 risk=3 winner=false elimination_note=Lower clarity.
OBJECTIONS:
obj-1
FRAMEWORKS:
framework-a
CONVERGENCE_PLAN:
Plan step 1.
STATE_SNAPSHOT:
Phase: Spec, Round: 3/5, Status: LOCKED
END_PAYLOAD [RX] [A1B2C3]";

    c.benchmark_group("parser/parse_partner_packet")
        .bench_function("full_7_sections", |b| {
            b.iter(|| parse_partner_packet(black_box(raw)))
        });
}

fn bench_verification_checklist(c: &mut Criterion) {
    let case = make_populated_case();
    let packet = "[RX] [ABC123] CONFIRMED";
    c.benchmark_group("contracts/VerificationChecklist")
        .bench_function("run", |b| {
            b.iter(|| contracts::VerificationChecklist::run(black_box(&case), black_box(packet)))
        });
}

fn bench_convergence_closure(c: &mut Criterion) {
    let case = make_populated_case();
    c.benchmark_group("contracts/ConvergenceClosure")
        .bench_function("from_case", |b| {
            b.iter(|| contracts::ConvergenceClosure::from_case(black_box(&case), "http://origin", "http://partner"))
        });
}

fn bench_assess_model_parity(c: &mut Criterion) {
    let mut group = c.benchmark_group("parity/assess_model_parity");
    group.bench_function("same_tier", |b| {
        b.iter(|| assess_model_parity(black_box("claude-sonnet-4"), black_box("claude-sonnet-4")))
    });
    group.bench_function("different_tier", |b| {
        b.iter(|| assess_model_parity(black_box("claude-opus-4"), black_box("claude-haiku-3")))
    });
    group.finish();
}

fn bench_next_round(c: &mut Criterion) {
    let mut group = c.benchmark_group("rounds/next_round");
    group.bench_function("foundation_to_spec", |b| {
        b.iter(|| next_round(Phase::Foundation, 0, black_box(&SessionStatus::EXPLORING)))
    });
    group.bench_function("spec_to_impl", |b| {
        b.iter(|| next_round(Phase::Spec, 2, black_box(&SessionStatus::LOCKED)))
    });
    group.bench_function("impl_advance", |b| {
        b.iter(|| next_round(Phase::Implementation, 3, black_box(&SessionStatus::LOCKED)))
    });
    group.bench_function("max_rounds_halt", |b| {
        b.iter(|| next_round(Phase::Implementation, 5, black_box(&SessionStatus::LOCKED)))
    });
    group.finish();
}

fn bench_context_engine(c: &mut Criterion) {
    let mut ctx = context::ContextEngine::new();
    for i in 0..50 {
        ctx.write(&format!("Entry {} about APAC revenue growth in quarterly report", i), "report", 0.8);
        ctx.write(&format!("Revenue entry {} for APAC region analysis", i), "finance", 0.9);
    }

    let mut group = c.benchmark_group("context/ContextEngine");
    group.bench_function("write", |b| {
        b.iter(|| ctx.write(black_box("New APAC revenue data shows strong quarterly growth"), black_box("report"), black_box(0.85)))
    });
    group.bench_function("select_5_results", |b| {
        b.iter(|| ctx.select(black_box("APAC revenue"), 5))
    });
    group.finish();
}

criterion_group!(
    benches,
    bench_r0_gate,
    bench_phase_gate,
    bench_payload_validator,
    bench_payload_id,
    bench_validate_envelope,
    bench_build_state_snapshot,
    bench_build_devils_advocate,
    bench_partner_packet_validate,
    bench_parse_partner_packet,
    bench_verification_checklist,
    bench_convergence_closure,
    bench_assess_model_parity,
    bench_next_round,
    bench_context_engine,
);
criterion_main!(benches);
