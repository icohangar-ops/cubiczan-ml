//! Governance policy engine with audit chain and model heterogeneity scoring.
//!
//! This module provides:
//! - Model family classification and heterogeneity scoring
//! - Policy gate with rate limiting, kill switch, and evidence requirements
//! - Audit kernel with HMAC-SHA256 chained hash verification

use crate::types::*;
use chrono::{DateTime, Utc};
use hmac::{Hmac, Mac};
use sha2::Sha256;
use std::collections::HashMap;

type HmacSha256 = Hmac<Sha256>;

// ---------------------------------------------------------------------------
// Model family classification
// ---------------------------------------------------------------------------

/// Classify a model name into its vendor family.
///
/// Examples:
/// - "gpt-4", "gpt-4o" → "openai"
/// - "claude-3-opus", "claude-3.5-sonnet" → "anthropic"
/// - "qwen2.5-72b" → "qwen"
/// - "llama-3.1-405b" → "meta"
/// - "gemini-pro" → "google"
pub fn model_family(name: &str) -> &str {
    let lower = name.to_lowercase();

    if lower.contains("gpt") || lower.contains("o1") || lower.contains("o3") {
        "openai"
    } else if lower.contains("claude") {
        "anthropic"
    } else if lower.contains("qwen") {
        "qwen"
    } else if lower.contains("llama") {
        "meta"
    } else if lower.contains("gemini") || lower.contains("palm") {
        "google"
    } else if lower.contains("mistral") || lower.contains("mixtral") || lower.contains("codestral") {
        "mistral"
    } else if lower.contains("deepseek") {
        "deepseek"
    } else if lower.contains("phi") {
        "microsoft"
    } else if lower.contains("command") || lower.contains("cohere") {
        "cohere"
    } else {
        "unknown"
    }
}

/// Compute a heterogeneity score based on model diversity.
///
/// The score is a weighted combination:
/// - 65% unique family ratio (unique families / total models)
/// - 35% anti-monoculture score (1 - dominant_family_count / total)
///
/// Returns (score, dominant_family_name).
pub fn compute_heterogeneity_score(model_names: &[&str]) -> (f64, String) {
    if model_names.is_empty() {
        return (0.0, String::from("none"));
    }

    let families: Vec<&str> = model_names.iter().map(|n| model_family(n)).collect();
    let unique_families: HashSet<&str> = families.iter().copied().collect();
    let total = model_names.len() as f64;

    // Count per family
    let mut counts: HashMap<&str, usize> = HashMap::new();
    for &family in &families {
        *counts.entry(family).or_insert(0) += 1;
    }

    // Find dominant family
    let dominant = counts
        .iter()
        .max_by_key(|(_, &c)| c)
        .map(|(&f, _)| f)
        .unwrap_or("none");
    let dominant_count = counts.get(dominant).copied().unwrap_or(0) as f64;

    // Compute scores
    let unique_ratio = unique_families.len() as f64 / total;
    let anti_monoculture = 1.0 - (dominant_count / total);

    let score = 0.65 * unique_ratio + 0.35 * anti_monoculture;

    (score, dominant.to_string())
}

// ---------------------------------------------------------------------------
// Policy Gate
// ---------------------------------------------------------------------------

/// A policy gate that evaluates tool access requests.
///
/// Evaluation order:
/// 1. Kill switch → Block everything
/// 2. Blocked actions → Block
/// 3. Rate budget (sliding window) → Block if exceeded
/// 4. Evidence requirements → RequireApproval if not met
/// 5. Approval-required actions → RequireApproval
/// 6. Otherwise → Allow
pub struct PolicyGate {
    policies: HashMap<String, ToolPolicy>,
    call_history: HashMap<String, Vec<DateTime<Utc>>>,
    kill_switch: bool,
}

impl PolicyGate {
    /// Create a new policy gate with no policies.
    pub fn new() -> Self {
        PolicyGate {
            policies: HashMap::new(),
            call_history: HashMap::new(),
            kill_switch: false,
        }
    }

    /// Builder: add a policy to the gate.
    pub fn with_policy(mut self, policy: ToolPolicy) -> Self {
        self.policies.insert(policy.tool.clone(), policy);
        self
    }

    /// Activate the kill switch (blocks all tool access).
    pub fn activate_kill_switch(&mut self) {
        self.kill_switch = true;
    }

    /// Deactivate the kill switch.
    pub fn deactivate_kill_switch(&mut self) {
        self.kill_switch = false;
    }

    /// Check if the kill switch is active.
    pub fn is_kill_switch_active(&self) -> bool {
        self.kill_switch
    }

    /// Add a policy after construction.
    pub fn add_policy(&mut self, policy: ToolPolicy) {
        self.policies.insert(policy.tool.clone(), policy);
    }

    /// Evaluate a tool access request.
    ///
    /// Returns a PolicyDecision with the action, reason, and optional approval ID.
    pub fn evaluate(
        &mut self,
        tool: &str,
        action: &str,
        actor: &str,
        evidence_sources: u32,
    ) -> PolicyDecision {
        // 1. Kill switch check
        if self.kill_switch {
            return PolicyDecision::new(
                PolicyAction::Block,
                "kill switch is active",
                tool,
                actor,
                action,
                "kill_switch",
            );
        }

        // Get policy (default deny if no policy)
        let policy = match self.policies.get(tool) {
            Some(p) => p,
            None => {
                return PolicyDecision::new(
                    PolicyAction::Block,
                    "no policy found for tool",
                    tool,
                    actor,
                    action,
                    "no_policy",
                );
            }
        };

        // 2. Blocked actions
        if policy.is_action_blocked(action) {
            return PolicyDecision::new(
                PolicyAction::Block,
                format!("action '{}' is blocked for tool '{}'", action, tool),
                tool,
                actor,
                action,
                &policy.policy_id,
            );
        }

        // 3. Rate budget check
        let remaining = self.remaining_budget(tool, policy.window_seconds);
        if remaining == 0 {
            return PolicyDecision::new(
                PolicyAction::Block,
                format!(
                    "rate limit exceeded for tool '{}' (window: {}s)",
                    tool, policy.window_seconds
                ),
                tool,
                actor,
                action,
                &policy.policy_id,
            );
        }

        // Record the call
        self.record_call(tool);

        // 4. Evidence requirement check
        if policy.min_evidence_sources > 0 && evidence_sources < policy.min_evidence_sources {
            return PolicyDecision::new(
                PolicyAction::RequireApproval,
                format!(
                    "insufficient evidence sources: {} < {} required",
                    evidence_sources, policy.min_evidence_sources
                ),
                tool,
                actor,
                action,
                &policy.policy_id,
            );
        }

        // 5. Approval-required actions
        if policy.is_approval_required(action) {
            return PolicyDecision::new(
                PolicyAction::RequireApproval,
                format!("action '{}' requires approval for tool '{}'", action, tool),
                tool,
                actor,
                action,
                &policy.policy_id,
            );
        }

        // 6. Allow
        PolicyDecision::new(
            PolicyAction::Allow,
            format!("action '{}' allowed for tool '{}'", action, tool),
            tool,
            actor,
            action,
            &policy.policy_id,
        )
    }

    /// Record a tool call for rate limiting.
    fn record_call(&mut self, tool: &str) {
        self.call_history
            .entry(tool.to_string())
            .or_insert_with(Vec::new)
            .push(Utc::now());
    }

    /// Compute the remaining budget for a tool in the given window.
    pub fn remaining_budget(&self, tool: &str, window_secs: u64) -> u32 {
        let policy = match self.policies.get(tool) {
            Some(p) => p,
            None => return 0,
        };

        let calls = self.call_history.get(tool).map(|v| v.as_slice()).unwrap_or(&[]);
        let cutoff = Utc::now() - chrono::Duration::seconds(window_secs as i64);

        let recent_calls = calls.iter().filter(|t| **t > cutoff).count() as u32;

        if recent_calls >= policy.max_calls {
            0
        } else {
            policy.max_calls - recent_calls
        }
    }

    /// Get the number of recorded calls for a tool.
    pub fn call_count(&self, tool: &str) -> usize {
        self.call_history.get(tool).map(|v| v.len()).unwrap_or(0)
    }

    /// Get all policy names.
    pub fn policy_names(&self) -> Vec<&str> {
        self.policies.keys().map(|s| s.as_str()).collect()
    }

    /// Reset call history for a specific tool.
    pub fn reset_history(&mut self, tool: &str) {
        self.call_history.remove(tool);
    }

    /// Reset all call history.
    pub fn reset_all_history(&mut self) {
        self.call_history.clear();
    }
}

impl Default for PolicyGate {
    fn default() -> Self {
        Self::new()
    }
}

// ---------------------------------------------------------------------------
// Audit Kernel
// ---------------------------------------------------------------------------

/// An append-only audit chain with HMAC-SHA256 integrity verification.
///
/// Each event contains:
/// - A digest computed as HMAC-SHA256(previous_hash + payload_json)
/// - A reference to the previous event's digest for chain linkage
///
/// The chain can be verified by replaying all events and checking
/// both the hash linkage and digest integrity.
pub struct AuditKernel {
    chain: Vec<AuditEvent>,
    secret: Option<String>,
}

/// A single event in the audit chain.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct AuditEvent {
    pub event_id: String,
    pub timestamp: DateTime<Utc>,
    pub actor: String,
    pub action: String,
    pub previous_hash: String,
    pub digest: String,
    pub payload: serde_json::Value,
}

/// Result of chain verification.
#[derive(Debug, Clone)]
pub struct VerificationReport {
    pub valid: bool,
    pub event_count: usize,
    pub first_violation: Option<String>,
}

impl VerificationReport {
    /// Create a successful verification report.
    pub fn valid(event_count: usize) -> Self {
        VerificationReport {
            valid: true,
            event_count,
            first_violation: None,
        }
    }

    /// Create a failed verification report.
    pub fn invalid(event_count: usize, reason: impl Into<String>) -> Self {
        VerificationReport {
            valid: false,
            event_count,
            first_violation: Some(reason.into()),
        }
    }
}

impl AuditKernel {
    /// Create a new audit kernel.
    ///
    /// If a secret is provided, HMAC-SHA256 digests will be computed.
    /// Without a secret, a simple SHA-256 hash is used.
    pub fn new(secret: Option<String>) -> Self {
        AuditKernel {
            chain: Vec::new(),
            secret,
        }
    }

    /// Record an event in the audit chain.
    ///
    /// Returns a reference to the newly created event.
    pub fn record(
        &mut self,
        actor: &str,
        action: &str,
        payload: serde_json::Value,
    ) -> &AuditEvent {
        let previous_hash = self
            .chain
            .last()
            .map(|e| e.digest.clone())
            .unwrap_or_else(|| "GENESIS".to_string());

        let event_id = uuid::Uuid::new_v4().to_string();
        let timestamp = Utc::now();

        let payload_json = serde_json::to_string(&payload).unwrap_or_default();
        let input = format!("{}{}", previous_hash, payload_json);
        let digest = self.compute_digest(&input);

        let event = AuditEvent {
            event_id,
            timestamp,
            actor: actor.to_string(),
            action: action.to_string(),
            previous_hash,
            digest,
            payload,
        };

        self.chain.push(event);
        self.chain.last().unwrap()
    }

    /// Verify the integrity of the entire audit chain.
    ///
    /// Checks:
    /// 1. Hash linkage (each event's previous_hash matches the prior event's digest)
    /// 2. Digest integrity (replay the HMAC computation)
    pub fn verify_chain(&self) -> Result<VerificationReport, String> {
        let mut expected_prev = "GENESIS".to_string();

        for (i, event) in self.chain.iter().enumerate() {
            // Check previous hash linkage
            if event.previous_hash != expected_prev {
                return Ok(VerificationReport::invalid(
                    self.chain.len(),
                    format!(
                        "event {} ({:?}): previous_hash mismatch (expected '{}', got '{}')",
                        i, event.action, expected_prev, event.previous_hash
                    ),
                ));
            }

            // Check digest integrity
            let payload_json = serde_json::to_string(&event.payload).unwrap_or_default();
            let input = format!("{}{}", event.previous_hash, payload_json);
            let expected_digest = self.compute_digest(&input);

            if event.digest != expected_digest {
                return Ok(VerificationReport::invalid(
                    self.chain.len(),
                    format!(
                        "event {} ({:?}): digest mismatch",
                        i, event.action
                    ),
                ));
            }

            expected_prev = event.digest.clone();
        }

        Ok(VerificationReport::valid(self.chain.len()))
    }

    /// Compute the HMAC-SHA256 or plain SHA-256 digest of a message.
    fn compute_digest(&self, message: &str) -> String {
        match &self.secret {
            Some(secret) => {
                let mut mac =
                    HmacSha256::new_from_slice(secret.as_bytes()).expect("HMAC key error");
                mac.update(message.as_bytes());
                let result = mac.finalize();
                hex::encode(result.into_bytes())
            }
            None => {
                use sha2::Digest;
                let mut hasher = Sha256::new();
                hasher.update(message.as_bytes());
                hex::encode(hasher.finalize())
            }
        }
    }

    /// Returns the number of events in the chain.
    pub fn len(&self) -> usize {
        self.chain.len()
    }

    /// Returns true if the chain has no events.
    pub fn is_empty(&self) -> bool {
        self.chain.is_empty()
    }

    /// Export the entire chain as a JSON value.
    pub fn export_json(&self) -> serde_json::Value {
        serde_json::to_value(&self.chain).unwrap_or(serde_json::Value::Null)
    }

    /// Get the last event in the chain.
    pub fn last_event(&self) -> Option<&AuditEvent> {
        self.chain.last()
    }

    /// Get an event by index.
    pub fn get_event(&self, index: usize) -> Option<&AuditEvent> {
        self.chain.get(index)
    }

    /// Get all events by a specific actor.
    pub fn events_by_actor(&self, actor: &str) -> Vec<&AuditEvent> {
        self.chain
            .iter()
            .filter(|e| e.actor == actor)
            .collect()
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    // -- Model family tests --

    #[test]
    fn test_model_family_openai() {
        assert_eq!(model_family("gpt-4"), "openai");
        assert_eq!(model_family("gpt-4o"), "openai");
        assert_eq!(model_family("GPT-4-Turbo"), "openai");
        assert_eq!(model_family("o1-preview"), "openai");
        assert_eq!(model_family("o3-mini"), "openai");
    }

    #[test]
    fn test_model_family_anthropic() {
        assert_eq!(model_family("claude-3-opus"), "anthropic");
        assert_eq!(model_family("claude-3.5-sonnet"), "anthropic");
        assert_eq!(model_family("CLAUDE-3-HAIKU"), "anthropic");
    }

    #[test]
    fn test_model_family_qwen() {
        assert_eq!(model_family("qwen2.5-72b"), "qwen");
        assert_eq!(model_family("Qwen-Coder"), "qwen");
    }

    #[test]
    fn test_model_family_meta() {
        assert_eq!(model_family("llama-3.1-405b"), "meta");
        assert_eq!(model_family("LLaMA-2-7B"), "meta");
    }

    #[test]
    fn test_model_family_google() {
        assert_eq!(model_family("gemini-pro"), "google");
        assert_eq!(model_family("palm-2"), "google");
    }

    #[test]
    fn test_model_family_mistral() {
        assert_eq!(model_family("mistral-7b"), "mistral");
        assert_eq!(model_family("mixtral-8x7b"), "mistral");
        assert_eq!(model_family("codestral"), "mistral");
    }

    #[test]
    fn test_model_family_deepseek() {
        assert_eq!(model_family("deepseek-v2"), "deepseek");
        assert_eq!(model_family("deepseek-coder"), "deepseek");
    }

    #[test]
    fn test_model_family_unknown() {
        assert_eq!(model_family("my-custom-model"), "unknown");
    }

    // -- Heterogeneity score tests --

    #[test]
    fn test_heterogeneity_empty() {
        let (score, dominant) = compute_heterogeneity_score(&[]);
        assert_eq!(score, 0.0);
        assert_eq!(dominant, "none");
    }

    #[test]
    fn test_heterogeneity_single() {
        let (score, dominant) = compute_heterogeneity_score(&["gpt-4"]);
        assert_eq!(dominant, "openai");
        assert!(score > 0.0);
    }

    #[test]
    fn test_heterogeneity_diverse() {
        let models = ["gpt-4", "claude-3-opus", "llama-3.1-70b", "gemini-pro"];
        let (score, _) = compute_heterogeneity_score(&models);
        // 4 unique families out of 4 models → unique_ratio = 1.0
        // No dominant family (each has 1/4) → anti_monoculture = 0.75
        // score = 0.65 * 1.0 + 0.35 * 0.75 = 0.9125
        assert!((score - 0.9125).abs() < 0.01);
    }

    #[test]
    fn test_heterogeneity_monoculture() {
        let models = ["gpt-4", "gpt-4o", "gpt-4-turbo"];
        let (score, dominant) = compute_heterogeneity_score(&models);
        assert_eq!(dominant, "openai");
        // unique_ratio = 1/3 ≈ 0.333
        // anti_monoculture = 0
        // score = 0.65 * 0.333 ≈ 0.217
        assert!(score < 0.3);
    }

    #[test]
    fn test_heterogeneity_dominant() {
        let models = ["gpt-4", "gpt-4o", "claude-3-opus"];
        let (score, dominant) = compute_heterogeneity_score(&models);
        assert_eq!(dominant, "openai");
        // unique_ratio = 2/3 ≈ 0.667
        // anti_monoculture = 1 - 2/3 = 0.333
        // score = 0.65 * 0.667 + 0.35 * 0.333 ≈ 0.556
        assert!((score - 0.556).abs() < 0.05);
    }

    // -- PolicyGate tests --

    #[test]
    fn test_policy_gate_new() {
        let gate = PolicyGate::new();
        assert!(!gate.is_kill_switch_active());
        assert!(gate.policy_names().is_empty());
    }

    #[test]
    fn test_policy_gate_kill_switch() {
        let mut gate = PolicyGate::new();
        gate.activate_kill_switch();
        assert!(gate.is_kill_switch_active());

        let policy = ToolPolicy::new("browser", TrustLevel::Autonomous);
        gate.add_policy(policy);

        let decision = gate.evaluate("browser", "navigate", "agent1", 0);
        assert_eq!(decision.action, PolicyAction::Block);
        assert!(decision.reason.contains("kill switch"));

        gate.deactivate_kill_switch();
        assert!(!gate.is_kill_switch_active());
    }

    #[test]
    fn test_policy_gate_no_policy() {
        let mut gate = PolicyGate::new();
        let decision = gate.evaluate("unknown_tool", "action", "agent1", 0);
        assert_eq!(decision.action, PolicyAction::Block);
    }

    #[test]
    fn test_policy_gate_allow() {
        let policy = ToolPolicy::new("browser", TrustLevel::Autonomous);
        let mut gate = PolicyGate::new().with_policy(policy);

        let decision = gate.evaluate("browser", "navigate", "agent1", 0);
        assert_eq!(decision.action, PolicyAction::Allow);
        assert!(decision.is_allowed());
    }

    #[test]
    fn test_policy_gate_blocked_action() {
        let policy = ToolPolicy::new("shell", TrustLevel::Autonomous).block_action("rm -rf");
        let mut gate = PolicyGate::new().with_policy(policy);

        let decision = gate.evaluate("shell", "rm -rf", "agent1", 0);
        assert_eq!(decision.action, PolicyAction::Block);
        assert!(decision.reason.contains("blocked"));
    }

    #[test]
    fn test_policy_gate_rate_limit() {
        let policy = ToolPolicy::new("api", TrustLevel::Autonomous)
            .with_max_calls(2)
            .with_window(3600);
        let mut gate = PolicyGate::new().with_policy(policy);

        // First call
        let d1 = gate.evaluate("api", "call", "agent1", 0);
        assert_eq!(d1.action, PolicyAction::Allow);

        // Second call
        let d2 = gate.evaluate("api", "call", "agent1", 0);
        assert_eq!(d2.action, PolicyAction::Allow);

        // Third call should be blocked
        let d3 = gate.evaluate("api", "call", "agent1", 0);
        assert_eq!(d3.action, PolicyAction::Block);
        assert!(d3.reason.contains("rate limit"));
    }

    #[test]
    fn test_policy_gate_evidence_requirement() {
        let policy = ToolPolicy::new("trading", TrustLevel::Supervised).with_min_evidence(2);
        let mut gate = PolicyGate::new().with_policy(policy);

        // Insufficient evidence
        let d1 = gate.evaluate("trading", "buy", "agent1", 1);
        assert_eq!(d1.action, PolicyAction::RequireApproval);

        // Sufficient evidence
        let d2 = gate.evaluate("trading", "buy", "agent1", 3);
        assert_eq!(d2.action, PolicyAction::Allow);
    }

    #[test]
    fn test_policy_gate_approval_required_action() {
        let policy =
            ToolPolicy::new("trading", TrustLevel::ApprovalRequired).require_approval_for("sell");
        let mut gate = PolicyGate::new().with_policy(policy);

        let decision = gate.evaluate("trading", "sell", "agent1", 5);
        assert_eq!(decision.action, PolicyAction::RequireApproval);
        assert!(decision.requires_approval);
        assert!(decision.approval_id.is_some());
    }

    #[test]
    fn test_policy_gate_remaining_budget() {
        let policy = ToolPolicy::new("api", TrustLevel::Autonomous)
            .with_max_calls(5)
            .with_window(3600);
        let mut gate = PolicyGate::new().with_policy(policy);

        assert_eq!(gate.remaining_budget("api", 3600), 5);

        gate.evaluate("api", "call", "a1", 0);
        assert_eq!(gate.remaining_budget("api", 3600), 4);
    }

    #[test]
    fn test_policy_gate_call_count() {
        let policy = ToolPolicy::new("api", TrustLevel::Autonomous);
        let mut gate = PolicyGate::new().with_policy(policy);

        assert_eq!(gate.call_count("api"), 0);
        gate.evaluate("api", "call", "a1", 0);
        gate.evaluate("api", "call", "a1", 0);
        assert_eq!(gate.call_count("api"), 2);
    }

    #[test]
    fn test_policy_gate_reset_history() {
        let policy = ToolPolicy::new("api", TrustLevel::Autonomous)
            .with_max_calls(1)
            .with_window(3600);
        let mut gate = PolicyGate::new().with_policy(policy);

        gate.evaluate("api", "call", "a1", 0);
        assert_eq!(gate.remaining_budget("api", 3600), 0);

        gate.reset_history("api");
        assert_eq!(gate.remaining_budget("api", 3600), 1);
    }

    #[test]
    fn test_policy_gate_with_policy_builder() {
        let policy = ToolPolicy::new("t1", TrustLevel::Autonomous);
        let gate = PolicyGate::new().with_policy(policy);
        assert!(gate.policy_names().contains(&"t1"));
    }

    // -- AuditKernel tests --

    #[test]
    fn test_audit_kernel_new() {
        let kernel = AuditKernel::new(None);
        assert!(kernel.is_empty());
    }

    #[test]
    fn test_audit_kernel_record() {
        let mut kernel = AuditKernel::new(Some("secret".to_string()));
        let event = kernel.record("agent1", "task_complete", serde_json::json!({"task": "t1"}));

        assert_eq!(event.actor, "agent1");
        assert_eq!(event.action, "task_complete");
        assert_eq!(event.previous_hash, "GENESIS");
        assert_eq!(kernel.len(), 1);
    }

    #[test]
    fn test_audit_kernel_chain_linkage() {
        let mut kernel = AuditKernel::new(Some("secret".to_string()));
        let e1 = kernel.record("a1", "action1", serde_json::json!(1));
        let e2 = kernel.record("a2", "action2", serde_json::json!(2));

        assert_eq!(e2.previous_hash, e1.digest);
        assert_eq!(kernel.len(), 2);
    }

    #[test]
    fn test_audit_kernel_verify_valid() {
        let mut kernel = AuditKernel::new(Some("test_secret".to_string()));
        kernel.record("a1", "action1", serde_json::json!({"x": 1}));
        kernel.record("a2", "action2", serde_json::json!({"y": 2}));
        kernel.record("a3", "action3", serde_json::json!({"z": 3}));

        let report = kernel.verify_chain().unwrap();
        assert!(report.valid);
        assert_eq!(report.event_count, 3);
    }

    #[test]
    fn test_audit_kernel_verify_no_secret() {
        let mut kernel = AuditKernel::new(None);
        kernel.record("a1", "action1", serde_json::json!(null));
        kernel.record("a2", "action2", serde_json::json!(null));

        let report = kernel.verify_chain().unwrap();
        assert!(report.valid);
    }

    #[test]
    fn test_audit_kernel_verify_tampered() {
        let mut kernel = AuditKernel::new(Some("secret".to_string()));
        kernel.record("a1", "action1", serde_json::json!({"original": true}));
        kernel.record("a2", "action2", serde_json::json!({"original": true}));

        // Tamper with an event
        kernel.chain[1].payload = serde_json::json!({"original": false, "tampered": true});

        let report = kernel.verify_chain().unwrap();
        assert!(!report.valid);
        assert!(report.first_violation.is_some());
    }

    #[test]
    fn test_audit_kernel_export_json() {
        let mut kernel = AuditKernel::new(None);
        kernel.record("a1", "action1", serde_json::json!(42));

        let json = kernel.export_json();
        assert!(json.is_array());
        assert_eq!(json.as_array().unwrap().len(), 1);
    }

    #[test]
    fn test_audit_kernel_last_event() {
        let mut kernel = AuditKernel::new(None);
        assert!(kernel.last_event().is_none());

        kernel.record("a1", "action1", serde_json::json!(null));
        assert!(kernel.last_event().is_some());
    }

    #[test]
    fn test_audit_kernel_get_event() {
        let mut kernel = AuditKernel::new(None);
        kernel.record("a1", "action1", serde_json::json!(null));
        kernel.record("a2", "action2", serde_json::json!(null));

        assert!(kernel.get_event(0).is_some());
        assert!(kernel.get_event(1).is_some());
        assert!(kernel.get_event(2).is_none());
    }

    #[test]
    fn test_audit_kernel_events_by_actor() {
        let mut kernel = AuditKernel::new(None);
        kernel.record("a1", "action1", serde_json::json!(null));
        kernel.record("a2", "action2", serde_json::json!(null));
        kernel.record("a1", "action3", serde_json::json!(null));

        let events = kernel.events_by_actor("a1");
        assert_eq!(events.len(), 2);
    }

    #[test]
    fn test_verification_report_valid() {
        let report = VerificationReport::valid(5);
        assert!(report.valid);
        assert_eq!(report.event_count, 5);
        assert!(report.first_violation.is_none());
    }

    #[test]
    fn test_verification_report_invalid() {
        let report = VerificationReport::invalid(3, "digest mismatch");
        assert!(!report.valid);
        assert_eq!(report.event_count, 3);
        assert_eq!(report.first_violation, Some("digest mismatch".to_string()));
    }

    #[test]
    fn test_policy_decision_approval_id() {
        let decision = PolicyDecision::new(
            PolicyAction::RequireApproval,
            "needs review",
            "tool",
            "actor",
            "action",
            "p1",
        );
        assert!(decision.requires_approval);
        assert!(decision.approval_id.is_some());
    }
}
