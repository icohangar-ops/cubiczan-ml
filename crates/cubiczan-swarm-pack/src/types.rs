//! Shared types for the swarm intelligence platform.
//!
//! This module defines all domain types used across the swarm pack:
//! domain classification, task status state machine, stigmergic scent types,
//! governance policies, consensus structures, and Solana CLI types.

use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::fmt;

// ---------------------------------------------------------------------------
// Domain classification
// ---------------------------------------------------------------------------

/// Application domain for swarm task routing.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, Hash)]
pub enum Domain {
    Financial,
    Cybersecurity,
    BusinessIntel,
    PredictiveSim,
    ContentMarketing,
    Healthcare,
    Political,
    RealEstate,
    TalentHR,
}

impl Domain {
    /// Returns all domain variants.
    pub fn all() -> &'static [Domain] {
        &[
            Domain::Financial,
            Domain::Cybersecurity,
            Domain::BusinessIntel,
            Domain::PredictiveSim,
            Domain::ContentMarketing,
            Domain::Healthcare,
            Domain::Political,
            Domain::RealEstate,
            Domain::TalentHR,
        ]
    }

    /// Returns the string label for this domain.
    pub fn as_str(&self) -> &'static str {
        match self {
            Domain::Financial => "financial",
            Domain::Cybersecurity => "cybersecurity",
            Domain::BusinessIntel => "business_intel",
            Domain::PredictiveSim => "predictive_sim",
            Domain::ContentMarketing => "content_marketing",
            Domain::Healthcare => "healthcare",
            Domain::Political => "political",
            Domain::RealEstate => "real_estate",
            Domain::TalentHR => "talent_hr",
        }
    }

    /// Parse a domain from a string label (case-insensitive).
    pub fn from_str_label(s: &str) -> Option<Domain> {
        match s.to_lowercase().as_str() {
            "financial" => Some(Domain::Financial),
            "cybersecurity" => Some(Domain::Cybersecurity),
            "business_intel" | "businessintel" => Some(Domain::BusinessIntel),
            "predictive_sim" | "predictivesim" => Some(Domain::PredictiveSim),
            "content_marketing" | "contentmarketing" => Some(Domain::ContentMarketing),
            "healthcare" => Some(Domain::Healthcare),
            "political" => Some(Domain::Political),
            "real_estate" | "realestate" => Some(Domain::RealEstate),
            "talent_hr" | "talenthr" => Some(Domain::TalentHR),
            _ => None,
        }
    }
}

impl fmt::Display for Domain {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.as_str())
    }
}

// ---------------------------------------------------------------------------
// Task status state machine
// ---------------------------------------------------------------------------

/// Task lifecycle status with valid transitions:
///
/// ```text
/// Pending → Ready → Active → Complete
///                  Active → Retry → Ready (loop)
///                  Active → Retry → Escalate (max retries)
///                  Active → Blocked
/// ```
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, Hash)]
pub enum TaskStatus {
    Pending,
    Ready,
    Active,
    Complete,
    Blocked,
    Retry,
    Escalate,
}

impl TaskStatus {
    /// Returns all status variants.
    pub fn all() -> &'static [TaskStatus] {
        &[
            TaskStatus::Pending,
            TaskStatus::Ready,
            TaskStatus::Active,
            TaskStatus::Complete,
            TaskStatus::Blocked,
            TaskStatus::Retry,
            TaskStatus::Escalate,
        ]
    }

    /// Returns true if the status is a terminal state.
    pub fn is_terminal(&self) -> bool {
        matches!(self, TaskStatus::Complete | TaskStatus::Blocked | TaskStatus::Escalate)
    }

    /// Returns true if the task can be worked on.
    pub fn is_workable(&self) -> bool {
        matches!(self, TaskStatus::Ready | TaskStatus::Active)
    }

    /// Returns true if the status is a final success state.
    pub fn is_success(&self) -> bool {
        *self == TaskStatus::Complete
    }

    /// Returns true if the status represents a failure or blocking state.
    pub fn is_failure(&self) -> bool {
        matches!(self, TaskStatus::Blocked | TaskStatus::Escalate)
    }

    /// Check whether a transition from `self` to `next` is valid.
    pub fn can_transition_to(&self, next: TaskStatus) -> bool {
        match (self, next) {
            // Normal flow
            (TaskStatus::Pending, TaskStatus::Ready) => true,
            (TaskStatus::Ready, TaskStatus::Active) => true,
            (TaskStatus::Active, TaskStatus::Complete) => true,
            // Failure flow
            (TaskStatus::Active, TaskStatus::Retry) => true,
            (TaskStatus::Retry, TaskStatus::Ready) => true,
            (TaskStatus::Retry, TaskStatus::Escalate) => true,
            // Blocking
            (TaskStatus::Active, TaskStatus::Blocked) => true,
            (TaskStatus::Pending, TaskStatus::Blocked) => true,
            // Re-activation from blocked
            (TaskStatus::Blocked, TaskStatus::Pending) => true,
            // Same state
            (a, b) if *a == b => true,
            _ => false,
        }
    }
}

impl fmt::Display for TaskStatus {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            TaskStatus::Pending => write!(f, "pending"),
            TaskStatus::Ready => write!(f, "ready"),
            TaskStatus::Active => write!(f, "active"),
            TaskStatus::Complete => write!(f, "complete"),
            TaskStatus::Blocked => write!(f, "blocked"),
            TaskStatus::Retry => write!(f, "retry"),
            TaskStatus::Escalate => write!(f, "escalate"),
        }
    }
}

// ---------------------------------------------------------------------------
// Scent types for stigmergic coordination
// ---------------------------------------------------------------------------

/// Types of pheromone signals emitted by agents for stigmergic coordination.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, Hash)]
pub enum ScentType {
    Completion,
    Failure,
    Difficulty,
    Urgency,
    Progress,
    HelpWanted,
}

impl ScentType {
    /// Returns all scent type variants.
    pub fn all() -> &'static [ScentType] {
        &[
            ScentType::Completion,
            ScentType::Failure,
            ScentType::Difficulty,
            ScentType::Urgency,
            ScentType::Progress,
            ScentType::HelpWanted,
        ]
    }

    /// Returns the default half-life in seconds for this scent type.
    /// A negative value means the scent grows over time (e.g., urgency).
    pub fn default_half_life(&self) -> f64 {
        match self {
            ScentType::Completion => 300.0,
            ScentType::Failure => 360.0,
            ScentType::Difficulty => 120.0,
            ScentType::Urgency => -1.0,
            ScentType::Progress => 20.0,
            ScentType::HelpWanted => 120.0,
        }
    }

    /// Returns true if this scent type grows over time instead of decaying.
    pub fn is_growing(&self) -> bool {
        *self == ScentType::Urgency
    }
}

impl fmt::Display for ScentType {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            ScentType::Completion => write!(f, "completion"),
            ScentType::Failure => write!(f, "failure"),
            ScentType::Difficulty => write!(f, "difficulty"),
            ScentType::Urgency => write!(f, "urgency"),
            ScentType::Progress => write!(f, "progress"),
            ScentType::HelpWanted => write!(f, "help_wanted"),
        }
    }
}

// ---------------------------------------------------------------------------
// Governance types
// ---------------------------------------------------------------------------

/// Trust level for a tool or operation.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, Hash)]
pub enum TrustLevel {
    Autonomous,
    Supervised,
    ApprovalRequired,
}

impl TrustLevel {
    /// Returns all trust level variants.
    pub fn all() -> &'static [TrustLevel] {
        &[TrustLevel::Autonomous, TrustLevel::Supervised, TrustLevel::ApprovalRequired]
    }

    /// Returns the numeric restriction level (higher = more restricted).
    pub fn restriction_level(&self) -> u8 {
        match self {
            TrustLevel::Autonomous => 0,
            TrustLevel::Supervised => 1,
            TrustLevel::ApprovalRequired => 2,
        }
    }
}

impl fmt::Display for TrustLevel {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            TrustLevel::Autonomous => write!(f, "autonomous"),
            TrustLevel::Supervised => write!(f, "supervised"),
            TrustLevel::ApprovalRequired => write!(f, "approval_required"),
        }
    }
}

/// Action determined by the policy engine.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, Hash)]
pub enum PolicyAction {
    Allow,
    RequireApproval,
    Block,
}

impl PolicyAction {
    /// Returns all policy action variants.
    pub fn all() -> &'static [PolicyAction] {
        &[PolicyAction::Allow, PolicyAction::RequireApproval, PolicyAction::Block]
    }

    /// Returns true if this action blocks execution.
    pub fn is_blocked(&self) -> bool {
        *self == PolicyAction::Block
    }

    /// Returns true if this action requires approval before execution.
    pub fn requires_approval(&self) -> bool {
        *self == PolicyAction::RequireApproval
    }
}

impl fmt::Display for PolicyAction {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            PolicyAction::Allow => write!(f, "allow"),
            PolicyAction::RequireApproval => write!(f, "require_approval"),
            PolicyAction::Block => write!(f, "block"),
        }
    }
}

// ---------------------------------------------------------------------------
// Solana types
// ---------------------------------------------------------------------------

/// Classification of Solana CLI commands.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, Hash)]
pub enum SolanaActionType {
    Read,
    Write,
    Deploy,
    Faucet,
    ConfigRead,
    Unsupported,
}

impl SolanaActionType {
    /// Returns all action type variants.
    pub fn all() -> &'static [SolanaActionType] {
        &[
            SolanaActionType::Read,
            SolanaActionType::Write,
            SolanaActionType::Deploy,
            SolanaActionType::Faucet,
            SolanaActionType::ConfigRead,
            SolanaActionType::Unsupported,
        ]
    }

    /// Returns true if the action modifies on-chain state.
    pub fn is_write(&self) -> bool {
        matches!(self, SolanaActionType::Write | SolanaActionType::Deploy | SolanaActionType::Faucet)
    }

    /// Returns true if the action is read-only.
    pub fn is_read_only(&self) -> bool {
        matches!(self, SolanaActionType::Read | SolanaActionType::ConfigRead)
    }

    /// Returns true if this action type requires approval by default.
    pub fn requires_approval(&self) -> bool {
        matches!(self, SolanaActionType::Deploy | SolanaActionType::Faucet)
    }
}

impl fmt::Display for SolanaActionType {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            SolanaActionType::Read => write!(f, "read"),
            SolanaActionType::Write => write!(f, "write"),
            SolanaActionType::Deploy => write!(f, "deploy"),
            SolanaActionType::Faucet => write!(f, "faucet"),
            SolanaActionType::ConfigRead => write!(f, "config_read"),
            SolanaActionType::Unsupported => write!(f, "unsupported"),
        }
    }
}

// ---------------------------------------------------------------------------
// DAG Task
// ---------------------------------------------------------------------------

/// A single task node in the DAG task graph.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DAGTask {
    pub task_id: String,
    pub description: String,
    pub agent_type: String,
    pub tags: Vec<String>,
    pub dependencies: Vec<String>,
    pub dependents: Vec<String>,
    pub status: TaskStatus,
    pub worker_id: Option<String>,
    pub result: Option<serde_json::Value>,
    pub retries: u32,
    pub max_retries: u32,
    pub created_at: DateTime<Utc>,
    pub started_at: Option<DateTime<Utc>>,
    pub completed_at: Option<DateTime<Utc>>,
}

impl DAGTask {
    /// Create a new pending task with the given parameters.
    pub fn new(
        task_id: impl Into<String>,
        description: impl Into<String>,
        agent_type: impl Into<String>,
        tags: Vec<String>,
        dependencies: Vec<String>,
        max_retries: u32,
    ) -> Self {
        DAGTask {
            task_id: task_id.into(),
            description: description.into(),
            agent_type: agent_type.into(),
            tags,
            dependencies,
            dependents: Vec::new(),
            status: TaskStatus::Pending,
            worker_id: None,
            result: None,
            retries: 0,
            max_retries,
            created_at: Utc::now(),
            started_at: None,
            completed_at: None,
        }
    }

    /// Returns true if the task has no unmet dependencies.
    pub fn dependencies_met(&self, tasks: &HashMap<String, DAGTask>) -> bool {
        self.dependencies.iter().all(|dep_id| {
            tasks
                .get(dep_id)
                .map(|t| t.status == TaskStatus::Complete)
                .unwrap_or(false)
        })
    }

    /// Returns the duration from start to completion, if both are set.
    pub fn duration(&self) -> Option<chrono::Duration> {
        match (self.started_at, self.completed_at) {
            (Some(start), Some(end)) => Some(end.signed_duration_since(start)),
            _ => None,
        }
    }

    /// Returns true if the task can be retried (retries < max_retries).
    pub fn can_retry(&self) -> bool {
        self.retries < self.max_retries
    }
}

// ---------------------------------------------------------------------------
// Task Graph
// ---------------------------------------------------------------------------

/// The full DAG task graph with metadata.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TaskGraph {
    pub graph_id: String,
    pub tasks: HashMap<String, DAGTask>,
    pub critical_path_length: u32,
    pub theoretical_speedup: f64,
}

impl TaskGraph {
    /// Create an empty task graph.
    pub fn new(graph_id: impl Into<String>) -> Self {
        TaskGraph {
            graph_id: graph_id.into(),
            tasks: HashMap::new(),
            critical_path_length: 0,
            theoretical_speedup: 1.0,
        }
    }

    /// Returns the total number of tasks in the graph.
    pub fn len(&self) -> usize {
        self.tasks.len()
    }

    /// Returns true if the graph has no tasks.
    pub fn is_empty(&self) -> bool {
        self.tasks.is_empty()
    }

    /// Returns the number of tasks in each status.
    pub fn status_counts(&self) -> HashMap<TaskStatus, usize> {
        let mut counts = HashMap::new();
        for task in self.tasks.values() {
            *counts.entry(task.status).or_insert(0) += 1;
        }
        counts
    }

    /// Returns the number of completed tasks.
    pub fn completed_count(&self) -> usize {
        self.tasks.values().filter(|t| t.status == TaskStatus::Complete).count()
    }

    /// Returns true if all tasks are in a terminal state.
    pub fn is_finished(&self) -> bool {
        !self.tasks.is_empty() && self.tasks.values().all(|t| t.status.is_terminal())
    }
}

// ---------------------------------------------------------------------------
// Scent Signal
// ---------------------------------------------------------------------------

/// A pheromone signal emitted by an agent.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ScentSignal {
    pub signal_id: String,
    pub task_id: String,
    pub worker_id: String,
    pub scent_type: ScentType,
    pub intensity: f64,
    pub emitted_at: DateTime<Utc>,
    pub metadata: serde_json::Value,
}

impl ScentSignal {
    /// Create a new scent signal.
    pub fn new(
        signal_id: impl Into<String>,
        task_id: impl Into<String>,
        worker_id: impl Into<String>,
        scent_type: ScentType,
        intensity: f64,
    ) -> Self {
        ScentSignal {
            signal_id: signal_id.into(),
            task_id: task_id.into(),
            worker_id: worker_id.into(),
            scent_type,
            intensity,
            emitted_at: Utc::now(),
            metadata: serde_json::Value::Null,
        }
    }

    /// Create a scent signal with custom metadata.
    pub fn with_metadata(
        signal_id: impl Into<String>,
        task_id: impl Into<String>,
        worker_id: impl Into<String>,
        scent_type: ScentType,
        intensity: f64,
        metadata: serde_json::Value,
    ) -> Self {
        ScentSignal {
            signal_id: signal_id.into(),
            task_id: task_id.into(),
            worker_id: worker_id.into(),
            scent_type,
            intensity,
            emitted_at: Utc::now(),
            metadata,
        }
    }
}

// ---------------------------------------------------------------------------
// Governance: Tool Policy & Decision
// ---------------------------------------------------------------------------

/// A governance policy for a specific tool.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ToolPolicy {
    pub tool: String,
    pub trust_level: TrustLevel,
    pub max_calls: u32,
    pub window_seconds: u64,
    pub approval_required_actions: Vec<String>,
    pub blocked_actions: Vec<String>,
    pub min_evidence_sources: u32,
    pub policy_id: String,
}

impl ToolPolicy {
    /// Create a new tool policy with default settings.
    pub fn new(tool: impl Into<String>, trust_level: TrustLevel) -> Self {
        ToolPolicy {
            tool: tool.into(),
            trust_level,
            max_calls: 100,
            window_seconds: 3600,
            approval_required_actions: Vec::new(),
            blocked_actions: Vec::new(),
            min_evidence_sources: 0,
            policy_id: uuid::Uuid::new_v4().to_string(),
        }
    }

    /// Builder: set maximum calls in the window.
    pub fn with_max_calls(mut self, max: u32) -> Self {
        self.max_calls = max;
        self
    }

    /// Builder: set window size in seconds.
    pub fn with_window(mut self, seconds: u64) -> Self {
        self.window_seconds = seconds;
        self
    }

    /// Builder: set minimum evidence sources required.
    pub fn with_min_evidence(mut self, min: u32) -> Self {
        self.min_evidence_sources = min;
        self
    }

    /// Builder: add a blocked action.
    pub fn block_action(mut self, action: impl Into<String>) -> Self {
        self.blocked_actions.push(action.into());
        self
    }

    /// Builder: add an action that requires approval.
    pub fn require_approval_for(mut self, action: impl Into<String>) -> Self {
        self.approval_required_actions.push(action.into());
        self
    }

    /// Check if a specific action is blocked by this policy.
    pub fn is_action_blocked(&self, action: &str) -> bool {
        self.blocked_actions.iter().any(|a| a.eq_ignore_ascii_case(action))
    }

    /// Check if a specific action requires approval.
    pub fn is_approval_required(&self, action: &str) -> bool {
        self.approval_required_actions.iter().any(|a| a.eq_ignore_ascii_case(action))
    }
}

/// The result of a policy evaluation.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PolicyDecision {
    pub action: PolicyAction,
    pub reason: String,
    pub tool: String,
    pub actor: String,
    pub requested_action: String,
    pub policy_id: String,
    pub requires_approval: bool,
    pub approval_id: Option<String>,
}

impl PolicyDecision {
    /// Create a new policy decision.
    pub fn new(
        action: PolicyAction,
        reason: impl Into<String>,
        tool: impl Into<String>,
        actor: impl Into<String>,
        requested_action: impl Into<String>,
        policy_id: impl Into<String>,
    ) -> Self {
        let requires_approval = action.requires_approval();
        PolicyDecision {
            action,
            reason: reason.into(),
            tool: tool.into(),
            actor: actor.into(),
            requested_action: requested_action.into(),
            policy_id: policy_id.into(),
            requires_approval,
            approval_id: if requires_approval {
                Some(uuid::Uuid::new_v4().to_string())
            } else {
                None
            },
        }
    }

    /// Returns true if this decision allows the action.
    pub fn is_allowed(&self) -> bool {
        self.action == PolicyAction::Allow
    }
}

// ---------------------------------------------------------------------------
// Consensus types
// ---------------------------------------------------------------------------

/// A vote cast by an agent in the consensus process.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AgentVote {
    pub agent_id: String,
    pub model_name: String,
    pub position: String,
    pub confidence: f64,
    pub reasoning: String,
    pub is_contrarian: bool,
}

impl AgentVote {
    /// Create a new agent vote.
    pub fn new(
        agent_id: impl Into<String>,
        model_name: impl Into<String>,
        position: impl Into<String>,
        confidence: f64,
        reasoning: impl Into<String>,
    ) -> Self {
        AgentVote {
            agent_id: agent_id.into(),
            model_name: model_name.into(),
            position: position.into(),
            confidence: confidence.clamp(0.0, 1.0),
            reasoning: reasoning.into(),
            is_contrarian: false,
        }
    }

    /// Create a contrarian vote (flagged for diversity).
    pub fn contrarian(
        agent_id: impl Into<String>,
        model_name: impl Into<String>,
        position: impl Into<String>,
        confidence: f64,
        reasoning: impl Into<String>,
    ) -> Self {
        AgentVote {
            agent_id: agent_id.into(),
            model_name: model_name.into(),
            position: position.into(),
            confidence: confidence.clamp(0.0, 1.0),
            reasoning: reasoning.into(),
            is_contrarian: true,
        }
    }
}

// ---------------------------------------------------------------------------
// Solana types
// ---------------------------------------------------------------------------

/// A classified plan for a Solana CLI command.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SolanaCommandPlan {
    pub args: Vec<String>,
    pub cluster: String,
    pub action_type: SolanaActionType,
    pub allowed: bool,
    pub requires_approval: bool,
    pub reason: String,
    pub command_preview: String,
}

impl SolanaCommandPlan {
    /// Create a new Solana command plan.
    pub fn new(
        args: Vec<String>,
        cluster: impl Into<String>,
        action_type: SolanaActionType,
        allowed: bool,
        requires_approval: bool,
        reason: impl Into<String>,
    ) -> Self {
        let command_preview = args.join(" ");
        SolanaCommandPlan {
            args,
            cluster: cluster.into(),
            action_type,
            allowed,
            requires_approval,
            reason: reason.into(),
            command_preview,
        }
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    // -- Domain tests --

    #[test]
    fn test_domain_all_count() {
        assert_eq!(Domain::all().len(), 9);
    }

    #[test]
    fn test_domain_from_str_label() {
        assert_eq!(Domain::from_str_label("financial"), Some(Domain::Financial));
        assert_eq!(Domain::from_str_label("FINANCIAL"), Some(Domain::Financial));
        assert_eq!(Domain::from_str_label("business_intel"), Some(Domain::BusinessIntel));
        assert_eq!(Domain::from_str_label("predictive_sim"), Some(Domain::PredictiveSim));
        assert_eq!(Domain::from_str_label("content_marketing"), Some(Domain::ContentMarketing));
        assert_eq!(Domain::from_str_label("unknown"), None);
    }

    #[test]
    fn test_domain_display() {
        assert_eq!(Domain::Financial.to_string(), "financial");
        assert_eq!(Domain::RealEstate.to_string(), "real_estate");
    }

    #[test]
    fn test_domain_serde_roundtrip() {
        let domain = Domain::Cybersecurity;
        let json = serde_json::to_string(&domain).unwrap();
        let parsed: Domain = serde_json::from_str(&json).unwrap();
        assert_eq!(domain, parsed);
    }

    // -- TaskStatus tests --

    #[test]
    fn test_task_status_all_count() {
        assert_eq!(TaskStatus::all().len(), 7);
    }

    #[test]
    fn test_task_status_transitions() {
        assert!(TaskStatus::Pending.can_transition_to(TaskStatus::Ready));
        assert!(TaskStatus::Ready.can_transition_to(TaskStatus::Active));
        assert!(TaskStatus::Active.can_transition_to(TaskStatus::Complete));
        assert!(TaskStatus::Active.can_transition_to(TaskStatus::Retry));
        assert!(TaskStatus::Retry.can_transition_to(TaskStatus::Ready));
        assert!(TaskStatus::Retry.can_transition_to(TaskStatus::Escalate));
        assert!(TaskStatus::Active.can_transition_to(TaskStatus::Blocked));

        // Invalid transitions
        assert!(!TaskStatus::Pending.can_transition_to(TaskStatus::Complete));
        assert!(!TaskStatus::Complete.can_transition_to(TaskStatus::Active));
        assert!(!TaskStatus::Escalate.can_transition_to(TaskStatus::Ready));
    }

    #[test]
    fn test_task_status_terminal() {
        assert!(TaskStatus::Complete.is_terminal());
        assert!(TaskStatus::Blocked.is_terminal());
        assert!(TaskStatus::Escalate.is_terminal());
        assert!(!TaskStatus::Active.is_terminal());
        assert!(!TaskStatus::Pending.is_terminal());
    }

    #[test]
    fn test_task_status_workable() {
        assert!(TaskStatus::Ready.is_workable());
        assert!(TaskStatus::Active.is_workable());
        assert!(!TaskStatus::Pending.is_workable());
        assert!(!TaskStatus::Complete.is_workable());
    }

    // -- ScentType tests --

    #[test]
    fn test_scent_type_all_count() {
        assert_eq!(ScentType::all().len(), 6);
    }

    #[test]
    fn test_scent_type_urgency_grows() {
        assert!(ScentType::Urgency.is_growing());
        assert!(!ScentType::Completion.is_growing());
        assert!(!ScentType::Difficulty.is_growing());
    }

    #[test]
    fn test_scent_type_default_half_life() {
        assert_eq!(ScentType::Completion.default_half_life(), 300.0);
        assert_eq!(ScentType::Urgency.default_half_life(), -1.0);
        assert_eq!(ScentType::Progress.default_half_life(), 20.0);
    }

    // -- TrustLevel tests --

    #[test]
    fn test_trust_level_ordering() {
        assert!(TrustLevel::Autonomous.restriction_level() < TrustLevel::Supervised.restriction_level());
        assert!(TrustLevel::Supervised.restriction_level() < TrustLevel::ApprovalRequired.restriction_level());
    }

    // -- PolicyAction tests --

    #[test]
    fn test_policy_action_blocked() {
        assert!(PolicyAction::Block.is_blocked());
        assert!(!PolicyAction::Allow.is_blocked());
        assert!(!PolicyAction::RequireApproval.is_blocked());
    }

    #[test]
    fn test_policy_action_requires_approval() {
        assert!(PolicyAction::RequireApproval.requires_approval());
        assert!(!PolicyAction::Allow.requires_approval());
    }

    // -- SolanaActionType tests --

    #[test]
    fn test_solana_action_type_is_write() {
        assert!(SolanaActionType::Write.is_write());
        assert!(SolanaActionType::Deploy.is_write());
        assert!(SolanaActionType::Faucet.is_write());
        assert!(!SolanaActionType::Read.is_write());
        assert!(!SolanaActionType::ConfigRead.is_write());
    }

    #[test]
    fn test_solana_action_type_requires_approval() {
        assert!(SolanaActionType::Deploy.requires_approval());
        assert!(SolanaActionType::Faucet.requires_approval());
        assert!(!SolanaActionType::Read.requires_approval());
        assert!(!SolanaActionType::Write.requires_approval());
    }

    // -- DAGTask tests --

    #[test]
    fn test_dag_task_new() {
        let task = DAGTask::new("t1", "desc", "agent", vec!["a".into()], vec![], 3);
        assert_eq!(task.task_id, "t1");
        assert_eq!(task.status, TaskStatus::Pending);
        assert_eq!(task.retries, 0);
        assert_eq!(task.max_retries, 3);
        assert!(task.worker_id.is_none());
    }

    #[test]
    fn test_dag_task_can_retry() {
        let task = DAGTask::new("t1", "desc", "agent", vec![], vec![], 2);
        assert!(task.can_retry());
        assert!(task.can_retry());
    }

    #[test]
    fn test_task_graph_new() {
        let graph = TaskGraph::new("g1");
        assert!(graph.is_empty());
        assert_eq!(graph.len(), 0);
        assert!(!graph.is_finished());
    }

    #[test]
    fn test_scent_signal_new() {
        let signal = ScentSignal::new("s1", "t1", "w1", ScentType::Urgency, 0.8);
        assert_eq!(signal.signal_id, "s1");
        assert_eq!(signal.intensity, 0.8);
        assert_eq!(signal.scent_type, ScentType::Urgency);
    }

    #[test]
    fn test_agent_vote_new() {
        let vote = AgentVote::new("a1", "gpt-4", "yes", 0.9, "because");
        assert_eq!(vote.confidence, 0.9);
        assert!(!vote.is_contrarian);
    }

    #[test]
    fn test_agent_vote_contrarian() {
        let vote = AgentVote::contrarian("a1", "claude-3", "no", 0.7, "dissent");
        assert!(vote.is_contrarian);
    }

    #[test]
    fn test_agent_vote_confidence_clamped() {
        let vote = AgentVote::new("a1", "m", "pos", 1.5, "r");
        assert_eq!(vote.confidence, 1.0);
        let vote2 = AgentVote::new("a1", "m", "pos", -0.5, "r");
        assert_eq!(vote2.confidence, 0.0);
    }

    #[test]
    fn test_solana_command_plan_new() {
        let plan = SolanaCommandPlan::new(
            vec!["solana".into(), "balance".into()],
            "devnet",
            SolanaActionType::Read,
            true,
            false,
            "Read-only command",
        );
        assert_eq!(plan.command_preview, "solana balance");
        assert!(plan.allowed);
        assert!(!plan.requires_approval);
    }

    #[test]
    fn test_tool_policy_builder() {
        let policy = ToolPolicy::new("browser", TrustLevel::Supervised)
            .with_max_calls(50)
            .with_window(1800)
            .with_min_evidence(2)
            .block_action("delete")
            .require_approval_for("upload");

        assert_eq!(policy.max_calls, 50);
        assert_eq!(policy.window_seconds, 1800);
        assert!(policy.is_action_blocked("delete"));
        assert!(policy.is_approval_required("upload"));
        assert!(!policy.is_action_blocked("read"));
    }

    #[test]
    fn test_policy_decision_new_allowed() {
        let decision = PolicyDecision::new(
            PolicyAction::Allow,
            "within budget",
            "browser",
            "agent1",
            "navigate",
            "p1",
        );
        assert!(decision.is_allowed());
        assert!(!decision.requires_approval);
        assert!(decision.approval_id.is_none());
    }

    #[test]
    fn test_policy_decision_new_blocked() {
        let decision = PolicyDecision::new(
            PolicyAction::Block,
            "kill switch active",
            "shell",
            "agent1",
            "rm -rf",
            "p1",
        );
        assert!(!decision.is_allowed());
        assert!(decision.action.is_blocked());
    }
}
