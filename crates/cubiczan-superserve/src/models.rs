//! Data models for the Superserve.ai API.
//!
//! Contains all request and response structs used by the client, with
//! serde serialisation for JSON interop. All structs use `camelCase`
//! field naming to match the Superserve.ai API convention.

use std::collections::HashMap;

use serde::{Deserialize, Serialize};

// ---------------------------------------------------------------------------
// Sandbox models
// ---------------------------------------------------------------------------

/// Network egress rules attached to a sandbox.
///
/// Controls which external destinations the sandbox can reach.
/// `allow_out` supports CIDRs and domains (including wildcards like `*.example.com`).
/// `deny_out` supports CIDRs only.
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
#[serde(rename_all = "camelCase")]
pub struct NetworkRules {
    /// CIDRs and domain patterns (with wildcards) to allow.
    #[serde(default)]
    pub allow_out: Vec<String>,
    /// CIDRs to explicitly deny.
    #[serde(default)]
    pub deny_out: Vec<String>,
}

impl NetworkRules {
    /// Create a new set of network rules.
    pub fn new(allow_out: Vec<String>, deny_out: Vec<String>) -> Self {
        Self {
            allow_out,
            deny_out,
        }
    }
}

/// Request body for creating a new sandbox.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct CreateSandboxRequest {
    /// Sandbox name (1–64 characters).
    pub name: String,
    /// Optional template ID to base the sandbox on.
    pub template_id: Option<String>,
    /// Maximum lifetime in seconds before automatic deletion (max 604 800 = 7 days).
    pub timeout_seconds: Option<u64>,
    /// Arbitrary key-value metadata (max 64 keys, 256 B key, 2 KB value).
    pub metadata: Option<HashMap<String, String>>,
    /// Environment variables injected into the sandbox.
    pub env_vars: Option<HashMap<String, String>>,
    /// Network egress rules.
    pub network: Option<NetworkRules>,
}

impl CreateSandboxRequest {
    /// Create a minimal sandbox creation request with just a name.
    pub fn new(name: impl Into<String>) -> Self {
        Self {
            name: name.into(),
            template_id: None,
            timeout_seconds: None,
            metadata: None,
            env_vars: None,
            network: None,
        }
    }

    /// Set the template ID.
    pub fn template_id(mut self, id: impl Into<String>) -> Self {
        self.template_id = Some(id.into());
        self
    }

    /// Set the timeout in seconds.
    pub fn timeout_seconds(mut self, secs: u64) -> Self {
        self.timeout_seconds = Some(secs);
        self
    }

    /// Set metadata key-value pairs.
    pub fn metadata(mut self, meta: HashMap<String, String>) -> Self {
        self.metadata = Some(meta);
        self
    }

    /// Set environment variables.
    pub fn env_vars(mut self, env: HashMap<String, String>) -> Self {
        self.env_vars = Some(env);
        self
    }

    /// Set network rules.
    pub fn network(mut self, rules: NetworkRules) -> Self {
        self.network = Some(rules);
        self
    }
}

/// Status of a sandbox.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "lowercase")]
pub enum SandboxStatus {
    /// The sandbox is running and ready to accept commands.
    Active,
    /// The sandbox has been paused (snapshot + suspend).
    Paused,
    /// The sandbox is in the process of resuming.
    Resuming,
    /// The sandbox has failed.
    Failed,
    /// The sandbox has been deleted.
    Deleted,
}

impl std::fmt::Display for SandboxStatus {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Active => write!(f, "active"),
            Self::Paused => write!(f, "paused"),
            Self::Resuming => write!(f, "resuming"),
            Self::Failed => write!(f, "failed"),
            Self::Deleted => write!(f, "deleted"),
        }
    }
}

/// Detailed information about a sandbox returned by the API.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct SandboxInfo {
    /// Unique identifier (UUID).
    pub id: String,
    /// Human-readable sandbox name.
    pub name: String,
    /// Current status of the sandbox.
    pub status: SandboxStatus,
    /// ISO 8601 creation timestamp.
    pub created_at: String,
    /// Template ID if the sandbox was created from a template.
    pub template_id: Option<String>,
    /// Short-lived bearer token used to authenticate against the sandbox.
    pub access_token: Option<String>,
    /// Timeout in seconds (automatic deletion deadline).
    pub timeout_seconds: Option<u64>,
    /// User-defined metadata.
    #[serde(default)]
    pub metadata: HashMap<String, String>,
    /// Environment variables visible inside the sandbox.
    #[serde(default)]
    pub env_vars: HashMap<String, String>,
}

/// Request body for partially updating a sandbox.
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
#[serde(rename_all = "camelCase")]
pub struct UpdateSandboxRequest {
    /// Updated network rules.
    pub network: Option<NetworkRules>,
    /// Updated metadata (merged with existing).
    pub metadata: Option<HashMap<String, String>>,
}

impl UpdateSandboxRequest {
    /// Create a new update request (all fields optional).
    pub fn new() -> Self {
        Self::default()
    }

    /// Set network rules.
    pub fn network(mut self, rules: NetworkRules) -> Self {
        self.network = Some(rules);
        self
    }

    /// Set metadata.
    pub fn metadata(mut self, meta: HashMap<String, String>) -> Self {
        self.metadata = Some(meta);
        self
    }
}

// ---------------------------------------------------------------------------
// Exec models
// ---------------------------------------------------------------------------

/// Request body for executing a command inside a sandbox.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct ExecRequest {
    /// The command to run (e.g., `"python3"` or `"/bin/sh"`).
    pub command: String,
    /// Arguments to pass to the command.
    pub args: Option<Vec<String>>,
    /// Additional environment variables for this execution.
    pub env: Option<HashMap<String, String>>,
    /// Working directory inside the sandbox (default `/home/user`).
    pub working_dir: Option<String>,
    /// Maximum execution time in seconds (default 30).
    pub timeout_s: Option<u64>,
}

impl ExecRequest {
    /// Create a new exec request with a command string.
    pub fn new(command: impl Into<String>) -> Self {
        Self {
            command: command.into(),
            args: None,
            env: None,
            working_dir: None,
            timeout_s: None,
        }
    }

    /// Set command arguments.
    pub fn args(mut self, args: Vec<String>) -> Self {
        self.args = Some(args);
        self
    }

    /// Set environment variables.
    pub fn env(mut self, env: HashMap<String, String>) -> Self {
        self.env = Some(env);
        self
    }

    /// Set working directory.
    pub fn working_dir(mut self, dir: impl Into<String>) -> Self {
        self.working_dir = Some(dir.into());
        self
    }

    /// Set timeout in seconds.
    pub fn timeout_s(mut self, secs: u64) -> Self {
        self.timeout_s = Some(secs);
        self
    }
}

/// Result of a synchronous command execution inside a sandbox.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct ExecResult {
    /// Standard output captured from the command.
    pub stdout: String,
    /// Standard error captured from the command.
    pub stderr: String,
    /// Exit code of the command.
    pub exit_code: i32,
}

// ---------------------------------------------------------------------------
// Template models
// ---------------------------------------------------------------------------

/// Resource constraints for a template.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct TemplateResources {
    /// CPU allocation in millicores (1000–4000).
    pub cpu_millis: u32,
    /// Memory allocation in megabytes (256–4096).
    pub memory_mb: u32,
    /// Disk allocation in megabytes (1024–8192).
    pub disk_mb: u32,
}

impl Default for TemplateResources {
    fn default() -> Self {
        Self {
            cpu_millis: 2000,
            memory_mb: 2048,
            disk_mb: 4096,
        }
    }
}

impl TemplateResources {
    /// Create resource constraints with the given values.
    pub fn new(cpu_millis: u32, memory_mb: u32, disk_mb: u32) -> Self {
        Self {
            cpu_millis,
            memory_mb,
            disk_mb,
        }
    }
}

/// A build step — runs a shell command during template image build.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BuildStep {
    /// Shell command to execute.
    pub run: String,
}

impl BuildStep {
    /// Create a new build step from a shell command.
    pub fn run(cmd: impl Into<String>) -> Self {
        Self { run: cmd.into() }
    }
}

/// An environment-variable step during template build.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BuildEnvStep {
    /// Environment variables to set.
    pub env: HashMap<String, String>,
}

impl BuildEnvStep {
    /// Create a new env step from a map.
    pub fn env(env: HashMap<String, String>) -> Self {
        Self { env }
    }
}

/// A user-configuration step during template build.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BuildUserStep {
    /// Username to create.
    pub name: String,
    /// Whether the user gets sudo access.
    pub sudo: Option<bool>,
}

impl BuildUserStep {
    /// Create a new user step.
    pub fn new(name: impl Into<String>) -> Self {
        Self {
            name: name.into(),
            sudo: None,
        }
    }

    /// Grant or revoke sudo for the user.
    pub fn sudo(mut self, enabled: bool) -> Self {
        self.sudo = Some(enabled);
        self
    }
}

/// Request body for creating a new template.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct CreateTemplateRequest {
    /// Template name (1–128 characters).
    pub name: String,
    /// Base OCI image (default `superserve/base`).
    pub base_image: Option<String>,
    /// Ordered list of build steps (shell commands).
    #[serde(default)]
    pub steps: Vec<BuildStep>,
    /// Environment variable configuration steps.
    pub env: Option<Vec<BuildEnvStep>>,
    /// Default working directory inside the sandbox.
    pub workdir: Option<String>,
    /// User configuration.
    pub user: Option<BuildUserStep>,
    /// Command executed on sandbox start.
    pub start_cmd: Option<String>,
    /// Command used to determine sandbox readiness.
    pub ready_cmd: Option<String>,
    /// Resource constraints.
    pub resources: Option<TemplateResources>,
}

impl CreateTemplateRequest {
    /// Create a minimal template request with a name.
    pub fn new(name: impl Into<String>) -> Self {
        Self {
            name: name.into(),
            base_image: None,
            steps: Vec::new(),
            env: None,
            workdir: None,
            user: None,
            start_cmd: None,
            ready_cmd: None,
            resources: None,
        }
    }

    /// Set the base OCI image.
    pub fn base_image(mut self, image: impl Into<String>) -> Self {
        self.base_image = Some(image.into());
        self
    }

    /// Add a build step.
    pub fn step(mut self, step: BuildStep) -> Self {
        self.steps.push(step);
        self
    }

    /// Add a shell-command build step (convenience shorthand).
    pub fn add_run(mut self, cmd: impl Into<String>) -> Self {
        self.steps.push(BuildStep::run(cmd));
        self
    }

    /// Set environment variable steps.
    pub fn env(mut self, env: Vec<BuildEnvStep>) -> Self {
        self.env = Some(env);
        self
    }

    /// Set the working directory.
    pub fn workdir(mut self, dir: impl Into<String>) -> Self {
        self.workdir = Some(dir.into());
        self
    }

    /// Set the user configuration.
    pub fn user(mut self, user: BuildUserStep) -> Self {
        self.user = Some(user);
        self
    }

    /// Set the start command.
    pub fn start_cmd(mut self, cmd: impl Into<String>) -> Self {
        self.start_cmd = Some(cmd.into());
        self
    }

    /// Set the readiness probe command.
    pub fn ready_cmd(mut self, cmd: impl Into<String>) -> Self {
        self.ready_cmd = Some(cmd.into());
        self
    }

    /// Set resource constraints.
    pub fn resources(mut self, res: TemplateResources) -> Self {
        self.resources = Some(res);
        self
    }
}

/// Status of a template.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "lowercase")]
pub enum TemplateStatus {
    /// Template image is ready to be used for sandboxes.
    Ready,
    /// Template is in the process of being created.
    Pending,
    /// Template image is currently being built.
    Building,
    /// Template build failed.
    Failed,
    /// Template build was cancelled.
    Cancelled,
}

impl std::fmt::Display for TemplateStatus {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Ready => write!(f, "ready"),
            Self::Pending => write!(f, "pending"),
            Self::Building => write!(f, "building"),
            Self::Failed => write!(f, "failed"),
            Self::Cancelled => write!(f, "cancelled"),
        }
    }
}

/// Detailed information about a template.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct TemplateInfo {
    /// Unique template identifier.
    pub id: String,
    /// Human-readable template name.
    pub name: String,
    /// Base OCI image.
    pub base_image: String,
    /// Current template status.
    pub status: TemplateStatus,
    /// Resource constraints for sandboxes created from this template.
    pub resources: TemplateResources,
    /// ISO 8601 creation timestamp.
    pub created_at: String,
}

/// Status of a template build.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "lowercase")]
pub enum BuildStatus {
    /// Build is queued.
    Pending,
    /// Build is in progress.
    Building,
    /// Build is creating a snapshot.
    Snapshotting,
    /// Build completed successfully.
    Ready,
    /// Build failed.
    Failed,
    /// Build was cancelled.
    Cancelled,
}

impl std::fmt::Display for BuildStatus {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Pending => write!(f, "pending"),
            Self::Building => write!(f, "building"),
            Self::Snapshotting => write!(f, "snapshotting"),
            Self::Ready => write!(f, "ready"),
            Self::Failed => write!(f, "failed"),
            Self::Cancelled => write!(f, "cancelled"),
        }
    }
}

/// Detailed information about a template build.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct BuildInfo {
    /// Unique build identifier.
    pub id: String,
    /// ID of the template this build belongs to.
    pub template_id: String,
    /// Current build status.
    pub status: BuildStatus,
    /// ISO 8601 creation timestamp.
    pub created_at: String,
}

// ---------------------------------------------------------------------------
// Health / misc
// ---------------------------------------------------------------------------

/// Response from the health-check endpoint.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct HealthResponse {
    /// Whether the service is healthy.
    pub ok: bool,
}

/// A single SSE event emitted by an exec/stream or build log stream.
#[derive(Debug, Clone)]
pub struct SseEvent {
    /// The event type (e.g. `"stdout"`, `"stderr"`, `"done"`).
    pub event: String,
    /// The event data payload.
    pub data: String,
}
