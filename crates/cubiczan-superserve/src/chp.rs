//! CHP integration: use Superserve sandboxes as third-party validators.
//!
//! This module provides a bridge between the Superserve sandbox platform
//! and the Consensus Hardening Protocol (CHP). It enables ephemeral sandbox
//! creation, command execution, and result capture for validating CHP
//! decision items.
//!
//! # Quick Start
//!
//! ```no_run
//! use cubiczan_superserve::SuperserveClient;
//! use cubiczan_superserve::chp::{ChpValidationConfig, run_chp_validation};
//!
//! #[tokio::main]
//! async fn main() -> Result<(), Box<dyn std::error::Error>> {
//!     let client = SuperserveClient::from_env()?;
//!
//!     let config = ChpValidationConfig {
//!         name_prefix: "chp-val".to_string(),
//!         template_id: "tpl_abc123".to_string(),
//!         command: "cargo test --locked".to_string(),
//!         working_dir: None,
//!         timeout_secs: 120,
//!         proposal_item: "foundation-lock-item-1".to_string(),
//!         metadata: std::collections::HashMap::new(),
//!     };
//!
//!     let result = run_chp_validation(&client, &config).await?;
//!     println!("passed: {}", result.passed);
//!
//!     // Convert to CHP-compatible JSON
//!     let json = result.to_chp_json();
//!     println!("{}", serde_json::to_string_pretty(&json)?);
//!
//!     Ok(())
//! }
//! ```

use std::collections::HashMap;

use crate::client::SuperserveClient;
use crate::error::SuperserveError;
use crate::models::{CreateSandboxRequest, ExecRequest};

/// Configuration for a CHP validation sandbox run.
///
/// Specifies the template to use, the command to execute, and the CHP
/// context (proposal item) being validated.
#[derive(Debug, Clone)]
pub struct ChpValidationConfig {
    /// Sandbox name prefix (will be appended with timestamp).
    pub name_prefix: String,
    /// Template ID to use (e.g., from `templates::CHP_VALIDATOR`).
    pub template_id: String,
    /// Shell command to execute inside the sandbox.
    pub command: String,
    /// Working directory inside the sandbox.
    pub working_dir: Option<String>,
    /// Timeout in seconds for command execution.
    pub timeout_secs: u64,
    /// The CHP decision item being validated.
    pub proposal_item: String,
    /// Metadata to attach to the sandbox.
    pub metadata: HashMap<String, String>,
}

impl ChpValidationConfig {
    /// Create a new CHP validation config with the minimum required fields.
    pub fn new(
        name_prefix: impl Into<String>,
        template_id: impl Into<String>,
        command: impl Into<String>,
        proposal_item: impl Into<String>,
    ) -> Self {
        Self {
            name_prefix: name_prefix.into(),
            template_id: template_id.into(),
            command: command.into(),
            working_dir: None,
            timeout_secs: 120,
            proposal_item: proposal_item.into(),
            metadata: HashMap::new(),
        }
    }

    /// Set the working directory inside the sandbox.
    pub fn working_dir(mut self, dir: impl Into<String>) -> Self {
        self.working_dir = Some(dir.into());
        self
    }

    /// Set the timeout in seconds for command execution.
    pub fn timeout_secs(mut self, secs: u64) -> Self {
        self.timeout_secs = secs;
        self
    }

    /// Set the metadata to attach to the sandbox.
    pub fn metadata(mut self, meta: HashMap<String, String>) -> Self {
        self.metadata = meta;
        self
    }
}

/// Result of a CHP validation run, ready for CHP consumption.
///
/// Contains the sandbox execution results plus audit trail information,
/// and can be converted to a `serde_json::Value` compatible with CHP's
/// [`SuperServeValidation`](consensus_hardening_protocol::models::SuperServeValidation).
#[derive(Debug, Clone)]
pub struct ChpValidationResult {
    /// The sandbox ID used (for audit trail).
    pub sandbox_id: String,
    /// The proposal item validated.
    pub proposal_item: String,
    /// Whether the validation passed (exit_code == 0).
    pub passed: bool,
    /// Exit code from the command.
    pub exit_code: i32,
    /// stdout from the command (full output).
    pub stdout: String,
    /// stderr from the command.
    pub stderr: String,
}

impl ChpValidationResult {
    /// Convert to a `serde_json::Value` compatible with CHP's `SuperServeValidation`.
    ///
    /// The `stdout` field is truncated to 200 characters for audit logging,
    /// matching the convention used in the CHP crate's rationale format.
    pub fn to_chp_json(&self) -> serde_json::Value {
        serde_json::json!({
            "sandbox_id": self.sandbox_id,
            "proposal_item": self.proposal_item,
            "exit_code": self.exit_code,
            "stdout": &self.stdout[..self.stdout.len().min(200)],
            "passed": self.passed,
        })
    }
}

/// Run a CHP validation in a Superserve sandbox.
///
/// This is the primary entry point for CHP integration. It performs the
/// full lifecycle of an ephemeral validation run:
///
/// 1. Creates a sandbox from the configured template
/// 2. Executes the validation command inside the sandbox
/// 3. Captures the execution result (stdout, stderr, exit code)
/// 4. Deletes the sandbox (best-effort; failure is logged but does not
///    fail the overall operation)
/// 5. Returns a [`ChpValidationResult`] for CHP consumption
///
/// # Errors
///
/// Returns an error if sandbox creation or command execution fails.
/// Sandbox deletion errors are silently logged (via `eprintln!`) to
/// avoid masking the actual validation result.
pub async fn run_chp_validation(
    client: &SuperserveClient,
    config: &ChpValidationConfig,
) -> Result<ChpValidationResult, SuperserveError> {
    // 1. Create sandbox with timestamped name
    let sandbox_name = format!(
        "{}-{}",
        config.name_prefix,
        chrono::Utc::now().timestamp()
    );

    let mut create_req = CreateSandboxRequest::new(&sandbox_name)
        .template_id(&config.template_id)
        .timeout_seconds(config.timeout_secs);

    if !config.metadata.is_empty() {
        create_req = create_req.metadata(config.metadata.clone());
    }

    let sandbox = client.create_sandbox(&create_req).await?;
    let sandbox_id = sandbox.id.clone();

    // 2. Execute command
    let mut exec_req = ExecRequest::new(&config.command)
        .timeout_s(config.timeout_secs);

    if let Some(ref dir) = config.working_dir {
        exec_req = exec_req.working_dir(dir);
    }

    let exec_result = client.exec(&sandbox_id, &exec_req).await?;

    // 3. Build result
    let result = ChpValidationResult {
        sandbox_id,
        proposal_item: config.proposal_item.clone(),
        passed: exec_result.exit_code == 0,
        exit_code: exec_result.exit_code,
        stdout: exec_result.stdout,
        stderr: exec_result.stderr,
    };

    // 4. Delete sandbox (best-effort)
    match client.delete_sandbox(&result.sandbox_id).await {
        Ok(_) => {}
        Err(e) => {
            eprintln!(
                "warning: failed to delete sandbox {} after CHP validation: {}",
                result.sandbox_id, e
            );
        }
    }

    // 5. Return result
    Ok(result)
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    fn make_result(
        sandbox_id: &str,
        proposal_item: &str,
        exit_code: i32,
        stdout: &str,
        stderr: &str,
    ) -> ChpValidationResult {
        ChpValidationResult {
            sandbox_id: sandbox_id.to_string(),
            proposal_item: proposal_item.to_string(),
            passed: exit_code == 0,
            exit_code,
            stdout: stdout.to_string(),
            stderr: stderr.to_string(),
        }
    }

    #[test]
    fn test_to_chp_json_passed() {
        let result = make_result("sb-001", "item-1", 0, "all tests passed", "");
        let json = result.to_chp_json();

        assert_eq!(json["sandbox_id"], "sb-001");
        assert_eq!(json["proposal_item"], "item-1");
        assert_eq!(json["exit_code"], 0);
        assert_eq!(json["passed"], true);
        assert_eq!(json["stdout"], "all tests passed");
    }

    #[test]
    fn test_to_chp_json_failed() {
        let result = make_result("sb-002", "item-2", 1, "test failed", "stack trace");
        let json = result.to_chp_json();

        assert_eq!(json["sandbox_id"], "sb-002");
        assert_eq!(json["exit_code"], 1);
        assert_eq!(json["passed"], false);
        // stderr is not included in the JSON output
        assert!(json.get("stderr").is_none());
    }

    #[test]
    fn test_to_chp_json_stdout_truncated() {
        let long_stdout = "X".repeat(500);
        let result = make_result("sb-003", "item-3", 0, &long_stdout, "");
        let json = result.to_chp_json();

        let stdout_str = json["stdout"].as_str().unwrap();
        assert_eq!(stdout_str.len(), 200);
        assert!(stdout_str.chars().all(|c| c == 'X'));
    }

    #[test]
    fn test_to_chp_json_stdout_short() {
        let short_stdout = "ok";
        let result = make_result("sb-004", "item-4", 0, short_stdout, "");
        let json = result.to_chp_json();

        assert_eq!(json["stdout"], "ok");
    }

    #[test]
    fn test_to_chp_json_empty_stdout() {
        let result = make_result("sb-005", "item-5", 0, "", "error output");
        let json = result.to_chp_json();

        assert_eq!(json["stdout"], "");
        assert_eq!(json["passed"], true);
    }

    #[test]
    fn test_chp_validation_config_new() {
        let config = ChpValidationConfig::new(
            "chp-val",
            "tpl_abc",
            "cargo test",
            "foundation-item-1",
        );
        assert_eq!(config.name_prefix, "chp-val");
        assert_eq!(config.template_id, "tpl_abc");
        assert_eq!(config.command, "cargo test");
        assert_eq!(config.proposal_item, "foundation-item-1");
        assert!(config.working_dir.is_none());
        assert_eq!(config.timeout_secs, 120);
        assert!(config.metadata.is_empty());
    }

    #[test]
    fn test_chp_validation_config_builder() {
        let mut meta = HashMap::new();
        meta.insert("source".to_string(), "chp".to_string());

        let config = ChpValidationConfig::new("chp", "tpl", "test", "item")
            .working_dir("/home/user/project")
            .timeout_secs(300)
            .metadata(meta);

        assert_eq!(config.working_dir.as_deref(), Some("/home/user/project"));
        assert_eq!(config.timeout_secs, 300);
        assert_eq!(config.metadata.get("source").unwrap(), "chp");
    }

    #[test]
    fn test_to_chp_json_is_valid_json() {
        let result = make_result("sb-006", "item-6", 0, "output", "");
        let json = result.to_chp_json();

        // Verify it serializes and deserializes cleanly
        let json_str = serde_json::to_string(&json).unwrap();
        let parsed: serde_json::Value = serde_json::from_str(&json_str).unwrap();
        assert_eq!(parsed["sandbox_id"], "sb-006");
        assert_eq!(parsed["proposal_item"], "item-6");
    }
}
