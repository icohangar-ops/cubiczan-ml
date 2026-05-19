//! Superserve sandbox bridge for CHP third-party validation.
//!
//! When the `superserve` feature is enabled, this module provides
//! `SuperserveBridge` — a thin adapter that executes CHP proposal
//! validations inside Superserve persistent sandboxes and converts
//! the results into CHP-native `ThirdPartyValidation` records.
//!
//! # Usage
//!
//! ```ignore
//! use chp::superserve_bridge::SuperserveBridge;
//!
//! #[tokio::main]
//! async fn main() {
//!     let bridge = SuperserveBridge::new("ss_live_...");
//!     let validation = bridge.validate_proposal(
//!         "chp-validator",
//!         "cargo test --package my-proposal",
//!         "decision-x: implement hedging strategy",
//!     ).await.unwrap();
//!
//!     // Feed into CHP orchestrator
//!     orchestrator.apply_validation("dc-001", validation).await.unwrap();
//! }
//! ```

use crate::models::{ThirdPartyValidation, ValidationResult};

/// Bridge between Superserve sandboxes and CHP validation.
///
/// Wraps a `cubiczan_superserve::SuperserveClient` and exposes
/// CHP-oriented methods that return `ThirdPartyValidation` records.
pub struct SuperserveBridge {
    client: cubiczan_superserve::SuperserveClient,
}

impl SuperserveBridge {
    /// Create a new bridge from a Superserve API key.
    pub fn new(api_key: &str) -> Self {
        Self {
            client: cubiczan_superserve::SuperserveClient::new(api_key),
        }
    }

    /// Create a bridge from the `SUPERSERVE_API_KEY` environment variable.
    pub fn from_env() -> Result<Self, String> {
        let key = std::env::var("SUPERSERVE_API_KEY")
            .map_err(|_| String::from("SUPERSERVE_API_KEY not set"))?;
        Ok(Self::new(&key))
    }

    /// Validate a CHP proposal by executing a command in a Superserve sandbox.
    ///
    /// Creates an ephemeral sandbox, runs the command, captures the result,
    /// deletes the sandbox, and returns a `ThirdPartyValidation` record
    /// compatible with CHP's validation pipeline.
    ///
    /// # Arguments
    /// * `template_id` — Superserve template to use (e.g., `"chp-validator"`)
    /// * `command` — Shell command to execute inside the sandbox
    /// * `proposal_item` — The CHP decision item being validated
    ///
    /// # Returns
    /// A `ThirdPartyValidation` record ready for `CHPOrchestrator::apply_validation()`.
    pub async fn validate_proposal(
        &self,
        template_id: &str,
        command: &str,
        proposal_item: &str,
    ) -> Result<ThirdPartyValidation, String> {
        use cubiczan_superserve::chp::{ChpValidationConfig, run_chp_validation};

        let config = ChpValidationConfig::new(
            "chp-validation",
            template_id,
            command,
            proposal_item,
        )
        .timeout_secs(300)
        .working_dir("/home/user");

        let result = run_chp_validation(&self.client, &config)
            .await
            .map_err(|e| format!("Superserve validation failed: {}", e))?;

        let rationale = if result.stdout.len() > 200 {
            format!(
                "exit_code={}, stdout_truncated={}",
                result.exit_code,
                &result.stdout[..200]
            )
        } else {
            format!(
                "exit_code={}, stdout={}",
                result.exit_code, result.stdout
            )
        };

        Ok(ThirdPartyValidation {
            validator: format!("superserve:{}", result.sandbox_id),
            item: proposal_item.to_string(),
            challenge: String::from("sandbox execution"),
            result: if result.passed {
                ValidationResult::CONFIRM
            } else {
                ValidationResult::REJECT
            },
            rationale,
        })
    }

    /// Run a validation with full result details (for audit logging).
    ///
    /// Returns both the CHP-native `ThirdPartyValidation` and the raw
    /// `ChpValidationResult` for detailed audit trails.
    pub async fn validate_proposal_with_details(
        &self,
        template_id: &str,
        command: &str,
        proposal_item: &str,
    ) -> Result<(ThirdPartyValidation, cubiczan_superserve::chp::ChpValidationResult), String> {
        use cubiczan_superserve::chp::{ChpValidationConfig, run_chp_validation};

        let config = ChpValidationConfig::new(
            "chp-validation",
            template_id,
            command,
            proposal_item,
        )
        .timeout_secs(300)
        .working_dir("/home/user");

        let result = run_chp_validation(&self.client, &config)
            .await
            .map_err(|e| format!("Superserve validation failed: {}", e))?;

        let passed = result.passed;
        let validation = ThirdPartyValidation {
            validator: format!("superserve:{}", result.sandbox_id),
            item: proposal_item.to_string(),
            challenge: String::from("sandbox execution"),
            result: if passed {
                ValidationResult::CONFIRM
            } else {
                ValidationResult::REJECT
            },
            rationale: format!(
                "exit_code={}, stdout={}",
                result.exit_code, result.stdout
            ),
        };

        Ok((validation, result))
    }

    /// Get a reference to the underlying Superserve client
    /// for advanced operations (sandbox lifecycle management, templates, etc.).
    pub fn client(&self) -> &cubiczan_superserve::SuperserveClient {
        &self.client
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_bridge_new() {
        let bridge = SuperserveBridge::new("test_key");
        // Verify bridge was created (client internally constructed)
        let _ = &bridge;
    }

    #[test]
    fn test_bridge_from_env_missing() {
        // Ensure no SUPERSERVE_API_KEY is set for this test
        std::env::remove_var("SUPERSERVE_API_KEY");
        let result = SuperserveBridge::from_env();
        assert!(result.is_err());
        assert!(result.unwrap_err().contains("SUPERSERVE_API_KEY"));
    }
}
