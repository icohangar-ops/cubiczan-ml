//! Pre-defined template configurations for common sandbox environments.
//!
//! These builder functions return fully-constructed [`CreateTemplateRequest`]
//! values that can be passed directly to [`SuperserveClient::create_template`].
//!
//! # Available Templates
//!
//! | Constant | Description |
//! |----------|-------------|
//! | [`RUST_CRATE`] | `superserve/base` + cargo, rustfmt, clippy |
//! | [`PYTHON_ML`] | `superserve/python-3.11` + numpy, pandas, scikit-learn |
//! | [`CHP_VALIDATOR`] | `superserve/base` + cargo + rust test deps for CHP |
//!
//! [`SuperserveClient::create_template`]: crate::client::SuperserveClient::create_template

use crate::models::{CreateTemplateRequest, TemplateResources};

// ---------------------------------------------------------------------------
// Pre-built template configs
// ---------------------------------------------------------------------------

/// Rust crate development environment.
///
/// Based on `superserve/base` with the Rust toolchain installed via rustup,
/// including `rustfmt` and `clippy`.
///
/// # Example
///
/// ```no_run
/// use cubiczan_superserve::SuperserveClient;
/// use cubiczan_superserve::templates::RUST_CRATE;
/// # async fn example(client: &SuperserveClient) -> Result<(), Box<dyn std::error::Error>> {
/// let req = RUST_CRATE("my-rust-env".to_string());
/// let template = client.create_template(&req).await?;
/// # Ok(())
/// # }
/// ```
#[allow(non_snake_case)]
pub fn RUST_CRATE(name: String) -> CreateTemplateRequest {
    CreateTemplateRequest::new(&name)
        .base_image("superserve/base")
        .add_run("curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y")
        .add_run("source $HOME/.cargo/env && rustup component add rustfmt clippy")
        .workdir("/home/user/project")
        .resources(TemplateResources::new(2000, 2048, 4096))
}

/// Python ML / data-science environment.
///
/// Based on `superserve/python-3.11` with numpy, pandas, and scikit-learn
/// installed via pip.
///
/// # Example
///
/// ```no_run
/// use cubiczan_superserve::SuperserveClient;
/// use cubiczan_superserve::templates::PYTHON_ML;
/// # async fn example(client: &SuperserveClient) -> Result<(), Box<dyn std::error::Error>> {
/// let req = PYTHON_ML("my-ml-env".to_string());
/// let template = client.create_template(&req).await?;
/// # Ok(())
/// # }
/// ```
#[allow(non_snake_case)]
pub fn PYTHON_ML(name: String) -> CreateTemplateRequest {
    CreateTemplateRequest::new(&name)
        .base_image("superserve/python-3.11")
        .add_run("pip install --no-cache-dir numpy pandas scikit-learn")
        .workdir("/home/user/notebooks")
        .resources(TemplateResources::new(2000, 4096, 8192))
}

/// Consensus Hardening Protocol (CHP) validator environment.
///
/// Based on `superserve/base` with the Rust toolchain and common test
/// dependencies needed for compiling and testing the CHP crate.
///
/// # Example
///
/// ```no_run
/// use cubiczan_superserve::SuperserveClient;
/// use cubiczan_superserve::templates::CHP_VALIDATOR;
/// # async fn example(client: &SuperserveClient) -> Result<(), Box<dyn std::error::Error>> {
/// let req = CHP_VALIDATOR("chp-validator".to_string());
/// let template = client.create_template(&req).await?;
/// # Ok(())
/// # }
/// ```
#[allow(non_snake_case)]
pub fn CHP_VALIDATOR(name: String) -> CreateTemplateRequest {
    CreateTemplateRequest::new(&name)
        .base_image("superserve/base")
        .add_run("curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y")
        .add_run("source $HOME/.cargo/env && rustup component add rustfmt clippy")
        // Install common test / CI dependencies
        .add_run("source $HOME/.cargo/env && cargo install cargo-nextest --locked || true")
        .workdir("/home/user/chp")
        .resources(TemplateResources::new(2000, 2048, 4096))
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn rust_crate_template_is_valid() {
        let req = RUST_CRATE("test-rust".to_string());
        assert_eq!(req.name, "test-rust");
        assert_eq!(req.base_image.as_deref(), Some("superserve/base"));
        assert!(req.steps.len() >= 2);
        assert_eq!(req.workdir.as_deref(), Some("/home/user/project"));
        let res = req.resources.unwrap();
        assert_eq!(res.cpu_millis, 2000);
        assert_eq!(res.memory_mb, 2048);
        assert_eq!(res.disk_mb, 4096);
    }

    #[test]
    fn python_ml_template_is_valid() {
        let req = PYTHON_ML("test-ml".to_string());
        assert_eq!(req.name, "test-ml");
        assert_eq!(req.base_image.as_deref(), Some("superserve/python-3.11"));
        assert!(!req.steps.is_empty());
        assert_eq!(req.workdir.as_deref(), Some("/home/user/notebooks"));
        let res = req.resources.unwrap();
        assert_eq!(res.cpu_millis, 2000);
        assert_eq!(res.memory_mb, 4096);
        assert_eq!(res.disk_mb, 8192);
    }

    #[test]
    fn chp_validator_template_is_valid() {
        let req = CHP_VALIDATOR("chp-val".to_string());
        assert_eq!(req.name, "chp-val");
        assert_eq!(req.base_image.as_deref(), Some("superserve/base"));
        assert!(req.steps.len() >= 3);
        assert_eq!(req.workdir.as_deref(), Some("/home/user/chp"));
        let res = req.resources.unwrap();
        assert_eq!(res.cpu_millis, 2000);
        assert_eq!(res.memory_mb, 2048);
        assert_eq!(res.disk_mb, 4096);
    }

    #[test]
    fn templates_serialize_to_valid_json() {
        let req = RUST_CRATE("serialize-test".to_string());
        let json = serde_json::to_string(&req).expect("serialization should succeed");
        assert!(json.contains("superserve/base"));
        assert!(json.contains("serialize-test"));

        let deserialized: CreateTemplateRequest =
            serde_json::from_str(&json).expect("deserialization should succeed");
        assert_eq!(deserialized.name, req.name);
        assert_eq!(deserialized.steps.len(), req.steps.len());
    }
}
