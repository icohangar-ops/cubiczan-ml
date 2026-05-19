//! # cubiczan-superserve
//!
//! Typed Rust client for the [Superserve.ai](https://superserve.ai) persistent sandbox API.
//!
//! This crate provides a fully-featured async client for managing sandboxes and
//! templates on the Superserve.ai platform. It supports command execution (both
//! synchronous and SSE-streaming), template lifecycle management, and build-log
//! streaming.
//!
//! ## Feature overview
//!
//! - **Sandbox CRUD** — create, list, get, update, pause, resume, delete
//! - **Command execution** — sync (`exec`) and SSE streaming (`exec_stream`)
//! - **Template management** — create, list, get, delete templates
//! - **Build management** — trigger rebuilds, monitor builds, stream build logs
//! - **Pre-built templates** — ready-to-use configs for Rust, Python ML, and CHP
//! - **Typed errors** — rich error types with auth, rate-limit, and not-found variants
//!
//! ## Quick start
//!
//! ```no_run
//! use cubiczan_superserve::SuperserveClient;
//! use cubiczan_superserve::CreateSandboxRequest;
//!
//! #[tokio::main]
//! async fn main() -> Result<(), Box<dyn std::error::Error>> {
//!     let client = SuperserveClient::from_env()?;
//!
//!     let health = client.health().await?;
//!     println!("healthy: {}", health.ok);
//!
//!     let sandbox = client
//!         .create_sandbox(&CreateSandboxRequest::new("demo"))
//!         .await?;
//!     println!("sandbox: {} ({})", sandbox.id, sandbox.status);
//!
//!     Ok(())
//! }
//! ```

// Core modules
pub mod chp;
pub mod client;
pub mod error;
pub mod models;
pub mod templates;

// Re-exports — public API surface

/// The main client struct and its constants.
pub use client::{DEFAULT_BASE_URL, API_KEY_ENV, SuperserveClient};

/// All request and response types.
pub use models::{
    BuildEnvStep, BuildInfo, BuildStep, BuildStatus, BuildUserStep, CreateSandboxRequest,
    CreateTemplateRequest, ExecRequest, ExecResult, HealthResponse, NetworkRules, SandboxInfo,
    SandboxStatus, SseEvent, TemplateInfo, TemplateResources, TemplateStatus,
    UpdateSandboxRequest,
};

/// SSE event stream for exec and build log streaming.
pub use client::SseEventStream;

/// Error types.
pub use error::{ApiErrorResponse, SuperserveError};
