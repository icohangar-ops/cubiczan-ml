//! Typed async client for the [Superserve.ai](https://superserve.ai) persistent sandbox API.
//!
//! # Quick Start
//!
//! ```no_run
//! use cubiczan_superserve::SuperserveClient;
//!
//! #[tokio::main]
//! async fn main() -> Result<(), Box<dyn std::error::Error>> {
//!     // Read API key from environment (SUPERSERVE_API_KEY)
//!     let client = SuperserveClient::from_env()?;
//!
//!     // Health check
//!     let health = client.health().await?;
//!     println!("API healthy: {}", health.ok);
//!
//!     // Create a sandbox
//!     let sandbox = client
//!         .create_sandbox(&cubiczan_superserve::CreateSandboxRequest::new("my-sandbox"))
//!         .await?;
//!     println!("Sandbox ID: {}", sandbox.id);
//!
//!     // Execute a command
//!     let result = client
//!         .exec(&sandbox.id, &cubiczan_superserve::ExecRequest::new("uname -a"))
//!         .await?;
//!     println!("stdout: {}", result.stdout);
//!
//!     Ok(())
//! }
//! ```

use std::collections::HashMap;
use std::env;
use std::pin::Pin;
use std::task::{Context, Poll};

use futures::Stream;
use reqwest::header::{HeaderMap, HeaderValue};
use tracing::debug;

use crate::error::SuperserveError;
use crate::models::*;

/// Default base URL for the Superserve.ai API.
pub const DEFAULT_BASE_URL: &str = "https://api.superserve.ai";

/// Environment variable name for the API key.
pub const API_KEY_ENV: &str = "SUPERSERVE_API_KEY";

/// Typed async client for the Superserve.ai API.
///
/// All methods return futures that resolve to strongly-typed models
/// or [`SuperserveError`] on failure.
#[derive(Debug, Clone)]
pub struct SuperserveClient {
    /// HTTP client (reused across requests for connection pooling).
    http: reqwest::Client,
    /// Base URL for all API requests.
    base_url: String,
}

impl SuperserveClient {
    // -----------------------------------------------------------------------
    // Constructors
    // -----------------------------------------------------------------------

    /// Create a new client with the given API key and the default base URL.
    ///
    /// # Example
    ///
    /// ```no_run
    /// use cubiczan_superserve::SuperserveClient;
    /// let client = SuperserveClient::new("ss_live_...");
    /// ```
    pub fn new(api_key: impl Into<String>) -> Self {
        Self::with_base_url(api_key, DEFAULT_BASE_URL)
    }

    /// Create a new client with a custom base URL (useful for testing or proxies).
    pub fn with_base_url(api_key: impl Into<String>, base_url: &str) -> Self {
        let mut headers = HeaderMap::new();
        headers.insert(
            "X-API-Key",
            HeaderValue::from_str(&api_key.into()).expect("API key must be valid header value"),
        );

        let http = reqwest::Client::builder()
            .default_headers(headers)
            .build()
            .expect("failed to build reqwest Client");

        Self {
            http,
            base_url: base_url.trim_end_matches('/').to_string(),
        }
    }

    /// Create a client by reading `SUPERSERVE_API_KEY` from the environment.
    ///
    /// # Errors
    ///
    /// Returns an error if the environment variable is not set.
    pub fn from_env() -> Result<Self, SuperserveError> {
        let key = env::var(API_KEY_ENV).map_err(|_| {
            SuperserveError::Authentication {
                detail: format!(
                    "environment variable {} is not set",
                    API_KEY_ENV
                ),
            }
        })?;
        Ok(Self::new(key))
    }

    /// Create a client by reading `SUPERSERVE_API_KEY` from the environment,
    /// with a custom base URL.
    pub fn from_env_with_base_url(base_url: &str) -> Result<Self, SuperserveError> {
        let key = env::var(API_KEY_ENV).map_err(|_| {
            SuperserveError::Authentication {
                detail: format!(
                    "environment variable {} is not set",
                    API_KEY_ENV
                ),
            }
        })?;
        Ok(Self::with_base_url(key, base_url))
    }

    /// Return a reference to the configured base URL.
    pub fn base_url(&self) -> &str {
        &self.base_url
    }

    // -----------------------------------------------------------------------
    // Helpers
    // -----------------------------------------------------------------------

    /// Build a full URL for the given path.
    fn url(&self, path: &str) -> String {
        format!("{}{}", self.base_url, path)
    }

    /// Send a GET request and deserialise the JSON response.
    async fn get<T: serde::de::DeserializeOwned>(&self, path: &str) -> Result<T, SuperserveError> {
        let url = self.url(path);
        debug!(%url, "GET");
        let resp = self.http.get(&url).send().await?;
        if !resp.status().is_success() {
            return Err(SuperserveError::from_response(resp).await);
        }
        resp.json::<T>().await.map_err(SuperserveError::from)
    }

    /// Send a POST request with a JSON body and deserialise the response.
    async fn post<T: serde::de::DeserializeOwned, B: serde::Serialize>(
        &self,
        path: &str,
        body: &B,
    ) -> Result<T, SuperserveError> {
        let url = self.url(path);
        debug!(%url, "POST");
        let resp = self.http.post(&url).json(body).send().await?;
        if !resp.status().is_success() {
            return Err(SuperserveError::from_response(resp).await);
        }
        resp.json::<T>().await.map_err(SuperserveError::from)
    }

    /// Send a PATCH request with a JSON body and deserialise the response.
    async fn patch<T: serde::de::DeserializeOwned, B: serde::Serialize>(
        &self,
        path: &str,
        body: &B,
    ) -> Result<T, SuperserveError> {
        let url = self.url(path);
        debug!(%url, "PATCH");
        let resp = self.http.patch(&url).json(body).send().await?;
        if !resp.status().is_success() {
            return Err(SuperserveError::from_response(resp).await);
        }
        resp.json::<T>().await.map_err(SuperserveError::from)
    }

    /// Send a DELETE request and deserialise the response.
    async fn delete<T: serde::de::DeserializeOwned>(&self, path: &str) -> Result<T, SuperserveError> {
        let url = self.url(path);
        debug!(%url, "DELETE");
        let resp = self.http.delete(&url).send().await?;
        if !resp.status().is_success() {
            return Err(SuperserveError::from_response(resp).await);
        }
        resp.json::<T>().await.map_err(SuperserveError::from)
    }

    /// Send a POST request with a JSON body and return the raw response for
    /// streaming purposes.
    async fn post_stream<B: serde::Serialize>(
        &self,
        path: &str,
        body: &B,
    ) -> Result<reqwest::Response, SuperserveError> {
        let url = self.url(path);
        debug!(%url, "POST (stream)");
        let resp = self.http.post(&url).json(body).send().await?;
        if !resp.status().is_success() {
            return Err(SuperserveError::from_response(resp).await);
        }
        Ok(resp)
    }

    /// Send a GET request and return the raw response for streaming purposes.
    async fn get_stream(&self, path: &str) -> Result<reqwest::Response, SuperserveError> {
        let url = self.url(path);
        debug!(%url, "GET (stream)");
        let resp = self.http.get(&url).send().await?;
        if !resp.status().is_success() {
            return Err(SuperserveError::from_response(resp).await);
        }
        Ok(resp)
    }

    // -----------------------------------------------------------------------
    // Health
    // -----------------------------------------------------------------------

    /// Check the health of the Superserve.ai API.
    ///
    /// `GET /health`
    pub async fn health(&self) -> Result<HealthResponse, SuperserveError> {
        self.get("/health").await
    }

    // =======================================================================
    // Sandbox endpoints
    // =======================================================================

    /// List all sandboxes, optionally filtered by metadata query parameters.
    ///
    /// `GET /sandboxes?metadata.key=value`
    ///
    /// Pass an empty map to retrieve all sandboxes.
    pub async fn list_sandboxes(
        &self,
        metadata_filter: &HashMap<String, String>,
    ) -> Result<Vec<SandboxInfo>, SuperserveError> {
        let mut path = String::from("/sandboxes");
        if !metadata_filter.is_empty() {
            let qs: Vec<String> = metadata_filter
                .iter()
                .map(|(k, v)| format!("metadata.{}={}", k, v))
                .collect();
            path.push_str(&format!("?{}", qs.join("&")));
        }
        self.get(&path).await
    }

    /// List all sandboxes without any filters.
    ///
    /// Convenience wrapper around [`Self::list_sandboxes`].
    pub async fn list_all_sandboxes(&self) -> Result<Vec<SandboxInfo>, SuperserveError> {
        self.list_sandboxes(&HashMap::new()).await
    }

    /// Create a new sandbox.
    ///
    /// `POST /sandboxes`
    pub async fn create_sandbox(
        &self,
        req: &CreateSandboxRequest,
    ) -> Result<SandboxInfo, SuperserveError> {
        self.post("/sandboxes", req).await
    }

    /// Get details for a specific sandbox.
    ///
    /// `GET /sandboxes/{id}`
    ///
    /// The response includes the `access_token` needed to communicate with
    /// the sandbox directly (e.g., over SSH or HTTP).
    pub async fn get_sandbox(&self, sandbox_id: &str) -> Result<SandboxInfo, SuperserveError> {
        let path = format!("/sandboxes/{}", sandbox_id);
        self.get(&path).await
    }

    /// Partially update a sandbox (network rules, metadata).
    ///
    /// `PATCH /sandboxes/{id}`
    pub async fn update_sandbox(
        &self,
        sandbox_id: &str,
        req: &UpdateSandboxRequest,
    ) -> Result<SandboxInfo, SuperserveError> {
        let path = format!("/sandboxes/{}", sandbox_id);
        self.patch(&path, req).await
    }

    /// Pause a sandbox (snapshot & suspend the VM).
    ///
    /// `POST /sandboxes/{id}/pause`
    ///
    /// A paused sandbox retains its disk state and can be resumed later.
    pub async fn pause_sandbox(&self, sandbox_id: &str) -> Result<SandboxInfo, SuperserveError> {
        let path = format!("/sandboxes/{}/pause", sandbox_id);
        self.post(&path, &serde_json::json!({})).await
    }

    /// Resume a paused sandbox.
    ///
    /// `POST /sandboxes/{id}/resume`
    ///
    /// Returns a **fresh** `access_token` that replaces the previous one.
    pub async fn resume_sandbox(&self, sandbox_id: &str) -> Result<SandboxInfo, SuperserveError> {
        let path = format!("/sandboxes/{}/resume", sandbox_id);
        self.post(&path, &serde_json::json!({})).await
    }

    /// Delete a sandbox permanently.
    ///
    /// `DELETE /sandboxes/{id}`
    pub async fn delete_sandbox(&self, sandbox_id: &str) -> Result<SandboxInfo, SuperserveError> {
        let path = format!("/sandboxes/{}", sandbox_id);
        self.delete(&path).await
    }

    /// Execute a command inside a sandbox (synchronous).
    ///
    /// `POST /sandboxes/{id}/exec`
    ///
    /// Blocks until the command completes and returns the full stdout, stderr,
    /// and exit code.
    pub async fn exec(
        &self,
        sandbox_id: &str,
        req: &ExecRequest,
    ) -> Result<ExecResult, SuperserveError> {
        let path = format!("/sandboxes/{}/exec", sandbox_id);
        self.post(&path, req).await
    }

    /// Execute a command inside a sandbox with SSE streaming output.
    ///
    /// `POST /sandboxes/{id}/exec/stream`
    ///
    /// Returns a stream of [`SseEvent`] values emitted by the server as the
    /// command produces output. The stream ends when the command finishes.
    pub async fn exec_stream(
        &self,
        sandbox_id: &str,
        req: &ExecRequest,
    ) -> Result<SseEventStream, SuperserveError> {
        let path = format!("/sandboxes/{}/exec/stream", sandbox_id);
        let resp = self.post_stream(&path, req).await?;
        Ok(SseEventStream::new(resp))
    }

    // =======================================================================
    // Template endpoints
    // =======================================================================

    /// List all templates.
    ///
    /// `GET /templates`
    pub async fn list_templates(&self) -> Result<Vec<TemplateInfo>, SuperserveError> {
        self.get("/templates").await
    }

    /// Create a new template.
    ///
    /// `POST /templates`
    ///
    /// Template creation triggers an asynchronous build. Use
    /// [`Self::get_template`] or [`Self::list_builds`] to monitor progress.
    pub async fn create_template(
        &self,
        req: &CreateTemplateRequest,
    ) -> Result<TemplateInfo, SuperserveError> {
        self.post("/templates", req).await
    }

    /// Get details for a specific template.
    ///
    /// `GET /templates/{id}`
    pub async fn get_template(&self, template_id: &str) -> Result<TemplateInfo, SuperserveError> {
        let path = format!("/templates/{}", template_id);
        self.get(&path).await
    }

    /// Delete a template.
    ///
    /// `DELETE /templates/{id}`
    pub async fn delete_template(&self, template_id: &str) -> Result<TemplateInfo, SuperserveError> {
        let path = format!("/templates/{}", template_id);
        self.delete(&path).await
    }

    /// List recent builds for a template.
    ///
    /// `GET /templates/{id}/builds`
    pub async fn list_builds(&self, template_id: &str) -> Result<Vec<BuildInfo>, SuperserveError> {
        let path = format!("/templates/{}/builds", template_id);
        self.get(&path).await
    }

    /// Trigger a rebuild of a template.
    ///
    /// `POST /templates/{id}/builds`
    pub async fn rebuild_template(
        &self,
        template_id: &str,
    ) -> Result<BuildInfo, SuperserveError> {
        let path = format!("/templates/{}/builds", template_id);
        self.post(&path, &serde_json::json!({})).await
    }

    /// Get details for a specific template build.
    ///
    /// `GET /templates/{id}/builds/{build_id}`
    pub async fn get_build(
        &self,
        template_id: &str,
        build_id: &str,
    ) -> Result<BuildInfo, SuperserveError> {
        let path = format!("/templates/{}/builds/{}", template_id, build_id);
        self.get(&path).await
    }

    /// Cancel a pending or in-progress template build.
    ///
    /// `DELETE /templates/{id}/builds/{build_id}`
    pub async fn cancel_build(
        &self,
        template_id: &str,
        build_id: &str,
    ) -> Result<BuildInfo, SuperserveError> {
        let path = format!("/templates/{}/builds/{}", template_id, build_id);
        self.delete(&path).await
    }

    /// Stream build logs for a specific template build (SSE).
    ///
    /// `GET /templates/{id}/builds/{build_id}/logs`
    ///
    /// Returns a stream of [`SseEvent`] values as the build progresses.
    pub async fn stream_build_logs(
        &self,
        template_id: &str,
        build_id: &str,
    ) -> Result<SseEventStream, SuperserveError> {
        let path = format!("/templates/{}/builds/{}/logs", template_id, build_id);
        let resp = self.get_stream(&path).await?;
        Ok(SseEventStream::new(resp))
    }
}

// ---------------------------------------------------------------------------
// SSE event stream
// ---------------------------------------------------------------------------

use eventsource_stream::Eventsource;

/// An asynchronous stream of Server-Sent Events (SSE) from the Superserve.ai API.
///
/// Implements [`futures::Stream`] so it can be used with `.next().await`,
/// `while let Some(event) = stream.next().await`, or combinators from the
/// `futures` crate.
pub struct SseEventStream {
    inner: Pin<
        Box<
            dyn Stream<Item = Result<eventsource_stream::Event, eventsource_stream::EventStreamError<reqwest::Error>>> + Send,
        >,
    >,
}

impl SseEventStream {
    fn new(resp: reqwest::Response) -> Self {
        let stream = resp.bytes_stream().eventsource();
        Self {
            inner: Box::pin(stream),
        }
    }
}

impl Stream for SseEventStream {
    type Item = Result<SseEvent, SuperserveError>;

    fn poll_next(mut self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<Option<Self::Item>> {
        match self.inner.as_mut().poll_next(cx) {
            Poll::Ready(Some(Ok(event))) => Poll::Ready(Some(Ok(SseEvent {
                event: event.event,
                data: event.data,
            }))),
            Poll::Ready(Some(Err(e))) => {
                // Map the eventsource-stream error variants into SuperserveError.
                let mapped = match e {
                    eventsource_stream::EventStreamError::Transport(e) => SuperserveError::Request(e),
                    eventsource_stream::EventStreamError::Parser(e) => SuperserveError::Api {
                        status: 0,
                        detail: format!("SSE parse error: {}", e),
                    },
                    eventsource_stream::EventStreamError::Utf8(e) => SuperserveError::Api {
                        status: 0,
                        detail: format!("SSE UTF-8 error: {}", e),
                    },
                };
                Poll::Ready(Some(Err(mapped)))
            }
            Poll::Ready(None) => Poll::Ready(None),
            Poll::Pending => Poll::Pending,
        }
    }
}
