//! Error types for the Superserve.ai API client.
//!
//! Provides typed errors for API interactions including authentication
//! failures, rate limiting, resource not found, and network issues.

use thiserror::Error;

/// Errors that can occur when interacting with the Superserve.ai API.
#[derive(Error, Debug)]
pub enum SuperserveError {
    /// Authentication failed — invalid or missing API key.
    #[error("authentication failed: {detail}")]
    Authentication {
        /// Human-readable error detail from the API.
        detail: String,
    },

    /// The requested resource was not found (e.g., invalid sandbox or template ID).
    #[error("resource not found: {resource}")]
    NotFound {
        /// The type of resource that was not found.
        resource: String,
    },

    /// The client is being rate limited by the API.
    #[error("rate limited: retry after {retry_after_secs}s")]
    RateLimited {
        /// Suggested number of seconds to wait before retrying.
        retry_after_secs: u64,
    },

    /// The API returned a non-success status code with an error body.
    #[error("API error (HTTP {status}): {detail}")]
    Api {
        /// HTTP status code returned by the API.
        status: u16,
        /// Human-readable error detail from the API response body.
        detail: String,
    },

    /// A network or transport-level error occurred while communicating with the API.
    #[error("request failed: {0}")]
    Request(#[from] reqwest::Error),

    /// Failed to parse or serialize JSON data.
    #[error("JSON error: {0}")]
    Json(#[from] serde_json::Error),
}

impl SuperserveError {
    /// Create an authentication error from a detail message.
    pub fn auth(detail: impl Into<String>) -> Self {
        Self::Authentication {
            detail: detail.into(),
        }
    }

    /// Create a "not found" error for a resource.
    pub fn not_found(resource: impl Into<String>) -> Self {
        Self::NotFound {
            resource: resource.into(),
        }
    }

    /// Create a rate-limit error with a suggested retry duration.
    pub fn rate_limited(retry_after_secs: u64) -> Self {
        Self::RateLimited {
            retry_after_secs,
        }
    }

    /// Create a generic API error from a status code and detail message.
    pub fn api(status: u16, detail: impl Into<String>) -> Self {
        Self::Api {
            status,
            detail: detail.into(),
        }
    }
}

/// A minimal error payload returned by the Superserve.ai API on failure.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct ApiErrorResponse {
    /// Error message from the API.
    pub error: Option<String>,
    /// Optional detail providing additional context.
    pub detail: Option<String>,
}

impl SuperserveError {
    /// Attempt to construct a typed error from an HTTP response.
    ///
    /// Inspects the status code and, if possible, deserialises the response
    /// body as [`ApiErrorResponse`].
    pub async fn from_response(response: reqwest::Response) -> Self {
        let status = response.status().as_u16();
        let body = response.text().await.unwrap_or_default();

        let detail = serde_json::from_str::<ApiErrorResponse>(&body)
            .ok()
            .and_then(|r| r.detail.or(r.error))
            .unwrap_or_else(|| body.clone());

        match status {
            401 => Self::auth(detail),
            403 => Self::auth(detail),
            404 => Self::not_found(detail),
            429 => {
                // Best-effort parse of Retry-After header.
                Self::RateLimited {
                    retry_after_secs: 1,
                }
            }
            _ => Self::api(status, detail),
        }
    }
}
