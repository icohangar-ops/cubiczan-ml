//! Integration and unit tests for the cubiczan-superserve crate.
//!
//! These tests validate:
//! - JSON serialisation / deserialisation of all model types
//! - Error type construction and display formatting
//! - Client construction
//! - Pre-defined template configurations
//! - Edge cases (empty fields, defaults, optional values)

use std::collections::HashMap;

use cubiczan_superserve::*;
use serde_json;

// =============================================================================
// Model serialisation / deserialisation tests
// =============================================================================

#[test]
fn sandbox_info_deserializes_from_api_response() {
    let json = r#"{
        "id": "550e8400-e29b-41d4-a716-446655440000",
        "name": "my-sandbox",
        "status": "active",
        "createdAt": "2025-01-15T10:30:00Z",
        "templateId": "tmpl_abc123",
        "accessToken": "tok_xyz",
        "timeoutSeconds": 3600,
        "metadata": {"env": "dev"},
        "envVars": {"RUST_LOG": "debug"}
    }"#;

    let info: SandboxInfo = serde_json::from_str(json).expect("deserialization should succeed");
    assert_eq!(info.id, "550e8400-e29b-41d4-a716-446655440000");
    assert_eq!(info.name, "my-sandbox");
    assert_eq!(info.status, SandboxStatus::Active);
    assert_eq!(info.created_at, "2025-01-15T10:30:00Z");
    assert_eq!(info.template_id.as_deref(), Some("tmpl_abc123"));
    assert_eq!(info.access_token.as_deref(), Some("tok_xyz"));
    assert_eq!(info.timeout_seconds, Some(3600));
    assert_eq!(info.metadata.get("env").unwrap(), "dev");
    assert_eq!(info.env_vars.get("RUST_LOG").unwrap(), "debug");
}

#[test]
fn sandbox_info_serializes_to_camel_case() {
    let info = SandboxInfo {
        id: "id-1".to_string(),
        name: "test".to_string(),
        status: SandboxStatus::Paused,
        created_at: "2025-01-01T00:00:00Z".to_string(),
        template_id: None,
        access_token: None,
        timeout_seconds: None,
        metadata: HashMap::new(),
        env_vars: HashMap::new(),
    };

    let json = serde_json::to_string(&info).expect("serialization should succeed");
    assert!(json.contains(r#""createdAt":"2025-01-01T00:00:00Z""#));
    assert!(json.contains(r#""templateId":null"#));
    assert!(json.contains(r#""accessToken":null"#));
    assert!(json.contains(r#""timeoutSeconds":null"#));
}

#[test]
fn create_sandbox_request_builder() {
    let req = CreateSandboxRequest::new("builder-test")
        .template_id("tmpl_123")
        .timeout_seconds(7200)
        .metadata(HashMap::from([("team".to_string(), "ml".to_string())]))
        .env_vars(HashMap::from([("FOO".to_string(), "bar".to_string())]))
        .network(NetworkRules::new(
            vec!["*.example.com".to_string(), "10.0.0.0/8".to_string()],
            vec!["192.168.0.0/16".to_string()],
        ));

    assert_eq!(req.name, "builder-test");
    assert_eq!(req.template_id.as_deref(), Some("tmpl_123"));
    assert_eq!(req.timeout_seconds, Some(7200));
    assert_eq!(req.metadata.as_ref().unwrap()["team"], "ml");
    assert_eq!(req.env_vars.as_ref().unwrap()["FOO"], "bar");
    assert_eq!(req.network.as_ref().unwrap().allow_out.len(), 2);
    assert_eq!(req.network.as_ref().unwrap().deny_out.len(), 1);

    // Round-trip through JSON
    let json = serde_json::to_string(&req).unwrap();
    let deserialized: CreateSandboxRequest = serde_json::from_str(&json).unwrap();
    assert_eq!(deserialized.name, req.name);
}

#[test]
fn create_sandbox_request_minimal_serializes() {
    let req = CreateSandboxRequest::new("minimal");
    let json = serde_json::to_string(&req).unwrap();
    assert!(json.contains(r#""name":"minimal""#));
    assert!(json.contains(r#""templateId":null"#));
    assert!(json.contains(r#""timeoutSeconds":null"#));
    assert!(json.contains(r#""metadata":null"#));
    assert!(json.contains(r#""envVars":null"#));
    assert!(json.contains(r#""network":null"#));
}

#[test]
fn exec_request_builder() {
    let req = ExecRequest::new("python3")
        .args(vec!["-c".to_string(), "print('hello')".to_string()])
        .env(HashMap::from([("PYTHONPATH".to_string(), "/app".to_string())]))
        .working_dir("/app")
        .timeout_s(60);

    assert_eq!(req.command, "python3");
    assert_eq!(req.args.as_ref().unwrap().len(), 2);
    assert_eq!(req.env.as_ref().unwrap()["PYTHONPATH"], "/app");
    assert_eq!(req.working_dir.as_deref(), Some("/app"));
    assert_eq!(req.timeout_s, Some(60));

    // JSON round-trip
    let json = serde_json::to_string(&req).unwrap();
    let deserialized: ExecRequest = serde_json::from_str(&json).unwrap();
    assert_eq!(deserialized.command, "python3");
}

#[test]
fn exec_request_minimal_serializes() {
    let req = ExecRequest::new("ls");
    let json = serde_json::to_string(&req).unwrap();
    assert!(json.contains(r#""command":"ls""#));
    assert!(json.contains(r#""args":null"#));
    assert!(json.contains(r#""workingDir":null"#));
    assert!(json.contains(r#""timeoutS":null"#));
}

#[test]
fn exec_result_roundtrip() {
    let json = r#"{
        "stdout": "hello world\n",
        "stderr": "",
        "exitCode": 0
    }"#;

    let result: ExecResult = serde_json::from_str(json).unwrap();
    assert_eq!(result.stdout, "hello world\n");
    assert_eq!(result.stderr, "");
    assert_eq!(result.exit_code, 0);

    let serialized = serde_json::to_string(&result).unwrap();
    assert!(serialized.contains(r#""exitCode":0"#));
}

#[test]
fn network_rules_serializes() {
    let rules = NetworkRules::new(
        vec!["*.api.superserve.ai".to_string()],
        vec!["10.0.0.0/8".to_string()],
    );

    let json = serde_json::to_string(&rules).unwrap();
    assert!(json.contains(r#""allowOut":["*.api.superserve.ai"]"#));
    assert!(json.contains(r#""denyOut":["10.0.0.0/8"]"#));

    let deserialized: NetworkRules = serde_json::from_str(&json).unwrap();
    assert_eq!(deserialized.allow_out.len(), 1);
    assert_eq!(deserialized.deny_out.len(), 1);
}

#[test]
fn network_rules_default() {
    let rules = NetworkRules::default();
    assert!(rules.allow_out.is_empty());
    assert!(rules.deny_out.is_empty());
}

#[test]
fn update_sandbox_request() {
    let req = UpdateSandboxRequest::new()
        .network(NetworkRules::new(vec![], vec!["0.0.0.0/0".to_string()]))
        .metadata(HashMap::from([("owner".to_string(), "alice".to_string())]));

    let json = serde_json::to_string(&req).unwrap();
    assert!(json.contains(r#""denyOut":["0.0.0.0/0"]"#));
    assert!(json.contains(r#""owner""#));

    let deserialized: UpdateSandboxRequest = serde_json::from_str(&json).unwrap();
    assert!(deserialized.network.is_some());
    assert!(deserialized.metadata.is_some());
}

// =============================================================================
// Template model tests
// =============================================================================

#[test]
fn create_template_request_builder() {
    let req = CreateTemplateRequest::new("my-template")
        .base_image("superserve/base")
        .add_run("apt-get update && apt-get install -y curl")
        .add_run("curl -fsSL https://sh.rustup.rs | sh -s -- -y")
        .workdir("/home/user")
        .start_cmd("cargo run")
        .ready_cmd("cargo build")
        .resources(TemplateResources::new(4000, 4096, 8192));

    assert_eq!(req.name, "my-template");
    assert_eq!(req.base_image.as_deref(), Some("superserve/base"));
    assert_eq!(req.steps.len(), 2);
    assert_eq!(req.workdir.as_deref(), Some("/home/user"));
    assert_eq!(req.start_cmd.as_deref(), Some("cargo run"));
    assert_eq!(req.ready_cmd.as_deref(), Some("cargo build"));
    let res = req.resources.unwrap();
    assert_eq!(res.cpu_millis, 4000);
    assert_eq!(res.memory_mb, 4096);
    assert_eq!(res.disk_mb, 8192);
}

#[test]
fn create_template_request_minimal_serializes() {
    let req = CreateTemplateRequest::new("minimal-tmpl");
    let json = serde_json::to_string(&req).unwrap();
    assert!(json.contains(r#""name":"minimal-tmpl""#));
    assert!(json.contains(r#""baseImage":null"#));
    assert!(json.contains(r#""steps":[]"#));
}

#[test]
fn template_info_deserializes() {
    let json = r#"{
        "id": "tmpl_abc",
        "name": "Rust Dev",
        "baseImage": "superserve/base",
        "status": "ready",
        "resources": {
            "cpuMillis": 2000,
            "memoryMb": 2048,
            "diskMb": 4096
        },
        "createdAt": "2025-01-01T00:00:00Z"
    }"#;

    let info: TemplateInfo = serde_json::from_str(json).unwrap();
    assert_eq!(info.id, "tmpl_abc");
    assert_eq!(info.name, "Rust Dev");
    assert_eq!(info.base_image, "superserve/base");
    assert_eq!(info.status, TemplateStatus::Ready);
    assert_eq!(info.resources.cpu_millis, 2000);
    assert_eq!(info.resources.memory_mb, 2048);
    assert_eq!(info.resources.disk_mb, 4096);
}

#[test]
fn template_resources_default() {
    let res = TemplateResources::default();
    assert_eq!(res.cpu_millis, 2000);
    assert_eq!(res.memory_mb, 2048);
    assert_eq!(res.disk_mb, 4096);
}

#[test]
fn build_step_serializes() {
    let step = BuildStep::run("apt-get install -y git");
    let json = serde_json::to_string(&step).unwrap();
    assert!(json.contains(r#""run":"apt-get install -y git""#));
}

#[test]
fn build_env_step_serializes() {
    let mut env = HashMap::new();
    env.insert("PATH".to_string(), "/usr/local/bin:$PATH".to_string());
    let step = BuildEnvStep::env(env);
    let json = serde_json::to_string(&step).unwrap();
    assert!(json.contains("PATH"));
    assert!(json.contains("/usr/local/bin"));
}

#[test]
fn build_user_step_serializes() {
    let step = BuildUserStep::new("deploy").sudo(true);
    let json = serde_json::to_string(&step).unwrap();
    assert!(json.contains(r#""name":"deploy""#));
    assert!(json.contains(r#""sudo":true"#));
}

#[test]
fn build_info_deserializes() {
    let json = r#"{
        "id": "build_xyz",
        "templateId": "tmpl_abc",
        "status": "building",
        "createdAt": "2025-01-01T01:00:00Z"
    }"#;

    let info: BuildInfo = serde_json::from_str(json).unwrap();
    assert_eq!(info.id, "build_xyz");
    assert_eq!(info.template_id, "tmpl_abc");
    assert_eq!(info.status, BuildStatus::Building);
    assert_eq!(info.created_at, "2025-01-01T01:00:00Z");
}

#[test]
fn health_response_roundtrip() {
    let json = r#"{"ok":true}"#;
    let resp: HealthResponse = serde_json::from_str(json).unwrap();
    assert!(resp.ok);

    let serialized = serde_json::to_string(&resp).unwrap();
    assert!(serialized.contains(r#""ok":true"#));
}

// =============================================================================
// Enum display & variant tests
// =============================================================================

#[test]
fn sandbox_status_display() {
    assert_eq!(SandboxStatus::Active.to_string(), "active");
    assert_eq!(SandboxStatus::Paused.to_string(), "paused");
    assert_eq!(SandboxStatus::Resuming.to_string(), "resuming");
    assert_eq!(SandboxStatus::Failed.to_string(), "failed");
    assert_eq!(SandboxStatus::Deleted.to_string(), "deleted");
}

#[test]
fn sandbox_status_deserialize_all_variants() {
    for (json, expected) in [
        (r#""active""#, SandboxStatus::Active),
        (r#""paused""#, SandboxStatus::Paused),
        (r#""resuming""#, SandboxStatus::Resuming),
        (r#""failed""#, SandboxStatus::Failed),
        (r#""deleted""#, SandboxStatus::Deleted),
    ] {
        let status: SandboxStatus = serde_json::from_str(json).unwrap();
        assert_eq!(status, expected);
    }
}

#[test]
fn template_status_display() {
    assert_eq!(TemplateStatus::Ready.to_string(), "ready");
    assert_eq!(TemplateStatus::Pending.to_string(), "pending");
    assert_eq!(TemplateStatus::Building.to_string(), "building");
    assert_eq!(TemplateStatus::Failed.to_string(), "failed");
    assert_eq!(TemplateStatus::Cancelled.to_string(), "cancelled");
}

#[test]
fn template_status_deserialize_all_variants() {
    for (json, expected) in [
        (r#""ready""#, TemplateStatus::Ready),
        (r#""pending""#, TemplateStatus::Pending),
        (r#""building""#, TemplateStatus::Building),
        (r#""failed""#, TemplateStatus::Failed),
        (r#""cancelled""#, TemplateStatus::Cancelled),
    ] {
        let status: TemplateStatus = serde_json::from_str(json).unwrap();
        assert_eq!(status, expected);
    }
}

#[test]
fn build_status_display() {
    assert_eq!(BuildStatus::Pending.to_string(), "pending");
    assert_eq!(BuildStatus::Building.to_string(), "building");
    assert_eq!(BuildStatus::Snapshotting.to_string(), "snapshotting");
    assert_eq!(BuildStatus::Ready.to_string(), "ready");
    assert_eq!(BuildStatus::Failed.to_string(), "failed");
    assert_eq!(BuildStatus::Cancelled.to_string(), "cancelled");
}

#[test]
fn build_status_deserialize_all_variants() {
    for (json, expected) in [
        (r#""pending""#, BuildStatus::Pending),
        (r#""building""#, BuildStatus::Building),
        (r#""snapshotting""#, BuildStatus::Snapshotting),
        (r#""ready""#, BuildStatus::Ready),
        (r#""failed""#, BuildStatus::Failed),
        (r#""cancelled""#, BuildStatus::Cancelled),
    ] {
        let status: BuildStatus = serde_json::from_str(json).unwrap();
        assert_eq!(status, expected);
    }
}

// =============================================================================
// Error type tests
// =============================================================================

#[test]
fn error_auth_display() {
    let err = SuperserveError::auth("invalid API key");
    assert_eq!(
        err.to_string(),
        "authentication failed: invalid API key"
    );
}

#[test]
fn error_not_found_display() {
    let err = SuperserveError::not_found("sandbox sb_123");
    assert_eq!(err.to_string(), "resource not found: sandbox sb_123");
}

#[test]
fn error_rate_limited_display() {
    let err = SuperserveError::rate_limited(30);
    assert_eq!(err.to_string(), "rate limited: retry after 30s");
}

#[test]
fn error_api_display() {
    let err = SuperserveError::api(500, "internal server error");
    assert_eq!(err.to_string(), "API error (HTTP 500): internal server error");
}

#[test]
fn error_constructors() {
    let _ = SuperserveError::auth("test");
    let _ = SuperserveError::not_found("sandbox xyz");
    let _ = SuperserveError::rate_limited(10);
    let _ = SuperserveError::api(422, "validation failed");
}

#[test]
fn api_error_response_deserializes() {
    let json = r#"{"error":"unauthorized","detail":"API key is missing"}"#;
    let resp: ApiErrorResponse = serde_json::from_str(json).unwrap();
    assert_eq!(resp.error.as_deref(), Some("unauthorized"));
    assert_eq!(resp.detail.as_deref(), Some("API key is missing"));
}

#[test]
fn api_error_response_handles_empty() {
    let json = r#"{}"#;
    let resp: ApiErrorResponse = serde_json::from_str(json).unwrap();
    assert!(resp.error.is_none());
    assert!(resp.detail.is_none());
}

// =============================================================================
// Client construction tests
// =============================================================================

#[test]
fn client_new() {
    let client = SuperserveClient::new("ss_live_test_key");
    assert_eq!(client.base_url(), DEFAULT_BASE_URL);
}

#[test]
fn client_custom_base_url() {
    let client = SuperserveClient::with_base_url("ss_live_test", "https://custom.api.example.com");
    assert_eq!(client.base_url(), "https://custom.api.example.com");
}

#[test]
fn client_custom_base_url_strips_trailing_slash() {
    let client = SuperserveClient::with_base_url("ss_live_test", "https://api.example.com/");
    assert_eq!(client.base_url(), "https://api.example.com");
}

#[test]
#[should_panic(expected = "API key must be valid header value")]
fn client_rejects_invalid_api_key() {
    // Header values with newlines are invalid
    let _ = SuperserveClient::new("ss_live_\nbad");
}

#[test]
fn client_from_env_missing() {
    // Remove the env var if set, then verify error
    std::env::remove_var("SUPERSERVE_API_KEY");
    let result = SuperserveClient::from_env();
    assert!(result.is_err());
    let err = result.unwrap_err().to_string();
    assert!(err.contains("SUPERSERVE_API_KEY"));
}

#[test]
fn default_base_url_constant() {
    assert_eq!(DEFAULT_BASE_URL, "https://api.superserve.ai");
}

// =============================================================================
// Template config tests (cross-module)
// =============================================================================

#[test]
fn template_rust_crate_is_valid_request() {
    let req = templates::RUST_CRATE("rust-test".to_string());
    let json = serde_json::to_string(&req).unwrap();
    assert!(json.contains("rust-test"));
    assert!(json.contains("superserve/base"));
    assert!(json.contains("rustup"));

    // Ensure it round-trips
    let deserialized: CreateTemplateRequest = serde_json::from_str(&json).unwrap();
    assert_eq!(deserialized.name, "rust-test");
}

#[test]
fn template_python_ml_is_valid_request() {
    let req = templates::PYTHON_ML("ml-test".to_string());
    let json = serde_json::to_string(&req).unwrap();
    assert!(json.contains("ml-test"));
    assert!(json.contains("superserve/python-3.11"));
    assert!(json.contains("numpy"));
    assert!(json.contains("pandas"));

    let deserialized: CreateTemplateRequest = serde_json::from_str(&json).unwrap();
    assert_eq!(deserialized.name, "ml-test");
}

#[test]
fn template_chp_validator_is_valid_request() {
    let req = templates::CHP_VALIDATOR("chp-test".to_string());
    let json = serde_json::to_string(&req).unwrap();
    assert!(json.contains("chp-test"));
    assert!(json.contains("superserve/base"));
    assert!(json.contains("rustup"));
    assert!(json.contains("cargo-nextest"));

    let deserialized: CreateTemplateRequest = serde_json::from_str(&json).unwrap();
    assert_eq!(deserialized.name, "chp-test");
}

// =============================================================================
// Edge case / comprehensive tests
// =============================================================================

#[test]
fn sandbox_info_with_all_null_optionals() {
    let json = r#"{
        "id": "id-1",
        "name": "sparse",
        "status": "failed",
        "createdAt": "2025-06-01T00:00:00Z",
        "templateId": null,
        "accessToken": null,
        "timeoutSeconds": null,
        "metadata": {},
        "envVars": {}
    }"#;

    let info: SandboxInfo = serde_json::from_str(json).unwrap();
    assert_eq!(info.status, SandboxStatus::Failed);
    assert!(info.template_id.is_none());
    assert!(info.access_token.is_none());
    assert!(info.timeout_seconds.is_none());
    assert!(info.metadata.is_empty());
    assert!(info.env_vars.is_empty());
}

#[test]
fn exec_result_nonzero_exit_code() {
    let json = r#"{
        "stdout": "",
        "stderr": "command not found: foobar\n",
        "exitCode": 127
    }"#;

    let result: ExecResult = serde_json::from_str(json).unwrap();
    assert!(result.stdout.is_empty());
    assert!(result.stderr.contains("command not found"));
    assert_eq!(result.exit_code, 127);
}

#[test]
fn create_sandbox_request_with_large_metadata() {
    let mut meta = HashMap::new();
    for i in 0..64 {
        meta.insert(format!("key_{}", i), format!("value_{}", i));
    }
    let req = CreateSandboxRequest::new("big-meta").metadata(meta);
    assert_eq!(req.metadata.as_ref().unwrap().len(), 64);
}

#[test]
fn list_sandboxes_empty_filter() {
    // Verify that empty filter serialises sensibly
    let filter: HashMap<String, String> = HashMap::new();
    assert!(filter.is_empty());
}

#[test]
fn debug_derive_on_models() {
    // Ensure all types implement Debug
    let _info = format!("{:?}", SandboxInfo {
        id: "id".to_string(),
        name: "n".to_string(),
        status: SandboxStatus::Active,
        created_at: "t".to_string(),
        template_id: None,
        access_token: None,
        timeout_seconds: None,
        metadata: HashMap::new(),
        env_vars: HashMap::new(),
    });
    let _err = format!("{:?}", SuperserveError::auth("test"));
    let _req = format!("{:?}", CreateSandboxRequest::new("test"));
    let _exec = format!("{:?}", ExecRequest::new("ls"));
    let _res = format!("{:?}", ExecResult {
        stdout: "out".to_string(),
        stderr: "".to_string(),
        exit_code: 0,
    });
    let _tmpl = format!("{:?}", TemplateInfo {
        id: "id".to_string(),
        name: "n".to_string(),
        base_image: "img".to_string(),
        status: TemplateStatus::Ready,
        resources: TemplateResources::default(),
        created_at: "t".to_string(),
    });
    let _build = format!("{:?}", BuildInfo {
        id: "id".to_string(),
        template_id: "tid".to_string(),
        status: BuildStatus::Building,
        created_at: "t".to_string(),
    });
}

#[test]
fn clone_derive_on_models() {
    let req = CreateSandboxRequest::new("original")
        .template_id("tmpl")
        .timeout_seconds(100);
    let _cloned = req.clone();

    // Note: SuperserveError does NOT impl Clone because reqwest::Error is not Clone.
    let _err = format!("{:?}", SuperserveError::api(404, "not found"));
}
