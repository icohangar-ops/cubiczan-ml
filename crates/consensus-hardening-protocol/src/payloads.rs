//! Payload-integrity helpers for CHP packet exchange.

use rand::Rng;
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone)]
pub struct PayloadEnvelope {
    pub route: String,
    pub payload_id: String,
    pub body: String,
}

impl PayloadEnvelope {
    pub fn render(&self) -> String {
        format!(
            "BEGIN_PAYLOAD [{}] [{}]\n{}\nEND_PAYLOAD [{}] [{}]",
            self.route, self.payload_id, self.body, self.route, self.payload_id,
        )
    }
}

/// Generate a random 6-char alphanumeric payload ID (uppercase + digits).
pub fn make_payload_id() -> String {
    let chars: Vec<char> = "ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789".chars().collect();
    let mut rng = rand::thread_rng();
    (0..6).map(|_| chars[rng.gen_range(0..chars.len())]).collect()
}

pub fn build_payload_envelope(body: &str, route: &str, payload_id: Option<&str>) -> PayloadEnvelope {
    PayloadEnvelope {
        route: route.to_string(),
        payload_id: payload_id.unwrap_or(&make_payload_id()).to_string(),
        body: body.to_string(),
    }
}

pub fn validate_payload_envelope(rendered: &str) -> bool {
    let lines: Vec<&str> = rendered.trim().lines().map(|l| l.trim_end()).collect();
    if lines.len() < 3 {
        return false;
    }
    let first = lines[0];
    let last = lines[lines.len() - 1];
    if !first.starts_with("BEGIN_PAYLOAD [") || !last.starts_with("END_PAYLOAD [") {
        return false;
    }
    let first_suffix = first.strip_prefix("BEGIN_PAYLOAD").unwrap().trim();
    let last_suffix = last.strip_prefix("END_PAYLOAD").unwrap().trim();
    first_suffix == last_suffix
}

pub fn payload_echo_confirmed(route: &str, payload_id: &str, echo: &str) -> bool {
    echo.trim() == format!("[{}] [{}] CONFIRMED", route, payload_id)
}

pub fn extract_payload_id(rendered: &str) -> Option<String> {
    let first_line = rendered.trim().lines().next()?;
    if !first_line.starts_with("BEGIN_PAYLOAD [") {
        return None;
    }
    let parts: Vec<&str> = first_line.split('[').collect();
    if parts.len() < 3 {
        return None;
    }
    let raw = parts[2].trim_end_matches(']');
    Some(raw.trim().to_string())
}

// ============================================================================
// EchoStatus & PayloadValidator
// ============================================================================

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum EchoStatus {
    CONFIRMED,
    MISMATCH,
    MISSING,
}

pub struct PayloadValidator;

impl PayloadValidator {
    pub fn validate_echo(received_id: &str, expected_id: &str) -> EchoStatus {
        if received_id.is_empty() || expected_id.is_empty() {
            return EchoStatus::MISSING;
        }
        if received_id == expected_id {
            EchoStatus::CONFIRMED
        } else {
            EchoStatus::MISMATCH
        }
    }

    pub fn on_mismatch(route: &str) -> (String, String) {
        let new_id = make_payload_id();
        let echo = format!("[{}] [{}] CONFIRMED", route, new_id);
        (new_id, echo)
    }

    pub fn on_missing_marker(route: &str) -> (String, String) {
        let new_id = make_payload_id();
        let echo = format!("[{}] [{}] CONFIRMED", route, new_id);
        (new_id, echo)
    }

    /// Check that echo is confirmed before allowing state advancement
    pub fn gate(payload_echo: &str, expected_route: &str, expected_id: &str) -> Result<(), String> {
        let expected_echo = format!("[{}] [{}] CONFIRMED", expected_route, expected_id);
        if payload_echo.trim() == expected_echo {
            Ok(())
        } else if payload_echo.is_empty() {
            Err("PAYLOAD_ECHO is missing — cannot advance state".into())
        } else {
            Err(format!("PAYLOAD_ECHO mismatch: expected '{}', got '{}'", expected_echo, payload_echo.trim()))
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_build_and_validate_envelope() {
        let env = build_payload_envelope("test body", "RX", Some("ABC123"));
        let rendered = env.render();
        assert!(validate_payload_envelope(&rendered));
    }

    #[test]
    fn test_invalid_envelope() {
        assert!(!validate_payload_envelope("random text"));
        assert!(!validate_payload_envelope("BEGIN_PAYLOAD [RX]\nbody\nEND_SOMETHING"));
    }

    #[test]
    fn test_extract_payload_id() {
        let env = build_payload_envelope("body", "RX", Some("XY99ZZ"));
        let rendered = env.render();
        assert_eq!(extract_payload_id(&rendered), Some("XY99ZZ".into()));
    }

    #[test]
    fn test_extract_payload_id_none() {
        assert_eq!(extract_payload_id("no payload here"), None);
    }

    #[test]
    fn test_payload_echo_confirmed() {
        assert!(payload_echo_confirmed("RX", "ABC123", "[RX] [ABC123] CONFIRMED"));
        assert!(!payload_echo_confirmed("RX", "ABC123", "wrong"));
    }

    #[test]
    fn test_make_payload_id_length() {
        let id = make_payload_id();
        assert_eq!(id.len(), 6);
        assert!(id.chars().all(|c| c.is_ascii_uppercase() || c.is_ascii_digit()));
    }

    // --- EchoStatus tests ---

    #[test]
    fn test_validate_echo_confirmed() {
        assert_eq!(PayloadValidator::validate_echo("ABC123", "ABC123"), EchoStatus::CONFIRMED);
    }

    #[test]
    fn test_validate_echo_mismatch() {
        assert_eq!(PayloadValidator::validate_echo("ABC123", "XYZ789"), EchoStatus::MISMATCH);
    }

    #[test]
    fn test_validate_echo_missing_received_empty() {
        assert_eq!(PayloadValidator::validate_echo("", "ABC123"), EchoStatus::MISSING);
    }

    #[test]
    fn test_validate_echo_missing_expected_empty() {
        assert_eq!(PayloadValidator::validate_echo("ABC123", ""), EchoStatus::MISSING);
    }

    #[test]
    fn test_validate_echo_missing_both_empty() {
        assert_eq!(PayloadValidator::validate_echo("", ""), EchoStatus::MISSING);
    }

    // --- PayloadValidator::on_mismatch tests ---

    #[test]
    fn test_on_mismatch_returns_new_id_and_echo() {
        let (id, echo) = PayloadValidator::on_mismatch("RX");
        assert_eq!(id.len(), 6);
        assert_eq!(echo, format!("[RX] [{}] CONFIRMED", id));
    }

    #[test]
    fn test_on_missing_marker_returns_new_id_and_echo() {
        let (id, echo) = PayloadValidator::on_missing_marker("TX");
        assert_eq!(id.len(), 6);
        assert_eq!(echo, format!("[TX] [{}] CONFIRMED", id));
    }

    // --- PayloadValidator::gate tests ---

    #[test]
    fn test_gate_pass() {
        let result = PayloadValidator::gate("[RX] [ABC123] CONFIRMED", "RX", "ABC123");
        assert!(result.is_ok());
    }

    #[test]
    fn test_gate_fail_empty() {
        let result = PayloadValidator::gate("", "RX", "ABC123");
        assert!(result.is_err());
        assert!(result.unwrap_err().contains("missing"));
    }

    #[test]
    fn test_gate_fail_mismatch() {
        let result = PayloadValidator::gate("[RX] [WRONG] CONFIRMED", "RX", "ABC123");
        assert!(result.is_err());
        assert!(result.unwrap_err().contains("mismatch"));
    }

    #[test]
    fn test_gate_pass_with_whitespace() {
        let result = PayloadValidator::gate("  [RX] [ABC123] CONFIRMED  ", "RX", "ABC123");
        assert!(result.is_ok());
    }

    #[test]
    fn test_echo_status_serde() {
        for status in [EchoStatus::CONFIRMED, EchoStatus::MISMATCH, EchoStatus::MISSING] {
            let json = serde_json::to_string(&status).unwrap();
            let restored: EchoStatus = serde_json::from_str(&json).unwrap();
            assert_eq!(status, restored);
        }
    }
}
