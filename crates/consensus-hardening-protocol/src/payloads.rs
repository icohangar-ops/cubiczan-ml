//! Payload-integrity helpers for CHP packet exchange.

use rand::Rng;

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
}
