//! Raw text parser for CHP partner packets.

pub struct ParsedPacket {
    pub sections: Vec<(String, String)>, // (section_name, content)
    pub payload_id: Option<String>,
    pub route: Option<String>,
    pub is_valid_envelope: bool,
    pub errors: Vec<String>,
}

pub fn parse_partner_packet(raw: &str) -> ParsedPacket {
    let mut result = ParsedPacket {
        sections: Vec::new(),
        payload_id: None,
        route: None,
        is_valid_envelope: false,
        errors: Vec::new(),
    };

    let lines: Vec<&str> = raw.lines().collect();
    if lines.is_empty() {
        result.errors.push("empty packet".into());
        return result;
    }

    // Extract envelope
    let first = lines[0].trim();
    if first.starts_with("BEGIN_PAYLOAD [") {
        // Extract [ROUTE] [PAYLOAD_ID] from "BEGIN_PAYLOAD [ROUTE] [PAYLOAD_ID]"
        let after_begin = first.strip_prefix("BEGIN_PAYLOAD").unwrap().trim();
        let bracket_content: Vec<&str> = after_begin.split(']').collect();
        for segment in bracket_content.iter() {
            let trimmed = segment.trim().trim_start_matches('[').trim();
            if result.route.is_none() {
                result.route = Some(trimmed.to_string());
            } else if result.payload_id.is_none() && !trimmed.is_empty() {
                result.payload_id = Some(trimmed.to_string());
            }
        }
    }

    let last = lines[lines.len() - 1].trim();
    result.is_valid_envelope = first.starts_with("BEGIN_PAYLOAD") && last.starts_with("END_PAYLOAD");

    if !result.is_valid_envelope {
        result.errors.push("invalid payload envelope".into());
    }

    // Parse sections — split on section headers (all caps with underscores)
    let mut current_section = String::new();
    let mut current_content: Vec<&str> = Vec::new();

    for line in &lines {
        let trimmed = line.trim();
        if trimmed.starts_with("BEGIN_PAYLOAD") || trimmed.starts_with("END_PAYLOAD") {
            continue;
        }
        // Detect section headers
        if is_section_header(trimmed) {
            if !current_section.is_empty() {
                result.sections.push((current_section.clone(), current_content.join("\n").trim().to_string()));
            }
            current_section = extract_section_name(trimmed);
            current_content.clear();
        } else if !current_section.is_empty() {
            current_content.push(*line);
        }
    }
    if !current_section.is_empty() {
        result.sections.push((current_section, current_content.join("\n").trim().to_string()));
    }

    result
}

fn is_section_header(line: &str) -> bool {
    // Match patterns like "ITEM_AGREEMENTS:", "1. CORE_PROBLEM_STATEMENT", "SCORING_TABLE:"
    let upper = line.to_uppercase();
    if upper.contains("CORE_PROBLEM") || upper.contains("PARTNER_SYSTEM")
        || upper.contains("TRANSMISSION_CHECKLIST") || upper.contains("ITEM_AGREEMENTS")
        || upper.contains("WINNER_FRAMING") || upper.contains("SCORING_TABLE")
        || upper.contains("OBJECTIONS") || upper.contains("FRAMEWORKS")
        || upper.contains("CONVERGENCE_PLAN") || upper.contains("STATE_SNAPSHOT")
        || upper.contains("R0_GATE") || upper.contains("FOUNDATION_DISCLOSURE")
        || upper.contains("FOUNDATION_ATTACK") || upper.contains("STYLE_GUIDE") {
        return true;
    }
    // Match ALL_CAPS followed by colon
    let trimmed = line.trim_end_matches(':');
    trimmed.chars().all(|c| c.is_ascii_uppercase() || c == '_' || c == ' ') && trimmed.len() > 2
}

fn extract_section_name(line: &str) -> String {
    // Remove leading number like "1. " and trailing colon
    let trimmed = line.trim().trim_end_matches(':');
    // Remove leading "N. " pattern
    if let Some(pos) = trimmed.find(". ") {
        trimmed[pos + 2..].trim().to_string()
    } else {
        trimmed.to_string()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_parse_valid_envelope() {
        let raw = "BEGIN_PAYLOAD [RX] [ABC123]\nbody\nEND_PAYLOAD [RX] [ABC123]";
        let parsed = parse_partner_packet(raw);
        assert!(parsed.is_valid_envelope);
        assert_eq!(parsed.route.as_deref(), Some("RX"));
        assert_eq!(parsed.payload_id.as_deref(), Some("ABC123"));
        assert!(parsed.errors.is_empty());
    }

    #[test]
    fn test_parse_invalid_envelope() {
        let raw = "random text\nno envelope";
        let parsed = parse_partner_packet(raw);
        assert!(!parsed.is_valid_envelope);
        assert!(parsed.errors.iter().any(|e| e.contains("invalid payload envelope")));
    }

    #[test]
    fn test_parse_empty_packet() {
        let parsed = parse_partner_packet("");
        assert!(parsed.errors.iter().any(|e| e.contains("empty packet")));
    }

    #[test]
    fn test_parse_sections() {
        let raw = "BEGIN_PAYLOAD [RX] [A1B2C3]\n\
                   ITEM_AGREEMENTS:\n\
                   item1: score 90, LOCKED\n\
                   WINNER_FRAMING:\n\
                   The winner is A.\n\
                   END_PAYLOAD [RX] [A1B2C3]";
        let parsed = parse_partner_packet(raw);
        assert_eq!(parsed.sections.len(), 2);
        assert_eq!(parsed.sections[0].0, "ITEM_AGREEMENTS");
        assert!(parsed.sections[0].1.contains("item1"));
        assert_eq!(parsed.sections[1].0, "WINNER_FRAMING");
    }

    #[test]
    fn test_parse_numbered_section() {
        let raw = "BEGIN_PAYLOAD [RX] [X1]\n\
                   1. CORE_PROBLEM_STATEMENT\n\
                   Some problem text\n\
                   END_PAYLOAD [RX] [X1]";
        let parsed = parse_partner_packet(raw);
        assert_eq!(parsed.sections.len(), 1);
        assert_eq!(parsed.sections[0].0, "CORE_PROBLEM_STATEMENT");
    }

    #[test]
    fn test_parse_all_caps_with_colon() {
        let raw = "BEGIN_PAYLOAD [RX] [X1]\n\
                   SOME_CUSTOM_HEADER:\n\
                   content here\n\
                   END_PAYLOAD [RX] [X1]";
        let parsed = parse_partner_packet(raw);
        assert_eq!(parsed.sections.len(), 1);
        assert_eq!(parsed.sections[0].0, "SOME_CUSTOM_HEADER");
    }

    #[test]
    fn test_parse_multiple_sections() {
        let raw = "BEGIN_PAYLOAD [RX] [PID]\n\
                   FOUNDATION_DISCLOSURE:\n\
                   content1\n\
                   FOUNDATION_ATTACK:\n\
                   content2\n\
                   STYLE_GUIDE:\n\
                   content3\n\
                   END_PAYLOAD [RX] [PID]";
        let parsed = parse_partner_packet(raw);
        assert_eq!(parsed.sections.len(), 3);
        let names: Vec<&str> = parsed.sections.iter().map(|(n, _)| n.as_str()).collect();
        assert_eq!(names, vec!["FOUNDATION_DISCLOSURE", "FOUNDATION_ATTACK", "STYLE_GUIDE"]);
    }

    #[test]
    fn test_parse_no_sections() {
        let raw = "BEGIN_PAYLOAD [RX] [PID]\n\
                   just some text\n\
                   more text\n\
                   END_PAYLOAD [RX] [PID]";
        let parsed = parse_partner_packet(raw);
        assert!(parsed.sections.is_empty());
    }

    #[test]
    fn test_parse_section_content_trimmed() {
        let raw = "BEGIN_PAYLOAD [RX] [PID]\n\
                   SECTION_A:\n\
                   \n\
                   content line\n\
                   \n\
                   END_PAYLOAD [RX] [PID]";
        let parsed = parse_partner_packet(raw);
        assert_eq!(parsed.sections.len(), 1);
        assert_eq!(parsed.sections[0].1, "content line");
    }
}
