//! Context Engine: layered memory with importance/relevance scoring.

use std::collections::HashMap;

#[derive(Debug, Clone)]
pub struct Entity {
    pub name: String,
    pub kind: String,
    pub attributes: serde_json::Value,
    pub last_seen: String,
}

#[derive(Debug, Clone)]
pub struct Event {
    pub description: String,
    pub payload: serde_json::Value,
    pub timestamp: String,
}

#[derive(Debug, Clone)]
pub struct Task {
    pub description: String,
    pub status: String,
    pub priority: u32,
}

#[derive(Debug, Clone)]
pub struct ContextEntry {
    pub content: String,
    pub source: String,
    pub importance: f64,
    pub timestamp: String,
    pub tokens: Vec<String>,
}

pub struct ContextEngine {
    short_term: Vec<ContextEntry>,
    long_term: Vec<ContextEntry>,
    entities: HashMap<String, Entity>,
    events: Vec<Event>,
    tasks: Vec<Task>,
    max_short: usize,
    max_long: usize,
}

impl ContextEngine {
    pub fn new() -> Self {
        Self {
            short_term: Vec::new(),
            long_term: Vec::new(),
            entities: HashMap::new(),
            events: Vec::new(),
            tasks: Vec::new(),
            max_short: 100,
            max_long: 500,
        }
    }

    pub fn write(&mut self, content: &str, source: &str, importance: f64) {
        let now = chrono::Utc::now().to_rfc3339();
        let tokens: Vec<String> = content
            .split_whitespace()
            .map(|w| w.to_lowercase())
            .filter(|w| w.len() >= 3)
            .collect();
        let entry = ContextEntry {
            content: content.into(),
            source: source.into(),
            importance,
            timestamp: now,
            tokens,
        };
        self.short_term.push(entry);
        if self.short_term.len() > self.max_short {
            self._promote_short_to_long();
        }
    }

    pub fn select(&self, query: &str, limit: usize) -> Vec<ContextEntry> {
        let query_tokens: Vec<String> = query
            .split_whitespace()
            .map(|w| w.to_lowercase())
            .collect();
        if query_tokens.is_empty() {
            return Vec::new();
        }

        let all_entries: Vec<&ContextEntry> = self.short_term.iter()
            .chain(self.long_term.iter())
            .collect();

        let mut scored: Vec<(f64, usize)> = all_entries.iter().enumerate().map(|(idx, entry)| {
            let token_overlap: f64 = entry.tokens.iter()
                .filter(|t| query_tokens.contains(t))
                .count() as f64;
            let max_possible = query_tokens.len().max(entry.tokens.len()) as f64;
            let cosine = if max_possible > 0.0 { token_overlap / max_possible } else { 0.0 };
            let score = cosine * entry.importance;
            (score, idx)
        }).collect();

        scored.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap_or(std::cmp::Ordering::Equal));

        scored.into_iter()
            .take(limit)
            .map(|(_, idx)| all_entries[idx].clone())
            .collect()
    }

    pub fn upsert_entity(&mut self, name: &str, kind: &str, attributes: serde_json::Value) {
        let now = chrono::Utc::now().to_rfc3339();
        self.entities.insert(name.to_string(), Entity {
            name: name.into(),
            kind: kind.into(),
            attributes,
            last_seen: now,
        });
    }

    pub fn record_event(&mut self, description: &str, payload: serde_json::Value) {
        let now = chrono::Utc::now().to_rfc3339();
        self.events.push(Event {
            description: description.into(),
            payload,
            timestamp: now,
        });
    }

    pub fn snapshot_for(&self, domain: &str) -> String {
        let entity_names: Vec<&str> = self.entities.keys().map(|s| s.as_str()).collect();
        let task_count = self.tasks.len();
        let entry_count = self.short_term.len() + self.long_term.len();
        format!(
            "Context snapshot for '{}': {} entities ({:?}), {} tasks, {} memory entries",
            domain, self.entities.len(), entity_names, task_count, entry_count,
        )
    }

    fn _promote_short_to_long(&mut self) {
        if let Some(oldest) = self.short_term.first() {
            self.long_term.push(oldest.clone());
            self.short_term.remove(0);
            if self.long_term.len() > self.max_long {
                self.long_term.remove(0);
            }
        }
    }

    pub fn entity_count(&self) -> usize {
        self.entities.len()
    }

    pub fn event_count(&self) -> usize {
        self.events.len()
    }

    pub fn entry_count(&self) -> usize {
        self.short_term.len() + self.long_term.len()
    }
}

impl Default for ContextEngine {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_write_and_select() {
        let mut ctx = ContextEngine::new();
        ctx.write("The quarterly revenue report shows strong growth in APAC region", "report", 0.8);
        ctx.write("APAC revenue grew 15% year over year", "finance", 0.9);

        let results = ctx.select("APAC revenue", 5);
        assert_eq!(results.len(), 2);
        // Higher importance should rank first
        assert_eq!(results[0].source, "finance");
    }

    #[test]
    fn test_upsert_entity() {
        let mut ctx = ContextEngine::new();
        ctx.upsert_entity("Acme Corp", "company", serde_json::json!({"sector": "tech"}));
        assert_eq!(ctx.entity_count(), 1);
    }

    #[test]
    fn test_record_event() {
        let mut ctx = ContextEngine::new();
        ctx.record_event("Decision made", serde_json::json!({"type": "approval"}));
        assert_eq!(ctx.event_count(), 1);
    }

    #[test]
    fn test_snapshot() {
        let ctx = ContextEngine::new();
        let snap = ctx.snapshot_for("finance");
        assert!(snap.contains("finance"));
    }

    #[test]
    fn test_promotion() {
        let mut ctx = ContextEngine::new();
        ctx.max_short = 2;
        ctx.write("entry 1", "s1", 0.5);
        ctx.write("entry 2", "s2", 0.5);
        ctx.write("entry 3", "s3", 0.5); // triggers promotion
        assert_eq!(ctx.short_term.len(), 2);
        assert_eq!(ctx.long_term.len(), 1);
    }
}
