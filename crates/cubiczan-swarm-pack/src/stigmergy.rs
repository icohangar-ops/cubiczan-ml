//! Stigmergic coordination with pheromone signals.
//!
//! This module implements the stigmergic coordination layer where agents
//! communicate indirectly through pheromone-like scent signals:
//! - Exponential decay for most scent types
//! - Growing intensity for urgency signals
//! - Task scoring using the TEMM1E formula
//! - Best task selection with random tie-breaking
//! - Garbage collection of expired signals

use crate::types::*;
use chrono::{DateTime, Utc};
use std::collections::{HashMap, HashSet};

// ---------------------------------------------------------------------------
// Constants
// ---------------------------------------------------------------------------

/// Half-life in seconds for each scent type.
/// A negative value means the scent grows over time (urgency).
pub const SCENT_HALF_LIVES: [(ScentType, f64); 6] = [
    (ScentType::Completion, 300.0),
    (ScentType::Failure, 360.0),
    (ScentType::Difficulty, 120.0),
    (ScentType::Urgency, -1.0),
    (ScentType::Progress, 20.0),
    (ScentType::HelpWanted, 120.0),
];

/// Signals below this intensity threshold are garbage collected.
pub const GC_THRESHOLD: f64 = 0.01;

// ---------------------------------------------------------------------------
// Decay computation
// ---------------------------------------------------------------------------

/// Compute the decay constant λ from a half-life.
///
/// λ = ln(2) / half_life
///
/// For growing scents (negative half-life), returns 0.0.
pub fn decay_constant(half_life: f64) -> f64 {
    if half_life <= 0.0 {
        return 0.0;
    }
    std::f64::consts::LN_2 / half_life
}

/// Compute the current intensity of a scent signal after decay.
///
/// For normal (decaying) scents:
///   current = original × e^(-λ × elapsed_seconds)
///
/// For growing scents (e.g., Urgency):
///   current = original × (1 + elapsed_seconds / 300.0)
///
/// Returns 0.0 if the signal has decayed below GC_THRESHOLD.
pub fn compute_current_intensity(
    original: f64,
    emitted_at: DateTime<Utc>,
    scent_type: ScentType,
) -> f64 {
    if original <= 0.0 {
        return 0.0;
    }

    let elapsed = Utc::now().signed_duration_since(emitted_at);
    let elapsed_secs = elapsed.num_seconds() as f64;

    if elapsed_secs < 0.0 {
        return original; // Future emission, return as-is
    }

    let current = if scent_type.is_growing() {
        // Growing scent: intensity increases over time
        original * (1.0 + elapsed_secs / 300.0)
    } else {
        // Decaying scent
        let half_life = scent_type.default_half_life();
        let lambda = decay_constant(half_life);
        original * (-lambda * elapsed_secs).exp()
    };

    if current < GC_THRESHOLD {
        0.0
    } else {
        current
    }
}

// ---------------------------------------------------------------------------
// Jaccard similarity
// ---------------------------------------------------------------------------

/// Compute the Jaccard similarity between two string sets.
///
/// J(A, B) = |A ∩ B| / |A ∪ B|
///
/// Returns 0.0 if both sets are empty, 1.0 if both are identical.
pub fn jaccard_similarity(set_a: &[String], set_b: &[String]) -> f64 {
    let set_a: HashSet<&str> = set_a.iter().map(|s| s.as_str()).collect();
    let set_b: HashSet<&str> = set_b.iter().map(|s| s.as_str()).collect();

    if set_a.is_empty() && set_b.is_empty() {
        return 1.0;
    }

    let intersection = set_a.intersection(&set_b).count();
    let union = set_a.union(&set_b).count();

    if union == 0 {
        return 0.0;
    }

    intersection as f64 / union as f64
}

// ---------------------------------------------------------------------------
// Task scoring (TEMM1E formula)
// ---------------------------------------------------------------------------

/// Compute a task score for a worker using the TEMM1E formula:
///
/// S = A^2.0 × U^1.5 × (1 - D)^1.0 × (1 - F)^0.8 × R^1.2
///
/// Where:
/// - A: affinity (tag match score, 0-1)
/// - U: aggregated urgency scent
/// - D: aggregated difficulty scent (dampening factor)
/// - F: aggregated failure scent (dampening factor)
/// - R: aggregated progress scent (boosting factor)
pub fn compute_task_score(
    worker_tags: &[String],
    task_tags: &[String],
    affinity: f64,
    urgency: f64,
    difficulty: f64,
    failure: f64,
    progress: f64,
) -> f64 {
    // Compute affinity if not provided, or use the given value
    let a = if affinity < 0.0 {
        jaccard_similarity(worker_tags, task_tags)
    } else {
        affinity
    };

    // Clamp all values to valid ranges
    let a = a.clamp(0.0, 1.0);
    let u = (1.0 + urgency).max(0.01); // Ensure > 0 for pow
    let d = difficulty.clamp(0.0, 1.0);
    let f = failure.clamp(0.0, 1.0);
    let r = (1.0 + progress).max(0.01); // Ensure > 0 for pow

    a.powi(2) * u.powf(1.5) * (1.0 - d) * (1.0 - f).powf(0.8) * r.powf(1.2)
}

// ---------------------------------------------------------------------------
// ScentField: The pheromone signal store
// ---------------------------------------------------------------------------

/// A field of pheromone signals, indexed by task ID and scent type.
///
/// Agents emit signals into this field and read aggregated intensities
/// for task selection decisions.
pub struct ScentField {
    signals: Vec<ScentSignal>,
}

impl ScentField {
    /// Create a new empty scent field.
    pub fn new() -> Self {
        ScentField {
            signals: Vec::new(),
        }
    }

    /// Emit a new scent signal into the field.
    pub fn emit(&mut self, signal: ScentSignal) {
        self.signals.push(signal);
    }

    /// Create and emit a simple signal.
    pub fn emit_signal(
        &mut self,
        task_id: impl Into<String>,
        worker_id: impl Into<String>,
        scent_type: ScentType,
        intensity: f64,
    ) {
        let signal = ScentSignal::new(
            uuid::Uuid::new_v4().to_string(),
            task_id,
            worker_id,
            scent_type,
            intensity,
        );
        self.signals.push(signal);
    }

    /// Read the current aggregated intensity for a specific task and scent type.
    ///
    /// Returns the sum of all current (decayed) intensities for matching signals.
    pub fn read(&self, task_id: &str, scent_type: ScentType) -> f64 {
        self.signals
            .iter()
            .filter(|s| s.task_id == task_id && s.scent_type == scent_type)
            .map(|s| compute_current_intensity(s.intensity, s.emitted_at, scent_type))
            .sum()
    }

    /// Read all current scent intensities for a specific task.
    ///
    /// Returns a map of scent type → current intensity.
    pub fn read_all(&self, task_id: &str) -> HashMap<ScentType, f64> {
        let mut result = HashMap::new();
        for scent_type in ScentType::all() {
            let intensity = self.read(task_id, *scent_type);
            if intensity > 0.0 {
                result.insert(*scent_type, intensity);
            }
        }
        result
    }

    /// Garbage collect expired signals (intensity below threshold).
    ///
    /// Returns the number of signals removed.
    pub fn garbage_collect(&mut self) -> usize {
        let before = self.signals.len();
        self.signals.retain(|s| {
            compute_current_intensity(s.intensity, s.emitted_at, s.scent_type) >= GC_THRESHOLD
        });
        before - self.signals.len()
    }

    /// Returns the total number of signals in the field.
    pub fn len(&self) -> usize {
        self.signals.len()
    }

    /// Returns true if the field has no signals.
    pub fn is_empty(&self) -> bool {
        self.signals.is_empty()
    }

    /// Get all signals for a specific task.
    pub fn signals_for_task(&self, task_id: &str) -> Vec<&ScentSignal> {
        self.signals
            .iter()
            .filter(|s| s.task_id == task_id)
            .collect()
    }

    /// Get all signals emitted by a specific worker.
    pub fn signals_by_worker(&self, worker_id: &str) -> Vec<&ScentSignal> {
        self.signals
            .iter()
            .filter(|s| s.worker_id == worker_id)
            .collect()
    }

    /// Read the current intensity at a specific past time (for testing).
    pub fn read_at(
        &self,
        task_id: &str,
        scent_type: ScentType,
        now: DateTime<Utc>,
    ) -> f64 {
        self.signals
            .iter()
            .filter(|s| s.task_id == task_id && s.scent_type == scent_type)
            .map(|s| {
                let elapsed = now.signed_duration_since(s.emitted_at);
                let elapsed_secs = elapsed.num_seconds() as f64;
                if elapsed_secs < 0.0 {
                    return s.intensity;
                }
                if scent_type.is_growing() {
                    s.intensity * (1.0 + elapsed_secs / 300.0)
                } else {
                    let half_life = scent_type.default_half_life();
                    let lambda = decay_constant(half_life);
                    s.intensity * (-lambda * elapsed_secs).exp()
                }
            })
            .sum()
    }

    /// Clear all signals from the field.
    pub fn clear(&mut self) {
        self.signals.clear();
    }
}

impl Default for ScentField {
    fn default() -> Self {
        Self::new()
    }
}

// ---------------------------------------------------------------------------
// Best task selection
// ---------------------------------------------------------------------------

/// Select the best task for a worker from the available tasks.
///
/// Scores all tasks using the TEMM1E formula and returns the task ID
/// with the highest score. Includes a 5% random tie-breaking factor
/// to prevent herd behavior.
///
/// Returns None if no tasks are available.
pub fn select_best_task(
    _worker_id: &str,
    worker_tags: &[String],
    available_tasks: &[&DAGTask],
    scent_field: &ScentField,
) -> Option<String> {
    if available_tasks.is_empty() {
        return None;
    }

    let affinity_noise: f64 = rand::random::<f64>() * 0.05;

    let mut scored: Vec<(String, f64)> = available_tasks
        .iter()
        .map(|task| {
            let task_tags = &task.tags;
            let affinity = jaccard_similarity(worker_tags, task_tags);
            let urgency = scent_field.read(&task.task_id, ScentType::Urgency);
            let difficulty = scent_field.read(&task.task_id, ScentType::Difficulty);
            let failure = scent_field.read(&task.task_id, ScentType::Failure);
            let progress = scent_field.read(&task.task_id, ScentType::Progress);

            let score = compute_task_score(
                worker_tags,
                task_tags,
                affinity,
                urgency,
                difficulty,
                failure,
                progress,
            );

            // Add small noise to prevent ties
            let noisy_score = score * (1.0 + affinity_noise);
            (task.task_id.clone(), noisy_score)
        })
        .collect();

    scored.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

    Some(scored[0].0.clone())
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    // -- Decay computation tests --

    #[test]
    fn test_decay_constant() {
        let lambda = decay_constant(100.0);
        let expected = std::f64::consts::LN_2 / 100.0;
        assert!((lambda - expected).abs() < 1e-10);
    }

    #[test]
    fn test_decay_constant_negative() {
        let lambda = decay_constant(-1.0);
        assert_eq!(lambda, 0.0);
    }

    #[test]
    fn test_decay_constant_zero() {
        let lambda = decay_constant(0.0);
        assert_eq!(lambda, 0.0);
    }

    #[test]
    fn test_compute_current_intensity_fresh() {
        let now = Utc::now();
        let intensity = compute_current_intensity(1.0, now, ScentType::Completion);
        assert!((intensity - 1.0).abs() < 0.01);
    }

    #[test]
    fn test_compute_current_intensity_zero() {
        let now = Utc::now();
        let intensity = compute_current_intensity(0.0, now, ScentType::Completion);
        assert_eq!(intensity, 0.0);
    }

    #[test]
    fn test_compute_current_intensity_negative() {
        let now = Utc::now();
        let intensity = compute_current_intensity(-0.5, now, ScentType::Completion);
        assert_eq!(intensity, 0.0);
    }

    #[test]
    fn test_compute_current_intensity_past() {
        let past = Utc::now() - chrono::Duration::seconds(300);
        // After one half-life of Completion (300s), should be ~0.5
        let intensity = compute_current_intensity(1.0, past, ScentType::Completion);
        assert!((intensity - 0.5).abs() < 0.01);
    }

    #[test]
    fn test_compute_current_intensity_urgency_grows() {
        let past = Utc::now() - chrono::Duration::seconds(300);
        // Urgency grows: 1.0 * (1 + 300/300) = 2.0
        let intensity = compute_current_intensity(1.0, past, ScentType::Urgency);
        assert!((intensity - 2.0).abs() < 0.01);
    }

    #[test]
    fn test_compute_current_intensity_decays_below_threshold() {
        // Far in the past, should decay below GC_THRESHOLD
        let far_past = Utc::now() - chrono::Duration::seconds(100000);
        let intensity = compute_current_intensity(0.5, far_past, ScentType::Progress);
        // Progress has 20s half-life, so after 100000s it should be essentially 0
        assert_eq!(intensity, 0.0);
    }

    // -- Jaccard similarity tests --

    #[test]
    fn test_jaccard_identical_sets() {
        let a = vec!["x".into(), "y".into(), "z".into()];
        let b = vec!["x".into(), "y".into(), "z".into()];
        assert!((jaccard_similarity(&a, &b) - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_jaccard_disjoint_sets() {
        let a = vec!["x".into(), "y".into()];
        let b = vec!["z".into(), "w".into()];
        assert!((jaccard_similarity(&a, &b) - 0.0).abs() < 1e-10);
    }

    #[test]
    fn test_jaccard_partial_overlap() {
        let a = vec!["x".into(), "y".into()];
        let b = vec!["y".into(), "z".into()];
        // intersection = {y}, union = {x, y, z}
        assert!((jaccard_similarity(&a, &b) - (1.0 / 3.0)).abs() < 1e-10);
    }

    #[test]
    fn test_jaccard_empty_sets() {
        let a: Vec<String> = vec![];
        let b: Vec<String> = vec![];
        assert!((jaccard_similarity(&a, &b) - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_jaccard_one_empty() {
        let a = vec!["x".into()];
        let b: Vec<String> = vec![];
        assert!((jaccard_similarity(&a, &b) - 0.0).abs() < 1e-10);
    }

    // -- Task scoring tests --

    #[test]
    fn test_task_score_perfect_match() {
        let worker = vec!["finance".into(), "analysis".into()];
        let task = vec!["finance".into(), "analysis".into()];
        let score = compute_task_score(&worker, &task, -1.0, 0.0, 0.0, 0.0, 0.0);
        // A=1.0, U=1.0, D=0, F=0, R=1.0
        // S = 1^2 * 1^1.5 * 1 * 1 * 1^1.2 = 1.0
        assert!((score - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_task_score_no_match() {
        let worker = vec!["code".into()];
        let task = vec!["finance".into()];
        let score = compute_task_score(&worker, &task, -1.0, 0.0, 0.0, 0.0, 0.0);
        // A=0.0, so S = 0
        assert_eq!(score, 0.0);
    }

    #[test]
    fn test_task_score_high_difficulty_dampens() {
        let worker = vec!["x".into()];
        let task = vec!["x".into()];
        let score_low = compute_task_score(&worker, &task, -1.0, 0.0, 0.0, 0.0, 0.0);
        let score_high = compute_task_score(&worker, &task, -1.0, 0.0, 0.8, 0.0, 0.0);
        assert!(score_low > score_high);
    }

    #[test]
    fn test_task_score_high_failure_dampens() {
        let worker = vec!["x".into()];
        let task = vec!["x".into()];
        let score_low = compute_task_score(&worker, &task, -1.0, 0.0, 0.0, 0.0, 0.0);
        let score_high = compute_task_score(&worker, &task, -1.0, 0.0, 0.0, 0.9, 0.0);
        assert!(score_low > score_high);
    }

    #[test]
    fn test_task_score_progress_boosts() {
        let worker = vec!["x".into()];
        let task = vec!["x".into()];
        let score_none = compute_task_score(&worker, &task, -1.0, 0.0, 0.0, 0.0, 0.0);
        let score_progress = compute_task_score(&worker, &task, -1.0, 0.0, 0.0, 0.0, 0.5);
        assert!(score_progress > score_none);
    }

    #[test]
    fn test_task_score_urgency_boosts() {
        let worker = vec!["x".into()];
        let task = vec!["x".into()];
        let score_none = compute_task_score(&worker, &task, -1.0, 0.0, 0.0, 0.0, 0.0);
        let score_urgent = compute_task_score(&worker, &task, -1.0, 0.5, 0.0, 0.0, 0.0);
        assert!(score_urgent > score_none);
    }

    // -- ScentField tests --

    #[test]
    fn test_scent_field_new() {
        let field = ScentField::new();
        assert!(field.is_empty());
        assert_eq!(field.len(), 0);
    }

    #[test]
    fn test_scent_field_emit_and_read() {
        let mut field = ScentField::new();
        field.emit_signal("t1", "w1", ScentType::Urgency, 0.8);

        let intensity = field.read("t1", ScentType::Urgency);
        assert!((intensity - 0.8).abs() < 0.01);
    }

    #[test]
    fn test_scent_field_read_nonexistent() {
        let field = ScentField::new();
        let intensity = field.read("nonexistent", ScentType::Completion);
        assert_eq!(intensity, 0.0);
    }

    #[test]
    fn test_scent_field_read_all() {
        let mut field = ScentField::new();
        field.emit_signal("t1", "w1", ScentType::Urgency, 0.5);
        field.emit_signal("t1", "w2", ScentType::Completion, 0.3);

        let all = field.read_all("t1");
        assert!(all.contains_key(&ScentType::Urgency));
        assert!(all.contains_key(&ScentType::Completion));
    }

    #[test]
    fn test_scent_field_aggregation() {
        let mut field = ScentField::new();
        field.emit_signal("t1", "w1", ScentType::Urgency, 0.3);
        field.emit_signal("t1", "w2", ScentType::Urgency, 0.4);
        field.emit_signal("t1", "w3", ScentType::Urgency, 0.3);

        let intensity = field.read("t1", ScentType::Urgency);
        assert!((intensity - 1.0).abs() < 0.01);
    }

    #[test]
    fn test_scent_field_garbage_collect() {
        let mut field = ScentField::new();
        // Emit a weak progress signal (short half-life)
        field.emit_signal("t1", "w1", ScentType::Progress, 0.02);

        // Simulate time passage by manually aging signals
        let old_time = Utc::now() - chrono::Duration::seconds(200);
        field.signals[0].emitted_at = old_time;

        let removed = field.garbage_collect();
        assert!(removed > 0);
    }

    #[test]
    fn test_scent_field_signals_for_task() {
        let mut field = ScentField::new();
        field.emit_signal("t1", "w1", ScentType::Urgency, 0.5);
        field.emit_signal("t2", "w1", ScentType::Completion, 0.5);

        let t1_signals = field.signals_for_task("t1");
        assert_eq!(t1_signals.len(), 1);
    }

    #[test]
    fn test_scent_field_signals_by_worker() {
        let mut field = ScentField::new();
        field.emit_signal("t1", "w1", ScentType::Urgency, 0.5);
        field.emit_signal("t2", "w2", ScentType::Completion, 0.5);

        let w1_signals = field.signals_by_worker("w1");
        assert_eq!(w1_signals.len(), 1);
    }

    #[test]
    fn test_scent_field_clear() {
        let mut field = ScentField::new();
        field.emit_signal("t1", "w1", ScentType::Urgency, 0.5);
        field.clear();
        assert!(field.is_empty());
    }

    #[test]
    fn test_scent_field_read_at() {
        let mut field = ScentField::new();
        field.emit_signal("t1", "w1", ScentType::Completion, 1.0);

        let past = field.signals[0].emitted_at + chrono::Duration::seconds(300);
        let intensity = field.read_at("t1", ScentType::Completion, past);
        assert!((intensity - 0.5).abs() < 0.01);
    }

    // -- Select best task tests --

    #[test]
    fn test_select_best_task_none_available() {
        let field = ScentField::new();
        let result = select_best_task("w1", &[], &[], &field);
        assert!(result.is_none());
    }

    #[test]
    fn test_select_best_task_single() {
        let field = ScentField::new();
        let tasks = vec![DAGTask::new("t1", "desc", "agent", vec!["x".into()], vec![], 3)];
        let refs: Vec<&DAGTask> = tasks.iter().collect();
        let result = select_best_task("w1", &["x".into()], &refs, &field);
        assert_eq!(result, Some("t1".to_string()));
    }

    #[test]
    fn test_select_best_task_prefers_match() {
        let mut field = ScentField::new();
        let t1 = DAGTask::new("t1", "desc", "agent", vec!["finance".into()], vec![], 3);
        let t2 = DAGTask::new("t2", "desc", "agent", vec!["code".into()], vec![], 3);

        let tasks = vec![&t1, &t2];
        let result = select_best_task("w1", &["finance".into()], &tasks, &field);
        assert_eq!(result, Some("t1".to_string()));
    }

    // -- Half-life table tests --

    #[test]
    fn test_scent_half_lives_table() {
        assert_eq!(SCENT_HALF_LIVES.len(), 6);
        for (scent, half_life) in &SCENT_HALF_LIVES {
            assert_eq!(*half_life, scent.default_half_life());
        }
    }

    #[test]
    fn test_urgency_is_growing() {
        assert!(ScentType::Urgency.is_growing());
        assert!(!ScentType::Completion.is_growing());
        assert!(!ScentType::Failure.is_growing());
        assert!(!ScentType::Difficulty.is_growing());
        assert!(!ScentType::Progress.is_growing());
        assert!(!ScentType::HelpWanted.is_growing());
    }
}
