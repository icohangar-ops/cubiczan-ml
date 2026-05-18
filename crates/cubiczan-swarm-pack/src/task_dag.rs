//! DAG Task Planning with topological sort and critical path analysis.
//!
//! This module provides a complete DAG (Directed Acyclic Graph) task planner:
//! - Validation via Kahn's algorithm (cycle detection + topological sort)
//! - Critical path computation via longest-path BFS
//! - Task state machine transitions (Pending → Ready → Active → Complete/Retry/Escalate)
//! - Ready task activation and worker claiming

use crate::types::*;
use serde::{Deserialize, Serialize};
use std::collections::{HashMap, HashSet, VecDeque};
use thiserror::Error;

// ---------------------------------------------------------------------------
// Errors
// ---------------------------------------------------------------------------

/// Errors that can occur during DAG operations.
#[derive(Debug, Error)]
pub enum DAGError {
    #[error("cycle detected in task graph")]
    CycleDetected,

    #[error("task not found: {0}")]
    TaskNotFound(String),

    #[error("invalid transition from {from:?} to {to:?}")]
    InvalidTransition {
        from: TaskStatus,
        to: TaskStatus,
    },

    #[error("task {0} is not in Ready state (current: {1:?})")]
    NotReady(String, TaskStatus),

    #[error("task {0} is not in Active state (current: {1:?})")]
    NotActive(String, TaskStatus),

    #[error("task {0} is already claimed by worker {1}")]
    AlreadyClaimed(String, String),

    #[error("dependency not found: {0}")]
    DependencyNotFound(String),

    #[error("self-dependency detected: {0}")]
    SelfDependency(String),
}

// ---------------------------------------------------------------------------
// Task Specification
// ---------------------------------------------------------------------------

/// Specification for creating a new task in the DAG.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TaskSpec {
    pub task_id: String,
    pub description: String,
    pub agent_type: String,
    pub tags: Vec<String>,
    pub dependencies: Vec<String>,
    pub max_retries: u32,
}

impl TaskSpec {
    /// Create a new task specification.
    pub fn new(
        task_id: impl Into<String>,
        description: impl Into<String>,
        agent_type: impl Into<String>,
    ) -> Self {
        TaskSpec {
            task_id: task_id.into(),
            description: description.into(),
            agent_type: agent_type.into(),
            tags: Vec::new(),
            dependencies: Vec::new(),
            max_retries: 3,
        }
    }

    /// Builder: set tags.
    pub fn with_tags(mut self, tags: Vec<String>) -> Self {
        self.tags = tags;
        self
    }

    /// Builder: set dependencies.
    pub fn depends_on(mut self, deps: Vec<String>) -> Self {
        self.dependencies = deps;
        self
    }

    /// Builder: set max retries.
    pub fn with_max_retries(mut self, max: u32) -> Self {
        self.max_retries = max;
        self
    }
}

// ---------------------------------------------------------------------------
// Core DAG operations
// ---------------------------------------------------------------------------

/// Validate the DAG and return a topological ordering of task IDs.
///
/// Uses Kahn's algorithm:
/// 1. Compute in-degrees for all nodes
/// 2. Add all zero in-degree nodes to a queue
/// 3. Process the queue, reducing in-degrees
/// 4. If not all nodes are processed, a cycle exists
pub fn validate_dag(tasks: &HashMap<String, DAGTask>) -> Result<Vec<String>, DAGError> {
    if tasks.is_empty() {
        return Ok(Vec::new());
    }

    // Compute in-degrees (only count dependencies that exist in the graph)
    let mut in_degree: HashMap<&str, u32> = HashMap::new();
    let mut dependents_map: HashMap<&str, Vec<&str>> = HashMap::new();

    for (id, task) in tasks {
        in_degree.entry(id.as_str()).or_insert(0);
        for dep in &task.dependencies {
            if !tasks.contains_key(dep) {
                // Skip dependencies not in the graph
                continue;
            }
            if dep == id.as_str() {
                return Err(DAGError::SelfDependency(id.clone()));
            }
            *in_degree.entry(dep).or_insert(0);
            *in_degree.entry(id.as_str()).or_insert(0) += 1;
            dependents_map
                .entry(dep)
                .or_insert_with(Vec::new)
                .push(id);
        }
    }

    // Initialize queue with zero in-degree nodes
    let mut queue: VecDeque<&str> = VecDeque::new();
    for (&id, &degree) in &in_degree {
        if degree == 0 {
            queue.push_back(id);
        }
    }

    let mut sorted = Vec::with_capacity(tasks.len());
    let mut processed = 0;

    while let Some(id) = queue.pop_front() {
        sorted.push(id.to_string());
        processed += 1;

        if let Some(deps) = dependents_map.get(id) {
            for &dep_id in deps {
                if let Some(degree) = in_degree.get_mut(dep_id) {
                    *degree -= 1;
                    if *degree == 0 {
                        queue.push_back(dep_id);
                    }
                }
            }
        }
    }

    if processed != tasks.len() {
        return Err(DAGError::CycleDetected);
    }

    Ok(sorted)
}

/// Compute the critical path through the DAG.
///
/// Returns a tuple of (critical_path_task_ids, critical_path_length).
/// Uses BFS with longest-path tracking after topological sort.
pub fn compute_critical_path(tasks: &HashMap<String, DAGTask>) -> Result<(Vec<String>, u32), DAGError> {
    if tasks.is_empty() {
        return Ok((Vec::new(), 0));
    }

    let sorted = validate_dag(tasks)?;

    // Compute longest distance from any root to each node
    let mut dist: HashMap<&str, u32> = HashMap::new();
    let mut predecessor: HashMap<&str, &str> = HashMap::new();

    for id in &sorted {
        let task = &tasks[id.as_str()];
        // Start with weight 1 for this task
        let mut max_dist = 1u32;
        let mut best_pred: Option<&str> = None;

        for dep in &task.dependencies {
            if let Some(&d) = dist.get(dep.as_str()) {
                let candidate = d + 1;
                if candidate > max_dist {
                    max_dist = candidate;
                    best_pred = Some(dep);
                }
            }
        }

        dist.insert(id, max_dist);
        if let Some(pred) = best_pred {
            predecessor.insert(id, pred);
        }
    }

    // Find the node with maximum distance (end of critical path)
    let mut end_node: &str = &sorted[0];
    let mut max_len: u32 = 0;

    for id in &sorted {
        if let Some(&d) = dist.get(id.as_str()) {
            if d > max_len {
                max_len = d;
                end_node = id;
            }
        }
    }

    // Reconstruct the critical path by backtracking
    let mut path = Vec::new();
    let mut current: Option<&str> = Some(end_node);

    while let Some(node) = current {
        path.push(node.to_string());
        current = predecessor.get(node).copied();
    }

    path.reverse();
    Ok((path, max_len))
}

/// Compute theoretical speedup using Amdahl's law approximation.
///
/// speedup = total_tasks / critical_path_length
pub fn compute_theoretical_speedup(tasks: &HashMap<String, DAGTask>, critical_path_len: u32) -> f64 {
    if critical_path_len == 0 {
        return 1.0;
    }
    let total = tasks.len() as f64;
    let cp = critical_path_len as f64;
    (total / cp).max(1.0)
}

/// Build a complete task graph from task specifications.
///
/// - Creates all tasks in Pending state
/// - Populates dependents lists
/// - Validates the DAG
/// - Computes critical path and speedup
pub fn build_task_graph(specs: Vec<TaskSpec>) -> Result<TaskGraph, DAGError> {
    let graph_id = uuid::Uuid::new_v4().to_string();
    let mut tasks: HashMap<String, DAGTask> = HashMap::new();

    // Validate all dependencies exist
    let spec_ids: HashSet<&str> = specs.iter().map(|s| s.task_id.as_str()).collect();

    for spec in &specs {
        for dep in &spec.dependencies {
            if !spec_ids.contains(dep.as_str()) {
                return Err(DAGError::DependencyNotFound(dep.clone()));
            }
            if dep == &spec.task_id {
                return Err(DAGError::SelfDependency(spec.task_id.clone()));
            }
        }
    }

    // Create tasks
    for spec in specs {
        let task = DAGTask::new(
            &spec.task_id,
            &spec.description,
            &spec.agent_type,
            spec.tags,
            spec.dependencies.clone(),
            spec.max_retries,
        );
        tasks.insert(spec.task_id, task);
    }

    // Populate dependents
    let task_ids: Vec<String> = tasks.keys().cloned().collect();
    for id in &task_ids {
        let deps: Vec<String> = tasks[id].dependencies.clone();
        for dep in deps {
            if let Some(dep_task) = tasks.get_mut(&dep) {
                dep_task.dependents.push(id.clone());
            }
        }
    }

    // Validate DAG
    validate_dag(&tasks)?;

    // Compute critical path
    let (critical_path, cp_len) = compute_critical_path(&tasks)?;
    let speedup = compute_theoretical_speedup(&tasks, cp_len);

    Ok(TaskGraph {
        graph_id,
        tasks,
        critical_path_length: cp_len,
        theoretical_speedup: speedup,
    })
}

/// Get the IDs of tasks that are ready to be activated.
///
/// A task is ready when:
/// - Its status is Pending, AND
/// - All its dependencies are Complete
pub fn get_ready_tasks(tasks: &HashMap<String, DAGTask>) -> Vec<String> {
    let mut ready = Vec::new();

    for (id, task) in tasks {
        if task.status != TaskStatus::Pending {
            continue;
        }
        if task.worker_id.is_some() {
            continue;
        }
        let all_deps_complete = task
            .dependencies
            .iter()
            .all(|dep_id| {
                tasks
                    .get(dep_id)
                    .map(|t| t.status == TaskStatus::Complete)
                    .unwrap_or(true) // Missing deps treated as met
            });
        if all_deps_complete {
            ready.push(id.clone());
        }
    }

    ready
}

/// Activate ready tasks by transitioning them from Pending → Ready,
/// and from Retry → Ready. Returns the IDs of activated tasks.
pub fn activate_ready_tasks(tasks: &mut HashMap<String, DAGTask>) -> Vec<String> {
    let mut activated = Vec::new();

    // First pass: collect status snapshot to avoid borrow conflicts
    let pending_deps: Vec<(String, Vec<String>, Option<String>)> = tasks
        .iter()
        .filter(|(_, t)| t.status == TaskStatus::Pending)
        .map(|(id, t)| (id.clone(), t.dependencies.clone(), t.worker_id.clone()))
        .collect();

    let retry_ids: Vec<String> = tasks
        .iter()
        .filter(|(_, t)| t.status == TaskStatus::Retry)
        .map(|(id, _)| id.clone())
        .collect();

    // Check which pending tasks have all deps complete
    let mut ready_pending = Vec::new();
    for (id, deps, worker_id) in &pending_deps {
        if worker_id.is_some() {
            continue;
        }
        let all_complete = deps.iter().all(|dep_id| {
            tasks
                .get(dep_id)
                .map(|t| t.status == TaskStatus::Complete)
                .unwrap_or(true)
        });
        if all_complete {
            ready_pending.push(id.clone());
        }
    }

    // Second pass: mutate statuses
    for id in &ready_pending {
        if let Some(task) = tasks.get_mut(id) {
            task.status = TaskStatus::Ready;
            activated.push(id.clone());
        }
    }
    for id in &retry_ids {
        if let Some(task) = tasks.get_mut(id) {
            task.status = TaskStatus::Ready;
            activated.push(id.clone());
        }
    }

    activated
}

/// Claim a task for a worker. Transitions the task from Ready → Active.
///
/// Returns an error if the task is not in Ready state or is already claimed.
pub fn claim_task(
    tasks: &mut HashMap<String, DAGTask>,
    task_id: &str,
    worker_id: &str,
) -> Result<(), DAGError> {
    let task = tasks
        .get_mut(task_id)
        .ok_or_else(|| DAGError::TaskNotFound(task_id.to_string()))?;

    if task.status != TaskStatus::Ready {
        return Err(DAGError::NotReady(
            task_id.to_string(),
            task.status,
        ));
    }

    if let Some(ref existing) = task.worker_id {
        return Err(DAGError::AlreadyClaimed(
            task_id.to_string(),
            existing.clone(),
        ));
    }

    task.status = TaskStatus::Active;
    task.worker_id = Some(worker_id.to_string());
    task.started_at = Some(chrono::Utc::now());

    Ok(())
}

/// Complete a task. Transitions the task from Active → Complete.
///
/// Sets the result and completed_at timestamp.
pub fn complete_task(
    tasks: &mut HashMap<String, DAGTask>,
    task_id: &str,
    result: serde_json::Value,
) -> Result<(), DAGError> {
    let task = tasks
        .get_mut(task_id)
        .ok_or_else(|| DAGError::TaskNotFound(task_id.to_string()))?;

    if task.status != TaskStatus::Active {
        return Err(DAGError::NotActive(
            task_id.to_string(),
            task.status,
        ));
    }

    task.status = TaskStatus::Complete;
    task.result = Some(result);
    task.completed_at = Some(chrono::Utc::now());

    Ok(())
}

/// Fail a task. Transitions the task from Active → Retry (if retries available)
/// or Active → Escalate (if max retries exceeded).
///
/// Returns the new status.
pub fn fail_task(
    tasks: &mut HashMap<String, DAGTask>,
    task_id: &str,
    error: &str,
) -> Result<TaskStatus, DAGError> {
    let task = tasks
        .get_mut(task_id)
        .ok_or_else(|| DAGError::TaskNotFound(task_id.to_string()))?;

    if task.status != TaskStatus::Active {
        return Err(DAGError::NotActive(
            task_id.to_string(),
            task.status,
        ));
    }

    if task.retries < task.max_retries {
        task.retries += 1;
        task.status = TaskStatus::Retry;
        task.worker_id = None;
        task.started_at = None;
        // Store error in result
        task.result = Some(serde_json::json!({
            "error": error,
            "retry_count": task.retries,
        }));
        Ok(TaskStatus::Retry)
    } else {
        task.status = TaskStatus::Escalate;
        task.result = Some(serde_json::json!({
            "error": error,
            "escalated": true,
            "retry_count": task.retries,
        }));
        Ok(TaskStatus::Escalate)
    }
}

/// Block a task. Transitions from Active → Blocked.
pub fn block_task(
    tasks: &mut HashMap<String, DAGTask>,
    task_id: &str,
    reason: &str,
) -> Result<(), DAGError> {
    let task = tasks
        .get_mut(task_id)
        .ok_or_else(|| DAGError::TaskNotFound(task_id.to_string()))?;

    if task.status != TaskStatus::Active {
        return Err(DAGError::NotActive(
            task_id.to_string(),
            task.status,
        ));
    }

    task.status = TaskStatus::Blocked;
    task.result = Some(serde_json::json!({
        "blocked": true,
        "reason": reason,
    }));

    Ok(())
}

/// Unblock a task by transitioning Blocked → Pending.
pub fn unblock_task(
    tasks: &mut HashMap<String, DAGTask>,
    task_id: &str,
) -> Result<(), DAGError> {
    let task = tasks
        .get_mut(task_id)
        .ok_or_else(|| DAGError::TaskNotFound(task_id.to_string()))?;

    if task.status != TaskStatus::Blocked {
        return Err(DAGError::InvalidTransition {
            from: task.status,
            to: TaskStatus::Pending,
        });
    }

    task.status = TaskStatus::Pending;
    task.worker_id = None;
    task.started_at = None;

    Ok(())
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    // -- validate_dag tests --

    #[test]
    fn test_validate_empty_dag() {
        let tasks: HashMap<String, DAGTask> = HashMap::new();
        let result = validate_dag(&tasks).unwrap();
        assert!(result.is_empty());
    }

    #[test]
    fn test_validate_single_task() {
        let mut tasks = HashMap::new();
        tasks.insert("t1".into(), DAGTask::new("t1", "desc", "agent", vec![], vec![], 3));
        let result = validate_dag(&tasks).unwrap();
        assert_eq!(result, vec!["t1"]);
    }

    #[test]
    fn test_validate_linear_dag() {
        let mut tasks = HashMap::new();
        tasks.insert("t1".into(), DAGTask::new("t1", "1", "a", vec![], vec![], 3));
        tasks.insert("t2".into(), DAGTask::new("t2", "2", "a", vec![], vec!["t1".into()], 3));
        tasks.insert("t3".into(), DAGTask::new("t3", "3", "a", vec![], vec!["t2".into()], 3));
        // Update dependents
        tasks.get_mut("t1").unwrap().dependents = vec!["t2".into()];
        tasks.get_mut("t2").unwrap().dependents = vec!["t3".into()];

        let result = validate_dag(&tasks).unwrap();
        assert_eq!(result.len(), 3);
        // t1 must come before t2, t2 before t3
        let pos = |id: &str| result.iter().position(|x| x == id).unwrap();
        assert!(pos("t1") < pos("t2"));
        assert!(pos("t2") < pos("t3"));
    }

    #[test]
    fn test_validate_diamond_dag() {
        let mut tasks = HashMap::new();
        tasks.insert("root".into(), DAGTask::new("root", "r", "a", vec![], vec![], 3));
        tasks.insert("left".into(), DAGTask::new("left", "l", "a", vec![], vec!["root".into()], 3));
        tasks.insert("right".into(), DAGTask::new("right", "ri", "a", vec![], vec!["root".into()], 3));
        tasks.insert("join".into(), DAGTask::new("join", "j", "a", vec![], vec!["left".into(), "right".into()], 3));
        tasks.get_mut("root").unwrap().dependents = vec!["left".into(), "right".into()];
        tasks.get_mut("left").unwrap().dependents = vec!["join".into()];
        tasks.get_mut("right").unwrap().dependents = vec!["join".into()];

        let result = validate_dag(&tasks).unwrap();
        assert_eq!(result.len(), 4);
        let pos = |id: &str| result.iter().position(|x| x == id).unwrap();
        assert!(pos("root") < pos("left"));
        assert!(pos("root") < pos("right"));
        assert!(pos("left") < pos("join"));
        assert!(pos("right") < pos("join"));
    }

    #[test]
    fn test_detect_cycle() {
        let mut tasks = HashMap::new();
        tasks.insert("a".into(), DAGTask::new("a", "a", "a", vec![], vec!["b".into()], 3));
        tasks.insert("b".into(), DAGTask::new("b", "b", "a", vec![], vec!["a".into()], 3));
        tasks.get_mut("a").unwrap().dependents = vec!["b".into()];
        tasks.get_mut("b").unwrap().dependents = vec!["a".into()];

        let result = validate_dag(&tasks);
        assert!(matches!(result, Err(DAGError::CycleDetected)));
    }

    #[test]
    fn test_detect_three_node_cycle() {
        let mut tasks = HashMap::new();
        tasks.insert("a".into(), DAGTask::new("a", "a", "a", vec![], vec!["c".into()], 3));
        tasks.insert("b".into(), DAGTask::new("b", "b", "a", vec![], vec!["a".into()], 3));
        tasks.insert("c".into(), DAGTask::new("c", "c", "a", vec![], vec!["b".into()], 3));
        tasks.get_mut("a").unwrap().dependents = vec!["b".into()];
        tasks.get_mut("b").unwrap().dependents = vec!["c".into()];
        tasks.get_mut("c").unwrap().dependents = vec!["a".into()];

        let result = validate_dag(&tasks);
        assert!(matches!(result, Err(DAGError::CycleDetected)));
    }

    #[test]
    fn test_detect_self_dependency() {
        let mut tasks = HashMap::new();
        tasks.insert("a".into(), DAGTask::new("a", "a", "a", vec![], vec!["a".into()], 3));

        let result = validate_dag(&tasks);
        assert!(matches!(result, Err(DAGError::SelfDependency(_))));
    }

    // -- compute_critical_path tests --

    #[test]
    fn test_critical_path_empty() {
        let tasks: HashMap<String, DAGTask> = HashMap::new();
        let (path, len) = compute_critical_path(&tasks).unwrap();
        assert!(path.is_empty());
        assert_eq!(len, 0);
    }

    #[test]
    fn test_critical_path_single() {
        let mut tasks = HashMap::new();
        tasks.insert("t1".into(), DAGTask::new("t1", "desc", "a", vec![], vec![], 3));
        let (path, len) = compute_critical_path(&tasks).unwrap();
        assert_eq!(path, vec!["t1"]);
        assert_eq!(len, 1);
    }

    #[test]
    fn test_critical_path_linear() {
        let mut tasks = HashMap::new();
        tasks.insert("t1".into(), DAGTask::new("t1", "1", "a", vec![], vec![], 3));
        tasks.insert("t2".into(), DAGTask::new("t2", "2", "a", vec![], vec!["t1".into()], 3));
        tasks.insert("t3".into(), DAGTask::new("t3", "3", "a", vec![], vec!["t2".into()], 3));
        tasks.get_mut("t1").unwrap().dependents = vec!["t2".into()];
        tasks.get_mut("t2").unwrap().dependents = vec!["t3".into()];

        let (path, len) = compute_critical_path(&tasks).unwrap();
        assert_eq!(path, vec!["t1", "t2", "t3"]);
        assert_eq!(len, 3);
    }

    #[test]
    fn test_critical_path_diamond() {
        let mut tasks = HashMap::new();
        tasks.insert("root".into(), DAGTask::new("root", "r", "a", vec![], vec![], 3));
        tasks.insert("left".into(), DAGTask::new("left", "l", "a", vec![], vec!["root".into()], 3));
        tasks.insert("right".into(), DAGTask::new("right", "ri", "a", vec![], vec!["root".into()], 3));
        tasks.insert("join".into(), DAGTask::new("join", "j", "a", vec![], vec!["left".into(), "right".into()], 3));
        tasks.get_mut("root").unwrap().dependents = vec!["left".into(), "right".into()];
        tasks.get_mut("left").unwrap().dependents = vec!["join".into()];
        tasks.get_mut("right").unwrap().dependents = vec!["join".into()];

        let (path, len) = compute_critical_path(&tasks).unwrap();
        assert_eq!(len, 3); // root → left/right → join = 3
        assert_eq!(path.len(), 3);
        assert_eq!(path[0], "root");
        assert_eq!(path[2], "join");
    }

    #[test]
    fn test_critical_path_complex() {
        let mut tasks = HashMap::new();
        // A → B → D → F
        // A → C → E → F
        // B → E (cross edge)
        tasks.insert("a".into(), DAGTask::new("a", "a", "a", vec![], vec![], 3));
        tasks.insert("b".into(), DAGTask::new("b", "b", "a", vec![], vec!["a".into()], 3));
        tasks.insert("c".into(), DAGTask::new("c", "c", "a", vec![], vec!["a".into()], 3));
        tasks.insert("d".into(), DAGTask::new("d", "d", "a", vec![], vec!["b".into()], 3));
        tasks.insert("e".into(), DAGTask::new("e", "e", "a", vec![], vec!["c".into(), "b".into()], 3));
        tasks.insert("f".into(), DAGTask::new("f", "f", "a", vec![], vec!["d".into(), "e".into()], 3));
        tasks.get_mut("a").unwrap().dependents = vec!["b".into(), "c".into()];
        tasks.get_mut("b").unwrap().dependents = vec!["d".into(), "e".into()];
        tasks.get_mut("c").unwrap().dependents = vec!["e".into()];
        tasks.get_mut("d").unwrap().dependents = vec!["f".into()];
        tasks.get_mut("e").unwrap().dependents = vec!["f".into()];

        let (_, len) = compute_critical_path(&tasks).unwrap();
        assert_eq!(len, 4); // a → b → e → f or a → b → d → f
    }

    // -- compute_theoretical_speedup tests --

    #[test]
    fn test_speedup_single_task() {
        let mut tasks = HashMap::new();
        tasks.insert("t1".into(), DAGTask::new("t1", "d", "a", vec![], vec![], 3));
        let speedup = compute_theoretical_speedup(&tasks, 1);
        assert_eq!(speedup, 1.0);
    }

    #[test]
    fn test_speedup_diamond() {
        // 4 tasks, critical path 3 → speedup = 4/3 ≈ 1.33
        let speedup = compute_theoretical_speedup(&HashMap::new(), 3);
        // With 0 tasks, returns 1.0
        assert_eq!(speedup, 1.0);

        let mut tasks = HashMap::new();
        tasks.insert("a".into(), DAGTask::new("a", "a", "a", vec![], vec![], 3));
        tasks.insert("b".into(), DAGTask::new("b", "b", "a", vec![], vec![], 3));
        tasks.insert("c".into(), DAGTask::new("c", "c", "a", vec![], vec![], 3));
        tasks.insert("d".into(), DAGTask::new("d", "d", "a", vec![], vec![], 3));
        let speedup = compute_theoretical_speedup(&tasks, 3);
        assert!((speedup - 4.0 / 3.0).abs() < 1e-10);
    }

    #[test]
    fn test_speedup_zero_critical_path() {
        let mut tasks = HashMap::new();
        tasks.insert("t1".into(), DAGTask::new("t1", "d", "a", vec![], vec![], 3));
        let speedup = compute_theoretical_speedup(&tasks, 0);
        assert_eq!(speedup, 1.0);
    }

    // -- build_task_graph tests --

    #[test]
    fn test_build_task_graph_empty() {
        let graph = build_task_graph(vec![]).unwrap();
        assert!(graph.is_empty());
        assert_eq!(graph.critical_path_length, 0);
    }

    #[test]
    fn test_build_task_graph_linear() {
        let specs = vec![
            TaskSpec::new("a", "task a", "agent").depends_on(vec![]),
            TaskSpec::new("b", "task b", "agent").depends_on(vec!["a".into()]),
            TaskSpec::new("c", "task c", "agent").depends_on(vec!["b".into()]),
        ];
        let graph = build_task_graph(specs).unwrap();
        assert_eq!(graph.len(), 3);
        assert_eq!(graph.critical_path_length, 3);
        assert!((graph.theoretical_speedup - 1.0).abs() < 1e-10);
        assert!(graph.tasks["a"].dependents.contains(&"b".to_string()));
        assert!(graph.tasks["b"].dependents.contains(&"c".to_string()));
    }

    #[test]
    fn test_build_task_graph_missing_dependency() {
        let specs = vec![
            TaskSpec::new("a", "task a", "agent").depends_on(vec!["z".into()]),
        ];
        let result = build_task_graph(specs);
        assert!(matches!(result, Err(DAGError::DependencyNotFound(_))));
    }

    #[test]
    fn test_build_task_graph_self_dependency() {
        let specs = vec![
            TaskSpec::new("a", "task a", "agent").depends_on(vec!["a".into()]),
        ];
        let result = build_task_graph(specs);
        assert!(matches!(result, Err(DAGError::SelfDependency(_))));
    }

    #[test]
    fn test_build_task_graph_with_tags() {
        let specs = vec![
            TaskSpec::new("a", "task a", "analyst")
                .with_tags(vec!["financial".into(), "research".into()]),
        ];
        let graph = build_task_graph(specs).unwrap();
        assert_eq!(graph.tasks["a"].tags, vec!["financial", "research"]);
    }

    // -- get_ready_tasks tests --

    #[test]
    fn test_get_ready_no_deps() {
        let mut tasks = HashMap::new();
        tasks.insert("t1".into(), DAGTask::new("t1", "d", "a", vec![], vec![], 3));
        let ready = get_ready_tasks(&tasks);
        assert_eq!(ready, vec!["t1"]);
    }

    #[test]
    fn test_get_ready_with_unmet_deps() {
        let mut tasks = HashMap::new();
        tasks.insert("t1".into(), DAGTask::new("t1", "d", "a", vec![], vec![], 3));
        tasks.insert("t2".into(), DAGTask::new("t2", "d", "a", vec![], vec!["t1".into()], 3));
        let ready = get_ready_tasks(&tasks);
        assert_eq!(ready, vec!["t1"]);
    }

    #[test]
    fn test_get_ready_with_met_deps() {
        let mut tasks = HashMap::new();
        let mut t1 = DAGTask::new("t1", "d", "a", vec![], vec![], 3);
        t1.status = TaskStatus::Complete;
        tasks.insert("t1".into(), t1);
        tasks.insert("t2".into(), DAGTask::new("t2", "d", "a", vec![], vec!["t1".into()], 3));
        let ready = get_ready_tasks(&tasks);
        assert_eq!(ready, vec!["t2"]);
    }

    // -- activate_ready_tasks tests --

    #[test]
    fn test_activate_ready_tasks_basic() {
        let mut tasks = HashMap::new();
        tasks.insert("t1".into(), DAGTask::new("t1", "d", "a", vec![], vec![], 3));
        let activated = activate_ready_tasks(&mut tasks);
        assert_eq!(activated, vec!["t1"]);
        assert_eq!(tasks["t1"].status, TaskStatus::Ready);
    }

    #[test]
    fn test_activate_retry_tasks() {
        let mut tasks = HashMap::new();
        let mut t1 = DAGTask::new("t1", "d", "a", vec![], vec![], 3);
        t1.status = TaskStatus::Retry;
        tasks.insert("t1".into(), t1);
        let activated = activate_ready_tasks(&mut tasks);
        assert_eq!(activated, vec!["t1"]);
        assert_eq!(tasks["t1"].status, TaskStatus::Ready);
    }

    #[test]
    fn test_activate_skips_active() {
        let mut tasks = HashMap::new();
        let mut t1 = DAGTask::new("t1", "d", "a", vec![], vec![], 3);
        t1.status = TaskStatus::Active;
        tasks.insert("t1".into(), t1);
        let activated = activate_ready_tasks(&mut tasks);
        assert!(activated.is_empty());
    }

    // -- claim_task tests --

    #[test]
    fn test_claim_task_success() {
        let mut tasks = HashMap::new();
        let mut t1 = DAGTask::new("t1", "d", "a", vec![], vec![], 3);
        t1.status = TaskStatus::Ready;
        tasks.insert("t1".into(), t1);

        claim_task(&mut tasks, "t1", "worker1").unwrap();
        assert_eq!(tasks["t1"].status, TaskStatus::Active);
        assert_eq!(tasks["t1"].worker_id.as_deref(), Some("worker1"));
        assert!(tasks["t1"].started_at.is_some());
    }

    #[test]
    fn test_claim_task_not_ready() {
        let mut tasks = HashMap::new();
        tasks.insert("t1".into(), DAGTask::new("t1", "d", "a", vec![], vec![], 3));

        let result = claim_task(&mut tasks, "t1", "worker1");
        assert!(matches!(result, Err(DAGError::NotReady(..))));
    }

    #[test]
    fn test_claim_task_already_claimed() {
        let mut tasks = HashMap::new();
        let mut t1 = DAGTask::new("t1", "d", "a", vec![], vec![], 3);
        t1.status = TaskStatus::Ready;
        t1.worker_id = Some("worker1".into());
        tasks.insert("t1".into(), t1);

        let result = claim_task(&mut tasks, "t1", "worker2");
        assert!(matches!(result, Err(DAGError::AlreadyClaimed(..))));
    }

    #[test]
    fn test_claim_task_not_found() {
        let mut tasks: HashMap<String, DAGTask> = HashMap::new();
        let result = claim_task(&mut tasks, "nonexistent", "w1");
        assert!(matches!(result, Err(DAGError::TaskNotFound(_))));
    }

    // -- complete_task tests --

    #[test]
    fn test_complete_task_success() {
        let mut tasks = HashMap::new();
        let mut t1 = DAGTask::new("t1", "d", "a", vec![], vec![], 3);
        t1.status = TaskStatus::Active;
        tasks.insert("t1".into(), t1);

        let result = serde_json::json!({"answer": 42});
        complete_task(&mut tasks, "t1", result.clone()).unwrap();
        assert_eq!(tasks["t1"].status, TaskStatus::Complete);
        assert_eq!(tasks["t1"].result, Some(result));
        assert!(tasks["t1"].completed_at.is_some());
    }

    #[test]
    fn test_complete_task_not_active() {
        let mut tasks = HashMap::new();
        tasks.insert("t1".into(), DAGTask::new("t1", "d", "a", vec![], vec![], 3));

        let result = complete_task(&mut tasks, "t1", serde_json::json!(null));
        assert!(matches!(result, Err(DAGError::NotActive(..))));
    }

    // -- fail_task tests --

    #[test]
    fn test_fail_task_retry() {
        let mut tasks = HashMap::new();
        let mut t1 = DAGTask::new("t1", "d", "a", vec![], vec![], 3);
        t1.status = TaskStatus::Active;
        tasks.insert("t1".into(), t1);

        let new_status = fail_task(&mut tasks, "t1", "something went wrong").unwrap();
        assert_eq!(new_status, TaskStatus::Retry);
        assert_eq!(tasks["t1"].retries, 1);
        assert!(tasks["t1"].worker_id.is_none());
    }

    #[test]
    fn test_fail_task_escalate() {
        let mut tasks = HashMap::new();
        let mut t1 = DAGTask::new("t1", "d", "a", vec![], vec![], 2);
        t1.status = TaskStatus::Active;
        t1.retries = 2; // already at max
        tasks.insert("t1".into(), t1);

        let new_status = fail_task(&mut tasks, "t1", "permanent failure").unwrap();
        assert_eq!(new_status, TaskStatus::Escalate);
    }

    #[test]
    fn test_fail_task_not_active() {
        let mut tasks = HashMap::new();
        tasks.insert("t1".into(), DAGTask::new("t1", "d", "a", vec![], vec![], 3));

        let result = fail_task(&mut tasks, "t1", "error");
        assert!(matches!(result, Err(DAGError::NotActive(..))));
    }

    #[test]
    fn test_fail_task_stores_error() {
        let mut tasks = HashMap::new();
        let mut t1 = DAGTask::new("t1", "d", "a", vec![], vec![], 3);
        t1.status = TaskStatus::Active;
        tasks.insert("t1".into(), t1);

        fail_task(&mut tasks, "t1", "timeout").unwrap();
        let result = &tasks["t1"].result;
        assert_eq!(result.as_ref().unwrap()["error"], "timeout");
    }

    // -- block/unblock tests --

    #[test]
    fn test_block_task_success() {
        let mut tasks = HashMap::new();
        let mut t1 = DAGTask::new("t1", "d", "a", vec![], vec![], 3);
        t1.status = TaskStatus::Active;
        tasks.insert("t1".into(), t1);

        block_task(&mut tasks, "t1", "dependency failed").unwrap();
        assert_eq!(tasks["t1"].status, TaskStatus::Blocked);
    }

    #[test]
    fn test_unblock_task_success() {
        let mut tasks = HashMap::new();
        let mut t1 = DAGTask::new("t1", "d", "a", vec![], vec![], 3);
        t1.status = TaskStatus::Blocked;
        tasks.insert("t1".into(), t1);

        unblock_task(&mut tasks, "t1").unwrap();
        assert_eq!(tasks["t1"].status, TaskStatus::Pending);
        assert!(tasks["t1"].worker_id.is_none());
    }

    // -- Integration tests --

    #[test]
    fn test_full_workflow_linear() {
        let specs = vec![
            TaskSpec::new("a", "first", "agent"),
            TaskSpec::new("b", "second", "agent").depends_on(vec!["a".into()]),
            TaskSpec::new("c", "third", "agent").depends_on(vec!["b".into()]),
        ];
        let mut graph = build_task_graph(specs).unwrap();

        // Activate ready tasks (a has no deps)
        let activated = activate_ready_tasks(&mut graph.tasks);
        assert_eq!(activated, vec!["a"]);

        // Claim a
        claim_task(&mut graph.tasks, "a", "w1").unwrap();

        // Complete a
        complete_task(&mut graph.tasks, "a", serde_json::json!("done")).unwrap();

        // Now b should be ready
        let activated = activate_ready_tasks(&mut graph.tasks);
        assert_eq!(activated, vec!["b"]);

        // Claim and complete b
        claim_task(&mut graph.tasks, "b", "w2").unwrap();
        complete_task(&mut graph.tasks, "b", serde_json::json!("done")).unwrap();

        // Now c should be ready
        let activated = activate_ready_tasks(&mut graph.tasks);
        assert_eq!(activated, vec!["c"]);

        // Complete the pipeline
        claim_task(&mut graph.tasks, "c", "w1").unwrap();
        complete_task(&mut graph.tasks, "c", serde_json::json!("done")).unwrap();

        assert!(graph.is_finished());
    }

    #[test]
    fn test_workflow_with_retry_and_recovery() {
        let mut tasks = HashMap::new();
        tasks.insert("t1".into(), DAGTask::new("t1", "d", "a", vec![], vec![], 3));
        let mut t1 = tasks.get_mut("t1").unwrap();
        t1.status = TaskStatus::Active;
        t1.worker_id = Some("w1".into());
        t1.started_at = Some(chrono::Utc::now());

        // Fail the task
        let status = fail_task(&mut tasks, "t1", "transient error").unwrap();
        assert_eq!(status, TaskStatus::Retry);
        assert_eq!(tasks["t1"].retries, 1);

        // Activate retry → Ready
        let activated = activate_ready_tasks(&mut tasks);
        assert_eq!(activated, vec!["t1"]);

        // Re-claim and complete
        claim_task(&mut tasks, "t1", "w2").unwrap();
        complete_task(&mut tasks, "t1", serde_json::json!("success")).unwrap();

        assert_eq!(tasks["t1"].status, TaskStatus::Complete);
    }

    #[test]
    fn test_parallel_branches() {
        let specs = vec![
            TaskSpec::new("root", "root", "a"),
            TaskSpec::new("left", "left", "a").depends_on(vec!["root".into()]),
            TaskSpec::new("right", "right", "a").depends_on(vec!["root".into()]),
        ];
        let mut graph = build_task_graph(specs).unwrap();

        // Only root is ready initially
        let activated = activate_ready_tasks(&mut graph.tasks);
        assert_eq!(activated, vec!["root"]);

        // Complete root
        claim_task(&mut graph.tasks, "root", "w1").unwrap();
        complete_task(&mut graph.tasks, "root", serde_json::json!("done")).unwrap();

        // Both left and right should be ready
        let activated = activate_ready_tasks(&mut graph.tasks);
        assert_eq!(activated.len(), 2);
        assert!(activated.contains(&"left".to_string()));
        assert!(activated.contains(&"right".to_string()));
    }

    #[test]
    fn test_task_duration() {
        let mut task = DAGTask::new("t1", "d", "a", vec![], vec![], 3);
        task.started_at = Some(chrono::Utc::now());
        task.completed_at = Some(chrono::Utc::now() + chrono::Duration::seconds(10));

        let duration = task.duration().unwrap();
        assert_eq!(duration.num_seconds(), 10);
    }

    #[test]
    fn test_task_duration_incomplete() {
        let task = DAGTask::new("t1", "d", "a", vec![], vec![], 3);
        assert!(task.duration().is_none());
    }

    #[test]
    fn test_task_spec_builder() {
        let spec = TaskSpec::new("t1", "desc", "analyst")
            .with_tags(vec!["finance".into()])
            .depends_on(vec!["t0".into()])
            .with_max_retries(5);

        assert_eq!(spec.task_id, "t1");
        assert_eq!(spec.tags, vec!["finance"]);
        assert_eq!(spec.dependencies, vec!["t0"]);
        assert_eq!(spec.max_retries, 5);
    }

    #[test]
    fn test_status_counts() {
        let specs = vec![
            TaskSpec::new("a", "a", "a"),
            TaskSpec::new("b", "b", "a"),
            TaskSpec::new("c", "c", "a"),
        ];
        let mut graph = build_task_graph(specs).unwrap();

        graph.tasks.get_mut("a").unwrap().status = TaskStatus::Complete;
        graph.tasks.get_mut("b").unwrap().status = TaskStatus::Active;

        let counts = graph.status_counts();
        assert_eq!(*counts.get(&TaskStatus::Complete).unwrap_or(&0), 1);
        assert_eq!(*counts.get(&TaskStatus::Active).unwrap_or(&0), 1);
        assert_eq!(*counts.get(&TaskStatus::Pending).unwrap_or(&0), 1);
        assert_eq!(graph.completed_count(), 1);
    }
}
