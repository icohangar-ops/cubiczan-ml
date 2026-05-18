//! Alert dispatcher – threshold-based alerts, multi-channel routing,
//! deduplication, escalation policies, and alert history.

use crate::detect::DetectionResult;
use crate::types::{Alert, Chain};
pub use crate::types::AlertLevel;
use chrono::{DateTime, Duration, Utc};
use sha3::{Digest, Keccak256};
use std::collections::HashMap;

// ---------------------------------------------------------------------------
// AlertChannel
// ---------------------------------------------------------------------------

/// Supported alert routing channels.
#[derive(Debug, Clone, PartialEq, Eq, Hash, serde::Serialize, serde::Deserialize)]
pub enum AlertChannel {
    Webhook,
    Email,
    Slack,
    PagerDuty,
    Log,
}

// ---------------------------------------------------------------------------
// EscalationPolicy
// ---------------------------------------------------------------------------

/// Defines when an alert should be escalated to a higher-priority channel.
#[derive(Debug, Clone)]
pub struct EscalationPolicy {
    /// How long to wait before escalating a recurring alert.
    pub escalation_window: Duration,
    /// How many occurrences of the same fingerprint within the window trigger escalation.
    pub threshold: u32,
    /// The channel to escalate to.
    pub escalate_to: AlertChannel,
}

impl Default for EscalationPolicy {
    fn default() -> Self {
        Self {
            escalation_window: Duration::minutes(15),
            threshold: 3,
            escalate_to: AlertChannel::PagerDuty,
        }
    }
}

// ---------------------------------------------------------------------------
// AlertConfig
// ---------------------------------------------------------------------------

/// Configuration for the alert dispatcher.
#[derive(Debug, Clone)]
pub struct AlertConfig {
    /// Minimum alert level to dispatch.
    pub min_level: AlertLevel,
    /// Channels for each alert level.
    pub channels: HashMap<AlertLevel, Vec<AlertChannel>>,
    /// Escalation policies (keyed by channel).
    pub escalation_policies: Vec<EscalationPolicy>,
    /// Deduplication window – alerts with the same fingerprint within this
    /// window are suppressed.
    pub dedup_window: Duration,
}

impl Default for AlertConfig {
    fn default() -> Self {
        let mut channels = HashMap::new();
        channels.insert(AlertLevel::Low, vec![AlertChannel::Log]);
        channels.insert(AlertLevel::Medium, vec![AlertChannel::Slack, AlertChannel::Log]);
        channels.insert(
            AlertLevel::High,
            vec![AlertChannel::Slack, AlertChannel::Email, AlertChannel::Log],
        );
        channels.insert(
            AlertLevel::Critical,
            vec![
                AlertChannel::PagerDuty,
                AlertChannel::Slack,
                AlertChannel::Email,
                AlertChannel::Log,
            ],
        );
        Self {
            min_level: AlertLevel::Low,
            channels,
            escalation_policies: vec![EscalationPolicy::default()],
            dedup_window: Duration::minutes(5),
        }
    }
}

// ---------------------------------------------------------------------------
// AlertRouter
// ---------------------------------------------------------------------------

/// Dispatches alerts, handles deduplication, and manages escalation.
pub struct AlertRouter {
    config: AlertConfig,
    /// Dispatched alerts (acts as history buffer).
    history: Vec<Alert>,
    /// Track when each fingerprint was last dispatched.
    last_dispatched: HashMap<String, DateTime<Utc>>,
    /// Track occurrence counts for escalation.
    occurrence_counts: HashMap<String, (u32, DateTime<Utc>)>,
    /// Collected dispatch actions (for inspection / testing).
    dispatched: Vec<(AlertChannel, Alert)>,
}

impl AlertRouter {
    pub fn new(config: AlertConfig) -> Self {
        Self {
            config,
            history: Vec::new(),
            last_dispatched: HashMap::new(),
            occurrence_counts: HashMap::new(),
            dispatched: Vec::new(),
        }
    }

    /// Create with default configuration.
    pub fn default_router() -> Self {
        Self::new(AlertConfig::default())
    }

    /// Convert a `DetectionResult` into an `Alert` and dispatch it.
    pub fn dispatch(
        &mut self,
        chain: Chain,
        result: DetectionResult,
        now: DateTime<Utc>,
    ) -> Option<Alert> {
        let level = AlertLevel::from_score(result.score);
        if level < self.config.min_level {
            return None;
        }

        let fingerprint = compute_fingerprint(&chain, &result);

        // Deduplication
        if let Some(&last) = self.last_dispatched.get(&fingerprint) {
            if now - last < self.config.dedup_window {
                // Still within dedup window – suppress but count occurrence
                self.increment_occurrence(&fingerprint, now);
                return None;
            }
        }

        let alert = Alert {
            id: uuid::Uuid::new_v4().to_string(),
            chain,
            anomaly_type: result.anomaly_type.clone(),
            score: result.score,
            level,
            message: result.description.clone(),
            tx_hash: result.tx_hash.clone(),
            timestamp: now,
            fingerprint: fingerprint.clone(),
        };

        self.last_dispatched.insert(fingerprint.clone(), now);
        self.increment_occurrence(&fingerprint, now);

        // Route to channels
        if let Some(chs) = self.config.channels.get(&level) {
            for ch in chs {
                self.dispatched.push((ch.clone(), alert.clone()));
            }
        }

        // Check escalation
        for policy in &self.config.escalation_policies {
            if self.should_escalate(&fingerprint, policy, now) {
                self.dispatched
                    .push((policy.escalate_to.clone(), alert.clone()));
            }
        }

        self.history.push(alert.clone());
        Some(alert)
    }

    /// Manually dispatch an alert (bypasses detection, but applies dedup + routing).
    pub fn dispatch_alert(&mut self, alert: Alert, now: DateTime<Utc>) -> Option<Alert> {
        if alert.level < self.config.min_level {
            return None;
        }
        if let Some(&last) = self.last_dispatched.get(&alert.fingerprint) {
            if now - last < self.config.dedup_window {
                return None;
            }
        }
        self.last_dispatched.insert(alert.fingerprint.clone(), now);
        self.increment_occurrence(&alert.fingerprint, now);

        if let Some(chs) = self.config.channels.get(&alert.level) {
            for ch in chs {
                self.dispatched.push((ch.clone(), alert.clone()));
            }
        }
        self.history.push(alert.clone());
        Some(alert)
    }

    // -- Accessors ----------------------------------------------------------

    /// Return the dispatched (channel, alert) pairs.
    pub fn dispatched(&self) -> &[(AlertChannel, Alert)] {
        &self.dispatched
    }

    /// Return the alert history.
    pub fn history(&self) -> &[Alert] {
        &self.history
    }

    /// Return total number of dispatched alerts (including channel copies).
    pub fn dispatch_count(&self) -> usize {
        self.dispatched.len()
    }

    /// Return number of unique alerts (history length).
    pub fn unique_alert_count(&self) -> usize {
        self.history.len()
    }

    /// Clear history and dispatched buffers.
    pub fn clear(&mut self) {
        self.history.clear();
        self.dispatched.clear();
        self.last_dispatched.clear();
        self.occurrence_counts.clear();
    }

    // -- private ------------------------------------------------------------

    fn increment_occurrence(&mut self, fp: &str, now: DateTime<Utc>) {
        let entry = self.occurrence_counts.entry(fp.to_string()).or_insert((0, now));
        entry.0 += 1;
        entry.1 = now;
    }

    fn should_escalate(
        &self,
        fp: &str,
        policy: &EscalationPolicy,
        now: DateTime<Utc>,
    ) -> bool {
        if let Some(&(count, first)) = self.occurrence_counts.get(fp) {
            if count >= policy.threshold && (now - first) < policy.escalation_window {
                return true;
            }
        }
        false
    }
}

// ---------------------------------------------------------------------------
// Fingerprinting
// ---------------------------------------------------------------------------

fn compute_fingerprint(chain: &Chain, result: &DetectionResult) -> String {
    let mut hasher = Keccak256::new();
    hasher.update(chain.chain_id().as_bytes());
    hasher.update(format!("{:?}", result.anomaly_type).as_bytes());
    hasher.update(
        result
            .tx_hash
            .as_deref()
            .unwrap_or("none")
            .as_bytes(),
    );
    let hash = hasher.finalize();
    hex::encode(hash)[..16].to_string()
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use crate::types::{AnomalyScore, AnomalyType};
    use super::*;

    fn make_result(anomaly_type: AnomalyType, score: f64) -> DetectionResult {
        DetectionResult {
            anomaly_type,
            score: AnomalyScore::new(score),
            description: "test alert".into(),
            tx_hash: Some("0xABC".into()),
        }
    }

    #[test]
    fn test_dispatch_basic() {
        let mut router = AlertRouter::default_router();
        let result = make_result(AnomalyType::ValueOutlier, 0.95);
        let now = Utc::now();
        let alert = router.dispatch(Chain::Ethereum, result, now);
        assert!(alert.is_some());
        assert_eq!(alert.unwrap().level, AlertLevel::Critical);
        assert_eq!(router.unique_alert_count(), 1);
        assert!(router.dispatch_count() >= 1);
    }

    #[test]
    fn test_dispatch_below_min_level() {
        let mut cfg = AlertConfig::default();
        cfg.min_level = AlertLevel::Critical;
        let mut router = AlertRouter::new(cfg);
        let result = make_result(AnomalyType::ValueOutlier, 0.5); // Medium
        let alert = router.dispatch(Chain::Ethereum, result, Utc::now());
        assert!(alert.is_none());
    }

    #[test]
    fn test_deduplication() {
        let mut router = AlertRouter::default_router();
        let now = Utc::now();
        let result = make_result(AnomalyType::ValueOutlier, 0.95);
        let a1 = router.dispatch(Chain::Ethereum, result.clone(), now);
        assert!(a1.is_some());
        let a2 = router.dispatch(Chain::Ethereum, result, now + Duration::seconds(10));
        assert!(a2.is_none()); // suppressed within dedup window
        assert_eq!(router.unique_alert_count(), 1);
    }

    #[test]
    fn test_dedup_expiry() {
        let mut router = AlertRouter::default_router();
        let now = Utc::now();
        let result = make_result(AnomalyType::VolumeSpike, 0.8);
        let _ = router.dispatch(Chain::Solana, result.clone(), now);
        // Wait past dedup window (5 min default)
        let later = now + Duration::minutes(6);
        let a2 = router.dispatch(Chain::Solana, result, later);
        assert!(a2.is_some());
        assert_eq!(router.unique_alert_count(), 2);
    }

    #[test]
    fn test_channel_routing() {
        let mut router = AlertRouter::default_router();
        let now = Utc::now();
        let result = make_result(AnomalyType::ValueOutlier, 0.95);
        router.dispatch(Chain::Ethereum, result, now);
        let dispatched = router.dispatched();
        // Critical should go to PagerDuty, Slack, Email, Log
        let channels: Vec<&AlertChannel> = dispatched.iter().map(|(c, _)| c).collect();
        assert!(channels.contains(&&AlertChannel::PagerDuty));
        assert!(channels.contains(&&AlertChannel::Slack));
        assert!(channels.contains(&&AlertChannel::Email));
        assert!(channels.contains(&&AlertChannel::Log));
    }

    #[test]
    fn test_escalation() {
        let mut cfg = AlertConfig::default();
        cfg.escalation_policies = vec![EscalationPolicy {
            escalation_window: Duration::minutes(15),
            threshold: 2,
            escalate_to: AlertChannel::PagerDuty,
        }];
        cfg.dedup_window = Duration::seconds(0); // disable dedup for this test
        let mut router = AlertRouter::new(cfg);

        let result = make_result(AnomalyType::PatternMatch, 0.75);
        let now = Utc::now();

        // First dispatch – no escalation
        router.dispatch(Chain::Ethereum, result.clone(), now);
        // Second dispatch – should trigger escalation
        router.dispatch(Chain::Ethereum, result.clone(), now + Duration::seconds(1));

        let dispatched = router.dispatched();
        // Should have at least one PagerDuty from escalation
        let pager_duty_count = dispatched
            .iter()
            .filter(|(c, _)| *c == AlertChannel::PagerDuty)
            .count();
        assert!(
            pager_duty_count >= 1,
            "Expected escalation to PagerDuty, got {}",
            pager_duty_count
        );
    }

    #[test]
    fn test_history() {
        let mut router = AlertRouter::default_router();
        let now = Utc::now();
        router.dispatch(
            Chain::Ethereum,
            make_result(AnomalyType::ValueOutlier, 0.5),
            now,
        );
        router.dispatch(
            Chain::Solana,
            make_result(AnomalyType::VolumeSpike, 0.8),
            now + Duration::seconds(60),
        );
        assert_eq!(router.history().len(), 2);
    }

    #[test]
    fn test_clear() {
        let mut router = AlertRouter::default_router();
        router.dispatch(
            Chain::Ethereum,
            make_result(AnomalyType::ValueOutlier, 0.9),
            Utc::now(),
        );
        router.clear();
        assert!(router.history().is_empty());
        assert_eq!(router.dispatch_count(), 0);
    }

    #[test]
    fn test_compute_fingerprint_deterministic() {
        let r1 = make_result(AnomalyType::ValueOutlier, 0.9);
        let r2 = make_result(AnomalyType::ValueOutlier, 0.9);
        let fp1 = compute_fingerprint(&Chain::Ethereum, &r1);
        let fp2 = compute_fingerprint(&Chain::Ethereum, &r2);
        assert_eq!(fp1, fp2);
    }

    #[test]
    fn test_compute_fingerprint_different() {
        let r1 = make_result(AnomalyType::ValueOutlier, 0.9);
        let r2 = make_result(AnomalyType::VolumeSpike, 0.9);
        let fp1 = compute_fingerprint(&Chain::Ethereum, &r1);
        let fp2 = compute_fingerprint(&Chain::Ethereum, &r2);
        assert_ne!(fp1, fp2);
    }
}
