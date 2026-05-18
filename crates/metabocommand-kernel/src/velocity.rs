use crate::types::*;

/// Calculate a composite velocity score from action metrics.
///
/// Weighted: 0.40 × actions_per_hour_norm + 0.30 × success_rate + 0.30 × speed_norm
///
/// - `actions_per_hour` is normalized by dividing by 100 (so 100 actions/hr = max contribution)
/// - `success_rate` should be in [0.0, 1.0]
/// - `speed_norm` = 1 / (1 + avg_response_time_secs / 60)
pub fn calculate_velocity_score(
    actions_per_hour: f64,
    success_rate: f64,
    avg_response_time_secs: f64,
) -> f64 {
    let actions_norm = (actions_per_hour / 100.0).min(1.0).max(0.0);
    let success = success_rate.clamp(0.0, 1.0);
    let speed_norm = 1.0 / (1.0 + avg_response_time_secs / 60.0);

    actions_norm * 0.40 + success * 0.30 + speed_norm * 0.30
}

/// Classify a velocity score into a discrete tier.
///
/// - ≥ 80 → Critical (extremely high throughput)
/// - ≥ 60 → Hot
/// - ≥ 40 → Warm
/// - < 40 → Cold
pub fn classify_velocity(score: f64) -> VelocityTier {
    if score >= 80.0 {
        VelocityTier::Critical
    } else if score >= 60.0 {
        VelocityTier::Hot
    } else if score >= 40.0 {
        VelocityTier::Warm
    } else {
        VelocityTier::Cold
    }
}

/// Determine if an action should be auto-executed.
///
/// Auto-execute when velocity score >= 70 AND action risk < 0.3.
pub fn should_auto_execute(score: f64, action_risk: f64) -> bool {
    score >= 70.0 && action_risk < 0.3
}

/// Check if a threshold breach is occurring.
///
/// Returns true if the current value exceeds the threshold AND
/// the majority (more than half) of the last `window_size` history values also exceed it.
pub fn compute_threshold_breach(
    current: f64,
    threshold: f64,
    window_size: usize,
    history: &[f64],
) -> bool {
    if current <= threshold {
        return false;
    }

    if window_size == 0 || history.is_empty() {
        return current > threshold;
    }

    let window: &[f64] = if history.len() >= window_size {
        &history[history.len() - window_size..]
    } else {
        history
    };

    let exceeding = window.iter().filter(|&&v| v > threshold).count();
    exceeding > window.len() / 2
}

/// Calculate the deceleration rate: negative velocity change per unit time.
pub fn deceleration_rate(recent_scores: &[f64]) -> f64 {
    if recent_scores.len() < 2 {
        return 0.0;
    }
    let last = recent_scores[recent_scores.len() - 1];
    let prev = recent_scores[recent_scores.len() - 2];
    prev - last // positive means slowing down
}

/// Compute the moving average of velocity scores.
pub fn velocity_moving_average(scores: &[f64], window: usize) -> Vec<f64> {
    if window == 0 || scores.len() < window {
        return vec![];
    }
    let mut result = Vec::with_capacity(scores.len() - window + 1);
    let mut sum: f64 = scores[..window].iter().sum();
    result.push(sum / window as f64);
    for i in window..scores.len() {
        sum += scores[i] - scores[i - window];
        result.push(sum / window as f64);
    }
    result
}

/// Generate a human-readable velocity report.
pub fn velocity_report(score: f64, tier: VelocityTier) -> String {
    format!(
        "Velocity Score: {:.1} | Tier: {} | Auto-execute (risk=0.2): {}",
        score,
        tier,
        should_auto_execute(score, 0.2)
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_velocity_score_high() {
        // 100 actions/hr (norm=1.0), 100% success, instant response (speed=1.0)
        let score = calculate_velocity_score(100.0, 1.0, 0.0);
        assert!((score - 1.0).abs() < 0.001);
    }

    #[test]
    fn test_velocity_score_low() {
        let score = calculate_velocity_score(0.0, 0.0, 300.0);
        assert!(score < 0.1);
    }

    #[test]
    fn test_velocity_score_moderate() {
        let score = calculate_velocity_score(50.0, 0.8, 30.0);
        assert!(score > 0.3 && score < 0.8);
    }

    #[test]
    fn test_classify_critical() {
        assert_eq!(classify_velocity(85.0), VelocityTier::Critical);
        assert_eq!(classify_velocity(80.0), VelocityTier::Critical);
    }

    #[test]
    fn test_classify_hot() {
        assert_eq!(classify_velocity(70.0), VelocityTier::Hot);
        assert_eq!(classify_velocity(60.0), VelocityTier::Hot);
    }

    #[test]
    fn test_classify_warm() {
        assert_eq!(classify_velocity(50.0), VelocityTier::Warm);
        assert_eq!(classify_velocity(40.0), VelocityTier::Warm);
    }

    #[test]
    fn test_classify_cold() {
        assert_eq!(classify_velocity(30.0), VelocityTier::Cold);
        assert_eq!(classify_velocity(0.0), VelocityTier::Cold);
    }

    #[test]
    fn test_should_auto_execute_true() {
        assert!(should_auto_execute(80.0, 0.1));
        assert!(should_auto_execute(70.0, 0.29));
    }

    #[test]
    fn test_should_auto_execute_false_low_score() {
        assert!(!should_auto_execute(60.0, 0.1));
    }

    #[test]
    fn test_should_auto_execute_false_high_risk() {
        assert!(!should_auto_execute(90.0, 0.5));
    }

    #[test]
    fn test_should_auto_execute_edge() {
        assert!(should_auto_execute(70.0, 0.299)); // just under risk threshold
        assert!(!should_auto_execute(70.0, 0.3)); // exactly at risk threshold
    }

    #[test]
    fn test_threshold_breach_simple() {
        assert!(compute_threshold_breach(100.0, 50.0, 0, &[]));
        assert!(!compute_threshold_breach(40.0, 50.0, 0, &[]));
    }

    #[test]
    fn test_threshold_breach_with_history_majority() {
        // Current exceeds, majority of window exceeds
        let history = vec![60.0, 55.0, 52.0, 51.0, 50.0]; // 4/5 above 50
        assert!(compute_threshold_breach(60.0, 50.0, 5, &history));
    }

    #[test]
    fn test_threshold_breach_with_history_minority() {
        // Current exceeds, but minority of window exceeds
        let history = vec![40.0, 45.0, 30.0, 20.0, 10.0]; // 0/5 above 50
        assert!(!compute_threshold_breach(60.0, 50.0, 5, &history));
    }

    #[test]
    fn test_deceleration_rate_positive() {
        let scores = vec![80.0, 60.0, 40.0];
        let rate = deceleration_rate(&scores);
        assert!((rate - 20.0).abs() < 0.001); // slowing down
    }

    #[test]
    fn test_deceleration_rate_negative() {
        let scores = vec![40.0, 60.0, 80.0];
        let rate = deceleration_rate(&scores);
        assert!((rate - (-20.0)).abs() < 0.001); // speeding up
    }

    #[test]
    fn test_deceleration_rate_insufficient() {
        assert!((deceleration_rate(&[50.0]) - 0.0).abs() < 0.001);
    }

    #[test]
    fn test_velocity_moving_average() {
        let scores = vec![10.0, 20.0, 30.0, 40.0, 50.0];
        let ma = velocity_moving_average(&scores, 3);
        assert_eq!(ma.len(), 3);
        assert!((ma[0] - 20.0).abs() < 0.001);
        assert!((ma[1] - 30.0).abs() < 0.001);
        assert!((ma[2] - 40.0).abs() < 0.001);
    }

    #[test]
    fn test_velocity_report() {
        let report = velocity_report(75.0, VelocityTier::Hot);
        assert!(report.contains("75.0"));
        assert!(report.contains("Hot"));
    }
}
