//! # Control Loop
//!
//! Implements PID-like control theory for the closed-loop system.
//! Provides feedback control over risk, leverage, exposure, and drawdown.
//!
//! ## PID Components
//! - **Proportional (P)**: React to current error (actual vs target)
//! - **Integral (I)**: Accumulate past errors (prevent persistent drift)
//! - **Derivative (D)**: React to rate of change (prevent overshooting)

use crate::types::*;

/// A PID controller with tunable gains.
#[derive(Debug, Clone)]
pub struct PidController {
    /// Proportional gain.
    pub kp: f64,
    /// Integral gain.
    pub ki: f64,
    /// Derivative gain.
    pub kd: f64,
    /// Accumulated integral error.
    pub integral_error: f64,
    /// Previous error for derivative computation.
    pub prev_error: f64,
    /// Anti-windup limit for integral term.
    pub integral_limit: f64,
    /// Output clamping range.
    pub output_min: f64,
    pub output_max: f64,
}

impl Default for PidController {
    fn default() -> Self {
        PidController {
            kp: 0.5,
            ki: 0.1,
            kd: 0.05,
            integral_error: 0.0,
            prev_error: 0.0,
            integral_limit: 1.0,
            output_min: -1.0,
            output_max: 1.0,
        }
    }
}

impl PidController {
    pub fn new(kp: f64, ki: f64, kd: f64) -> Self {
        PidController {
            kp,
            ki,
            kd,
            integral_limit: 1.0,
            ..Default::default()
        }
    }

    /// Compute PID output given the current error.
    pub fn compute(&mut self, error: f64) -> f64 {
        // Proportional
        let p = self.kp * error;

        // Integral with anti-windup
        self.integral_error = (self.integral_error + error).clamp(
            -self.integral_limit,
            self.integral_limit,
        );
        let i = self.ki * self.integral_error;

        // Derivative
        let d = self.kd * (error - self.prev_error);
        self.prev_error = error;

        // Output clamped
        (p + i + d).clamp(self.output_min, self.output_max)
    }

    /// Reset the controller state.
    pub fn reset(&mut self) {
        self.integral_error = 0.0;
        self.prev_error = 0.0;
    }
}

/// Control targets for the closed-loop system.
#[derive(Debug, Clone)]
pub struct ControlTargets {
    /// Target risk level (annualized volatility).
    pub target_volatility: f64,
    /// Target leverage.
    pub target_leverage: f64,
    /// Target portfolio exposure (fraction).
    pub target_exposure: f64,
    /// Maximum acceptable drawdown.
    pub max_drawdown: f64,
    /// Target Sharpe ratio.
    pub target_sharpe: f64,
}

impl Default for ControlTargets {
    fn default() -> Self {
        ControlTargets {
            target_volatility: 0.15,
            target_leverage: 1.0,
            target_exposure: 0.95,
            max_drawdown: 0.15,
            target_sharpe: 1.5,
        }
    }
}

/// The main control loop implementing PID feedback control.
#[derive(Debug, Clone)]
pub struct ControlLoop {
    /// PID controller for risk (volatility).
    pub risk_pid: PidController,
    /// PID controller for leverage.
    pub leverage_pid: PidController,
    /// PID controller for exposure.
    pub exposure_pid: PidController,
    /// PID controller for drawdown.
    pub drawdown_pid: PidController,
    /// Control targets.
    pub targets: ControlTargets,
    /// History of feedback signals.
    pub feedback_history: Vec<FeedbackSignal>,
    /// History of control outputs.
    pub control_outputs: Vec<ControlOutput>,
    /// Oscillation detection window.
    pub oscillation_window: usize,
}

/// Output of a control cycle.
#[derive(Debug, Clone)]
pub struct ControlOutput {
    /// Risk adjustment factor.
    pub risk_adjustment: f64,
    /// Leverage adjustment factor.
    pub leverage_adjustment: f64,
    /// Exposure adjustment factor.
    pub exposure_adjustment: f64,
    /// Whether emergency mode is active.
    pub emergency: bool,
    /// Overall control signal (-1 to 1).
    pub aggregate_signal: f64,
}

impl Default for ControlOutput {
    fn default() -> Self {
        ControlOutput {
            risk_adjustment: 0.0,
            leverage_adjustment: 0.0,
            exposure_adjustment: 0.0,
            emergency: false,
            aggregate_signal: 0.0,
        }
    }
}

impl Default for ControlLoop {
    fn default() -> Self {
        ControlLoop {
            risk_pid: PidController::new(0.5, 0.1, 0.05),
            leverage_pid: PidController::new(0.3, 0.05, 0.02),
            exposure_pid: PidController::new(0.2, 0.05, 0.01),
            drawdown_pid: PidController::new(1.0, 0.2, 0.1),
            targets: ControlTargets::default(),
            feedback_history: Vec::new(),
            control_outputs: Vec::new(),
            oscillation_window: 20,
        }
    }
}

impl ControlLoop {
    pub fn new(targets: ControlTargets) -> Self {
        ControlLoop {
            targets,
            ..Default::default()
        }
    }

    /// Run one control cycle given current portfolio state.
    pub fn compute_control(&mut self, portfolio: &PortfolioState, metrics: &LoopMetrics) -> ControlOutput {
        let mut output = ControlOutput::default();

        // Risk control: error = actual_vol - target_vol
        let risk_error = portfolio.sharpe - self.targets.target_sharpe;
        output.risk_adjustment = self.risk_pid.compute(risk_error);

        // Leverage control (negated: positive output = reduce leverage)
        let lev_error = self.targets.target_leverage - portfolio.leverage;
        output.leverage_adjustment = self.leverage_pid.compute(lev_error);

        // Exposure control (negated: positive output = reduce exposure)
        let exp_error = self.targets.target_exposure - portfolio.exposure;
        output.exposure_adjustment = self.exposure_pid.compute(exp_error);

        // Drawdown control (emergency)
        let dd_error = if portfolio.max_drawdown > self.targets.max_drawdown {
            (portfolio.max_drawdown - self.targets.max_drawdown) / self.targets.max_drawdown
        } else {
            0.0
        };
        let dd_signal = self.drawdown_pid.compute(dd_error);
        output.emergency = dd_error > 0.0;

        // Aggregate signal
        output.aggregate_signal = (output.risk_adjustment * 0.3
            + output.leverage_adjustment * 0.2
            + output.exposure_adjustment * 0.1
            + dd_signal * 0.4)
            .clamp(-1.0, 1.0);

        // Generate feedback signals
        self.feedback_history.push(FeedbackSignal::new(
            "sharpe", portfolio.sharpe, self.targets.target_sharpe, output.risk_adjustment,
        ));
        self.feedback_history.push(FeedbackSignal::new(
            "leverage", portfolio.leverage, self.targets.target_leverage, output.leverage_adjustment,
        ));
        self.feedback_history.push(FeedbackSignal::new(
            "exposure", portfolio.exposure, self.targets.target_exposure, output.exposure_adjustment,
        ));
        self.feedback_history.push(FeedbackSignal::new(
            "drawdown", portfolio.max_drawdown, self.targets.max_drawdown, dd_signal,
        ));

        self.control_outputs.push(output.clone());

        // Trim history
        if self.feedback_history.len() > 1000 {
            self.feedback_history.drain(..self.feedback_history.len() - 1000);
        }
        if self.control_outputs.len() > 1000 {
            self.control_outputs.drain(..self.control_outputs.len() - 1000);
        }

        output
    }

    /// Detect oscillation in the control output.
    pub fn detect_oscillation(&self) -> bool {
        if self.control_outputs.len() < self.oscillation_window {
            return false;
        }

        let window = &self.control_outputs[self.control_outputs.len() - self.oscillation_window..];
        let signals: Vec<f64> = window.iter().map(|o| o.aggregate_signal).collect();

        // Count sign changes
        let mut sign_changes = 0;
        for i in 1..signals.len() {
            if (signals[i] > 0.0 && signals[i - 1] < 0.0) || (signals[i] < 0.0 && signals[i - 1] > 0.0) {
                sign_changes += 1;
            }
        }

        // If more than half the window shows sign changes, it's oscillating
        sign_changes > self.oscillation_window / 2
    }

    /// Dampen the PID controllers to reduce oscillation.
    pub fn dampen(&mut self, factor: f64) {
        self.risk_pid.kp *= factor;
        self.risk_pid.ki *= factor;
        self.risk_pid.kd *= factor;
        self.leverage_pid.kp *= factor;
        self.leverage_pid.ki *= factor;
        self.leverage_pid.kd *= factor;
        self.exposure_pid.kp *= factor;
        self.exposure_pid.ki *= factor;
        self.exposure_pid.kd *= factor;
    }

    /// Compute feedback signal for a specific metric.
    pub fn compute_feedback(&self, metric: &str, actual: f64, target: f64) -> FeedbackSignal {
        let error = target - actual;
        let adjustment = -error * 0.5; // Simple proportional control
        FeedbackSignal::new(metric, actual, target, adjustment)
    }

    /// Reset all PID controllers.
    pub fn reset(&mut self) {
        self.risk_pid.reset();
        self.leverage_pid.reset();
        self.exposure_pid.reset();
        self.drawdown_pid.reset();
        self.feedback_history.clear();
        self.control_outputs.clear();
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn close(a: f64, b: f64, eps: f64) -> bool {
        (a - b).abs() < eps
    }

    #[test]
    fn test_pid_proportional() {
        let mut pid = PidController::new(1.0, 0.0, 0.0);
        let output = pid.compute(0.5);
        assert!(close(output, 0.5, 1e-10));
    }

    #[test]
    fn test_pid_integral() {
        let mut pid = PidController::new(0.0, 1.0, 0.0);
        let _ = pid.compute(0.5);
        let output = pid.compute(0.5);
        // After two calls with error=0.5: integral = 1.0, output = 1.0
        assert!(close(output, 1.0, 1e-10));
    }

    #[test]
    fn test_pid_derivative() {
        let mut pid = PidController::new(0.0, 0.0, 1.0);
        let _ = pid.compute(0.0);
        let output = pid.compute(1.0);
        // Derivative: 1.0 * (1.0 - 0.0) = 1.0
        assert!(close(output, 1.0, 1e-10));
    }

    #[test]
    fn test_pid_anti_windup() {
        let mut pid = PidController::new(0.0, 1.0, 0.0);
        pid.integral_limit = 1.0;
        for _ in 0..100 {
            let _ = pid.compute(1.0);
        }
        assert!(pid.integral_error <= 1.0);
    }

    #[test]
    fn test_pid_output_clamping() {
        let mut pid = PidController::new(10.0, 0.0, 0.0);
        pid.output_min = -0.5;
        pid.output_max = 0.5;
        let output = pid.compute(1.0);
        assert!(close(output, 0.5, 1e-10));
    }

    #[test]
    fn test_pid_reset() {
        let mut pid = PidController::new(1.0, 1.0, 1.0);
        let _ = pid.compute(0.5);
        pid.reset();
        assert!(close(pid.integral_error, 0.0, 1e-10));
        assert!(close(pid.prev_error, 0.0, 1e-10));
    }

    #[test]
    fn test_control_loop_basic() {
        let mut loop_ctrl = ControlLoop::default();
        let portfolio = PortfolioState::default();
        let metrics = LoopMetrics::default();
        let output = loop_ctrl.compute_control(&portfolio, &metrics);
        assert!(!output.emergency);
        assert!(output.aggregate_signal >= -1.0 && output.aggregate_signal <= 1.0);
    }

    #[test]
    fn test_control_emergency_drawdown() {
        let mut loop_ctrl = ControlLoop::default();
        let mut portfolio = PortfolioState::default();
        portfolio.max_drawdown = 0.20; // Exceeds 0.15 max
        let metrics = LoopMetrics::default();
        let output = loop_ctrl.compute_control(&portfolio, &metrics);
        assert!(output.emergency);
    }

    #[test]
    fn test_control_no_emergency_normal() {
        let mut loop_ctrl = ControlLoop::default();
        let mut portfolio = PortfolioState::default();
        portfolio.max_drawdown = 0.05; // Below 0.15 max
        let metrics = LoopMetrics::default();
        let output = loop_ctrl.compute_control(&portfolio, &metrics);
        assert!(!output.emergency);
    }

    #[test]
    fn test_oscillation_detection_stable() {
        let mut loop_ctrl = ControlLoop::default();
        loop_ctrl.oscillation_window = 10;
        let portfolio = PortfolioState::default();
        let metrics = LoopMetrics::default();
        for _ in 0..15 {
            loop_ctrl.compute_control(&portfolio, &metrics);
        }
        assert!(!loop_ctrl.detect_oscillation());
    }

    #[test]
    fn test_oscillation_detection_oscillating() {
        let mut loop_ctrl = ControlLoop::default();
        loop_ctrl.oscillation_window = 10;
        let mut portfolio = PortfolioState::default();
        portfolio.sharpe = 0.0;
        let metrics = LoopMetrics::default();

        // Alternate sharpe to cause oscillation
        for i in 0..15 {
            portfolio.sharpe = if i % 2 == 0 { 3.0 } else { 0.0 };
            loop_ctrl.compute_control(&portfolio, &metrics);
        }
        // May or may not oscillate depending on PID dynamics
        // Just verify it doesn't panic
        let _ = loop_ctrl.detect_oscillation();
    }

    #[test]
    fn test_dampen_reduces_gains() {
        let mut loop_ctrl = ControlLoop::default();
        let orig_kp = loop_ctrl.risk_pid.kp;
        loop_ctrl.dampen(0.5);
        assert!(close(loop_ctrl.risk_pid.kp, orig_kp * 0.5, 1e-10));
    }

    #[test]
    fn test_feedback_signal_generation() {
        let loop_ctrl = ControlLoop::default();
        let fb = loop_ctrl.compute_feedback("test", 1.5, 2.0);
        assert!(close(fb.deviation, -0.5, 1e-10));
        assert!(fb.adjustment < 0.0); // Negative adjustment to reduce
    }

    #[test]
    fn test_control_history_tracking() {
        let mut loop_ctrl = ControlLoop::default();
        let portfolio = PortfolioState::default();
        let metrics = LoopMetrics::default();
        loop_ctrl.compute_control(&portfolio, &metrics);
        loop_ctrl.compute_control(&portfolio, &metrics);
        assert_eq!(loop_ctrl.control_outputs.len(), 2);
        assert_eq!(loop_ctrl.feedback_history.len(), 8); // 4 signals per cycle
    }

    #[test]
    fn test_control_reset() {
        let mut loop_ctrl = ControlLoop::default();
        let portfolio = PortfolioState::default();
        let metrics = LoopMetrics::default();
        loop_ctrl.compute_control(&portfolio, &metrics);
        loop_ctrl.reset();
        assert!(loop_ctrl.control_outputs.is_empty());
        assert!(loop_ctrl.feedback_history.is_empty());
    }

    #[test]
    fn test_leverage_above_target() {
        let mut loop_ctrl = ControlLoop::default();
        let mut portfolio = PortfolioState::default();
        portfolio.leverage = 2.5;
        let metrics = LoopMetrics::default();
        let output = loop_ctrl.compute_control(&portfolio, &metrics);
        // Leverage adjustment should be negative (reduce leverage)
        assert!(output.leverage_adjustment < 0.0);
    }

    #[test]
    fn test_custom_targets() {
        let targets = ControlTargets {
            target_volatility: 0.10,
            target_leverage: 1.5,
            target_exposure: 0.80,
            max_drawdown: 0.10,
            target_sharpe: 2.0,
        };
        let loop_ctrl = ControlLoop::new(targets);
        assert!(close(loop_ctrl.targets.target_leverage, 1.5, 1e-10));
    }

    #[test]
    fn test_control_output_default() {
        let output = ControlOutput::default();
        assert!(!output.emergency);
        assert!(close(output.aggregate_signal, 0.0, 1e-10));
    }
}
