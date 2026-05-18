//! Mathematical utility functions for the swarm engine.

/// Clamp a value between lo and hi (inclusive).
pub fn clamp(v: f64, lo: f64, hi: f64) -> f64 {
    v.max(lo).min(hi)
}

/// Simple moving average of a slice of f64 values.
/// Returns 0.0 for an empty slice.
pub fn sma(values: &[f64]) -> f64 {
    if values.is_empty() {
        return 0.0;
    }
    values.iter().sum::<f64>() / values.len() as f64
}

/// Sample standard deviation (Bessel-corrected: divides by N-1).
/// Returns 0.0 for slices with fewer than 2 elements.
pub fn std_dev(values: &[f64]) -> f64 {
    if values.len() < 2 {
        return 0.0;
    }
    let mean = sma(values);
    let variance: f64 = values.iter().map(|v| (v - mean).powi(2)).sum::<f64>()
        / (values.len() - 1) as f64;
    variance.sqrt()
}

/// Parse a string that may represent a number (f64), returning 0.0 on failure.
pub fn parse_f64_or_zero(s: &str) -> f64 {
    s.parse::<f64>().unwrap_or(0.0)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_clamp_within() {
        assert_eq!(clamp(5.0, 0.0, 10.0), 5.0);
    }

    #[test]
    fn test_clamp_below() {
        assert_eq!(clamp(-1.0, 0.0, 10.0), 0.0);
    }

    #[test]
    fn test_clamp_above() {
        assert_eq!(clamp(15.0, 0.0, 10.0), 10.0);
    }

    #[test]
    fn test_sma_basic() {
        assert_eq!(sma(&[2.0, 4.0, 6.0]), 4.0);
    }

    #[test]
    fn test_sma_empty() {
        assert_eq!(sma(&[]), 0.0);
    }

    #[test]
    fn test_sma_single() {
        assert_eq!(sma(&[42.0]), 42.0);
    }

    #[test]
    fn test_std_dev_population_known() {
        // Values [2, 4, 4, 4, 5, 5, 7, 9] — population std dev ≈ 2.0
        let values = vec![2.0, 4.0, 4.0, 4.0, 5.0, 5.0, 7.0, 9.0];
        let sd = std_dev(&values);
        // Sample std dev of these values ≈ 2.138
        assert!((sd - 2.138).abs() < 0.01);
    }

    #[test]
    fn test_std_dev_empty() {
        assert_eq!(std_dev(&[]), 0.0);
    }

    #[test]
    fn test_std_dev_single() {
        assert_eq!(std_dev(&[5.0]), 0.0);
    }

    #[test]
    fn test_std_dev_two() {
        let sd = std_dev(&[0.0, 10.0]);
        assert!((sd - 7.071).abs() < 0.01);
    }

    #[test]
    fn test_parse_f64_or_zero() {
        assert_eq!(parse_f64_or_zero("3.14"), 3.14);
        assert_eq!(parse_f64_or_zero(""), 0.0);
        assert_eq!(parse_f64_or_zero("abc"), 0.0);
        assert_eq!(parse_f64_or_zero("-2.5"), -2.5);
    }
}
