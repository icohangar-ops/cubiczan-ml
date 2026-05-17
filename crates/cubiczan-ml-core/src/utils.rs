//! # Utility Functions
//!
//! General-purpose helpers used across the CubicZan ML ecosystem.

/// Compute the number of parameters for a dense layer with bias.
///
/// # Arguments
/// * `input_dim` — Number of input features.
/// * `output_dim` — Number of output features.
/// * `use_bias` — Whether to include bias parameters.
pub fn dense_params(input_dim: usize, output_dim: usize, use_bias: bool) -> usize {
    let weights = input_dim * output_dim;
    if use_bias {
        weights + output_dim
    } else {
        weights
    }
}

/// Reshape a flat vector into a 2-D layout (row-major).
///
/// Returns `None` if `len(data) != rows * cols`.
pub fn reshape_2d(data: &[f64], rows: usize, cols: usize) -> Option<Vec<Vec<f64>>> {
    if data.len() != rows * cols {
        return None;
    }
    let mut result = Vec::with_capacity(rows);
    for r in 0..rows {
        result.push(data[r * cols..(r + 1) * cols].to_vec());
    }
    Some(result)
}

/// Flatten a 2-D vector into a single contiguous `Vec<f64>`.
pub fn flatten_2d(data: &[Vec<f64>]) -> Vec<f64> {
    data.iter().flat_map(|row| row.iter().copied()).collect()
}

/// Clamp every element of a mutable slice to the range `[min, max]`.
pub fn clip_slice(data: &mut [f64], min: f64, max: f64) {
    for v in data.iter_mut() {
        *v = v.clamp(min, max);
    }
}

/// Compute the softmax of a slice **in place**.
///
/// Uses the numerically stable formulation: subtract the max first.
pub fn softmax(slice: &mut [f64]) {
    if slice.is_empty() {
        return;
    }
    let max_val = slice.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
    let mut sum = 0.0;
    for v in slice.iter_mut() {
        *v = (*v - max_val).exp();
        sum += *v;
    }
    for v in slice.iter_mut() {
        *v /= sum;
    }
}

/// Compute the sigmoid (logistic) function for a scalar value.
pub fn sigmoid(x: f64) -> f64 {
    1.0 / (1.0 + (-x).exp())
}

/// Compute the ReLU function element-wise, returning a new `Vec`.
pub fn relu(data: &[f64]) -> Vec<f64> {
    data.iter().map(|&x| x.max(0.0)).collect()
}

/// One-hot encode a class index.
///
/// Returns a `Vec<f64>` of length `n_classes` with 1.0 at `index` and 0.0 elsewhere.
pub fn one_hot(index: usize, n_classes: usize) -> Vec<f64> {
    let mut v = vec![0.0; n_classes];
    if index < n_classes {
        v[index] = 1.0;
    }
    v
}

/// Compute the mean squared error between two slices.
///
/// Returns `None` if the slices have different lengths.
pub fn mse(a: &[f64], b: &[f64]) -> Option<f64> {
    if a.len() != b.len() {
        return None;
    }
    if a.is_empty() {
        return Some(0.0);
    }
    let sum: f64 = a.iter().zip(b.iter()).map(|(x, y)| (x - y).powi(2)).sum();
    Some(sum / a.len() as f64)
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_dense_params() {
        assert_eq!(dense_params(3, 4, true), 16);  // 12 weights + 4 biases
        assert_eq!(dense_params(3, 4, false), 12);
    }

    #[test]
    fn test_reshape_2d() {
        let flat = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
        let matrix = reshape_2d(&flat, 2, 3).unwrap();
        assert_eq!(matrix[0], vec![1.0, 2.0, 3.0]);
        assert_eq!(matrix[1], vec![4.0, 5.0, 6.0]);
        assert!(reshape_2d(&flat, 2, 4).is_none());
    }

    #[test]
    fn test_flatten_and_reshape_roundtrip() {
        let data = vec![vec![1.0, 2.0], vec![3.0, 4.0], vec![5.0, 6.0]];
        let flat = flatten_2d(&data);
        let restored = reshape_2d(&flat, 3, 2).unwrap();
        assert_eq!(restored, data);
    }

    #[test]
    fn test_softmax() {
        let mut v = vec![1.0, 2.0, 3.0];
        softmax(&mut v);
        assert!((v.iter().sum::<f64>() - 1.0).abs() < 1e-10);
        assert!(v[2] > v[1] && v[1] > v[0]);
    }

    #[test]
    fn test_sigmoid() {
        assert!((sigmoid(0.0) - 0.5).abs() < 1e-10);
        assert!(sigmoid(100.0) > 0.99);
        assert!(sigmoid(-100.0) < 0.01);
    }

    #[test]
    fn test_relu() {
        assert_eq!(relu(&[-1.0, 0.0, 1.0, 2.0]), vec![0.0, 0.0, 1.0, 2.0]);
    }

    #[test]
    fn test_one_hot() {
        assert_eq!(one_hot(0, 3), vec![1.0, 0.0, 0.0]);
        assert_eq!(one_hot(2, 3), vec![0.0, 0.0, 1.0]);
        // out of range → all zeros
        assert_eq!(one_hot(5, 3), vec![0.0, 0.0, 0.0]);
    }

    #[test]
    fn test_mse() {
        let a = vec![1.0, 2.0, 3.0];
        let b = vec![1.5, 2.5, 3.5];
        let err = mse(&a, &b).unwrap();
        assert!((err - 0.25).abs() < 1e-10);
        assert!(mse(&a, &[1.0]).is_none()); // length mismatch
    }
}
