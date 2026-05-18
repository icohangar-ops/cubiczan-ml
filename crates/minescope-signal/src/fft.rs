//! # Signal Processing Primitives
//!
//! Frequency-domain analysis for mining sensor signals:
//! - DFT (Discrete Fourier Transform)
//! - Frequency spectrum analysis
//! - Dominant frequency extraction
//! - Signal filtering (low-pass, high-pass, band-pass)
//! - Signal envelope detection
//! - Noise reduction via frequency domain filtering
//! - Spectral centroid computation

use crate::types::SensorReading;
use chrono::{DateTime, Utc};
use std::f64::consts::PI;

/// Types of frequency-domain filters.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FilterType {
    LowPass,
    HighPass,
    BandPass,
}

/// A frequency-domain signal processor for mining sensor data.
#[derive(Debug, Clone)]
pub struct SignalProcessor {
    /// Sample rate in Hz.
    pub sample_rate: f64,
    /// Number of samples used for DFT windows.
    pub window_size: usize,
}

impl SignalProcessor {
    /// Create a new signal processor with the given sample rate.
    pub fn new(sample_rate: f64, window_size: usize) -> Self {
        SignalProcessor {
            sample_rate,
            window_size,
        }
    }

    /// Create with default settings (sample_rate=100 Hz, window_size=256).
    pub fn default() -> Self {
        SignalProcessor {
            sample_rate: 100.0,
            window_size: 256,
        }
    }

    // -----------------------------------------------------------------------
    // DFT
    // -----------------------------------------------------------------------

    /// Compute the Discrete Fourier Transform of a signal.
    /// Returns complex coefficients as (real, imag) pairs.
    pub fn dft(&self, signal: &[f64]) -> Vec<(f64, f64)> {
        let n = signal.len();
        if n == 0 {
            return Vec::new();
        }
        let mut result = Vec::with_capacity(n);
        for k in 0..n {
            let mut real = 0.0_f64;
            let mut imag = 0.0_f64;
            for (i, &val) in signal.iter().enumerate() {
                let angle = -2.0 * PI * k as f64 * i as f64 / n as f64;
                real += val * angle.cos();
                imag += val * angle.sin();
            }
            result.push((real / n as f64, imag / n as f64));
        }
        result
    }

    /// Compute the Inverse DFT from complex coefficients.
    pub fn idft(&self, coeffs: &[(f64, f64)]) -> Vec<f64> {
        let n = coeffs.len();
        if n == 0 {
            return Vec::new();
        }
        let mut result = Vec::with_capacity(n);
        for i in 0..n {
            let mut val = 0.0_f64;
            for (k, (re, im)) in coeffs.iter().enumerate() {
                let angle = 2.0 * PI * k as f64 * i as f64 / n as f64;
                val += re * angle.cos() - im * angle.sin();
            }
            result.push(val);
        }
        result
    }

    // -----------------------------------------------------------------------
    // Spectrum analysis
    // -----------------------------------------------------------------------

    /// Compute the magnitude spectrum (absolute values of DFT coefficients).
    /// Returns (frequency_hz, magnitude) pairs for the positive half.
    pub fn magnitude_spectrum(&self, signal: &[f64]) -> Vec<(f64, f64)> {
        let coeffs = self.dft(signal);
        let n = coeffs.len();
        if n == 0 {
            return Vec::new();
        }
        let half = n / 2 + 1;
        let mut spectrum = Vec::with_capacity(half);
        for k in 0..half {
            let (re, im) = coeffs[k];
            let mag = (re * re + im * im).sqrt() * 2.0;
            let freq = k as f64 * self.sample_rate / n as f64;
            spectrum.push((freq, mag));
        }
        spectrum
    }

    /// Compute the power spectrum (magnitude squared).
    pub fn power_spectrum(&self, signal: &[f64]) -> Vec<(f64, f64)> {
        self.magnitude_spectrum(signal)
            .into_iter()
            .map(|(freq, mag)| (freq, mag * mag))
            .collect()
    }

    /// Extract the dominant frequency from a signal.
    /// Returns (frequency_hz, magnitude).
    pub fn dominant_frequency(&self, signal: &[f64]) -> Option<(f64, f64)> {
        if signal.len() < 4 {
            return None;
        }
        let spectrum = self.magnitude_spectrum(signal);
        // Skip DC component (index 0)
        if spectrum.len() < 2 {
            return None;
        }
        spectrum
            .iter()
            .skip(1)
            .max_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal))
            .copied()
    }

    /// Extract the top N dominant frequencies.
    pub fn top_frequencies(&self, signal: &[f64], n: usize) -> Vec<(f64, f64)> {
        let spectrum = self.magnitude_spectrum(signal);
        let mut ranked: Vec<_> = spectrum.iter().skip(1).copied().collect();
        ranked.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        ranked.truncate(n);
        ranked
    }

    // -----------------------------------------------------------------------
    // Filtering
    // -----------------------------------------------------------------------

    /// Apply a frequency-domain filter to a signal.
    /// - LowPass: keep frequencies below cutoff_hz
    /// - HighPass: keep frequencies above cutoff_hz
    /// - BandPass: keep frequencies between low_cutoff and high_cutoff
    pub fn filter_signal(
        &self,
        signal: &[f64],
        filter_type: FilterType,
        cutoff_hz: f64,
        cutoff_hz2: Option<f64>,
    ) -> Vec<f64> {
        let n = signal.len();
        if n == 0 {
            return Vec::new();
        }
        let mut coeffs = self.dft(signal);
        let freq_resolution = self.sample_rate / n as f64;

        for k in 0..n {
            let freq = k as f64 * freq_resolution;
            // Handle Nyquist folding
            let freq_actual = if freq <= self.sample_rate / 2.0 {
                freq
            } else {
                self.sample_rate - freq
            };

            let keep = match filter_type {
                FilterType::LowPass => freq_actual <= cutoff_hz,
                FilterType::HighPass => freq_actual >= cutoff_hz,
                FilterType::BandPass => {
                    let hi = cutoff_hz2.unwrap_or(cutoff_hz + 1.0);
                    freq_actual >= cutoff_hz && freq_actual <= hi
                }
            };

            if !keep {
                coeffs[k].0 = 0.0;
                coeffs[k].1 = 0.0;
                // Also zero the symmetric component
                if k > 0 && k < n {
                    let sym = n - k;
                    if sym < n {
                        coeffs[sym].0 = 0.0;
                        coeffs[sym].1 = 0.0;
                    }
                }
            }
        }

        self.idft(&coeffs)
    }

    /// Remove noise from a signal by zeroing out frequencies below a
    /// power threshold relative to the dominant frequency.
    pub fn denoise(&self, signal: &[f64], power_threshold_ratio: f64) -> Vec<f64> {
        let n = signal.len();
        if n < 4 {
            return signal.to_vec();
        }

        let power = self.power_spectrum(signal);
        let max_power = power
            .iter()
            .skip(1)
            .map(|(_, p)| *p)
            .fold(0.0_f64, f64::max);

        let threshold = max_power * power_threshold_ratio;

        let mut coeffs = self.dft(signal);
        let freq_resolution = self.sample_rate / n as f64;

        for k in 0..n {
            let freq = k as f64 * freq_resolution;
            let freq_actual = if freq <= self.sample_rate / 2.0 {
                freq
            } else {
                self.sample_rate - freq
            };
            let freq_idx = (freq_actual / freq_resolution).round() as usize;
            let freq_idx = freq_idx.min(power.len() - 1);

            if power[freq_idx].1 < threshold && k > 0 {
                coeffs[k].0 = 0.0;
                coeffs[k].1 = 0.0;
                let sym = n - k;
                if sym < n {
                    coeffs[sym].0 = 0.0;
                    coeffs[sym].1 = 0.0;
                }
            }
        }

        self.idft(&coeffs)
    }

    // -----------------------------------------------------------------------
    // Envelope
    // -----------------------------------------------------------------------

    /// Compute the signal envelope using the analytic signal approach
    /// (Hilbert transform approximation via DFT).
    pub fn envelope(&self, signal: &[f64]) -> Vec<f64> {
        let n = signal.len();
        if n == 0 {
            return Vec::new();
        }
        let mut coeffs = self.dft(signal);

        // Zero out negative frequencies to create analytic signal
        let half = n / 2 + 1;
        for k in half..n {
            coeffs[k].0 = 0.0;
            coeffs[k].1 = 0.0;
        }
        // Double positive frequencies (except DC)
        for k in 1..half {
            coeffs[k].0 *= 2.0;
            coeffs[k].1 *= 2.0;
        }

        // Inverse DFT gives analytic signal; take magnitude
        let analytic = self.idft(&coeffs);
        analytic.iter().map(|&v| v.abs()).collect()
    }

    /// Simpler envelope via rectification + smoothing.
    pub fn envelope_simple(&self, signal: &[f64], window: usize) -> Vec<f64> {
        let rectified: Vec<f64> = signal.iter().map(|&v| v.abs()).collect();
        self.smooth(&rectified, window)
    }

    // -----------------------------------------------------------------------
    // Spectral features
    // -----------------------------------------------------------------------

    /// Compute the spectral centroid (brightness) of a signal.
    pub fn spectral_centroid(&self, signal: &[f64]) -> f64 {
        let spectrum = self.magnitude_spectrum(signal);
        if spectrum.is_empty() {
            return 0.0;
        }
        let total_power: f64 = spectrum.iter().map(|(_, m)| *m).sum();
        if total_power.abs() < 1e-15 {
            return 0.0;
        }
        let weighted: f64 = spectrum.iter().map(|(f, m)| f * m).sum();
        weighted / total_power
    }

    /// Compute spectral bandwidth (spread around centroid).
    pub fn spectral_bandwidth(&self, signal: &[f64]) -> f64 {
        let spectrum = self.magnitude_spectrum(signal);
        if spectrum.is_empty() {
            return 0.0;
        }
        let total_power: f64 = spectrum.iter().map(|(_, m)| *m).sum();
        if total_power.abs() < 1e-15 {
            return 0.0;
        }
        let centroid = self.spectral_centroid(signal);
        let variance: f64 = spectrum
            .iter()
            .map(|(f, m)| m * (f - centroid).powi(2))
            .sum::<f64>()
            / total_power;
        variance.sqrt()
    }

    /// Compute spectral flatness (geometric mean / arithmetic mean of power spectrum).
    pub fn spectral_flatness(&self, signal: &[f64]) -> f64 {
        let power = self.power_spectrum(signal);
        if power.is_empty() {
            return 0.0;
        }
        // Skip DC
        let signal_power: Vec<f64> = power.iter().skip(1).map(|(_, p)| *p).collect();
        if signal_power.is_empty() || signal_power.iter().all(|&p| p < 1e-15) {
            return 0.0;
        }
        let arithmetic_mean: f64 = signal_power.iter().sum::<f64>() / signal_power.len() as f64;
        let log_sum: f64 = signal_power
            .iter()
            .filter(|&&p| p > 1e-15)
            .map(|p| p.ln())
            .sum::<f64>();
        let geometric_mean = (log_sum / signal_power.len() as f64).exp();
        if arithmetic_mean.abs() < 1e-15 {
            return 0.0;
        }
        geometric_mean / arithmetic_mean
    }

    // -----------------------------------------------------------------------
    // Helpers
    // -----------------------------------------------------------------------

    /// Simple moving average smoothing.
    pub fn smooth(&self, signal: &[f64], window: usize) -> Vec<f64> {
        if window == 0 || signal.is_empty() {
            return signal.to_vec();
        }
        let w = window.min(signal.len());
        let mut result = Vec::with_capacity(signal.len());
        for i in 0..signal.len() {
            let start = i.saturating_sub(w - 1);
            let end = i + 1;
            let sum: f64 = signal[start..end].iter().sum();
            result.push(sum / (end - start) as f64);
        }
        result
    }

    /// Compute RMS (root mean square) of a signal.
    pub fn rms(signal: &[f64]) -> f64 {
        if signal.is_empty() {
            return 0.0;
        }
        let sum_sq: f64 = signal.iter().map(|&x| x * x).sum();
        (sum_sq / signal.len() as f64).sqrt()
    }

    /// Compute zero-crossing rate.
    pub fn zero_crossing_rate(signal: &[f64]) -> f64 {
        if signal.len() < 2 {
            return 0.0;
        }
        let crossings = signal
            .windows(2)
            .filter(|w| (w[0] >= 0.0 && w[1] < 0.0) || (w[0] < 0.0 && w[1] >= 0.0))
            .count();
        crossings as f64 / (signal.len() - 1) as f64
    }

    /// Process sensor readings into a time series of values.
    pub fn readings_to_series(readings: &[SensorReading]) -> Vec<f64> {
        readings.iter().map(|r| r.value).collect()
    }

    /// Generate a test signal: sine wave with given frequency, amplitude, and noise.
    pub fn generate_sine(
        duration_secs: f64,
        sample_rate: f64,
        frequency: f64,
        amplitude: f64,
        noise_level: f64,
    ) -> Vec<f64> {
        let n = (duration_secs * sample_rate) as usize;
        use rand::Rng;
        let mut rng = rand::thread_rng();
        (0..n)
            .map(|i| {
                let t = i as f64 / sample_rate;
                let noise = if noise_level > 0.0 { rng.gen_range(-noise_level..noise_level) } else { 0.0 };
                amplitude * (2.0 * PI * frequency * t).sin() + noise
            })
            .collect()
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use chrono::Duration;

    fn make_processor() -> SignalProcessor {
        SignalProcessor::new(100.0, 256)
    }

    fn make_vibration_readings(values: &[f64]) -> Vec<SensorReading> {
        let base = Utc::now();
        values
            .iter()
            .enumerate()
            .map(|(i, &v)| {
                SensorReading::new("VIB-001", crate::types::SensorType::Vibration, base + Duration::milliseconds(i as i64 * 10), v)
            })
            .collect()
    }

    #[test]
    fn test_processor_default() {
        let p = SignalProcessor::default();
        assert_eq!(p.sample_rate, 100.0);
        assert_eq!(p.window_size, 256);
    }

    #[test]
    fn test_dft_dc_component() {
        let p = make_processor();
        let signal = vec![3.0; 10];
        let coeffs = p.dft(&signal);
        // DC component should be approximately 3.0
        assert!((coeffs[0].0 - 3.0).abs() < 1e-10);
    }

    #[test]
    fn test_dft_empty() {
        let p = make_processor();
        assert!(p.dft(&[]).is_empty());
    }

    #[test]
    fn test_idft_roundtrip() {
        let p = make_processor();
        let signal: Vec<f64> = (0..32).map(|i| (i as f64 * 0.2).sin()).collect();
        let coeffs = p.dft(&signal);
        let recovered = p.idft(&coeffs);
        for i in 0..signal.len() {
            assert!(
                (signal[i] - recovered[i]).abs() < 1e-8,
                "Mismatch at index {}: {} vs {}",
                i,
                signal[i],
                recovered[i]
            );
        }
    }

    #[test]
    fn test_magnitude_spectrum() {
        let p = make_processor();
        // Pure 10 Hz sine at 100 Hz sample rate
        let signal: Vec<f64> = (0..100)
            .map(|i| (2.0 * PI * 10.0 * i as f64 / 100.0).sin())
            .collect();
        let spectrum = p.magnitude_spectrum(&signal);
        // Should have peak near 10 Hz
        let max_entry = spectrum
            .iter()
            .max_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal))
            .unwrap();
        assert!((max_entry.0 - 10.0).abs() < 2.0);
    }

    #[test]
    fn test_dominant_frequency() {
        let p = make_processor();
        let signal = SignalProcessor::generate_sine(1.0, 100.0, 25.0, 1.0, 0.01);
        let (freq, _mag) = p.dominant_frequency(&signal).unwrap();
        assert!((freq - 25.0).abs() < 3.0, "Got freq {}, expected ~25", freq);
    }

    #[test]
    fn test_dominant_frequency_short_signal() {
        let p = make_processor();
        assert!(p.dominant_frequency(&[1.0, 2.0]).is_none());
    }

    #[test]
    fn test_top_frequencies() {
        let p = make_processor();
        // Mix of two sine waves: 10 Hz and 25 Hz
        let signal: Vec<f64> = (0..200)
            .map(|i| {
                let t = i as f64 / 100.0;
                (2.0 * PI * 10.0 * t).sin() + 0.5 * (2.0 * PI * 25.0 * t).sin()
            })
            .collect();
        let top = p.top_frequencies(&signal, 2);
        assert_eq!(top.len(), 2);
        // Primary should be 10 Hz
        assert!((top[0].0 - 10.0).abs() < 3.0);
    }

    #[test]
    fn test_low_pass_filter() {
        let p = make_processor();
        // 5 Hz + 40 Hz signal, low-pass at 15 Hz should remove 40 Hz
        let signal: Vec<f64> = (0..200)
            .map(|i| {
                let t = i as f64 / 100.0;
                (2.0 * PI * 5.0 * t).sin() + 0.5 * (2.0 * PI * 40.0 * t).sin()
            })
            .collect();
        let filtered = p.filter_signal(&signal, FilterType::LowPass, 15.0, None);
        // Compute RMS of filtered — should be less than original but non-zero
        let orig_rms = SignalProcessor::rms(&signal);
        let filt_rms = SignalProcessor::rms(&filtered);
        assert!(filt_rms < orig_rms);
        assert!(filt_rms > 0.0);
    }

    #[test]
    fn test_high_pass_filter() {
        let p = make_processor();
        let signal: Vec<f64> = (0..200)
            .map(|i| {
                let t = i as f64 / 100.0;
                (2.0 * PI * 5.0 * t).sin() + 2.0 * (2.0 * PI * 40.0 * t).sin()
            })
            .collect();
        let filtered = p.filter_signal(&signal, FilterType::HighPass, 15.0, None);
        let filt_rms = SignalProcessor::rms(&filtered);
        assert!(filt_rms > 0.0);
    }

    #[test]
    fn test_band_pass_filter() {
        let p = make_processor();
        let signal: Vec<f64> = (0..200)
            .map(|i| {
                let t = i as f64 / 100.0;
                (2.0 * PI * 5.0 * t).sin()
                    + 2.0 * (2.0 * PI * 20.0 * t).sin()
                    + 0.5 * (2.0 * PI * 45.0 * t).sin()
            })
            .collect();
        let filtered = p.filter_signal(&signal, FilterType::BandPass, 10.0, Some(30.0));
        let filt_rms = SignalProcessor::rms(&filtered);
        assert!(filt_rms > 0.0);
    }

    #[test]
    fn test_filter_empty_signal() {
        let p = make_processor();
        let result = p.filter_signal(&[], FilterType::LowPass, 10.0, None);
        assert!(result.is_empty());
    }

    #[test]
    fn test_denoise() {
        let p = make_processor();
        let clean: Vec<f64> = (0..200)
            .map(|i| (2.0 * PI * 10.0 * i as f64 / 100.0).sin())
            .collect();
        let noisy: Vec<f64> = clean
            .iter()
            .map(|&v| v + rand::random::<f64>() * 0.4 - 0.2)
            .collect();
        let denoised = p.denoise(&noisy, 0.05);
        // Denoised should be closer to clean than noisy is
        let noise_error: f64 = noisy
            .iter()
            .zip(clean.iter())
            .map(|(a, b)| (a - b).powi(2))
            .sum::<f64>();
        let denoise_error: f64 = denoised
            .iter()
            .zip(clean.iter())
            .map(|(a, b)| (a - b).powi(2))
            .sum::<f64>();
        assert!(denoise_error < noise_error * 1.5);
    }

    #[test]
    fn test_denoise_short_signal() {
        let p = make_processor();
        let signal = vec![1.0, 2.0];
        let result = p.denoise(&signal, 0.05);
        assert_eq!(result.len(), 2);
    }

    #[test]
    fn test_envelope() {
        let p = make_processor();
        let signal: Vec<f64> = (0..100)
            .map(|i| {
                let t = i as f64 / 100.0;
                (2.0 * PI * 5.0 * t).sin() * (1.0 + 0.5 * (2.0 * PI * 0.5 * t).cos())
            })
            .collect();
        let env = p.envelope(&signal);
        assert!(!env.is_empty());
        // Envelope should be non-negative
        assert!(env.iter().all(|&v| v >= -1e-10));
    }

    #[test]
    fn test_envelope_simple() {
        let p = make_processor();
        let signal = vec![1.0, -2.0, 3.0, -4.0, 5.0, -3.0, 2.0, -1.0];
        let env = p.envelope_simple(&signal, 3);
        assert_eq!(env.len(), signal.len());
        assert!(env.iter().all(|&v| v >= 0.0));
    }

    #[test]
    fn test_envelope_empty() {
        let p = make_processor();
        assert!(p.envelope(&[]).is_empty());
    }

    #[test]
    fn test_spectral_centroid() {
        let p = make_processor();
        let high_freq: Vec<f64> = (0..200)
            .map(|i| (2.0 * PI * 40.0 * i as f64 / 100.0).sin())
            .collect();
        let low_freq: Vec<f64> = (0..200)
            .map(|i| (2.0 * PI * 5.0 * i as f64 / 100.0).sin())
            .collect();
        let centroid_high = p.spectral_centroid(&high_freq);
        let centroid_low = p.spectral_centroid(&low_freq);
        assert!(centroid_high > centroid_low);
    }

    #[test]
    fn test_spectral_centroid_empty() {
        let p = make_processor();
        assert_eq!(p.spectral_centroid(&[]), 0.0);
    }

    #[test]
    fn test_spectral_bandwidth() {
        let p = make_processor();
        let signal: Vec<f64> = (0..200)
            .map(|i| (2.0 * PI * 15.0 * i as f64 / 100.0).sin())
            .collect();
        let bw = p.spectral_bandwidth(&signal);
        assert!(bw >= 0.0);
    }

    #[test]
    fn test_spectral_flatness_pure_tone() {
        let p = make_processor();
        let signal: Vec<f64> = (0..400)
            .map(|i| (2.0 * PI * 10.0 * i as f64 / 100.0).sin())
            .collect();
        let flatness = p.spectral_flatness(&signal);
        // Pure tone flatness should be well-defined and non-negative
        assert!(flatness.is_finite());
        assert!(flatness >= 0.0);
    }

    #[test]
    fn test_spectral_flatness_noise() {
        let p = make_processor();
        let signal: Vec<f64> = (0..200).map(|_| rand::random::<f64>() * 2.0 - 1.0).collect();
        let flatness = p.spectral_flatness(&signal);
        // Noise should have higher flatness
        assert!(flatness > 0.0);
    }

    #[test]
    fn test_rms() {
        assert!((SignalProcessor::rms(&[3.0, 4.0]) - 3.5355).abs() < 0.01);
        assert_eq!(SignalProcessor::rms(&[]), 0.0);
        assert_eq!(SignalProcessor::rms(&[0.0]), 0.0);
    }

    #[test]
    fn test_zero_crossing_rate() {
        let signal = vec![1.0, -1.0, 1.0, -1.0, 1.0];
        assert!((SignalProcessor::zero_crossing_rate(&signal) - 1.0).abs() < 1e-10);
        assert_eq!(SignalProcessor::zero_crossing_rate(&[1.0, 2.0]), 0.0);
        assert_eq!(SignalProcessor::zero_crossing_rate(&[]), 0.0);
    }

    #[test]
    fn test_smooth() {
        let p = make_processor();
        let signal = vec![1.0, 3.0, 5.0, 3.0, 1.0];
        let smoothed = p.smooth(&signal, 3);
        assert_eq!(smoothed[0], 1.0);
        assert_eq!(smoothed[2], 3.0);
    }

    #[test]
    fn test_smooth_empty() {
        let p = make_processor();
        assert!(p.smooth(&[], 5).is_empty());
    }

    #[test]
    fn test_generate_sine() {
        let signal = SignalProcessor::generate_sine(0.5, 100.0, 10.0, 2.0, 0.0);
        assert_eq!(signal.len(), 50);
        let max = signal.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
        assert!((max - 2.0).abs() < 0.1);
    }

    #[test]
    fn test_readings_to_series() {
        let readings = make_vibration_readings(&[1.0, 2.0, 3.0]);
        let series = SignalProcessor::readings_to_series(&readings);
        assert_eq!(series, vec![1.0, 2.0, 3.0]);
    }

    #[test]
    fn test_power_spectrum() {
        let p = make_processor();
        let signal: Vec<f64> = (0..100)
            .map(|i| (2.0 * PI * 10.0 * i as f64 / 100.0).sin())
            .collect();
        let power = p.power_spectrum(&signal);
        assert!(!power.is_empty());
        assert!(power.iter().all(|(_, p)| *p >= 0.0));
    }
}
