//! # Mining Process Signal Analysis
//!
//! Stage-specific signal benchmarks, process efficiency scoring,
//! throughput monitoring, energy analysis, bottleneck detection,
//! and cross-stage correlation analysis.

use crate::types::{ProcessingStage, ProcessSignal};
use chrono::{DateTime, Utc};
use std::collections::HashMap;

/// Typical benchmark ranges for each processing stage.
#[derive(Debug, Clone)]
pub struct StageBenchmark {
    pub stage: ProcessingStage,
    /// Typical throughput range (tons/hour).
    pub throughput_range: (f64, f64),
    /// Optimal throughput (tons/hour).
    pub optimal_throughput: f64,
    /// Typical energy per ton (kWh/ton).
    pub energy_per_ton_range: (f64, f64),
    /// Typical particle size after this stage (mm).
    pub particle_size_range: (f64, f64),
    /// Typical temperature range (°C).
    pub temperature_range: (f64, f64),
    /// Typical pressure range (kPa).
    pub pressure_range: (f64, f64),
}

impl StageBenchmark {
    /// Get benchmarks for a specific processing stage.
    pub fn for_stage(stage: ProcessingStage) -> Self {
        match stage {
            ProcessingStage::Crushing => StageBenchmark {
                stage,
                throughput_range: (500.0, 2000.0),
                optimal_throughput: 1500.0,
                energy_per_ton_range: (0.5, 2.0),
                particle_size_range: (50.0, 200.0),
                temperature_range: (20.0, 60.0),
                pressure_range: (0.0, 100.0),
            },
            ProcessingStage::Grinding => StageBenchmark {
                stage,
                throughput_range: (100.0, 500.0),
                optimal_throughput: 350.0,
                energy_per_ton_range: (5.0, 25.0),
                particle_size_range: (0.05, 0.5),
                temperature_range: (30.0, 80.0),
                pressure_range: (0.0, 200.0),
            },
            ProcessingStage::Flotation => StageBenchmark {
                stage,
                throughput_range: (50.0, 300.0),
                optimal_throughput: 200.0,
                energy_per_ton_range: (1.0, 5.0),
                particle_size_range: (0.01, 0.1),
                temperature_range: (20.0, 40.0),
                pressure_range: (80.0, 200.0),
            },
            ProcessingStage::Leaching => StageBenchmark {
                stage,
                throughput_range: (30.0, 150.0),
                optimal_throughput: 100.0,
                energy_per_ton_range: (10.0, 40.0),
                particle_size_range: (0.0, 0.05),
                temperature_range: (40.0, 90.0),
                pressure_range: (100.0, 500.0),
            },
            ProcessingStage::Smelting => StageBenchmark {
                stage,
                throughput_range: (20.0, 100.0),
                optimal_throughput: 60.0,
                energy_per_ton_range: (300.0, 800.0),
                particle_size_range: (0.0, 0.01),
                temperature_range: (1000.0, 1500.0),
                pressure_range: (100.0, 300.0),
            },
            ProcessingStage::Refining => StageBenchmark {
                stage,
                throughput_range: (10.0, 50.0),
                optimal_throughput: 30.0,
                energy_per_ton_range: (200.0, 600.0),
                particle_size_range: (0.0, 0.01),
                temperature_range: (800.0, 1200.0),
                pressure_range: (50.0, 200.0),
            },
            ProcessingStage::Conveying => StageBenchmark {
                stage,
                throughput_range: (500.0, 3000.0),
                optimal_throughput: 2000.0,
                energy_per_ton_range: (0.1, 0.5),
                particle_size_range: (0.0, 200.0),
                temperature_range: (15.0, 50.0),
                pressure_range: (0.0, 50.0),
            },
            ProcessingStage::Sorting => StageBenchmark {
                stage,
                throughput_range: (100.0, 800.0),
                optimal_throughput: 500.0,
                energy_per_ton_range: (0.5, 3.0),
                particle_size_range: (5.0, 100.0),
                temperature_range: (15.0, 40.0),
                pressure_range: (0.0, 100.0),
            },
        }
    }
}

/// Process efficiency analysis result.
#[derive(Debug, Clone)]
pub struct EfficiencyReport {
    pub stage: ProcessingStage,
    pub throughput_score: f64,    // 0-100
    pub energy_score: f64,        // 0-100 (higher = more efficient)
    pub overall_efficiency: f64,  // 0-100
    pub throughput_actual: f64,
    pub throughput_optimal: f64,
    pub energy_per_ton: f64,
    pub energy_optimal: f64,
    pub bottlenecks: Vec<String>,
}

/// The process analyzer for mining operations.
#[derive(Debug, Clone)]
pub struct ProcessAnalyzer {
    benchmarks: HashMap<ProcessingStage, StageBenchmark>,
    /// Historical throughput data per stage.
    throughput_history: HashMap<ProcessingStage, Vec<f64>>,
    /// Historical energy data per stage (kWh/ton).
    energy_history: HashMap<ProcessingStage, Vec<f64>>,
}

impl ProcessAnalyzer {
    /// Create a new process analyzer with default benchmarks.
    pub fn new() -> Self {
        let benchmarks = ProcessingStage::all_ordered()
            .iter()
            .map(|&stage| (stage, StageBenchmark::for_stage(stage)))
            .collect();
        ProcessAnalyzer {
            benchmarks,
            throughput_history: HashMap::new(),
            energy_history: HashMap::new(),
        }
    }

    /// Get benchmark for a stage.
    pub fn benchmark(&self, stage: ProcessingStage) -> Option<&StageBenchmark> {
        self.benchmarks.get(&stage)
    }

    /// Record throughput measurement for a stage.
    pub fn record_throughput(&mut self, stage: ProcessingStage, throughput: f64) {
        self.throughput_history
            .entry(stage)
            .or_default()
            .push(throughput);
    }

    /// Record energy measurement for a stage.
    pub fn record_energy(&mut self, stage: ProcessingStage, energy_per_ton: f64) {
        self.energy_history
            .entry(stage)
            .or_default()
            .push(energy_per_ton);
    }

    // -----------------------------------------------------------------------
    // Efficiency scoring
    // -----------------------------------------------------------------------

    /// Compute process efficiency for a stage given actual throughput and energy.
    pub fn compute_efficiency(
        &self,
        stage: ProcessingStage,
        actual_throughput: f64,
        energy_per_ton: f64,
    ) -> EfficiencyReport {
        let bm = self.benchmarks.get(&stage).cloned().unwrap_or_else(|| StageBenchmark::for_stage(stage));

        // Throughput score: how close to optimal
        let throughput_score = if bm.optimal_throughput > 0.0 {
            let ratio = actual_throughput / bm.optimal_throughput;
            // Optimal at ratio=1.0, penalize both under and over
            (1.0 - (ratio - 1.0).abs()).clamp(0.0, 1.0) * 100.0
        } else {
            50.0
        };

        // Energy score: lower energy per ton is better
        let energy_optimal = bm.energy_per_ton_range.0;
        let energy_score = if energy_optimal > 0.0 {
            let ratio = energy_per_ton / energy_optimal;
            // Better when close to or below optimal
            (2.0 - ratio).clamp(0.0, 1.0) * 100.0
        } else {
            50.0
        };

        let overall_efficiency = throughput_score * 0.6 + energy_score * 0.4;

        // Bottleneck detection
        let mut bottlenecks = Vec::new();
        if throughput_score < 40.0 {
            bottlenecks.push(format!(
                "Low throughput: {:.0} t/h vs optimal {:.0} t/h",
                actual_throughput, bm.optimal_throughput
            ));
        }
        if energy_score < 40.0 {
            bottlenecks.push(format!(
                "High energy consumption: {:.1} kWh/t vs optimal {:.1} kWh/t",
                energy_per_ton, energy_optimal
            ));
        }

        EfficiencyReport {
            stage,
            throughput_score,
            energy_score,
            overall_efficiency,
            throughput_actual: actual_throughput,
            throughput_optimal: bm.optimal_throughput,
            energy_per_ton,
            energy_optimal,
            bottlenecks,
        }
    }

    // -----------------------------------------------------------------------
    // Throughput monitoring
    // -----------------------------------------------------------------------

    /// Compute average throughput for a stage.
    pub fn average_throughput(&self, stage: ProcessingStage) -> f64 {
        self.throughput_history
            .get(&stage)
            .map(|v| v.iter().sum::<f64>() / v.len().max(1) as f64)
            .unwrap_or(0.0)
    }

    /// Compute throughput trend (positive = increasing, negative = decreasing).
    pub fn throughput_trend(&self, stage: ProcessingStage) -> f64 {
        let history = match self.throughput_history.get(&stage) {
            Some(h) if h.len() >= 4 => h,
            _ => return 0.0,
        };
        let n = history.len();
        let first_half_avg: f64 = history[..n / 2].iter().sum::<f64>() / (n / 2) as f64;
        let second_half_avg: f64 = history[n / 2..].iter().sum::<f64>() / (n - n / 2) as f64;
        if first_half_avg.abs() < 1e-15 {
            return 0.0;
        }
        (second_half_avg - first_half_avg) / first_half_avg
    }

    /// Check if throughput is within normal range.
    pub fn throughput_in_range(&self, stage: ProcessingStage, throughput: f64) -> bool {
        let bm = self.benchmarks.get(&stage).cloned().unwrap_or_else(|| StageBenchmark::for_stage(stage));
        throughput >= bm.throughput_range.0 && throughput <= bm.throughput_range.1
    }

    // -----------------------------------------------------------------------
    // Energy analysis
    // -----------------------------------------------------------------------

    /// Compute average energy per ton for a stage.
    pub fn average_energy(&self, stage: ProcessingStage) -> f64 {
        self.energy_history
            .get(&stage)
            .map(|v| v.iter().sum::<f64>() / v.len().max(1) as f64)
            .unwrap_or(0.0)
    }

    /// Compute energy efficiency trend.
    pub fn energy_trend(&self, stage: ProcessingStage) -> f64 {
        let history = match self.energy_history.get(&stage) {
            Some(h) if h.len() >= 4 => h,
            _ => return 0.0,
        };
        let n = history.len();
        let first_half_avg: f64 = history[..n / 2].iter().sum::<f64>() / (n / 2) as f64;
        let second_half_avg: f64 = history[n / 2..].iter().sum::<f64>() / (n - n / 2) as f64;
        if first_half_avg.abs() < 1e-15 {
            return 0.0;
        }
        (second_half_avg - first_half_avg) / first_half_avg
    }

    /// Compute total energy consumption across all recorded stages.
    pub fn total_energy_consumption(&self) -> f64 {
        self.energy_history
            .values()
            .flat_map(|v| v.iter())
            .sum()
    }

    // -----------------------------------------------------------------------
    // Bottleneck detection
    // -----------------------------------------------------------------------

    /// Find the bottleneck stage (lowest efficiency).
    pub fn find_bottleneck(
        &self,
        current_throughputs: &HashMap<ProcessingStage, f64>,
        current_energies: &HashMap<ProcessingStage, f64>,
    ) -> Option<ProcessingStage> {
        let mut min_efficiency = f64::MAX;
        let mut bottleneck = None;

        for (&stage, bm) in &self.benchmarks {
            let throughput = current_throughputs.get(&stage).copied().unwrap_or(0.0);
            let energy = current_energies.get(&stage).copied().unwrap_or(bm.energy_per_ton_range.1);

            if bm.optimal_throughput > 0.0 {
                let tp_score = (1.0 - (throughput / bm.optimal_throughput - 1.0).abs()).clamp(0.0, 1.0);
                let en_score = if bm.energy_per_ton_range.0 > 0.0 {
                    (2.0 - energy / bm.energy_per_ton_range.0).clamp(0.0, 1.0)
                } else {
                    0.5
                };
                let efficiency = tp_score * 0.6 + en_score * 0.4;

                if efficiency < min_efficiency {
                    min_efficiency = efficiency;
                    bottleneck = Some(stage);
                }
            }
        }

        bottleneck
    }

    /// Get all stages ranked by efficiency (worst first).
    pub fn rank_stages_by_efficiency(
        &self,
        current_throughputs: &HashMap<ProcessingStage, f64>,
        current_energies: &HashMap<ProcessingStage, f64>,
    ) -> Vec<(ProcessingStage, f64)> {
        let mut ranked: Vec<(ProcessingStage, f64)> = ProcessingStage::all_ordered()
            .iter()
            .map(|&stage| {
                let throughput = current_throughputs.get(&stage).copied().unwrap_or(0.0);
                let energy = current_energies.get(&stage).copied().unwrap_or(10.0);
                let report = self.compute_efficiency(stage, throughput, energy);
                (stage, report.overall_efficiency)
            })
            .collect();
        ranked.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal));
        ranked
    }

    // -----------------------------------------------------------------------
    // Cross-stage correlation
    // -----------------------------------------------------------------------

    /// Analyze correlation between throughput of two stages.
    /// High correlation suggests the stages are well-balanced.
    pub fn cross_stage_correlation(&self, stage_a: ProcessingStage, stage_b: ProcessingStage) -> f64 {
        let hist_a = match self.throughput_history.get(&stage_a) {
            Some(h) => h,
            None => return 0.0,
        };
        let hist_b = match self.throughput_history.get(&stage_b) {
            Some(h) => h,
            None => return 0.0,
        };

        let len = hist_a.len().min(hist_b.len());
        if len < 3 {
            return 0.0;
        }

        let a = &hist_a[hist_a.len() - len..];
        let b = &hist_b[hist_b.len() - len..];
        let n = len as f64;
        let ma = a.iter().sum::<f64>() / n;
        let mb = b.iter().sum::<f64>() / n;

        let mut cov = 0.0_f64;
        let mut va = 0.0_f64;
        let mut vb = 0.0_f64;
        for i in 0..len {
            let da = a[i] - ma;
            let db = b[i] - mb;
            cov += da * db;
            va += da * da;
            vb += db * db;
        }

        let denom = (va * vb).sqrt();
        if denom.abs() < 1e-15 {
            0.0
        } else {
            cov / denom
        }
    }

    /// Compute the overall pipeline efficiency (product of stage efficiencies).
    pub fn pipeline_efficiency(
        &self,
        current_throughputs: &HashMap<ProcessingStage, f64>,
        current_energies: &HashMap<ProcessingStage, f64>,
    ) -> f64 {
        let stages = ProcessingStage::all_ordered();
        if stages.is_empty() {
            return 0.0;
        }

        let product: f64 = stages
            .iter()
            .map(|&stage| {
                let tp = current_throughputs.get(&stage).copied().unwrap_or(0.0);
                let en = current_energies.get(&stage).copied().unwrap_or(10.0);
                let report = self.compute_efficiency(stage, tp, en);
                report.overall_efficiency / 100.0
            })
            .product();

        product * 100.0
    }

    // -----------------------------------------------------------------------
    // Process signal analysis
    // -----------------------------------------------------------------------

    /// Analyze a stream of process signals and generate stage-level summaries.
    pub fn analyze_signals(&self, signals: &[ProcessSignal]) -> HashMap<ProcessingStage, Vec<f64>> {
        let mut stage_values: HashMap<ProcessingStage, Vec<f64>> = HashMap::new();
        for signal in signals {
            stage_values
                .entry(signal.stage)
                .or_default()
                .push(signal.value);
        }
        stage_values
    }

    /// Generate a process health summary across all stages.
    pub fn health_summary(
        &self,
        current_throughputs: &HashMap<ProcessingStage, f64>,
        current_energies: &HashMap<ProcessingStage, f64>,
    ) -> String {
        let rankings = self.rank_stages_by_efficiency(current_throughputs, current_energies);
        let bottleneck = self.find_bottleneck(current_throughputs, current_energies);
        let pipeline_eff = self.pipeline_efficiency(current_throughputs, current_energies);

        let mut summary = format!("Pipeline Efficiency: {:.1}%\n", pipeline_eff);
        if let Some(stage) = bottleneck {
            summary.push_str(&format!("Bottleneck: {}\n", stage));
        }
        summary.push_str("Stage Rankings (worst → best):\n");
        for (stage, eff) in &rankings {
            summary.push_str(&format!("  {}: {:.1}%\n", stage, eff));
        }
        summary
    }
}

impl Default for ProcessAnalyzer {
    fn default() -> Self {
        Self::new()
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    fn make_analyzer_with_data() -> ProcessAnalyzer {
        let mut analyzer = ProcessAnalyzer::new();
        for &stage in ProcessingStage::all_ordered() {
            let bm = StageBenchmark::for_stage(stage);
            for i in 0..20 {
                let noise = rand::random::<f64>() * 0.1;
                analyzer.record_throughput(stage, bm.optimal_throughput * (0.8 + noise));
                analyzer.record_energy(stage, bm.energy_per_ton_range.0 * (1.0 + noise));
            }
        }
        analyzer
    }

    #[test]
    fn test_new_analyzer() {
        let analyzer = ProcessAnalyzer::new();
        assert!(analyzer.benchmark(ProcessingStage::Crushing).is_some());
    }

    #[test]
    fn test_stage_benchmark() {
        let bm = StageBenchmark::for_stage(ProcessingStage::Grinding);
        assert!(bm.optimal_throughput > 0.0);
        assert!(bm.energy_per_ton_range.0 > 0.0);
    }

    #[test]
    fn test_compute_efficiency_good() {
        let analyzer = ProcessAnalyzer::new();
        let report = analyzer.compute_efficiency(ProcessingStage::Crushing, 1500.0, 0.5);
        assert!(report.throughput_score > 80.0);
        assert!(report.overall_efficiency > 70.0);
        assert!(report.bottlenecks.is_empty());
    }

    #[test]
    fn test_compute_efficiency_poor() {
        let analyzer = ProcessAnalyzer::new();
        let report = analyzer.compute_efficiency(ProcessingStage::Crushing, 100.0, 50.0);
        assert!(report.throughput_score < 50.0);
        assert!(!report.bottlenecks.is_empty());
    }

    #[test]
    fn test_average_throughput() {
        let analyzer = make_analyzer_with_data();
        let avg = analyzer.average_throughput(ProcessingStage::Crushing);
        assert!(avg > 0.0);
    }

    #[test]
    fn test_throughput_trend() {
        let mut analyzer = ProcessAnalyzer::new();
        // Increasing trend
        for i in 0..20 {
            analyzer.record_throughput(ProcessingStage::Grinding, 100.0 + i as f64 * 10.0);
        }
        let trend = analyzer.throughput_trend(ProcessingStage::Grinding);
        assert!(trend > 0.0);
    }

    #[test]
    fn test_throughput_trend_insufficient() {
        let analyzer = ProcessAnalyzer::new();
        assert_eq!(analyzer.throughput_trend(ProcessingStage::Crushing), 0.0);
    }

    #[test]
    fn test_throughput_in_range() {
        let analyzer = ProcessAnalyzer::new();
        assert!(analyzer.throughput_in_range(ProcessingStage::Crushing, 1000.0));
        assert!(!analyzer.throughput_in_range(ProcessingStage::Crushing, 5000.0));
    }

    #[test]
    fn test_average_energy() {
        let analyzer = make_analyzer_with_data();
        let avg = analyzer.average_energy(ProcessingStage::Smelting);
        assert!(avg > 0.0);
    }

    #[test]
    fn test_energy_trend() {
        let mut analyzer = ProcessAnalyzer::new();
        for i in 0..20 {
            analyzer.record_energy(ProcessingStage::Grinding, 10.0 - i as f64 * 0.3);
        }
        let trend = analyzer.energy_trend(ProcessingStage::Grinding);
        assert!(trend < 0.0);
    }

    #[test]
    fn test_energy_trend_insufficient() {
        let analyzer = ProcessAnalyzer::new();
        assert_eq!(analyzer.energy_trend(ProcessingStage::Crushing), 0.0);
    }

    #[test]
    fn test_total_energy_consumption() {
        let analyzer = make_analyzer_with_data();
        assert!(analyzer.total_energy_consumption() > 0.0);
    }

    #[test]
    fn test_find_bottleneck() {
        let analyzer = ProcessAnalyzer::new();
        let mut throughputs = HashMap::new();
        let mut energies = HashMap::new();
        for &stage in ProcessingStage::all_ordered() {
            let bm = StageBenchmark::for_stage(stage);
            throughputs.insert(stage, bm.optimal_throughput);
            energies.insert(stage, bm.energy_per_ton_range.0);
        }
        // Make crushing bad
        throughputs.insert(ProcessingStage::Crushing, 50.0);
        energies.insert(ProcessingStage::Crushing, 100.0);

        let bottleneck = analyzer.find_bottleneck(&throughputs, &energies);
        assert_eq!(bottleneck, Some(ProcessingStage::Crushing));
    }

    #[test]
    fn test_find_bottleneck_empty() {
        let analyzer = ProcessAnalyzer::new();
        assert!(analyzer.find_bottleneck(&HashMap::new(), &HashMap::new()).is_some());
    }

    #[test]
    fn test_rank_stages() {
        let analyzer = ProcessAnalyzer::new();
        let mut tps = HashMap::new();
        let mut ens = HashMap::new();
        for &stage in ProcessingStage::all_ordered() {
            tps.insert(stage, 100.0);
            ens.insert(stage, 10.0);
        }
        let ranked = analyzer.rank_stages_by_efficiency(&tps, &ens);
        assert_eq!(ranked.len(), 8);
        // Worst should have lowest efficiency
        assert!(ranked[0].1 <= ranked[1].1);
    }

    #[test]
    fn test_cross_stage_correlation() {
        let mut analyzer = ProcessAnalyzer::new();
        for i in 0..50 {
            let v = 100.0 * (i as f64 * 0.1).sin();
            analyzer.record_throughput(ProcessingStage::Crushing, v);
            analyzer.record_throughput(ProcessingStage::Grinding, v * 0.5);
        }
        let corr = analyzer.cross_stage_correlation(ProcessingStage::Crushing, ProcessingStage::Grinding);
        assert!(corr.abs() > 0.5);
    }

    #[test]
    fn test_cross_stage_correlation_no_data() {
        let analyzer = ProcessAnalyzer::new();
        assert_eq!(
            analyzer.cross_stage_correlation(ProcessingStage::Crushing, ProcessingStage::Grinding),
            0.0
        );
    }

    #[test]
    fn test_pipeline_efficiency() {
        let analyzer = ProcessAnalyzer::new();
        let mut tps = HashMap::new();
        let mut ens = HashMap::new();
        for &stage in ProcessingStage::all_ordered() {
            let bm = StageBenchmark::for_stage(stage);
            tps.insert(stage, bm.optimal_throughput);
            ens.insert(stage, bm.energy_per_ton_range.0);
        }
        let eff = analyzer.pipeline_efficiency(&tps, &ens);
        assert!(eff > 50.0);
    }

    #[test]
    fn test_analyze_signals() {
        let analyzer = ProcessAnalyzer::new();
        let signals = vec![
            ProcessSignal::new(ProcessingStage::Crushing, "throughput", 1200.0, Utc::now(), 0.9),
            ProcessSignal::new(ProcessingStage::Crushing, "throughput", 1300.0, Utc::now(), 0.85),
            ProcessSignal::new(ProcessingStage::Grinding, "throughput", 300.0, Utc::now(), 0.8),
        ];
        let values = analyzer.analyze_signals(&signals);
        assert_eq!(values.get(&ProcessingStage::Crushing).unwrap().len(), 2);
        assert_eq!(values.get(&ProcessingStage::Grinding).unwrap().len(), 1);
    }

    #[test]
    fn test_health_summary() {
        let analyzer = ProcessAnalyzer::new();
        let mut tps = HashMap::new();
        let mut ens = HashMap::new();
        for &stage in ProcessingStage::all_ordered() {
            tps.insert(stage, 500.0);
            ens.insert(stage, 5.0);
        }
        let summary = analyzer.health_summary(&tps, &ens);
        assert!(summary.contains("Pipeline Efficiency"));
        assert!(summary.contains("Bottleneck"));
    }
}
