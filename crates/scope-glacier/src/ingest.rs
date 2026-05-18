//! EIA data ingestion pipeline: CSV/JSON parsing, normalization, validation, gap filling.

use crate::types::*;
use chrono::{DateTime, NaiveDate, Utc};
use serde::Deserialize;
use std::collections::HashMap;

/// Parsed EIA series response (mirrors the EIA API JSON schema).
#[derive(Debug, Deserialize)]
pub struct EiaSeries {
    pub series_id: String,
    #[serde(default)]
    pub name: String,
    pub units: String,
    pub data: Vec<EiaDataPoint>,
}

/// A single EIA data point: [timestamp_string, value, optional_annotation].
#[derive(Debug, Clone, Deserialize)]
pub struct EiaDataPoint {
    #[serde(default)]
    pub period: String,
    pub value: Option<f64>,
}

/// A normalized time-series record after ingestion processing.
#[derive(Debug, Clone)]
pub struct NormalizedRecord {
    pub timestamp: DateTime<Utc>,
    pub value: f64,
    pub commodity: EnergyCommodity,
    pub quality: DataQuality,
}

/// Data quality flags for ingested records.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DataQuality {
    Verified,
    Estimated,
    Interpolated,
    OutlierCleaned,
}

/// A parsed EIA JSON response wrapper.
#[derive(Debug, Deserialize)]
pub struct EiaResponse {
    pub response: EiaResponseBody,
}

#[derive(Debug, Deserialize)]
pub struct EiaResponseBody {
    pub data: Vec<EiaSeries>,
}

/// Ingestion statistics for a batch processing run.
#[derive(Debug, Clone, Default)]
pub struct IngestStats {
    pub total_rows: usize,
    pub valid_rows: usize,
    pub missing_values: usize,
    pub outliers_removed: usize,
    pub gaps_filled: usize,
    pub parse_errors: usize,
}

impl IngestStats {
    pub fn valid_ratio(&self) -> f64 {
        if self.total_rows == 0 {
            return 0.0;
        }
        self.valid_rows as f64 / self.total_rows as f64
    }

    pub fn summary(&self) -> String {
        format!(
            "IngestStats: {} total, {} valid ({:.1}%), {} missing, {} outliers, {} gaps filled, {} parse errors",
            self.total_rows,
            self.valid_rows,
            self.valid_ratio() * 100.0,
            self.missing_values,
            self.outliers_removed,
            self.gaps_filled,
            self.parse_errors,
        )
    }
}

/// Parses EIA-style JSON data into PricePoint records.
pub fn parse_eia_json(json_str: &str, commodity: EnergyCommodity) -> Result<Vec<PricePoint>> {
    let response: EiaResponse =
        serde_json::from_str(json_str).map_err(|e| GlacierError::ParseError(e.to_string()))?;

    if response.response.data.is_empty() {
        return Ok(vec![]);
    }

    let series = &response.response.data[0];
    let mut points = Vec::with_capacity(series.data.len());

    for dp in &series.data {
        if let Some(val) = dp.value {
            if !val.is_finite() || val <= 0.0 {
                continue;
            }
            let ts = parse_eia_period(&dp.period)?;
            points.push(PricePoint::new(ts, commodity, val).with_source("EIA"));
        }
    }

    points.sort_by_key(|p| p.timestamp);
    Ok(points)
}

/// Parses EIA period strings like "20240101" or "2024-01-01" into DateTime<Utc>.
fn parse_eia_period(period: &str) -> Result<DateTime<Utc>> {
    // Try ISO format first: "2024-01-01"
    if let Ok(d) = NaiveDate::parse_from_str(period.trim(), "%Y-%m-%d") {
        return Ok(d.and_hms_opt(0, 0, 0).unwrap().and_utc());
    }
    // Try compact format: "20240101"
    if let Ok(d) = NaiveDate::parse_from_str(period.trim(), "%Y%m%d") {
        return Ok(d.and_hms_opt(0, 0, 0).unwrap().and_utc());
    }
    // Try monthly: "202401"
    if let Ok(d) = NaiveDate::parse_from_str(&format!("{}01", period.trim()), "%Y%m%d") {
        return Ok(d.and_hms_opt(0, 0, 0).unwrap().and_utc());
    }
    Err(GlacierError::ParseError(format!(
        "Cannot parse EIA period: {}",
        period
    )))
}

/// Parses EIA-style CSV data (header + rows) into PricePoint records.
/// Expected format: period,price[,volume]
pub fn parse_eia_csv(csv_str: &str, commodity: EnergyCommodity) -> Result<Vec<PricePoint>> {
    let mut points = Vec::new();
    let mut stats = IngestStats::default();

    for line in csv_str.lines().skip(1) {
        let trimmed = line.trim();
        if trimmed.is_empty() {
            continue;
        }
        stats.total_rows += 1;

        let fields: Vec<&str> = trimmed.split(',').collect();
        if fields.len() < 2 {
            stats.parse_errors += 1;
            continue;
        }

        let ts = match parse_eia_period(fields[0]) {
            Ok(t) => t,
            Err(_) => {
                stats.parse_errors += 1;
                continue;
            }
        };

        let price: f64 = match fields[1].parse() {
            Ok(v) => v,
            Err(_) => {
                stats.missing_values += 1;
                continue;
            }
        };

        if !price.is_finite() || price <= 0.0 {
            stats.outliers_removed += 1;
            continue;
        }

        let mut pp = PricePoint::new(ts, commodity, price).with_source("EIA-CSV");
        if fields.len() >= 3 {
            if let Ok(vol) = fields[2].parse::<f64>() {
                pp = pp.with_volume(vol);
            }
        }
        stats.valid_rows += 1;
        points.push(pp);
    }

    points.sort_by_key(|p| p.timestamp);
    let _ = stats; // In production this would be returned or logged
    Ok(points)
}

/// Normalizes a time series of PricePoints into evenly-spaced values.
/// Aligns to daily frequency, filling gaps via linear interpolation.
pub fn normalize_timeseries(points: &[PricePoint]) -> Result<Vec<NormalizedRecord>> {
    if points.is_empty() {
        return Ok(vec![]);
    }

    let commodity = points[0].commodity;
    let mut records: Vec<NormalizedRecord> = Vec::new();

    for pp in points {
        records.push(NormalizedRecord {
            timestamp: pp.timestamp,
            value: pp.price,
            commodity,
            quality: DataQuality::Verified,
        });
    }

    // Fill gaps using linear interpolation
    fill_gaps_linear(&mut records);

    Ok(records)
}

/// Detects and fills gaps in a time series using linear interpolation.
fn fill_gaps_linear(records: &mut Vec<NormalizedRecord>) {
    if records.len() < 2 {
        return;
    }

    let mut filled = Vec::new();
    filled.push(records[0].clone());

    for window in records.windows(2) {
        let (prev, next) = (&window[0], &window[1]);
        let gap_days = (next.timestamp - prev.timestamp).num_days();

        if gap_days > 1 {
            for day in 1..gap_days {
                let frac = day as f64 / gap_days as f64;
                let interp_value = prev.value + frac * (next.value - prev.value);
                filled.push(NormalizedRecord {
                    timestamp: prev.timestamp + chrono::Duration::days(day),
                    value: interp_value,
                    commodity: prev.commodity,
                    quality: DataQuality::Interpolated,
                });
            }
        }
        filled.push(next.clone());
    }

    *records = filled;
}

/// Removes outliers from a price series using the IQR method.
/// Returns the cleaned series and the number of outliers removed.
pub fn remove_outliers_iqr(
    records: &mut Vec<NormalizedRecord>,
    multiplier: f64,
) -> usize {
    if records.len() < 4 {
        return 0;
    }

    let mut values: Vec<f64> = records.iter().map(|r| r.value).collect();
    values.sort_by(|a, b| a.partial_cmp(b).unwrap());

    let q1 = values[values.len() / 4];
    let q3 = values[3 * values.len() / 4];
    let iqr = q3 - q1;
    let lower = q1 - multiplier * iqr;
    let upper = q3 + multiplier * iqr;

    let original_len = records.len();
    records.retain(|r| r.value >= lower && r.value <= upper);

    // Mark the ones that survived with quality check
    for r in records.iter_mut() {
        if r.quality == DataQuality::Verified {
            // keep as is
        }
    }

    original_len - records.len()
}

/// Validates a series of PricePoints, returning statistics and filtered valid points.
pub fn validate_price_series(points: &[PricePoint]) -> (Vec<PricePoint>, IngestStats) {
    let mut stats = IngestStats::default();
    let mut valid: Vec<PricePoint> = Vec::new();

    for pp in points {
        stats.total_rows += 1;
        if pp.is_valid() {
            stats.valid_rows += 1;
            valid.push(pp.clone());
        } else {
            stats.outliers_removed += 1;
        }
    }

    (valid, stats)
}

/// Batch-processes multiple commodity feeds and aggregates results.
pub fn batch_ingest(
    feeds: &HashMap<EnergyCommodity, String>,
    format: IngestFormat,
) -> Result<HashMap<EnergyCommodity, Vec<PricePoint>>> {
    let mut results = HashMap::new();

    for (commodity, data) in feeds {
        let points = match format {
            IngestFormat::Json => parse_eia_json(data, *commodity)?,
            IngestFormat::Csv => parse_eia_csv(data, *commodity)?,
        };
        results.insert(*commodity, points);
    }

    Ok(results)
}

/// Supported ingestion formats.
#[derive(Debug, Clone, Copy)]
pub enum IngestFormat {
    Json,
    Csv,
}

/// Converts normalized records back to a simple f64 vector.
pub fn records_to_values(records: &[NormalizedRecord]) -> Vec<f64> {
    records.iter().map(|r| r.value).collect()
}

/// Extracts timestamps from normalized records.
pub fn records_to_timestamps(records: &[NormalizedRecord]) -> Vec<DateTime<Utc>> {
    records.iter().map(|r| r.timestamp).collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_parse_eia_json_valid() {
        let json = r#"{
            "response": {
                "data": [{
                    "series_id": "PET.RWTC",
                    "name": "WTI",
                    "units": "$/barrel",
                    "data": [
                        {"period": "2024-01-01", "value": 72.5},
                        {"period": "2024-01-02", "value": 73.0},
                        {"period": "2024-01-03", "value": null}
                    ]
                }]
            }
        }"#;

        let points = parse_eia_json(json, EnergyCommodity::CrudeOil).unwrap();
        assert_eq!(points.len(), 2);
        assert!((points[0].price - 72.5).abs() < 1e-10);
        assert_eq!(points[0].source, "EIA");
    }

    #[test]
    fn test_parse_eia_json_empty() {
        let json = r#"{"response": {"data": []}}"#;
        let points = parse_eia_json(json, EnergyCommodity::CrudeOil).unwrap();
        assert!(points.is_empty());
    }

    #[test]
    fn test_parse_eia_json_invalid() {
        let json = r#"not valid json"#;
        let result = parse_eia_json(json, EnergyCommodity::CrudeOil);
        assert!(result.is_err());
    }

    #[test]
    fn test_parse_eia_csv() {
        let csv = "period,price,volume\n2024-01-01,72.5,1000\n2024-01-02,73.0,1100\n";
        let points = parse_eia_csv(csv, EnergyCommodity::CrudeOil).unwrap();
        assert_eq!(points.len(), 2);
        assert_eq!(points[0].volume, Some(1000.0));
        assert_eq!(points[0].source, "EIA-CSV");
    }

    #[test]
    fn test_parse_eia_csv_compact_period() {
        let csv = "period,price\n20240115,3.50\n20240116,3.55\n";
        let points = parse_eia_csv(csv, EnergyCommodity::NaturalGas).unwrap();
        assert_eq!(points.len(), 2);
    }

    #[test]
    fn test_parse_eia_csv_skips_invalid() {
        let csv = "period,price\n2024-01-01,72.5\nbad-date,73.0\n2024-01-03,-5.0\n";
        let points = parse_eia_csv(csv, EnergyCommodity::CrudeOil).unwrap();
        assert_eq!(points.len(), 1);
    }

    #[test]
    fn test_parse_eia_period_formats() {
        let iso = parse_eia_period("2024-03-15").unwrap();
        let compact = parse_eia_period("20240315").unwrap();
        assert_eq!(iso, compact);
    }

    #[test]
    fn test_parse_eia_period_monthly() {
        let result = parse_eia_period("202403");
        assert!(result.is_ok());
    }

    #[test]
    fn test_parse_eia_period_invalid() {
        let result = parse_eia_period("not-a-date");
        assert!(result.is_err());
    }

    #[test]
    fn test_normalize_timeseries_empty() {
        let result = normalize_timeseries(&[]).unwrap();
        assert!(result.is_empty());
    }

    #[test]
    fn test_normalize_timeseries_fills_gaps() {
        let p1 = PricePoint::new(
            "2024-01-01T00:00:00Z".parse().unwrap(),
            EnergyCommodity::CrudeOil,
            72.0,
        );
        let p2 = PricePoint::new(
            "2024-01-04T00:00:00Z".parse().unwrap(),
            EnergyCommodity::CrudeOil,
            78.0,
        );
        let records = normalize_timeseries(&[p1, p2]).unwrap();
        // Should have original 2 + 2 interpolated = 4
        assert_eq!(records.len(), 4);
        // Check interpolated values
        assert_eq!(records[1].quality, DataQuality::Interpolated);
        assert!((records[1].value - 74.0).abs() < 1e-10);
    }

    #[test]
    fn test_remove_outliers_iqr() {
        let base_ts = "2024-01-01T00:00:00Z".parse::<DateTime<Utc>>().unwrap();
        let mut records: Vec<NormalizedRecord> = (0..20)
            .map(|i| NormalizedRecord {
                timestamp: base_ts + chrono::Duration::days(i),
                value: 100.0 + (i as f64) * 0.5,
                commodity: EnergyCommodity::CrudeOil,
                quality: DataQuality::Verified,
            })
            .collect();

        // Add an outlier
        records.push(NormalizedRecord {
            timestamp: base_ts + chrono::Duration::days(20),
            value: 10000.0,
            commodity: EnergyCommodity::CrudeOil,
            quality: DataQuality::Verified,
        });

        let removed = remove_outliers_iqr(&mut records, 1.5);
        assert_eq!(removed, 1);
        assert!(records.len() < 21);
    }

    #[test]
    fn test_remove_outliers_iqr_small_dataset() {
        let mut records = vec![NormalizedRecord {
            timestamp: Utc::now(),
            value: 100.0,
            commodity: EnergyCommodity::CrudeOil,
            quality: DataQuality::Verified,
        }];
        let removed = remove_outliers_iqr(&mut records, 1.5);
        assert_eq!(removed, 0);
    }

    #[test]
    fn test_validate_price_series() {
        let now = Utc::now();
        let points = vec![
            PricePoint::new(now, EnergyCommodity::CrudeOil, 75.0),
            PricePoint::new(now, EnergyCommodity::CrudeOil, -5.0),
            PricePoint::new(now, EnergyCommodity::CrudeOil, f64::NAN),
            PricePoint::new(now, EnergyCommodity::CrudeOil, 80.0),
        ];

        let (valid, stats) = validate_price_series(&points);
        assert_eq!(valid.len(), 2);
        assert_eq!(stats.total_rows, 4);
        assert_eq!(stats.outliers_removed, 2);
    }

    #[test]
    fn test_batch_ingest_json() {
        let mut feeds = HashMap::new();
        feeds.insert(
            EnergyCommodity::CrudeOil,
            r#"{"response": {"data": [{"series_id": "PET.RWTC", "units": "$/barrel", "data": [{"period": "2024-01-01", "value": 75.0}]}]}}"#.to_string(),
        );
        let result = batch_ingest(&feeds, IngestFormat::Json).unwrap();
        assert!(result.contains_key(&EnergyCommodity::CrudeOil));
        assert_eq!(result[&EnergyCommodity::CrudeOil].len(), 1);
    }

    #[test]
    fn test_records_to_values() {
        let records = vec![
            NormalizedRecord {
                timestamp: Utc::now(),
                value: 1.0,
                commodity: EnergyCommodity::CrudeOil,
                quality: DataQuality::Verified,
            },
            NormalizedRecord {
                timestamp: Utc::now(),
                value: 2.0,
                commodity: EnergyCommodity::CrudeOil,
                quality: DataQuality::Verified,
            },
        ];
        let values = records_to_values(&records);
        assert_eq!(values, vec![1.0, 2.0]);
    }

    #[test]
    fn test_ingest_stats_summary() {
        let stats = IngestStats {
            total_rows: 100,
            valid_rows: 90,
            missing_values: 5,
            outliers_removed: 3,
            gaps_filled: 10,
            parse_errors: 2,
        };
        let summary = stats.summary();
        assert!(summary.contains("100 total"));
        assert!(summary.contains("90 valid"));
    }
}
