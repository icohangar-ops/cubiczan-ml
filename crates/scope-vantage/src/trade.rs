// scope-vantage/src/trade.rs — UN Comtrade data ingestion

use crate::types::*;
use anyhow::{bail, Result};
use serde_json::Value;
use std::collections::HashMap;

/// Common ISO 3166-1 numeric → country name mapping (subset).
pub fn country_registry() -> HashMap<&'static str, &'static str> {
    let mut m = HashMap::new();
    m.insert("004", "Afghanistan");
    m.insert("008", "Albania");
    m.insert("012", "Algeria");
    m.insert("036", "Australia");
    m.insert("076", "Brazil");
    m.insert("124", "Canada");
    m.insert("156", "China");
    m.insert("250", "France");
    m.insert("276", "Germany");
    m.insert("356", "India");
    m.insert("360", "Indonesia");
    m.insert("380", "Italy");
    m.insert("392", "Japan");
    m.insert("410", "South Korea");
    m.insert("484", "Mexico");
    m.insert("566", "Nigeria");
    m.insert("586", "Pakistan");
    m.insert("643", "Russia");
    m.insert("682", "Saudi Arabia");
    m.insert("710", "South Africa");
    m.insert("724", "Spain");
    m.insert("764", "Thailand");
    m.insert("792", "Turkey");
    m.insert("804", "Ukraine");
    m.insert("826", "United Kingdom");
    m.insert("840", "United States");
    m.insert("704", "Vietnam");
    m.insert("032", "Argentina");
    m.insert("152", "Chile");
    m.insert("170", "Colombia");
    m.insert("818", "Egypt");
    m.insert("364", "Iran");
    m.insert("372", "Ireland");
    m.insert("376", "Israel");
    m.insert("458", "Malaysia");
    m.insert("578", "Norway");
    m.insert("608", "Philippines");
    m.insert("616", "Poland");
    m.insert("642", "Romania");
    m.insert("702", "Singapore");
    m.insert("752", "Sweden");
    m.insert("756", "Switzerland");
    m.insert("682", "Saudi Arabia");
    m.insert("784", "United Arab Emirates");
    m
}

/// Resolve a country code to a `Country` struct using the built-in registry.
pub fn resolve_country(code: &str) -> Country {
    let reg = country_registry();
    let trimmed = code.trim();
    let name = reg.get(trimmed).copied().unwrap_or("Unknown");
    Country::new(trimmed, name)
}

/// Normalize an HS code to exactly 6 digits, zero-padding on the right.
/// Accepts 2, 4, 6, 8, or 10-digit inputs.
pub fn normalize_hs_code(raw: &str) -> Result<CommodityCode> {
    let trimmed = raw.trim();
    let digits: String = trimmed.chars().filter(|c| c.is_ascii_digit()).collect();
    if digits.is_empty() {
        bail!("HS code contains no digits: '{}'", raw);
    }
    if digits.len() > 10 {
        bail!("HS code too long: '{}'", raw);
    }
    // Pad to 6 digits with zeros on the right
    let padded = format!("{:0<6}", &digits[..std::cmp::min(digits.len(), 6)]);
    CommodityCode::new(&padded)
}

/// Parse a single CSV row (as a string slice) into a `TradeRecord`.
/// Expected columns: reporter_code, partner_code, commodity_code, flow, trade_value_usd, net_weight_kg, year
pub fn parse_csv_row(row: &str) -> Result<TradeRecord> {
    let cols: Vec<&str> = row.split(',').collect();
    if cols.len() < 7 {
        bail!("CSV row must have at least 7 columns, got {}", cols.len());
    }
    let reporter = resolve_country(cols[0]);
    let partner = resolve_country(cols[1]);
    let commodity = normalize_hs_code(cols[2])?;
    let flow = TradeFlow::from_str(cols[3])?;
    let trade_value_usd: f64 = cols[4].trim().parse()?;
    let net_weight_kg: f64 = cols[5].trim().parse()?;
    let year: u32 = cols[6].trim().parse()?;

    let record = TradeRecord::new(reporter, partner, commodity, flow, trade_value_usd, net_weight_kg, year);
    record.validate()?;
    Ok(record)
}

/// Parse a batch of CSV rows into `TradeRecord` list, skipping invalid rows.
pub fn parse_csv_batch(rows: &[&str]) -> (Vec<TradeRecord>, Vec<String>) {
    let mut records = Vec::new();
    let mut errors = Vec::new();
    for (i, row) in rows.iter().enumerate() {
        match parse_csv_row(row) {
            Ok(rec) => records.push(rec),
            Err(e) => errors.push(format!("Row {}: {}", i, e)),
        }
    }
    (records, errors)
}

/// Parse a JSON object representing a single trade record.
/// Expected fields: reporter_code, partner_code, commodity_code, flow, trade_value_usd, net_weight_kg, year
pub fn parse_json_record(val: &Value) -> Result<TradeRecord> {
    let reporter_code = val["reporter_code"].as_str().unwrap_or("");
    let partner_code = val["partner_code"].as_str().unwrap_or("");
    let commodity_code = val["commodity_code"].as_str().unwrap_or("");
    let flow_str = val["flow"].as_str().unwrap_or("");

    let trade_value_usd = val["trade_value_usd"].as_f64().unwrap_or(0.0);
    let net_weight_kg = val["net_weight_kg"].as_f64().unwrap_or(0.0);
    let year = val["year"].as_u64().unwrap_or(0) as u32;

    let reporter = resolve_country(reporter_code);
    let partner = resolve_country(partner_code);
    let commodity = normalize_hs_code(commodity_code)?;
    let flow = TradeFlow::from_str(flow_str)?;

    let record = TradeRecord::new(reporter, partner, commodity, flow, trade_value_usd, net_weight_kg, year);
    record.validate()?;
    Ok(record)
}

/// Parse a JSON array of trade records.
pub fn parse_json_batch(json: &str) -> Result<Vec<TradeRecord>> {
    let arr: Vec<Value> = serde_json::from_str(json)?;
    let mut records = Vec::new();
    for val in arr {
        records.push(parse_json_record(&val)?);
    }
    Ok(records)
}

/// Deduplicate trade records by dedup_key, keeping the one with the higher value.
pub fn dedup_records(records: Vec<TradeRecord>) -> Vec<TradeRecord> {
    let mut map: HashMap<String, TradeRecord> = HashMap::new();
    for rec in records {
        let key = rec.dedup_key();
        map.entry(key)
            .and_modify(|existing| {
                if rec.trade_value_usd > existing.trade_value_usd {
                    *existing = rec.clone();
                }
            })
            .or_insert(rec);
    }
    map.into_values().collect()
}

/// Construct bilateral trade flows from a set of records.
/// Returns a list of (source_country, target_country, commodity, total_value).
pub fn bilateral_flows(records: &[TradeRecord]) -> Vec<(String, String, String, f64)> {
    let mut map: HashMap<(String, String, String), f64> = HashMap::new();
    for rec in records {
        let (src, tgt) = match rec.flow {
            TradeFlow::Export => (rec.reporter.code.clone(), rec.partner.code.clone()),
            TradeFlow::Import => (rec.partner.code.clone(), rec.reporter.code.clone()),
            TradeFlow::ReExport => continue,
        };
        *map.entry((src, tgt, rec.commodity.0.clone()))
            .or_insert(0.0) += rec.trade_value_usd;
    }
    map.into_iter()
        .map(|((s, t, c), v)| (s, t, c, v))
        .collect()
}

// ─── Tests ──────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn resolve_known_country() {
        let c = resolve_country("840");
        assert_eq!(c.code, "840");
        assert_eq!(c.name, "United States");
    }

    #[test]
    fn resolve_unknown_country() {
        let c = resolve_country("999");
        assert_eq!(c.name, "Unknown");
    }

    #[test]
    fn country_registry_nonempty() {
        let reg = country_registry();
        assert!(reg.len() > 30);
    }

    #[test]
    fn normalize_hs6() {
        let code = normalize_hs_code("870323").unwrap();
        assert_eq!(code.as_str(), "870323");
    }

    #[test]
    fn normalize_hs4_padded() {
        let code = normalize_hs_code("8703").unwrap();
        assert_eq!(code.as_str(), "870300");
    }

    #[test]
    fn normalize_hs2_padded() {
        let code = normalize_hs_code("87").unwrap();
        assert_eq!(code.as_str(), "870000");
    }

    #[test]
    fn normalize_hs_invalid_chars() {
        assert!(normalize_hs_code("ABCDEFGH").is_err());
    }

    #[test]
    fn normalize_hs_too_long() {
        assert!(normalize_hs_code("123456789012345").is_err());
    }

    #[test]
    fn parse_csv_row_valid() {
        let row = "840,156,870323,import,1500000.0,75000.0,2023";
        let rec = parse_csv_row(row).unwrap();
        assert_eq!(rec.reporter.code, "840");
        assert_eq!(rec.partner.code, "156");
        assert_eq!(rec.flow, TradeFlow::Import);
        assert!((rec.trade_value_usd - 1_500_000.0).abs() < 1e-6);
    }

    #[test]
    fn parse_csv_row_export() {
        let row = "156,840,870323,export,2000000.0,100000.0,2023";
        let rec = parse_csv_row(row).unwrap();
        assert_eq!(rec.flow, TradeFlow::Export);
    }

    #[test]
    fn parse_csv_row_too_few_columns() {
        let row = "840,156,870323";
        assert!(parse_csv_row(row).is_err());
    }

    #[test]
    fn parse_csv_row_invalid_flow() {
        let row = "840,156,870323,unknown,100.0,10.0,2023";
        assert!(parse_csv_row(row).is_err());
    }

    #[test]
    fn parse_csv_batch_mixed() {
        let rows = vec![
            "840,156,870323,import,100.0,10.0,2023",
            "BAD ROW",
            "156,840,270900,export,200.0,20.0,2023",
        ];
        let (records, errors) = parse_csv_batch(&rows);
        assert_eq!(records.len(), 2);
        assert_eq!(errors.len(), 1);
    }

    #[test]
    fn parse_json_record_valid() {
        let json = serde_json::json!({
            "reporter_code": "840",
            "partner_code": "156",
            "commodity_code": "870323",
            "flow": "import",
            "trade_value_usd": 500000.0,
            "net_weight_kg": 25000.0,
            "year": 2023
        });
        let rec = parse_json_record(&json).unwrap();
        assert_eq!(rec.trade_value_usd, 500000.0);
    }

    #[test]
    fn parse_json_batch_valid() {
        let json = r#"[
            {"reporter_code":"840","partner_code":"156","commodity_code":"870323","flow":"import","trade_value_usd":100.0,"net_weight_kg":10.0,"year":2023},
            {"reporter_code":"156","partner_code":"840","commodity_code":"270900","flow":"export","trade_value_usd":200.0,"net_weight_kg":20.0,"year":2023}
        ]"#;
        let records = parse_json_batch(json).unwrap();
        assert_eq!(records.len(), 2);
    }

    #[test]
    fn parse_json_batch_invalid() {
        assert!(parse_json_batch("not json").is_err());
    }

    #[test]
    fn dedup_records_keeps_higher() {
        let r1 = TradeRecord::new(
            Country::new("840", "USA"),
            Country::new("156", "CHN"),
            CommodityCode::new("870323").unwrap(),
            TradeFlow::Import,
            100.0,
            10.0,
            2023,
        );
        let r2 = TradeRecord::new(
            Country::new("840", "USA"),
            Country::new("156", "CHN"),
            CommodityCode::new("870323").unwrap(),
            TradeFlow::Import,
            200.0,
            20.0,
            2023,
        );
        let deduped = dedup_records(vec![r1, r2]);
        assert_eq!(deduped.len(), 1);
        assert!((deduped[0].trade_value_usd - 200.0).abs() < 1e-9);
    }

    #[test]
    fn dedup_records_distinct_keys() {
        let r1 = TradeRecord::new(
            Country::new("840", "USA"),
            Country::new("156", "CHN"),
            CommodityCode::new("870323").unwrap(),
            TradeFlow::Import,
            100.0,
            10.0,
            2023,
        );
        let r2 = TradeRecord::new(
            Country::new("840", "USA"),
            Country::new("276", "DEU"),
            CommodityCode::new("870323").unwrap(),
            TradeFlow::Import,
            200.0,
            20.0,
            2023,
        );
        let deduped = dedup_records(vec![r1, r2]);
        assert_eq!(deduped.len(), 2);
    }

    #[test]
    fn bilateral_flows_aggregation() {
        let records = vec![
            TradeRecord::new(
                Country::new("840", "USA"),
                Country::new("156", "CHN"),
                CommodityCode::new("870323").unwrap(),
                TradeFlow::Export,
                100.0,
                10.0,
                2023,
            ),
            TradeRecord::new(
                Country::new("840", "USA"),
                Country::new("156", "CHN"),
                CommodityCode::new("870323").unwrap(),
                TradeFlow::Export,
                50.0,
                5.0,
                2023,
            ),
        ];
        let flows = bilateral_flows(&records);
        assert_eq!(flows.len(), 1);
        assert!((flows[0].3 - 150.0).abs() < 1e-9);
        assert_eq!(flows[0].0, "840");
        assert_eq!(flows[0].1, "156");
    }

    #[test]
    fn bilateral_flows_import_reversed() {
        let records = vec![
            TradeRecord::new(
                Country::new("840", "USA"),
                Country::new("156", "CHN"),
                CommodityCode::new("870323").unwrap(),
                TradeFlow::Import,
                100.0,
                10.0,
                2023,
            ),
        ];
        let flows = bilateral_flows(&records);
        assert_eq!(flows.len(), 1);
        // Import: source is partner, target is reporter
        assert_eq!(flows[0].0, "156");
        assert_eq!(flows[0].1, "840");
    }

    #[test]
    fn bilateral_flows_skip_reexport() {
        let records = vec![
            TradeRecord::new(
                Country::new("840", "USA"),
                Country::new("156", "CHN"),
                CommodityCode::new("870323").unwrap(),
                TradeFlow::ReExport,
                100.0,
                10.0,
                2023,
            ),
        ];
        let flows = bilateral_flows(&records);
        assert!(flows.is_empty());
    }
}
