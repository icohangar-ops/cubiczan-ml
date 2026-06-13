//! Macroeconomic data fetching and mock generation.

use serde::{Deserialize, Serialize};
#[cfg(feature = "live")]
use resilient_call::{retry, with_timeout, RetryPolicy};

/// Macro economic indicators.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MacroData {
    pub ppi_metals: f64,
    pub ppi_metals_change_1y: f64,
    pub industrial_production: f64,
    pub industrial_production_change_1y: f64,
    pub manufacturing_pmi: f64,
    pub usd_index: f64,
    pub timestamp: String,
}

/// Generate mock macroeconomic indicators.
pub fn generate_mock_macro_data() -> MacroData {
    let now = chrono::Utc::now().to_rfc3339();
    MacroData {
        ppi_metals: 200.0,
        ppi_metals_change_1y: 5.0,
        industrial_production: 101.5,
        industrial_production_change_1y: 1.0,
        manufacturing_pmi: 53.0,
        usd_index: 102.5,
        timestamp: now,
    }
}

/// Fetch live macro data from FRED API.
#[cfg(feature = "live")]
pub async fn fetch_fred_macro_data(
    api_key: &str,
    client: &reqwest::Client,
) -> anyhow::Result<MacroData> {
    let mut macro_data = MacroData {
        ppi_metals: 0.0,
        ppi_metals_change_1y: 0.0,
        industrial_production: 0.0,
        industrial_production_change_1y: 0.0,
        manufacturing_pmi: 0.0,
        usd_index: 0.0,
        timestamp: chrono::Utc::now().to_rfc3339(),
    };

    // Fetch PPI for metals (WPU101)
    let ppi_url = format!(
        "https://api.stlouisfed.org/fred/series/observations?series_id=WPU101&api_key={}&file_type=json&sort_order=desc&limit=2",
        api_key
    );
    let resp: serde_json::Value = with_timeout(
        retry(
            || async {
                let r = client.get(&ppi_url).send().await?;
                r.error_for_status()?.json::<serde_json::Value>().await
            },
            &RetryPolicy::with_max_attempts(4),
            crate::prices::is_retryable_reqwest,
        ),
        std::time::Duration::from_secs(20),
    )
    .await
    .map_err(|e| anyhow::anyhow!("FRED PPI fetch failed: {e}"))?;
    if let Some(observations) = resp.get("observations").and_then(|v| v.as_array()) {
        if observations.len() >= 2 {
            let current = observations[0]["value"].as_str().and_then(|v| v.parse::<f64>().ok()).unwrap_or(0.0);
            let previous = observations[1]["value"].as_str().and_then(|v| v.parse::<f64>().ok()).unwrap_or(0.0);
            macro_data.ppi_metals = current;
            macro_data.ppi_metals_change_1y = ((current - previous) / previous * 100.0 * 100.0).round() / 100.0;
        }
    }

    // Fetch Industrial Production (IPMAN)
    let ip_url = format!(
        "https://api.stlouisfed.org/fred/series/observations?series_id=IPMAN&api_key={}&file_type=json&sort_order=desc&limit=2",
        api_key
    );
    let resp: serde_json::Value = with_timeout(
        retry(
            || async {
                let r = client.get(&ip_url).send().await?;
                r.error_for_status()?.json::<serde_json::Value>().await
            },
            &RetryPolicy::with_max_attempts(4),
            crate::prices::is_retryable_reqwest,
        ),
        std::time::Duration::from_secs(20),
    )
    .await
    .map_err(|e| anyhow::anyhow!("FRED industrial production fetch failed: {e}"))?;
    if let Some(observations) = resp.get("observations").and_then(|v| v.as_array()) {
        if observations.len() >= 2 {
            let current = observations[0]["value"].as_str().and_then(|v| v.parse::<f64>().ok()).unwrap_or(0.0);
            let previous = observations[1]["value"].as_str().and_then(|v| v.parse::<f64>().ok()).unwrap_or(0.0);
            macro_data.industrial_production = current;
            macro_data.industrial_production_change_1y = ((current - previous) / previous * 100.0 * 100.0).round() / 100.0;
        }
    }

    Ok(macro_data)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_mock_macro_data() {
        let data = generate_mock_macro_data();
        assert!(data.ppi_metals > 0.0);
        assert!(data.manufacturing_pmi > 0.0);
        assert!(!data.timestamp.is_empty());
    }
}
