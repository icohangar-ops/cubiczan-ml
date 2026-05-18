//! Commodity price fetching and mock generation.

use serde::{Deserialize, Serialize};
use crate::config::MINERALS;

/// Price data for a single mineral.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MineralPrice {
    pub current_price: f64,
    pub forecast_price: f64,
    pub timestamp: String,
}

/// Generate realistic mock commodity prices for demonstration.
pub fn generate_mock_prices() -> Vec<(&'static str, MineralPrice)> {
    let base_prices: [(&str, f64); 3] = [
        ("LITHIUM", 14_250.0),
        ("NICKEL", 17_200.0),
        ("COBALT", 33_500.0),
    ];

    let now = chrono::Utc::now().to_rfc3339();

    base_prices.iter().map(|&(mineral, base)| {
        let config = &MINERALS.iter().find(|(s, _)| *s == mineral).unwrap().1;
        let (low, high) = config.typical_price_range;
        let variation = base * 0.15;

        // Deterministic-ish random using time
        let seed = chrono::Utc::now().timestamp() as u64 % 10_000;
        let rv = pseudo_random(seed, mineral);
        let current = (base + (rv - 0.5) * 2.0 * variation)
            .clamp(low * 0.8, high * 1.2);

        let rv2 = pseudo_random(seed + 1, mineral);
        let forecast_change = -0.05 + rv2 * 0.20; // slight bullish bias
        let forecast = current * (1.0 + forecast_change);

        (mineral, MineralPrice {
            current_price: (current * 100.0).round() / 100.0,
            forecast_price: (forecast * 100.0).round() / 100.0,
            timestamp: now.clone(),
        })
    }).collect()
}

/// Simple pseudo-random number 0..1 from seed + mineral name.
fn pseudo_random(seed: u64, mineral: &str) -> f64 {
    let mut s = seed.wrapping_mul(mineral.len() as u64 + 7).wrapping_add(0x9e3779b9);
    s = s.wrapping_mul(1_103_515_245).wrapping_add(12_345);
    (s >> 16) as f64 / 65_536.0
}

/// Fetch live commodity prices from Alpha Vantage API.
#[cfg(feature = "live")]
pub async fn fetch_alpha_vantage_prices(
    api_key: &str,
    client: &reqwest::Client,
) -> anyhow::Result<Vec<(&'static str, MineralPrice)>> {
    let mut results = Vec::new();
    let now = chrono::Utc::now().to_rfc3339();

    for &(mineral, config) in MINERALS {
        let url = format!(
            "https://www.alphavantage.co/query?function=TIME_SERIES_MONTHLY&symbol={}&apikey={}",
            config.alpha_vantage_symbol, api_key
        );

        let resp: serde_json::Value = client.get(&url).send().await?.json().await?;

        if let Some(ts) = resp.get("Monthly Time Series").and_then(|v| v.as_object()) {
            let mut dates: Vec<_> = ts.keys().collect();
            dates.sort_by(|a, b| b.cmp(a));

            if let Some(current_str) = dates.first().and_then(|d| ts[*d].get("4. close")).and_then(|v| v.as_str()) {
                let current: f64 = current_str.parse()?;

                let forecast = if dates.len() > 12 {
                    let year_ago_str = dates.get(12).and_then(|d| ts[*d].get("4. close")).and_then(|v| v.as_str());
                    if let Some(ya) = year_ago_str {
                        let year_ago: f64 = ya.parse()?;
                        current * (1.0 + (current - year_ago) / year_ago)
                    } else {
                        current * 1.05
                    }
                } else {
                    current * 1.05
                };

                results.push((mineral, MineralPrice {
                    current_price: (current * 100.0).round() / 100.0,
                    forecast_price: (forecast * 100.0).round() / 100.0,
                    timestamp: now.clone(),
                }));
            }
        }
    }

    if results.is_empty() {
        anyhow::bail!("No price data returned from Alpha Vantage");
    }
    Ok(results)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_mock_prices() {
        let prices = generate_mock_prices();
        assert_eq!(prices.len(), 3);
        for (mineral, data) in &prices {
            assert!(data.current_price > 0.0, "{} price should be positive", mineral);
            assert!(data.forecast_price > 0.0, "{} forecast should be positive", mineral);
        }
    }

    #[test]
    fn test_pseudo_random_deterministic() {
        let a = pseudo_random(42, "LITHIUM");
        let b = pseudo_random(42, "LITHIUM");
        assert_eq!(a, b, "Should be deterministic for same seed");
    }
}
