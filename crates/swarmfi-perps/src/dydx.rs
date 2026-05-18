//! dYdX v4 Indexer API client.
//!
//! Public, read-only HTTP client for the dYdX v4 Indexer API.
//! No authentication required. Base URL: https://indexer.v4prod.dydx.exchange

use crate::types::*;
use serde::{Deserialize, Serialize};

const BASE_URL: &str = "https://indexer.v4prod.dydx.exchange";

// ── API Response Types ───────────────────────────────────────────

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PerpetualMarketsResponse {
    pub markets: std::collections::HashMap<String, PerpetualMarket>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct PerpetualMarket {
    pub ticker: String,
    pub oracle_price: String,
    pub open_interest: String,
    pub volume_24_h: String,
    pub next_funding_time: String,
    #[serde(default)]
    pub incremental_liquidation_fund: String,
    #[serde(default)]
    pub market_liquidation_fund: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OrderbookResponse {
    pub orderbook: OrderbookData,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OrderbookData {
    pub bids: Vec<OrderbookLevel>,
    pub asks: Vec<OrderbookLevel>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TradesResponse {
    pub trades: Vec<TradeData>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct TradeData {
    pub id: String,
    pub side: String,
    pub size: f64,
    pub price: f64,
    pub created_at: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CandlesResponse {
    pub candles: Vec<CandleData>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct CandleData {
    pub started_at: String,
    pub open: f64,
    pub high: f64,
    pub low: f64,
    pub close: f64,
    pub base_token_volume: f64,
    pub usd_volume: f64,
    pub trades: u32,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HistoricalFundingResponse {
    pub historical_funding: Vec<HistoricalFundingData>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct HistoricalFundingData {
    pub market: String,
    pub effective_at: String,
    pub rate: String,
    pub price: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HeightResponse {
    pub height: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SparklineResponse {
    pub sparklines: Vec<SparklineEntry>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct SparklineEntry {
    pub started_at: String,
    pub quote_volume: f64,
}

// ── Client ───────────────────────────────────────────────────────

/// Synchronous dYdX Indexer API client.
/// For async usage, wrap calls in `tokio::task::spawn_blocking`.
pub struct DydxClient {
    base_url: String,
    client: reqwest::blocking::Client,
}

impl Default for DydxClient {
    fn default() -> Self {
        Self::new()
    }
}

impl DydxClient {
    /// Create a new client with the default dYdX v4 Indexer URL.
    pub fn new() -> Self {
        Self {
            base_url: BASE_URL.to_string(),
            client: reqwest::blocking::Client::builder()
                .timeout(std::time::Duration::from_secs(10))
                .build()
                .unwrap_or_else(|_| reqwest::blocking::Client::new()),
        }
    }

    /// Create a client with a custom base URL (for testing).
    pub fn with_base_url(base_url: &str) -> Self {
        Self {
            base_url: base_url.to_string(),
            client: reqwest::blocking::Client::new(),
        }
    }

    /// Fetch all perpetual markets.
    pub fn get_perpetual_markets(&self) -> anyhow::Result<PerpetualMarketsResponse> {
        let url = format!("{}/v4/perpetualMarkets", self.base_url);
        let resp: PerpetualMarketsResponse = self.client.get(&url).send()?.json()?;
        Ok(resp)
    }

    /// Fetch the orderbook for a market.
    pub fn get_orderbook(&self, market: &str) -> anyhow::Result<OrderbookResponse> {
        let url = format!("{}/v4/orderbook/{}", self.base_url, market);
        let resp: OrderbookResponse = self.client.get(&url).send()?.json()?;
        Ok(resp)
    }

    /// Fetch recent trades for a market.
    pub fn get_trades(&self, market: &str, limit: u32) -> anyhow::Result<TradesResponse> {
        let url = format!("{}/v4/trades/{}?limit={}", self.base_url, market, limit);
        let resp: TradesResponse = self.client.get(&url).send()?.json()?;
        Ok(resp)
    }

    /// Fetch OHLCV candles for a market.
    pub fn get_candles(&self, market: &str, resolution: &str, limit: u32) -> anyhow::Result<CandlesResponse> {
        let url = format!(
            "{}/v4/candles/{}?resolution={}&limit={}",
            self.base_url, market, resolution, limit
        );
        let resp: CandlesResponse = self.client.get(&url).send()?.json()?;
        Ok(resp)
    }

    /// Fetch historical funding rates for a market.
    pub fn get_historical_funding(&self, market: &str, limit: u32) -> anyhow::Result<HistoricalFundingResponse> {
        let url = format!("{}/v4/historicalFunding/{}?limit={}", self.base_url, market, limit);
        let resp: HistoricalFundingResponse = self.client.get(&url).send()?.json()?;
        Ok(resp)
    }

    /// Fetch sparklines for a market.
    pub fn get_sparklines(&self, market: &str) -> anyhow::Result<SparklineResponse> {
        let url = format!("{}/v4/sparklines/{}", self.base_url, market);
        let resp: SparklineResponse = self.client.get(&url).send()?.json()?;
        Ok(resp)
    }

    /// Fetch the current block height.
    pub fn get_height(&self) -> anyhow::Result<HeightResponse> {
        let url = format!("{}/v4/height", self.base_url);
        let resp: HeightResponse = self.client.get(&url).send()?.json()?;
        Ok(resp)
    }
}

// ── Conversions ──────────────────────────────────────────────────

impl From<&PerpetualMarket> for MarketInfo {
    fn from(m: &PerpetualMarket) -> Self {
        MarketInfo {
            ticker: m.ticker.clone(),
            oracle_price: m.oracle_price.clone(),
            open_interest: m.open_interest.clone(),
            volume_24h: m.volume_24_h.clone(),
            next_funding_time: m.next_funding_time.clone(),
        }
    }
}

impl From<&OrderbookData> for Orderbook {
    fn from(ob: &OrderbookData) -> Self {
        Orderbook {
            bids: ob.bids.iter().map(|b| OrderbookLevel { price: b.price, size: b.size }).collect(),
            asks: ob.asks.iter().map(|a| OrderbookLevel { price: a.price, size: a.size }).collect(),
        }
    }
}

impl From<&TradeData> for Trade {
    fn from(t: &TradeData) -> Self {
        Trade {
            side: if t.side == "BUY" { TradeSide::Buy } else { TradeSide::Sell },
            size: t.size,
            price: t.price,
            created_at: t.created_at,
        }
    }
}

impl From<&CandleData> for Candle {
    fn from(c: &CandleData) -> Self {
        Candle {
            started_at: c.started_at.clone(),
            open: c.open,
            high: c.high,
            low: c.low,
            close: c.close,
            base_token_volume: c.base_token_volume,
            usd_volume: c.usd_volume,
            trades: c.trades,
        }
    }
}

impl From<&HistoricalFundingData> for FundingEntry {
    fn from(f: &HistoricalFundingData) -> Self {
        FundingEntry {
            rate: f.rate.clone(),
            effective_at: f.effective_at.clone(),
            price: f.price.clone(),
        }
    }
}

/// Build a `MarketDataBundle` by fetching live data from dYdX.
/// Individual fetch failures are handled gracefully (partial data is OK).
pub fn build_live_market_data(client: &DydxClient, market: &str) -> MarketDataBundle {
    // Fetch all data sources, handling failures individually
    let markets_result = client.get_perpetual_markets().ok();
    let orderbook_result = client.get_orderbook(market).ok();
    let trades_result = client.get_trades(market, 100).ok();
    let candles_result = client.get_candles(market, "1HOURS", 50).ok();
    let funding_result = client.get_historical_funding(market, 20).ok();

    // Extract orderbook
    let orderbook = orderbook_result.map(|ob| Orderbook::from(&ob.orderbook));

    // Extract trades
    let trades: Vec<Trade> = trades_result
        .map(|t| t.trades.iter().map(Trade::from).collect())
        .unwrap_or_default();

    // Extract candles
    let candles: Vec<Candle> = candles_result
        .map(|c| c.candles.iter().map(Candle::from).collect())
        .unwrap_or_default();

    // Extract funding
    let funding: Vec<FundingEntry> = funding_result
        .map(|f| f.historical_funding.iter().map(FundingEntry::from).collect())
        .unwrap_or_default();

    // Extract market info
    let market_info = markets_result
        .and_then(|m| m.markets.get(market).map(MarketInfo::from));

    // Compute stats
    let mid_price = if let Some(ref ob) = orderbook {
        if !ob.bids.is_empty() && !ob.asks.is_empty() {
            (ob.bids[0].price + ob.asks[0].price) / 2.0
        } else {
            0.0
        }
    } else {
        0.0
    };

    let mid_price = if mid_price > 0.0 {
        mid_price
    } else if let Some(ref mi) = market_info {
        crate::math::parse_f64_or_zero(&mi.oracle_price)
    } else if let Some(last_candle) = candles.last() {
        last_candle.close
    } else {
        0.0
    };

    let spread = if let Some(ref ob) = orderbook {
        if !ob.bids.is_empty() && !ob.asks.is_empty() {
            ob.asks[0].price - ob.bids[0].price
        } else {
            0.0
        }
    } else {
        0.0
    };

    let volume_24h = market_info
        .as_ref()
        .map(|m| crate::math::parse_f64_or_zero(&m.volume_24h))
        .unwrap_or(0.0);

    let open_interest = market_info
        .as_ref()
        .map(|m| crate::math::parse_f64_or_zero(&m.open_interest))
        .unwrap_or(0.0);

    let mut funding_rate_1h = 0.0;
    if let Some(first_funding) = funding.first() {
        funding_rate_1h = crate::math::parse_f64_or_zero(&first_funding.rate) * 24.0 * 365.0 * 100.0;
    }

    MarketDataBundle {
        orderbook,
        trades,
        candles,
        funding,
        market: market_info,
        stats: MarketStats {
            mid_price,
            spread,
            volume_24h,
            open_interest,
            funding_rate_1h,
        },
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_client_new() {
        let client = DydxClient::new();
        assert_eq!(client.base_url, BASE_URL);
    }

    #[test]
    fn test_client_custom_url() {
        let client = DydxClient::with_base_url("http://localhost:8080");
        assert_eq!(client.base_url, "http://localhost:8080");
    }

    #[test]
    fn test_perpetual_market_to_market_info() {
        let pm = PerpetualMarket {
            ticker: "BTC-USD".into(),
            oracle_price: "67000".into(),
            open_interest: "500000000".into(),
            volume_24_h: "1000000000".into(),
            next_funding_time: "2025-01-01".into(),
            incremental_liquidation_fund: "1000".into(),
            market_liquidation_fund: "2000".into(),
        };
        let mi = MarketInfo::from(&pm);
        assert_eq!(mi.ticker, "BTC-USD");
        assert_eq!(mi.oracle_price, "67000");
    }

    #[test]
    fn test_trade_conversion() {
        let td = TradeData {
            id: "123".into(),
            side: "BUY".into(),
            size: 0.5,
            price: 67000.0,
            created_at: 1704067200.0,
        };
        let trade = Trade::from(&td);
        assert_eq!(trade.side, TradeSide::Buy);
        assert_eq!(trade.size, 0.5);
    }

    #[test]
    fn test_candle_conversion() {
        let cd = CandleData {
            started_at: "2025-01-01T00:00:00Z".into(),
            open: 65000.0, high: 65500.0, low: 64800.0, close: 65300.0,
            base_token_volume: 100.0, usd_volume: 6530000.0, trades: 500,
        };
        let candle = Candle::from(&cd);
        assert_eq!(candle.close, 65300.0);
    }

    #[test]
    fn test_funding_conversion() {
        let hf = HistoricalFundingData {
            market: "BTC-USD".into(),
            effective_at: "2025-01-01".into(),
            rate: "0.0001".into(),
            price: "67000".into(),
        };
        let entry = FundingEntry::from(&hf);
        assert_eq!(entry.rate, "0.0001");
    }

    #[test]
    fn test_height_response_type() {
        let hr = HeightResponse { height: "12345".into() };
        assert_eq!(hr.height, "12345");
    }
}
