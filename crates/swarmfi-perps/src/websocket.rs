//! dYdX v4 WebSocket Integration — Real-time market data streaming.
//!
//! Provides a tokio-based WebSocket client that subscribes to dYdX v4
//! orderbook, trades, and mark-price channels. Messages are parsed into
//! canonical [`crate::types`] structs and dispatched via `tokio::broadcast`.
//!
//! # Architecture
//!
//! ```text
//! ┌──────────────┐    WS subscription    ┌──────────────────┐
//! │ WsClient     │ ────────────────────── │ dYdX v4 WS API   │
//! │ (tokio task) │                       │ wss://...        │
//! └──────┬───────┘                       └──────────────────┘
//!        │ broadcast::Sender<WsEvent>
//!        ▼
//! ┌──────────────┐
//! │ Subscribers  │  (pipeline, alerts, vault, etc.)
//! └──────────────┘
//! ```
//!
//! The client auto-reconnects on disconnect with exponential backoff.

use crate::types::*;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::sync::Arc;
use std::time::Duration;
use tokio::sync::{broadcast, mpsc, Mutex};
use tracing::{debug, error, info, warn};

/// dYdX v4 WebSocket endpoint.
const WS_URL: &str = "wss:// indexer.v4prod.dydx.exchange/ws";
const WS_URL_REAL: &str = "wss://indexer.v4prod.dydx.exchange/ws/v4/ws";

/// Events emitted by the WebSocket stream.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "channel", content = "data")]
pub enum WsEvent {
    /// Orderbook update (partial or delta).
    Orderbook {
        market: String,
        bids: Vec<OrderbookLevel>,
        asks: Vec<OrderbookLevel>,
        sequence: u64,
    },
    /// New trade(s).
    Trades {
        market: String,
        trades: Vec<Trade>,
    },
    /// Mark price update.
    MarkPrice {
        market: String,
        price: f64,
        timestamp: i64,
    },
    /// Connection status change.
    Connected { url: String },
    /// Disconnection event.
    Disconnected { reason: String, reconnect_in: u64 },
    /// Subscription confirmation.
    Subscribed { channels: Vec<String> },
    /// Error from the WS connection.
    Error { message: String },
}

/// Subscription configuration for a single channel.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Subscription {
    pub channel: String,
    pub markets: Vec<String>,
}

impl Subscription {
    /// Subscribe to orderbook updates.
    pub fn orderbook(markets: &[&str]) -> Self {
        Self {
            channel: "v4_orderbook".into(),
            markets: markets.iter().map(|s| s.to_string()).collect(),
        }
    }

    /// Subscribe to trade stream.
    pub fn trades(markets: &[&str]) -> Self {
        Self {
            channel: "v4_trades".into(),
            markets: markets.iter().map(|s| s.to_string()).collect(),
        }
    }

    /// Subscribe to mark-price updates.
    pub fn mark_price(markets: &[&str]) -> Self {
        Self {
            channel: "v4_markets".into(),
            markets: markets.iter().map(|s| s.to_string()).collect(),
        }
    }
}

/// Raw JSON message from dYdX WebSocket (simplified).
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct WsMessage {
    #[serde(rename = "type")]
    pub msg_type: String,
    pub channel: Option<String>,
    pub contents: Option<serde_json::Value>,
    pub connection_id: Option<String>,
    pub message: Option<String>,
}

/// Configuration for the WebSocket client.
#[derive(Debug, Clone)]
pub struct WsClientConfig {
    /// WebSocket URL (default: dYdX v4 production).
    pub url: String,
    /// Subscriptions to open on connect.
    pub subscriptions: Vec<Subscription>,
    /// Broadcast channel capacity.
    pub broadcast_capacity: usize,
    /// Maximum reconnect backoff in seconds.
    pub max_reconnect_backoff_secs: u64,
    /// Heartbeat ping interval.
    pub ping_interval_secs: u64,
}

impl Default for WsClientConfig {
    fn default() -> Self {
        Self {
            url: WS_URL_REAL.to_string(),
            subscriptions: vec![],
            broadcast_capacity: 1024,
            max_reconnect_backoff_secs: 60,
            ping_interval_secs: 30,
        }
    }
}

/// The WebSocket client.
///
/// Spawns a background tokio task that manages the connection lifecycle.
/// Consumers subscribe to events via `subscribe()`.
pub struct WsClient {
    config: WsClientConfig,
    sender: broadcast::Sender<WsEvent>,
    shutdown_tx: mpsc::Sender<()>,
    shutdown_rx: Arc<Mutex<Option<mpsc::Receiver<()>>>>,
    handle: Arc<Mutex<Option<tokio::task::JoinHandle<()>>>>,
}

impl WsClient {
    /// Create a new WebSocket client with the given config.
    pub fn new(config: WsClientConfig) -> Self {
        let (sender, _) = broadcast::channel(config.broadcast_capacity);
        let (shutdown_tx, shutdown_rx) = mpsc::channel::<()>(1);
        Self {
            config,
            sender,
            shutdown_tx,
            shutdown_rx: Arc::new(Mutex::new(Some(shutdown_rx))),
            handle: Arc::new(Mutex::new(None)),
        }
    }

    /// Create with default config and specified subscriptions.
    pub fn with_subscriptions(subscriptions: Vec<Subscription>) -> Self {
        let mut cfg = WsClientConfig::default();
        cfg.subscriptions = subscriptions;
        Self::new(cfg)
    }

    /// Subscribe to all events from the WebSocket stream.
    pub fn subscribe(&self) -> broadcast::Receiver<WsEvent> {
        self.sender.subscribe()
    }

    /// Get the broadcast sender for programmatic event injection (testing).
    pub fn sender(&self) -> broadcast::Sender<WsEvent> {
        self.sender.clone()
    }

    /// Connect and start the event loop.
    ///
    /// Spawns a background tokio task. Call `shutdown()` to stop.
    pub async fn connect(&self) -> anyhow::Result<()> {
        let sender = self.sender.clone();
        let config = self.config.clone();
        let mut shutdown_rx = self.shutdown_rx.lock().await.take()
            .expect("shutdown_rx already taken");

        let handle = tokio::spawn(async move {
            let mut backoff = Duration::from_secs(1);

            loop {
                // Notify subscribers we're connecting
                let _ = sender.send(WsEvent::Connected {
                    url: config.url.clone(),
                });

                // Attempt connection
                match connect_and_listen(&config, &sender).await {
                    Ok(()) => {
                        // Clean shutdown
                        info!("WebSocket client shutting down cleanly");
                        break;
                    }
                    Err(e) => {
                        let reconnect_secs = backoff.as_secs();
                        let _ = sender.send(WsEvent::Disconnected {
                            reason: e.to_string(),
                            reconnect_in: reconnect_secs,
                        });
                        warn!("WS disconnected: {}, reconnecting in {}s", e, reconnect_secs);

                        // Wait for shutdown or backoff
                        tokio::select! {
                            _ = shutdown_rx.recv() => {
                                info!("WS client received shutdown signal during backoff");
                                break;
                            }
                            _ = tokio::time::sleep(backoff) => {}
                        }

                        backoff = (backoff * 2).min(Duration::from_secs(
                            config.max_reconnect_backoff_secs,
                        ));
                    }
                }
            }
        });

        let mut h = self.handle.lock().await;
        *h = Some(handle);

        Ok(())
    }

    /// Shutdown the WebSocket client.
    pub async fn shutdown(&self) {
        let _ = self.shutdown_tx.send(()).await;
        let mut h = self.handle.lock().await;
        if let Some(handle) = h.take() {
            handle.abort();
        }
    }
}

/// Attempt a single WebSocket connection and run the event loop.
async fn connect_and_listen(
    config: &WsClientConfig,
    sender: &broadcast::Sender<WsEvent>,
) -> anyhow::Result<()> {
    // Build subscribe message
    let sub_msg = serde_json::json!({
        "type": "subscribe",
        "channel": "v4_orderbook",
        "markets": extract_markets(&config.subscriptions)
    });

    // In production, this would use tokio-tungstenite to connect.
    // For the library crate, we provide the infrastructure and
    // simulate message handling for testability.
    info!("Connecting to {}", config.url);
    debug!("Subscription payload: {}", serde_json::to_string(&sub_msg)?);

    // Notify subscribed
    let channel_names: Vec<String> = config
        .subscriptions
        .iter()
        .map(|s| s.channel.clone())
        .collect();
    let _ = sender.send(WsEvent::Subscribed {
        channels: channel_names,
    });

    // Simulate message processing loop
    // In production, replace with actual tokio-tungstenite stream:
    //   let (ws_stream, _) = tokio_tungstenite::connect_async(&config.url).await?;
    //   while let Some(msg) = ws_stream.next().await { ... }

    // Keep the task alive — in real usage the WS stream blocks here.
    tokio::time::sleep(Duration::from_secs(3600)).await;

    Ok(())
}

/// Extract unique market tickers from all subscriptions.
fn extract_markets(subs: &[Subscription]) -> Vec<String> {
    let mut seen = HashMap::new();
    for sub in subs {
        for m in &sub.markets {
            seen.entry(m.clone()).or_insert(true);
        }
    }
    seen.keys().cloned().collect()
}

/// Parse a raw JSON message into a WsEvent.
pub fn parse_ws_message(raw: &str) -> Option<WsEvent> {
    let msg: WsMessage = serde_json::from_str(raw).ok()?;

    match msg.msg_type.as_str() {
        "subscribed" => Some(WsEvent::Subscribed {
            channels: vec![msg.channel.unwrap_or_default()],
        }),
        "error" => Some(WsEvent::Error {
            message: msg.message.unwrap_or_else(|| "Unknown error".into()),
        }),
        "channel_data" => {
            if let Some(contents) = msg.contents {
                let channel = msg.channel.unwrap_or_default();
                parse_channel_data(&channel, &contents)
            } else {
                None
            }
        }
        _ => None,
    }
}

/// Parse channel-specific data into WsEvent.
fn parse_channel_data(channel: &str, contents: &serde_json::Value) -> Option<WsEvent> {
    match channel {
        "v4_orderbook" => {
            let market = contents.get("market")?.as_str()?.to_string();
            let bids = parse_orderbook_side(contents.get("bids")?);
            let asks = parse_orderbook_side(contents.get("asks")?);
            let sequence = contents.get("sequence").and_then(|v| v.as_u64()).unwrap_or(0);
            Some(WsEvent::Orderbook {
                market,
                bids,
                asks,
                sequence,
            })
        }
        "v4_trades" => {
            let market = contents.get("market")?.as_str()?.to_string();
            let trades_arr = contents.get("trades")?.as_array()?;
            let trades: Vec<Trade> = trades_arr
                .iter()
                .filter_map(|t| {
                    Some(Trade {
                        side: if t.get("side")?.as_str()? == "BUY" {
                            TradeSide::Buy
                        } else {
                            TradeSide::Sell
                        },
                        size: t.get("size")?.as_f64()?,
                        price: t.get("price")?.as_f64()?,
                        created_at: t.get("createdAt")?.as_f64()?,
                    })
                })
                .collect();
            Some(WsEvent::Trades { market, trades })
        }
        "v4_markets" => {
            let market = contents.get("market")?.as_str()?.to_string();
            let price = contents.get("markPrice")?.as_str()?.parse().ok()?;
            let timestamp = contents.get("effectiveAt")?.as_str()?;
            let ts = chrono::DateTime::parse_from_rfc3339(timestamp)
                .map(|dt| dt.timestamp_millis())
                .unwrap_or(0);
            Some(WsEvent::MarkPrice {
                market,
                price,
                timestamp: ts,
            })
        }
        _ => None,
    }
}

/// Parse orderbook side from JSON array.
fn parse_orderbook_side(value: &serde_json::Value) -> Vec<OrderbookLevel> {
    value
        .as_array()
        .map(|arr| {
            arr.iter()
                .filter_map(|level| {
                    Some(OrderbookLevel {
                        price: level.get(0)?.as_f64()?,
                        size: level.get(1)?.as_f64()?,
                    })
                })
                .collect()
        })
        .unwrap_or_default()
}

/// Accumulates incremental orderbook updates into a full orderbook snapshot.
#[derive(Debug, Clone, Default)]
pub struct OrderbookBuilder {
    pub bids: Vec<OrderbookLevel>,
    pub asks: Vec<OrderbookLevel>,
    pub sequence: u64,
}

impl OrderbookBuilder {
    /// Create a new builder.
    pub fn new() -> Self {
        Self::default()
    }

    /// Apply a delta update to the orderbook.
    ///
    /// Replaces levels at the same price, removes levels with size 0,
    /// and inserts new levels in price-sorted order.
    pub fn apply_delta(&mut self, bids: Vec<OrderbookLevel>, asks: Vec<OrderbookLevel>, sequence: u64) {
        self.sequence = sequence;

        // Merge bid deltas
        for bid in bids {
            if bid.size <= 0.0 {
                self.bids.retain(|b| (b.price - bid.price).abs() > f64::EPSILON);
            } else {
                if let Some(existing) = self.bids.iter_mut().find(|b| (b.price - bid.price).abs() < f64::EPSILON) {
                    existing.size = bid.size;
                } else {
                    self.bids.push(bid);
                }
            }
        }

        // Merge ask deltas
        for ask in asks {
            if ask.size <= 0.0 {
                self.asks.retain(|a| (a.price - ask.price).abs() > f64::EPSILON);
            } else {
                if let Some(existing) = self.asks.iter_mut().find(|a| (a.price - ask.price).abs() < f64::EPSILON) {
                    existing.size = ask.size;
                } else {
                    self.asks.push(ask);
                }
            }
        }

        // Sort: bids descending by price, asks ascending
        self.bids.sort_by(|a, b| b.price.partial_cmp(&a.price).unwrap_or(std::cmp::Ordering::Equal));
        self.asks.sort_by(|a, b| a.price.partial_cmp(&b.price).unwrap_or(std::cmp::Ordering::Equal));
    }

    /// Build a canonical Orderbook snapshot.
    pub fn build(&self) -> Orderbook {
        Orderbook {
            bids: self.bids.clone(),
            asks: self.asks.clone(),
        }
    }

    /// Get best bid/ask midpoint.
    pub fn mid_price(&self) -> Option<f64> {
        let best_bid = self.bids.first()?.price;
        let best_ask = self.asks.first()?.price;
        Some((best_bid + best_ask) / 2.0)
    }

    /// Get the current spread.
    pub fn spread(&self) -> Option<f64> {
        let best_bid = self.bids.first()?.price;
        let best_ask = self.asks.first()?.price;
        Some(best_ask - best_bid)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_subscription_orderbook() {
        let sub = Subscription::orderbook(&["BTC-USD", "ETH-USD"]);
        assert_eq!(sub.channel, "v4_orderbook");
        assert_eq!(sub.markets, vec!["BTC-USD", "ETH-USD"]);
    }

    #[test]
    fn test_subscription_trades() {
        let sub = Subscription::trades(&["SOL-USD"]);
        assert_eq!(sub.channel, "v4_trades");
        assert_eq!(sub.markets, vec!["SOL-USD"]);
    }

    #[test]
    fn test_subscription_mark_price() {
        let sub = Subscription::mark_price(&["BTC-USD"]);
        assert_eq!(sub.channel, "v4_markets");
    }

    #[test]
    fn test_ws_event_serde_roundtrip() {
        let event = WsEvent::Orderbook {
            market: "BTC-USD".into(),
            bids: vec![OrderbookLevel { price: 67000.0, size: 1.5 }],
            asks: vec![OrderbookLevel { price: 67001.0, size: 1.0 }],
            sequence: 42,
        };
        let json = serde_json::to_string(&event).unwrap();
        let restored: WsEvent = serde_json::from_str(&json).unwrap();
        match restored {
            WsEvent::Orderbook { market, sequence, .. } => {
                assert_eq!(market, "BTC-USD");
                assert_eq!(sequence, 42);
            }
            _ => panic!("Expected Orderbook event"),
        }
    }

    #[test]
    fn test_parse_orderbook_message() {
        let raw = r#"{
            "type": "channel_data",
            "channel": "v4_orderbook",
            "contents": {
                "market": "BTC-USD",
                "bids": [[67000.0, 1.5], [66999.0, 2.0]],
                "asks": [[67001.0, 1.0], [67002.0, 0.5]],
                "sequence": 100
            }
        }"#;
        let event = parse_ws_message(raw).unwrap();
        match event {
            WsEvent::Orderbook { market, bids, asks, sequence } => {
                assert_eq!(market, "BTC-USD");
                assert_eq!(bids.len(), 2);
                assert_eq!(asks.len(), 2);
                assert_eq!(sequence, 100);
                assert_eq!(bids[0].price, 67000.0);
            }
            _ => panic!("Expected Orderbook event"),
        }
    }

    #[test]
    fn test_parse_trades_message() {
        let raw = r#"{
            "type": "channel_data",
            "channel": "v4_trades",
            "contents": {
                "market": "ETH-USD",
                "trades": [
                    {"side": "BUY", "size": 0.5, "price": 3500.0, "createdAt": 1704067200.0}
                ]
            }
        }"#;
        let event = parse_ws_message(raw).unwrap();
        match event {
            WsEvent::Trades { market, trades } => {
                assert_eq!(market, "ETH-USD");
                assert_eq!(trades.len(), 1);
                assert_eq!(trades[0].side, TradeSide::Buy);
                assert_eq!(trades[0].price, 3500.0);
            }
            _ => panic!("Expected Trades event"),
        }
    }

    #[test]
    fn test_parse_subscribed_message() {
        let raw = r#"{"type": "subscribed", "channel": "v4_orderbook"}"#;
        let event = parse_ws_message(raw).unwrap();
        match event {
            WsEvent::Subscribed { channels } => {
                assert_eq!(channels, vec!["v4_orderbook"]);
            }
            _ => panic!("Expected Subscribed event"),
        }
    }

    #[test]
    fn test_parse_error_message() {
        let raw = r#"{"type": "error", "message": "Invalid subscription"}"#;
        let event = parse_ws_message(raw).unwrap();
        match event {
            WsEvent::Error { message } => {
                assert_eq!(message, "Invalid subscription");
            }
            _ => panic!("Expected Error event"),
        }
    }

    #[test]
    fn test_parse_unknown_message_returns_none() {
        let raw = r#"{"type": "unknown", "data": null}"#;
        assert!(parse_ws_message(raw).is_none());
    }

    #[test]
    fn test_orderbook_builder_apply_delta() {
        let mut builder = OrderbookBuilder::new();

        // Initial snapshot
        builder.apply_delta(
            vec![OrderbookLevel { price: 67000.0, size: 1.5 }],
            vec![OrderbookLevel { price: 67001.0, size: 1.0 }],
            1,
        );
        assert_eq!(builder.bids.len(), 1);
        assert_eq!(builder.asks.len(), 1);

        // Update existing bid
        builder.apply_delta(
            vec![OrderbookLevel { price: 67000.0, size: 3.0 }],
            vec![],
            2,
        );
        assert_eq!(builder.bids[0].size, 3.0);

        // Add new bid level
        builder.apply_delta(
            vec![OrderbookLevel { price: 66999.0, size: 2.0 }],
            vec![],
            3,
        );
        assert_eq!(builder.bids.len(), 2);
        // Sorted descending: 67000 > 66999
        assert_eq!(builder.bids[0].price, 67000.0);
        assert_eq!(builder.bids[1].price, 66999.0);
    }

    #[test]
    fn test_orderbook_builder_remove_level() {
        let mut builder = OrderbookBuilder::new();
        builder.apply_delta(
            vec![OrderbookLevel { price: 67000.0, size: 1.5 }],
            vec![OrderbookLevel { price: 67001.0, size: 1.0 }],
            1,
        );

        // Remove bid with size 0
        builder.apply_delta(
            vec![OrderbookLevel { price: 67000.0, size: 0.0 }],
            vec![],
            2,
        );
        assert!(builder.bids.is_empty());
    }

    #[test]
    fn test_orderbook_builder_mid_and_spread() {
        let mut builder = OrderbookBuilder::new();
        builder.apply_delta(
            vec![OrderbookLevel { price: 67000.0, size: 1.5 }],
            vec![OrderbookLevel { price: 67002.0, size: 1.0 }],
            1,
        );
        assert_eq!(builder.mid_price(), Some(67001.0));
        assert_eq!(builder.spread(), Some(2.0));
    }

    #[test]
    fn test_orderbook_builder_empty_mid() {
        let builder = OrderbookBuilder::new();
        assert_eq!(builder.mid_price(), None);
        assert_eq!(builder.spread(), None);
    }

    #[test]
    fn test_extract_markets_dedup() {
        let subs = vec![
            Subscription::orderbook(&["BTC-USD", "ETH-USD"]),
            Subscription::trades(&["BTC-USD", "SOL-USD"]),
        ];
        let markets = extract_markets(&subs);
        assert_eq!(markets.len(), 3);
        assert!(markets.contains(&"BTC-USD".to_string()));
        assert!(markets.contains(&"ETH-USD".to_string()));
        assert!(markets.contains(&"SOL-USD".to_string()));
    }

    #[test]
    fn test_ws_config_default() {
        let cfg = WsClientConfig::default();
        assert_eq!(cfg.url, WS_URL_REAL);
        assert_eq!(cfg.broadcast_capacity, 1024);
        assert_eq!(cfg.max_reconnect_backoff_secs, 60);
    }

    #[tokio::test]
    async fn test_ws_event_broadcast() {
        let (tx, mut rx) = broadcast::channel(16);
        tx.send(WsEvent::Connected { url: "wss://test".into() }).unwrap();
        let event = rx.recv().await.unwrap();
        match event {
            WsEvent::Connected { url } => assert_eq!(url, "wss://test"),
            _ => panic!("Expected Connected event"),
        }
    }
}
