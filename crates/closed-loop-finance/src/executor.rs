//! # Execution Engine
//!
//! Simulates order execution with realistic slippage, fees, and partial fills.
//!
//! ## Features
//! - Market and limit order simulation
//! - Volatility-adjusted slippage model
//! - Fee model with maker/taker distinction
//! - Market impact estimation for large orders
//! - Partial fill simulation
//! - Execution quality scoring

use chrono::Utc;
use serde::{Deserialize, Serialize};
use crate::types::*;

/// Order type for execution.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum OrderType {
    Market,
    Limit,
}

/// Order side.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum OrderSide {
    Buy,
    Sell,
}

/// An order to be executed.
#[derive(Debug, Clone)]
pub struct Order {
    pub symbol: String,
    pub side: OrderSide,
    pub order_type: OrderType,
    pub quantity: f64,
    pub price: f64,         // For limit orders; ignored for market
    pub stop_price: Option<f64>,
    pub current_market_price: f64,
    pub current_volatility: f64,
    pub avg_daily_volume: f64,
}

/// The execution engine simulating trade execution.
#[derive(Debug, Clone)]
pub struct ExecutionEngine {
    pub fee_config: FeeConfig,
    pub slippage_config: SlippageConfig,
    pub partial_fill_probability: f64,
    pub max_partial_fill_ratio: f64,
}

impl Default for ExecutionEngine {
    fn default() -> Self {
        ExecutionEngine {
            fee_config: FeeConfig::default(),
            slippage_config: SlippageConfig::default(),
            partial_fill_probability: 0.1,
            max_partial_fill_ratio: 0.7,
        }
    }
}

impl ExecutionEngine {
    pub fn new(fee_config: FeeConfig, slippage_config: SlippageConfig) -> Self {
        ExecutionEngine {
            fee_config,
            slippage_config,
            ..Default::default()
        }
    }

    /// Execute an order and return the result.
    pub fn execute(&self, order: &Order) -> ExecutionResult {
        // Calculate slippage
        let slippage = self.calculate_slippage(order);

        // Determine fill price
        let raw_price = match order.order_type {
            OrderType::Market => order.current_market_price,
            OrderType::Limit => {
                let limit_ok = match order.side {
                    OrderSide::Buy => order.price >= order.current_market_price,
                    OrderSide::Sell => order.price <= order.current_market_price,
                };
                if !limit_ok {
                    return ExecutionResult {
                        action: match order.side { OrderSide::Buy => Action::Buy, OrderSide::Sell => Action::Sell },
                        filled_price: 0.0,
                        filled_quantity: 0.0,
                        slippage: 0.0,
                        fees: 0.0,
                        timestamp: Utc::now(),
                        success: false,
                        quality_score: 0.0,
                        symbol: order.symbol.clone(),
                        order_type: format!("{:?}", order.order_type),
                    };
                }
                order.price
            }
        };

        let filled_price = match order.side {
            OrderSide::Buy => raw_price * (1.0 + slippage),
            OrderSide::Sell => raw_price * (1.0 - slippage),
        };

        // Partial fill simulation
        let fill_ratio = if self.partial_fill_probability > 0.0 {
            // Deterministic: use order size relative to avg daily volume
            let size_ratio = (order.quantity * filled_price) / order.avg_daily_volume.max(1.0);
            if size_ratio > 0.05 {
                // Large orders get partial fills
                1.0 - (size_ratio - 0.05) * self.max_partial_fill_ratio
            } else {
                1.0
            }
        } else {
            1.0
        };

        let filled_quantity = order.quantity * fill_ratio.clamp(0.0, 1.0);

        // Calculate fees
        let notional = filled_price * filled_quantity;
        let fees = self.fee_config.fixed_fee
            + notional * self.fee_config.percentage_fee
            + notional * if order.order_type == OrderType::Limit {
                self.fee_config.maker_fee
            } else {
                self.fee_config.taker_fee
            };

        // Execution quality score
        let quality = self.score_execution(slippage, fill_ratio, fees / notional.max(1e-15));

        let action = match order.side {
            OrderSide::Buy => Action::Buy,
            OrderSide::Sell => Action::Sell,
        };

        ExecutionResult {
            action,
            filled_price,
            filled_quantity,
            slippage,
            fees,
            timestamp: Utc::now(),
            success: filled_quantity > 0.0,
            quality_score: quality,
            symbol: order.symbol.clone(),
            order_type: format!("{:?}", order.order_type),
        }
    }

    /// Calculate slippage based on volatility and order size.
    pub fn calculate_slippage(&self, order: &Order) -> f64 {
        let vol_component = order.current_volatility * self.slippage_config.volatility_multiplier;
        let size_component = {
            let notional = order.quantity * order.current_market_price;
            let size_ratio = notional / order.avg_daily_volume.max(1.0);
            size_ratio * self.slippage_config.volume_impact_factor
        };
        self.slippage_config.base_slippage + vol_component + size_component
    }

    /// Estimate market impact for a large order.
    pub fn estimate_market_impact(&self, quantity: f64, price: f64, avg_daily_volume: f64) -> f64 {
        let notional = quantity * price;
        let participation_rate = notional / avg_daily_volume.max(1.0);
        // Square root market impact model (common in practice)
        0.1 * participation_rate.sqrt()
    }

    /// Calculate fees for a given notional value.
    pub fn calculate_fees(&self, notional: f64, order_type: OrderType) -> f64 {
        let rate = match order_type {
            OrderType::Market => self.fee_config.taker_fee,
            OrderType::Limit => self.fee_config.maker_fee,
        };
        self.fee_config.fixed_fee + notional * (self.fee_config.percentage_fee + rate)
    }

    /// Score execution quality in [0, 1].
    pub fn score_execution(&self, slippage: f64, fill_ratio: f64, fee_rate: f64) -> f64 {
        let slippage_score = (1.0 - slippage * 100.0).max(0.0); // Lower slippage = better
        let fill_score = fill_ratio; // Higher fill ratio = better
        let fee_score = (1.0 - fee_rate * 100.0).max(0.0);

        // Weighted average
        (slippage_score * 0.4 + fill_score * 0.4 + fee_score * 0.2).clamp(0.0, 1.0)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn market_buy_order(quantity: f64, price: f64) -> Order {
        Order {
            symbol: "BTC/USDT".into(),
            side: OrderSide::Buy,
            order_type: OrderType::Market,
            quantity,
            price,
            stop_price: None,
            current_market_price: price,
            current_volatility: 0.15,
            avg_daily_volume: 1_000_000.0,
        }
    }

    #[test]
    fn test_execute_market_buy() {
        let engine = ExecutionEngine::default();
        let order = market_buy_order(1.0, 100.0);
        let result = engine.execute(&order);
        assert!(result.success);
        assert_eq!(result.filled_quantity, 1.0);
        assert!(result.filled_price >= 100.0); // Buy slippage adds
        assert!(result.fees > 0.0);
    }

    #[test]
    fn test_execute_market_sell() {
        let engine = ExecutionEngine::default();
        let mut order = market_buy_order(1.0, 100.0);
        order.side = OrderSide::Sell;
        let result = engine.execute(&order);
        assert!(result.success);
        assert!(result.filled_price <= 100.0); // Sell slippage subtracts
    }

    #[test]
    fn test_execute_limit_buy_fill() {
        let engine = ExecutionEngine::default();
        let mut order = market_buy_order(1.0, 100.0);
        order.order_type = OrderType::Limit;
        order.price = 105.0; // Above market, should fill
        let result = engine.execute(&order);
        assert!(result.success);
    }

    #[test]
    fn test_execute_limit_buy_no_fill() {
        let engine = ExecutionEngine::default();
        let mut order = market_buy_order(1.0, 100.0);
        order.order_type = OrderType::Limit;
        order.price = 95.0; // Below market, should NOT fill for buy
        let result = engine.execute(&order);
        assert!(!result.success);
        assert_eq!(result.filled_quantity, 0.0);
    }

    #[test]
    fn test_slippage_calculation() {
        let engine = ExecutionEngine::default();
        let order = market_buy_order(1.0, 100.0);
        let slip = engine.calculate_slippage(&order);
        assert!(slip > 0.0);
    }

    #[test]
    fn test_slippage_increases_with_volatility() {
        let engine = ExecutionEngine::default();
        let mut order = market_buy_order(1.0, 100.0);
        order.current_volatility = 0.05;
        let slip_low = engine.calculate_slippage(&order);
        order.current_volatility = 0.50;
        let slip_high = engine.calculate_slippage(&order);
        assert!(slip_high > slip_low);
    }

    #[test]
    fn test_market_impact() {
        let engine = ExecutionEngine::default();
        let impact = engine.estimate_market_impact(100.0, 100.0, 1_000_000.0);
        assert!(impact > 0.0);
        assert!(impact < 1.0);
    }

    #[test]
    fn test_market_impact_large_order() {
        let engine = ExecutionEngine::default();
        let small = engine.estimate_market_impact(10.0, 100.0, 1_000_000.0);
        let large = engine.estimate_market_impact(10000.0, 100.0, 1_000_000.0);
        assert!(large > small);
    }

    #[test]
    fn test_fee_calculation_market() {
        let engine = ExecutionEngine::default();
        let fee = engine.calculate_fees(10_000.0, OrderType::Market);
        assert!(fee > 0.0);
        // Taker fee should be higher than maker
        let fee_maker = engine.calculate_fees(10_000.0, OrderType::Limit);
        assert!(fee > fee_maker);
    }

    #[test]
    fn test_execution_quality_score() {
        let engine = ExecutionEngine::default();
        let score = engine.score_execution(0.001, 1.0, 0.001);
        assert!(score > 0.8); // Good execution
    }

    #[test]
    fn test_execution_quality_poor() {
        let engine = ExecutionEngine::default();
        let score = engine.score_execution(0.05, 0.5, 0.01);
        assert!(score < 0.8); // Poor execution
    }

    #[test]
    fn test_partial_fill_large_order() {
        let engine = ExecutionEngine::default();
        let order = Order {
            symbol: "BTC/USDT".into(),
            side: OrderSide::Buy,
            order_type: OrderType::Market,
            quantity: 100.0,
            price: 100.0,
            stop_price: None,
            current_market_price: 100.0,
            current_volatility: 0.15,
            avg_daily_volume: 10_000.0, // Small daily volume
        };
        let result = engine.execute(&order);
        assert!(result.success);
        assert!(result.filled_quantity < 100.0);
    }

    #[test]
    fn test_zero_quantity_order() {
        let engine = ExecutionEngine::default();
        let order = market_buy_order(0.0, 100.0);
        let result = engine.execute(&order);
        assert_eq!(result.filled_quantity, 0.0);
    }

    #[test]
    fn test_fees_include_fixed_and_percentage() {
        let engine = ExecutionEngine {
            fee_config: FeeConfig {
                fixed_fee: 1.0,
                percentage_fee: 0.01,
                maker_fee: 0.0,
                taker_fee: 0.0,
            },
            slippage_config: SlippageConfig::default(),
            ..Default::default()
        };
        let order = market_buy_order(1.0, 100.0);
        let result = engine.execute(&order);
        // Fee = 1.0 + 100.0 * 0.01 = 2.0
        assert!((result.fees - 2.0).abs() < 0.01);
    }
}
