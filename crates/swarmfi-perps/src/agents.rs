//! Nine specialized market analysis agents.
//!
//! Each agent analyzes a different aspect of market data and returns
//! an AgentVote with signal, confidence, and reasoning.

use crate::math::{clamp, parse_f64_or_zero, sma, std_dev};
use crate::types::*;

// ── Core Agents ──────────────────────────────────────────────────

/// 1. FundingAgent — Analyzes historical funding rates for long/short bias.
///
/// Very positive funding → overleveraged long → SHORT.
/// Negative funding → overleveraged short → LONG.
pub fn funding_agent(data: &MarketDataBundle) -> AgentVote {
    let funding_rate = data.stats.funding_rate_1h;

    if data.funding.is_empty() {
        return AgentVote {
            agent_type: "FundingAgent".into(),
            signal: Signal::Neutral,
            confidence: 20.0,
            reasoning: "No funding data available".into(),
        };
    }

    let mut reasons = Vec::new();
    reasons.push(format!("Current 1h funding: {:.4}% APR", funding_rate));

    // Average funding over history (annualized)
    let rates: Vec<f64> = data.funding.iter().map(|f| parse_f64_or_zero(&f.rate)).collect();
    let avg_funding = sma(&rates) * 100.0 * 24.0 * 365.0;
    reasons.push(format!("Avg funding (annualized): {:.4}%", avg_funding));

    // Count consecutive same-direction funding periods
    let mut consecutive_positive = 0u32;
    let mut consecutive_negative = 0u32;
    for r in &rates {
        if *r > 0.0 {
            consecutive_negative = 0;
            consecutive_positive += 1;
        } else if *r < 0.0 {
            consecutive_positive = 0;
            consecutive_negative += 1;
        } else {
            break;
        }
    }

    let (signal, confidence) = if funding_rate > 0.05 {
        (
            Signal::Short,
            clamp(40.0 + consecutive_positive as f64 * 5.0, 45.0, 85.0),
        )
    } else if funding_rate < -0.05 {
        (
            Signal::Long,
            clamp(40.0 + consecutive_negative as f64 * 5.0, 45.0, 85.0),
        )
    } else if funding_rate > 0.01 {
        (
            Signal::Short,
            clamp(35.0 + consecutive_positive as f64 * 3.0, 35.0, 60.0),
        )
    } else if funding_rate < -0.01 {
        (
            Signal::Long,
            clamp(35.0 + consecutive_negative as f64 * 3.0, 35.0, 60.0),
        )
    } else {
        (Signal::Neutral, 30.0)
    };

    let reason_suffix = match signal {
        Signal::Short if funding_rate > 0.01 => format!(
            "({} consecutive positive periods)",
            consecutive_positive
        ),
        Signal::Long if funding_rate < -0.01 => format!(
            "({} consecutive negative periods)",
            consecutive_negative
        ),
        _ => "no strong directional bias".into(),
    };
    reasons.push(reason_suffix);

    AgentVote {
        agent_type: "FundingAgent".into(),
        signal,
        confidence,
        reasoning: reasons.join(". "),
    }
}

/// 2. MomentumAgent — Price momentum via SMA comparison and candle streaks.
pub fn momentum_agent(data: &MarketDataBundle) -> AgentVote {
    if data.candles.len() < 5 {
        return AgentVote {
            agent_type: "MomentumAgent".into(),
            signal: Signal::Neutral,
            confidence: 20.0,
            reasoning: "Insufficient candle data".into(),
        };
    }

    let closes: Vec<f64> = data.candles.iter().map(|c| c.close).collect();
    let current_price = *closes.last().unwrap();
    let sma10 = sma(&closes[closes.len().saturating_sub(10)..]);
    let sma5 = sma(&closes[closes.len().saturating_sub(5)..]);

    // Count consecutive green/red candles from the end
    let mut consecutive_green = 0u32;
    let mut consecutive_red = 0u32;
    for candle in data.candles.iter().rev() {
        let change = candle.close - candle.open;
        if change > 0.0 {
            consecutive_green += 1;
            if consecutive_red > 0 { break; }
        } else if change < 0.0 {
            consecutive_red += 1;
            if consecutive_green > 0 { break; }
        } else {
            break;
        }
    }

    let lookback = 5.min(closes.len());
    let price_change = current_price - closes[closes.len() - lookback];
    let price_change_pct = (price_change / current_price) * 100.0;

    let mut reasons = Vec::new();
    if sma5 > 0.0 {
        reasons.push(format!("Price vs SMA5: {:.3}%", ((current_price / sma5 - 1.0) * 100.0)));
    }
    if sma10 > 0.0 {
        reasons.push(format!("Price vs SMA10: {:.3}%", ((current_price / sma10 - 1.0) * 100.0)));
    }
    reasons.push(format!(
        "{} consecutive {} candles",
        consecutive_green.max(consecutive_red),
        if consecutive_green > consecutive_red { "green" } else { "red" }
    ));

    let (signal, confidence) =
        if current_price > sma5 && current_price > sma10 && consecutive_green >= 3 {
            (
                Signal::Long,
                clamp(50.0 + consecutive_green as f64 * 5.0 + price_change_pct.abs() * 2.0, 50.0, 85.0),
            )
        } else if current_price < sma5 && current_price < sma10 && consecutive_red >= 3 {
            (
                Signal::Short,
                clamp(50.0 + consecutive_red as f64 * 5.0 + price_change_pct.abs() * 2.0, 50.0, 85.0),
            )
        } else if current_price > sma10 {
            (
                Signal::Long,
                clamp(40.0 + consecutive_green as f64 * 3.0, 40.0, 65.0),
            )
        } else if current_price < sma10 {
            (
                Signal::Short,
                clamp(40.0 + consecutive_red as f64 * 3.0, 40.0, 65.0),
            )
        } else {
            (Signal::Neutral, 30.0)
        };

    AgentVote {
        agent_type: "MomentumAgent".into(),
        signal,
        confidence,
        reasoning: reasons.join(". "),
    }
}

/// 3. VolatilityAgent — Measures hourly return volatility.
///
/// High volatility → NEUTRAL (uncertainty).
/// Low volatility → looks for breakout direction.
pub fn volatility_agent(data: &MarketDataBundle) -> AgentVote {
    if data.candles.len() < 5 {
        return AgentVote {
            agent_type: "VolatilityAgent".into(),
            signal: Signal::Neutral,
            confidence: 20.0,
            reasoning: "Insufficient candle data for volatility analysis".into(),
        };
    }

    // Hourly returns
    let returns: Vec<f64> = data.candles.windows(2)
        .map(|w| (w[1].close - w[0].close) / w[0].close)
        .collect();

    let vol = std_dev(&returns);
    let avg_vol = vol * 100.0;

    // Range compression
    let recent: Vec<_> = data.candles[data.candles.len().saturating_sub(5)..].to_vec();
    let recent_high = recent.iter().map(|c| c.high).fold(f64::NEG_INFINITY, f64::max);
    let recent_low = recent.iter().map(|c| c.low).fold(f64::INFINITY, f64::min);
    let current_price = data.candles.last().unwrap().close;
    let range_pct = if current_price > 0.0 {
        ((recent_high - recent_low) / current_price) * 100.0
    } else {
        0.0
    };

    let mut reasons = Vec::new();
    reasons.push(format!("Hourly volatility: {:.4}%", avg_vol));
    reasons.push(format!("5-candle range: {:.3}%", range_pct));

    let (signal, confidence) = if avg_vol > 0.03 || range_pct > 2.0 {
        (
            Signal::Neutral,
            clamp(55.0 + avg_vol * 200.0, 55.0, 75.0),
        )
    } else if avg_vol < 0.005 && range_pct < 0.3 {
        // Compressed volatility — check for micro-trend
        let last_three: Vec<_> = data.candles[data.candles.len().saturating_sub(3)..].to_vec();
        let trend_up = last_three.iter().all(|c| c.close > c.open);
        let trend_down = last_three.iter().all(|c| c.close < c.open);
        if trend_up {
            (Signal::Long, 55.0)
        } else if trend_down {
            (Signal::Short, 55.0)
        } else {
            (Signal::Neutral, 50.0)
        }
    } else {
        // Normal volatility — lean towards SMA direction
        let closes: Vec<f64> = data.candles.iter().map(|c| c.close).collect();
        let sma_val = sma(&closes[closes.len().saturating_sub(10)..]);
        if current_price > sma_val {
            (Signal::Long, 45.0)
        } else {
            (Signal::Short, 45.0)
        }
    };

    AgentVote {
        agent_type: "VolatilityAgent".into(),
        signal,
        confidence,
        reasoning: reasons.join(". "),
    }
}

/// 4. VolumeAgent — Compares recent volume to average and trade flow imbalance.
pub fn volume_agent(data: &MarketDataBundle) -> AgentVote {
    let mut reasons = Vec::new();

    if data.candles.len() >= 2 {
        let volumes: Vec<f64> = data.candles.iter().map(|c| c.usd_volume).filter(|&v| v > 0.0).collect();
        let recent_vol: Vec<f64> = data.candles[data.candles.len().saturating_sub(3)..]
            .iter().map(|c| c.usd_volume).filter(|&v| v > 0.0).collect();

        if volumes.len() >= 5 && !recent_vol.is_empty() {
            let avg_vol = sma(&volumes[volumes.len().saturating_sub(10)..]);
            let recent_avg_vol = sma(&recent_vol);
            let volume_ratio = if avg_vol > 0.0 { recent_avg_vol / avg_vol } else { 1.0 };

            reasons.push(format!("Volume ratio (recent/avg): {:.2}x", volume_ratio));

            let last_candle = data.candles.last().unwrap();
            let price_change = last_candle.close - last_candle.open;

            if volume_ratio > 1.5 {
                if price_change > 0.0 {
                    return AgentVote {
                        agent_type: "VolumeAgent".into(),
                        signal: Signal::Long,
                        confidence: clamp(50.0 + volume_ratio * 10.0, 50.0, 80.0),
                        reasoning: format!(
                            "{}. High volume ({:.2}x average) with upward price action — strong buying conviction",
                            reasons.join(". "), volume_ratio
                        ),
                    };
                } else if price_change < 0.0 {
                    return AgentVote {
                        agent_type: "VolumeAgent".into(),
                        signal: Signal::Short,
                        confidence: clamp(50.0 + volume_ratio * 10.0, 50.0, 80.0),
                        reasoning: format!(
                            "{}. High volume ({:.2}x average) with downward price action — strong selling conviction",
                            reasons.join(". "), volume_ratio
                        ),
                    };
                }
            } else if volume_ratio < 0.5 {
                return AgentVote {
                    agent_type: "VolumeAgent".into(),
                    signal: Signal::Neutral,
                    confidence: 50.0,
                    reasoning: format!(
                        "{}. Low volume ({:.2}x average) — low conviction",
                        reasons.join(". "), volume_ratio
                    ),
                };
            }
        }
    }

    // Fallback: trade flow imbalance
    if !data.trades.is_empty() {
        let recent_trades = &data.trades[..50.min(data.trades.len())];
        let mut buy_volume = 0.0;
        let mut sell_volume = 0.0;

        for t in recent_trades {
            let usd = t.size * t.price;
            match t.side {
                TradeSide::Buy => buy_volume += usd,
                TradeSide::Sell => sell_volume += usd,
            }
        }

        let total_volume = buy_volume + sell_volume;
        let buy_ratio = if total_volume > 0.0 { buy_volume / total_volume } else { 0.5 };

        reasons.push(format!("Buy/sell ratio from trades: {:.1}%", buy_ratio * 100.0));

        if buy_ratio > 0.6 {
            return AgentVote {
                agent_type: "VolumeAgent".into(),
                signal: Signal::Long,
                confidence: clamp(40.0 + (buy_ratio - 0.5) * 200.0, 40.0, 70.0),
                reasoning: reasons.join(". "),
            };
        } else if buy_ratio < 0.4 {
            return AgentVote {
                agent_type: "VolumeAgent".into(),
                signal: Signal::Short,
                confidence: clamp(40.0 + (0.5 - buy_ratio) * 200.0, 40.0, 70.0),
                reasoning: reasons.join(". "),
            };
        }
    }

    AgentVote {
        agent_type: "VolumeAgent".into(),
        signal: Signal::Neutral,
        confidence: 35.0,
        reasoning: if reasons.is_empty() {
            "Insufficient volume data".into()
        } else {
            reasons.join(". ")
        },
    }
}

/// 5. OrderbookAgent — Analyzes bid/ask depth imbalance.
pub fn orderbook_agent(data: &MarketDataBundle) -> AgentVote {
    let ob = match &data.orderbook {
        Some(ob) => ob,
        None => {
            return AgentVote {
                agent_type: "OrderbookAgent".into(),
                signal: Signal::Neutral,
                confidence: 20.0,
                reasoning: "No orderbook data available".into(),
            };
        }
    };

    if ob.bids.is_empty() || ob.asks.is_empty() {
        return AgentVote {
            agent_type: "OrderbookAgent".into(),
            signal: Signal::Neutral,
            confidence: 20.0,
            reasoning: "Empty orderbook".into(),
        };
    }

    let mut reasons = Vec::new();

    // Aggregate bid/ask depth
    let bid_depth: f64 = ob.bids.iter().map(|b| b.size * b.price).sum();
    let ask_depth: f64 = ob.asks.iter().map(|a| a.size * a.price).sum();
    let total_depth = bid_depth + ask_depth;
    let bid_ratio = if total_depth > 0.0 { bid_depth / total_depth } else { 0.5 };

    reasons.push(format!("Bid depth: ${:.2}, Ask depth: ${:.2}", bid_depth, ask_depth));
    reasons.push(format!("Bid/Ask ratio: {:.1}%", bid_ratio * 100.0));

    // Top 5 depth
    let top5_bid: f64 = ob.bids[..5.min(ob.bids.len())].iter().map(|b| b.size * b.price).sum();
    let top5_ask: f64 = ob.asks[..5.min(ob.asks.len())].iter().map(|a| a.size * a.price).sum();
    let top5_total = top5_bid + top5_ask;
    let top5_bid_ratio = if top5_total > 0.0 { top5_bid / top5_total } else { 0.5 };

    reasons.push(format!("Top-5 bid/ask ratio: {:.1}%", top5_bid_ratio * 100.0));

    // Spread
    let best_bid = ob.bids[0].price;
    let best_ask = ob.asks[0].price;
    let spread = if best_ask > 0.0 {
        ((best_ask - best_bid) / best_ask) * 100.0
    } else {
        0.0
    };
    reasons.push(format!("Spread: {:.4}%", spread));

    let combined_bid_ratio = (bid_ratio + top5_bid_ratio) / 2.0;

    let (signal, confidence) = if combined_bid_ratio > 0.6 {
        (
            Signal::Long,
            clamp(45.0 + (combined_bid_ratio - 0.5) * 150.0, 45.0, 75.0),
        )
    } else if combined_bid_ratio < 0.4 {
        (
            Signal::Short,
            clamp(45.0 + (0.5 - combined_bid_ratio) * 150.0, 45.0, 75.0),
        )
    } else if spread > 0.1 {
        (Signal::Neutral, 55.0)
    } else {
        (Signal::Neutral, 35.0)
    };

    AgentVote {
        agent_type: "OrderbookAgent".into(),
        signal,
        confidence,
        reasoning: reasons.join(". "),
    }
}

/// 6. LiquidationAgent — Estimates liquidation risk from funding + volatility + wick patterns.
pub fn liquidation_agent(data: &MarketDataBundle) -> AgentVote {
    let mut reasons = Vec::new();

    // Volatility
    let vol = if data.candles.len() >= 5 {
        let returns: Vec<f64> = data.candles.windows(2)
            .map(|w| (w[1].close - w[0].close) / w[0].close)
            .collect();
        std_dev(&returns) * 100.0
    } else {
        0.0
    };

    let funding_rate = data.stats.funding_rate_1h;
    let funding_abs = funding_rate.abs();

    reasons.push(format!("Volatility: {:.4}%", vol));
    reasons.push(format!("Funding rate (annualized): {:.4}%", funding_rate));

    // Risk score: 0–10
    let mut risk_score: u32 = 0;

    // Funding contribution (0–3)
    if funding_abs > 0.05 { risk_score += 3; }
    else if funding_abs > 0.02 { risk_score += 2; }
    else if funding_abs > 0.01 { risk_score += 1; }

    // Volatility contribution (0–3)
    if vol > 0.03 { risk_score += 3; }
    else if vol > 0.01 { risk_score += 2; }
    else if vol > 0.005 { risk_score += 1; }

    // Wick rejection patterns (0–3 from last 3 candles)
    if data.candles.len() >= 3 {
        for c in data.candles[data.candles.len().saturating_sub(3)..].iter() {
            let body = (c.close - c.open).abs();
            let upper_wick = c.high - c.close.max(c.open);
            let lower_wick = c.close.min(c.open) - c.low;
            if upper_wick > body * 2.0 {
                risk_score += 1;
                reasons.push("Detected upper wick rejection — potential selling pressure at highs".into());
            }
            if lower_wick > body * 2.0 {
                risk_score += 1;
                reasons.push("Detected lower wick rejection — potential buying support at lows".into());
            }
        }
    }

    reasons.push(format!("Liquidation risk score: {}/10", risk_score));

    let (signal, confidence) = if risk_score >= 6 {
        let (sig, _suffix) = if funding_rate > 0.0 {
            (Signal::Short, "long squeeze potential")
        } else {
            (Signal::Long, "short squeeze potential")
        };
        (sig, clamp(55.0 + risk_score as f64 * 3.0, 55.0, 80.0))
    } else if risk_score >= 3 {
        let sig = if funding_rate > 0.0 { Signal::Short } else { Signal::Long };
        (sig, clamp(42.0 + risk_score as f64 * 2.0, 42.0, 60.0))
    } else {
        (Signal::Neutral, 30.0)
    };

    AgentVote {
        agent_type: "LiquidationAgent".into(),
        signal,
        confidence,
        reasoning: reasons.join(". "),
    }
}

/// 7. MeanReversionAgent — Z-score based mean reversion signal.
pub fn mean_reversion_agent(data: &MarketDataBundle) -> AgentVote {
    if data.candles.len() < 10 {
        return AgentVote {
            agent_type: "MeanReversionAgent".into(),
            signal: Signal::Neutral,
            confidence: 20.0,
            reasoning: "Insufficient candle data for mean reversion analysis".into(),
        };
    }

    let closes: Vec<f64> = data.candles.iter().map(|c| c.close).collect();
    let current_price = *closes.last().unwrap();
    let mean = sma(&closes);
    let sd = std_dev(&closes);
    let z_score = if sd > 0.0 { (current_price - mean) / sd } else { 0.0 };

    let mut reasons = Vec::new();
    reasons.push(format!("Current price: {:.2}", current_price));
    reasons.push(format!("Mean price: {:.2}", mean));
    reasons.push(format!("Std deviation: {:.2}", sd));
    reasons.push(format!("Z-score: {:.3}", z_score));

    let (signal, confidence) = if z_score > 2.0 {
        (Signal::Short, clamp(50.0 + (z_score - 2.0) * 10.0, 50.0, 80.0))
    } else if z_score < -2.0 {
        (Signal::Long, clamp(50.0 + (z_score.abs() - 2.0) * 10.0, 50.0, 80.0))
    } else if z_score > 1.5 {
        (Signal::Short, clamp(40.0 + (z_score - 1.5) * 10.0, 40.0, 60.0))
    } else if z_score < -1.5 {
        (Signal::Long, clamp(40.0 + (z_score.abs() - 1.5) * 10.0, 40.0, 60.0))
    } else {
        (Signal::Neutral, 30.0)
    };

    AgentVote {
        agent_type: "MeanReversionAgent".into(),
        signal,
        confidence,
        reasoning: reasons.join(". "),
    }
}

/// 8. TrendAgent — Multi-timeframe SMA alignment analysis.
pub fn trend_agent(data: &MarketDataBundle) -> AgentVote {
    if data.candles.len() < 10 {
        return AgentVote {
            agent_type: "TrendAgent".into(),
            signal: Signal::Neutral,
            confidence: 20.0,
            reasoning: "Insufficient candle data for trend analysis".into(),
        };
    }

    let closes: Vec<f64> = data.candles.iter().map(|c| c.close).collect();
    let current_price = *closes.last().unwrap();

    let mut reasons = Vec::new();

    // Short-term (5 candles)
    let short_sma = sma(&closes[closes.len().saturating_sub(5)..]);
    let short_trend = current_price > short_sma;
    let short_dist = if short_sma > 0.0 { (current_price / short_sma - 1.0) * 100.0 } else { 0.0 };
    reasons.push(format!(
        "Short-term trend (5-candle): {} ({:.3}% from SMA)",
        if short_trend { "UP" } else { "DOWN" },
        short_dist
    ));

    // Medium-term (10 candles)
    let med_sma = sma(&closes[closes.len().saturating_sub(10)..]);
    let med_trend = current_price > med_sma;
    let med_dist = if med_sma > 0.0 { (current_price / med_sma - 1.0) * 100.0 } else { 0.0 };
    reasons.push(format!(
        "Medium-term trend (10-candle): {} ({:.3}% from SMA)",
        if med_trend { "UP" } else { "DOWN" },
        med_dist
    ));

    // Longer-term (20 candles) if available
    if closes.len() >= 20 {
        let long_sma = sma(&closes[closes.len().saturating_sub(20)..]);
        let long_trend = current_price > long_sma;
        let long_dist = if long_sma > 0.0 { (current_price / long_sma - 1.0) * 100.0 } else { 0.0 };
        reasons.push(format!(
            "Longer-term trend (20-candle): {} ({:.3}% from SMA)",
            if long_trend { "UP" } else { "DOWN" },
            long_dist
        ));

        // SMA alignment
        let aligned_up = short_sma > med_sma && med_sma > long_sma && current_price > short_sma;
        let aligned_down = short_sma < med_sma && med_sma < long_sma && current_price < short_sma;

        if aligned_up {
            return AgentVote {
                agent_type: "TrendAgent".into(),
                signal: Signal::Long,
                confidence: 75.0,
                reasoning: format!(
                    "{}. STRONG BULLISH: All SMAs aligned upward with price above all averages",
                    reasons.join(". ")
                ),
            };
        } else if aligned_down {
            return AgentVote {
                agent_type: "TrendAgent".into(),
                signal: Signal::Short,
                confidence: 75.0,
                reasoning: format!(
                    "{}. STRONG BEARISH: All SMAs aligned downward with price below all averages",
                    reasons.join(". ")
                ),
            };
        }
    }

    // Short/medium alignment
    if short_trend == med_trend {
        let signal = if short_trend { Signal::Long } else { Signal::Short };
        let reason = if short_trend {
            "Bullish: Short and medium term trends aligned upward"
        } else {
            "Bearish: Short and medium term trends aligned downward"
        };
        AgentVote {
            agent_type: "TrendAgent".into(),
            signal,
            confidence: 60.0,
            reasoning: format!("{}. {}", reasons.join(". "), reason),
        }
    } else {
        AgentVote {
            agent_type: "TrendAgent".into(),
            signal: Signal::Neutral,
            confidence: 40.0,
            reasoning: format!("{}. Mixed trend signals — short and medium term trends diverging", reasons.join(". ")),
        }
    }
}

/// 9. SentimentAgent (Meta-Agent) — Synthesizes other agents' signals.
pub fn sentiment_agent(_data: &MarketDataBundle, previous_votes: &[AgentVote]) -> AgentVote {
    if previous_votes.is_empty() {
        return AgentVote {
            agent_type: "SentimentAgent".into(),
            signal: Signal::Neutral,
            confidence: 20.0,
            reasoning: "No other agent votes to aggregate for sentiment analysis".into(),
        };
    }

    let mut longs = 0u32;
    let mut shorts = 0u32;
    let mut neutrals = 0u32;
    let mut weighted_long = 0.0;
    let mut weighted_short = 0.0;

    for vote in previous_votes {
        match vote.signal {
            Signal::Long => { longs += 1; weighted_long += vote.confidence; }
            Signal::Short => { shorts += 1; weighted_short += vote.confidence; }
            Signal::Neutral => { neutrals += 1; }
        }
    }

    let total = previous_votes.len() as f64;
    let long_pct = (longs as f64 / total) * 100.0;
    let short_pct = (shorts as f64 / total) * 100.0;
    let neutral_pct = (neutrals as f64 / total) * 100.0;

    let divergence = (longs as i32 - shorts as i32).abs() as f64 / total;

    let avg_long_conviction = if longs > 0 { weighted_long / longs as f64 } else { 0.0 };
    let avg_short_conviction = if shorts > 0 { weighted_short / shorts as f64 } else { 0.0 };

    let mut reasons = Vec::new();
    reasons.push(format!("Agent sentiment: {} LONG, {} SHORT, {} NEUTRAL", longs, shorts, neutrals));
    reasons.push(format!("Sentiment split: {:.0}% bullish, {:.0}% bearish, {:.0}% neutral", long_pct, short_pct, neutral_pct));
    reasons.push(format!("Sentiment divergence: {:.2}", divergence));
    reasons.push(format!("Avg LONG conviction: {:.1}, Avg SHORT conviction: {:.1}", avg_long_conviction, avg_short_conviction));

    let (signal, confidence) = if divergence < 0.2 {
        (Signal::Neutral, clamp(50.0 + neutrals as f64 * 5.0, 50.0, 70.0))
    } else if longs > shorts && long_pct >= 60.0 {
        (Signal::Long, clamp(45.0 + long_pct * 0.3 + avg_long_conviction * 0.1, 45.0, 75.0))
    } else if shorts > longs && short_pct >= 60.0 {
        (Signal::Short, clamp(45.0 + short_pct * 0.3 + avg_short_conviction * 0.1, 45.0, 75.0))
    } else {
        let sig = if longs > shorts { Signal::Long } else if shorts > longs { Signal::Short } else { Signal::Neutral };
        (sig, 35.0)
    };

    AgentVote {
        agent_type: "SentimentAgent".into(),
        signal,
        confidence,
        reasoning: reasons.join(". "),
    }
}

// ── Agent Registry & Runner ───────────────────────────────────────

/// The 8 core agent types (SentimentAgent is meta and runs separately).
pub const CORE_AGENTS: &[&str] = &[
    "FundingAgent",
    "MomentumAgent",
    "VolatilityAgent",
    "VolumeAgent",
    "OrderbookAgent",
    "LiquidationAgent",
    "MeanReversionAgent",
    "TrendAgent",
];

/// All 9 agent types including SentimentAgent.
pub const ALL_AGENT_TYPES: &[&str] = &[
    "FundingAgent", "MomentumAgent", "VolatilityAgent", "VolumeAgent",
    "OrderbookAgent", "LiquidationAgent", "MeanReversionAgent", "TrendAgent",
    "SentimentAgent",
];

/// Run all 9 agents on market data. Returns all 9 votes.
pub fn run_all_agents(data: &MarketDataBundle) -> Vec<AgentVote> {
    let core_votes = vec![
        funding_agent(data),
        momentum_agent(data),
        volatility_agent(data),
        volume_agent(data),
        orderbook_agent(data),
        liquidation_agent(data),
        mean_reversion_agent(data),
        trend_agent(data),
    ];

    // SentimentAgent runs last with access to all core votes
    let sentiment_vote = sentiment_agent(data, &core_votes);

    let mut all = core_votes;
    all.push(sentiment_vote);
    all
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_test_bundle() -> MarketDataBundle {
        let candles: Vec<Candle> = (0..20)
            .map(|i| {
                let base = 65000.0 + (i as f64) * 50.0;
                Candle {
                    started_at: format!("2025-01-01T{:02}:00:00Z", i),
                    open: base,
                    high: base + 100.0,
                    low: base - 50.0,
                    close: base + 75.0,
                    base_token_volume: 100.0,
                    usd_volume: 6_500_000.0 + i as f64 * 100_000.0,
                    trades: 500,
                }
            })
            .collect();

        MarketDataBundle {
            orderbook: Some(Orderbook {
                bids: vec![
                    OrderbookLevel { price: 66000.0, size: 1.0 },
                    OrderbookLevel { price: 65900.0, size: 2.0 },
                    OrderbookLevel { price: 65800.0, size: 1.5 },
                ],
                asks: vec![
                    OrderbookLevel { price: 66100.0, size: 0.5 },
                    OrderbookLevel { price: 66200.0, size: 1.0 },
                    OrderbookLevel { price: 66300.0, size: 2.0 },
                ],
            }),
            trades: vec![
                Trade { side: TradeSide::Buy, size: 0.1, price: 66000.0, created_at: 1704067200.0 },
                Trade { side: TradeSide::Buy, size: 0.2, price: 66050.0, created_at: 1704067260.0 },
                Trade { side: TradeSide::Sell, size: 0.05, price: 66025.0, created_at: 1704067320.0 },
            ],
            candles: candles.clone(),
            funding: vec![
                FundingEntry { rate: "0.0001".into(), effective_at: "2025-01-01T00:00:00Z".into(), price: "66000".into() },
                FundingEntry { rate: "0.00015".into(), effective_at: "2025-01-01T01:00:00Z".into(), price: "66000".into() },
            ],
            market: Some(MarketInfo {
                ticker: "BTC-USD".into(),
                oracle_price: "66000".into(),
                open_interest: "500000000".into(),
                volume_24h: "1000000000".into(),
                next_funding_time: "2025-01-01T01:00:00Z".into(),
            }),
            stats: MarketStats {
                mid_price: 66050.0,
                spread: 100.0,
                volume_24h: 1_000_000_000.0,
                open_interest: 500_000_000.0,
                funding_rate_1h: 87.6, // 0.0001 * 24 * 365 * 100
            },
        }
    }

    #[test]
    fn test_all_agents_run() {
        let bundle = make_test_bundle();
        let votes = run_all_agents(&bundle);
        assert_eq!(votes.len(), 9);
        assert_eq!(votes[0].agent_type, "FundingAgent");
        assert_eq!(votes[8].agent_type, "SentimentAgent");
    }

    #[test]
    fn test_funding_agent_high_positive() {
        let mut bundle = make_test_bundle();
        bundle.stats.funding_rate_1h = 0.10; // >0.05
        let vote = funding_agent(&bundle);
        assert_eq!(vote.signal, Signal::Short);
        assert!(vote.confidence >= 45.0);
    }

    #[test]
    fn test_funding_agent_negative() {
        let mut bundle = make_test_bundle();
        bundle.stats.funding_rate_1h = -0.10;
        let vote = funding_agent(&bundle);
        assert_eq!(vote.signal, Signal::Long);
    }

    #[test]
    fn test_funding_agent_neutral() {
        let mut bundle = make_test_bundle();
        bundle.stats.funding_rate_1h = 0.005;
        let vote = funding_agent(&bundle);
        assert_eq!(vote.signal, Signal::Neutral);
    }

    #[test]
    fn test_funding_agent_empty() {
        let mut bundle = make_test_bundle();
        bundle.funding = vec![];
        bundle.stats.funding_rate_1h = 0.0;
        let vote = funding_agent(&bundle);
        assert_eq!(vote.signal, Signal::Neutral);
        assert_eq!(vote.confidence, 20.0);
    }

    #[test]
    fn test_momentum_agent_bullish() {
        let bundle = make_test_bundle(); // 20 green candles
        let vote = momentum_agent(&bundle);
        assert_eq!(vote.signal, Signal::Long);
    }

    #[test]
    fn test_momentum_agent_insufficient() {
        let mut bundle = make_test_bundle();
        bundle.candles = bundle.candles[..3].to_vec();
        let vote = momentum_agent(&bundle);
        assert_eq!(vote.confidence, 20.0);
    }

    #[test]
    fn test_volatility_agent_runs() {
        let bundle = make_test_bundle();
        let vote = volatility_agent(&bundle);
        assert!(vote.confidence > 0.0);
        assert!(!vote.reasoning.is_empty());
    }

    #[test]
    fn test_volume_agent_runs() {
        let bundle = make_test_bundle();
        let vote = volume_agent(&bundle);
        assert!(vote.confidence > 0.0);
    }

    #[test]
    fn test_orderbook_agent_runs() {
        let bundle = make_test_bundle();
        let vote = orderbook_agent(&bundle);
        assert!(vote.confidence > 0.0);
        assert!(vote.reasoning.contains("Bid depth"));
    }

    #[test]
    fn test_orderbook_agent_empty() {
        let mut bundle = make_test_bundle();
        bundle.orderbook = None;
        let vote = orderbook_agent(&bundle);
        assert_eq!(vote.confidence, 20.0);
    }

    #[test]
    fn test_liquidation_agent_runs() {
        let bundle = make_test_bundle();
        let vote = liquidation_agent(&bundle);
        assert!(vote.reasoning.contains("Liquidation risk score"));
    }

    #[test]
    fn test_mean_reversion_agent_runs() {
        let bundle = make_test_bundle();
        let vote = mean_reversion_agent(&bundle);
        assert!(vote.reasoning.contains("Z-score"));
    }

    #[test]
    fn test_mean_reversion_insufficient() {
        let mut bundle = make_test_bundle();
        bundle.candles = bundle.candles[..5].to_vec();
        let vote = mean_reversion_agent(&bundle);
        assert_eq!(vote.confidence, 20.0);
    }

    #[test]
    fn test_trend_agent_bullish_aligned() {
        let bundle = make_test_bundle(); // 20 consecutive green candles → all SMAs aligned up
        let vote = trend_agent(&bundle);
        assert_eq!(vote.signal, Signal::Long);
        assert_eq!(vote.confidence, 75.0);
    }

    #[test]
    fn test_trend_agent_insufficient() {
        let mut bundle = make_test_bundle();
        bundle.candles = bundle.candles[..5].to_vec();
        let vote = trend_agent(&bundle);
        assert_eq!(vote.confidence, 20.0);
    }

    #[test]
    fn test_sentiment_agent_empty() {
        let bundle = make_test_bundle();
        let vote = sentiment_agent(&bundle, &[]);
        assert_eq!(vote.confidence, 20.0);
    }

    #[test]
    fn test_sentiment_agent_divided() {
        let bundle = make_test_bundle();
        let votes = vec![
            AgentVote { agent_type: "A".into(), signal: Signal::Long, confidence: 60.0, reasoning: "".into() },
            AgentVote { agent_type: "B".into(), signal: Signal::Short, confidence: 60.0, reasoning: "".into() },
            AgentVote { agent_type: "C".into(), signal: Signal::Neutral, confidence: 40.0, reasoning: "".into() },
            AgentVote { agent_type: "D".into(), signal: Signal::Long, confidence: 55.0, reasoning: "".into() },
            AgentVote { agent_type: "E".into(), signal: Signal::Short, confidence: 55.0, reasoning: "".into() },
        ];
        let vote = sentiment_agent(&bundle, &votes);
        // divergence = |2-2|/5 = 0.0 < 0.2 → NEUTRAL
        assert_eq!(vote.signal, Signal::Neutral);
    }

    #[test]
    fn test_sentiment_agent_bullish_majority() {
        let bundle = make_test_bundle();
        let votes = vec![
            AgentVote { agent_type: "A".into(), signal: Signal::Long, confidence: 70.0, reasoning: "".into() },
            AgentVote { agent_type: "B".into(), signal: Signal::Long, confidence: 65.0, reasoning: "".into() },
            AgentVote { agent_type: "C".into(), signal: Signal::Long, confidence: 60.0, reasoning: "".into() },
            AgentVote { agent_type: "D".into(), signal: Signal::Short, confidence: 40.0, reasoning: "".into() },
            AgentVote { agent_type: "E".into(), signal: Signal::Neutral, confidence: 30.0, reasoning: "".into() },
        ];
        let vote = sentiment_agent(&bundle, &votes);
        assert_eq!(vote.signal, Signal::Long);
    }

    #[test]
    fn test_core_agents_length() {
        assert_eq!(CORE_AGENTS.len(), 8);
        assert_eq!(ALL_AGENT_TYPES.len(), 9);
    }
}
