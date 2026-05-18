# CubicZan ML

**Shared Rust ML layer for the Cubiczan AI / DeFi / Finance ecosystem.**

A high-performance, zero-dependency-on-Python machine learning library written in pure Rust. Provides the foundational ML infrastructure shared across all Cubiczan projects — from commodity price prediction and critical mineral traceability to on-chain inference and financial sentiment analysis.

## Architecture

```
cubiczan-ml/
├── Cargo.toml                          # Workspace root
├── crates/
│   ├── cubiczan-ml-core/               # Foundation layer
│   │   └── math · time_series · signal · risk · preprocessing · utils
│   ├── cubiczan-ml-nlp/                # Natural language processing
│   │   └── tokenizer · sentiment · classifier · ner · embeddings · summarizer
│   ├── cubiczan-ml-dl/                 # Deep learning (Candle)
│   │   └── models · inference · on_chain · time_series · training
│   ├── cubiczan-ml-rl/                 # Reinforcement learning
│   │   └── agents · environment · policy · exploration · backtest
│   ├── cubiczan-ml-tf/                 # TensorFlow bridge
│   │   └── session · bridge · models
│   ├── cubiczan-ml-py/                 # PyO3 Python bindings
│   │   └── core · nlp · dl · rl · tf
│   ├── finflowrl/                      # FinFlowRL Rust port
│   │   └── HFT flow-matching RL (PPO, market-making)
│   ├── critmin-oracle/                # CritMin Oracle Rust port
│   │   └── risk scoring · sentiment · scaling · keccak256
│   └── consensus-hardening-protocol/    # CHP Rust port
│       └── state machine · gates · devil's advocate · parity · contracts
```

## Crates

### `cubiczan-ml-core` — Foundation

Financial math, time series, trading signals, risk management, and data preprocessing. The bedrock every other crate builds on.

| Module | Highlights |
|--------|-----------|
| **math** | Moving averages (SMA/EMA/WMA/DEMA), Bollinger Bands, RSI, MACD, portfolio metrics, correlation, statistical tests |
| **time_series** | OHLCV candles, resampling, returns (log/simple), stationarity tests (ADF), seasonality detection, rolling stats |
| **signal** | Trading signal types, strength/confidence scoring, signal aggregation, consensus voting, conflict detection |
| **risk** | Kelly criterion, position sizing, Value-at-Risk, CVaR, max drawdown tracking, exposure limits, margin calculation |
| **preprocessing** | MinMax/Standard/Robust scalers, label encoders, train/test splits, feature engineering (lags, rolling stats), NaN handling |
| **utils** | Softmax, sigmoid, ReLU, one-hot encoding, MSE, clipping, dense parameter counting |
| **device** | Compute device enumeration (CPU, CUDA), device-aware dispatch |
| **error** | Unified `MlError` enum with `Result<T>` alias, serde/bincode interop |
| **metrics** | Training metrics (loss, accuracy, epoch, timing), serialization support |
| **normalization** | Online normalization stats with incremental mean/std computation |

### `cubiczan-ml-nlp` — Financial NLP

Text analysis specialized for financial documents — SEC filings, earnings calls, crypto social media, commodity reports.

| Module | Highlights |
|--------|-----------|
| **tokenizer** | HuggingFace tokenizers wrapper, financial-aware preprocessing, subword tokenization, padding/truncation, batch encode |
| **sentiment** | Sector-specific lexicons, Fed-speak decoder, emoji/emoticon handling, cashtag detection, confidence scoring, aggregate scoring |
| **classifier** | Zero-shot classification, multi-label pipeline, keyword-based, FinBERT-ready integration, thresholded confidence |
| **ner** | Named entity recognition for companies (`ORG`), currencies (`MONEY`), dates (`DATE`), percentages, SEC filing entities |
| **embeddings** | Sentence embeddings with in-memory cache, cosine similarity search, TF-IDF fallback, batch processing |
| **summarizer** | Extractive summarization (TextRank-style), abstractive hooks, sentence scoring, configurable length limits |

### `cubiczan-ml-dl` — Deep Learning

Neural network architectures and inference powered by [HuggingFace Candle](https://github.com/huggingface/candle) — pure Rust, no GPU required for inference.

| Module | Highlights |
|--------|-----------|
| **models** | LSTM, Transformer, Autoencoder, MLP, Conv1D architectures with configurable hyperparameters |
| **inference** | Fast inference engine, framework abstraction layer, batch prediction, model checkpointing |
| **on_chain** | Blockchain transaction analysis, on-chain ML inference, wallet behavior profiling, fraud detection |
| **time_series** | DL-based time series forecasting, feature normalization, sliding window datasets |
| **training** | Learning rate schedules (step decay, cosine annealing, warmup), early stopping, gradient clipping, Adam optimizer config |

### `cubiczan-ml-rl` — Reinforcement Learning

A complete RL framework for building autonomous trading agents. Train, evaluate, and backtest strategies in simulated market environments.

| Module | Highlights |
|--------|-----------|
| **agents** | Q-learning, Deep Q-Network (DQN), Policy Gradient, Actor-Critic, ensemble agents with weight averaging |
| **environment** | Simple trading (long/short/hold), portfolio management with multi-asset support, order book simulation, configurable commissions/slippage |
| **policy** | Kelly criterion, momentum, mean-reversion, risk parity, adaptive policy switching, policy chaining |
| **exploration** | Epsilon-greedy, Boltzmann softmax, UCB1, Thompson sampling, entropy-regularized exploration |
| **backtest** | Event-driven backtesting engine, equity curve tracking, trade logging, performance metrics (Sharpe, Sortino, max DD, win rate) |

### `cubiczan-ml-tf` — TensorFlow Bridge

Load and run Python-trained TensorFlow/Keras models from Rust. Bridges existing ML pipelines into the Cubiczan ecosystem without rewriting.

| Module | Highlights |
|--------|-----------|
| **session** | SavedModel and frozen graph loading, batch inference, session pooling for concurrency, inference stats tracking |
| **bridge** | PyTfBridge for importing Python-trained models, ONNX import/validation, auto-generated Rust wrapper code |
| **models** | Pre-built interfaces for TF LSTM, Transformer, Classifier, and Risk Model inference |

### `cubiczan-ml-py` — Python Bindings (PyO3)

Zero-copy Rust-to-Python bindings via PyO3 + maturin. Install with `pip install cubiczan-ml` to access all 38 ML APIs directly from Python — no Rust knowledge needed.

### `finflowrl` — FinFlowRL Rust Port

Complete Rust port of the FinFlowRL HFT flow-matching reinforcement learning system. Pure-numpy neural net and PPO trainer rewritten using `cubiczan-ml-rl` + `cubiczan-ml-dl` crates.

### `critmin-oracle` — CritMin Oracle Rust Port

AI-powered critical minerals supply chain risk oracle, rewritten from Python to Rust. Computes on-chain risk scores for lithium, nickel, and cobalt using sentiment analysis, regulatory keyword scoring, and price forecasting.

| Module | Highlights |
|--------|-----------|
| **config** | Mineral metadata, scaling constants (match Solidity contract), regulatory keyword weights |
| **scaling** | On-chain value scaling, keccak256 hashing (Solidity-compatible via sha3 crate) |
| **sentiment** | Keyword-based NLP sentiment analyzer for SEC filings, regulatory risk scorer |
| **forecast** | Price forecasting via linear regression on log prices, R-squared confidence |
| **prices** | Commodity price generation (mock) and Alpha Vantage API fetching (live mode) |
| **macro_data** | Macroeconomic indicator generation and FRED API fetching |
| **pipeline** | Full orchestration: composite risk scoring, demo/live modes, JSON output |

### `consensus-hardening-protocol` — CHP Rust Port

Complete Rust port of the Consensus Hardening Protocol decision-governance layer. Provides state machine, gates, adversarial validation, parity checks, payload envelopes, and orchestration for multi-agent AI systems.

| Module | Highlights |
|--------|-----------|
| **models** | Canonical data model: DecisionCase, SessionStatus, Verdict, Dossier, FoundationDisclosure/Attack, DevilsAdvocateRound |
| **gates** | R0 gate evaluation, phase gate enforcement (Foundation→Spec→Implementation) |
| **foundation** | Foundation disclosure validation, attack scoring, verdict computation |
| **parity** | Model tier inference (SMALL/MID/HIGH/FRONTIER), parity gap assessment |
| **devil** | Devil's advocate construction (Phase 0 + Round 3), VCL diagnosis, vulnerability merging |
| **payloads** | Payload envelope validation, payload ID generation, echo confirmation |
| **contracts** | Item agreement scoring, verification checklists, ASCII enforcement, council spawn |
| **registry** | Decision case registry with search, related-case finding, JSON persistence |
| **context** | Context engine with entity/event/task tracking, relevance scoring |
| **validators** | Third-party validation for lock progression (PROVISIONAL_LOCK→LOCKED) |
| **orchestrator** | Full CHP session orchestration: initial session, partner packet ingestion, report rendering |

## Quick Start

### Prerequisites

- Rust 1.80+ (tested on 1.95.0 stable)
- No Python runtime required

### Add as dependency

```toml
# In your Cargo.toml
[dependencies]
cubiczan-ml-core = { git = "https://github.com/Cubiczan/cubiczan-ml", branch = "main" }
cubiczan-ml-nlp  = { git = "https://github.com/Cubiczan/cubiczan-ml", branch = "main" }
cubiczan-ml-dl   = { git = "https://github.com/Cubiczan/cubiczan-ml", branch = "main" }
cubiczan-ml-rl   = { git = "https://github.com/Cubiczan/cubiczan-ml", branch = "main" }
cubiczan-ml-tf   = { git = "https://github.com/Cubiczan/cubiczan-ml", branch = "main" }
critmin-oracle   = { git = "https://github.com/Cubiczan/cubiczan-ml", branch = "main" }
consensus-hardening-protocol = { git = "https://github.com/Cubiczan/cubiczan-ml", branch = "main" }
```

### Build from source

```bash
git clone https://github.com/Cubiczan/cubiczan-ml.git
cd cubiczan-ml
cargo build
cargo test
```

### Usage examples

```rust
use cubiczan_ml_core::{
    math::{MovingAverage, MovingAverageType},
    time_series::OhlcvCandle,
    risk::KellyCriterion,
    preprocessing::MinMaxScaler,
};

// Compute moving averages
let prices = vec![100.0, 102.0, 101.0, 103.0, 105.0, 104.0, 106.0];
let sma = MovingAverage::compute(&prices, 3, MovingAverageType::SMA);
let ema = MovingAverage::compute(&prices, 3, MovingAverageType::EMA);

// Kelly criterion position sizing
let kelly = KellyCriterion::new(0.6, 2.0);
let fraction = kelly.compute_fraction();

// Scale features for ML
let mut scaler = MinMaxScaler::new();
let scaled = scaler.fit_transform(&data)?;
```

```rust
use cubiczan_ml_nlp::{
    sentiment::FinSentimentAnalyzer,
    tokenizer::FinTokenizer,
    classifier::TextClassifier,
};

// Analyze financial sentiment
let analyzer = FinSentimentAnalyzer::new();
let result = analyzer.analyze("Fed signals potential rate cut in Q3")?;
println!("Sentiment: {:?} (confidence: {:.2})", result.label, result.confidence);
```

```rust
use cubiczan_ml_rl::{
    agents::QLearningAgent,
    environment::SimpleTradingEnv,
    exploration::EpsilonGreedy,
};

// Build a trading agent
let env = SimpleTradingEnv::new(prices, 100_000.0);
let exploration = EpsilonGreedy::new(0.1, 0.995, 1000);
let mut agent = QLearningAgent::new(
    env.state_size(),
    env.action_count(),
    0.1,    // learning rate
    0.99,   // discount factor
    exploration,
);

// Train
for episode in 0..500 {
    let mut state = env.reset();
    let mut total_reward = 0.0;
    loop {
        let action = agent.select_action(&state);
        let (next_state, reward, done) = env.step(action);
        agent.update(&state, action, reward, &next_state, done);
        state = next_state;
        total_reward += reward;
        if done { break; }
    }
}
```

## Key Dependencies

| Crate | Purpose |
|-------|---------|
| `ndarray` / `nalgebra` | n-dimensional arrays and linear algebra |
| `candle-core` / `candle-nn` | Pure-Rust deep learning (HuggingFace) |
| `tokenizers` | HuggingFace tokenizers (bindings) |
| `serde` / `serde_json` | Serialization framework |
| `statrs` | Statistical distributions and tests |
| `tracing` | Structured logging and diagnostics |
| `anyhow` / `thiserror` | Ergonomic error handling |
| `chrono` | Date/time for financial time series |
| `rand` | RNG for exploration strategies |
| `sha3` / `hex` | Keccak256 hashing (Solidity-compatible) |
| `reqwest` / `tokio` | Async HTTP for API fetching (FRED, Alpha Vantage) |

## Stats

| Metric | Value |
|--------|-------|
| Total lines of Rust | ~23,000+ |
| Source files | 65+ |
| Crates | 9 |
| Tests passing | **403 / 403** |
| Build errors | **0** |
| Minimum Rust version | 1.80+ (tested 1.95.0) |

## Integration Targets

This shared ML layer is designed to be integrated into the following Cubiczan ecosystem projects:

- **Commodity-Price-Analyzer** — Price prediction and signal generation
- **closed-loop-finance** — Autonomous finance loop with ML-driven decisions
- **FinFlowRL** — RL-based trading strategies (Rust port complete)
- **critmin-oracle** — Critical minerals blockchain risk oracle (Rust port complete)
- **minescope-signal** — Mining signal processing and anomaly detection
- **sec-earnings-workbench** — SEC filing NLP analysis
- **Stellar-critical-metal-traceability** — Supply chain traceability ML
- **consensus-hardening-protocol** — Multi-agent decision governance (Rust port complete)
- **courtvision-ai** — Sports analytics with ML
- **greenverify-ai** — ESG verification and scoring

## License

MIT

## Author

Shyam Desigan &lt;sam@cubiczan.com&gt;

---

Built with [Candle](https://github.com/huggingface/candle), [ndarray](https://github.com/rust-ndarray/ndarray), and [tokenizers](https://github.com/huggingface/tokenizers).
