//! Core type definitions for the scope-sentinel REIT analytics platform.
//!
//! Defines domain models for REITs, financial statements, portfolio positions,
//! trading signals, and risk ratings used across all modules.

use serde::{Deserialize, Serialize};
use chrono::{DateTime, Utc};

/// REIT sector classification aligned with NAREIT/GICS taxonomy.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum REITSector {
    Residential,
    Commercial,
    Healthcare,
    Industrial,
    DataCenter,
    Retail,
    Office,
    Lodging,
    SelfStorage,
    Specialty,
    Mortgage,
    Infrastructure,
}

impl REITSector {
    /// Returns a human-readable label for the sector.
    pub fn label(&self) -> &'static str {
        match self {
            Self::Residential => "Residential",
            Self::Commercial => "Commercial",
            Self::Healthcare => "Healthcare",
            Self::Industrial => "Industrial",
            Self::DataCenter => "Data Center",
            Self::Retail => "Retail",
            Self::Office => "Office",
            Self::Lodging => "Lodging / Resort",
            Self::SelfStorage => "Self-Storage",
            Self::Specialty => "Specialty",
            Self::Mortgage => "Mortgage (mREIT)",
            Self::Infrastructure => "Infrastructure",
        }
    }

    /// Returns all sector variants.
    pub fn all() -> &'static [REITSector] {
        &[
            Self::Residential,
            Self::Commercial,
            Self::Healthcare,
            Self::Industrial,
            Self::DataCenter,
            Self::Retail,
            Self::Office,
            Self::Lodging,
            Self::SelfStorage,
            Self::Specialty,
            Self::Mortgage,
            Self::Infrastructure,
        ]
    }
}

impl std::fmt::Display for REITSector {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.label())
    }
}

/// Core REIT entity representing a single real estate investment trust.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct REIT {
    /// Ticker symbol (e.g., "O", "AMT", "PSA").
    pub ticker: String,
    /// Full legal name.
    pub name: String,
    /// CIK number assigned by the SEC.
    pub cik: String,
    /// Primary sector classification.
    pub sector: REITSector,
    /// Date the company was incorporated / IPO'd.
    pub inception_date: Option<DateTime<Utc>>,
    /// Total market capitalization in USD.
    pub market_cap: f64,
    /// Latest share price in USD.
    pub share_price: f64,
    /// Number of shares outstanding.
    pub shares_outstanding: f64,
    /// Most recent balance sheet.
    pub balance_sheet: Option<BalanceSheet>,
    /// Most recent income statement.
    pub income_statement: Option<IncomeStatement>,
    /// Computed financial ratios.
    pub ratios: Option<FinancialRatios>,
}

/// Balance sheet financial data (in thousands USD unless noted).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BalanceSheet {
    pub period_end: DateTime<Utc>,
    /// Total real estate assets (gross).
    pub real_estate_assets: f64,
    /// Accumulated depreciation on real estate.
    pub accumulated_depreciation: f64,
    /// Net real estate assets.
    pub net_real_estate_assets: f64,
    /// Total assets.
    pub total_assets: f64,
    /// Total current assets.
    pub current_assets: f64,
    /// Total liabilities.
    pub total_liabilities: f64,
    /// Total current liabilities.
    pub current_liabilities: f64,
    /// Mortgage debt (secured).
    pub mortgage_debt: f64,
    /// Unsecured debt / senior notes.
    pub unsecured_debt: f64,
    /// Total debt (mortgage + unsecured + other).
    pub total_debt: f64,
    /// Total stockholders' equity.
    pub shareholders_equity: f64,
    /// Cash and cash equivalents.
    pub cash: f64,
    /// Restricted cash.
    pub restricted_cash: f64,
    /// Total number of shares outstanding at period end.
    pub shares_outstanding: f64,
}

/// Income statement data (in thousands USD unless noted).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct IncomeStatement {
    pub period_start: DateTime<Utc>,
    pub period_end: DateTime<Utc>,
    /// Total rental revenue.
    pub rental_revenue: f64,
    /// Total revenue (rental + other).
    pub total_revenue: f64,
    /// Same-store NOI change (%).
    pub same_store_noi_growth: Option<f64>,
    /// Cost of operations / property operating expenses.
    pub operating_expenses: f64,
    /// Net Operating Income (NOI).
    pub noi: f64,
    /// Depreciation and amortization.
    pub depreciation_amortization: f64,
    /// General and administrative expenses.
    pub general_admin_expenses: f64,
    /// Interest expense (total).
    pub interest_expense: f64,
    /// Interest income.
    pub interest_income: Option<f64>,
    /// Earnings before interest and taxes.
    pub ebit: f64,
    /// Income tax expense.
    pub income_tax_expense: f64,
    /// Net income attributable to common shareholders.
    pub net_income: f64,
    /// Gains / losses on property dispositions.
    pub gains_losses_on_sales: Option<f64>,
    /// Funds From Operations (FFO) per share.
    pub ffo_per_share: Option<f64>,
    /// Dividends declared per share.
    pub dividends_per_share: Option<f64>,
    /// Weighted average diluted shares outstanding.
    pub weighted_avg_shares: Option<f64>,
}

/// Computed financial ratios and metrics for a REIT.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FinancialRatios {
    pub computed_at: DateTime<Utc>,
    /// Funds From Operations (FFO) in millions.
    pub ffo: f64,
    /// FFO per share.
    pub ffo_per_share: f64,
    /// Adjusted FFO (AFFO) per share.
    pub affo_per_share: f64,
    /// Net Asset Value (NAV) per share.
    pub nav_per_share: f64,
    /// Price / FFO ratio.
    pub price_to_ffo: f64,
    /// Dividend yield (annualized).
    pub dividend_yield: f64,
    /// Debt to EBITDA ratio.
    pub debt_to_ebitda: f64,
    /// Interest coverage ratio (EBIT / Interest Expense).
    pub interest_coverage: f64,
    /// Capitalization rate (NOI / Property Value).
    pub cap_rate: f64,
    /// Same-store NOI growth (%).
    pub same_store_noi_growth: f64,
    /// Current ratio (Current Assets / Current Liabilities).
    pub current_ratio: f64,
    /// Debt to equity ratio.
    pub debt_to_equity: f64,
    /// Return on equity (%).
    pub return_on_equity: f64,
    /// Operating margin (%).
    pub operating_margin: f64,
}

/// A single position held within a REIT portfolio.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PortfolioPosition {
    /// Ticker symbol of the REIT.
    pub ticker: String,
    /// REIT sector.
    pub sector: REITSector,
    /// Number of shares held.
    pub shares: f64,
    /// Average cost basis per share.
    pub cost_basis: f64,
    /// Current market price per share.
    pub current_price: f64,
    /// Portfolio weight (0.0 to 1.0).
    pub weight: f64,
    /// Total position value (shares * current_price).
    pub market_value: f64,
    /// Unrealized P&L.
    pub unrealized_pnl: f64,
    /// Unrealized return (%).
    pub unrealized_return_pct: f64,
    /// Computed financial ratios (if available).
    pub ratios: Option<FinancialRatios>,
}

/// Trading signal generated for a specific REIT.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum Signal {
    Buy,
    Sell,
    Hold,
}

impl Signal {
    pub fn label(&self) -> &'static str {
        match self {
            Self::Buy => "BUY",
            Self::Sell => "SELL",
            Self::Hold => "HOLD",
        }
    }

    /// Numerical score for composite scoring: Buy=1, Hold=0, Sell=-1.
    pub fn score(&self) -> i32 {
        match self {
            Self::Buy => 1,
            Self::Hold => 0,
            Self::Sell => -1,
        }
    }
}

impl std::fmt::Display for Signal {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.label())
    }
}

/// Risk rating classification for portfolio and individual positions.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
pub enum RiskRating {
    VeryLow,
    Low,
    Medium,
    High,
    VeryHigh,
}

impl RiskRating {
    /// Numeric risk score: 1 (VeryLow) to 5 (VeryHigh).
    pub fn score(&self) -> u8 {
        match self {
            Self::VeryLow => 1,
            Self::Low => 2,
            Self::Medium => 3,
            Self::High => 4,
            Self::VeryHigh => 5,
        }
    }

    /// Parse from numeric score.
    pub fn from_score(score: u8) -> Option<Self> {
        match score {
            1 => Some(Self::VeryLow),
            2 => Some(Self::Low),
            3 => Some(Self::Medium),
            4 => Some(Self::High),
            5 => Some(Self::VeryHigh),
            _ => None,
        }
    }
}

impl std::fmt::Display for RiskRating {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let s = match self {
            Self::VeryLow => "Very Low",
            Self::Low => "Low",
            Self::Medium => "Medium",
            Self::High => "High",
            Self::VeryHigh => "Very High",
        };
        write!(f, "{}", s)
    }
}

/// A detailed signal result with rationale and confidence.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SignalResult {
    pub ticker: String,
    pub signal: Signal,
    pub confidence: f64,      // 0.0 to 1.0
    pub composite_score: f64, // weighted score from multiple factors
    pub rationale: String,
    pub generated_at: DateTime<Utc>,
}

/// Portfolio-level analytics summary.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PortfolioAnalytics {
    pub total_value: f64,
    pub total_cost_basis: f64,
    pub total_pnl: f64,
    pub total_return_pct: f64,
    pub sector_weights: std::collections::HashMap<REITSector, f64>,
    pub diversification_score: f64, // 0.0 to 1.0 (Herfindahl-based)
    pub risk_rating: RiskRating,
    pub weighted_avg_dividend_yield: f64,
    pub weighted_avg_ffo_yield: f64,
    pub weighted_avg_debt_to_ebitda: f64,
    pub sharpe_ratio: Option<f64>,
    pub sortino_ratio: Option<f64>,
    pub treynor_ratio: Option<f64>,
}

/// Cash flow statement data.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CashFlowStatement {
    pub period_start: DateTime<Utc>,
    pub period_end: DateTime<Utc>,
    /// Net income (starting line).
    pub net_income: f64,
    /// Depreciation and amortization (add-back).
    pub depreciation_amortization: f64,
    /// Gains on property sales (deducted).
    pub gains_on_sales: f64,
    /// Operating cash flow.
    pub operating_cash_flow: f64,
    /// Capital expenditures (acquisitions, developments).
    pub capital_expenditures: f64,
    /// Cash from property dispositions.
    pub property_dispositions: f64,
    /// Investing cash flow.
    pub investing_cash_flow: f64,
    /// Debt issued.
    pub debt_issued: f64,
    /// Debt repaid.
    pub debt_repaid: f64,
    /// Dividends paid.
    pub dividends_paid: f64,
    /// Financing cash flow.
    pub financing_cash_flow: f64,
    /// Net change in cash.
    pub net_change_in_cash: f64,
}

/// Backtest trade record.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BacktestTrade {
    pub ticker: String,
    pub entry_date: DateTime<Utc>,
    pub exit_date: DateTime<Utc>,
    pub entry_price: f64,
    pub exit_price: f64,
    pub shares: f64,
    pub signal: Signal,
    pub return_pct: f64,
    pub holding_days: i64,
}

/// Backtest summary statistics.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BacktestSummary {
    pub total_trades: usize,
    pub winning_trades: usize,
    pub losing_trades: usize,
    pub win_rate: f64,
    pub avg_return_pct: f64,
    pub best_return_pct: f64,
    pub worst_return_pct: f64,
    pub total_return_pct: f64,
    pub max_drawdown_pct: f64,
    pub sharpe_ratio: f64,
    pub avg_holding_days: f64,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_reit_sector_labels() {
        assert_eq!(REITSector::Residential.label(), "Residential");
        assert_eq!(REITSector::DataCenter.label(), "Data Center");
        assert_eq!(REITSector::Mortgage.label(), "Mortgage (mREIT)");
    }

    #[test]
    fn test_reit_sector_all_count() {
        let all = REITSector::all();
        assert_eq!(all.len(), 12);
    }

    #[test]
    fn test_reit_sector_display() {
        assert_eq!(format!("{}", REITSector::Industrial), "Industrial");
    }

    #[test]
    fn test_signal_score() {
        assert_eq!(Signal::Buy.score(), 1);
        assert_eq!(Signal::Hold.score(), 0);
        assert_eq!(Signal::Sell.score(), -1);
    }

    #[test]
    fn test_signal_display() {
        assert_eq!(format!("{}", Signal::Buy), "BUY");
        assert_eq!(format!("{}", Signal::Sell), "SELL");
        assert_eq!(format!("{}", Signal::Hold), "HOLD");
    }

    #[test]
    fn test_risk_rating_score() {
        assert_eq!(RiskRating::VeryLow.score(), 1);
        assert_eq!(RiskRating::Medium.score(), 3);
        assert_eq!(RiskRating::VeryHigh.score(), 5);
    }

    #[test]
    fn test_risk_rating_from_score() {
        assert_eq!(RiskRating::from_score(1), Some(RiskRating::VeryLow));
        assert_eq!(RiskRating::from_score(3), Some(RiskRating::Medium));
        assert_eq!(RiskRating::from_score(5), Some(RiskRating::VeryHigh));
        assert_eq!(RiskRating::from_score(0), None);
        assert_eq!(RiskRating::from_score(6), None);
    }

    #[test]
    fn test_risk_rating_ord() {
        assert!(RiskRating::Low < RiskRating::High);
        assert!(RiskRating::VeryLow <= RiskRating::VeryLow);
    }

    #[test]
    fn test_portfolio_position_pnl() {
        let pos = PortfolioPosition {
            ticker: "O".into(),
            sector: REITSector::Residential,
            shares: 100.0,
            cost_basis: 50.0,
            current_price: 60.0,
            weight: 0.25,
            market_value: 6000.0,
            unrealized_pnl: 1000.0,
            unrealized_return_pct: 20.0,
            ratios: None,
        };
        assert_eq!(pos.unrealized_pnl, 1000.0);
        assert_eq!(pos.unrealized_return_pct, 20.0);
    }

    #[test]
    fn test_signal_result_creation() {
        let result = SignalResult {
            ticker: "AMT".into(),
            signal: Signal::Buy,
            confidence: 0.85,
            composite_score: 0.72,
            rationale: "Strong AFFO growth".into(),
            generated_at: Utc::now(),
        };
        assert_eq!(result.signal, Signal::Buy);
        assert!((result.confidence - 0.85).abs() < 1e-9);
    }

    #[test]
    fn test_balance_sheet_defaults() {
        let bs = BalanceSheet {
            period_end: Utc::now(),
            real_estate_assets: 1_000_000.0,
            accumulated_depreciation: 200_000.0,
            net_real_estate_assets: 800_000.0,
            total_assets: 1_200_000.0,
            current_assets: 100_000.0,
            total_liabilities: 600_000.0,
            current_liabilities: 50_000.0,
            mortgage_debt: 400_000.0,
            unsecured_debt: 100_000.0,
            total_debt: 500_000.0,
            shareholders_equity: 600_000.0,
            cash: 80_000.0,
            restricted_cash: 20_000.0,
            shares_outstanding: 50_000_000.0,
        };
        assert_eq!(bs.net_real_estate_assets, bs.real_estate_assets - bs.accumulated_depreciation);
    }

    #[test]
    fn test_financial_ratios_serde_roundtrip() {
        let ratios = FinancialRatios {
            computed_at: Utc::now(),
            ffo: 500.0,
            ffo_per_share: 5.0,
            affo_per_share: 4.5,
            nav_per_share: 80.0,
            price_to_ffo: 16.0,
            dividend_yield: 0.04,
            debt_to_ebitda: 5.2,
            interest_coverage: 4.0,
            cap_rate: 0.055,
            same_store_noi_growth: 0.03,
            current_ratio: 2.0,
            debt_to_equity: 0.8,
            return_on_equity: 0.08,
            operating_margin: 0.45,
        };
        let json = serde_json::to_string(&ratios).unwrap();
        let decoded: FinancialRatios = serde_json::from_str(&json).unwrap();
        assert!((decoded.affo_per_share - ratios.affo_per_share).abs() < 1e-9);
        assert!((decoded.cap_rate - 0.055).abs() < 1e-9);
    }

    #[test]
    fn test_reit_serde_roundtrip() {
        let reit = REIT {
            ticker: "PSA".into(),
            name: "Public Storage".into(),
            cik: "0000732987".into(),
            sector: REITSector::SelfStorage,
            inception_date: None,
            market_cap: 50_000_000_000.0,
            share_price: 350.0,
            shares_outstanding: 142_857_142.0,
            balance_sheet: None,
            income_statement: None,
            ratios: None,
        };
        let json = serde_json::to_string(&reit).unwrap();
        let decoded: REIT = serde_json::from_str(&json).unwrap();
        assert_eq!(decoded.ticker, "PSA");
        assert_eq!(decoded.sector, REITSector::SelfStorage);
    }

    #[test]
    fn test_cash_flow_statement() {
        let cf = CashFlowStatement {
            period_start: Utc::now(),
            period_end: Utc::now(),
            net_income: 300_000.0,
            depreciation_amortization: 150_000.0,
            gains_on_sales: 20_000.0,
            operating_cash_flow: 430_000.0,
            capital_expenditures: 200_000.0,
            property_dispositions: 50_000.0,
            investing_cash_flow: -150_000.0,
            debt_issued: 100_000.0,
            debt_repaid: 80_000.0,
            dividends_paid: 200_000.0,
            financing_cash_flow: -180_000.0,
            net_change_in_cash: 100_000.0,
        };
        // Verify cash flow identity
        let check = cf.operating_cash_flow + cf.investing_cash_flow + cf.financing_cash_flow;
        assert!((check - cf.net_change_in_cash).abs() < 1.0);
    }

    #[test]
    fn test_backtest_summary() {
        let summary = BacktestSummary {
            total_trades: 100,
            winning_trades: 55,
            losing_trades: 45,
            win_rate: 0.55,
            avg_return_pct: 1.2,
            best_return_pct: 25.0,
            worst_return_pct: -15.0,
            total_return_pct: 120.0,
            max_drawdown_pct: 12.0,
            sharpe_ratio: 1.5,
            avg_holding_days: 45.0,
        };
        assert_eq!(summary.winning_trades + summary.losing_trades, summary.total_trades);
        assert!((summary.win_rate - 0.55).abs() < 1e-9);
    }
}
