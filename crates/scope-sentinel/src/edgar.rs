//! SEC EDGAR filing parser for XBRL/XML financial data extraction.
//!
//! Provides parsing utilities for extracting balance sheets, income statements,
//! and cash flow data from SEC filings. Handles XBRL tag extraction, filing
//! metadata, and data normalization.

use chrono::{DateTime, Utc};
use thiserror::Error;
use crate::types::{
    BalanceSheet, CashFlowStatement, IncomeStatement, REIT, REITSector,
};

/// Errors that can occur during EDGAR filing parsing.
#[derive(Debug, Error)]
pub enum EdgarError {
    #[error("Missing required XBRL tag: {tag}")]
    MissingTag { tag: String },

    #[error("Failed to parse numeric value for tag '{tag}': {raw}")]
    ParseError { tag: String, raw: String },

    #[error("Filing period metadata missing or invalid")]
    InvalidPeriod,

    #[error("XML structure error: {0}")]
    XmlError(String),

    #[error("Invalid CIK format: {0}")]
    InvalidCik(String),
}

/// Metadata extracted from an SEC EDGAR filing.
#[derive(Debug, Clone)]
pub struct FilingMetadata {
    pub accession_number: String,
    pub filing_type: String,
    pub filing_date: DateTime<Utc>,
    pub period_end: DateTime<Utc>,
    pub period_start: DateTime<Utc>,
    pub fiscal_year: u32,
    pub fiscal_period: String,
    pub cik: String,
    pub entity_name: String,
    pub sic_code: Option<String>,
    pub document_type: String,
    pub is_amended: bool,
}

/// Parsed XBRL context with period and entity information.
#[derive(Debug, Clone)]
pub struct XbrlContext {
    pub entity_id: String,
    pub period_start: DateTime<Utc>,
    pub period_end: DateTime<Utc>,
    pub is_instant: bool,
    pub dimensions: Vec<(String, String)>,
}

/// A single XBRL fact extracted from a filing.
#[derive(Debug, Clone)]
pub struct XbrlFact {
    pub tag: String,
    pub value: String,
    pub context_ref: String,
    pub unit_ref: Option<String>,
    pub decimals: Option<i32>,
}

/// Result of parsing a complete filing.
#[derive(Debug)]
pub struct ParsedFiling {
    pub metadata: FilingMetadata,
    pub facts: Vec<XbrlFact>,
    pub contexts: Vec<XbrlContext>,
    pub balance_sheet: Option<BalanceSheet>,
    pub income_statement: Option<IncomeStatement>,
    pub cash_flow: Option<CashFlowStatement>,
}

/// Common XBRL tags used in REIT financial reporting (US-GAAP / REIT taxonomy).
pub struct XbrlTags;

impl XbrlTags {
    // Balance Sheet tags
    pub const REAL_ESTATE_ASSETS: &'static str = "RealEstateAssets";
    pub const ACCUMULATED_DEPRECIATION: &'static str = "AccumulatedDepreciationAmortizationPropertyPlantAndEquipment";
    pub const TOTAL_ASSETS: &'static str = "Assets";
    pub const CURRENT_ASSETS: &'static str = "AssetsCurrent";
    pub const TOTAL_LIABILITIES: &'static str = "Liabilities";
    pub const CURRENT_LIABILITIES: &'static str = "LiabilitiesCurrent";
    pub const MORTGAGE_DEBT: &'static str = "MortgageAndOtherDebtSecured";
    pub const SECURED_DEBT: &'static str = "LongTermDebtSecured";
    pub const UNSCURED_DEBT: &'static str = "LongTermDebtUnsecured";
    pub const LONG_TERM_DEBT: &'static str = "LongTermDebt";
    pub const SHORT_TERM_DEBT: &'static str = "ShortTermBorrowings";
    pub const STOCKHOLDERS_EQUITY: &'static str = "StockholdersEquity";
    pub const CASH: &'static str = "CashAndCashEquivalentsAtCarryingValue";
    pub const RESTRICTED_CASH: &'static str = "CashAndCashEquivalentsRestrictedCashAndRestrictedCashEquivalents";

    // Income Statement tags
    pub const RENTAL_REVENUE: &'static str = "RentalRevenue";
    pub const TOTAL_REVENUE: &'static str = "Revenues";
    pub const OPERATING_EXPENSES: &'static str = "OperatingExpenses";
    pub const NOI: &'static str = "NetOperatingIncomeLoss";
    pub const DEPRECIATION: &'static str = "DepreciationAndAmortization";
    pub const G_AND_A: &'static str = "GeneralAndAdministrativeExpense";
    pub const INTEREST_EXPENSE: &'static str = "InterestExpense";
    pub const INTEREST_INCOME: &'static str = "InterestIncome";
    pub const EBIT: &'static str = "OperatingIncomeLoss";
    pub const INCOME_TAX: &'static str = "IncomeTaxExpenseBenefit";
    pub const NET_INCOME: &'static str = "NetIncomeLoss";
    pub const FFO_PER_SHARE: &'static str = "FundsFromOperationsPerShare";
    pub const DIVIDENDS_PER_SHARE: &'static str = "CommonStockDividendsPerShareCashPaid";
    pub const WEIGHTED_AVG_SHARES: &'static str = "WeightedAverageNumberOfDilutedSharesOutstanding";

    // Cash Flow tags
    pub const CAPITAL_EXPENDITURES: &'static str = "PaymentsForAcquisitionOfPropertyPlantAndEquipment";
    pub const PROPERTY_DISPOSITIONS: &'static str = "ProceedsFromSaleOfPropertyPlantAndEquipment";
    pub const DEBT_ISSUED: &'static str = "ProceedsFromIssuanceOfLongTermDebt";
    pub const DEBT_REPAID: &'static str = "PaymentsForRepaymentOfLongTermDebt";
    pub const DIVIDENDS_PAID: &'static str = "PaymentsOfDividends";
}

/// Parse a numeric string from XBRL, handling commas, parentheses for negatives, etc.
pub fn parse_xbrl_numeric(raw: &str) -> Result<f64, EdgarError> {
    let trimmed = raw.trim();
    if trimmed.is_empty() || trimmed == "-" {
        return Ok(0.0);
    }
    // Remove commas
    let cleaned: String = trimmed.chars().filter(|c| *c != ',').collect();
    // Handle parenthesized negatives like "(1,234.56)"
    if cleaned.starts_with('(') && cleaned.ends_with(')') {
        let inner = &cleaned[1..cleaned.len() - 1];
        return inner
            .parse::<f64>()
            .map(|v| -v)
            .map_err(|_| EdgarError::ParseError {
                tag: "unknown".into(),
                raw: raw.into(),
            });
    }
    cleaned.parse::<f64>().map_err(|_| EdgarError::ParseError {
        tag: "unknown".into(),
        raw: raw.into(),
    })
}

/// Extract a numeric fact value from the facts list by tag name.
pub fn extract_fact(facts: &[XbrlFact], tag: &str) -> Result<f64, EdgarError> {
    let raw_tag = tag.to_lowercase();
    for fact in facts.iter().rev() {
        if fact.tag.to_lowercase() == raw_tag {
            return parse_xbrl_numeric(&fact.value).map_err(|e| EdgarError::ParseError {
                tag: tag.into(),
                raw: fact.value.clone(),
            });
        }
    }
    Err(EdgarError::MissingTag { tag: tag.into() })
}

/// Extract a numeric fact, returning a default if the tag is missing.
pub fn extract_fact_or_default(facts: &[XbrlFact], tag: &str, default: f64) -> f64 {
    extract_fact(facts, tag).unwrap_or(default)
}

/// Parse a CIK string, normalizing to 10-digit zero-padded format.
pub fn normalize_cik(raw: &str) -> Result<String, EdgarError> {
    let digits: String = raw.chars().filter(|c| c.is_ascii_digit()).collect();
    if digits.is_empty() {
        return Err(EdgarError::InvalidCik(raw.into()));
    }
    let num: u64 = digits.parse().map_err(|_| EdgarError::InvalidCik(raw.into()))?;
    Ok(format!("{:010}", num))
}

/// Parse filing type from document description (e.g., "10-K", "10-K/A").
pub fn parse_filing_type(raw: &str) -> (String, bool) {
    let trimmed = raw.trim().to_uppercase();
    let is_amended = trimmed.ends_with("/A");
    let base = if is_amended {
        trimmed.trim_end_matches("/A")
    } else {
        &trimmed
    };
    (base.to_string(), is_amended)
}

/// Normalize financial values from various scales (millions, thousands, etc.)
/// to a target scale. SEC filings typically report in thousands.
pub fn normalize_value(value: f64, _scale: u32) -> f64 {
    // SEC EDGAR XBRL data is typically in thousands USD.
    // This function can be extended for different reporting scales.
    value
}

/// Build a BalanceSheet from extracted facts.
pub fn build_balance_sheet(
    facts: &[XbrlFact],
    period_end: DateTime<Utc>,
    shares_outstanding: f64,
) -> Result<BalanceSheet, EdgarError> {
    let real_estate_assets = extract_fact_or_default(facts, XbrlTags::REAL_ESTATE_ASSETS, 0.0);
    let accumulated_dep = extract_fact_or_default(facts, XbrlTags::ACCUMULATED_DEPRECIATION, 0.0);

    let total_assets = extract_fact(facts, XbrlTags::TOTAL_ASSETS)
        .or_else(|_| {
            extract_fact_or_default(facts, "Assets", 0.0);
            extract_fact(facts, "Assets")
        })
        .unwrap_or(0.0);

    let current_assets = extract_fact_or_default(facts, XbrlTags::CURRENT_ASSETS, 0.0);
    let total_liabilities = extract_fact_or_default(facts, XbrlTags::TOTAL_LIABILITIES, 0.0);
    let current_liabilities = extract_fact_or_default(facts, XbrlTags::CURRENT_LIABILITIES, 0.0);

    let mortgage_debt = extract_fact_or_default(facts, XbrlTags::MORTGAGE_DEBT, 0.0);
    let secured_debt = extract_fact_or_default(facts, XbrlTags::SECURED_DEBT, 0.0);
    let unsecured_debt = extract_fact_or_default(facts, XbrlTags::UNSCURED_DEBT, 0.0);
    let long_term_debt = extract_fact_or_default(facts, XbrlTags::LONG_TERM_DEBT, 0.0);
    let short_term_debt = extract_fact_or_default(facts, XbrlTags::SHORT_TERM_DEBT, 0.0);

    let total_debt = mortgage_debt.max(secured_debt) + unsecured_debt + long_term_debt + short_term_debt;
    let equity = total_assets - total_liabilities;
    let cash = extract_fact_or_default(facts, XbrlTags::CASH, 0.0);
    let restricted_cash = extract_fact_or_default(facts, XbrlTags::RESTRICTED_CASH, 0.0);

    Ok(BalanceSheet {
        period_end,
        real_estate_assets,
        accumulated_depreciation: accumulated_dep,
        net_real_estate_assets: (real_estate_assets - accumulated_dep).max(0.0),
        total_assets,
        current_assets,
        total_liabilities,
        current_liabilities,
        mortgage_debt,
        unsecured_debt,
        total_debt,
        shareholders_equity: equity,
        cash,
        restricted_cash,
        shares_outstanding,
    })
}

/// Build an IncomeStatement from extracted facts.
pub fn build_income_statement(
    facts: &[XbrlFact],
    period_start: DateTime<Utc>,
    period_end: DateTime<Utc>,
) -> Result<IncomeStatement, EdgarError> {
    let rental_revenue = extract_fact_or_default(facts, XbrlTags::RENTAL_REVENUE, 0.0);
    let total_revenue = extract_fact_or_default(facts, XbrlTags::TOTAL_REVENUE, 0.0);
    let operating_expenses = extract_fact_or_default(facts, XbrlTags::OPERATING_EXPENSES, 0.0);
    let noi = extract_fact_or_default(facts, XbrlTags::NOI, 0.0);
    let depreciation = extract_fact_or_default(facts, XbrlTags::DEPRECIATION, 0.0);
    let ga = extract_fact_or_default(facts, XbrlTags::G_AND_A, 0.0);
    let interest_expense = extract_fact_or_default(facts, XbrlTags::INTEREST_EXPENSE, 0.0);
    let interest_income = extract_fact(facts, XbrlTags::INTEREST_INCOME).ok();
    let ebit = extract_fact_or_default(facts, XbrlTags::EBIT, 0.0);
    let income_tax = extract_fact_or_default(facts, XbrlTags::INCOME_TAX, 0.0);
    let net_income = extract_fact_or_default(facts, XbrlTags::NET_INCOME, 0.0);
    let ffo_ps = extract_fact(facts, XbrlTags::FFO_PER_SHARE).ok();
    let divs_ps = extract_fact(facts, XbrlTags::DIVIDENDS_PER_SHARE).ok();
    let wavg = extract_fact(facts, XbrlTags::WEIGHTED_AVG_SHARES).ok();

    Ok(IncomeStatement {
        period_start,
        period_end,
        rental_revenue,
        total_revenue,
        same_store_noi_growth: None, // Not directly in XBRL
        operating_expenses,
        noi,
        depreciation_amortization: depreciation,
        general_admin_expenses: ga,
        interest_expense,
        interest_income,
        ebit,
        income_tax_expense: income_tax,
        net_income,
        gains_losses_on_sales: None,
        ffo_per_share: ffo_ps,
        dividends_per_share: divs_ps,
        weighted_avg_shares: wavg,
    })
}

/// Build a CashFlowStatement from extracted facts.
pub fn build_cash_flow_statement(
    facts: &[XbrlFact],
    period_start: DateTime<Utc>,
    period_end: DateTime<Utc>,
) -> Result<CashFlowStatement, EdgarError> {
    let net_income = extract_fact_or_default(facts, XbrlTags::NET_INCOME, 0.0);
    let depreciation = extract_fact_or_default(facts, XbrlTags::DEPRECIATION, 0.0);
    let capex = extract_fact_or_default(facts, XbrlTags::CAPITAL_EXPENDITURES, 0.0);
    let dispositions = extract_fact_or_default(facts, XbrlTags::PROPERTY_DISPOSITIONS, 0.0);
    let debt_issued = extract_fact_or_default(facts, XbrlTags::DEBT_ISSUED, 0.0);
    let debt_repaid = extract_fact_or_default(facts, XbrlTags::DEBT_REPAID, 0.0);
    let dividends_paid = extract_fact_or_default(facts, XbrlTags::DIVIDENDS_PAID, 0.0);

    let operating_cf = net_income + depreciation;
    let investing_cf = -capex + dispositions;
    let financing_cf = debt_issued - debt_repaid - dividends_paid;
    let net_change = operating_cf + investing_cf + financing_cf;

    Ok(CashFlowStatement {
        period_start,
        period_end,
        net_income,
        depreciation_amortization: depreciation,
        gains_on_sales: 0.0,
        operating_cash_flow: operating_cf,
        capital_expenditures: capex,
        property_dispositions: dispositions,
        investing_cash_flow: investing_cf,
        debt_issued,
        debt_repaid,
        dividends_paid,
        financing_cash_flow: financing_cf,
        net_change_in_cash: net_change,
    })
}

/// Build a complete REIT struct from parsed filing data.
pub fn build_reit_from_filing(
    metadata: &FilingMetadata,
    balance_sheet: BalanceSheet,
    income_statement: IncomeStatement,
) -> REIT {
    REIT {
        ticker: String::new(), // Needs to be set externally
        name: metadata.entity_name.clone(),
        cik: metadata.cik.clone(),
        sector: REITSector::Specialty, // Default; needs SIC code mapping
        inception_date: None,
        market_cap: 0.0,
        share_price: 0.0,
        shares_outstanding: balance_sheet.shares_outstanding,
        balance_sheet: Some(balance_sheet),
        income_statement: Some(income_statement),
        ratios: None,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn sample_facts() -> Vec<XbrlFact> {
        vec![
            XbrlFact {
                tag: "Assets".into(),
                value: "1,200,000".into(),
                context_ref: "ctx_2024".into(),
                unit_ref: Some("USD".into()),
                decimals: Some(-3),
            },
            XbrlFact {
                tag: "AssetsCurrent".into(),
                value: "100,000".into(),
                context_ref: "ctx_2024".into(),
                unit_ref: Some("USD".into()),
                decimals: Some(-3),
            },
            XbrlFact {
                tag: "Liabilities".into(),
                value: "600,000".into(),
                context_ref: "ctx_2024".into(),
                unit_ref: Some("USD".into()),
                decimals: Some(-3),
            },
            XbrlFact {
                tag: "LiabilitiesCurrent".into(),
                value: "50,000".into(),
                context_ref: "ctx_2024".into(),
                unit_ref: Some("USD".into()),
                decimals: Some(-3),
            },
            XbrlFact {
                tag: "LongTermDebt".into(),
                value: "300,000".into(),
                context_ref: "ctx_2024".into(),
                unit_ref: Some("USD".into()),
                decimals: Some(-3),
            },
            XbrlFact {
                tag: "Revenues".into(),
                value: "500,000".into(),
                context_ref: "ctx_2024".into(),
                unit_ref: Some("USD".into()),
                decimals: Some(-3),
            },
            XbrlFact {
                tag: "NetIncomeLoss".into(),
                value: "200,000".into(),
                context_ref: "ctx_2024".into(),
                unit_ref: Some("USD".into()),
                decimals: Some(-3),
            },
            XbrlFact {
                tag: "DepreciationAndAmortization".into(),
                value: "80,000".into(),
                context_ref: "ctx_2024".into(),
                unit_ref: Some("USD".into()),
                decimals: Some(-3),
            },
            XbrlFact {
                tag: "InterestExpense".into(),
                value: "25,000".into(),
                context_ref: "ctx_2024".into(),
                decimals: Some(-3),
                unit_ref: Some("USD".into()),
            },
            XbrlFact {
                tag: "OperatingIncomeLoss".into(),
                value: "300,000".into(),
                context_ref: "ctx_2024".into(),
                unit_ref: Some("USD".into()),
                decimals: Some(-3),
            },
            XbrlFact {
                tag: "CashAndCashEquivalentsAtCarryingValue".into(),
                value: "50,000".into(),
                context_ref: "ctx_2024".into(),
                unit_ref: Some("USD".into()),
                decimals: Some(-3),
            },
            XbrlFact {
                tag: "IncomeTaxExpenseBenefit".into(),
                value: "40,000".into(),
                context_ref: "ctx_2024".into(),
                unit_ref: Some("USD".into()),
                decimals: Some(-3),
            },
        ]
    }

    #[test]
    fn test_parse_xbrl_numeric_positive() {
        assert!((parse_xbrl_numeric("1,234,567.89").unwrap() - 1_234_567.89).abs() < 0.01);
    }

    #[test]
    fn test_parse_xbrl_numeric_negative_parens() {
        assert!((parse_xbrl_numeric("(1,234.56)").unwrap() - (-1_234.56)).abs() < 0.01);
    }

    #[test]
    fn test_parse_xbrl_numeric_empty() {
        assert!((parse_xbrl_numeric("").unwrap() - 0.0).abs() < 1e-9);
    }

    #[test]
    fn test_parse_xbrl_numeric_dash() {
        assert!((parse_xbrl_numeric("-").unwrap() - 0.0).abs() < 1e-9);
    }

    #[test]
    fn test_extract_fact_found() {
        let facts = sample_facts();
        let val = extract_fact(&facts, "Assets").unwrap();
        assert!((val - 1_200_000.0).abs() < 1.0);
    }

    #[test]
    fn test_extract_fact_missing() {
        let facts = sample_facts();
        let result = extract_fact(&facts, "NonexistentTag");
        assert!(result.is_err());
        match result.unwrap_err() {
            EdgarError::MissingTag { tag } => assert_eq!(tag, "NonexistentTag"),
            _ => panic!("Expected MissingTag error"),
        }
    }

    #[test]
    fn test_extract_fact_or_default() {
        let facts = sample_facts();
        assert!((extract_fact_or_default(&facts, "Missing", 42.0) - 42.0).abs() < 1e-9);
    }

    #[test]
    fn test_normalize_cik() {
        assert_eq!(normalize_cik("0000732987").unwrap(), "0000732987");
        assert_eq!(normalize_cik("732987").unwrap(), "0000732987");
        assert_eq!(normalize_cik("CIK:0000732987").unwrap(), "0000732987");
    }

    #[test]
    fn test_normalize_cik_invalid() {
        assert!(normalize_cik("ABC").is_err());
        assert!(normalize_cik("").is_err());
    }

    #[test]
    fn test_parse_filing_type() {
        let (ft, amended) = parse_filing_type("10-K");
        assert_eq!(ft, "10-K");
        assert!(!amended);

        let (ft, amended) = parse_filing_type("10-Q/A");
        assert_eq!(ft, "10-Q");
        assert!(amended);
    }

    #[test]
    fn test_build_balance_sheet() {
        let facts = sample_facts();
        let now = Utc::now();
        let bs = build_balance_sheet(&facts, now, 50_000_000.0).unwrap();
        assert!((bs.total_assets - 1_200_000.0).abs() < 1.0);
        assert!((bs.current_assets - 100_000.0).abs() < 1.0);
        assert!((bs.total_liabilities - 600_000.0).abs() < 1.0);
        assert!((bs.cash - 50_000.0).abs() < 1.0);
        assert_eq!(bs.shares_outstanding, 50_000_000.0);
    }

    #[test]
    fn test_build_income_statement() {
        let facts = sample_facts();
        let now = Utc::now();
        let is = build_income_statement(&facts, now, now).unwrap();
        assert!((is.total_revenue - 500_000.0).abs() < 1.0);
        assert!((is.net_income - 200_000.0).abs() < 1.0);
        assert!((is.depreciation_amortization - 80_000.0).abs() < 1.0);
        assert!((is.interest_expense - 25_000.0).abs() < 1.0);
    }

    #[test]
    fn test_build_cash_flow_statement() {
        let facts = sample_facts();
        let now = Utc::now();
        let cf = build_cash_flow_statement(&facts, now, now).unwrap();
        assert!((cf.net_income - 200_000.0).abs() < 1.0);
        assert!((cf.operating_cash_flow - 280_000.0).abs() < 1.0);
    }

    #[test]
    fn test_build_reit_from_filing() {
        let metadata = FilingMetadata {
            accession_number: "0001193125-24-123456".into(),
            filing_type: "10-K".into(),
            filing_date: Utc::now(),
            period_end: Utc::now(),
            period_start: Utc::now(),
            fiscal_year: 2024,
            fiscal_period: "FY".into(),
            cik: "0000732987".into(),
            entity_name: "Test REIT Inc.".into(),
            sic_code: Some("6798".into()),
            document_type: "10-K".into(),
            is_amended: false,
        };
        let facts = sample_facts();
        let bs = build_balance_sheet(&facts, Utc::now(), 50_000_000.0).unwrap();
        let is = build_income_statement(&facts, Utc::now(), Utc::now()).unwrap();
        let reit = build_reit_from_filing(&metadata, bs, is);
        assert_eq!(reit.name, "Test REIT Inc.");
        assert_eq!(reit.cik, "0000732987");
        assert!(reit.balance_sheet.is_some());
        assert!(reit.income_statement.is_some());
    }

    #[test]
    fn test_parse_filing_type_mixed_case() {
        let (ft, amended) = parse_filing_type("  10-k  ");
        assert_eq!(ft, "10-K");
        assert!(!amended);
    }
}
