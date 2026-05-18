//! # market-sentiment-fedgpt
//!
//! A deterministic, auditable analysis engine for Fed tone analysis,
//! macro sentiment scoring, and portfolio risk briefing with
//! verification-gated outputs.

pub mod types;
pub mod parser;
pub mod tone;
pub mod risk;
pub mod brief;
pub mod pipeline;

pub use types::*;
pub use parser::FedTextParser;
pub use tone::ToneAnalyzer;
pub use risk::RiskCalculator;
pub use brief::BriefingGenerator;
pub use pipeline::AnalysisPipeline;

#[cfg(test)]
mod integration_tests {
    use super::*;

    #[test]
    fn test_end_to_end_hawkish_pipeline() {
        let statement = types::FedStatement {
            date: chrono::NaiveDate::from_ymd_opt(2024, 6, 12).unwrap(),
            text: "The Committee decided to maintain the target range for the federal \
                   funds rate at 5.25 to 5.50 percent. Inflation remains elevated and \
                   the labor market continues to be tight. The Committee is strongly \
                   committed to returning inflation to its 2 percent objective. \
                   Additional rate increases may be appropriate.".to_string(),
            statement_type: types::StatementType::FOMC,
        };

        let positions = vec![
            types::PortfolioPosition {
                asset: "10Y Treasury".to_string(),
                duration: 8.5,
                convexity: 75.0,
                notional: 1_000_000.0,
                weight: 0.5,
            },
            types::PortfolioPosition {
                asset: "S&P 500 ETF".to_string(),
                duration: 0.0,
                convexity: 0.0,
                notional: 500_000.0,
                weight: 0.5,
            },
        ];

        let pipeline = AnalysisPipeline::new(0.7);
        let briefing = pipeline.run(&statement, None, &positions).unwrap();

        assert!(briefing.verification_gate.passed);
        assert_eq!(briefing.sentiment.monetary_policy, types::MonetaryPolicy::Hawkish);
    }

    #[test]
    fn test_end_to_end_dovish_pipeline() {
        let statement = types::FedStatement {
            date: chrono::NaiveDate::from_ymd_opt(2024, 12, 18).unwrap(),
            text: "The Committee decided to lower the target range for the federal \
                   funds rate by 25 basis points. Inflation has moved closer to 2 \
                   percent and the labor market is moderating. The Committee \
                   anticipates that further gradual adjustments may be appropriate.".to_string(),
            statement_type: types::StatementType::FOMC,
        };

        let positions = vec![
            types::PortfolioPosition {
                asset: "Aggregate Bond".to_string(),
                duration: 6.0,
                convexity: 50.0,
                notional: 2_000_000.0,
                weight: 1.0,
            },
        ];

        let pipeline = AnalysisPipeline::new(0.5);
        let briefing = pipeline.run(&statement, None, &positions).unwrap();

        assert!(briefing.verification_gate.confidence >= 0.5);
    }

    #[test]
    fn test_end_to_end_with_prior_statement() {
        let prior = types::FedStatement {
            date: chrono::NaiveDate::from_ymd_opt(2024, 3, 20).unwrap(),
            text: "The Committee is attentive to inflation risks and is prepared to \
                   raise rates further if needed. Economic activity has been expanding \
                   at a strong pace.".to_string(),
            statement_type: types::StatementType::FOMC,
        };

        let current = types::FedStatement {
            date: chrono::NaiveDate::from_ymd_opt(2024, 6, 12).unwrap(),
            text: "The Committee has decided to maintain the target range. Inflation \
                   has moderated somewhat but remains elevated. The Committee will \
                   carefully assess incoming data.".to_string(),
            statement_type: types::StatementType::FOMC,
        };

        let positions = vec![
            types::PortfolioPosition {
                asset: "Corporate Bond".to_string(),
                duration: 5.0,
                convexity: 40.0,
                notional: 1_000_000.0,
                weight: 1.0,
            },
        ];

        let pipeline = AnalysisPipeline::new(0.5);
        let briefing = pipeline.run(&current, Some(&prior), &positions).unwrap();

        // Tone shift should be detected
        assert!(briefing.sentiment.tone_shift.is_some());
    }

    #[test]
    fn test_low_confidence_rejected() {
        let statement = types::FedStatement {
            date: chrono::NaiveDate::from_ymd_opt(2024, 6, 12).unwrap(),
            text: "The Committee met today.".to_string(),
            statement_type: types::StatementType::FOMC,
        };

        let positions = vec![
            types::PortfolioPosition {
                asset: "Cash".to_string(),
                duration: 0.0,
                convexity: 0.0,
                notional: 1_000_000.0,
                weight: 1.0,
            },
        ];

        let pipeline = AnalysisPipeline::new(0.95);
        let briefing = pipeline.run(&statement, None, &positions).unwrap();

        assert!(!briefing.verification_gate.passed);
    }
}
