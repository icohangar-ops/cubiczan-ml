use crate::types::{GovernanceMetric, ESGRating};

/// Governance scoring engine.
#[derive(Debug, Clone)]
pub struct GovernanceScorer {
    big4_auditors: Vec<String>,
    anti_corruption_keywords: Vec<String>,
    disclosure_topics: Vec<String>,
    related_party_keywords: Vec<String>,
}

impl GovernanceScorer {
    pub fn new() -> Self {
        Self {
            big4_auditors: vec![
                "deloitte".into(),
                "pricewaterhousecoopers".into(),
                "pwc".into(),
                "ernst & young".into(),
                "ey".into(),
                "kpmg".into(),
                "grant thornton".into(),
            ],
            anti_corruption_keywords: vec![
                "anti-corruption".into(),
                "anti corruption".into(),
                "bribery".into(),
                "fraud prevention".into(),
                "whistleblower".into(),
                "whistle-blower".into(),
                "conflict of interest".into(),
                "ethics".into(),
                "code of conduct".into(),
                "compliance".into(),
                "due diligence".into(),
                "facilitation payments".into(),
            ],
            disclosure_topics: vec![
                "climate".into(),
                "carbon".into(),
                "emissions".into(),
                "water".into(),
                "waste".into(),
                "biodiversity".into(),
                "human rights".into(),
                "labor".into(),
                "supply chain".into(),
                "governance".into(),
                "board".into(),
                "compensation".into(),
                "diversity".into(),
                "inclusion".into(),
                "safety".into(),
                "community".into(),
            ],
            related_party_keywords: vec![
                "related party".into(),
                "related-party".into(),
                "insider transaction".into(),
                "insider dealing".into(),
                "self-dealing".into(),
                "self dealing".into(),
                "affiliated transaction".into(),
                "family member".into(),
                "controlling shareholder".into(),
            ],
        }
    }

    // ─── Board Composition Analysis ─────────────────────────────

    /// Score board independence ratio (0-100).
    /// A higher ratio of independent directors is better.
    pub fn board_independence_score(independent_ratio: f64) -> f64 {
        match independent_ratio {
            r if r >= 0.80 => 95.0,
            r if r >= 0.67 => 80.0,
            r if r >= 0.50 => 65.0,
            r if r >= 0.33 => 45.0,
            r if r >= 0.20 => 30.0,
            _ => 15.0,
        }
    }

    /// Score board size (0-100). Moderate board sizes tend to be optimal.
    pub fn board_size_score(board_size: u32) -> f64 {
        match board_size {
            0 => 0.0,
            s if s <= 3 => 30.0,
            s if s <= 5 => 50.0,
            s if s <= 8 => 80.0,
            s if s <= 12 => 95.0,
            s if s <= 15 => 85.0,
            s if s <= 20 => 65.0,
            _ => 40.0,
        }
    }

    /// Combined board composition score.
    pub fn board_composition_score(independent_ratio: f64, board_size: u32) -> f64 {
        let ind = Self::board_independence_score(independent_ratio) * 0.70;
        let size = Self::board_size_score(board_size) * 0.30;
        ind + size
    }

    // ─── Executive Compensation ────────────────────────────────

    /// Score executive compensation reasonableness.
    /// `pay_ratio` is the ratio of CEO pay to median employee pay.
    pub fn executive_compensation_score(pay_ratio: f64) -> f64 {
        match pay_ratio {
            r if r <= 20.0 => 95.0,
            r if r <= 50.0 => 85.0,
            r if r <= 100.0 => 70.0,
            r if r <= 200.0 => 55.0,
            r if r <= 300.0 => 40.0,
            r if r <= 500.0 => 25.0,
            _ => 10.0,
        }
    }

    // ─── Audit Quality ─────────────────────────────────────────

    /// Score audit quality based on auditor name.
    pub fn audit_quality_score(&self, auditor_name: &str) -> f64 {
        let lower = auditor_name.to_lowercase();
        if self.big4_auditors.iter().any(|b| lower.contains(b.as_str())) {
            90.0
        } else if lower.contains("audit") || lower.contains("accounting") {
            60.0
        } else if lower.is_empty() || lower == "none" || lower == "n/a" {
            10.0
        } else {
            50.0
        }
    }

    /// Score audit quality considering tenure and rotation.
    pub fn audit_tenure_score(auditor_tenure_years: f64) -> f64 {
        // Best practice: rotate every 10-20 years
        match auditor_tenure_years {
            t if t <= 2.0 => 60.0, // New auditor — acceptable
            t if t <= 10.0 => 90.0, // Sweet spot
            t if t <= 20.0 => 75.0, // Getting long
            t if t <= 30.0 => 50.0, // Too long
            _ => 30.0,              // Way too long
        }
    }

    // ─── Shareholder Rights ────────────────────────────────────

    /// Score shareholder rights based on voting and proxy features.
    pub fn shareholder_rights_score(
        has_cumulative_voting: bool,
        proxy_access_pct: f64, // minimum % to nominate directors via proxy
        has_majority_voting: bool,
        has_say_on_pay: bool,
        dual_class_shares: bool,
    ) -> f64 {
        let mut score: f64 = 0.0;

        if has_cumulative_voting { score += 20.0; }

        score += match proxy_access_pct {
            p if p > 0.0 && p <= 3.0 => 25.0,
            p if p > 3.0 && p <= 5.0 => 20.0,
            p if p > 5.0 && p <= 10.0 => 15.0,
            p if p > 10.0 => 10.0,
            _ => 0.0,
        };

        if has_majority_voting { score += 15.0; }
        if has_say_on_pay { score += 15.0; }
        if dual_class_shares { score -= 15.0; }

        score.clamp(0.0, 100.0)
    }

    // ─── Anti-Corruption Policy ────────────────────────────────

    /// Score anti-corruption policy based on keyword detection (0-100).
    pub fn anti_corruption_score(&self, policy_text: &str) -> f64 {
        let lower = policy_text.to_lowercase();
        let matched = self.anti_corruption_keywords.iter()
            .filter(|kw| lower.contains(kw.as_str()))
            .count();

        let total = self.anti_corruption_keywords.len();
        let raw = (matched as f64 / total as f64) * 100.0;

        // Bonus for having a policy at all
        let has_content = policy_text.trim().len() > 50;
        let base = if has_content { 10.0 } else { 0.0 };

        (raw + base).min(100.0)
    }

    // ─── Transparency & Disclosure ─────────────────────────────

    /// Score transparency based on how many disclosure topics are mentioned.
    pub fn transparency_score(&self, disclosure_text: &str) -> f64 {
        let lower = disclosure_text.to_lowercase();
        let matched = self.disclosure_topics.iter()
            .filter(|topic| lower.contains(topic.as_str()))
            .count();

        let total = self.disclosure_topics.len();
        (matched as f64 / total as f64) * 100.0
    }

    // ─── Related Party Transaction Detection ───────────────────

    /// Detect presence of related party transaction concerns (0-100, higher = more concerns).
    pub fn related_party_risk(&self, text: &str) -> f64 {
        let lower = text.to_lowercase();
        let matched = self.related_party_keywords.iter()
            .filter(|kw| lower.contains(kw.as_str()))
            .count();

        let total = self.related_party_keywords.len();
        (matched as f64 / total as f64) * 100.0
    }

    /// Check if related party transactions are a concern.
    pub fn has_related_party_concerns(&self, text: &str) -> bool {
        self.related_party_risk(text) > 20.0
    }

    // ─── Composite Governance Score ────────────────────────────

    /// Compute overall governance score and return `GovernanceMetric`.
    pub fn composite_governance_score(
        &self,
        independent_ratio: f64,
        board_size: u32,
        pay_ratio: f64,
        auditor_name: &str,
        has_cumulative_voting: bool,
        proxy_access_pct: f64,
        has_majority_voting: bool,
        has_say_on_pay: bool,
        dual_class_shares: bool,
        anti_corruption_text: &str,
        disclosure_text: &str,
    ) -> GovernanceMetric {
        let board_independence = Self::board_composition_score(independent_ratio, board_size);
        let executive_compensation = Self::executive_compensation_score(pay_ratio);
        let audit_quality = self.audit_quality_score(auditor_name);
        let shareholder_rights = Self::shareholder_rights_score(
            has_cumulative_voting, proxy_access_pct, has_majority_voting,
            has_say_on_pay, dual_class_shares,
        );
        let anti_corruption = self.anti_corruption_score(anti_corruption_text);
        let transparency = self.transparency_score(disclosure_text);

        GovernanceMetric {
            board_independence,
            executive_compensation,
            audit_quality,
            shareholder_rights,
            anti_corruption,
            transparency,
        }
    }

    /// Generate detailed governance scoring breakdown.
    pub fn detailed_score(
        &self,
        independent_ratio: f64,
        board_size: u32,
        pay_ratio: f64,
        auditor_name: &str,
        has_cumulative_voting: bool,
        proxy_access_pct: f64,
        has_majority_voting: bool,
        has_say_on_pay: bool,
        dual_class_shares: bool,
        anti_corruption_text: &str,
        disclosure_text: &str,
        related_party_text: &str,
    ) -> GovernanceScoreBreakdown {
        let metric = self.composite_governance_score(
            independent_ratio, board_size, pay_ratio, auditor_name,
            has_cumulative_voting, proxy_access_pct, has_majority_voting,
            has_say_on_pay, dual_class_shares,
            anti_corruption_text, disclosure_text,
        );
        let overall = metric.average();
        let rating = ESGRating::from_score(overall);
        let related_party_concern = self.has_related_party_concerns(related_party_text);

        GovernanceScoreBreakdown {
            overall,
            rating,
            metric,
            related_party_concern,
        }
    }
}

impl Default for GovernanceScorer {
    fn default() -> Self {
        Self::new()
    }
}

/// Detailed governance scoring breakdown.
#[derive(Debug, Clone)]
pub struct GovernanceScoreBreakdown {
    pub overall: f64,
    pub rating: ESGRating,
    pub metric: GovernanceMetric,
    pub related_party_concern: bool,
}

#[cfg(test)]
mod tests {
    use super::*;

    fn scorer() -> GovernanceScorer {
        GovernanceScorer::new()
    }

    #[test]
    fn test_board_independence_high() {
        assert_eq!(GovernanceScorer::board_independence_score(0.85), 95.0);
    }

    #[test]
    fn test_board_independence_medium() {
        assert_eq!(GovernanceScorer::board_independence_score(0.67), 80.0);
    }

    #[test]
    fn test_board_independence_low() {
        assert_eq!(GovernanceScorer::board_independence_score(0.15), 15.0);
    }

    #[test]
    fn test_board_size_optimal() {
        assert_eq!(GovernanceScorer::board_size_score(9), 95.0);
    }

    #[test]
    fn test_board_size_small() {
        assert_eq!(GovernanceScorer::board_size_score(2), 30.0);
    }

    #[test]
    fn test_board_size_zero() {
        assert_eq!(GovernanceScorer::board_size_score(0), 0.0);
    }

    #[test]
    fn test_board_size_oversized() {
        assert_eq!(GovernanceScorer::board_size_score(25), 40.0);
    }

    #[test]
    fn test_board_composition_combined() {
        let score = GovernanceScorer::board_composition_score(0.75, 9);
        // 80*0.70 + 95*0.30 = 56 + 28.5 = 84.5
        assert!((score - 84.5).abs() < 1e-9);
    }

    #[test]
    fn test_executive_compensation_reasonable() {
        assert_eq!(GovernanceScorer::executive_compensation_score(30.0), 85.0);
    }

    #[test]
    fn test_executive_compensation_extreme() {
        assert_eq!(GovernanceScorer::executive_compensation_score(1000.0), 10.0);
    }

    #[test]
    fn test_audit_quality_big4() {
        let s = scorer();
        assert_eq!(s.audit_quality_score("Deloitte & Touche LLP"), 90.0);
        assert_eq!(s.audit_quality_score("PwC Audit"), 90.0);
        assert_eq!(s.audit_quality_score("KPMG International"), 90.0);
    }

    #[test]
    fn test_audit_quality_non_big4() {
        let s = scorer();
        assert_eq!(s.audit_quality_score("Local CPA Firm"), 50.0);
    }

    #[test]
    fn test_audit_quality_none() {
        let s = scorer();
        assert_eq!(s.audit_quality_score("none"), 10.0);
    }

    #[test]
    fn test_audit_tenure_score() {
        assert_eq!(GovernanceScorer::audit_tenure_score(5.0), 90.0);
        assert_eq!(GovernanceScorer::audit_tenure_score(25.0), 50.0);
    }

    #[test]
    fn test_shareholder_rights_full() {
        let score = GovernanceScorer::shareholder_rights_score(true, 3.0, true, true, false);
        // 20 + 25 + 15 + 15 + 0 = 75
        assert_eq!(score, 75.0);
    }

    #[test]
    fn test_shareholder_rights_dual_class_penalty() {
        let score = GovernanceScorer::shareholder_rights_score(true, 3.0, true, true, true);
        // 20 + 25 + 15 + 15 - 15 = 60
        assert_eq!(score, 60.0);
    }

    #[test]
    fn test_shareholder_rights_minimal() {
        let score = GovernanceScorer::shareholder_rights_score(false, 0.0, false, false, false);
        assert_eq!(score, 0.0);
    }

    #[test]
    fn test_anti_corruption_empty() {
        let s = scorer();
        assert_eq!(s.anti_corruption_score(""), 0.0);
    }

    #[test]
    fn test_anti_corruption_strong() {
        let s = scorer();
        let policy = "Our anti-corruption policy includes whistleblower protection, \
                      fraud prevention measures, ethics training, code of conduct, \
                      compliance programs, and due diligence procedures for \
                      conflict of interest management.";
        let score = s.anti_corruption_score(policy);
        assert!(score > 50.0);
    }

    #[test]
    fn test_transparency_score_full() {
        let s = scorer();
        let text = "We report on climate, carbon emissions, water, waste, biodiversity, \
                    human rights, labor practices, supply chain, governance, board \
                    composition, compensation, diversity, inclusion, safety, community.";
        let score = s.transparency_score(text);
        assert!(score > 80.0);
    }

    #[test]
    fn test_transparency_score_minimal() {
        let s = scorer();
        assert_eq!(s.transparency_score("no disclosures"), 0.0);
    }

    #[test]
    fn test_related_party_risk_none() {
        let s = scorer();
        assert_eq!(s.related_party_risk("normal business operations"), 0.0);
    }

    #[test]
    fn test_related_party_risk_present() {
        let s = scorer();
        let text = "The company engaged in a related-party transaction with a family member. \
                    Insider transactions were disclosed in the annual report.";
        assert!(s.has_related_party_concerns(text));
    }

    #[test]
    fn test_composite_governance_score() {
        let s = scorer();
        let m = s.composite_governance_score(
            0.75, 10, 50.0, "PwC",
            true, 3.0, true, true, false,
            "Strong anti-corruption and whistleblower policies",
            "Full disclosure on climate, carbon, emissions, governance",
        );
        assert!(m.board_independence > 0.0);
        assert!(m.audit_quality > 0.0);
        assert!(m.transparency > 0.0);
    }

    #[test]
    fn test_detailed_score() {
        let s = scorer();
        let breakdown = s.detailed_score(
            0.75, 10, 50.0, "Deloitte",
            true, 3.0, true, true, false,
            "anti-corruption whistleblower ethics compliance",
            "climate carbon emissions water waste biodiversity governance board",
            "no related party concerns here",
        );
        assert!(breakdown.overall > 0.0);
        assert!(!breakdown.related_party_concern);
    }

    #[test]
    fn test_default_scorer() {
        let s = GovernanceScorer::default();
        assert!(!s.big4_auditors.is_empty());
    }
}
