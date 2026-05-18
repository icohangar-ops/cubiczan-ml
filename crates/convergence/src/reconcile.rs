// ─── Financial data reconciliation ────────────────────────────────────────────

use crate::types::{MergeEntity, ReconciliationEntry};
use std::collections::HashMap;

/// Result of a fuzzy account-match operation.
#[derive(Debug, Clone)]
pub struct FuzzyMatch {
    pub target_account: String,
    pub candidate_account: String,
    pub score: f64, // 0.0 – 1.0
}

/// Result of a balance-sheet reconciliation run.
#[derive(Debug, Clone)]
pub struct ReconciliationResult {
    pub total_entries: usize,
    pub reconciled: usize,
    pub unreconciled: usize,
    pub total_debit: f64,
    pub total_credit: f64,
    pub net_variance: f64,
    pub match_rate: f64,
}

/// Result of a trial balance comparison between two entities.
#[derive(Debug, Clone)]
pub struct TrialBalanceComparison {
    pub acquirer_total: f64,
    pub target_total: f64,
    pub difference: f64,
    pub matched_accounts: usize,
    pub unmatched_accounts: usize,
    pub variances: Vec<AccountVariance>,
}

/// Per-account variance from trial balance comparison.
#[derive(Debug, Clone)]
pub struct AccountVariance {
    pub account_code: String,
    pub acquirer_value: f64,
    pub target_value: f64,
    pub variance: f64,
    pub percent_variance: f64,
}

// ── Account mapping ──────────────────────────────────────────────────────────

/// Build an account-code mapping from one entity's chart of accounts to another's.
/// Matches exact codes first, then falls back to fuzzy matching on account names.
pub fn build_account_map(
    acquirer: &MergeEntity,
    target: &MergeEntity,
    fuzzy_threshold: f64,
) -> HashMap<String, String> {
    let mut map = HashMap::new();

    // Pass 1: exact code match
    for t_entry in &target.entries {
        for a_entry in &acquirer.entries {
            if a_entry.account_code == t_entry.account_code {
                map.insert(
                    t_entry.account_code.clone(),
                    a_entry.account_code.clone(),
                );
            }
        }
    }

    // Pass 2: fuzzy name match for unmatched target codes
    let matched_targets: std::collections::HashSet<String> = map.keys().cloned().collect();
    for t_entry in &target.entries {
        if matched_targets.contains(&t_entry.account_code) {
            continue;
        }
        let target_lower = t_entry.account_name.to_lowercase();
        let mut best_score = 0.0_f64;
        let mut best_code = String::new();
        for a_entry in &acquirer.entries {
            let acq_lower = a_entry.account_name.to_lowercase();
            let score = jaro_winkler_similarity(&acq_lower, &target_lower);
            if score > best_score {
                best_score = score;
                best_code = a_entry.account_code.clone();
            }
        }
        if best_score >= fuzzy_threshold && !best_code.is_empty() {
            map.insert(t_entry.account_code.clone(), best_code);
        }
    }

    map
}

/// Compute the Jaro–Winkler similarity between two strings (0.0 – 1.0).
pub fn jaro_winkler_similarity(a: &str, b: &str) -> f64 {
    if a.is_empty() && b.is_empty() {
        return 1.0;
    }
    if a.is_empty() || b.is_empty() {
        return 0.0;
    }

    let a_chars: Vec<char> = a.chars().collect();
    let b_chars: Vec<char> = b.chars().collect();
    let match_distance = std::cmp::max(a_chars.len(), b_chars.len()) / 2;

    let mut a_matched = vec![false; a_chars.len()];
    let mut b_matched = vec![false; b_chars.len()];
    let mut matches = 0_usize;
    let mut transpositions = 0_usize;

    for i in 0..a_chars.len() {
        let start = i.saturating_sub(match_distance);
        let end = (i + match_distance + 1).min(b_chars.len());
        for j in start..end {
            if b_matched[j] || a_chars[i] != b_chars[j] {
                continue;
            }
            a_matched[i] = true;
            b_matched[j] = true;
            matches += 1;
            break;
        }
    }

    if matches == 0 {
        return 0.0;
    }

    let mut k = 0_usize;
    for i in 0..a_chars.len() {
        if !a_matched[i] {
            continue;
        }
        while !b_matched[k] {
            k += 1;
        }
        if a_chars[i] != b_chars[k] {
            transpositions += 1;
        }
        k += 1;
    }

    let jaro = (matches as f64 / a_chars.len() as f64
        + matches as f64 / b_chars.len() as f64
        + (matches as f64 - transpositions as f64 / 2.0) / matches as f64)
        / 3.0;

    // Winkler prefix bonus (up to 4 characters)
    let prefix_len = a_chars
        .iter()
        .zip(b_chars.iter())
        .take(4)
        .take_while(|(ac, bc)| ac == bc)
        .count();

    let p = 0.1;
    jaro + prefix_len as f64 * p * (1.0 - jaro)
}

// ── Duplicate detection ──────────────────────────────────────────────────────

/// Find potential duplicate accounts within a single entity using fuzzy matching.
pub fn detect_duplicates(entity: &MergeEntity, threshold: f64) -> Vec<FuzzyMatch> {
    let mut matches = Vec::new();
    let entries = &entity.entries;
    for i in 0..entries.len() {
        for j in (i + 1)..entries.len() {
            let score = jaro_winkler_similarity(
                &entries[i].account_name.to_lowercase(),
                &entries[j].account_name.to_lowercase(),
            );
            if score >= threshold {
                matches.push(FuzzyMatch {
                    target_account: entries[i].account_code.clone(),
                    candidate_account: entries[j].account_code.clone(),
                    score,
                });
            }
        }
    }
    matches
}

// ── Data normalization ───────────────────────────────────────────────────────

/// Normalize account names: lowercase, collapse whitespace, trim.
pub fn normalize_account_name(name: &str) -> String {
    name.to_lowercase()
        .split_whitespace()
        .collect::<Vec<_>>()
        .join(" ")
        .trim()
        .to_owned()
}

/// Normalize monetary values by rounding to the nearest cent.
pub fn normalize_amount(value: f64) -> f64 {
    (value * 100.0).round() / 100.0
}

/// Normalize all entries in an entity in-place.
pub fn normalize_entity(entity: &mut MergeEntity) {
    for entry in &mut entity.entries {
        entry.account_name = normalize_account_name(&entry.account_name);
        entry.debit = normalize_amount(entry.debit);
        entry.credit = normalize_amount(entry.credit);
        entry.net = normalize_amount(entry.debit - entry.credit);
    }
}

// ── Balance sheet reconciliation ─────────────────────────────────────────────

/// Reconcile the combined balance sheet across two merge entities.
/// Returns summary statistics.
pub fn reconcile_balance_sheet(acquirer: &MergeEntity, target: &MergeEntity) -> ReconciliationResult {
    let total_entries = acquirer.entries.len() + target.entries.len();
    let reconciled = acquirer.reconciled_count() + target.reconciled_count();
    let unreconciled = total_entries - reconciled;

    let total_debit: f64 = acquirer
        .entries
        .iter()
        .chain(target.entries.iter())
        .map(|e| e.debit)
        .sum();
    let total_credit: f64 = acquirer
        .entries
        .iter()
        .chain(target.entries.iter())
        .map(|e| e.credit)
        .sum();
    let net_variance = normalize_amount(total_debit - total_credit);
    let match_rate = if total_entries > 0 {
        reconciled as f64 / total_entries as f64
    } else {
        0.0
    };

    ReconciliationResult {
        total_entries,
        reconciled,
        unreconciled,
        total_debit: normalize_amount(total_debit),
        total_credit: normalize_amount(total_credit),
        net_variance,
        match_rate,
    }
}

// ── Trial balance comparison ─────────────────────────────────────────────────

/// Compare trial balances between the acquirer and target entities.
/// Matches accounts by code and computes variances.
pub fn compare_trial_balances(acquirer: &MergeEntity, target: &MergeEntity) -> TrialBalanceComparison {
    let acquirer_total = normalize_amount(acquirer.total_net());
    let target_total = normalize_amount(target.total_net());
    let difference = normalize_amount(acquirer_total - target_total);

    // Build a lookup of acquirer entries by code
    let acq_map: HashMap<&str, &ReconciliationEntry> = acquirer
        .entries
        .iter()
        .map(|e| (e.account_code.as_str(), e))
        .collect();

    let mut matched_accounts = 0_usize;
    let mut variances = Vec::new();

    for t_entry in &target.entries {
        if let Some(a_entry) = acq_map.get(t_entry.account_code.as_str()) {
            matched_accounts += 1;
            let acq_net = normalize_amount(a_entry.net);
            let tgt_net = normalize_amount(t_entry.net);
            let variance = normalize_amount(acq_net - tgt_net);
            let pct = if tgt_net.abs() > 0.01 {
                (variance / tgt_net.abs()) * 100.0
            } else if variance.abs() > 0.01 {
                100.0
            } else {
                0.0
            };
            variances.push(AccountVariance {
                account_code: t_entry.account_code.clone(),
                acquirer_value: acq_net,
                target_value: tgt_net,
                variance,
                percent_variance: pct,
            });
        }
    }

    let unmatched_accounts = target.entries.len().saturating_sub(matched_accounts);

    TrialBalanceComparison {
        acquirer_total,
        target_total,
        difference,
        matched_accounts,
        unmatched_accounts,
        variances,
    }
}

// ── Auto-reconciliation ──────────────────────────────────────────────────────

/// Attempt to auto-reconcile entries between two entities using the account map.
/// Mutates both entities' entries in place.
pub fn auto_reconcile(
    acquirer: &mut MergeEntity,
    target: &mut MergeEntity,
    threshold: f64,
) -> usize {
    let map = build_account_map(acquirer, target, threshold);
    let mut reconciled_count = 0_usize;

    for t_entry in &mut target.entries {
        if t_entry.is_reconciled {
            continue;
        }
        if let Some(acq_code) = map.get(&t_entry.account_code) {
            t_entry.reconcile(acq_code);
            // Also mark the corresponding acquirer entry
            for a_entry in &mut acquirer.entries {
                if a_entry.account_code == *acq_code && !a_entry.is_reconciled {
                    a_entry.reconcile(&t_entry.account_code);
                    break;
                }
            }
            reconciled_count += 1;
        }
    }

    reconciled_count
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::types::EntityType;

    fn make_acquirer() -> MergeEntity {
        let mut e = MergeEntity::new("AcquirerCo", EntityType::Acquirer, "USD");
        e.add_entry(ReconciliationEntry::new("AcquirerCo", "1000", "Cash and Equivalents", 1_000_000.0, 0.0));
        e.add_entry(ReconciliationEntry::new("AcquirerCo", "1200", "Accounts Receivable", 500_000.0, 0.0));
        e.add_entry(ReconciliationEntry::new("AcquirerCo", "2000", "Accounts Payable", 0.0, 300_000.0));
        e.add_entry(ReconciliationEntry::new("AcquirerCo", "3000", "Revenue", 0.0, 2_000_000.0));
        e.add_entry(ReconciliationEntry::new("AcquirerCo", "4000", "Cost of Goods Sold", 1_200_000.0, 0.0));
        e
    }

    fn make_target() -> MergeEntity {
        let mut e = MergeEntity::new("TargetCo", EntityType::Target, "USD");
        e.add_entry(ReconciliationEntry::new("TargetCo", "1000", "Cash & Equivalents", 600_000.0, 0.0));
        e.add_entry(ReconciliationEntry::new("TargetCo", "1200", "Accounts Receivable", 250_000.0, 0.0));
        e.add_entry(ReconciliationEntry::new("TargetCo", "2000", "Accounts Payable", 0.0, 150_000.0));
        e.add_entry(ReconciliationEntry::new("TargetCo", "3000", "Revenue", 0.0, 1_000_000.0));
        e.add_entry(ReconciliationEntry::new("TargetCo", "9999", "Intangible Assets", 400_000.0, 0.0));
        e
    }

    #[test]
    fn test_jaro_winkler_identical() {
        let score = jaro_winkler_similarity("cash", "cash");
        assert!((score - 1.0).abs() < 0.001);
    }

    #[test]
    fn test_jaro_winkler_different() {
        let score = jaro_winkler_similarity("cash", "revenue");
        assert!(score < 0.5);
    }

    #[test]
    fn test_jaro_winkler_similar() {
        let score = jaro_winkler_similarity("accounts receivable", "accounts receivables");
        assert!(score > 0.9);
    }

    #[test]
    fn test_jaro_winkler_empty() {
        assert!((jaro_winkler_similarity("", "") - 1.0).abs() < 0.001);
        assert!((jaro_winkler_similarity("abc", "") - 0.0).abs() < 0.001);
    }

    #[test]
    fn test_build_account_map_exact() {
        let acq = make_acquirer();
        let tgt = make_target();
        let map = build_account_map(&acq, &tgt, 0.8);
        // Codes 1000, 1200, 2000, 3000 should match exactly
        assert_eq!(map.get("1000").unwrap(), "1000");
        assert_eq!(map.get("1200").unwrap(), "1200");
        assert_eq!(map.get("2000").unwrap(), "2000");
        assert_eq!(map.get("3000").unwrap(), "3000");
        // 9999 should not match
        assert!(!map.contains_key("9999"));
    }

    #[test]
    fn test_detect_duplicates() {
        let mut entity = MergeEntity::new("Test", EntityType::Target, "USD");
        entity.add_entry(ReconciliationEntry::new("Test", "100", "Cash", 100.0, 0.0));
        entity.add_entry(ReconciliationEntry::new("Test", "101", "Cash and cash equivalents", 100.0, 0.0));
        entity.add_entry(ReconciliationEntry::new("Test", "200", "Revenue", 0.0, 200.0));
        let dups = detect_duplicates(&entity, 0.6);
        assert!(!dups.is_empty());
    }

    #[test]
    fn test_detect_duplicates_no_false_positives() {
        let mut entity = MergeEntity::new("Test", EntityType::Target, "USD");
        entity.add_entry(ReconciliationEntry::new("Test", "100", "Cash", 100.0, 0.0));
        entity.add_entry(ReconciliationEntry::new("Test", "200", "Revenue", 0.0, 200.0));
        entity.add_entry(ReconciliationEntry::new("Test", "300", "Equipment", 300.0, 0.0));
        let dups = detect_duplicates(&entity, 0.85);
        assert!(dups.is_empty());
    }

    #[test]
    fn test_normalize_account_name() {
        assert_eq!(normalize_account_name("  Cash   and  Equivalents  "), "cash and equivalents");
        assert_eq!(normalize_account_name("ACCOUNTS RECEIVABLE"), "accounts receivable");
    }

    #[test]
    fn test_normalize_amount() {
        assert!((normalize_amount(123.456) - 123.46).abs() < 0.001);
        assert!((normalize_amount(100.0) - 100.0).abs() < 0.001);
    }

    #[test]
    fn test_normalize_entity() {
        let mut entity = make_acquirer();
        normalize_entity(&mut entity);
        // Check first entry was normalized
        assert_eq!(entity.entries[0].account_name, "cash and equivalents");
    }

    #[test]
    fn test_reconcile_balance_sheet() {
        let acq = make_acquirer();
        let tgt = make_target();
        let result = reconcile_balance_sheet(&acq, &tgt);
        assert_eq!(result.total_entries, 10); // 5 + 5
        assert_eq!(result.reconciled, 0); // none reconciled yet
        assert_eq!(result.unreconciled, 10);
        assert!(result.match_rate < 0.01);
    }

    #[test]
    fn test_compare_trial_balances() {
        let acq = make_acquirer();
        let tgt = make_target();
        let result = compare_trial_balances(&acq, &tgt);
        assert_eq!(result.matched_accounts, 4); // 1000, 1200, 2000, 3000
        assert_eq!(result.unmatched_accounts, 1); // 9999
        // Acquirer net: 1000000+500000-300000-2000000+1200000 = 400000
        // Target net: 600000+250000-150000-1000000+400000 = 100000
        assert!((result.acquirer_total - 400_000.0).abs() < 0.01);
        assert!((result.target_total - 100_000.0).abs() < 0.01);
    }

    #[test]
    fn test_auto_reconcile() {
        let mut acq = make_acquirer();
        let mut tgt = make_target();
        let count = auto_reconcile(&mut acq, &mut tgt, 0.8);
        assert!(count >= 4); // at least the 4 exact matches
        assert!(tgt.reconciled_count() >= 4);
    }

    #[test]
    fn test_auto_reconcile_marks_both_sides() {
        let mut acq = make_acquirer();
        let mut tgt = make_target();
        auto_reconcile(&mut acq, &mut tgt, 0.8);
        assert!(acq.reconciled_count() > 0);
        assert!(tgt.reconciled_count() > 0);
    }

    #[test]
    fn test_variance_computation() {
        let acq = make_acquirer();
        let tgt = make_target();
        let result = compare_trial_balances(&acq, &tgt);
        // Account 1000: acquirer=1000000, target=600000 => variance=400000
        let v1000 = result.variances.iter().find(|v| v.account_code == "1000").unwrap();
        assert!((v1000.variance - 400_000.0).abs() < 0.01);
        assert!((v1000.percent_variance - 66.67).abs() < 1.0);
    }
}
