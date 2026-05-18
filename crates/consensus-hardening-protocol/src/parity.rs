//! Model parity assessment for CHP sessions.

use crate::models::*;

fn infer_tier(model_name: &str) -> ModelTier {
    let name = model_name.to_lowercase();
    if ["opus", "max", "frontier"].iter().any(|t| name.contains(t)) {
        return ModelTier::FRONTIER;
    }
    if ["gpt-5", "claude 4", "claude-4", "high"].iter().any(|t| name.contains(t)) {
        return ModelTier::HIGH;
    }
    if ["sonnet", "4o", "mid", "gpt-4"].iter().any(|t| name.contains(t)) {
        return ModelTier::MID;
    }
    if ["mini", "small", "haiku"].iter().any(|t| name.contains(t)) {
        return ModelTier::SMALL;
    }
    ModelTier::UNKNOWN
}

const ALL_TIERS: &[ModelTier] = &[
    ModelTier::SMALL,
    ModelTier::MID,
    ModelTier::HIGH,
    ModelTier::FRONTIER,
    ModelTier::UNKNOWN,
];

pub fn assess_model_parity(origin_model: &str, partner_model: &str) -> ModelParityCheck {
    let origin_tier = infer_tier(origin_model);
    let partner_tier = infer_tier(partner_model);

    if origin_tier == ModelTier::UNKNOWN || partner_tier == ModelTier::UNKNOWN {
        return ModelParityCheck {
            origin: origin_model.into(),
            partner: partner_model.into(),
            delta: ModelParityDelta::MINOR,
            advisory: Some("One or both model tiers are unknown. Treat parity as advisory only.".into()),
        };
    }

    let gap = ALL_TIERS.iter().position(|t| *t == origin_tier)
        .expect("tier index")
        .abs_diff(ALL_TIERS.iter().position(|t| *t == partner_tier).expect("tier index"));

    match gap {
        0 => ModelParityCheck {
            origin: origin_model.into(),
            partner: partner_model.into(),
            delta: ModelParityDelta::NONE,
            advisory: None,
        },
        1 => ModelParityCheck {
            origin: origin_model.into(),
            partner: partner_model.into(),
            delta: ModelParityDelta::MINOR,
            advisory: Some("Slight analytical weight difference. Monitor for dominance bias.".into()),
        },
        _ => ModelParityCheck {
            origin: origin_model.into(),
            partner: partner_model.into(),
            delta: ModelParityDelta::SIGNIFICANT,
            advisory: None,
        },
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_infer_tier_claude_opus() {
        assert_eq!(infer_tier("claude-opus-4"), ModelTier::FRONTIER);
    }

    #[test]
    fn test_infer_tier_claude_sonnet() {
        assert_eq!(infer_tier("claude-sonnet-4"), ModelTier::MID);
    }

    #[test]
    fn test_infer_tier_haiku() {
        assert_eq!(infer_tier("claude-3-haiku"), ModelTier::SMALL);
    }

    #[test]
    fn test_infer_tier_unknown() {
        assert_eq!(infer_tier("some-random-model"), ModelTier::UNKNOWN);
    }

    #[test]
    fn test_assess_parity_same_tier() {
        let p = assess_model_parity("claude-sonnet-4", "claude-sonnet-4");
        assert_eq!(p.delta, ModelParityDelta::NONE);
        assert!(p.advisory.is_none());
    }

    #[test]
    fn test_assess_parity_minor_gap() {
        let p = assess_model_parity("claude-sonnet-4", "claude-haiku-3");
        assert_eq!(p.delta, ModelParityDelta::MINOR);
    }

    #[test]
    fn test_assess_parity_significant_gap() {
        let p = assess_model_parity("claude-opus-4", "claude-sonnet-4");
        assert_eq!(p.delta, ModelParityDelta::SIGNIFICANT);
    }

    #[test]
    fn test_assess_parity_unknown() {
        let p = assess_model_parity("unknown-model", "claude-sonnet-4");
        assert_eq!(p.delta, ModelParityDelta::MINOR);
        assert!(p.advisory.is_some());
    }

    #[test]
    fn test_model_parity_check_delta_is_enum() {
        let p = assess_model_parity("claude-opus-4", "claude-opus-4");
        // Should be NONE (same tier)
        assert_eq!(p.delta, ModelParityDelta::NONE);
    }

    #[test]
    fn test_assess_parity_serde_roundtrip() {
        let p = assess_model_parity("claude-sonnet-4", "claude-opus-4");
        let json = serde_json::to_string(&p).unwrap();
        let restored: ModelParityCheck = serde_json::from_str(&json).unwrap();
        assert_eq!(restored.delta, p.delta);
        assert_eq!(restored.origin, p.origin);
        assert_eq!(restored.partner, p.partner);
    }
}
