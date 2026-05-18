pub mod types;
pub mod environmental;
pub mod social;
pub mod governance;
pub mod verification;
pub mod bonds;
pub mod pipeline;

// Re-export key types
pub use types::{
    ESGCategory,
    ESGRating,
    ESGScore,
    ESGReport,
    EnvironmentalMetric,
    SocialMetric,
    GovernanceMetric,
    VerificationStatus,
    VerificationClaim,
    GreenBond,
    CarbonCredit,
    CarbonRetirement,
};

pub use environmental::EnvironmentalScorer;
pub use social::SocialScorer;
pub use governance::GovernanceScorer;
pub use verification::VerificationEngine;
pub use bonds::GreenBondAnalyzer;
pub use pipeline::ESGPipeline;
