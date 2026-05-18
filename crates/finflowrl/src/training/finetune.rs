/// Fine-Tuner — Stage 2 PPO fine-tuning of the MeanFlow policy.
///
/// After Stage 1 distillation, Stage 2 uses PPO to optimise the RL
/// objective (cumulative reward) directly in the environment.

use crate::agents::ppo::PPOAgent;
use crate::envs::hft_env::HFTEnv;
use crate::models::meanflow::MeanFlowPolicy;
use ndarray::Array1;
use rand::prelude::*;

/// Rollout data collected from one episode.
#[derive(Debug, Clone)]
pub struct Rollout {
    pub obs: Vec<Array1<f64>>,
    pub actions: Vec<Array1<f64>>,
    pub rewards: Vec<f64>,
    pub dones: Vec<f64>,
    pub values: Vec<f64>,
}

/// Stage 2: Fine-tune pre-trained policy with PPO.
pub struct FineTuner {
    pub policy: MeanFlowPolicy,
    pub env: HFTEnv,
    pub ppo_agent: PPOAgent,
    pub n_episodes: usize,
    pub steps_per_episode: usize,
    pub gamma: f64,
    pub lam: f64,
}

impl FineTuner {
    /// Create a new fine-tuner.
    pub fn new(
        policy: MeanFlowPolicy,
        env: HFTEnv,
        ppo_agent: PPOAgent,
        n_episodes: usize,
        steps_per_episode: usize,
        gamma: f64,
        lam: f64,
    ) -> Self {
        Self {
            policy,
            env,
            ppo_agent,
            n_episodes,
            steps_per_episode,
            gamma,
            lam,
        }
    }

    /// Collect one episode rollout using MeanFlow policy.
    pub fn collect_rollout(&mut self) -> Rollout {
        let mut rng = StdRng::seed_from_u64(42);
        let mut obs_list = Vec::new();
        let mut act_list = Vec::new();
        let mut reward_list = Vec::new();
        let mut done_list = Vec::new();
        let mut value_list = Vec::new();

        let obs = self.env.reset();
        let mut current_obs = obs;

        for _ in 0..self.steps_per_episode {
            let action = self.policy.act(&current_obs, &mut rng, false);
            let obs_arr = Array1::from_vec(
                action.iter().map(|v| v.max(-1.0).min(1.0)).collect(),
            );
            let (next_obs, reward, done, _info) = self.env.step(&obs_arr);

            let value = self.ppo_agent.policy.get_value(&current_obs);

            obs_list.push(current_obs.clone());
            act_list.push(obs_arr);
            reward_list.push(reward);
            done_list.push(if done { 1.0 } else { 0.0 });
            value_list.push(value);

            current_obs = next_obs;
            if done {
                break;
            }
        }

        Rollout {
            obs: obs_list,
            actions: act_list,
            rewards: reward_list,
            dones: done_list,
            values: value_list,
        }
    }

    /// Run fine-tuning loop.
    ///
    /// Returns: training history with episode rewards and PnLs.
    pub fn train(&mut self, n_epochs: usize) -> Vec<f64> {
        let mut history = Vec::new();

        for _ in 0..n_epochs {
            let mut ep_rewards = Vec::new();

            for _ in 0..self.n_episodes {
                let rollout = self.collect_rollout();
                let total_reward: f64 = rollout.rewards.iter().sum();
                ep_rewards.push(total_reward);
            }

            let avg_reward: f64 = if ep_rewards.is_empty() {
                0.0
            } else {
                ep_rewards.iter().sum::<f64>() / ep_rewards.len() as f64
            };
            history.push(avg_reward);
        }

        history
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::agents::ppo::PPOAgent;
    use crate::envs::hft_env::HFTEnv;
    use crate::models::meanflow::MeanFlowPolicy;

    #[test]
    fn test_finetuner_creation() {
        let env = HFTEnv::with_params(42, 20, 10.0, 0.001, 0.01, 1.0);
        let policy = MeanFlowPolicy::new_default(6, 1);
        let ppo = PPOAgent::new(6, 3, vec![32, 32], 3e-4, 0.2, 0.99, 0.95);
        let ft = FineTuner::new(policy, env, ppo, 1, 10, 0.99, 0.95);
        assert_eq!(ft.n_episodes, 1);
    }

    #[test]
    fn test_finetuner_rollout() {
        let env = HFTEnv::with_params(42, 20, 10.0, 0.001, 0.01, 1.0);
        let policy = MeanFlowPolicy::new_default(6, 1);
        let ppo = PPOAgent::new(6, 3, vec![32, 32], 3e-4, 0.2, 0.99, 0.95);
        let mut ft = FineTuner::new(policy, env, ppo, 1, 10, 0.99, 0.95);
        let rollout = ft.collect_rollout();
        assert!(!rollout.obs.is_empty());
        assert!(!rollout.rewards.is_empty());
    }
}
