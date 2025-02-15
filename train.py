import os
from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.callbacks import CheckpointCallback, EvalCallback, BaseCallback
from racing_env import RacingEnv
from utils import find_latest_model
import numpy as np

class SaveOnBestTrainingRewardCallback(BaseCallback):
    def __init__(self, check_freq: int, save_path: str, keep_n_models: int = 5, verbose: int = 1):
        super().__init__(verbose)
        self.check_freq = check_freq
        self.save_path = save_path
        self.keep_n_models = keep_n_models
        self.best_mean_reward = -np.inf
        self.saved_models = []  # Keep track of saved model paths

    def _init_callback(self) -> None:
        if self.save_path is not None:
            os.makedirs(self.save_path, exist_ok=True)

    def _cleanup_old_models(self):
        """Remove old model files keeping only the N most recent ones"""
        while len(self.saved_models) > self.keep_n_models:
            old_model = self.saved_models.pop(0)  # Remove oldest model
            if os.path.exists(old_model):
                os.remove(old_model)
                if self.verbose > 1:
                    print(f"Removed old model: {old_model}")

    def _on_step(self) -> bool:
        if self.n_calls % self.check_freq == 0:
            if len(self.model.ep_info_buffer) > 0:
                ep_info_buffer = list(self.model.ep_info_buffer)
                last_episodes = ep_info_buffer[-min(10, len(ep_info_buffer)):]
                ep_rew_mean = np.mean([ep_info["r"] for ep_info in last_episodes])

                if self.verbose > 1:
                    print(f"Mean reward over last {len(last_episodes)} episodes: {ep_rew_mean:.2f}, "
                        f"best mean reward: {self.best_mean_reward:.2f}")
                
                if ep_rew_mean > self.best_mean_reward * 1.05:
                    self.best_mean_reward = ep_rew_mean
                    path = os.path.join(self.save_path, 
                        f"{self.num_timesteps}-steps-reward-{ep_rew_mean:.1f}.zip")
                    self.model.save(path)
                    self.saved_models.append(path)
                    self._cleanup_old_models()
                    
                    if self.verbose > 0:
                        print(f"Step: {self.num_timesteps} - new best reward: {ep_rew_mean:.2f}. "
                              f"Saving to {path}")
        return True

def train():
    # Create environments
    env = Monitor(RacingEnv())
    eval_env = Monitor(RacingEnv(render_mode="human"))
    
    # Initialize PPO agent
    model = PPO(
        "MlpPolicy",
        env,
        learning_rate=3e-4,
        n_steps=2048,
        batch_size=64,
        gamma=0.99,
        verbose=0,
        tensorboard_log="./logs/",
        device="cpu",
        policy_kwargs=dict(
            net_arch=dict(
                pi=[128, 128, 64],  # Deeper actor network
                vf=[128, 128, 64]   # Deeper critic network
            ),
        )
    )
    
    # Load latest model if available
    latest_model = find_latest_model()
    if (latest_model):
        print(f"Loading parameters from existing model: {latest_model}")
        model.set_parameters(latest_model)
    else:
        print("No existing model found, starting from scratch.")
    
    callbacks = [
        EvalCallback(
            eval_env,
            best_model_save_path="./checkpoints/",
            log_path="./logs/",
            eval_freq=200_000,
            deterministic=True,
            render=True,
            n_eval_episodes=3
        ),
        SaveOnBestTrainingRewardCallback(
            check_freq=100_000,
            save_path="./checkpoints/",
            keep_n_models=1
        )
        # CheckpointCallback(
        #     save_freq=500_000,
        #     save_path="./checkpoints/",
        #     name_prefix="racing_model"
        # ),
    ]
    
    # Train
    model.learn(
        total_timesteps=100_000_000,
        progress_bar=True,
        callback=callbacks,
    )
    
    # Save final model and cleanup
    model.save("./checkpoints/racing_model_final")
    env.close()
    eval_env.close()

if __name__ == "__main__":
    train()
