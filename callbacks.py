import os
import numpy as np
from stable_baselines3.common.callbacks import BaseCallback

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
                last_episodes = ep_info_buffer[-min(30, len(ep_info_buffer)):]
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