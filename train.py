from stable_baselines3 import SAC
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.callbacks import EvalCallback
from racing_env import RacingEnv
from utils import find_latest_model, linear_schedule, setup_run_dir
from callbacks import SaveOnBestTrainingRewardCallback
import numpy as np

class TrainingConfig:
    # Training duration
    TOTAL_TIMESTEPS = 10_000_000
    
    # Evaluation settings
    EVAL_FREQ = 500_000
    EVAL_EPISODES = 3
    
    # Checkpoint settings
    CHECK_FREQ = 100_000
    KEEP_N_MODELS = 1
    
    # Paths
    CHECKPOINT_DIR = "checkpoints"
    LOG_DIR = "logs"


def train():
    run_dir = setup_run_dir(run_name="asdasd")
    
    env = Monitor(RacingEnv())
    eval_env = Monitor(RacingEnv(render_mode="human"))
    
    model = SAC(
        "MlpPolicy",
        env,
        learning_rate=linear_schedule(6e-4),
        gamma=0.98,
        verbose=0,
        tensorboard_log=TrainingConfig.LOG_DIR,
        device="cpu",
        policy_kwargs=dict(net_arch=dict(pi=[64, 64], qf=[400, 300])),
    )
    
    latest_model = find_latest_model()
    if (latest_model):
        print(f"Loading parameters from existing model: {latest_model}")
        try:
            model.set_parameters(latest_model)
            print("Successfully loaded model parameters")
        except Exception as e:
            print(f"Failed to load model parameters: {e}")
            print("Starting from scratch instead")
    else:
        print("No existing model found, starting from scratch.")
    
    callbacks = [
        EvalCallback(
            eval_env,
            best_model_save_path=TrainingConfig.CHECKPOINT_DIR,
            log_path=TrainingConfig.LOG_DIR,
            eval_freq=TrainingConfig.EVAL_FREQ,
            deterministic=True,
            render=True,
            n_eval_episodes=TrainingConfig.EVAL_EPISODES
        ),
        SaveOnBestTrainingRewardCallback(
            check_freq=TrainingConfig.CHECK_FREQ,
            save_path=TrainingConfig.CHECKPOINT_DIR,
            keep_n_models=TrainingConfig.KEEP_N_MODELS
        )
    ]
    
    model.learn(
        total_timesteps=TrainingConfig.TOTAL_TIMESTEPS,
        progress_bar=True,
        callback=callbacks,
    )
    
    model.save(f"{TrainingConfig.CHECKPOINT_DIR}/racing_model_final")
    env.close()
    eval_env.close()


if __name__ == "__main__":
    train()
