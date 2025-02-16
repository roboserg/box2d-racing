from dataclasses import dataclass
from stable_baselines3 import SAC, PPO
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.callbacks import EvalCallback
from racing_env import RacingEnv
from utils import linear_schedule, setup_run_dir
from callbacks import SaveOnBestTrainingRewardCallback


@dataclass
class Config:
    RUN_NAME: str = "12-SAC"
    TOTAL_TIMESTEPS: int = 10_000_000
    EVAL_FREQ: int = 500_000
    EVAL_EPISODES: int = 1
    CHECK_FREQ: int = 50_000
    KEEP_N_MODELS: int = 1
    LOG_DIR: str = "logs"


def train():
    config = Config()
    run_dir = setup_run_dir(config)
    env, eval_env = Monitor(RacingEnv()), Monitor(RacingEnv(render_mode="human"))
    
    model = SAC(
        "MlpPolicy",
        env,
        learning_rate=linear_schedule(3e-4),
        gamma=0.99,
        tensorboard_log=run_dir,
        device="cpu",
    )
     
    callbacks = [
        EvalCallback(
            eval_env,
            eval_freq=config.EVAL_FREQ,
            n_eval_episodes=config.EVAL_EPISODES,
            render=True,
        ),
        SaveOnBestTrainingRewardCallback(
            check_freq=config.CHECK_FREQ,
            save_path=run_dir,
            keep_n_models=config.KEEP_N_MODELS
        )
    ]
    
    model.learn(total_timesteps=config.TOTAL_TIMESTEPS, progress_bar=True, callback=callbacks, tb_log_name="run")
   
    model.save(f"{run_dir}/racing_model_final")
    env.close()
    eval_env.close()


if __name__ == "__main__":
    train()
