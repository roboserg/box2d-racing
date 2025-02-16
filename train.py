from stable_baselines3 import SAC
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.callbacks import EvalCallback
from racing_env import RacingEnv
from utils import find_latest_model, linear_schedule, setup_run_dir
from callbacks import SaveOnBestTrainingRewardCallback


class Config:
    RUN_NAME = "03-test"
    TOTAL_TIMESTEPS = 10_000_000
    EVAL_FREQ = 500_000
    EVAL_EPISODES = 3
    CHECK_FREQ = 100_000
    KEEP_N_MODELS = 1
    LOG_DIR = "logs"
    

def train():
    run_dir = setup_run_dir(Config.RUN_NAME, Config.LOG_DIR)
    env, eval_env = Monitor(RacingEnv()), Monitor(RacingEnv(render_mode="human"))
    
    model = SAC(
        "MlpPolicy",
        env,
        learning_rate=linear_schedule(6e-4),
        gamma=0.98,
        verbose=0,
        tensorboard_log=run_dir,
        device="cpu",
        policy_kwargs=dict(net_arch=dict(pi=[64, 64], qf=[400, 300])),
    )
     
    callbacks = [
        EvalCallback(
            eval_env,
            best_model_save_path=run_dir,
            log_path=run_dir,
            eval_freq=Config.EVAL_FREQ,
            n_eval_episodes=Config.EVAL_EPISODES,
            render=True,
        ),
        SaveOnBestTrainingRewardCallback(
            check_freq=Config.CHECK_FREQ,
            save_path=run_dir,
            keep_n_models=Config.KEEP_N_MODELS
        )
    ]
    
    model.learn(total_timesteps=Config.TOTAL_TIMESTEPS, progress_bar=True, callback=callbacks)   
    model.save(f"{run_dir}/racing_model_final")
    env.close()
    eval_env.close()


if __name__ == "__main__":
    train()
