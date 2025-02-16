from stable_baselines3 import SAC, PPO
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.callbacks import EvalCallback
from racing_env import RacingEnv
from utils import linear_schedule, setup_run_dir
from callbacks import SaveOnBestTrainingRewardCallback, SaveLatestCallback
from config import Config


def train():
    config = Config()
    run_dir = setup_run_dir(config)
    env, eval_env = Monitor(RacingEnv()), Monitor(RacingEnv(render_mode="human"))
    
    model = PPO(
        "MlpPolicy",
        env,
        learning_rate=linear_schedule(config.LEARNING_RATE),
        gamma=config.GAMMA,
        use_sde=True,
        tensorboard_log=run_dir,
        device="cpu",
    )

    print(f"Starting new training run in {run_dir}")
    if config.RESUME_FROM:
        print(f"Using weights from: {config.RESUME_FROM}")
        model.set_parameters(config.RESUME_FROM)
     
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
        ),
        SaveLatestCallback(
            save_freq=config.EVAL_FREQ,
            save_path=run_dir
        )
    ]
    
    model.learn(
        total_timesteps=config.TOTAL_TIMESTEPS, 
        progress_bar=True, 
        callback=callbacks, 
        tb_log_name="run"
    )
   
    model.save(f"{run_dir}/racing_model_final")
    env.close()
    eval_env.close()


if __name__ == "__main__":
    train()
