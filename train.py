from stable_baselines3 import SAC, PPO
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.callbacks import EvalCallback
from racing_env import RacingEnv
from utils import linear_schedule, setup_run_dir
from callbacks import SaveOnBestTrainingRewardCallback, SaveLatestCallback
from config import Config


def train():
    run_dir = setup_run_dir(Config)
    env, eval_env = Monitor(RacingEnv()), Monitor(RacingEnv())
    
    model = SAC(
        "MlpPolicy",
        env,
        learning_rate=linear_schedule(Config.LEARNING_RATE),
        gamma=Config.GAMMA,
        use_sde=Config.USE_SDE,
        tensorboard_log=run_dir,
        device="cpu",
        policy_kwargs=dict(net_arch=[256, 256])
    )

    print(f"Starting new training run in {run_dir}")
    if Config.RESUME_FROM:
        try:
            print(f"Using weights from: {Config.RESUME_FROM}")
            model.set_parameters(Config.RESUME_FROM)
        except Exception as e:
            print(f"Failed to load weights from {Config.RESUME_FROM}: {str(e)}")
            print("Starting with fresh weights instead.")
     
    callbacks = [
        EvalCallback(
            eval_env,
            eval_freq=Config.EVAL_FREQ,
            n_eval_episodes=Config.EVAL_EPISODES,
            render=True,
        ),
        SaveOnBestTrainingRewardCallback(
            check_freq=Config.CHECK_FREQ,
            save_path=run_dir,
            keep_n_models=Config.KEEP_N_MODELS
        ),
        SaveLatestCallback(
            save_freq=Config.EVAL_FREQ,
            save_path=run_dir
        )
    ]
    
    model.learn(
        total_timesteps=Config.TOTAL_TIMESTEPS, 
        progress_bar=True, 
        callback=callbacks, 
        tb_log_name="run"
    )
   
    model.save(f"{run_dir}/racing_model_final")
    env.close()
    eval_env.close()


if __name__ == "__main__":
    train()
