from stable_baselines3 import PPO, SAC
from stable_baselines3.common.callbacks import EvalCallback
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv, VecMonitor

from callbacks import SaveLatestCallback, SaveOnBestTrainingRewardCallback
from config import Config
from racing_env import RacingEnv
from utils import linear_schedule, load_model_weights, make_env, setup_run_dir


def train():
    env = VecMonitor(SubprocVecEnv([make_env(i) for i in range(Config.NUM_ENVS)]))
    eval_env = VecMonitor(RacingEnv())
    run_dir = setup_run_dir(Config)

    model = SAC(
        "MlpPolicy",
        env,
        learning_rate=linear_schedule(Config.LEARNING_RATE),
        gamma=Config.GAMMA,
        use_sde=Config.USE_SDE,
        tensorboard_log=run_dir,
        device="cpu",
        policy_kwargs=dict(net_arch=[256, 256]),
    )

    if Config.RESUME_FROM:
        load_model_weights(model, Config.RESUME_FROM)

    callbacks = [
        EvalCallback(
            eval_env,
            eval_freq=Config.EVAL_FREQ // Config.NUM_ENVS,
            n_eval_episodes=Config.EVAL_EPISODES,
            render=True,
        ),
        SaveOnBestTrainingRewardCallback(
            check_freq=Config.CHECK_FREQ // Config.NUM_ENVS, save_path=run_dir, keep_n_models=Config.KEEP_N_MODELS
        ),
        SaveLatestCallback(save_freq=Config.EVAL_FREQ // Config.NUM_ENVS, save_path=run_dir),
    ]

    model.learn(total_timesteps=Config.TOTAL_TIMESTEPS, progress_bar=True, callback=callbacks, tb_log_name="run")
    model.save(f"{run_dir}/racing_model_final")
    env.close()
    eval_env.close()


if __name__ == "__main__":
    train()
