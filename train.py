import os
import glob
import re
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize, VecVideoRecorder
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.callbacks import CheckpointCallback, EvalCallback
from racing_env import RacingEnv

def find_latest_model(checkpoint_dir="./checkpoints/"):
    # Check for final model first
    final_model = os.path.join(checkpoint_dir, "racing_model_final.zip")
    if os.path.exists(final_model):
        return final_model
    
    # Find all checkpoint files
    files = glob.glob(os.path.join(checkpoint_dir, "racing_model_*.zip"))
    if not files:
        return None
        
    # Extract step numbers and find max
    steps = []
    for f in files:
        match = re.search(r'racing_model_(\d+)_steps\.zip', f)
        if match:
            steps.append((int(match.group(1)), f))
    
    return max(steps)[1] if steps else None


def train(total_timesteps):
    # Create environments
    env = DummyVecEnv([lambda: Monitor(RacingEnv())])
    env = VecNormalize(env, norm_obs=True, norm_reward=False, clip_obs=10.)
    
    eval_env = DummyVecEnv([lambda: Monitor(RacingEnv(render_mode="human"))])
    eval_env = VecNormalize(eval_env, norm_obs=True, norm_reward=False, clip_obs=10.)
    
    # Sync normalization stats
    eval_env.obs_rms = env.obs_rms
    eval_env.ret_rms = env.ret_rms
    
    # Initialize PPO agent
    model = PPO(
        "MlpPolicy",
        env,
        learning_rate=3e-4,
        n_steps=2048,
        batch_size=128,
        n_epochs=10,
        gamma=0.95,
        verbose=0,
        tensorboard_log="./logs/"
    )
    
    # Load latest model if available
    latest_model = find_latest_model()
    if (latest_model):
        print(f"Loading parameters from existing model: {latest_model}")
        model.set_parameters(latest_model)
    
    # Setup callbacks
    video_folder = "./videos"
    os.makedirs(video_folder, exist_ok=True)
    
    callbacks = [
        CheckpointCallback(
            save_freq=100_000,
            save_path="./checkpoints/",
            name_prefix="racing_model"
        ),
        VideoRecordCallback(
            eval_env=eval_env,
            video_folder=video_folder,
            eval_freq=10_000,
            n_eval_episodes=1,
            deterministic=True
        )
    ]
    
    # Train
    model.learn(
        total_timesteps=total_timesteps,
        callback=callbacks,
        progress_bar=True
    )
    
    # Save final model and cleanup
    model.save("./checkpoints/racing_model_final")
    env.close()
    eval_env.close()

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--timesteps", type=int, default=10_000_000,
                      help="Total timesteps to train for")
    
    args = parser.parse_args()
    train(args.timesteps)
