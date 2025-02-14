import os
import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from stable_baselines3.common.monitor import Monitor
from racing_env import RacingEnv
from train import find_latest_model

def evaluate(n_episodes=10, render=True):
    # Create environment
    env = RacingEnv(render_mode="human" if render else None)
    # Ensure the environment is using the correct action space
    env = Monitor(env)
    env = DummyVecEnv([lambda: env])
    env = VecNormalize(env, norm_obs=True, norm_reward=False, clip_obs=10.)
    
    # Load latest model
    model_path = find_latest_model()
    if not model_path:
        print("No model found in checkpoints directory!")
        return
        
    print(f"Loading model from: {model_path}")
    model = PPO.load(model_path)
    model.device = "cpu"
    
    # Evaluation loop
    rewards = []
    episode_lengths = []
    
    for episode in range(n_episodes):
        obs = env.reset()[0]
        done = False
        total_reward = 0
        steps = 0
        
        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            total_reward += reward
            steps += 1
            
        rewards.append(total_reward)
        episode_lengths.append(steps)
        print(f"Episode {episode + 1}: Reward = {total_reward:.2f}, Steps = {steps}")
    
    print("\nEvaluation Results:")
    print(f"Average Reward: {np.mean(rewards):.2f} ± {np.std(rewards):.2f}")
    print(f"Average Episode Length: {np.mean(episode_lengths):.1f} ± {np.std(episode_lengths):.1f}")
    
    env.close()

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--episodes", type=int, default=10,
                      help="Number of episodes to evaluate")
    parser.add_argument("--no-render", action="store_true",
                      help="Disable rendering")
    
    args = parser.parse_args()
    evaluate(args.episodes, not args.no_render)
