import numpy as np
import pygame
from stable_baselines3 import SAC 
from stable_baselines3.common.monitor import Monitor
from racing_env import RacingEnv
from utils import find_latest_model

def evaluate(n_episodes=10, render=True):
    # Initialize pygame first
    pygame.init()
    pygame.display.init()
    
    env = Monitor(RacingEnv(render_mode="human" if render else None))
    
    model_path = find_latest_model()
    if not model_path:
        print(f"No model found in {model_path}. Exiting...")
        return
        
    print(f"Loading model from: {model_path}")
    model = SAC.load(model_path, device="cpu", env=env, training=False)
    
    # Evaluation loop
    rewards = []
    episode_lengths = []
    
    episode = 0
    while episode < n_episodes:
        obs = env.reset()[0]
        done = False
        total_reward, steps = 0, 0
        paused = False
        
        while not done:
            # Handle events
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    env.close()
                    return
                elif event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_q:  # Quit
                        env.close()
                        return
                    elif event.key == pygame.K_p:  # Pause
                        paused = not paused
                        print("Paused" if paused else "Resumed")
                    elif event.key == pygame.K_r:  # Restart episode
                        obs = env.reset()[0]  # Reset the environment
                        total_reward = 0
                        steps = 0
                        print("Restarting episode...")
                        continue
            
            if paused:
                env.render()
                continue
                
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            total_reward += reward
            steps += 1
            env.render()
            
        if not done:  # If episode completed normally (not restarted)
            rewards.append(total_reward)
            episode_lengths.append(steps)
            print(f"Episode {episode + 1}: Reward = {total_reward:.2f}, Steps = {steps}")
            episode += 1
    
    print(f"\nEvaluation Results after {n_episodes} episodes:")
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
