import numpy as np
import pygame
from stable_baselines3 import SAC
from stable_baselines3.common.monitor import Monitor
from racing_env import RacingEnv
from utils import find_latest_model


class Config:
    RUN_DIR = "logs/SAC_2"
    NUM_EVAL_EPISODES = 10


def evaluate(run_dir, n_episodes=10):
    pygame.init()
    
    env = Monitor(RacingEnv(render_mode="human"))
    
    model_path = find_latest_model(checkpoint_dir=run_dir)
    if not model_path:
        print(f"No model found in {run_dir}. Exiting...")
        return
        
    print(f"Loading model from: {model_path}")
    model = SAC.load(model_path, device="cpu", env=env, training=False)
    
    # Evaluation loop
    rewards = []
    episode_lengths = []
    
    for episode in range(n_episodes):
        obs = env.reset()[0]
        done = False
        total_reward, steps = 0, 0
        paused = False
        
        while not done:
            restart_episode, paused = handle_key_events(paused)
                
            if restart_episode:
                obs = env.reset()[0]
                total_reward, steps = 0, 0
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
    
    print(f"\nEvaluation Results after {n_episodes} episodes:")
    print(f"Average Reward: {np.mean(rewards):.2f} ± {np.std(rewards):.2f}")
    print(f"Average Episode Length: {np.mean(episode_lengths):.1f} ± {np.std(episode_lengths):.1f}")
    
    env.close()


def handle_key_events(paused):
    """
    Handle pygame key events during evaluation.
    
    Args:
        paused: Current pause state
        
    Returns:
        tuple: (restart_episode, new_paused)
    """
    for event in pygame.event.get():
        if event.type == pygame.QUIT or (event.type == pygame.KEYDOWN and event.key == pygame.K_q):
            pygame.quit()
            exit()
            
        elif event.type == pygame.KEYDOWN:
            if event.key == pygame.K_p:  # Pause
                paused = not paused
                print("Paused" if paused else "Resumed")
                
            elif event.key == pygame.K_r:  # Restart episode
                print("Restarting episode...")
                restart_episode = True
                return restart_episode, paused
    
    restart_episode = False            
    return restart_episode, paused


if __name__ == "__main__":
    evaluate(Config.RUN_DIR, Config.NUM_EVAL_EPISODES)
