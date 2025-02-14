import gymnasium as gym
import pygame
import numpy as np
from racing_env import RacingEnv

def main():
    try:
        # Initialize pygame first
        if not pygame.get_init():
            pygame.init()
            if not pygame.display.get_init():
                pygame.display.init()
        
        # Initialize environment
        env = RacingEnv(render_mode="human")
        observation, info = env.reset()
        clock = pygame.time.Clock()

        running = True
        while running:
            # Process input
            throttle, steering, drift = 0.0, 0.0, 0.0
            
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                elif event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE:
                    running = False

            # Get pressed keys
            keys = pygame.key.get_pressed()
            
            # Throttle (Up/Down arrows)
            if keys[pygame.K_UP]:
                throttle = 1.0
            elif keys[pygame.K_DOWN]:
                throttle = -1.0
                
            # Steering (Left/Right arrows)
            if keys[pygame.K_LEFT]:
                steering = -1.0
            elif keys[pygame.K_RIGHT]:
                steering = 1.0
                
            # Drift (Spacebar)
            if keys[pygame.K_SPACE]:
                drift = 1.0

            # Take action
            action = np.array([throttle, steering, drift])
            observation, reward, terminated, truncated, info = env.step(action)

            # Render the environment
            env.render()
            
            if terminated or truncated:
                observation, info = env.reset()

            clock.tick(60)  # 60 FPS

    except Exception as e:
        print(f"Error: {e}")
    finally:
        try:
            env.close()
        except:
            pass
        pygame.quit()

if __name__ == "__main__":
    main()
