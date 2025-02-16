import pygame
import gymnasium as gym
from gymnasium import spaces
import numpy as np
import Box2D
from Box2D.b2 import world
from car import Car
from track import Track
from physics_utils import ContactListener, get_ray_distances

class RacingEnv(gym.Env):
    def __init__(self, render_mode=None):
        super().__init__()
        self.render_mode = render_mode
        self.FPS = 60
        
        # Action space: [throttle, steering, drift]
        self.action_space = spaces.Box(
            low=np.array([-1, -1, 0]), high=np.array([1, 1, 1]), dtype=np.float32)
        
        # Observation space: [car_x, car_y, car_angle, car_speed, car_angular_vel, distances_to_track_edges, collision_flag]
        self.observation_space = spaces.Box(
            low=np.array([-np.inf] * 13, dtype=np.float32),   # 5 car states + 7 rays + 1 collision flag
            high=np.array([np.inf] * 13, dtype=np.float32))
        
        self.world = None
        self.car = None
        self.track = None
        self.renderer = None
        self.step_count = 0
        self.max_steps = 5000
        self.cumulative_reward = 0.0
        self.car_touched_boundary = False
        
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.step_count = self.cumulative_reward = 0.0
        self.car_touched_boundary = False
        
        self.world = world(gravity=(0, 0), doSleep=True)  # Create world
        self.track = Track(self.world)  # Create track
        
        # Create car at starting position with random angle
        start_pos = self.track.get_start_position()
        self.car = Car(self.world, start_pos, angle=np.random.uniform(0, 2*np.pi))
        
        self.contact_listener = ContactListener(self)
        self.world.contactListener = self.contact_listener
        
        return self._get_observation(), {}

    def _get_observation(self):
        obs = np.array([
            *self.car.get_position(),          # x, y
            self.car.get_heading(),            # angle
            self.car.get_linear_velocity(),    # velocity
            self.car.get_angular_velocity(),   # angular velocity
            *self._get_ray_distances(),        # ray distances
            float(self.car_touched_boundary)   # collision flag
        ])
        return self._normalize_observation(obs)

    def _normalize_observation(self, obs):
        obs[0:2] /= 100.0    # Position (x,y) -> [-1,1]
        obs[2] /= np.pi      # Angle [-pi,pi] -> [-1,1]
        obs[3] /= 20.0       # Linear vel [-20,20] -> [-1,1]
        obs[4] /= 5.0        # Angular vel [-5,5] -> [-1,1]
        obs[5:12] /= 100.0   # Ray distances [0,100] -> [0,1]
        return obs           # obs[12] is binary collision flag (0,1)

    def _print_observation_table(self, observation):
        headers = ["X", "Y", "Car Angle", "Car Speed", "Car Angular Vel"] + \
                 [f"Ray {i+1}" for i in range(7)] + ["Collision Flag"]
        row_format = "{:>15}" * len(headers)
        print(row_format.format(*headers))
        print(row_format.format(*[f"{x:.2f}" for x in observation]))

    def _get_ray_distances(self):
        distances, self.ray_endpoints = get_ray_distances(
            self.world, self.car.get_position(), self.car.get_angle())
        return distances

    def step(self, action):
        self.step_count += 1
        # Convert normalized action to car controls
        car_action = {
            'throttle': float(action[0]),
            'steer': float(action[1]),
            'drift': bool(action[2] > 0)
        }
        
        self.car.step(car_action)  # Update physics
        self.world.Step(1.0/self.FPS, 6, 2)
        
        obs = self._get_observation()
        reward = self._calculate_reward()
        self.cumulative_reward += reward
        
        terminated = self._is_done()
        return obs, reward, terminated, False, {}

    def _calculate_reward(self):
        reward = self.car.get_linear_velocity() / 1000
        if self.car_touched_boundary: reward -= 1.0  # Add penalty for touching boundary
        return reward

    def _is_done(self):
        return self.step_count >= self.max_steps or self.car_touched_boundary

    def render(self):
        if self.render_mode is None: return None
            
        if self.renderer is None:
            if not pygame.get_init(): pygame.init()
            if not pygame.display.get_init(): pygame.display.init()
            from renderer import Renderer
            self.renderer = Renderer(self.render_mode)
        
        outer_track, inner_track = self.track.get_boundaries()
        return self.renderer.render(
            self.car, outer_track, inner_track,
            ray_endpoints=self.ray_endpoints,
            mode=self.render_mode,
            step_count=self.step_count,
            cumulative_reward=self.cumulative_reward,
            show_ray_distances=False
        )

    def close(self):
        if self.renderer: self.renderer.close()
