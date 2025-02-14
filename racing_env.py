import pygame
import gymnasium as gym
from gymnasium import spaces
import numpy as np
import Box2D
from Box2D.b2 import (world, polygonShape, edgeShape, vec2)
from car import Car

class RayCastCallback(Box2D.b2RayCastCallback):
    def __init__(self):
        Box2D.b2RayCastCallback.__init__(self)
        self.fixture = None
        self.hit_point = None
        self.normal = None
        self.fraction = 1.0

    def ReportFixture(self, fixture, point, normal, fraction):
        self.fixture = fixture
        self.hit_point = point
        self.normal = normal
        self.fraction = fraction
        return fraction  # Return fraction to get closest hit

    def reset(self):
        self.fixture = None
        self.hit_point = None
        self.normal = None
        self.fraction = 1.0

class ContactListener(Box2D.b2ContactListener):
    def __init__(self, env):
        Box2D.b2ContactListener.__init__(self)
        self.env = env

    def BeginContact(self, contact):
        if (contact.fixtureA.body == self.env.car.body or 
            contact.fixtureB.body == self.env.car.body):
            self.env.car_touched_boundary = True

class RacingEnv(gym.Env):
    def __init__(self, render_mode=None):
        super().__init__()
        self.render_mode = render_mode
        self.FPS = 60  # Match with renderer FPS
        
        # Action space: [throttle, steering, drift]
        self.action_space = spaces.Box(
            low=np.array([-1, -1, 0]),
            high=np.array([1, 1, 1]),
            dtype=np.float32
        )
        
        # Observation space: [car_x, car_y, car_angle, car_speed, car_angular_vel, distances_to_track_edges, collision_flag]
        self.observation_space = spaces.Box(
            low=np.array([-np.inf, -np.inf, -np.pi, -50, -50] + [0]*8 + [0], dtype=np.float32),
            high=np.array([np.inf, np.inf, np.pi, 50, 50] + [100]*8 + [1], dtype=np.float32),
        )
        
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
        self.step_count = 0
        self.cumulative_reward = 0.0
        self.car_touched_boundary = False
        
        # Create world
        self.world = world(gravity=(0, 0), doSleep=True)
        
        # Create track
        self._create_track()
        
        # Create car at starting position (adjusted to use track_radius)
        start_pos = (self.track_center[0] - self.track_radius * 0.85, self.track_center[1])
        self.car = Car(self.world, start_pos, angle=np.random.uniform(0, 2*np.pi))
        
        self.contact_listener = ContactListener(self)
        self.world.contactListener = self.contact_listener
        
        observation = self._get_observation()
        info = {}
        
        return observation, info

    def _create_track(self):
        self.outer_track, self.inner_track = self._generate_track()
        
        # Create physical boundaries
        for points in [self.outer_track, self.inner_track]:
            for i in range(len(points)):
                p1 = points[i]
                p2 = points[(i + 1) % len(points)]
                self.world.CreateStaticBody(shapes=edgeShape(vertices=[p1, p2]))

    def _generate_track(self):
        # Center the track around (0,0) in world coordinates
        self.track_radius = 20.0  # Store as instance variable
        num_points = 30     # Increase for smoother track
        
        # Set track center
        self.track_center = (0, 0)  # Track is centered at origin in world coordinates
        
        # Generate outer track points
        outer_track = []
        for i in range(num_points):
            angle = (i / num_points) * 2 * np.pi
            # Add some variation to radius for more interesting track
            r = self.track_radius * (1 + 0.2 * np.sin(3 * angle))
            x = r * np.cos(angle)
            y = r * np.sin(angle)
            outer_track.append((x, y))
            
        # Generate inner track points with wider gap (changed from 0.7 to 0.6)
        inner_track = []
        inner_radius = self.track_radius * 0.6  # Increased track width by reducing this ratio
        for i in range(num_points):
            angle = (i / num_points) * 2 * np.pi
            r = inner_radius * (1 + 0.2 * np.sin(3 * angle))
            x = r * np.cos(angle)
            y = r * np.sin(angle)
            inner_track.append((x, y))
            
        return outer_track, inner_track

    def _get_observation(self):
        car_pos = self.car.get_position()
        car_angle = self.car.get_angle()
        car_vel = self.car.get_linear_velocity()
        car_ang_vel = self.car.get_angular_velocity()
        
        # Calculate distances to track boundaries in 8 directions
        rays = self._get_ray_distances()
        
        return np.array([
            car_pos[0], car_pos[1],
            car_angle,
            car_vel.length,
            car_ang_vel
        ] + rays + [float(self.car_touched_boundary)])

    def _get_ray_distances(self):
        ray_angles = [0, np.pi/4, np.pi/2, 3*np.pi/4, np.pi, 5*np.pi/4, 3*np.pi/2, 7*np.pi/4]
        ray_length = 100.0  # Maximum ray length
        car_pos = self.car.get_position()
        car_angle = self.car.get_angle()
        
        distances = []
        self.ray_endpoints = []  # Store for visualization
        callback = RayCastCallback()
        
        for angle in ray_angles:
            total_angle = car_angle + angle
            ray_dir = vec2(np.cos(total_angle), np.sin(total_angle))
            end_point = vec2(
                car_pos[0] + ray_dir[0] * ray_length,
                car_pos[1] + ray_dir[1] * ray_length
            )
            
            callback.reset()
            self.world.RayCast(callback, car_pos, end_point)
            
            if callback.hit_point:
                distance = np.sqrt(
                    (callback.hit_point[0] - car_pos[0])**2 +
                    (callback.hit_point[1] - car_pos[1])**2
                )
                self.ray_endpoints.append(callback.hit_point)
            else:
                distance = ray_length
                self.ray_endpoints.append(end_point)
            
            distances.append(distance)
        
        return distances

    def step(self, action):
        self.step_count += 1
        # Convert normalized action to car controls
        car_action = {
            'throttle': float(action[0]),
            'steer': float(action[1]),
            'drift': bool(action[2] > 0)
        }
        
        # Update physics
        self.car.step(car_action)
        self.world.Step(1.0/self.FPS, 6, 2)
        
        # Get new state
        obs = self._get_observation()
        
        # Calculate reward
        reward = self._calculate_reward()
        self.cumulative_reward += reward
        
        # Check if episode is done
        terminated = self._is_done()
        truncated = False
        
        info = {}
        
        return obs, reward, terminated, truncated, info

    def _calculate_reward(self):
        reward = self.car.get_linear_velocity().length / 1000
        # Add penalty for touching boundary
        if self.car_touched_boundary:
            reward -= 1.0
        return reward

    def _is_done(self):
        if self.step_count >= self.max_steps or self.car_touched_boundary:
            return True
        return False

    def render(self):
        if self.render_mode is None:
            return None
            
        if self.renderer is None:
            if not pygame.get_init():
                pygame.init()
            if not pygame.display.get_init():
                pygame.display.init()
                
            from renderer import Renderer
            self.renderer = Renderer(self.render_mode)
        
        result = self.renderer.render(
            self.car,
            self.outer_track,
            self.inner_track,
            ray_endpoints=self.ray_endpoints,
            mode=self.render_mode,
            step_count=self.step_count,
            cumulative_reward=self.cumulative_reward,
            show_ray_distances=False  # Set to True if you want to see distances
        )
        
        return result

    def close(self):
        if self.renderer:
            self.renderer.close()
