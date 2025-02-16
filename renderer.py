import pygame
import numpy as np
import random
from collections import deque

class Particle:
    def __init__(self, pos, velocity, color=(128, 128, 128), lifetime=0.5):
        self.x, self.y = pos
        self.vx, self.vy = velocity
        self.color = color
        self.lifetime = lifetime
        self.age = 0.0
        self.size = random.randint(4, 8)
        self.alpha = 255
        
    def update(self, dt):
        self.x += self.vx * dt
        self.y += self.vy * dt
        self.age += dt
        self.alpha = int(255 * (1 - self.age / self.lifetime))
        self.size = max(1, self.size - dt * 4)
        
    def is_alive(self):
        return self.age < self.lifetime

class SkidMark:
    def __init__(self, pos, lifetime=2.0):
        self.x, self.y = pos
        self.lifetime = lifetime
        self.age = 0.0
        self.alpha = 255
        self.width = 4  # Width of skid mark

    def update(self, dt):
        self.age += dt
        # Fade out gradually in the last 0.5 seconds
        fade_start = self.lifetime - 0.5
        if self.age > fade_start:
            self.alpha = int(255 * (1 - (self.age - fade_start) / 0.5))
        
    def is_alive(self):
        return self.age < self.lifetime

class Renderer:
    def __init__(self, render_mode=None):
        # Initialize pygame first
        if not pygame.get_init():
            pygame.init()
        if not pygame.display.get_init():
            pygame.display.init()
            
        self.render_mode = render_mode
        self.screen = None
        self.clock = None
        self.isopen = True
        self.SCREEN_WIDTH = 2000
        self.SCREEN_HEIGHT = 1200
        self.FPS = 60
        
        # Calculate PPM based on track size (from racing_env.py track_radius = 21.0)
        track_radius = 21.0  # Match with racing_env.py
        margin = 100  # Pixels margin from screen edges
        
        # Calculate PPM to fit track with margin - reduced divisor to make track larger
        self.PPM = min(
            (self.SCREEN_WIDTH - 2 * margin) / (2.5 * track_radius),  # Changed from 4 to 2.5
            (self.SCREEN_HEIGHT - 2 * margin) / (2.5 * track_radius)  # Changed from 4 to 2.5
        )
        
        # Center the track
        self.TRACK_OFFSET_X = self.SCREEN_WIDTH / 2
        self.TRACK_OFFSET_Y = self.SCREEN_HEIGHT / 2
        
        # Define colors
        self.GRAY = (70, 70, 70)
        self.WHITE = (255, 255, 255)
        self.BLACK = (0, 0, 0)
        self.RED = (255, 0, 0)
        self.GREEN = (0, 255, 0)
        self.YELLOW = (255, 255, 0)
        
        self.font = None
        self.particles = []
        self.skid_marks = deque(maxlen=1000)  # Limit total number of skid marks
        self.init_pygame()
        
        # Speed graph settings - moved to center top
        self.speed_history = []
        self.max_history = 100  # Number of speed points to show
        graph_width = 300  # Increased width
        graph_height = 150  # Increased height
        # Position graph in center top, below stats text
        self.graph_rect = pygame.Rect(
            (self.SCREEN_WIDTH - graph_width) // 2,  # Centered horizontally
            50,  # Just below the stats text
            graph_width,
            graph_height
        )
        self.graph_color = (0, 255, 0)  # Green line
        self.graph_bg = (0, 0, 0, 128)  # Semi-transparent black background

    def init_pygame(self):
        if not pygame.get_init():
            pygame.init()
        if self.font is None:
            pygame.font.init()
            self.font = pygame.font.Font(None, 36)
        
    def to_screen(self, point):
        """Convert world coordinates to screen coordinates with centering offset"""
        return (
            int(point[0] * self.PPM + self.TRACK_OFFSET_X),
            int(self.SCREEN_HEIGHT - (point[1] * self.PPM + self.TRACK_OFFSET_Y))
        )
        
    def spawn_drift_particles(self, car):
        if car.is_drifting():
            forward_normal = car.body.GetWorldVector(localVector=(1, 0))
            right_normal = car.body.GetWorldVector(localVector=(0, 1))
            
            # Create skid marks at both rear wheels
            rear_pos = car.get_position() - forward_normal * (car.length / 2)
            rear_left = rear_pos + right_normal * (car.width / 2)
            rear_right = rear_pos - right_normal * (car.width / 2)
            
            # Add skid marks
            self.skid_marks.append(SkidMark(self.to_screen(rear_left)))
            self.skid_marks.append(SkidMark(self.to_screen(rear_right)))
            
            # Original particle effects for smoke
            pos = self.to_screen(rear_pos)
            for _ in range(2):
                velocity = (random.uniform(-100, 100), random.uniform(-100, 100))
                self.particles.append(Particle(pos, velocity, color=(128, 128, 128)))
        
        if car.is_braking():
            forward_normal = car.body.GetWorldVector(localVector=(1, 0))
            right_normal = car.body.GetWorldVector(localVector=(0, 1))
            rear_pos = car.get_position() - forward_normal * (car.length / 2)
            rear_left = rear_pos + right_normal * (car.width / 2)
            rear_right = rear_pos - right_normal * (car.width / 2)
            
            # Add brake skid marks
            self.skid_marks.append(SkidMark(self.to_screen(rear_left)))
            self.skid_marks.append(SkidMark(self.to_screen(rear_right)))
            
            # Original brake particles
            for pos in [self.to_screen(rear_left), self.to_screen(rear_right)]:
                velocity = (random.uniform(-50, 50), random.uniform(-20, 20))
                self.particles.append(Particle(pos, velocity, color=(0, 0, 0)))

    def update_effects(self, dt):
        # Update particles
        for p in self.particles[:]:
            p.update(dt)
            if not p.is_alive():
                self.particles.remove(p)
        
        # Update skid marks
        for mark in list(self.skid_marks):
            mark.update(dt)
            if not mark.is_alive():
                self.skid_marks.remove(mark)

    def render(self, car, outer_track, inner_track, ray_endpoints=None, mode='human', 
              step_count=0, cumulative_reward=0.0, show_ray_distances=False):
        if not self.isopen: return None
        self.init_pygame()
        
        if self.screen is None:
            self.screen = pygame.display.set_mode((self.SCREEN_WIDTH, self.SCREEN_HEIGHT))
            self.clock = pygame.time.Clock()

        self.screen.fill(self.GRAY)
        
        # Draw skid marks on separate surface for alpha blending
        skid_surface = pygame.Surface((self.SCREEN_WIDTH, self.SCREEN_HEIGHT), pygame.SRCALPHA)
        for mark in self.skid_marks:
            color = (30, 30, 30, mark.alpha)  # Dark gray with alpha
            pygame.draw.circle(skid_surface, color, (int(mark.x), int(mark.y)), mark.width)
        self.screen.blit(skid_surface, (0, 0))
        
        # Draw track
        pygame.draw.lines(self.screen, self.WHITE, True,
                         [self.to_screen(p) for p in outer_track], 5)
        pygame.draw.lines(self.screen, self.BLACK, True,
                         [self.to_screen(p) for p in inner_track], 5)
        
        # Draw car
        car_pos = car.get_position()
        car_angle = car.get_angle()
        
        # Create car polygon (adjusted for smaller size)
        vertices = []
        for x, y in [(-0.5, -0.25), (0.5, -0.25), (0.5, 0.25), (-0.5, 0.25)]:  # Smaller vertices
            vertex = (
                car_pos[0] + x * np.cos(car_angle) - y * np.sin(car_angle),
                car_pos[1] + x * np.sin(car_angle) + y * np.cos(car_angle)
            )
            vertices.append(self.to_screen(vertex))
        
        # Draw car body
        pygame.draw.polygon(self.screen, self.RED, vertices)
        
        # Draw front indicator (adjusted size)
        front_center = (
            car_pos[0] + 0.6 * np.cos(car_angle),  # Reduced from 2.2
            car_pos[1] + 0.6 * np.sin(car_angle)
        )
        front_left = (
            car_pos[0] + 0.5 * np.cos(car_angle) - 0.2 * np.sin(car_angle),  # Reduced from 1.8 and 0.4
            car_pos[1] + 0.5 * np.sin(car_angle) + 0.2 * np.cos(car_angle)
        )
        front_right = (
            car_pos[0] + 0.5 * np.cos(car_angle) + 0.2 * np.sin(car_angle),
            car_pos[1] + 0.5 * np.sin(car_angle) - 0.2 * np.cos(car_angle)
        )
        front_triangle = [
            self.to_screen(front_center),
            self.to_screen(front_left),
            self.to_screen(front_right)
        ]
        pygame.draw.polygon(self.screen, self.WHITE, front_triangle)

        # Draw rays and their distances
        if ray_endpoints:
            car_pos_screen = self.to_screen(car.get_position())
            car_pos = car.get_position()
            for endpoint in ray_endpoints:
                endpoint_screen = self.to_screen(endpoint)
                # Draw the ray with semi-transparent red
                ray_surface = pygame.Surface((self.SCREEN_WIDTH, self.SCREEN_HEIGHT), pygame.SRCALPHA)
                pygame.draw.line(ray_surface, (255, 0, 0, 64), car_pos_screen, endpoint_screen, 2)
                self.screen.blit(ray_surface, (0, 0))
                
                # Only show distance text if enabled
                if show_ray_distances:
                    # Calculate distance
                    distance = np.sqrt(
                        (endpoint[0] - car_pos[0])**2 +
                        (endpoint[1] - car_pos[1])**2
                    )
                    
                    # Position the text in the middle of the ray
                    text_pos = (
                        (car_pos_screen[0] + endpoint_screen[0]) // 2,
                        (car_pos_screen[1] + endpoint_screen[1]) // 2
                    )
                    
                    # Render distance text
                    distance_text = f"{distance:.1f}"
                    text_surface = self.font.render(distance_text, True, self.YELLOW)
                    # Add a black outline/background for better visibility
                    text_background = self.font.render(distance_text, True, self.BLACK)
                    text_rect = text_surface.get_rect(center=text_pos)
                    
                    # Draw text with offset for outline effect
                    for dx, dy in [(-1,-1), (-1,1), (1,-1), (1,1)]:
                        self.screen.blit(text_background, 
                                       (text_rect.x + dx, text_rect.y + dy))
                    self.screen.blit(text_surface, text_rect)

        # Draw particles
        particle_surface = pygame.Surface((self.SCREEN_WIDTH, self.SCREEN_HEIGHT), pygame.SRCALPHA)
        for p in self.particles:
            # Use the particle's own color instead of forcing grey
            color = (*p.color, p.alpha)  # Convert RGB to RGBA using particle's color
            pygame.draw.circle(particle_surface, color, (int(p.x), int(p.y)), int(p.size))
        self.screen.blit(particle_surface, (0, 0))

        # Update speed history
        forward_velocity = car.get_forward_velocity().length
        self.speed_history.append(forward_velocity)
        if len(self.speed_history) > self.max_history:
            self.speed_history.pop(0)
            
        # Draw speed graph
        graph_surface = pygame.Surface((self.graph_rect.width, self.graph_rect.height), pygame.SRCALPHA)
        pygame.draw.rect(graph_surface, self.graph_bg, graph_surface.get_rect())
        
        # Draw graph lines
        if len(self.speed_history) > 1:
            # Keep track of all-time max speed
            if not hasattr(self, 'all_time_max_speed'):
                self.all_time_max_speed = max(self.speed_history)
            else:
                self.all_time_max_speed = max(self.all_time_max_speed, max(self.speed_history))
            
            # Use all-time max for scaling, with minimum of 20
            max_speed = max(self.all_time_max_speed, 20)
            
            # Draw points
            points = []
            for i, speed in enumerate(self.speed_history):
                x = i * (self.graph_rect.width / self.max_history)
                y = self.graph_rect.height * (1 - speed / max_speed)
                points.append((x, y))
            
            # Draw left side ticks (0 and max)
            left_ticks = [
                {"value": 0, "y": self.graph_rect.height},  # Bottom (0)
                {"value": max_speed, "y": 0}  # Top (max)
            ]
            
            # Draw left ticks and labels
            for tick in left_ticks:
                value = tick["value"]
                y_pos = int(tick["y"])
                
                # Draw label on left
                label = self.font.render(f"{value:.1f}", True, (255, 255, 255))
                graph_surface.blit(label, (-10 - label.get_width(), y_pos - label.get_height() / 2))
                
                # Draw tick mark
                pygame.draw.line(graph_surface, (255, 255, 255),
                               (-5, y_pos), (5, y_pos), 2)
            
            # Draw current speed on right side
            current_y = int(self.graph_rect.height * (1 - forward_velocity / max_speed))
            
            # Draw horizontal reference line for current speed
            pygame.draw.line(graph_surface, (255, 255, 255, 64),
                           (0, current_y), (self.graph_rect.width, current_y), 1)
            
            # Draw current speed label on right
            current_label = self.font.render(f"{forward_velocity:.1f}", True, (255, 255, 255))
            graph_surface.blit(current_label, 
                             (self.graph_rect.width + 10, 
                              current_y - current_label.get_height() / 2))
            
            # Draw the speed graph
            pygame.draw.lines(graph_surface, self.graph_color, False, points, 2)
            
            # Draw axes last to ensure they're on top
            pygame.draw.line(graph_surface, (255, 255, 255), 
                           (0, 0), (0, self.graph_rect.height), 2)  # Y axis
            pygame.draw.line(graph_surface, (255, 255, 255),
                           (0, self.graph_rect.height), 
                           (self.graph_rect.width, self.graph_rect.height), 2)  # X axis
        
        # Add speed label
        speed_label = self.font.render(f"{forward_velocity:.1f}", True, (255, 255, 255))
        graph_surface.blit(speed_label, (5, 5))
        
        # Blit graph to main screen
        self.screen.blit(graph_surface, self.graph_rect)
        
        # Draw stats
        current_fps = int(self.clock.get_fps())  # Get current FPS
        drift_active = "Yes" if car.current_action['drift'] else "No"
        stats_text = f"Step: {step_count} | Reward: {cumulative_reward:.1f} | Speed: {forward_velocity:.1f} | Drift Key: {drift_active} | FPS: {current_fps}"
        text_surface = self.font.render(stats_text, True, self.BLACK)
        text_rect = text_surface.get_rect(center=(self.SCREEN_WIDTH // 2, 20))
        self.screen.blit(text_surface, text_rect)

        # Update effects
        self.spawn_drift_particles(car)
        self.update_effects(1.0 / self.FPS)

        pygame.display.flip()
        self.clock.tick(self.FPS)
        
        return True

    def close(self):
        if self.screen is not None:
            pygame.display.quit()
        self.isopen = False
