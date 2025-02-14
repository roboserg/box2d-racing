import pygame
import numpy as np
import random
from particle import Particle

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
        self.SCREEN_WIDTH = 2000  # Increased from 1200
        self.SCREEN_HEIGHT = 1200  # Increased from 800
        self.PPM = 25.0  # Increased from 20.0 to make everything bigger
        self.FPS = 60
        
        # Add track offset to center it
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
        self.init_pygame()
        
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
            rear_pos = car.get_position() - forward_normal * (car.length / 2)
            pos = self.to_screen(rear_pos)
            for _ in range(2):  # Spawn multiple particles per frame
                velocity = (random.uniform(-100, 100), random.uniform(-100, 100))
                # Grey particles for drifting
                self.particles.append(Particle(pos, velocity, color=(128, 128, 128)))
        
            
        if car.is_braking():
            forward_normal = car.body.GetWorldVector(localVector=(1, 0))
            rear_left = car.get_position() - forward_normal * (car.length / 2) + \
                       car.body.GetWorldVector(localVector=(0, car.width / 2))
            rear_right = car.get_position() - forward_normal * (car.length / 2) - \
                        car.body.GetWorldVector(localVector=(0, car.width / 2))
            
            for pos in [self.to_screen(rear_left), self.to_screen(rear_right)]:
                velocity = (random.uniform(-50, 50), random.uniform(-20, 20))
                # Black particles for braking
                self.particles.append(Particle(pos, velocity, color=(0, 0, 0)))

    def update_particles(self, dt):
        for p in self.particles[:]:
            p.update(dt)
            if not p.is_alive():
                self.particles.remove(p)

    def render(self, car, outer_track, inner_track, ray_endpoints=None, mode='human', step_count=0, cumulative_reward=0.0, show_ray_distances=False):
        if not self.isopen:
            return None
            
        # Ensure pygame and font are initialized
        self.init_pygame()
        
        if self.screen is None:
            self.screen = pygame.display.set_mode((self.SCREEN_WIDTH, self.SCREEN_HEIGHT))
            self.clock = pygame.time.Clock()

        self.screen.fill(self.GRAY)
    
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

        # Draw stats
        forward_velocity = car.get_forward_velocity().length
        drift_active = "Yes" if car.current_action['drift'] else "No"
        stats_text = f"Step: {step_count} | Reward: {cumulative_reward:.1f} | Speed: {forward_velocity:.1f} | Drift Key: {drift_active}"
        text_surface = self.font.render(stats_text, True, self.BLACK)
        self.screen.blit(text_surface, (10, 10))

        # Update particles
        self.spawn_drift_particles(car)
        self.update_particles(1.0 / self.FPS)

        pygame.display.flip()
        self.clock.tick(self.FPS)
        
        return True

    def close(self):
        if self.screen is not None:
            pygame.display.quit()
        self.isopen = False
