import pygame
import numpy as np
from collections import deque
from car import Particle, SkidMark
import random

class Renderer:
    def __init__(self, render_mode='human'):
        self.render_mode = render_mode
        self.isopen = True
        self.screen = None
        self.clock = None
        self.particles = []
        self.skid_marks = []
        self.crash_locations = []  # Store crash locations
        
        # Colors
        self.GRAY = (70, 70, 70)
        self.WHITE = (255, 255, 255)
        self.BLACK = (0, 0, 0)
        self.GREEN = (0, 255, 0)
        self.YELLOW = (255, 255, 0)
        
        self.font = None
        self.SCREEN_WIDTH = 2000
        self.SCREEN_HEIGHT = 1200
        self.FPS = 60
        
        # Initialize with temporary values
        self.PPM = 10
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
        
        # Pre-compute common transformations
        self.screen_transform = np.array([
            [self.PPM, 0],
            [0, -self.PPM]
        ])
        self.screen_offset = np.array([self.TRACK_OFFSET_X, self.SCREEN_HEIGHT - self.TRACK_OFFSET_Y])
        
        # Initialize other attributes
        self.init_pygame()
        
        # Speed graph settings
        self.speed_history = []
        self.max_history = 100
        graph_width = 300
        graph_height = 150
        margin = 20  # Margin from screen edges
        
        self.graph_rect = pygame.Rect(
            self.SCREEN_WIDTH - graph_width - margin,  # Right align with margin
            self.SCREEN_HEIGHT - graph_height - margin,  # Bottom align with margin
            graph_width,
            graph_height
        )
        self.graph_color = (0, 255, 0)
        self.graph_bg = (0, 0, 0, 128)

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
        if car.is_physically_drifting():
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

    def _render_rays(self, car_pos, ray_endpoints, show_ray_distances=False):
        """
        Render ray sensors and their distance labels.
        
        Args:
            car_pos: numpy array or tuple of car's position
            ray_endpoints: list of ray endpoint positions
            show_ray_distances: boolean to toggle distance label display
        
        Returns:
            list of surfaces to be blitted to the main screen
        """
        if not ray_endpoints:
            return []
        
        surfaces_to_blit = []
        
        # Convert to numpy arrays for vectorized operations
        car_pos = np.array(car_pos)
        car_pos_screen = np.array(self.to_screen(car_pos))
        endpoints = np.array([self.to_screen(ep) for ep in ray_endpoints])
        
        # Create ray surface
        ray_surface = pygame.Surface((self.SCREEN_WIDTH, self.SCREEN_HEIGHT), pygame.SRCALPHA)
        for endpoint in endpoints:
            pygame.draw.line(ray_surface, (0, 255, 0, 64), car_pos_screen, endpoint, 2)
        surfaces_to_blit.append((ray_surface, (0, 0)))
        
        if show_ray_distances:
            # Vectorized distance calculation
            ray_vectors = np.array(ray_endpoints) - car_pos
            distances = np.linalg.norm(ray_vectors, axis=1)
            
            # Calculate text positions vectorized
            midpoints = (car_pos_screen + endpoints) // 2
            
            # Pre-render all distance texts
            distance_texts = [f"{d:.1f}" for d in distances]
            text_surfaces = [self.font.render(text, True, self.YELLOW) for text in distance_texts]
            text_backgrounds = [self.font.render(text, True, self.BLACK) for text in distance_texts]
            text_rects = [surf.get_rect(center=pos) for surf, pos in zip(text_surfaces, midpoints)]
            
            # Create text surface
            text_surface = pygame.Surface((self.SCREEN_WIDTH, self.SCREEN_HEIGHT), pygame.SRCALPHA)
            
            # Draw all texts with outline effect
            offsets = np.array([(-1,-1), (-1,1), (1,-1), (1,1)])
            for text_bg, text_surf, rect in zip(text_backgrounds, text_surfaces, text_rects):
                # Draw outline
                for offset in offsets:
                    text_surface.blit(text_bg, (rect.x + offset[0], rect.y + offset[1]))
                # Draw main text
                text_surface.blit(text_surf, rect)
            
            surfaces_to_blit.append((text_surface, (0, 0)))
        
        return surfaces_to_blit

    def _render_speed_graph(self, forward_velocity):
        """
        Render the speed graph with history, labels, and axes.
        
        Args:
            forward_velocity: current forward velocity of the car
            
        Returns:
            Surface: rendered graph surface
        """
        # Create graph surface
        graph_surface = pygame.Surface((self.graph_rect.width, self.graph_rect.height), pygame.SRCALPHA)
        pygame.draw.rect(graph_surface, self.graph_bg, graph_surface.get_rect())
        
        # Update speed history
        self.speed_history.append(forward_velocity)
        if len(self.speed_history) > self.max_history:
            self.speed_history.pop(0)
        
        # Draw graph lines
        if len(self.speed_history) > 1:
            # Keep track of all-time max speed
            if not hasattr(self, 'all_time_max_speed'):
                self.all_time_max_speed = max(self.speed_history)
            else:
                self.all_time_max_speed = max(self.all_time_max_speed, max(self.speed_history))
            
            # Use all-time max for scaling, with minimum of 20
            max_speed = max(self.all_time_max_speed, 20)
            
            # Draw points using numpy for vectorization
            points = np.array([
                (i * (self.graph_rect.width / self.max_history),
                 self.graph_rect.height * (1 - speed / max_speed))
                for i, speed in enumerate(self.speed_history)
            ])
            
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
        
        return graph_surface
    
    def configure_display(self, outer_track):
        """Calculate PPM and offsets based on track boundaries"""
        points = np.array(outer_track)
        min_x, min_y = np.min(points, axis=0)
        max_x, max_y = np.max(points, axis=0)
        
        track_width = max_x - min_x
        track_height = max_y - min_y
        
        self.PPM = min(
            self.SCREEN_WIDTH * 0.58 / track_width,
            self.SCREEN_HEIGHT * 0.58 / track_height
        )
        
        # Center the track
        center_x = (min_x + max_x) / 2
        center_y = (min_y + max_y) / 2
        
        # Update screen transform matrix and offset
        self.TRACK_OFFSET_X = self.SCREEN_WIDTH/2 - center_x * self.PPM 
        self.TRACK_OFFSET_Y = self.SCREEN_HEIGHT/2 + center_y * self.PPM + 70
        
        # Update the screen transform matrix
        self.screen_transform = np.array([
            [self.PPM, 0],
            [0, -self.PPM]
        ])
        self.screen_offset = np.array([self.TRACK_OFFSET_X, self.SCREEN_HEIGHT - self.TRACK_OFFSET_Y])

    def render(self, car, outer_track, inner_track, ray_endpoints=None, mode='human', 
              step_count=0, cumulative_reward=0.0, show_ray_distances=False):
        if not self.isopen: return None
        
        self.configure_display(outer_track)
            
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
        pygame.draw.lines(self.screen, self.BLACK, True,
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
        
        # Draw front indicator line
        front_start = (
            car_pos[0] + 0.4 * np.cos(car_angle),
            car_pos[1] + 0.4 * np.sin(car_angle)
        )
        front_end = (
            car_pos[0] + 0.9 * np.cos(car_angle),
            car_pos[1] + 0.9 * np.sin(car_angle)
        )
        pygame.draw.line(self.screen, self.BLACK, 
                        self.to_screen(front_start), 
                        self.to_screen(front_end), 
                        4)

        # Draw rays
        ray_surfaces = self._render_rays(car_pos, ray_endpoints, show_ray_distances)
        for surface, pos in ray_surfaces:
            self.screen.blit(surface, pos)
            
        # Draw black circles at ray endpoints
        if ray_endpoints:
            for endpoint in ray_endpoints:
                screen_pos = self.to_screen(endpoint)
                pygame.draw.circle(self.screen, self.GREEN, screen_pos, 3)

        # Draw particles
        particle_surface = pygame.Surface((self.SCREEN_WIDTH, self.SCREEN_HEIGHT), pygame.SRCALPHA)
        for p in self.particles:
            color = (*p.color, p.alpha)  # Convert RGB to RGBA using particle's color
            pygame.draw.circle(particle_surface, color, (int(p.x), int(p.y)), int(p.size))
        self.screen.blit(particle_surface, (0, 0))

        # Get and render speed graph
        forward_velocity = car.get_forward_velocity().length
        graph_surface = self._render_speed_graph(forward_velocity)
        self.screen.blit(graph_surface, self.graph_rect)
        
        # Draw stats
        current_fps = int(self.clock.get_fps())
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
