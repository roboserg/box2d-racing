import numpy as np
from Box2D import b2Vec2, b2EdgeShape

class Track:
    def __init__(self, world, radius=30.0, num_points=50):
        self.world = world
        self.radius = radius
        self.num_points = num_points
        self.track_width = self.radius * 0.5
        self.center = (0, 0)
        
        # Track boundaries
        self.center_points = []
        self.outer_points = []
        self.inner_points = []
        
        # Generate track points
        self._generate_center_spline()
        self._generate_boundaries()
        self._create_physical_boundaries()
    
    def _generate_center_spline(self):
        """Generate the center spline of the track using a parametric equation."""
        self.center_points = []
        
        for t in np.linspace(0, 2*np.pi, self.num_points):
            # Complex track with multiple features:
            # - Base circle
            # - Multiple sine waves of different frequencies
            # - Asymmetric variations
            
            # Base radius variation
            r = self.radius * (1 + 0.3 * np.sin(3 * t)  # Triple-lobed variation
                             + 0.15 * np.sin(5 * t)      # Five-lobed variation
                             + 0.1 * np.cos(2 * t))      # Double-lobed asymmetric variation
            
            # Add some asymmetric displacement
            x = r * np.cos(t) + 0.2 * self.radius * np.sin(1 * t)
            y = r * np.sin(t) + 0.15 * self.radius * np.cos(1 * t)
            
            self.center_points.append(np.array([float(x), float(y)]))
    
    def _generate_boundaries(self):
        """Generate inner and outer track boundaries from center spline."""
        self.inner_points = []
        self.outer_points = []
        
        # Pre-calculate all normals first
        normals = []
        for i in range(len(self.center_points)):
            p1 = self.center_points[i]
            p2 = self.center_points[(i + 1) % len(self.center_points)]
            
            # Calculate direction and normal
            direction = p2 - p1
            normal = np.array([-direction[1], direction[0]])
            normal = normal / (np.linalg.norm(normal) + 1e-8)
            normals.append(normal)
        
        # Smooth out normals at the seam and around the track
        smoothed_normals = []
        window_size = 5  # Use 5 points for smoothing
        for i in range(len(normals)):
            # Get window of normals centered at current point
            window = []
            for j in range(-window_size//2, window_size//2 + 1):
                idx = (i + j) % len(normals)
                window.append(normals[idx])
            
            # Average the normals in the window
            avg_normal = np.mean(window, axis=0)
            avg_normal = avg_normal / (np.linalg.norm(avg_normal) + 1e-8)
            smoothed_normals.append(avg_normal)
        
        # Generate boundary points using smoothed normals
        for i in range(len(self.center_points)):
            p = self.center_points[i]
            normal = smoothed_normals[i]
            
            inner_point = p - normal * (self.track_width / 2)
            outer_point = p + normal * (self.track_width / 2)
            
            self.inner_points.append((float(inner_point[0]), float(inner_point[1])))
            self.outer_points.append((float(outer_point[0]), float(outer_point[1])))
    
    def _create_physical_boundaries(self):
        """Create physical boundaries in Box2D world."""
        # Create inner track boundaries
        for i in range(len(self.inner_points)):
            p1 = self.inner_points[i]
            p2 = self.inner_points[(i + 1) % len(self.inner_points)]
            self.world.CreateStaticBody(shapes=b2EdgeShape(vertices=[p1, p2]))
        
        # Create outer track boundaries
        for i in range(len(self.outer_points)):
            p1 = self.outer_points[i]
            p2 = self.outer_points[(i + 1) % len(self.outer_points)]
            self.world.CreateStaticBody(shapes=b2EdgeShape(vertices=[p1, p2]))
    
    def get_start_position(self):
        """Get a random starting position from the track's center points."""
        start_point = self.center_points[np.random.randint(0, len(self.center_points))]
        return (float(start_point[0]), float(start_point[1]))
    
    def get_boundaries(self):
        """Get the track boundary points."""
        return self.outer_points, self.inner_points
