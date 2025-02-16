import numpy as np
from Box2D.b2 import edgeShape

class Track:
    def __init__(self, world, radius=21.0, num_points=31):
        self.world = world
        self.radius = radius
        self.num_points = num_points
        self.center = (0, 0)
        
        # Track boundaries
        self.outer_points = []
        self.inner_points = []
        
        # Generate and create the track
        self._generate_track_points()
        self._create_physical_boundaries()
    
    def _generate_track_points(self):
        """Generate the points that define the track boundaries."""
        # Generate outer track points
        self.outer_points = []
        for i in range(self.num_points):
            angle = (i / self.num_points) * 2 * np.pi
            # Add some variation to radius for more interesting track
            r = self.radius * (1 + 0.2 * np.sin(3 * angle))
            x = r * np.cos(angle)
            y = r * np.sin(angle)
            self.outer_points.append((x, y))
            
        # Generate inner track points with wider gap
        self.inner_points = []
        inner_radius = self.radius * 0.6  # Increased track width by reducing this ratio
        for i in range(self.num_points):
            angle = (i / self.num_points) * 2 * np.pi
            r = inner_radius * (1 + 0.2 * np.sin(3 * angle))
            x = r * np.cos(angle)
            y = r * np.sin(angle)
            self.inner_points.append((x, y))
    
    def _create_physical_boundaries(self):
        """Create physical boundaries in the Box2D world."""
        for points in [self.outer_points, self.inner_points]:
            for i in range(len(points)):
                p1 = points[i]
                p2 = points[(i + 1) % len(points)]
                self.world.CreateStaticBody(shapes=edgeShape(vertices=[p1, p2]))
    
    def get_start_position(self):
        """Get the recommended starting position for the car."""
        return (self.center[0] - self.radius * 0.85, self.center[1])
    
    def get_boundaries(self):
        """Get the track boundary points."""
        return self.outer_points, self.inner_points