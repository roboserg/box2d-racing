import Box2D
import numpy as np
from Box2D.b2 import vec2

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

def get_ray_distances(world, car_pos, car_angle, ray_length=100.0):
    ray_angles = [-np.pi/2, -np.pi/3, -np.pi/6, 0, np.pi/6, np.pi/3, np.pi/2]
    distances = []
    ray_endpoints = []
    callback = RayCastCallback()
    
    for angle in ray_angles:
        total_angle = car_angle + angle
        ray_dir = vec2(np.cos(total_angle), np.sin(total_angle))
        end_point = vec2(
            car_pos[0] + ray_dir[0] * ray_length,
            car_pos[1] + ray_dir[1] * ray_length
        )
        
        callback.reset()
        world.RayCast(callback, car_pos, end_point)
        
        if callback.hit_point:
            distance = np.sqrt(
                (callback.hit_point[0] - car_pos[0])**2 +
                (callback.hit_point[1] - car_pos[1])**2
            )
            ray_endpoints.append(callback.hit_point)
        else:
            distance = ray_length
            ray_endpoints.append(end_point)
        
        distances.append(distance)
    
    return distances, ray_endpoints
