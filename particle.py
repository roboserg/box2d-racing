import random

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
        # Fade out as particle ages
        self.alpha = int(255 * (1 - self.age / self.lifetime))
        self.size = max(1, self.size - dt * 4)
        
    def is_alive(self):
        return self.age < self.lifetime
