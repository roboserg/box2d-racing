import Box2D
from Box2D.b2 import vec2, polygonShape
import math

class Car:
    def __init__(self, world, position, angle=0):
        self.world = world
        self.width = 0.5   # Reduced from 1.0 meters
        self.length = 1.0  # Reduced from 2.0 meters
        self.current_action = {'throttle': 0, 'steer': 0, 'drift': False}
        
        # Create body
        self.body = self.world.CreateDynamicBody(
            position=position,
            angle=angle
        )
        
        # Create fixture separately with proper shape definition
        fixture_def = {}
        fixture_def['shape'] = polygonShape(box=(self.length/2, self.width/2))
        fixture_def['density'] = 1.0
        fixture_def['friction'] = 0.1  # Reduced from 0.3 for less grip
        
        # Add fixture to body
        self.body.CreateFixture(**fixture_def)
        
        # Adjust physics parameters
        self.max_drive_force = 20.0
        self.max_lateral_impulse = 6.0
        self.brake_drag_multiplier = 4.0  # Increased from 1.5 for stronger braking
        self.base_drag = 0.2

    def get_position(self):
        return self.body.position

    def get_angle(self):
        return self.body.angle

    def get_linear_velocity(self):
        return self.body.linearVelocity

    def get_angular_velocity(self):
        return self.body.angularVelocity

    def step(self, action):
        """
        action: dict with keys: 'throttle', 'steer', 'drift'
        throttle: float [-1, 1] for backward/forward
        steer: float [-1, 1] for right/left
        drift: bool for drift mode
        """
        self.current_action = action  # Store the current action
        # Update friction
        right_normal = self.body.GetWorldVector(localVector=(0, 1))
        lateral_velocity = right_normal.dot(self.body.linearVelocity) * right_normal
        
        # Enable drift when braking or when drift button is pressed
        is_braking = self.is_braking()
        drift_enabled = action['drift'] or is_braking
        drift_factor = 0.05 if drift_enabled else 1.0
        
        impulse = -self.body.mass * lateral_velocity * drift_factor
        
        if impulse.length > self.max_lateral_impulse:
            impulse *= self.max_lateral_impulse / impulse.length
        
        self.body.ApplyLinearImpulse(impulse, self.body.worldCenter, True)
        self.body.angularVelocity *= 0.7
        
        # Apply driving force
        forward_normal = self.body.GetWorldVector(localVector=(1, 0))
        forward_velocity = forward_normal.dot(self.body.linearVelocity) * forward_normal
        forward_speed = forward_velocity.length
        
        # Driving and braking
        if action['throttle'] < 0 and forward_speed > 1.0:  # Braking
            forward_drag = -self.base_drag * forward_velocity * self.brake_drag_multiplier
        else:
            force = self.max_drive_force * action['throttle']
            self.body.ApplyForce(force * forward_normal, self.body.worldCenter, True)
            # Gentler speed-dependent drag
            forward_drag = -self.base_drag * forward_velocity * (1.0 + 0.05 * forward_speed)
            
        self.body.ApplyForce(forward_drag, self.body.worldCenter, True)
        
        # Steering
        if forward_speed > 0.5:  # Only steer if moving
            # Speed-dependent steering with drift modification
            if action['drift']:
                # During drift: tighter turns and less speed limitation
                speed_factor = min(1.0, 12.0 / forward_speed) if forward_speed > 12.0 else 1.0
                steer_amount = 0.035 * action['steer'] * speed_factor
            else:
                # Normal steering
                speed_factor = min(1.0, 8.0 / forward_speed) if forward_speed > 8.0 else 1.0
                steer_amount = 0.025 * action['steer'] * speed_factor
            
            self.body.angle -= steer_amount

    def is_drifting(self):
        right_normal = self.body.GetWorldVector(localVector=(0, 1))
        lateral_speed = abs(right_normal.dot(self.body.linearVelocity))
        forward_speed = self.get_forward_velocity().length
        # Consider both manual drift and brake-induced drift
        return (lateral_speed > 3 and forward_speed > 3) or self.is_braking()

    def get_forward_velocity(self):
        forward_normal = self.body.GetWorldVector(localVector=(1, 0))
        return forward_normal.dot(self.body.linearVelocity) * forward_normal

    def is_braking(self):
        forward_normal = self.body.GetWorldVector(localVector=(1, 0))
        forward_velocity = forward_normal.dot(self.body.linearVelocity)
        return forward_velocity > 3.0 and self.body.linearVelocity.dot(forward_normal) > 0 and self.current_action['throttle'] < -0.5
