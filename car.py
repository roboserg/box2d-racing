import Box2D
from Box2D.b2 import vec2, polygonShape
import math
import numpy as np
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
        # These parameters control the car's movement and handling characteristics
        self.max_drive_force = 20.0  # Maximum force applied for acceleration
        self.max_lateral_impulse = 6.0  # Maximum sideways force for grip
        self.brake_drag_multiplier = 5.0  # Multiplier for increased drag when braking
        self.base_drag = 0.2  # Base drag force applied to the car

        # Add slippage-related parameters
        self.grip_loss_threshold = 12.0  # Increased speed threshold before grip loss
        self.turn_grip_loss_factor = 0.4  # Increased grip loss factor for turns
        self.recovery_rate = 0.2  # Faster recovery
        self.current_grip = 1.0
        self.min_grip = 0.4  # Increased minimum grip
        self.drift_threshold = 4.0  # Minimum lateral velocity for drift effects

    def get_position(self):
        return self.body.position

    def get_angle(self):
        return self.body.angle

    def get_linear_velocity(self):
        """Returns signed velocity (positive = forward, negative = backward)"""
        forward_normal = self.body.GetWorldVector(localVector=(1, 0))
        return forward_normal.dot(self.body.linearVelocity)

    def get_angular_velocity(self):
        return self.body.angularVelocity

    def step(self, action):
        """
        Process car physics step with given control inputs.
        
        Args:
            action: dict with keys:
                - throttle: float [-1, 1] for backward/forward
                - steer: float [-1, 1] for right/left
                - drift: bool for drift mode
        """
        self.current_action = action
        forward_normal = self.body.GetWorldVector(localVector=(1, 0))
        forward_velocity = forward_normal.dot(self.body.linearVelocity) * forward_normal
        forward_speed = forward_velocity.length
        steering_intensity = abs(action['steer'])
        
        # Calculate grip loss based on speed and steering
        if forward_speed > self.grip_loss_threshold:
            speed_factor = (forward_speed - self.grip_loss_threshold) / 15.0
            turn_factor = steering_intensity * self.turn_grip_loss_factor
            grip_loss = speed_factor * turn_factor * 0.5
            self.current_grip = max(
                self.min_grip,
                self.current_grip - grip_loss
            )
        else:
            self.current_grip = min(
                1.0,
                self.current_grip + self.recovery_rate
            )
        
        # Apply lateral impulse with grip factor
        right_normal = self.body.GetWorldVector(localVector=(0, 1))
        lateral_velocity = right_normal.dot(self.body.linearVelocity) * right_normal
        
        drift_enabled = action['drift'] or self.is_physically_drifting()
        base_drift_factor = 0.05 if drift_enabled else 1.0
        final_drift_factor = base_drift_factor * self.current_grip
        
        impulse = -self.body.mass * lateral_velocity * final_drift_factor
        
        if impulse.length > self.max_lateral_impulse:
            impulse *= self.max_lateral_impulse / impulse.length
        
        self.body.ApplyLinearImpulse(impulse, self.body.worldCenter, True)
        
        # Adjust angular damping based on grip
        self.body.angularDamping = 0.7 * self.current_grip
        
        # Apply driving force and braking
        is_braking = action['throttle'] < 0 and forward_speed > 1.0
        if is_braking:
            forward_drag = -self.base_drag * forward_velocity * self.brake_drag_multiplier
        else:
            force = self.max_drive_force * action['throttle']
            self.body.ApplyForce(force * forward_normal, self.body.worldCenter, True)
            forward_drag = -self.base_drag * forward_velocity * (1.0 + 0.05 * forward_speed)
            
        self.body.ApplyForce(forward_drag, self.body.worldCenter, True)
        
        # Modified steering with reduced control during braking
        min_speed_for_steering = 1.0
        speed_factor = min(1.0, forward_speed / min_speed_for_steering)
        
        if forward_speed > 0.1:
            max_steering_speed = 10.0
            steering_speed_factor = min(forward_speed / max_steering_speed, 1.0)
            
            # Calculate braking influence on steering
            brake_steering_reduction = 0.3 if is_braking else 1.0
            turn_reduction = (1.0 / (1.0 + (forward_speed / 20.0))) * brake_steering_reduction
            
            if action['drift']:
                # During drift: maintain better steering control than braking
                base_steer = -6.0 * action['steer'] * steering_speed_factor
                desired_angular_velocity = base_steer * turn_reduction * 0.8
            else:
                # Normal steering: significantly reduced during braking
                grip_modified_steer = action['steer'] * (
                    0.8 + (1.0 - self.current_grip) * 0.3
                )
                base_steer = -4.0 * grip_modified_steer * steering_speed_factor
                desired_angular_velocity = base_steer * turn_reduction
            
            # Apply speed factor to limit low-speed turning
            desired_angular_velocity *= speed_factor
            
            # Limit maximum turning rate
            max_angular_velocity = 3.0
            desired_angular_velocity = max(
                min(desired_angular_velocity, max_angular_velocity),
                -max_angular_velocity
            )
            
            current_angular_velocity = self.body.angularVelocity
            torque = (desired_angular_velocity - current_angular_velocity) * self.body.mass
            self.body.ApplyTorque(torque * self.current_grip, True)

    def is_physically_drifting(self):
        # Get lateral velocity magnitude
        right_normal = self.body.GetWorldVector(localVector=(0, 1))
        lateral_velocity = right_normal.dot(self.body.linearVelocity) * right_normal
        return lateral_velocity.length > self.drift_threshold

    def is_drifting(self):
        """Returns true if the car is drifting (includes both intentional drift and slippage)"""
        right_normal = self.body.GetWorldVector(localVector=(0, 1))
        lateral_speed = abs(right_normal.dot(self.body.linearVelocity))
        forward_speed = self.get_forward_velocity().length
        return (lateral_speed > 3 and forward_speed > 3) or \
               self.is_physically_drifting() or \
               self.current_grip < 0.6  # Changed from 0.8 to make drift indication less sensitive

    def get_forward_velocity(self):
        forward_normal = self.body.GetWorldVector(localVector=(1, 0))
        return forward_normal.dot(self.body.linearVelocity) * forward_normal

    def is_braking(self):
        # Only consider braking if we have significant forward speed
        forward_speed = self.get_linear_velocity()
        return self.current_action['throttle'] < -0.1 and forward_speed > 2.0

    def get_heading(self):
        """Returns the car's heading angle in radians (0 = right, pi/2 = up)"""
        forward_vec = self.body.GetWorldVector(localVector=(1, 0))
        return math.atan2(forward_vec.y, forward_vec.x)
