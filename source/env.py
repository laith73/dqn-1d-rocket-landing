import numpy as np


class RocketLandingEnv:
    """
    Simple 1D vertical rocket landing environment.
    
    The rocket starts at a random height and velocity and the goal is to apply thrust
    to land it smoothly.

    State: [height, velocity]
    Action: torque in range [0, max_thrust]
    """

    def __init__(self, max_thrust=20, drag_constant=1,max_velocity=20, max_altitude=50,dt=0.1, mass=1.0,  g=10.0):
        """
        Initialize the pendulum environment.

        Args:
            max_height: maximum thrust that can be applied
            drag_constant: the drag constant
            max_velocity: the maximum rocket initial random velocity
            max_altitude: the maximum rocket initial random altitude
            dt: the timestep for integration
            m: the mass of the rocket
            g: the gravity acceleration (m/s^2)
        """
        
        self.max_thrust = max_thrust
        self.drag_constant = drag_constant
        self.max_velocity = max_velocity
        self.max_altitude = max_altitude
        self.dt = dt
        self.mass = mass
        self.g = g

        # State variables
        self.velocity = None  # rocket velocity (m/s)
        self.altitude = None  # rocket height (m)
        self.steps = 0
        self.max_steps = 200

    def reset(self, seed=None, altitude_range=None, velocity_range=None):
        """
        Reset the environment to initial state.

        Args:
            seed: random seed for reproducibility
            altitude_range: range for initial height (low, high)
            velocity_range: range for initial velocity (low, high)

        Returns:
            observation: [altitude, velocity]
        """

        if seed is not None:
            np.random.seed(seed)
        
        if altitude_range is None:
            altitude_range = [self.max_altitude/2, self.max_altitude]
        
        if velocity_range is None:
            velocity_range = [-self.max_velocity, -self.max_velocity/2]
            
        # Random initial angle and velocity
        self.altitude = np.random.uniform(low=altitude_range[0], high=altitude_range[1])
        self.velocity = np.random.uniform(low=velocity_range[0], high=velocity_range[1])
        self.steps = 0

        return self._get_obs()

    def _get_obs(self):
        """
        Get current observation.

        Returns:
            observation: [height, velocity]
        """
        return np.array([self.altitude, self.velocity], dtype=np.float32)

    def step(self, action):
        """
        Take a step in the environment.

        Args:
            action: trust to apply (will be clipped to [0, max_thrust])

        Returns:
            observation: next state [height, velocity]
            reward: reward for this step
            done: whether episode is finished
            info: additional information dictionary
        """
        # Clip action to valid range
        thrust = np.clip(action, 0, self.max_thrust)

        # Store current state for reward calculation
        altitude = self.altitude
        velocity = self.velocity

        # Physics simulation using Euler integration
        # Equation of motion: a = (thrust - mg - kv)/m
        acceleration = (thrust - self.mass*self.g - self.drag_constant*velocity) / self.mass

        # Update velocity
        self.velocity = velocity + acceleration * self.dt
        # Update altitude
        self.altitude = altitude + self.velocity * self.dt

        # Update step counter
        self.steps += 1
        done = self.steps >= self.max_steps
        
        # Calculate reward
        # Goal: minimize altitude, velocity, and thrust usage
        if self.altitude > 5:
            reward = -(0.002*self.altitude**2 + 0.001*self.velocity**2 + 0.04*thrust**2)
        elif self.altitude > 1:
            reward = -(0.002*self.altitude**2 + 0.002*self.velocity**2 + 0.005*thrust**2)
        elif self.altitude >=0:
            reward = 100*(1-self.altitude)**2 - (0.004*self.velocity**2 + 0.001*thrust**2)
        else:
            reward = -100
 
        # Additional info
        info = {
            'height': self.altitude,
            'velocity': self.velocity,
            'thrust': thrust
        }

        # if self.steps % 10  == 0:
        #     print(self.get_state_description())

        return self._get_obs(), reward, done, info

    def get_state_description(self):
        """
        Get human-readable description of current state.

        Returns:
            dict with state information
        """
        return {
            'altitude': self.altitude,
            'velocity': self.velocity,
            'steps': self.steps,
            'position_description': self._describe_position()
        }

    def _describe_position(self):
        """
        Describe rocket altitude in human-readable form.
        """
        alt = self.altitude

        if alt > 100:
            return "high altitude"
        elif 50 < alt <= 100:
            return "mid altitude"
        elif 10 < alt <= 50:
            return "low altitude"
        elif 0 <= alt <= 10:
            return "near landing zone"
        else: 
            return "crashed"

        
class Environment:
    """
    MDP Environment Template.

    Students should implement:
    - step(): state transition logic
    - reward(): reward function
    """

    def __init__(self):
        # State space bounds
        self.S_low_limits = np.array([0., 0.])
        self.S_upper_limits = np.array([1., 1.])

        # Action space bounds
        self.A_low_limits = np.array([0.])
        self.A_upper_limits = np.array([1.])

        # Current state
        self.s = None

        # Internal rocket
        self._rocket = RocketLandingEnv()

    def reset(self):
        """Reset environment to initial state. Returns initial state."""
        self._rocket.reset()
        self.s = self._normalize_state()
        return self.s.copy()

    def _normalize_state(self):
        """Map rocket state to [0,1]²."""
        s0 = self._rocket.altitude / self._rocket.max_altitude
        s1 = (self._rocket.velocity + self._rocket.max_velocity) / (2 * self._rocket.max_velocity)
        return np.array([np.clip(s0, 0, 1), np.clip(s1, 0, 1)])

    def _denormalize_action(self, action):
        """Map action from [0,1] to [0, max_thrust]."""
        a = action[0] if isinstance(action, np.ndarray) else action
        return a * self._rocket.max_thrust

    def step(self, action):
        """
        Execute action and transition to next state.

        Args:
            action: Action in [A_low_limits, A_upper_limits]

        Returns:
            next_state: The resulting state
            reward: The reward for this transition
            done: Whether the episode has ended
            info: Additional information (dict)

        *** STUDENTS IMPLEMENT THIS ***
        """
        

        # Convert normalized action to thrust
        thrust = self._denormalize_action(action)
        # print("Thrust applied " + str(thrust) + " Current state: " + str(self._rocket.get_state_description()))
        # Step pendulum
        obs, r, done, info = self._rocket.step(thrust)

        # Get normalized next state
        next_state = self._normalize_state()

        # Update state
        self.s = next_state

        return next_state.copy(), r, done, {}