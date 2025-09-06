import gymnasium as gym
import numpy as np
import pygame

# Define some colors for Pygame
BLACK = (0, 0, 0)
WHITE = (255, 255, 255)
GREEN = (0, 255, 0)
RED = (255, 0, 0)

class CannonEnv(gym.Env):
    """
    A custom environment for a cannon shooting a projectile at a target.

    Observation Space: The position (x-coordinate) of the target.
    Action Space: The angle of the cannon (0 to 90 degrees).
    Reward: Based on the negative distance to the target. Closer is better.
    """
    metadata = {"render_modes": ["human"], "render_fps": 30}

    def __init__(self, render_mode=None):
        super(CannonEnv, self).__init__()

        # --- Define Action and Observation Spaces ---
        # Action: A single continuous value for the angle (0 to 90 degrees)
        self.action_space = gym.spaces.Box(low=0, high=90, shape=(1,), dtype=np.float32)

        # Observation: A single continuous value for the target's x-position
        # The target can be anywhere from 100 to 700 pixels horizontally.
        self.observation_space = gym.spaces.Box(low=100, high=700, shape=(1,), dtype=np.float32)

        # --- Environment Parameters ---
        self.gravity = 9.8
        self.force = 100  # Constant launch force
        self.cannon_pos = (50, 550) # (x, y) position on the screen
        self.screen_width = 800
        self.screen_height = 600

        # This will hold the current target's position
        self.target_pos = 0

        # Fixed target postion for easy practice lol
        self.fixed_target_pos = 400

        # --- Pygame Setup for Rendering ---
        self.render_mode = render_mode
        self.screen = None
        self.clock = None

    def reset(self, seed=None, options=None):
        """
        Called at the beginning of each episode. Resets the environment.
        """
        super().reset(seed=seed)

        # Place the target at a new random location
        self.target_pos = self.observation_space.sample()[0]
        # In the begining placing in a single position everytime
        #self.target_pos = self.fixed_target_pos

        # Return the initial observation (state) and an empty info dict
        observation = np.array([self.target_pos], dtype=np.float32)
        info = {}

        if self.render_mode == "human":
            self._render_frame()

        return observation, info

    def step(self, action):
        """
        Takes an action and returns the outcome.
        """
        angle_degrees = action[0]
        angle_radians = np.deg2rad(angle_degrees)

        # --- Physics Simulation ---
        # Calculate initial velocities
        vx = self.force * np.cos(angle_radians)
        vy = -self.force * np.sin(angle_radians) # Negative because Pygame's y-axis is inverted

        # Simulate projectile motion until it hits the ground (y >= ground_level)
        t = (2 * vy) / -self.gravity # time of flight formula simplified
        landing_x = self.cannon_pos[0] + vx * t

        # --- Calculate Reward ---
        # The reward is the negative distance from the landing spot to the target.
        # We want to maximize reward, which means minimizing distance.
        distance = abs(landing_x - self.target_pos)

        # A simple reward function. The max reward is 0 (perfect hit).
        # Normalize reward
        # Explicitly cast the reward to a standard Python float
        reward = float(-distance / self.screen_width)

        # --- Define Episode End ---
        # In this environment, an episode is over after every single shot.
        terminated = True
        truncated = False # Not using a time limit

        # The new observation is the same target position (since we reset right after)
        observation = np.array([self.target_pos], dtype=np.float32)
        info = {}

        if self.render_mode == "human":
            self._render_frame(projectile_path=self._get_trajectory(vx, vy))

        return observation, reward, terminated, truncated, info

    def render(self):
        """
        Renders the current state of the environment using Pygame.
        """
        if self.render_mode == "human":
            self._render_frame()

    def _get_trajectory(self, vx, vy):
        """Helper to calculate points for drawing the projectile path."""
        path = []
        x, y = self.cannon_pos
        t = 0
        while y < self.screen_height:
            x = self.cannon_pos[0] + vx * t
            y = self.cannon_pos[1] + vy * t + 0.5 * self.gravity * t**2
            # Convert NumPy floats to integers for Pygame
            path.append((int(x), int(y)))
            t += 0.2
        return path

    def _render_frame(self, projectile_path=None):
        if self.screen is None:
            pygame.init()
            pygame.display.set_caption("Cannon Environment") # Added a caption
            self.screen = pygame.display.set_mode((self.screen_width, self.screen_height))
            self.clock = pygame.time.Clock()

        # --- Process Pygame Events ---
        # This is crucial to keep the window from becoming unresponsive
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self.close()

        # Fill the background
        self.screen.fill(BLACK)

        # Draw the ground
        pygame.draw.line(self.screen, WHITE, (0, self.cannon_pos[1] + 10), (self.screen_width, self.cannon_pos[1] + 10), 2)

        # Draw the cannon
        pygame.draw.circle(self.screen, WHITE, (int(self.cannon_pos[0]), int(self.cannon_pos[1])), 15)

        # Draw the target
        # Cast self.target_pos to an integer for Pygame
        pygame.draw.rect(self.screen, GREEN, (int(self.target_pos) - 10, self.cannon_pos[1] - 10, 20, 20))

        # Draw the projectile path if provided
        if projectile_path and len(projectile_path) > 1:
            # Convert points to integers for Pygame
            int_path = [(int(p[0]), int(p[1])) for p in projectile_path]
            pygame.draw.lines(self.screen, RED, False, int_path, 2)

        # Update the display
        pygame.display.flip()

        # --- Pause After Firing ---
        # If we drew a projectile, wait for a moment to show the result
        if projectile_path:
            pygame.time.wait(500) # Wait for 500 milliseconds (0.5 seconds)

        self.clock.tick(self.metadata["render_fps"])

    def close(self):
        if self.screen is not None:
            pygame.quit()
            self.screen = None
