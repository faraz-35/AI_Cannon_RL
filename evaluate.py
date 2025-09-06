import os
import time
from stable_baselines3 import PPO
from env.cannon_env import CannonEnv

# --- 1. Define the model path ---
# This must match the name you used in train.py
model_path = os.path.join("models", "ppo_cannon_model.zip")

# --- 2. Create the environment with rendering ---
# IMPORTANT: We must pass render_mode="human" to see the simulation.
env = CannonEnv(render_mode="human")

# --- 3. Load the trained model ---
# The agent's "brain" is loaded from the file.
print("Loading the trained model...")
model = PPO.load(model_path, env=env)
print("Model loaded.")

# --- 4. Run the evaluation loop ---
# We'll watch the agent play for 10 episodes (10 shots).
episodes = 10
print(f"Running evaluation for {episodes} episodes...")

for ep in range(episodes):
    # Reset the environment to get a new target position
    obs, info = env.reset()

    # The episode is not done yet
    done = False

    # Since our episode ends in one step, this loop will only run once per episode.
    # It's written this way to be compatible with more complex environments.
    while not done:
        # Use the model to predict the best action for the current observation (target position)
        # deterministic=True makes the agent choose the single best action it knows.
        action, _states = model.predict(obs, deterministic=True)

        # Take the action in the environment
        obs, reward, terminated, truncated, info = env.step(action)

        # Check if the episode is finished
        done = terminated or truncated

        print(f"Episode {ep+1}: Target at {obs[0]:.2f}, Action (Angle): {action[0]:.2f}, Reward: {reward:.4f}")

    # Pause for a moment between shots to make it easier to watch
    time.sleep(1)

# --- 5. Clean up ---
env.close()
print("Evaluation finished.")
