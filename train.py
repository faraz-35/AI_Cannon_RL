import os
from stable_baselines3 import PPO
from stable_baselines3.common.env_checker import check_env
from env.cannon_env import CannonEnv

# --- 1. Create the environment ---
# This will be the world your agent learns in.
env = CannonEnv()

# It's good practice to check your custom environment to ensure it follows the Gym API.
# This will print a warning if something is wrong.
check_env(env)

# --- 2. Define the Model ---
# We will use the Proximal Policy Optimization (PPO) algorithm.
# 'MlpPolicy' means the agent will use a simple Multi-Layer Perceptron (a basic neural network)
# to decide its actions.
# verbose=1 will print out the training progress.
model = PPO('MlpPolicy', env, verbose=1)

# --- 3. Train the Model ---
# The agent will learn by interacting with the environment for a total of 30,000 steps.
# This number is tunable - more steps can lead to better performance but take longer.
print("Starting model training...")
model.learn(total_timesteps=30000)
print("Model training finished.")

# --- 4. Save the Trained Model ---
# Create a directory to save the models if it doesn't exist
models_dir = "models"
os.makedirs(models_dir, exist_ok=True)

# Save the model with a specific name
model_path = os.path.join(models_dir, "ppo_cannon_model.zip")
model.save(model_path)

print(f"Model saved to {model_path}")

# --- 5. Clean up ---
# Don't forget to close the environment
env.close()
