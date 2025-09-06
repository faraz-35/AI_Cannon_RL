# test_env.py
from env.cannon_env import CannonEnv

# Create the environment with rendering enabled
env = CannonEnv(render_mode="human")

# Reset the environment to get the first observation
observation, info = env.reset()

# Run for 10 "episodes" (10 shots)
for i in range(10):
    # Take a random action (a random angle)
    action = env.action_space.sample()
    print(f"Episode {i+1}: Firing at angle {action[0]:.2f}")

    # Perform the action
    observation, reward, terminated, truncated, info = env.step(action)

    print(f"  -> Reward: {reward:.4f}")

    # If the episode is done, reset for the next one
    if terminated or truncated:
        print("  -> Episode finished. Resetting.\n")
        observation, info = env.reset()

# Close the environment
env.close()
