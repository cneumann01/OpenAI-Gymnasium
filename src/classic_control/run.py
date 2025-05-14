import gymnasium as gym

def run():
    env = gym.make("CartPole-v1", render_mode="human")
    observation, info = env.reset(seed=42)

    for _ in range(1000):
        # action = env.action_space.sample()  # Random action

        action = 1 if observation[2] > 0 else 0  # Simple policy: move right if pole is tilted right, else left

        observation, reward, terminated, truncated, info = env.step(action)

        if terminated or truncated:
            observation, info = env.reset()

    env.close()

if __name__ == "__main__":
    run()
