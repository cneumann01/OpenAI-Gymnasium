import gymnasium as gym

def main():
    env = gym.make("CartPole-v1", render_mode="human")
    observation, info = env.reset(seed=42)

    for _ in range(1000):
        action = env.action_space.sample()  # Random action
        observation, reward, terminated, truncated, info = env.step(action)

        if terminated or truncated:
            observation, info = env.reset()

    env.close()

if __name__ == "__main__":
    main()
