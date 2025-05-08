import gymnasium as gym
import numpy as np
import matplotlib.pyplot as plot
import pickle as file_manager

def run(episodes=15000, is_training=True, render=False, is_slippery=False):
    """
    Run the FrozenLake environment with Q-learning.

    Parameters:
    - episodes: Number of episodes to run.
    - is_training: If True, train the agent; if False, load the Q-table from file.
    - render: If True, render the environment in a graphical display.
    - is_slippery: If True, use a slippery version of the environment, which only follows the intended action/direction 1/3 of the time,
        and has a 2/3's chance of "slipping" in another direction than the one intended.
    """

# Create the FrozenLake environment
    env = gym.make("FrozenLake-v1", map_name="8x8", render_mode="human" if render==True else None, is_slippery=is_slippery)
    
    q_array = None
    if(is_training == False):

        # Load a previously trained Q-table from file if not training
        with open("./agents/q_array.pkl", "rb") as file:
            q_array_from_file = file_manager.load(file)
        env.close()
        q_array = q_array_from_file
    else:

        # Initialize a blank Q-table with zeros if training
        q_array = np.zeros((env.observation_space.n, env.action_space.n))

    learning_rate = 0.9
    discount_factor = 0.9
    epsilon = 1  # 1 = 100% random actions
    epsilon_decay_rate = 0.00001  # Decay rate per episode at which actions become less random and start following the policy
    random_number_generator = np.random.default_rng(42)  # Ideal seed for training/reproducibility. May need to set episodes closer to 100,000 and decrease episolon_decay_rate for random environments.

    rewards_per_episode = np.zeros(episodes)

    for episode_number in range(episodes):
        state = env.reset()[0]
        terminated = False
        truncated = False

        # Loop until the episode is terminated or truncated
        while not (terminated or truncated):

            if is_training and random_number_generator.random() < epsilon:
                action = env.action_space.sample() # Random action
            else:
                action = np.argmax(q_array[state])

            new_state, reward, terminated, truncated, info = env.step(action)

            if is_training:
                q_array[state, action] = q_array[state, action] + learning_rate * (reward + discount_factor * np.max(q_array[new_state]) - q_array[state, action])

            state = new_state

        # Decay epsilon after each episode
        epsilon = max(epsilon - epsilon_decay_rate, 0)
    
        if epsilon == 0:
            learning_rate = 0.0001  # Reduce learning rate when epsilon is 0

        if reward == 1:
            rewards_per_episode[episode_number] = 1

    env.close()

    sum_rewards = np.zeros(episodes)
    for t in range(episodes):
        sum_rewards[t] = np.sum(rewards_per_episode[max(0, t-100):t+1])
    # Plot the rewards
    plot.plot(sum_rewards)
    plot.xlabel('Episode')
    plot.ylabel('Sum of Rewards (last 100 episodes)')
    plot.title('Sum of Rewards Over Episodes')
    plot.savefig("rewards_plot.png")

    if is_training:
        # Save the Q-table to file after training
        file = open("./agents/q_array.pkl", "wb")
        file_manager.dump(q_array, file)
        file.close()

if __name__ == "__main__":
    # Uncomment the following lines and run one at a time to train or test the agent.

    # Example training config, approx. 50% success rate with is_slippery=True, 100% success rate with is_slippery=False:
    run(episodes=110000, is_training=True, render=False, is_slippery=False)

    # Example pre-trained config, is_slippery state must match training state:
    # run(episodes=1, is_training=False, render=True, is_slippery=False)
