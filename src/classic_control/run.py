import gymnasium as gym
import numpy as np
import matplotlib.pyplot as plot
import pickle as file_manager
import math


def run(episodes=15000, is_training=True, render=False):

    # Create the CartPole environment
    env = gym.make("CartPole-v1", render_mode="human" if render else None)

    # We need to break down our continious observation space into discrete bins
    # The following array defines the number of bins for each observation:
    # [cart_position, cart_velocity, pole_angle, pole_velocity]
    number_of_bins = [12, 12, 48, 48]

    # Create a list of bin edges for each observation
    bins = [
        np.linspace(-4.8, 4.8, number_of_bins[0] - 1),   # Cart position
        np.linspace(-4, 4, number_of_bins[1] - 1),       # Cart velocity
        np.linspace(-0.418, 0.418, number_of_bins[2] - 1),  # Pole angle
        # Pole angular velocity
        np.linspace(-4, 4, number_of_bins[3] - 1)
    ]

    # Convert the continuous observation to discrete bins based on previous parameters

    def continuous_observation_to_discrete(observation, bins):
        return tuple(
            np.digitize(
                np.clip(observation[i], bins[i][0], bins[i][-1]), bins[i])
            for i in range(len(observation))
        )

    # q_array = None
    # if (is_training == True):

    #     # Initialize a blank Q-table with zeros if training
    #     q_array = np.zeros(number_of_bins + [env.action_space.n])

    # else:
    #     # Code for loading a previously trained Q-table from file if not training
    #     with open("./agents/q_array.pkl", "rb") as file:
    #         q_array_from_file = file_manager.load(file)
    #     q_array = q_array_from_file

    with open("./agents/q_array317.pkl", "rb") as file:
        q_array_from_file = file_manager.load(file)
    q_array = q_array_from_file

    # Initialize the Q-learning parameters
    learning_rate = 0.4
    discount_factor = 0.9
    epsilon = .9  # 1 = 100% random actions
    minimum_epsilon = 0.1  # Minimum epsilon value
    # Decay rate per episode at which actions become less random and start following the policy
    epsilon_decay_rate = 0.0001
    random_number_generator = np.random.default_rng()

    lowest_reward_to_save = 309
    rewards_per_episode = np.zeros(episodes)

    for episode_number in range(episodes):
        state = env.reset()[0]
        terminated = False
        truncated = False

        # Loop until the episode is terminated or truncated

        total_reward = 0
        while not (terminated or truncated):
            discrete_state = continuous_observation_to_discrete(state, bins)

            # Choose an action based on epsilon-greedy policy
            if is_training and random_number_generator.random() < epsilon:
                action = env.action_space.sample()
            else:
                action = np.argmax(q_array[discrete_state])

            new_state, reward, terminated, truncated, info = env.step(action)
            total_reward += reward

            if is_training:
                # Convert the new continuous observation to discrete bins
                discrete_new_state = continuous_observation_to_discrete(
                    new_state, bins)
                
                # Positive encouragement
                cart_position = state[0]
                center_bonus = max(0, 1 - abs(cart_position) / 2.4)
                adjusted_reward = reward + 0.1 * center_bonus


                # Update the Q-value using the Q-learning formula
                q_array[discrete_state][action] = q_array[discrete_state][action] + learning_rate * (
                    adjusted_reward + discount_factor * np.max(q_array[discrete_new_state]) - q_array[discrete_state][action])

            state = new_state

        # Decay epsilon after each episode
        epsilon = max(epsilon - epsilon_decay_rate, minimum_epsilon)

        rewards_per_episode[episode_number] = total_reward

        if epsilon == 0:
            learning_rate = 0.0001

        if episode_number % 100 == 0 and episode_number != 0:
            avg_last_100 = np.mean(
                rewards_per_episode[episode_number-100:episode_number])
            print(
                f"Episode {episode_number}: Avg Reward (last 100): {avg_last_100:.2f}, Epsilon: {epsilon:.4f}")
            
            if avg_last_100 > lowest_reward_to_save:

                print(f"Saving Q-table with avg reward {avg_last_100:.2f}")
                # Save the Q-table to a file
                with open(f"./agents/q_array{math.floor(avg_last_100)}.pkl", "wb") as file:
                    file_manager.dump(q_array, file)
                lowest_reward_to_save = avg_last_100

    env.close()

    if is_training:
        average_reward_per_100_episodes = np.zeros(episodes)
        for t in range(episodes):
            average_reward_per_100_episodes[t] = np.mean(
                rewards_per_episode[max(0, t-100):t+1])

        # Plot the rewards
        plot.plot(average_reward_per_100_episodes)
        plot.xlabel('Episode')
        plot.ylabel('Average of Rewards for last 100 Episodes')
        plot.title('Training Rewards')
        plot.savefig("rewards_plot.png")
        plot.show()

        # Save the Q-table to a file
        with open("./agents/q_array.pkl", "wb") as file:
            file_manager.dump(q_array, file)


if __name__ == "__main__":
    run(episodes=25000, is_training=True, render=False)
    # run(episodes=5, is_training=False, render=True)