from dqn import DeepQNetwork
import gym
import numpy as np
import os
import time
import matplotlib.pyplot as plt

if __name__ == "__main__":
    env = gym.make("Taxi-v2")
    # env = gym.make("FrozenLake-v0")
    episodes = 20
    dqn = DeepQNetwork(env.action_space.n, episodes=episodes, observation_space=env.observation_space.n)
    episode_reward_np_array = np.zeros(episodes)
    random_probability = 1

    for episode in range(episodes):
        done = False
        episode_reward = 0
        current_state = env.reset()
        step_counter = 0
        while not done:
            dqn.set_random_probability(random_probability)
            # os.system("clear")
            action = dqn.choose_action(current_state)
            next_state, reward, done, info = env.step(action)

            dqn.store_experience(current_state, action, reward, next_state, done)

            if step_counter > 100 and step_counter % 5 == 0:
                dqn.learn()

            current_state = next_state

            episode_reward += reward
            step_counter += 1

            # env.render()
            # time.sleep(0.1)
        random_probability -= 1 / episodes
        print(episode, step_counter)
        episode_reward_np_array[episode] = episode_reward

    dqn.plot_loss()
    print(episode_reward_np_array)
    plt.plot(episode_reward_np_array)
    plt.ylabel("Accumulated reward")
    plt.xlabel("Episode")
    plt.show()

    # dqn.print_loss()
    dqn.save_net()

    dqn.set_random_probability(0.1)
    for episode in range(3):
        done = False
        current_state = env.reset()
        while not done:
            os.system("clear")
            action = dqn.choose_action(current_state)

            next_state, reward, done, info = env.step(action)

            dqn.learn()

            current_state = next_state

            env.render()

            time.sleep(1)
