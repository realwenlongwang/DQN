from dqn import DeepQNetwork
import gym
import os
import time


def main():
    print("This is a toy test of DQN algorithm")


if __name__ == "__main__":
    env = gym.make("Taxi-v2")
    dqn = DeepQNetwork(env.action_space.n)

    episodes = 50
    current_state = env.reset()

    for episode in range(episodes):
        os.system("clear")
        action = dqn.choose_action([[current_state]])
        next_state, reward, done, info = env.step(action)

        dqn.learn(current_state=current_state, next_state=next_state,
                  action=action, reward=reward, done=done)

        current_state = next_state

        env.render()

        time.sleep(1)
