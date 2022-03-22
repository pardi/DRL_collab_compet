import torch
from collections import deque
from ddpg_agent import DDPQAgent
from RobotEnv import TennisEnv

import numpy as np
import matplotlib.pyplot as plt

import sys


def mov_avg(data, window):
    v = deque(maxlen=window)

    ma_data = []

    for d in data:
        v.append(d)
        ma_data.append(np.average(v))

    return ma_data


def main(file_env_path, train=True, n_episodes=1000, best_weight_path="best_weights/"):

    # Define the device to run the code into: GPU when available, CPU otherwise
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # Load environment
    env = TennisEnv(file_env_path, train=train)

    # Network parameters
    gamma = 0.99

    # Set number of episodes
    if not train:
        n_episodes = 1

    # Set timeout
    max_t = 5000

    # Final score
    final_score = 0.5
    best_score = 0
    solved_flag = False

    # Create agent
    agents = [DDPQAgent(gamma, env.state_size, env.action_size, device, name="agent1"),
              DDPQAgent(gamma, env.state_size, env.action_size, device, name="agent2")]

    if not train:
        for agent in agents:
            agent.load(best_weight_path)

    # list containing scores from each episode
    scores_list = []

    if train:
        score_window_size = 10
    else:
        score_window_size = 1

    for episode in range(1, n_episodes + 1):

        # Reset Noise
        for agent in agents:
            agent.reset()

        # Reset environment
        states = env.reset()
        scores = np.zeros(2)

        for t in range(max_t):

            actions = []

            # Get action
            for agent, state in zip(agents, states):
                actions.append(agent.get_action(state[np.newaxis, :])[0])

            actions = np.array(actions)

            # Get (s', r)
            next_states, rewards, dones, _ = env.step(actions)

            if train:
                for agent, state, action, reward, next_state, done in zip(agents, states, actions, rewards, next_states, dones):
                    # Update Actor-Critic with (s, a, r, s')
                    agent.step(state[np.newaxis, :], action[np.newaxis, :], reward, next_state[np.newaxis, :], done, t)

            states = next_states

            scores += np.array(rewards)

            # Break if episode is finished
            if np.any(dones):
                break

        # save most recent score
        scores_list.append(np.max(scores))

        if episode % score_window_size == 0:
            print('\rEpisode {}\tMax average Score : {:.2f}'.format(episode, np.mean(np.array(scores_list[-score_window_size:]))))

        # Check if we hit the final score
        if train:
            avg_score = np.mean(np.array(scores_list[-score_window_size:]))

            if avg_score >= final_score and not solved_flag:
                print('\nEnvironment solved in {:d} episodes!'.format(episode))
                solved_flag = True
                best_score = avg_score

            if solved_flag and avg_score > best_score:
                print('Saved better solution! Average Score: {:.2f}'.format(avg_score))
                for agent in agents:
                    agent.save(best_weight_path)

                best_score = avg_score

    if train:

        with open("scores_list.csv", 'w') as f:
            for s in scores_list:
                f.write(str(s) + ',')

        # Average all scores
        window_avg = score_window_size
        ma_data = mov_avg(scores_list, window_avg)

        plt.plot(scores_list, alpha=0.5)
        plt.plot(ma_data, alpha=1)
        plt.ylabel('Rewards')
        plt.savefig("img/p3_collab_compet.jpg")

        plt.show()


if __name__ == "__main__":
    # Set training:
    #   True - for training
    #   False - for executing best weight (when present)
    # if len(sys.argv) < 2:
    #     print("Input not recognised!")
    #     print("Usage: --no_training : run a single episode\n"
    #           "       --training : train the algorithm for 2k episodes")
    # else:
    #     if sys.argv[1] == "--no_training":
    #         main(file_env_path="Tennis_Linux/Tennis.x86_64", train=False)
    #     elif sys.argv[1] == "--training":
    #         main(file_env_path="Tennis_Linux/Tennis.x86_64", train=True)
    main(file_env_path="Tennis_Linux/Tennis.x86_64", n_episodes=1000, train=True)





