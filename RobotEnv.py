from unityagents import UnityEnvironment
import numpy as np


class TennisEnv(object):

    def __init__(self, file_path, train=True):

        # Load the environment
        self.unityEnv = UnityEnvironment(file_name=file_path)

        # get the default brain
        self.brain_name = self.unityEnv.brain_names[0]
        self.brain = self.unityEnv.brains[self.brain_name]

        # Get action sizes
        self.action_size = self.brain.vector_action_space_size

        env_info = self.unityEnv.reset(train_mode=train)[self.brain_name]
        self.state_size = env_info.vector_observations.shape[1]
        self.num_agents = len(env_info.agents)

    def step(self, action):

        # take the action to the environment
        env_info = self.unityEnv.step(action)[self.brain_name]

        # get the current state
        next_state = env_info.vector_observations

        # get the reward
        reward = env_info.rewards
        done = env_info.local_done

        return next_state, reward, done, env_info

    def reset(self, train_mode=True):
        # Reset the environment
        env_info = self.unityEnv.reset(train_mode=train_mode)[self.brain_name]

        # get the current state
        state = env_info.vector_observations

        return state

