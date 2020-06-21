from torchagents import Agent
from torchagents import NetworkPolicy
from torchagents.utilities import ReplayBuffer
import torch
import torch.nn as nn
from typing import Tuple
import numpy as np
import gym


class CrossEntropy(Agent):
    def __init__(self, state_shape: Tuple,
                 num_actions: int,
                 percentile_threshold: float = 0.7):
        policy = NetworkPolicy(state_shape=state_shape,
                               num_actions=num_actions)
        super(CrossEntropy, self).__init__(policy=policy,
                                           state_shape=state_shape,
                                           num_actions=num_actions)
        self._percentile_threshold = percentile_threshold
        self.current_episode = 0
        self._softmax = nn.Softmax(dim=1)
        self._off_policy = True

    def filter_episodes(self):
        # TODO: Filter list of episodes with mean reward percentile threshold
        return self._episodes

    def train_n_episodes(self, env: gym.Env, num_episodes: int):
        # Maybe env should never be used within an agent?
        self._episodes = [ReplayBuffer(buffer_size=100, state_shape=self._state_shape)
                         for _ in range(num_episodes)]
        for i in range(num_episodes):
            return

    def get_action(self, state: torch.Tensor):
        action_probs = self.get_action_distribution(state)
        if self._off_policy:
            return np.random.choice(len(action_probs), p=action_probs)
        return action_probs

    def SetOnPolicy(self):
        self._off_policy = False
