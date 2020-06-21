from torchagents import Agent
from torchagents import NetworkPolicy
from torchagents.utilities import ReplayBuffer
import torch
from typing import Tuple
import numpy as np


class Episode:
    def __init__(self, buffer_size, state_shape):
        device = torch.device("cpu")
        if torch.cuda.is_available():
            device = torch.device("cuda:0")
        self.reward = torch.tensor([0], dtype=torch.float32, device=device)
        self.experiences = ReplayBuffer(buffer_size=buffer_size, state_shape=state_shape)


class CrossEntropyAgent(Agent):
    def __init__(self, state_shape: Tuple,
                 num_actions: int,
                 max_episodes: int,
                 percentile_threshold: float = 0.7):
        policy = NetworkPolicy(state_shape=state_shape,
                               num_actions=num_actions)
        super(CrossEntropyAgent, self).__init__(policy=policy,
                                                state_shape=state_shape,
                                                num_actions=num_actions)
        self._percentile_threshold = percentile_threshold
        self.off_policy = True
        self._max_episodes = max_episodes

        self._episodes = [Episode(buffer_size=100, state_shape=self._state_shape)
                          for _ in range(self._max_episodes)]
        self._current_episode_index = 0

    def filter_episodes(self) -> None:
        rewards = [e.reward for e in self._episodes[:self._current_episode_index]]
        reward_bound = np.percentile(rewards, self._percentile_threshold)
        self._episodes = [e for e in self._episodes[:self._current_episode_index]
                          if e.reward >= reward_bound]

    def update_current_episode(self, state: torch.Tensor,
                               action: torch.Tensor,
                               reward: torch.Tensor,
                               next_state: torch.Tensor) -> None:
        self._episodes[self._current_episode_index].experiences.\
            update(state, action, reward, next_state)
        self._episodes[self._current_episode_index].reward += reward

    def finished_episode(self) -> None:
        self._current_episode_index += 1

    def train(self):
        self.filter_episodes()
        states = torch.cat([e.experiences.get_states()
                            for e in self._episodes[:self._current_episode_index]])
        actions = torch.cat([e.experiences.get_actions()
                             for e in self._episodes[:self._current_episode_index]]).long()
        rewards = torch.cat([e.experiences.get_rewards()
                             for e in self._episodes[:self._current_episode_index]])
        self._policy.train(states, actions, rewards)

    def reset_episodes(self):
        self._episodes = [Episode(buffer_size=100, state_shape=self._state_shape)
                          for _ in range(self._max_episodes)]
        self._current_episode_index = 0

    def get_avg_reward(self):
        return np.mean([torch.sum(e.experiences.get_rewards()).cpu().numpy()
                             for e in self._episodes[:self._current_episode_index]])

    def get_action(self, state: torch.Tensor) -> torch.Tensor:
        action_probs = self.get_action_distribution(state)
        if self.off_policy:
            return np.random.choice(len(action_probs), p=action_probs)
        return action_probs
