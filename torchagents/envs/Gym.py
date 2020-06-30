import numpy as np
import gym
from torchagents.envs import Env
import torch
from typing import Tuple


class Gym(Env):
    def __init__(self, env_name: str, render: bool = False, device: torch.device = None):
        super(Gym, self).__init__(env_name, render)
        self.env = gym.make(env_name)
        if self.is_discrete_action():
            self._action_space = tuple([self.env.action_space.n])
        else:
            self._action_space = self.env.action_space.shape
        self._state_space = self.env.observation_space.shape
        self._device = device

    def step(self, action: np.array) -> (torch.Tensor, torch.Tensor, bool):
        self.n_steps += 1

        self._action = self._clean_actions(action)
        self._state, self._reward, self._done, _ = self.env.step(self._action)

        self._state = torch.tensor(self._state)
        self._reward = torch.tensor(self._reward)

        if self.device is not None:
            self._state.to(self._device)
            self._reward.to(self._device)

        if self._done:
            self.n_dones += 1

        if self._render:
            self.env.render()

        return self._state, self._reward, self._done

    def _clean_actions(self, action: np.array) -> np.array:
        if self.is_discrete_action():
            return np.argmax(action)
        else:
            return np.array(action)

    def is_discrete_action(self) -> bool:
        return self.env.action_space.shape == ()

    def get_last_action(self) -> np.array:
        return self._action

    def get_state(self) -> torch.Tensor:
        return self._state

    def get_reward(self) -> torch.Tensor:
        return self._reward

    def is_done(self) -> bool:
        return self._done

    def get_action_space(self) -> Tuple:
        return self._action_space

    def get_state_space(self) -> Tuple:
        return self._state_space

    def reset(self):
        self.env.reset()

    def close(self):
        self.env.close()
