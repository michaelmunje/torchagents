from abc import ABC
from abc import abstractmethod
import torch

class Env(ABC):
    def __init__(self, env_name: str, render: bool = False, device: torch.device = None):
        self.n_steps = 0
        self.n_dones = 0
        self._render = render
        self.env_name = env_name
        self._device = device
        self._state = None
        self._reward = None
        self._done = False

    @abstractmethod
    def step(self, action):
        pass

    @abstractmethod
    def _clean_actions(self, action):
        pass

    @abstractmethod
    def is_discrete_action(self):
        pass

    @abstractmethod
    def get_last_action(self):
        pass

    @abstractmethod
    def get_state(self):
        pass

    @abstractmethod
    def get_reward(self):
        pass

    @abstractmethod
    def is_done(self):
        pass

    @abstractmethod
    def get_action_space(self):
        pass

    @abstractmethod
    def get_state_space(self):
        pass

    @abstractmethod
    def reset(self):
        pass

    @abstractmethod
    def close(self):
        pass
