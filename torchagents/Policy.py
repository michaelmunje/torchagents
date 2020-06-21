from abc import ABC
from abc import abstractmethod
from typing import Tuple
import torch


class Policy(ABC):
    def __init__(self, state_shape: Tuple,
                 num_actions: int):
        self._num_actions = num_actions
        self._state_shape = state_shape

    @abstractmethod
    def get_action(self, state: torch.Tensor) -> torch.Tensor:
        pass
