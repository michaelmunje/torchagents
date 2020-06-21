import torch
from torch.utils.tensorboard import SummaryWriter
from typing import Tuple
from .Policy import Policy
from .utilities import ReplayBuffer


class Agent:
    def __init__(self, policy: Policy,
                 state_shape: Tuple,
                 num_actions: int,
                 off_policy: bool = False,
                 buffer_size: int = 100,
                 epsilon: float = 0.05):

        self._policy = policy
        self._state_shape = state_shape
        self._num_actions = num_actions
        self._epsilon = epsilon
        self._off_policy = off_policy

        if off_policy:
            self._buffer = ReplayBuffer(buffer_size, state_shape)

    def get_action(self, state: torch.Tensor):
        if self._off_policy:
            if torch.rand(1) < self._epsilon:
                return torch.randint(high=self._num_actions, size=[1])
        return self._policy.get_action(state)

    def update(self):
        pass
