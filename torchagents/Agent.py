import torch
from torch.utils.tensorboard import SummaryWriter
from typing import Tuple
import numpy as np
from torchagents import Policy
from torchagents.utilities import ReplayBuffer


class Agent:
    def __init__(self, policy: Policy,
                 state_shape: Tuple,
                 num_actions: int,
                 off_policy: bool = False,
                 buffer_size: int = 100,
                 epsilon: float = 0.05):
        """

        Parameters
        ----------
        policy
        state_shape
        num_actions
        off_policy
        buffer_size
        epsilon
        """
        self._policy = policy
        self._state_shape = state_shape
        self._num_actions = num_actions
        self._epsilon = epsilon
        self.off_policy = off_policy

        if off_policy:
            self._buffer = ReplayBuffer(buffer_size, state_shape)

    def get_action(self, state: torch.Tensor) -> torch.Tensor:
        """
        Gets an an action from an agent. If on-policy, we essentially just call the policy here. Otherwise, perform
        off-policy algorithm here.

        By defaylt, the action is greedy epsilon. We can override this method from
        agents inheriting this abstract class.

        Parameters
        ----------
        state: torch.Tensor
            Current environment state to determine action from.

        Returns
        -------
        torch.Tensor
            Action for the agent.

        """
        if self.off_policy:
            if torch.rand(1) < self._epsilon:
                return torch.randint(high=self._num_actions, size=[1])
        return torch.argmax(self.get_action_distribution(state))

    def get_action_distribution(self, state: torch.Tensor) -> torch.Tensor:
        return self._policy.get_action(state)

    def update(self):
        pass
