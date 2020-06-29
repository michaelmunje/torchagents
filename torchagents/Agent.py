import torch
from typing import Tuple
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
        Class containing abstraction for an agent, which essentially will give us actions for a given state.
        There also will be training involved.

        Parameters
        ----------
        policy (Policy): Policy to determine actions from a given state
        state_shape (Tuple): Shape of the environment state.
        num_actions (int): Number of possible actions
        off_policy (bool): Whether the algorithm is off-policy or returns output from current policy.
        buffer_size (int): Size of the replay buffer, only needed if replay buffer is used by the agent.
        epsilon (float): Epsilon from epsilon greedy off-policy algorithm.
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
        state (torch.Tensor): Current environment state to determine action from.

        Returns
        -------
        (torch.Tensor) Action for the agent.

        """
        if self.off_policy:
            if torch.rand(1) < self._epsilon:
                return torch.randint(high=self._num_actions, size=[1])
        return torch.argmax(self.get_action_distribution(state))

    def get_action_distribution(self, state: torch.Tensor) -> torch.Tensor:
        """
        Gets the Agent's probability distribution over actions for a given state.

        Parameters
        ----------
        state (torch.Tensor): Current environment state to determine action from.

        Returns
        -------
        (torch.Tensor) Action distribution for the agent.

        """
        return self._policy.get_action(state)

    def update(self):
        pass
