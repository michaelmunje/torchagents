from abc import ABC
from abc import abstractmethod
from typing import Tuple
import torch.nn as nn
import torch.optim as optim
import torch


class Policy(ABC):
    def __init__(self, state_shape: Tuple,
                 num_actions: int):
        self._num_actions = num_actions
        self._state_shape = state_shape

    @abstractmethod
    def get_action(self, state: torch.Tensor) -> torch.Tensor:
        pass


class BasicNetwork(nn.Module):
    def __init__(self, state_shape: Tuple,
                 num_actions: int,
                 hidden_size: int = 128):
        super(BasicNetwork, self).__init__()
        num_input = list(state_shape)[0]
        self.net = nn.Sequential(
            #  May need to flatten state_shape
            nn.Linear(num_input, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, num_actions)
        )

    def forward(self, x):
        return self.net(x)


class NetworkPolicy(Policy):
    def __init__(self, state_shape: Tuple,
                 num_actions: int,
                 network=BasicNetwork,
                 optimizer=None,
                 writer=None):

        super(NetworkPolicy, self).__init__(state_shape, num_actions)

        self._network = network(state_shape, num_actions)
        if torch.cuda.is_available():
            self._network.cuda()
        self._softmax = nn.Softmax(dim=1)
        if optimizer is not None:
            self._optimizer = optimizer
        else:
            self._optimizer = optim.Adam(params=self._network.parameters(), lr=0.01)
        self._writer = writer
        self._training_iter = 0

    def get_action(self, state: torch.Tensor) -> torch.Tensor:
        #  Returns stochastic distribution over actions
        action_probs = self._softmax(self._network(state))
        return action_probs.data.cpu().numpy()[0]

    def train(self, states: torch.Tensor,
              actions: torch.Tensor,
              rewards: torch.Tensor):
        self._optimizer.zero_grad()
        actions_pred = self._network(states)
        loss_v = nn.CrossEntropyLoss((actions_pred, actions))
        loss_v.backward()
        self._optimizer.step()
        if self._writer is not None:
            self._writer.add_scalar("loss", loss_v.item(), self._training_iter)
        self._training_iter += 1
