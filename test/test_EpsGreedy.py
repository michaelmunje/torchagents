from unittest import TestCase
from torchagents import Policy
from torchagents import Agent
import numpy as np
import torch


class TestEpsGreedy(TestCase):

    def test_get_eps_greedy_action(self):
        class DumbPolicy(Policy):
            def get_action(self, state: torch.Tensor) -> torch.Tensor:
                return torch.tensor([0])  # always return first action
        torch.manual_seed(1337)
        policy = DumbPolicy(state_shape=(1,), num_actions=20)
        dummy = Agent(policy=policy, state_shape=(1,), num_actions=20,
                      off_policy=True, epsilon=0.05)

        num_test_actions = 100
        actions = np.zeros(num_test_actions)
        for i in range(num_test_actions):
            actions[i] = dummy.get_action(torch.tensor([0]))
        num_first = len(actions[actions == 0])
        p_first = num_first / num_test_actions
        self.assertTrue(0.93 < p_first < 0.97)
