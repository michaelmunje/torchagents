from unittest import TestCase
from torchagents.utilities import ReplayBuffer
import torch


class TestReplayBuffer(TestCase):
    def test_update(self):
        # buffer = ReplayBuffer(buffer_size=20, state_shape=(1, 1))
        # experience: state, action, reward, next state
        buffer = ReplayBuffer(buffer_size=20, state_shape=(1, 2))
        state1 = torch.tensor([[5.0, 10.0]], dtype=torch.float32)
        action1 = torch.tensor([3])
        reward1 = torch.tensor([5.0])
        next_state1 = torch.tensor([[2.0, 3.0]])
        buffer.update(state=state1, action=action1, reward=reward1, next_state=next_state1)
        i = 0
        for (s, a, r, ns) in buffer.get_values():
            self.assertTrue(torch.eq(s, state1).all().numpy())
            self.assertFalse(torch.eq(s, action1).all().numpy())
            self.assertFalse(torch.eq(s, next_state1).all().numpy())
            self.assertTrue(torch.eq(a, action1).all().numpy())
            self.assertTrue(torch.eq(r, reward1).all().numpy())
            self.assertTrue(torch.eq(ns, next_state1).all().numpy())

        buffer.reset()
