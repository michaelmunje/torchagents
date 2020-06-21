from unittest import TestCase
from torchagents.utilities import ReplayBuffer
import torch


class TestReplayBuffer(TestCase):
    def test_update(self):
        # experience: state, action, reward, next state
        buffer = ReplayBuffer(buffer_size=2, state_shape=(1, 2))

        state1 = torch.tensor([[5.0, 10.0]], dtype=torch.float32)
        action1 = torch.tensor([3])
        reward1 = torch.tensor([5.0])
        next_state1 = torch.tensor([[2.0, 3.0]])
        buffer.update(state=state1, action=action1,
                      reward=reward1, next_state=next_state1)

        state2 = torch.tensor([[10.0, 15.0]], dtype=torch.float32)
        action2 = torch.tensor([2])
        reward2 = torch.tensor([3.0])
        next_state2 = torch.tensor([[55.0, 22.0]])
        buffer.update(state=state2, action=action2,
                      reward=reward2, next_state=next_state2)

        (s, a, r, ns) = buffer.get_random_batch(2)

        self.assertTrue(torch.eq(s[0], state1).all().numpy()
                        or torch.eq(s[0], state2).all().numpy())

        self.assertFalse(torch.eq(s[0], action1).all().numpy())
        self.assertFalse(torch.eq(s[0], next_state1).all().numpy())

        self.assertTrue(torch.eq(a[0], action1).all().numpy()
                        or torch.eq(a[0], action2).all().numpy())

        self.assertTrue(torch.eq(r[0], reward1).all().numpy()
                        or torch.eq(r[0], reward2).all().numpy())

        self.assertTrue(torch.eq(ns[0], next_state1).all().numpy()
                        or torch.eq(ns[0], next_state2).all().numpy())

        buffer.reset()
