from unittest import TestCase
from torchagents.utilities import ReplayBuffer
import torch


class TestReplayBuffer(TestCase):
    def test_update(self):

        device = torch.device("cpu")
        if torch.cuda.is_available():
            device = torch.device("cuda:0")

        # experience: state, action, reward, next state
        buffer = ReplayBuffer(buffer_size=2, state_shape=(1, 2))

        state1 = torch.tensor([[5.0, 10.0]], dtype=torch.float32, device=device)
        action1 = torch.tensor([3], device=device)
        reward1 = torch.tensor([5.0], device=device)
        next_state1 = torch.tensor([[2.0, 3.0]], device=device)
        buffer.update(state=state1, action=action1,
                      reward=reward1, next_state=next_state1)

        state2 = torch.tensor([[10.0, 15.0]], dtype=torch.float32, device=device)
        action2 = torch.tensor([2], device=device)
        reward2 = torch.tensor([3.0], device=device)
        next_state2 = torch.tensor([[55.0, 22.0]], device=device)
        buffer.update(state=state2, action=action2,
                      reward=reward2, next_state=next_state2)

        (s, a, r, ns) = buffer.get_random_batch(2)

        self.assertTrue(torch.eq(s[0], state1).all().cpu().numpy()
                        or torch.eq(s[0], state2).all().cpu().numpy())

        self.assertFalse(torch.eq(s[0], action1).all().cpu().numpy())
        self.assertFalse(torch.eq(s[0], next_state1).all().cpu().numpy())

        self.assertTrue(torch.eq(a[0], action1).all().cpu().numpy()
                        or torch.eq(a[0], action2).all().cpu().numpy())

        self.assertTrue(torch.eq(r[0], reward1).all().cpu().numpy()
                        or torch.eq(r[0], reward2).all().cpu().numpy())

        self.assertTrue(torch.eq(ns[0], next_state1).all().cpu().numpy()
                        or torch.eq(ns[0], next_state2).all().cpu().numpy())

        buffer.reset()
