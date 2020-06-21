import torch
from typing import Tuple


class ReplayBuffer:
    def __init__(self, buffer_size: int, state_shape: Tuple):

        device = torch.device("cpu")
        if torch.cuda.is_available():
            device = torch.device("cuda:0")

        # Experience consists of:
        # state, action, reward, nextState
        self._states = torch.zeros(size=[buffer_size, *state_shape],
                                   dtype=torch.float32, device=device)
        self._actions = torch.zeros(size=[buffer_size], dtype=torch.int, device=device)
        self._rewards = torch.zeros(size=[buffer_size], dtype=torch.float32, device=device)
        self._next_states = torch.zeros(size=[buffer_size, *state_shape],
                                        dtype=torch.float32, device=device)
        self._current_index = 0
        self.buffer_size = buffer_size

    def update(self, state: torch.Tensor, action: torch.Tensor,
               reward: torch.Tensor, next_state: torch.Tensor) -> None:
        idx = self._current_index % self.buffer_size
        self._states[idx] = state
        self._actions[idx] = action
        self._rewards[idx] = reward
        self._next_states[idx] = next_state
        self._current_index += 1

    def get_random_batch(self, batch_size: int) -> (torch.Tensor, torch.Tensor,
                                                    torch.Tensor, torch.Tensor):
        # rand_idx = torch.randint(self.buffer_size, (batch_size,)) without replacement
        if self._current_index < batch_size:
            raise ValueError('Batch size greater than number of replays.')
        rand_idx = torch.randperm(batch_size)
        return (self._states[rand_idx], self._actions[rand_idx],
                self._rewards[rand_idx], self._next_states[rand_idx])

    def reset(self):
        self._current_index = 0
