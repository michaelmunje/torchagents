import torch
from typing import Tuple


class ReplayBuffer:
    def __init__(self, buffer_size: int, state_shape: Tuple):
        # Experience consists of:
        # state, action, reward, nextState
        self.states = torch.zeros(size=[buffer_size, *state_shape], dtype=torch.float32)
        self.actions = torch.zeros(size=[buffer_size], dtype=torch.int)
        self.rewards = torch.zeros(size=[buffer_size], dtype=torch.float32)
        self.next_states = torch.zeros(size=[buffer_size, *state_shape], dtype=torch.float32)
        self.current_index = 0
        self.buffer_size = buffer_size

    def update(self, state: torch.Tensor, action: torch.Tensor,
               reward: torch.Tensor, next_state: torch.Tensor) -> None:
        if self.current_index >= self.buffer_size:
            raise ValueError('Cannot update once replay buffer is full.')
        self.states[self.current_index] = state
        self.actions[self.current_index] = action
        self.rewards[self.current_index] = reward
        self.next_states[self.current_index] = next_state
        self.current_index += 1

    def get_values(self):
        i = 0
        while i < self.current_index:
            yield self.states[i], self.actions[i], self.rewards[i], self.next_states[i]
            i += 1

    def reset(self) -> None:
        self.current_index = 0
