import torch
from typing import Tuple


class ReplayBuffer:
    def __init__(self, buffer_size: int, state_shape: Tuple):

        device = torch.device("cpu")
        if torch.cuda.is_available():
            device = torch.device("cuda:0")

        # Experience consists of:
        # state, action, reward, nextState
        self.states = torch.zeros(size=[buffer_size, *state_shape],
                                  dtype=torch.float32, device=device)
        self.actions = torch.zeros(size=[buffer_size], dtype=torch.int, device=device)
        self.rewards = torch.zeros(size=[buffer_size], dtype=torch.float32, device=device)
        self.next_states = torch.zeros(size=[buffer_size, *state_shape],
                                       dtype=torch.float32, device=device)
        self.current_index = 0
        self.buffer_size = buffer_size

    def update(self, state: torch.Tensor, action: torch.Tensor,
               reward: torch.Tensor, next_state: torch.Tensor) -> None:
        self.states[self.current_index] = state
        self.actions[self.current_index] = action
        self.rewards[self.current_index] = reward
        self.next_states[self.current_index] = next_state
        self.current_index += 1
        self.current_index %= self.buffer_size

    def get_random_batch(self, batch_size: int) -> (torch.Tensor, torch.Tensor,
                                                    torch.Tensor, torch.Tensor):
        # rand_idx = torch.randint(self.buffer_size, (batch_size,)) without replacement
        rand_idx = torch.randperm(batch_size)
        return (self.states[rand_idx], self.actions[rand_idx],
                self.rewards[rand_idx], self.next_states[rand_idx])

    def reset(self) -> None:
        # This function has no real use right now
        self.current_index = 0
