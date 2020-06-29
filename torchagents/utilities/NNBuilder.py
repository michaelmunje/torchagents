import torch
from typing import List


def LinearSequentialBuilder(input_size: int,
                            output_size: int,
                            hidden_layers: List[int],
                            output_function: torch.nn = None):

    activation_function = torch.nn.ReLU

    layer_sizes = [input_size] + hidden_layers + [output_size]
    net_layers = [torch.nn.Linear(layer_sizes[i//2], layer_sizes[i//2+1])
                  if i % 2 == 0 else activation_function()
                  for i in range(2*len(layer_sizes)-2)]

    if output_function is not None:
        net_layers += [output_function(dim=-1)]

    return torch.nn.Sequential(*net_layers)
