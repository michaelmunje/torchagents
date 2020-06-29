from unittest import TestCase
import torch
import numpy as np
from torchagents.utilities import NNBuilder

class TestNNBuilder(TestCase):

    def tearDown(self):
        pass

    def setUp(self):
        self.config = {'input_size': 50, 'output_size': 15, 'hidden_layers': [128, 128, 256], 'output_function': torch.nn.Softmax}
        self.nn = NNBuilder.LinearSequentialBuilder(**self.config)

    def test_linear_sequential_builder_output(self):
        t = torch.rand(self.config['input_size'])
        self.assertTrue(self.nn(t).shape, torch.Size([self.config['output_size']]))