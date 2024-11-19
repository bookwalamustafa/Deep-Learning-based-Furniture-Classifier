import os
import random

import torch
import torch.nn as nn
import torchinfo


class Model(nn.Sequential):

    def __init__(self, input_shape, layers):
        super().__init__(*layers)
        self.input_shape = input_shape
        seed = 123
        random.seed(seed)
        torch.manual_seed(seed)
        torch.use_deterministic_algorithms(True)

    def print(self):
        print()
        torchinfo.summary(self, input_size=self.input_shape,
                          col_names=['input_size', 'output_size', 'num_params', 'kernel_size'])
        print()

    def save(self, path, warn_if_exists=True):
        if warn_if_exists and os.path.exists(path):
            response = input(f'File {path} exists; replace? (y/n) ')
            if response.lower().startswith('n'):
                return
            os.remove(path)
        else:
            os.makedirs(os.path.dirname(path), exist_ok=True)
        torch.save(self.state_dict(), path)

    def load(self, path):
        self.load_state_dict(torch.load(path))

    def freeze(self):
        for param in self.parameters():
            param.requires_grad = False
        return self

    def unfreeze(self):
        for param in self.parameters():
            param.requires_grad = True
        return self

    def start_training(self):
        super().train(True)

    def stop_training(self):
        super().train(False)
