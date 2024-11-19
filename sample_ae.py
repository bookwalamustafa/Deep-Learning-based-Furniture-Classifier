import sys

import torch.nn as nn
from autoencoder import _Autoencoder
from data import Data
from model import Model


class Sample_AE(_Autoencoder):

    def __init__(self, path):
        super().__init__(path)

        n_kernels = 64

        self.encoder = Model(
            input_shape=(self.BATCH_SIZE, 3, 64, 64),
            layers=[
                nn.Conv2d(3, n_kernels, kernel_size=3, padding=1),
                nn.ReLU(),
                nn.MaxPool2d(kernel_size=2),
                nn.Conv2d(n_kernels, n_kernels, kernel_size=3, padding=1),
                nn.ReLU(),
                nn.MaxPool2d(kernel_size=2),
                nn.Conv2d(n_kernels, n_kernels, kernel_size=3, padding=1),
                nn.ReLU(),
                nn.MaxPool2d(kernel_size=2),
                nn.Flatten(),
                nn.Dropout(p=0.2),
                nn.Linear(n_kernels * 8 * 8, 256),
                nn.ReLU(),
                nn.Dropout(p=0.2),
                nn.Linear(256, 16),
                nn.ReLU(),
            ]
        )

        self.decoder = Model(
            input_shape=(self.BATCH_SIZE, 16),
            layers=[
                nn.Linear(16, 256),
                nn.ReLU(),
                nn.Linear(256, n_kernels * 8 * 8),
                nn.ReLU(),
                nn.Unflatten(1, (n_kernels, 8, 8)),
                nn.ConvTranspose2d(n_kernels, n_kernels, kernel_size=3, stride=2,
                                   padding=1, output_padding=1),
                nn.ReLU(),
                nn.ConvTranspose2d(n_kernels, n_kernels, kernel_size=3, stride=2,
                                   padding=1, output_padding=1),
                nn.ReLU(),
                nn.ConvTranspose2d(n_kernels, 3, kernel_size=3, stride=2,
                                   padding=1, output_padding=1),
                nn.Sigmoid(),
            ]
        )

        self.model = Model(
            input_shape=self.encoder.input_shape,
            layers=[
                self.encoder,
                self.decoder
            ]
        )


if __name__ == '__main__':

    epochs = int(sys.argv[1]) if len(sys.argv) > 1 else None

    data = Data.load('data', image_size=64)
    data.shuffle()

    ae = Sample_AE('models/sample_ae.pt')
    ae.print()

    if not epochs:
        print(f'\nLoading {ae.path}...')
        ae.load()
    else:
        print(f'\nTraining...')
        ae.train(epochs, data)
        print(f'\nSaving {ae.path}...')
        ae.save()

    print(f'\nGenerating samples...')
    samples = ae.generate(data)
    data.display(32)
    samples.display(32)
