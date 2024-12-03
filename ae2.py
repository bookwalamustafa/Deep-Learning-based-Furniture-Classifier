import sys

import torch.nn as nn
from autoencoder import _Autoencoder
from data import Data
from model import Model


class AE2(_Autoencoder):

    def __init__(self, path):
        super().__init__(path)

        n_kernels = 64

        # Encoder: Single convolution + ReLU + max-pooling
        self.encoder = Model(
            input_shape=(self.BATCH_SIZE, n_kernels, 32, 32),
            layers=[
                nn.Conv2d(n_kernels, n_kernels, kernel_size=3, padding=1),
                nn.ReLU(),
                nn.MaxPool2d(kernel_size=2),
            ]
        )

        # Decoder: Single deconvolution + ReLU
        self.decoder = Model(
            input_shape=(self.BATCH_SIZE, n_kernels, 16, 16),
            layers=[
                nn.ConvTranspose2d(n_kernels, n_kernels, kernel_size=2, stride=2),
                nn.ReLU(),
            ]
        )

        # Combine encoder and decoder into the autoencoder model
        self.model = Model(
            input_shape=self.encoder.input_shape,
            layers=[
                self.encoder,
                self.decoder
            ]
        )


if __name__ == '__main__':

    epochs = int(sys.argv[1]) if len(sys.argv) > 1 else None

    # Load the original data
    data = Data.load('data', image_size=64)
    data.shuffle()

    # Load AE1 and use its encoder to transform the data
    from ae1 import AE1
    ae1 = AE1('models/ae1.pt')
    ae1.load()
    encoded_data = ae1.encode(data)

    # Initialize AE2 with the path to save the model
    ae2 = AE2('models/ae2.pt')
    ae2.print()

    if not epochs:
        print(f'\nLoading {ae2.path}...')
        ae2.load()
    else:
        print(f'\nTraining...')
        ae2.train(epochs, encoded_data)
        print(f'\nSaving {ae2.path}...')
        ae2.save()

    print(f'\nGenerating samples...')
    samples = ae2.generate(encoded_data)

    # Decode the samples back to image space for visualization
    decoded_samples = ae1.decode(samples)
    data.display(32)
    decoded_samples.display(32)
