import sys

import torch.nn as nn
from autoencoder import _Autoencoder
from data import Data
from model import Model


class AE3(_Autoencoder):

    def __init__(self, path):
        super().__init__(path)

        n_kernels = 64

        # Encoder: Same layers as AE2
        self.encoder = Model(
            input_shape=(self.BATCH_SIZE, n_kernels, 16, 16),
            layers=[
                nn.Conv2d(n_kernels, n_kernels, kernel_size=3, padding=1),
                nn.ReLU(),
                nn.MaxPool2d(kernel_size=2),
            ]
        )

        # Decoder: Same layers as AE2's decoder
        self.decoder = Model(
            input_shape=(self.BATCH_SIZE, n_kernels, 8, 8),
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

    # Load AE1 and use its encoder
    from ae1 import AE1
    ae1 = AE1('models/ae1.pt')
    ae1.load()

    # Encode data using AE1's encoder
    encoded_data_ae1 = ae1.encode(data)

    # Load AE2 and use its encoder
    from ae2 import AE2
    ae2 = AE2('models/ae2.pt')
    ae2.load()

    # Encode data using AE2's encoder
    encoded_data_ae2 = ae2.encode(encoded_data_ae1)

    # Initialize AE3 with the path to save the model
    ae3 = AE3('models/ae3.pt')
    ae3.print()

    if not epochs:
        print(f'\nLoading {ae3.path}...')
        ae3.load()
    else:
        print(f'\nTraining...')
        ae3.train(epochs, encoded_data_ae2)
        print(f'\nSaving {ae3.path}...')
        ae3.save()

    print(f'\nGenerating samples...')
    samples = ae3.generate(encoded_data_ae2)

    # Decode the samples back through AE2 and AE1 decoders
    decoded_samples_ae2 = ae2.decode(samples)
    decoded_samples_ae1 = ae1.decode(decoded_samples_ae2)

    # Display the original data and reconstructed data
    data.display(32)
    decoded_samples_ae1.display(32)
