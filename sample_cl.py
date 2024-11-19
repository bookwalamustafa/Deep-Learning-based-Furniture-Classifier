import sys

import torch.nn as nn
from classifier import _Classifier
from data import Data
from model import Model


class Sample_Cl(_Classifier):

    def __init__(self, path):
        super().__init__(path)

        n_kernels = 64

        self.model = Model(

            input_shape=(self.BATCH_SIZE, 3, 64, 64),

            layers=[
                nn.Conv2d(3, n_kernels, kernel_size=3, padding=1),
                nn.ReLU(),
                nn.MaxPool2d(kernel_size=2),
                nn.Conv2d(n_kernels, n_kernels, kernel_size=3, padding=1),
                nn.ReLU(),
                nn.MaxPool2d(kernel_size=2),
                nn.Flatten(),
                nn.Dropout(p=0.2),
                nn.Linear(n_kernels * 16 * 16, 128),
                nn.ReLU(),
                nn.Dropout(p=0.2),
                nn.Linear(128, 4),
            ]
        )


if __name__ == '__main__':

    epochs = int(sys.argv[1]) if len(sys.argv) > 1 else None

    data = Data.load('data', image_size=64)
    data.shuffle()

    cl = Sample_Cl('models/sample_cl.pt')
    cl.print()

    if not epochs:
        print(f'\nLoading {cl.path}...')
        cl.load()
    else:
        train_data, test_data = data.split(.8)
        print(f'\nTraining...')
        cl.train(epochs, train_data, test_data)
        print(f'\nSaving {cl.path}...')
        cl.save()

    results = cl.classify(data)
    print(f'\nAccuracy: {results.accuracy(data):.1f}%')
    print(f'\nConfusion Matrix:\n\n{results.confusion_matrix(data)}')
    results.display(32, data)
