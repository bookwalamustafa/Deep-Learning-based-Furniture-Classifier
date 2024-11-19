import torch.nn as nn
import torch.optim as optim
from data import Data


class _Autoencoder:

    BATCH_SIZE = 16

    def __init__(self, path):
        self.path = path

        # Override this method in subclasses to create encoder, decoder, and model
        self.encoder = None
        self.decoder = None
        self.model = None

    def print(self):
        self.model.print()

    def train(self, epochs, data):
        self.model.start_training()

        criterion = nn.MSELoss()
        optimizer = optim.Adam(self.model.parameters(),
                               lr=.001, betas=(0.5, 0.999))

        batches = data.batches(self.BATCH_SIZE)
        for epoch in range(epochs):

            train_loss = 0.0
            for batch in batches:
                optimizer.zero_grad()
                inputs = batch.images
                outputs = self.model(inputs)
                loss = criterion(outputs, inputs)
                loss.backward()
                optimizer.step()
                train_loss += loss.item()

            print(f'[Epoch {epoch + 1}/{epochs}]   loss: {train_loss:.5f}')

        self.model.stop_training()

    def freeze(self):
        self.model.freeze()
        return self

    def unfreeze(self):
        self.model.unfreeze()
        return self

    def save(self):
        self.model.save(self.path)
        return self

    def load(self):
        self.model.load(self.path)
        return self

    def encode(self, data):
        codes = self.encoder(data.images)
        return Data(codes, data.labels, data.classes)

    def decode(self, data):
        images = self.decoder(data.images)
        return Data(images, data.labels, data.classes)

    def generate(self, data):
        inputs = data.images
        outputs = self.model(inputs)
        return Data(outputs, data.labels, data.classes)
