import torch
import torch.nn as nn
import torch.optim as optim
from data import Data


class _Classifier:

    BATCH_SIZE = 16

    def __init__(self, path):
        self.path = path

        # Override this method in subclasses to create a model
        self.model = None

    def print(self):
        self.model.print()

    def train(self, epochs, train_data, test_data):
        self.model.start_training()

        criterion = nn.CrossEntropyLoss()
        optimizer = optim.SGD(self.model.parameters(), lr=0.001, momentum=0.9)

        batches = train_data.batches(self.BATCH_SIZE)
        for epoch in range(epochs):

            train_loss = 0.0
            for batch in batches:
                optimizer.zero_grad()
                inputs = batch.images
                outputs = self.model(inputs)
                loss = criterion(outputs, batch.labels)
                loss.backward()
                optimizer.step()
                train_loss += loss.item()

            train_acc = self.classify(train_data).accuracy(train_data)
            test_acc = self.classify(test_data).accuracy(test_data)
            print(f'[Epoch {epoch + 1}/{epochs}]   loss: {train_loss:.5f}   '
                  + f'train_acc: {train_acc:.1f}%   test_acc: {test_acc:.1f}%')

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

    def classify(self, data):
        inputs = data.images
        outputs = self.model(inputs)
        labels = torch.argmax(outputs, 1)
        return Data(data.images, labels, data.classes)
