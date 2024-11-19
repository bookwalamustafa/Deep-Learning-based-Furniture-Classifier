import random

import matplotlib.pyplot as plt
import numpy as np
import torch
import torchvision.datasets as datasets
import torchvision.transforms as transforms


class Data:

    @classmethod
    def load(cls, path, image_size):
        dataset = datasets.ImageFolder(
            root=path,
            transform=transforms.Compose([
                transforms.Resize(image_size),
                transforms.CenterCrop(image_size),
                transforms.ToTensor(),
            ])
        )
        images, labels = [], []
        for i in range(len(dataset)):
            image, label = dataset[i]
            images.append(image)
            labels.append(label)
        classes = dataset.classes
        return cls(images, labels, classes)

    def _to_tensor(self, x):
        if isinstance(x, torch.Tensor):
            x = x.detach()
        elif isinstance(x, list):
            if isinstance(x[0], torch.Tensor):
                x = torch.stack(x)
            elif isinstance(x[0], int):
                x = torch.LongTensor(x)
        return x

    def __init__(self, images, labels, classes):
        self.images = self._to_tensor(images)
        self.labels = self._to_tensor(labels)
        self.classes = classes
        self.image_channels = (self.images.size(dim=1)
                               if self.images.dim() > 1 else None)
        self.image_size = (self.images.size(dim=2)
                           if self.images.dim() > 2 else None)

    def __len__(self):
        return len(self.images)

    def __iter__(self):
        return zip(self.images, self.labels)

    def __getitem__(self, key):
        if isinstance(key, slice):
            return Data(self.images[key], self.labels[key], self.classes)
        else:
            return (self.images[key], self.labels[key])

    def clone(self):
        return Data(self.images.clone(), self.labels.clone(), self.classes)

    def shuffle(self):
        pairs = list(zip(self.images, self.labels))
        random.shuffle(pairs)
        self.images = self._to_tensor([image for image, _ in pairs])
        self.labels = self._to_tensor([label for _, label in pairs])
        return self

    def split(self, ratio):
        i = int(ratio * len(self))
        return self[:i], self[i:]

    def class_name(self, label):
        return self.classes[label]

    def batches(self, size):
        return [self[i:i+size] for i in range(0, len(self), size)]

    def random_samples(self, n):
        return self.clone().shuffle()[:n]

    def accuracy(self, truth):
        correct = (self.labels == truth.labels).float().sum()
        return 100 * correct / len(self)

    def confusion_matrix(self, truth, w=10):
        m = np.zeros((len(self.classes), len(self.classes)), dtype=np.int8)
        for label, label2 in zip(self.labels, truth.labels):
            m[label2][label] += 1
        s = 'true\\pred' + ' ' * \
            (w-9) + ''.join([f'{x:>{w}}' for x in self.classes]) + '\n'
        for i in range(len(self.classes)):
            s += (f'{self.classes[i]:>{w}}'
                  + ''.join([f'{x:>{w}}' for x in m[i]])
                  + '\n')
        return s

    def display(self, n=None, truth=None):
        n = n or len(self)
        fig = plt.figure(figsize=(12, 8))
        fig.subplots_adjust(wspace=.5, hspace=.5)
        ncols = 8
        nrows = n // ncols
        for i in range(n):
            image = self.images[i]
            image = image[:3, :, :]
            image = np.transpose(image, (1, 2, 0))
            image = (image * 255).numpy().astype(np.uint8)
            label = self.labels[i]
            x = i % ncols
            y = (i // ncols)
            index = (ncols * y) + x + 1
            plt.subplot(nrows, ncols, index)
            plt.axis('off')
            if truth:
                plt.title(self.class_name(label), fontsize=10,
                          backgroundcolor=('green' if label ==
                                           truth.labels[i] else 'red'),
                          fontdict={'color': 'white'})
            else:
                plt.title(self.class_name(label), fontsize=10)
            plt.imshow(image)
        plt.show()
        return self
