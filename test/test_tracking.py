import os
import tempfile
import time
import unittest

from tensorflow import keras

from segmind_track import log_params_decorator

import torch
from torch.utils.data import DataLoader, random_split
from torch.nn import functional as F
from torchvision.datasets import MNIST
from torchvision import transforms

import pytorch_lightning as pl

@log_params_decorator
def define_mnist_model(input_shape, hidden_neurons, num_classes=10):
    model = keras.Sequential([
        keras.layers.Flatten(input_shape=(28, 28)),
        keras.layers.Dense(hidden_neurons, activation='relu'),
        keras.layers.Dense(num_classes, activation='softmax')
    ])

    model.compile(
        optimizer='adam',
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy'])

    return model


class LightningMNISTClassifier(pl.core.LightningModule):

    def __init__(self):
        super(LightningMNISTClassifier, self).__init__()
        # mnist images are (1, 28, 28) (channels, width, height)
        # num_classes = 10
        self.layer_1 = torch.nn.Linear(28 * 28, 128)
        self.layer_2 = torch.nn.Linear(128, 256)
        self.layer_3 = torch.nn.Linear(256, 10)

    def forward(self, x):
        batch_size, channels, width, height = x.size()

        # (b, 1, 28, 28) -> (b, 1*28*28)
        x = x.view(batch_size, -1)

        # layer 1 (b, 1*28*28) -> (b, 128)
        x = self.layer_1(x)
        x = torch.relu(x)

        # layer 2 (b, 128) -> (b, 256)
        x = self.layer_2(x)
        x = torch.relu(x)

        # layer 3 (b, 256) -> (b, 10)
        x = self.layer_3(x)

        # probability distribution over labels
        x = torch.log_softmax(x, dim=1)

        return x

    def cross_entropy_loss(self, logits, labels):
        return F.nll_loss(logits, labels)

    def training_step(self, train_batch, batch_idx):
        x, y = train_batch
        logits = self.forward(x)
        loss = self.cross_entropy_loss(logits, y)

        logs = {'train_loss': loss}
        return {'loss': loss, 'log': logs}

    def validation_step(self, val_batch, batch_idx):
        x, y = val_batch
        logits = self.forward(x)
        loss = self.cross_entropy_loss(logits, y)
        return {'val_loss': loss}

    def validation_epoch_end(self, outputs):
        # called at the end of the validation epoch
        # outputs is an array with what you returned in validation_step
        # for each batch
        # outputs = [{'loss': batch_0_loss}, {'loss': batch_1_loss},
        # ..., {'loss': batch_n_loss}]
        avg_loss = torch.stack([x['val_loss'] for x in outputs]).mean()
        tensorboard_logs = {'val_loss': avg_loss}
        return {'avg_val_loss': avg_loss, 'log': tensorboard_logs}

    def prepare_data(self):
        # transforms for images
        transform=transforms.Compose([transforms.ToTensor(),
                                      transforms.Normalize((0.1307,),
                                      (0.3081,))])

        # prepare transforms standard to MNIST
        mnist_train = MNIST(os.getcwd(), train=True, download=True,
                            transform=transform)
        mnist_test = MNIST(os.getcwd(), train=False, download=True,
                           transform=transform)

        self.mnist_train, self.mnist_val = random_split(mnist_train,
                                                        [55000, 5000])

    def train_dataloader(self):
        return DataLoader(self.mnist_train, batch_size=64)

    def val_dataloader(self):
        return DataLoader(self.mnist_val, batch_size=64)

    def test_dataloader(self):
        return DataLoader(self.mnist_test, batch_size=64)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-3)
        return optimizer


class TestTracking(unittest.TestCase):
    """docstring for TestTracking."""

    def setUp(self):
        mocked_experiment_id = '521e7ebc-36c9-4ae8-8507-f7b31a5bd963_20'
        from segmind_track import set_experiment, set_runid
        set_experiment('a0583ec5-bdf3-4526-a985-05be15e62f16')

    def test_log_param(self):
        from segmind_track import log_param

    def test_log_metric(self):
        from segmind_track import log_metric

    def test_log_batch(self):
        from segmind_track import log_batch

        for i in range(1, 3):
            log_batch(
                metrics={'batch_test_metric': 1 + i},
                params={'batch_test_param': 10},
                tags={'batch_test_tag': 100},
                step=100 + i)
            time.sleep(0.5)

    def test_log_artifact(self):
        from segmind_track import log_artifact

        tempdir = tempfile.mkdtemp()

        with open(os.path.join(tempdir, 'lalala.txt'), 'w') as f:
            f.write('lolololololo')

        log_artifact(
            key='lalala_artifact.txt',
            path=os.path.join(tempdir, 'lalala.txt'))


class TestKerasCallback(unittest.TestCase):
    """docstring for TestKerasCallback."""

    def setUp(self):
        mocked_experiment_id = '521e7ebc-36c9-4ae8-8507-f7b31a5bd963_20'
        from segmind_track import set_experiment
        set_experiment('a0583ec5-bdf3-4526-a985-05be15e62f16')

    def test_callback(self):
        from segmind_track import KerasCallback

        fashion_mnist = keras.datasets.fashion_mnist
        (train_images,
         train_labels), (test_images, test_labels) = fashion_mnist.load_data()

        train_images = train_images / 255.0

        model = define_mnist_model((28, 28), hidden_neurons=28)

        keras_cb = KerasCallback()

        model.fit(
            train_images,
            train_labels,
            epochs=2,
            steps_per_epoch=10,
            callbacks=[keras_cb])


class TestLightningCallback(unittest.TestCase):
    """docstring for TestLightningCallback."""

    def setUp(self):
        mocked_experiment_id = '521e7ebc-36c9-4ae8-8507-f7b31a5bd963_20'
        from segmind_track import set_experiment
        set_experiment('a0583ec5-bdf3-4526-a985-05be15e62f16')

    def test_callback(self):
        from segmind_track import LightningCallback

        model = LightningMNISTClassifier()

        lightning_cb = LightningCallback()
        trainer = pl.Trainer(callbacks=[lightning_cb], max_epochs=2)

        trainer.fit(model)


if __name__ == '__main__':
    unittest.main()
