# import os
# import pytorch_lightning as pl
# import tempfile
# import time
# import torch
# import unittest
# from tensorflow import keras
# from torch.nn import functional as F
# from torch.utils.data import DataLoader, random_split
# from torchvision import transforms
# from torchvision.datasets import MNIST
#
# from segmind import log_params_decorator
#
#
# @log_params_decorator
# def define_mnist_model(input_shape, hidden_neurons, num_classes=10):
#     model = keras.Sequential([
#         keras.layers.Flatten(input_shape=(28, 28)),
#         keras.layers.Dense(hidden_neurons, activation='relu'),
#         keras.layers.Dense(num_classes, activation='softmax')
#     ])
#
#     model.compile(
#         optimizer='adam',
#         loss='sparse_categorical_crossentropy',
#         metrics=['accuracy'])
#
#     return model
#
#
# class MNIST_small(MNIST):
#     """docstring for MNIST_small."""
#     def __init__(self, *args, **kwargs):
#         super(MNIST_small, self).__init__(*args, **kwargs)
#
#     def __len__(self):
#         return 100
#
#
# class LightningMNISTClassifier(pl.core.LightningModule):
#
#     def __init__(self):
#         super(LightningMNISTClassifier, self).__init__()
#         # mnist images are (1, 28, 28) (channels, width, height)
#         # num_classes = 10
#         self.layer_1 = torch.nn.Linear(28 * 28, 128)
#         self.layer_2 = torch.nn.Linear(128, 256)
#         self.layer_3 = torch.nn.Linear(256, 10)
#
#     def forward(self, x):
#         batch_size, channels, width, height = x.size()
#
#         # (b, 1, 28, 28) -> (b, 1*28*28)
#         x = x.view(batch_size, -1)
#
#         # layer 1 (b, 1*28*28) -> (b, 128)
#         x = self.layer_1(x)
#         x = torch.relu(x)
#
#         # layer 2 (b, 128) -> (b, 256)
#         x = self.layer_2(x)
#         x = torch.relu(x)
#
#         # layer 3 (b, 256) -> (b, 10)
#         x = self.layer_3(x)
#
#         # probability distribution over labels
#         x = torch.log_softmax(x, dim=1)
#
#         return x
#
#     def cross_entropy_loss(self, logits, labels):
#         return F.nll_loss(logits, labels)
#
#     def training_step(self, train_batch, batch_idx):
#         x, y = train_batch
#         logits = self.forward(x)
#         loss = self.cross_entropy_loss(logits, y)
#
#         logs = {'train_loss': loss}
#         return {'loss': loss, 'log': logs}
#
#     def validation_step(self, val_batch, batch_idx):
#         x, y = val_batch
#         logits = self.forward(x)
#         loss = self.cross_entropy_loss(logits, y)
#         return {'val_loss': loss}
#
#     def validation_epoch_end(self, outputs):
#         # called at the end of the validation epoch
#         # outputs is an array with what you returned in validation_step
#         # for each batch
#         # outputs = [{'loss': batch_0_loss}, {'loss': batch_1_loss},
#         # ..., {'loss': batch_n_loss}]
#         avg_loss = torch.stack([x['val_loss'] for x in outputs]).mean()
#         tensorboard_logs = {'val_loss': avg_loss}
#         return {'avg_val_loss': avg_loss, 'log': tensorboard_logs}
#
#     def prepare_data(self):
#         # transforms for images
#         transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])  # noqa: E501
#
#         # prepare transforms standard to MNIST
#         tempdir = tempfile.mkdtemp()
#         mnist_train = MNIST_small(tempdir, train=True, download=True,
#                             transform=transform)
#         mnist_test = MNIST_small(tempdir, train=False, download=True,  # noqa: F841, E501
#                            transform=transform)
#
#         self.mnist_train, self.mnist_val = random_split(mnist_train,
#                                                         [80, 20])
#
#         # def train_len(self):
#         #     return 100
#
#         # self.mnist_train.__len__ = 100
#         # self.mnist_val.__len__ = 100
#
#
#     def train_dataloader(self):
#         return DataLoader(self.mnist_train, batch_size=10)
#
#     def val_dataloader(self):
#         return DataLoader(self.mnist_val, batch_size=10)
#
#     def test_dataloader(self):
#         return DataLoader(self.mnist_test, batch_size=10)
#
#     def configure_optimizers(self):
#         optimizer = torch.optim.Adam(self.parameters(), lr=1e-3)
#         return optimizer
#
#
# class TestTracking(unittest.TestCase):
#     """docstring for TestTracking."""
#
#     def setUp(self):
#         pass
#         # from segmind import set_project, set_runid
#         # set_project('a0583ec5-bdf3-4526-a985-05be15e62f16')
#
#     def test_log_param(self):
#         from segmind import log_param
#         log_param(
#             key='test_param',
#             value='A')
#
#     def test_log_params(self):
#         from segmind import log_params
#         log_params(
#             params={
#                 'test_params1': 100,
#                 'test_params2': 56
#             })
#
#     def test_log_metric(self):
#         from segmind import log_metric
#
#         log_metric(key='test_step_metric', value=100, step=1)
#         log_metric(key='test_step_metric', value=200, step=5)
#         log_metric(key='test_step_metric', value=300, step=10)
#
#         log_metric(key='test_epoch_metric', value=100, epoch=1)
#         log_metric(key='test_epoch_metric', value=150, epoch=2)
#         log_metric(key='test_epoch_metric', value=200, epoch=3)
#
#     def test_log_metrics(self):
#         from segmind import log_metrics
#
#         log_metrics(metrics={
#                 'test_batch_metric1': 50,
#                 'test_batch_metric2': 1
#             },
#             step=10)
#         log_metrics(metrics={
#                 'test_batch_metric1': 100,
#                 'test_batch_metric2': 50
#             },
#             step=20)
#
#     def test_log_batch(self):
#         from segmind import log_batch
#
#         for i in range(1, 3):
#             log_batch(
#                 metrics={'batch_test_metric': 1 + i},
#                 params={'batch_test_param': 10},
#                 tags={'batch_test_tag': 100},
#                 step=100 + i,
#                 epoch=i)
#             time.sleep(0.5)
#
#     def test_log_artifact(self):
#         from segmind import log_artifact
#
#         tempdir = tempfile.mkdtemp()
#
#         with open(os.path.join(tempdir, 'lalala.txt'), 'w') as f:
#             f.write('lolololololo')
#
#         log_artifact(
#             key='lalala_artifact.txt',
#             path=os.path.join(tempdir, 'lalala.txt'))
#
#
# class TestKerasCallback(unittest.TestCase):
#     """docstring for TestKerasCallback."""
#
#     def setUp(self):
#         pass
#         # from segmind import set_project
#         # set_project('a0583ec5-bdf3-4526-a985-05be15e62f16')
#
#     def test_callback(self):
#         from segmind.keras import CheckpointCallback, KerasCallback
#
#         fashion_mnist = keras.datasets.fashion_mnist
#         (train_images,
#          train_labels), (test_images, test_labels) = fashion_mnist.load_data()
#
#         train_images = train_images / 255.0
#
#         model = define_mnist_model((28, 28), hidden_neurons=28)
#
#         keras_cb = KerasCallback()
#         ckpt_cb = CheckpointCallback(
#             snapshot_interval=10,
#             snapshot_path='/tmp',
#             checkpoint_prefix='keras_callback_mnist')
#
#         # model.fit(
#         #     train_images,
#         #     train_labels,
#         #     epochs=10,
#         #     steps_per_epoch=10,
#         #     callbacks=[keras_cb, ckpt_cb])
#
#
# class TestLightningCallback(unittest.TestCase):
#     """docstring for TestLightningCallback."""
#
#     def setUp(self):
#         pass
#         # from segmind import set_project
#         # set_project('a0583ec5-bdf3-4526-a985-05be15e62f16')
#
#     def test_callback(self):
#         from segmind.pytorch_lightning import LightningCallback
#
#         if torch.cuda.is_available():
#             gpus = -1
#         else:
#             gpus = None
#
#         model = LightningMNISTClassifier()
#
#         lightning_cb = LightningCallback()
#         trainer = pl.Trainer(
#             callbacks=[lightning_cb],
#             gpus=gpus,
#             max_epochs=10)
#
#         trainer.fit(model)
#
#
# class TestXGBoostCallback(unittest.TestCase):
#     """docstring for TestXGBoostCallback."""
#     def setUp(self):
#         pass
#
#     def test_callback(self):
#         import xgboost as xgb
#         from sklearn.datasets import load_breast_cancer
#         from sklearn.model_selection import train_test_split
#
#         from segmind.xgboost import XGBoost_callback
#
#         X, y = load_breast_cancer(return_X_y=True)
#         X_train, X_valid, y_train, y_valid = train_test_split(X, y, random_state=0)
#         D_train = xgb.DMatrix(X_train, y_train)
#         D_valid = xgb.DMatrix(X_valid, y_valid)
#
#         num_boost_round = 10
#
#         # Pass it to the `callbacks` parameter as a list.
#         xgb.train(
#             {
#                 'objective': 'binary:logistic',
#                 'eval_metric': ['error', 'rmse'],
#                 'tree_method': 'hist'
#             },
#             D_train,
#             evals=[(D_train, 'Train'), (D_valid, 'Valid')],
#             num_boost_round=num_boost_round,
#             callbacks=[XGBoost_callback(period=5)])


if __name__ == '__main__':
    # unittest.main(verbosity=2)
    pass
