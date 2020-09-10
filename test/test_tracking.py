import os
import tempfile
import time
import unittest

from tensorflow import keras

from segmind_track import log_params_decorator


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


class TestTracking(unittest.TestCase):
    """docstring for TestTracking."""

    def setUp(self):
        mocked_experiment_id = '521e7ebc-36c9-4ae8-8507-f7b31a5bd963_20'
        from segmind_track import set_experiment, set_runid
        set_experiment('487f0813-c080-4d8b-8a9d-792b0acf8ad9')

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
        set_experiment('487f0813-c080-4d8b-8a9d-792b0acf8ad9')

    def test_callback(self):
        from segmind_track import KerasCallback

        fashion_mnist = keras.datasets.fashion_mnist
        (train_images,
         train_labels), (test_images, test_labels) = fashion_mnist.load_data()

        train_images = train_images / 255.0

        model = define_mnist_model((28, 28), hidden_neurons=28)

        keras_cb = KerasCallback(log_evry_n_step=5)

        model.fit(
            train_images,
            train_labels,
            epochs=2,
            steps_per_epoch=10,
            callbacks=[keras_cb])


if __name__ == '__main__':
    unittest.main()
