"""Summary."""
from __future__ import absolute_import
import copy
import os
import shutil
import tempfile

import GPUtil
from tensorflow import keras

from .utils.logging_utils import try_mlflow_log
from .tracking.fluent import log_artifact, log_metrics, log_param, set_tag


def log_gpu_metric():
    gpu_data = {}
    try:
        i = 0
        gpus = GPUtil.getGPUs()
        if len(gpus) > 1:
            for gpu in gpus:
                # gpu_data[f'GPU_Name_{i}']=gpu.name
                gpu_data[f'GPU_load_{i}'] = gpu.load
                gpu_data[f'GPU_Memory_util_{i}'] = gpu.memoryUtil
                # gpu_data[f'GPU_Memory_total_{i}']=gpu.memoryTotal
                i += 1
        else:
            gpu = gpus[0]
            gpu_data[f'GPU_load'] = gpu.load
            gpu_data[f'GPU_Memory_util'] = gpu.memoryUtil
            # gpu_data[f'GPU_Memory_total']=gpu.memoryTotal
    except Exception as e:
        e = e
    return gpu_data


class KerasCallback(keras.callbacks.Callback):
    """Callback for auto-logging metrics and parameters. Records available logs
    after each epoch. Records model structural information as params when
    training begins.

    Attributes:
        current_epoch (int): The current epoch being run
        log_evry_n_step (int): every nth step to log value
        num_step (int): number of steps run since last logging
        step_logging (bool): to use step logging or epoch logging
    """

    def __init__(self, log_evry_n_step=None):
        """Constructor of class KerasCallback.

        Args:
            log_evry_n_step (None, optional): If Assigned, then algo with log metrics at every consecutive step,
                                              else at the end of every epoch(default).
        """
        super(KerasCallback, self).__init__()

        if log_evry_n_step != None:
            assert isinstance(
                log_evry_n_step, int
            ), 'argument `log_evry_n_step` expects an integer value got {}'.format(
                type(log_evry_n_step))
        self.num_step = 0
        self.num_test_step = 0
        self.log_evry_n_step = log_evry_n_step
        self.step_logging = False
        if self.log_evry_n_step != None:
            self.step_logging = True
        self.current_epoch = 0

    def on_train_begin(self, logs=None):  # pylint: disable=unused-argument
        """Summary.

        Args:
            logs (None, optional): Description
        """
        try_mlflow_log(log_param, 'num_layers', len(self.model.layers))
        try_mlflow_log(log_param, 'optimizer_name',
                       type(self.model.optimizer).__name__)
        if hasattr(self.model.optimizer, 'lr'):
            lr = self.model.optimizer.lr if \
                type(self.model.optimizer.lr) is float \
                else keras.backend.eval(self.model.optimizer.lr)
            try_mlflow_log(log_param, 'learning_rate', lr)
        if hasattr(self.model.optimizer, 'epsilon'):
            epsilon = self.model.optimizer.epsilon if \
                type(self.model.optimizer.epsilon) is float \
                else keras.backend.eval(self.model.optimizer.epsilon)
            try_mlflow_log(log_param, 'epsilon', epsilon)

        sum_list = []
        self.model.summary(print_fn=sum_list.append)
        summary = '\n'.join(sum_list)
        # try_mlflow_log(set_tag, 'model_summary', summary)

        tempdir = tempfile.mkdtemp()
        try:
            summary_file = os.path.join(tempdir, 'model_summary.txt')
            with open(summary_file, 'w') as f:
                f.write(summary)
            try_mlflow_log(
                log_artifact, key='model_summary.txt', path=summary_file)
        finally:
            shutil.rmtree(tempdir)

    def on_train_batch_end(self, batch, logs=None):
        """Summary.

        Args:
            batch (TYPE): Description
            logs (None, optional): Description

        Returns:
            TYPE: Description
        """
        self.num_step += 1
        if not logs:
            return
        if self.step_logging and self.num_step % self.log_evry_n_step == 0:
            gpu_data = log_gpu_metric()
            logs_copy = copy.deepcopy(logs)
            logs_copy.update(gpu_data)
            try_mlflow_log(log_metrics, logs_copy, step=self.num_step)

    def on_test_batch_end(self, batch, logs=None):
        """Summary.

        Args:
            batch (TYPE): Description
            logs (None, optional): Description

        Returns:
            TYPE: Description
        """
        self.num_test_step += 1
        if not logs:
            return
        if self.step_logging and self.num_test_step % self.log_evry_n_step == 0:
            gpu_data = log_gpu_metric()
            logs_copy = copy.deepcopy(logs)
            logs_copy.update(gpu_data)
            try_mlflow_log(log_metrics, logs_copy, step=self.num_step)

    def on_test_end(self, logs=None):
        if not logs:
            return
        else:
            try_mlflow_log(log_metrics, logs, step=self.num_step)

    def on_test_begin(self, batch, logs=None):
        self.num_test_step = 0

    # def on_test_batch_end(self, batch, logs=None):
    #     self.test_step = batch

    def on_epoch_end(self, epoch, logs=None):
        """Summary.

        Args:
            epoch (TYPE): Description
            logs (None, optional): Description

        Returns:
            TYPE: Description
        """
        self.current_epoch = epoch
        if not logs:
            return
        #try_mlflow_log(log_metrics, logs, step=epoch)
        # if not self.step_logging:
        try_mlflow_log(log_metrics, logs, step=self.num_step)

    #def on_train_end(self, logs=None):
    #    try_mlflow_log(log_model, self.model, artifact_path='model')
