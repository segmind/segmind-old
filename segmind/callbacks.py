"""Summary."""
from __future__ import absolute_import, print_function

import copy
import os
import pytorch_lightning as pl
import shutil
import tempfile
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.callbacks import ModelCheckpoint

from segmind.sys_data import gpu_metrics, system_metrics
from segmind.tracking.fluent import log_artifact, log_metrics, log_param
from segmind.utils.logging_utils import try_mlflow_log


class ModelCheckpointAndUpload(ModelCheckpoint):
    """docstring for ModelCheckpointAndUpload."""

    def _save_model(self, epoch, logs):
        super(ModelCheckpointAndUpload, self)._save_model(epoch, logs)
        filepath = self._get_file_path(epoch, logs)

        if os.path.exists(filepath):
            output_filename = os.path.join(tempfile.gettempdir(),
                                           'checkpoint_segmind_track')

            if os.path.isfile(output_filename):
                os.remove(output_filename)
            # zip filepath folder

            if os.path.isdir(filepath):
                shutil.make_archive(output_filename, 'zip', filepath)
                # log as artifact
                print(f'Uploading checkpoint {output_filename} ...')
                try_mlflow_log(
                    log_artifact,
                    key=os.path.basename(filepath) + '.zip',
                    path=output_filename + '.zip')

            else:
                # log as artifact
                print(f'Uploading checkpoint {filepath} ...')
                try_mlflow_log(
                    log_artifact,
                    key=os.path.basename(filepath),
                    path=filepath)


def CheckpointCallback(snapshot_interval,
                       snapshot_path,
                       checkpoint_prefix,
                       save_h5=True):

    assert isinstance(save_h5, bool)

    tf.io.gfile.makedirs(snapshot_path)

    checkpoint_name = os.path.join(snapshot_path,
                                   str(checkpoint_prefix) + '_{epoch:02d}')

    if save_h5:
        checkpoint_name += '.h5'

    return ModelCheckpointAndUpload(
        checkpoint_name, verbose=1, period=snapshot_interval // 1)


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
            log_evry_n_step (None, optional): If Assigned, then algo with log
            metrics at every consecutive step, else at the end of every epoch
            (default).
        """
        super(KerasCallback, self).__init__()

        if log_evry_n_step is not None:
            assert isinstance(
                log_evry_n_step, int
            ), f'argument `log_evry_n_step` expects an integer value go \
            {type(log_evry_n_step)}'

        self.num_step = 0
        self.num_test_step = 0
        self.log_evry_n_step = log_evry_n_step
        self.step_logging = False

        if self.log_evry_n_step is not None:
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
            gpu_data = gpu_metrics()
            logs_copy = copy.deepcopy(logs)
            logs_copy.update(gpu_data)

            cpu_data = system_metrics()
            logs_copy.update(cpu_data)

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
        if self.step_logging and self.num_test_step % self.log_evry_n_step == 0:  # noqa: E501
            gpu_data = gpu_metrics()
            logs_copy = copy.deepcopy(logs)
            logs_copy.update(gpu_data)
            cpu_data = system_metrics()
            logs_copy.update(cpu_data)
            try_mlflow_log(log_metrics, logs_copy, step=self.num_step)

    def on_test_end(self, logs=None):
        if not logs:
            return
        else:
            try_mlflow_log(log_metrics, logs, step=self.num_step)

    def on_test_begin(self, batch, logs=None):
        self.num_test_step = 0

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
        gpu_data = gpu_metrics()
        logs_copy = copy.deepcopy(logs)
        logs_copy.update(gpu_data)

        cpu_data = system_metrics()
        logs_copy.update(cpu_data)

        try_mlflow_log(log_metrics, logs_copy, step=self.num_step)


class PytorchModelCheckpointAndUpload(pl.callbacks.ModelCheckpoint):
    """docstring for PytorchModelCheckpointAndUpload."""

    def _save_model(self, filepath: str, trainer, pl_module):
        super(PytorchModelCheckpointAndUpload, self)._save_model(filepath, trainer, pl_module)  # noqa: E501
        # filepath = self._get_file_path(epoch, logs)
        trainer.dev_debugger.track_checkpointing_history(filepath)
        if trainer.is_global_zero:
            self._fs.makedirs(os.path.dirname(filepath), exist_ok=True)
        if self.save_function is not None:
            self.save_function(filepath, self.save_weights_only)
        if os.path.exists(filepath):
            output_filename = os.path.join(tempfile.gettempdir(),
                                           'checkpoint_segmind_track')

            if os.path.isfile(output_filename):
                os.remove(output_filename)
            # zip filepath folder

            if os.path.isdir(filepath):
                shutil.make_archive(output_filename, 'zip', filepath)
                # log as artifact
                print(f'Uploading checkpoint {output_filename} ...')
                try_mlflow_log(
                    log_artifact,
                    key=os.path.basename(filepath) + '.zip',
                    path=output_filename + '.zip')

            else:
                # log as artifact
                print(f'Uploading checkpoint {filepath} ...')
                try_mlflow_log(
                    log_artifact,
                    key=os.path.basename(filepath),
                    path=filepath)


def PytorchCheckpointCallback(snapshot_interval,
                              snapshot_path,
                              checkpoint_prefix,
                              save_h5=False):

    assert isinstance(save_h5, bool)

    tf.io.gfile.makedirs(snapshot_path)

    checkpoint_name = os.path.join(snapshot_path,
                                   str(checkpoint_prefix) + '_{epoch:02d}')

    return PytorchModelCheckpointAndUpload(
        checkpoint_name, save_top_k=-1, verbose=True, period=snapshot_interval // 1)  # noqa: E501


class LightningCallback(pl.callbacks.base.Callback):
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
        """Constructor of class pytorch_lightning.callbacks.base.Callback.

        Args:
            log_evry_n_step (None, optional): If Assigned, then algo with
            log metrics at every consecutive step, else at the end
            of every epoch(default).
        """
        super(LightningCallback, self).__init__()

        if log_evry_n_step is not None:
            assert isinstance(
                log_evry_n_step, int
            ), f'argument `log_evry_n_step` expects an integer value go \
            {type(log_evry_n_step)}'

        self.num_step = 0
        self.num_test_step = 0
        self.log_evry_n_step = log_evry_n_step
        self.step_logging = False

        if self.log_evry_n_step is not None:
            self.step_logging = True
        self.current_epoch = 0

    def on_train_start(self, trainer, pl_module):  # pylint: disable=unused-arg

        """Summary.

        Args:
            trainer (TYPE): Description
            pl_module (None, optional): Description

        Returns:
            TYPE: Description
        """

        optimizer = pl_module.configure_optimizers()
        try_mlflow_log(log_param, 'optimizer_name',
                       optimizer.__class__.__name__)
        lr = optimizer.param_groups[0]['lr']
        try_mlflow_log(log_param, 'learning_rate', lr)
        print('learning rate value is ', lr)
        epsilon = optimizer.param_groups[0]['eps']
        try_mlflow_log(log_param, 'epsilon', epsilon)
        print('epsilon value is ', epsilon)

        sum_list = []
        x = pl_module.summarize()
        x = str(x)
        sum_list.append(x)
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

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx,
                           dataloader_idx):
        """Summary.

        Args:
            trainer (TYPE): Description
            pl_module (None, optional): Description
            batch (TYPE) : Description
            batch_idx (TYPE) : Description
            dataloader_idx (TYPE) : Description
        Returns:
            TYPE: Description
        """

        self.num_step += 1
        logs = trainer.logger_connector.callback_metrics

        if self.step_logging and self.num_step % self.log_evry_n_step == 0:
            gpu_data = gpu_metrics()
            logs_copy = copy.deepcopy(logs)
            logs_copy.update(gpu_data)

            cpu_data = system_metrics()
            logs_copy.update(cpu_data)

            try_mlflow_log(log_metrics, logs_copy, step=self.num_step)

    def on_test_batch_end(self, trainer, pl_module, outputs, batch, batch_idx,
                          dataloader_idx):
        """Summary.

        Args:
            trainer (TYPE): Description
            pl_module (None, optional): Description
            batch (TYPE) : Description
            batch_idx (TYPE) : Description
            dataloader_idx (TYPE) : Description
        Returns:
            TYPE: Description
        """
        self.num_test_step += 1
        logs = trainer.logger_connector.callback_metrics

        if self.step_logging and self.num_test_step % self.log_evry_n_step == 0:  # noqa: E501
            gpu_data = gpu_metrics()
            logs_copy = copy.deepcopy(logs)
            logs_copy.update(gpu_data)
            cpu_data = system_metrics()
            logs_copy.update(cpu_data)
            try_mlflow_log(log_metrics, logs_copy, step=self.num_step)

    def on_test_end(self, trainer, pl_module):
        """Summary.

        Args:
            trainer (TYPE): Description
            pl_module (None, optional): Description

        Returns:
            TYPE: Description
        """
        logs = trainer.logger_connector.callback_metrics
        try_mlflow_log(log_metrics, logs, step=self.num_step)

    def on_test_start(self, trainer, pl_module):
        self.num_test_step = 0

    # def on_test_batch_end(self, batch, logs=None):
    #     self.test_step = batch

    def on_epoch_end(self, trainer, pl_module):
        """Summary.

        Args:
            trainer (TYPE): Description
            pl_module (None, optional): Description

        Returns:
            TYPE: Description
        """
        # self.current_epoch = epoch
        logs = trainer.logger_connector.callback_metrics

        gpu_data = gpu_metrics()
        logs_copy = copy.deepcopy(logs)
        logs_copy.update(gpu_data)

        cpu_data = system_metrics()
        logs_copy.update(cpu_data)

        try_mlflow_log(log_metrics, logs_copy, step=self.num_step)
        print('end of epoch')


def XGBoost_callback(period=1):
    """Create a callback that print evaluation result.
    We print the evaluation results every **period** iterations
    and on the first and the last iterations.
    Parameters
    ----------
    period : int
        The period to log the evaluation results
    show_stdv : bool, optional
         Whether show stdv if provided
    Returns
    -------
    callback : function
        A callback that print evaluation every period iterations.
    """
    def callback(env):
        """internal function."""
        if env.rank != 0 or (not env.evaluation_result_list) or period is False or period == 0:  # noqa: E501
            return
        step = env.iteration

        results = {}
        gpu_data = gpu_metrics()
        results.update(gpu_data)

        cpu_data = system_metrics()
        results.update(cpu_data)
        if step % period == 0 or step + 1 == env.begin_iteration or step + 1 == env.end_iteration:  # noqa: E501
            for x in env.evaluation_result_list:
                results[x[0]] = x[1]
            try_mlflow_log(log_metrics, results, step=step)

    return callback
