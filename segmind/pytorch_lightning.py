from __future__ import absolute_import, print_function

import os
import pytorch_lightning as pl
import shutil
import tempfile

from segmind.sys_data import gpu_metrics, system_metrics
from segmind.tracking.fluent import log_artifact, log_metrics, log_param
from segmind.utils.logging_utils import try_mlflow_log


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
                              checkpoint_prefix):

    os.makedirs(snapshot_path, exist_ok=True)
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

        # sys_data = gpu_metrics()
        # Removing system_metrics for now, as these are not frequently used
        # sys_data.update(system_metrics())
        if self.step_logging and self.num_step % self.log_evry_n_step == 0:
            # try_mlflow_log(
            #     log_metrics,
            #     sys_data,
            #     step=self.num_step,
            #     epoch=self.current_epoch,
            #     tags={'sys_metric': 'yes'})
            try_mlflow_log(
                log_metrics,
                logs,
                epoch=self.current_epoch,
                step=self.num_step)

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
            # sys_data = gpu_metrics()
            # sys_data.update(system_metrics())
            # try_mlflow_log(
            #     log_metrics,
            #     sys_data,
            #     step=self.num_step,
            #     epoch=self.current_epoch,
            #     tags={'sys_metric': 'yes'})
            try_mlflow_log(
                log_metrics,
                logs,
                step=self.num_step,
                epoch=self.current_epoch,)

    def on_test_end(self, trainer, pl_module):
        """Summary.

        Args:
            trainer (TYPE): Description
            pl_module (None, optional): Description

        Returns:
            TYPE: Description
        """
        logs = trainer.logger_connector.callback_metrics
        try_mlflow_log(
            log_metrics,
            logs,
            step=self.num_step)

    def on_test_start(self, trainer, pl_module):
        self.num_test_step = 0

    # def on_epoch_start(self, trainer, pl_module):
    #     pass

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
        self.current_epoch += 1

        # sys_data = gpu_metrics()
        # Removing system_metrics for now, as these are not frequently used
        # sys_data.update(system_metrics())
        # try_mlflow_log(
        #     log_metrics,
        #     sys_data,
        #     step=self.num_step,
        #     epoch=self.current_epoch,
        #     tags={'sys_metric': 'yes'})
        try_mlflow_log(
            log_metrics,
            logs,
            step=self.num_step,
            epoch=self.current_epoch)
        # print('end of epoch')
