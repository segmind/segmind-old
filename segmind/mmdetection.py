from mmcv.runner import HOOKS, LoggerHook, master_only

from segmind.tracking.fluent import log_metrics
from segmind.utils.logging_utils import try_mlflow_log


@HOOKS.register_module()
class SegmindLoggerHook(LoggerHook):
    """log metrics to Segmind.

    Args:
        interval (int): Logging interval (every k iterations). Default: 10.
        ignore_last (bool): Ignore the log of last iterations in each epoch
            if less than `interval`. Default True.
        reset_flag (bool): Whether to clear the output buffer after logging.
            Default False.
        by_epoch (bool): Whether EpochBasedRunner is used. Default True.
    """

    def __init__(self, interval=10, ignore_last=True, reset_flag=False, by_epoch=True):
        super(SegmindLoggerHook, self).__init__(
            interval, ignore_last, reset_flag, by_epoch
        )

    @master_only
    def log(self, runner):
        tags = self.get_loggable_tags(runner)
        if tags:
            # logging metrics to segmind
            try_mlflow_log(log_metrics, tags, step=runner.epoch, epoch=runner.epoch)


def init_segmind_hook(cfg=None):
    custom_imports = dict(imports=["SegmindLoggerHook"], allow_failed_imports=False)
    custom_hooks = [dict(type="SegmindLoggerHook")]

    if cfg:
        cfg.custom_imports = custom_imports
        cfg.custom_hooks = custom_hooks

    return custom_imports, custom_hooks, cfg
