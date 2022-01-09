import numbers
from mmcv.runner import HOOKS, LoggerHook, master_only

from segmind.tracking.fluent import log_metrics
from segmind.utils.logging_utils import try_mlflow_log


@HOOKS.register_module()
class SegmindLoggerHook(LoggerHook):
    def __init__(
        self, init_kwargs=None, interval=10, ignore_last=True, reset_flag=True
    ):
        super(SegmindLoggerHook, self).__init__(interval, ignore_last, reset_flag)
        self.init_kwargs = init_kwargs

    @master_only
    def before_run(self, runner):
        pass

    @master_only
    def log(self, runner):
        metrics = {}

        for var, val in runner.log_buffer.output.items():
            if var in ["time", "data_time"]:
                continue

            tag = f"{var}_{runner.mode}"
            if isinstance(val, numbers.Number):
                metrics[tag] = val

        metrics["learning_rate"] = runner.current_lr()[0]
        metrics["momentum"] = runner.current_momentum()[0]

        # logging metrics to segmind
        try_mlflow_log(log_metrics, metrics, step=runner.epoch, epoch=runner.epoch)

    @master_only
    def after_run(self, runner):
        pass


def init_segmind_hook(cfg=None):
    custom_imports = dict(imports=["SegmindLoggerHook"], allow_failed_imports=False)
    custom_hooks = [dict(type="SegmindLoggerHook")]

    if cfg:
        cfg.custom_imports = custom_imports
        cfg.custom_hooks = custom_hooks

    return custom_imports, custom_hooks, cfg
