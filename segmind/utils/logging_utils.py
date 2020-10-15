from __future__ import print_function

import inspect
import logging
import logging.config
import sys
import warnings

from segmind.tracking import fluent

# Logging format example:
# 2018/11/20 12:36:37 INFO cral.tracking.sagemaker: Creating new SageMaker endpoint # noqa: E501
LOGGING_LINE_FORMAT = '%(asctime)s %(levelname)s %(name)s: %(message)s'
LOGGING_DATETIME_FORMAT = '%Y/%m/%d %H:%M:%S'


def try_mlflow_log(fn, *args, **kwargs):
    """Catch exceptions and log a warning to avoid autolog throwing."""
    try:
        fn(*args, **kwargs)
    except Exception as e:  # pylint: disable=broad-except
        warnings.warn(
            'Logging to Refine server failed: ' + str(e), stacklevel=2)


def log_params_decorator(fn):

    def log_fn_args_as_params(*args, **kwargs):  # pylint: disable=W0102
        """Log parameters explicitly passed to a function.

        :param fn: function whose parameters are to be logged
        :param args: arguments explicitly passed into fn
        :param kwargs: kwargs explicitly passed into fn
        :return: None
        """
        # all_default_values has length n, corresponding to values of the
        # last n elements in all_param_names
        all_param_names, _, _, all_default_values = inspect.getargspec(fn)

        unnamed_args_length = len(args)

        _args_as_dict = dict(zip(all_param_names[:unnamed_args_length], args))

        num_args_without_default_value = len(all_param_names) - len(
            all_default_values)
        default_param_names = all_param_names[num_args_without_default_value:]
        _defaults_as_dict = dict(zip(default_param_names, all_default_values))
        _defaults_as_dict.update(**kwargs)

        _args_as_dict.update(**_defaults_as_dict)

        result = fn(*args, **kwargs)

        try_mlflow_log(fluent.log_params, _args_as_dict)

        return result

    return log_fn_args_as_params


def _configure_mlflow_loggers(root_module_name):
    logging.config.dictConfig({
        'version': 1,
        'disable_existing_loggers': False,
        'formatters': {
            'mlflow_formatter': {
                'format': LOGGING_LINE_FORMAT,
                'datefmt': LOGGING_DATETIME_FORMAT,
            },
        },
        'handlers': {
            'mlflow_handler': {
                'level': 'INFO',
                'formatter': 'mlflow_formatter',
                'class': 'logging.StreamHandler',
                'stream': sys.stderr,
            },
        },
        'loggers': {
            root_module_name: {
                'handlers': ['mlflow_handler'],
                'level': 'INFO',
                'propagate': False,
            },
        },
    })


def eprint(*args, **kwargs):
    print(*args, file=sys.stderr, **kwargs)
