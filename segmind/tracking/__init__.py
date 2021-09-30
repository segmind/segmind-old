"""The ``segmind_track.tracking`` module provides a Python CRUD interface to
MLflow experiments and runs.

This is a lower level API that directly translates to MLflow `REST API
<../rest-api.html>`_ calls. For a higher level API for managing an "active
run", use the :py:mod:`mlflow` module.
"""

from segmind.tracking._tracking_service.utils import (_TRACKING_URI_ENV_VAR,
                                                      _get_store,
                                                      get_tracking_uri,
                                                      is_tracking_uri_set,
                                                      set_tracking_uri)
from segmind.tracking.client import MlflowClient
from segmind.tracking.fluent import _EXPERIMENT_ID_ENV_VAR, _RUN_ID_ENV_VAR
from segmind.tracking.fluent import set_project, create_experiment, log_param, log_params, log_metric, log_metrics, \
    log_artifact, log_image, log_table, set_runid, start_run, active_run, end_run, search_runs, get_run, delete_run
from segmind.utils.logging_utils import log_params_decorator

__all__ = [
    'MlflowClient',
    'get_tracking_uri',
    'set_tracking_uri',
    'is_tracking_uri_set',
    '_get_store',
    '_EXPERIMENT_ID_ENV_VAR',
    '_RUN_ID_ENV_VAR',
    '_TRACKING_URI_ENV_VAR',

    # tracking endpoints
    'set_project',
    'create_experiment',
    'start_run',
    'set_runid',
    'log_param',
    'log_params',
    'log_params_decorator',
    'log_metric',
    'log_metrics',
    'log_artifact',
    'log_image',
    'log_table',
    'active_run',
    'end_run',
    'search_runs',
    'get_run',
    'delete_run',
]
