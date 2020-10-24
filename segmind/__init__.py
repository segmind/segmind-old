"""The `segmind_track` module provides a high-level API for tracking and
logging metadata and artifacts for DeepLearning Algorithms."""

# Filter annoying Cython warnings that serve no good purpose, and so before
# importing other modules.
# See: https://github.com/numpy/numpy/pull/432/commits/170ed4e33d6196d7
import warnings

# pylint: disable=wrong-import-position
from segmind import projects  # noqa
from segmind import tracking  # noqa
from .callbacks import (CheckpointCallback, KerasCallback,  # noqa: F401
                        LightningCallback, PytorchCheckpointCallback,
                        XGBoost_callback)
from .tracking import fluent
from .utils.logging_utils import log_params_decorator  # noqa: F401
from .utils.logging_utils import _configure_mlflow_loggers
from .version import VERSION as __version__  # noqa: F401

warnings.filterwarnings(
    'ignore', message='numpy.dtype size changed')  # noqa: E402
warnings.filterwarnings(
    'ignore', message='numpy.ufunc size changed')  # noqa: E402
# log a deprecated warning only once per function per module
warnings.filterwarnings('module', category=DeprecationWarning)

_configure_mlflow_loggers(__name__)


def ActiveRun(*args, **kwargs):
    """Wrapper around :py:class:`cral.tracking.entities.Run` to enable using
    Python ``with`` syntax.

    Args:
        *args: Description
        **kwargs: Description

    Returns:
        TYPE: Description
    """
    return fluent.ActiveRun(*args, **kwargs)


def log_param(key, value):
    """Log a parameter under the current run. If no run is active, this method
    will create a new active run.

    Args:
        key: Parameter name (string)
        value: Parameter value (string, but will be string-ified if not)

    Examples:
        Logging a simple parameter.

        ```
        from segmind import set_project, log_param
        set_project('xxx-xxx-xxx-xxx');
        log_param('param1', 100);
        ```
    """
    return fluent.log_param(key, value)


def log_params(params):
    """Log a batch of params for the current run. If no run is active, this
    method will create a new active run.

    Args:
        params: Dictionary of param_name: String -> value: (String, but will
                be string-ified if not)
    """
    return fluent.log_params(params)


def log_metric(key, value, step=None):
    """Log a metric under the current run. If no run is active, this method
    will create a new active run.

    Args:
        key: Metric name (string).
        value: Metric value (float). Note that some special values such
             as +/- Infinity may be replaced by other values
             depending on the store. For example, sFor example, the
             SQLAlchemy store replaces +/- Inf with max / min float
             values.
        step: Metric step (int). Defaults to zero if unspecified.
    """
    return fluent.log_metric(key, value, step)


def log_metrics(metrics, step=None):
    """Log multiple metrics for the current run. If no run is active, this
    method will create a new active run.

    Args:
        metrics: Dictionary of metric_name: String -> value: Float. Note that
                 some special values such as +/- Infinity may be replaced by
                 other values depending on the store. For example, sql based
                 store may replace +/- Inf with max / min float values.
        step: A single integer step at which to log the specified
                 Metrics. If unspecified, each metric is logged at step zero.
    """
    return fluent.log_metrics(metrics, step)


# def set_tag(key, value):
#     """Set a tag under the current run. If no run is active, this method will
#     create a new active run.

#     Args:
#         key: Tag name (string)
#         value: Tag value (string, but will be string-ified if not)
#     """
#     return fluent.set_tag(key, value)

# def set_tags(*args, **kwargs):
#     """Log a batch of tags for the current run. If no run is active, this
#     method will create a new active run.

#     Args:
#         tags: Dictionary of tag_name: String -> value: (String, but will
#               be string-ified if not)
#     """
#     return fluent.set_tags(*args, **kwargs)

# def delete_tag(*args, **kwargs):
#     """Delete a tag from a run. This is irreversible. If no run is active,
#        this method will create a new active run.

#     :param key: Name of the tag

#     Args:
#         *args: Description
#         **kwargs: Description

#     Returns:
#         TYPE: Description
#     """
#     return fluent.delete_tag(*args, **kwargs)


def active_run(*args, **kwargs):
    """Get the currently active ``Run``, or None if no such run exists.

    **Note**: You cannot access currently-active run attributes
    (parameters, metrics, etc.) through the run returned by
    ``cral.tracking.active_run``. In order to access such attributes, use the
    :py:class:`cral.tracking.tracking.MlflowClient` as follows:

    .. code-block:: py

        client = cral.tracking.tracking.MlflowClient()
        data = client.get_run(cral.tracking.active_run().info.run_id).data

    Args:
        *args: Description
        **kwargs: Description

    Returns:
        TYPE: Description
    """
    return fluent.active_run(*args, **kwargs)


def get_run(*args, **kwargs):
    """
    Fetch the run from backend store. The resulting
    :py:class:`Run <cral.tracking.entities.Run>` contains a collection of run
    metadata -- :py:class:`RunInfo <cral.tracking.entities.RunInfo>`, as well
    as a collection of run parameters, tags, and metrics --
    :py:class:`RunData <cral.tracking.entities.RunData>`. In the case where
    multiple metrics with the same key are logged for the run, the
    :py:class:`RunData <cral.tracking.entities.RunData>` contains
    the most recently logged value at the largest step for each metric.

    :param run_id: Unique identifier for the run.

    :return: A single :py:class:`cral.tracking.entities.Run` object, if the
             run exists. Otherwise, raises an exception.

    Args:
        *args: Description
        **kwargs: Description

    Returns:
        TYPE: Description
    """
    return fluent.get_run(*args, **kwargs)


def start_run(*args, **kwargs):
    """Start a new MLflow run, setting it as the active run under which metrics
    and parameters will be logged. The return value can be used as a context
    manager within a ``with`` block; otherwise, you must call ``end_run()`` to
    terminate the current run.

    If you pass a ``run_id`` or the ``MLFLOW_RUN_ID`` environment variable is
    set, ``start_run`` attempts to resume a run with the specified run ID and
    other parameters are ignored. ``run_id`` takes precedence over
    ``MLFLOW_RUN_ID``.

    MLflow sets a variety of default tags on the run, as defined in
    :ref:`MLflow system tags <system_tags>`.

    Args:
        run_id: If specified, get the run with the specified UUID and log
                   parameters and metrics under that run. The run's end time
                   is unset and its status is set to running, but the run's
                   other attributes (``source_version``, ``source_type``,
                   etc.) are not changed.
        experiment_id: ID of the experiment under which to create the
                   current run (applicable only when ``run_id`` is not
                   specified). If ``experiment_id`` argument is unspecified,
                   will look for valid experiment in the following order:
                   activated using ``set_project``,
                   ``MLFLOW_EXPERIMENT_NAME`` environment variable,
                   ``MLFLOW_EXPERIMENT_ID`` environment variable, or the
                   default experiment as defined by the tracking server.
        run_name: Name of new run (stored as a ``cral.tracking.runName``
                     tag). Used only when ``run_id`` is unspecified.
        nested: Controls whether run is nested in parent run. ``True``
                   creates a nest run.

    Returns:
        py:class:`cral.tracking.ActiveRun` object that acts as a context
                 manager wrapping the run's state.
    """
    return fluent.start_run(*args, **kwargs)


def end_run(*args, **kwargs):
    """End an active MLflow run (if there is one).

    Args:
        *args: Description
        **kwargs: Description

    Returns:
        TYPE: Description
    """
    return fluent.end_run(*args, **kwargs)


def search_runs(*args, **kwargs):
    """Get a pandas DataFrame of runs that fit the search criteria.

    Args:
    experiment_ids: List of experiment IDs. None will default to the active
                    experiment.
    filter_string: Filter query string, defaults to searching all runs.
    run_view_type: one of enum values ``ACTIVE_ONLY``, ``DELETED_ONLY``, or
                   ``ALL`` runs defined in :py:class:`cral.tracking.entities.ViewType`. # noqa
    max_results: The maximum number of runs to put in the dataframe. Default
                 is 100,000 to avoid causing out-of-memory issues on the
                 user's machine.
    order_by: List of columns to order by (e.g., "metrics.rmse"). The
              ``order_by`` column can contain an optional ``DESC`` or
              ``ASC`` value. The default is ``ASC``. The default ordering
              is to sort by ``start_time DESC``, then ``run_id``.

    Returns:
        pandas.DataFrame: A pandas.DataFrame of runs, where each metric,
        parameter, and tag are expanded into their own columns named
        metrics.*, params.*, and tags.* respectively. For runs that don't have
        a particular metric, parameter, or tag, their value will be (NumPy)
        Nan, None, or None respectively.
    """
    return fluent.search_runs(*args, **kwargs)


def set_project(project_id):
    """Set given experiment as active experiment. If experiment does not exist,
    create an experiment with provided name.

    Args:
        project_id: uuid of experiment to be activated.
    """
    return fluent.set_project(project_id)


def set_runid(*args, **kwargs):
    """Summary.

    Args:
        *args: Description
        **kwargs: Description

    Returns:
        TYPE: Description
    """
    return fluent.set_runid(*args, **kwargs)


def log_batch(*args, **kwargs):
    """Summary.

    Args:
        *args: Description
        **kwargs: Description

    Returns:
        TYPE: Description
    """
    return fluent.log_batch(*args, **kwargs)


get_experiment = fluent.get_experiment
# get_experiment_by_name = fluent.get_experiment_by_name

# TO DO: @pk00095
# Enable only for server
# create_experiment = fluent.create_experiment
# delete_experiment = fluent.delete_experiment
# delete_run = fluent.delete_run

# TO DO: @pk00095
# remove:
# get_artifact_uri = fluent.get_artifact_uri
# set_tracking_uri = tracking.set_tracking_uri
get_tracking_uri = tracking.get_tracking_uri
# register_model = _model_registry.fluent.register_model
# log_artifacts = fluent.log_artifacts
log_artifact = fluent.log_artifact
log_table = fluent.log_table
log_image = fluent.log_image
log_bbox_prediction = fluent.log_bbox_prediction
log_mask_prediction = fluent.log_mask_prediction

run = projects.run

__all__ = [
    'ActiveRun', 'log_param', 'log_params', 'log_metric', 'log_metrics',
    'set_tag', 'set_tags', 'delete_tag', 'log_artifacts', 'log_artifact',
    'active_run', 'start_run', 'end_run', 'search_runs', 'get_artifact_uri',
    'set_tracking_uri', 'create_experiment', 'set_project',
    'delete_experiment', 'delete_run', 'run', 'register_model'
]
