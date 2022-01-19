"""Internal module implementing the fluent API, allowing management of an
active MLflow run.

This module is exposed to users at the top-level :py:mod:`mlflow` module.

Attributes:
    NUM_RUNS_PER_PAGE_PANDAS (int): Description
    SEARCH_MAX_RESULTS_PANDAS (int): Description
"""

from __future__ import print_function

import atexit
import logging
import numpy as np
import os
import pandas as pd
import PIL.Image
import time
from google.protobuf.struct_pb2 import Struct
from tempfile import gettempdir

from segmind.entities import Metric, Param, Run, RunStatus, RunTag, ViewType
from segmind.entities.lifecycle_stage import LifecycleStage
from segmind.exceptions import MlflowException
from segmind.lite_extensions.client_utils import (_EXPERIMENT_ID_ENV_VAR,
                                                  _RUN_ID_ENV_VAR,
                                                  _get_experiment_id,
                                                  _runid_exists,
                                                  get_token)
from segmind.protos.service_lite_pb2 import RunTag as RunTagProto
from segmind.tracking.client import MlflowClient
from segmind.tracking.context import registry as context_registry
from segmind.utils import env
from segmind.utils.mlflow_tags import MLFLOW_PARENT_RUN_ID, MLFLOW_RUN_NAME
from segmind.utils.validation import _validate_run_id
from segmind.data.public import upload as upload_data

# _EXPERIMENT_ID_ENV_VAR = "MLFLOW_EXPERIMENT_ID"
# _EXPERIMENT_NAME_ENV_VAR = 'MLFLOW_EXPERIMENT_NAME'
# _RUN_ID_ENV_VAR = "MLFLOW_RUN_ID"
_active_run_stack = []
_active_experiment_id = None

SEARCH_MAX_RESULTS_PANDAS = 100000
NUM_RUNS_PER_PAGE_PANDAS = 10000

_logger = logging.getLogger(__name__)


def convert_to_imagefile(image):
    image_name = os.path.join(gettempdir(), 'temp_image.jpg')

    if isinstance(image, PIL.Image.Image):
        image.save(image_name)
        path = image_name
    elif isinstance(image, np.ndarray):
        image = PIL.Image.fromarray(image.astype(np.uint8))
        image.save(image_name)
        path = image_name
    else:
        path = image

    return path


def set_project(experiment_id):
    """Set given experiment as active experiment. If experiment does not exist,
    create an experiment with provided name.

    Args:
        experiment_id (str): id of experiment to be activated.

    Raises:
        MlflowException: Description
    """
    os.environ[_EXPERIMENT_ID_ENV_VAR] = experiment_id

    client = MlflowClient()

    experiment = client.get_experiment(experiment_id)
    if experiment_id is None:  # id can be 0
        print(f"INFO: '{experiment_id}' does not exist.")
        # experiment_id = client.create_experiment(experiment_name)
    elif experiment.lifecycle_stage == LifecycleStage.DELETED:
        raise MlflowException(
            "Cannot set a deleted experiment '%s' as the active experiment."
            ' You can restore the experiment, or permanently delete the '
            ' experiment to create a new one.' % experiment.name)
    global _active_experiment_id
    _active_experiment_id = experiment_id


def set_runid(run_id):
    """Summary.

    Args:
        run_id (TYPE): Description
    """
    os.environ[_RUN_ID_ENV_VAR] = run_id


class ActiveRun(Run):  # pylint: disable=W0223
    """Wrapper around :py:class:`segmind_track.entities.Run` to enable using
    Python ``with`` syntax."""

    def __init__(self, run):
        """Summary.

        Args:
            run (TYPE): Description
        """
        Run.__init__(self, run.info, run.data)

    def __enter__(self):
        """Summary.

        Returns:
            TYPE: Description
        """
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Summary.

        Args:
            exc_type (TYPE): Description
            exc_val (TYPE): Description
            exc_tb (TYPE): Description

        Returns:
            TYPE: Description
        """
        status = RunStatus.FINISHED if exc_type is None else RunStatus.FAILED
        end_run(RunStatus.to_string(status))
        return exc_type is None


def start_run(run_name=None, nested=False):
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
            parameters and metrics under that run. The run's end time is unset
            and its status is set to running, but the run's other attributes
            (``source_version``, ``source_type``, etc.) are not changed.
        experiment_id: ID of the experiment under which to create the current
            run (applicable only when ``run_id`` is not specified). If
            ``experiment_id`` argument is unspecified, will look for valid
            experiment in the following order: activated using
            ``set_project``, ``MLFLOW_EXPERIMENT_NAME`` environment
            variable, ``MLFLOW_EXPERIMENT_ID`` environment variable,
            or the default experiment as defined by the tracking server.
        run_name: Name of new run (stored as a ``segmind_track.runName`` tag).
            Used only when ``run_id`` is unspecified.
        nested: Controls whether run is nested in parent run. ``True`` creates
            a nest run.

    Returns:
        :py:class:`segmind_track.ActiveRun`: object that acts as a context
            manager wrappings the run's state.


    Raises:
        Exception: Description
        MlflowException: Description
    """
    global _active_run_stack
    # back compat for int experiment_id

    experiment_id = str(_get_experiment_id())

    if _runid_exists():
        existing_run_id = env.get_env(_RUN_ID_ENV_VAR)
    else:
        existing_run_id = None
    if len(_active_run_stack) > 0 and not nested:
        raise Exception(
            ('Run with UUID {} is already active. To start a new run, first ' +
             'end the current run with segmind_track.end_run().' +
             ' To start a nested run, call start_run with nested=True').format(
                 _active_run_stack[0].info.run_id))

    if existing_run_id is not None:
        _validate_run_id(existing_run_id)
        active_run_obj = MlflowClient().get_run(existing_run_id)
        # Check to see if experiment_id from environment matches experiment_id
        # from set_project()
        if (_active_experiment_id is not None and
                _active_experiment_id != active_run_obj.info.experiment_id):
            raise MlflowException(
                'Cannot start run with ID {} because active run ID '
                'does not match environment run ID. Make sure '
                '--experiment-name or --experiment-id matches experiment '
                'set with set_project(), or just use command-line '
                'arguments'.format(existing_run_id))
        # Check to see if current run isn't deleted
        if active_run_obj.info.lifecycle_stage == LifecycleStage.DELETED:
            raise MlflowException(
                'Cannot start run with ID {} because it is in the '
                'deleted state.'.format(existing_run_id))

    else:
        if len(_active_run_stack) > 0:
            parent_run_id = _active_run_stack[-1].info.run_id
        else:
            parent_run_id = None

        exp_id_for_run = experiment_id

        user_specified_tags = {}
        if parent_run_id is not None:
            user_specified_tags[MLFLOW_PARENT_RUN_ID] = parent_run_id
        if run_name is not None:
            user_specified_tags[MLFLOW_RUN_NAME] = run_name

        tags = context_registry.resolve_tags(user_specified_tags)

        active_run_obj = MlflowClient().create_run(
            experiment_id=exp_id_for_run, tags=tags)

    _active_run_stack.append(ActiveRun(active_run_obj))
    return _active_run_stack[-1]


def get_experiment_run(experiment_id=None, run_id=None, create_run_if_not_exists=False):
    experiment_obj = MlflowClient().get_experiment(experiment_id=experiment_id)
    os.environ[_EXPERIMENT_ID_ENV_VAR] = experiment_obj.experiment_id
    if not run_id:
        pass
    else:
        run_obj = MlflowClient().get_run(run_id=run_id)

    return experiment_obj, run_obj


def end_run(status=RunStatus.to_string(RunStatus.FINISHED)):
    """End an active MLflow run (if there is one).

    Args:
        status (TYPE, optional): Description
    """
    global _active_run_stack
    if len(_active_run_stack) > 0:
        MlflowClient().set_terminated(_active_run_stack[-1].info.run_id,
                                      status)
        # Clear out the global existing run environment variable as well.
        env.unset_variable(_RUN_ID_ENV_VAR)
        _active_run_stack.pop()


atexit.register(end_run)


def active_run():
    """Get the currently active ``Run``, or None if no such run exists.

    **Note**: You cannot access currently-active run attributes
    (parameters, metrics, etc.) through the run returned by
    ``segmind_track.active_run``. In order to access such attributes,
    use the :py:class:`segmind_track.tracking.MlflowClient` as follows:

    .. code-block:: py

        client = segmind_track.tracking.MlflowClient()
        data = client.get_run(segmind_track.active_run().info.run_id).data

    Returns:
        TYPE: Description
    """
    return _active_run_stack[-1] if len(_active_run_stack) > 0 else None


def get_run(run_id):
    """
    Fetch the run from backend store. The resulting
    :py:class:`Run <segmind_track.entities.Run>` contains a collection of run
    metadata -- :py:class:`RunInfo <segmind_track.entities.RunInfo>`, as well
    as a collection of run parameters, tags, and metrics --
    :py:class:`RunData <segmind_track.entities.RunData>`. In the case where
    multiple metrics with the same key are logged for the run, the
    :py:class:`RunData <segmind_track.entities.RunData>` contains
    the most recently logged value at the largest step for each metric.

    Args:
        run_id (TYPE): Unique identifier for the run.

    Returns:
        :py:class:`segmind_track.entities.Run`: object, if the run exists.
        Otherwise, raises an exception.
    """
    return MlflowClient().get_run(run_id)


def _get_or_start_run():
    """Summary.

    Returns:
        TYPE: Description
    """
    if len(_active_run_stack) > 0:
        return _active_run_stack[-1]

    return start_run()



def set_tag(key, value):
    """Set a tag under the current run. If no run is active, this method will
    create a new active run.

    Args:
        key (str): Tag name
        value (str): Tag value (string, but will be string-ified if not)
    """
    run_id = _get_or_start_run().info.run_id
    MlflowClient().set_tag(run_id, key, value)


def delete_tag(key):
    """Delete a tag from a run. This is irreversible. If no run is active, this
    method will create a new active run.

    Args:
        key (str): Name of the tag
    """
    run_id = _get_or_start_run().info.run_id
    MlflowClient().delete_tag(run_id, key)


def log_metric(key, value, step=None, epoch=None, tags={'sys_metric': 'no'}):
    """Log a metric under the current run. If no run is active, this method
    will create a new active run.

    Args:
        key (str): Metric name (string).
        value (float): Metric value (float). Note that some special values
                such as +/- Infinity may be replaced by other values depending
                on the store. For example, sFor example, the SQLAlchemy store
                replaces +/- Inf with max / min float values.
        step (int, optional): Metric step (int). Defaults to zero if
                            unspecified.
    """
    tags_arr = [RunTagProto(
        key=key,
        value=str(value)) for key, value in tags.items()]
    run_id = _get_or_start_run().info.run_id
    MlflowClient().log_metric(run_id, key, value, int(time.time() * 1000), step
                              or 0, epoch=epoch or 0, tags=tags_arr)


def log_param(key, value, tags={'sys_param': 'no'}):
    """Log a parameter under the current run. If no run is active, this method
    will create a new active run.

    Args:
        key (str: Parameter name (string)
        value (str): Parameter value (string, but will be string-ified if not)
    """
    tags_arr = [RunTagProto(
        key=key,
        value=str(value)) for key, value in tags.items()]
    run_id = _get_or_start_run().info.run_id
    MlflowClient().log_param(run_id, key, value, tags=tags_arr)


def log_params(params, tags={'sys_param': 'no'}):
    """Log a batch of params for the current run. If no run is active, this
    method will create a new active run.

    Args:
        params (dict): Dictionary of param_name: String -> value: (String, but
                    will be string-ified if not)
    """
    run_id = _get_or_start_run().info.run_id
    tags_arr = [RunTagProto(
        key=key,
        value=str(value)) for key, value in tags.items()]
    params_arr = [Param(
        key,
        str(value),
        tags=tags_arr) for key, value in params.items()]
    MlflowClient().log_batch(
        run_id=run_id, metrics=[], params=params_arr, tags=tags_arr)


def log_metrics(metrics, step=None, epoch=0, tags={'sys_metric': 'no'}):
    """Log multiple metrics for the current run. If no run is active, this
    method will create a new active run.

    Args:
        metrics (TYPE): Dictionary of metric_name: String -> value: Float.
                Note that some special values such as +/- Infinity may be
                replaced by other values depending on the store. For example,
                sql based store may replace +/- Inf with max / min float
                values.
        step (None, optional): A single integer step at which to log the
                specified Metrics. If unspecified, each metric is logged at
                step zero.
    """
    run_id = _get_or_start_run().info.run_id
    timestamp = int(time.time() * 1000)
    tags_arr = [RunTagProto(
        key=key,
        value=str(value)) for key, value in tags.items()]
    metrics_arr = [
        Metric(
            str(key).replace("/", "-"),
            value,
            timestamp,
            step or 0,
            epoch=epoch or 0,
            tags=tags_arr)
        for key, value in metrics.items()
    ]
    MlflowClient().log_batch(
        run_id=run_id, metrics=metrics_arr, params=[], tags=tags_arr)


def log_batch(metrics={}, params={}, tags={}, step=None, epoch=None):
    """Summary.

    Args:
        metrics (dict, optional): Description
        params (dict, optional): Description
        tags (dict, optional): Description
        step (None, optional): Description
    """
    run_id = _get_or_start_run().info.run_id
    tags_arr = [RunTagProto(
        key=key,
        value=str(value)) for key, value in tags.items()]
    params_arr = [Param(key, str(value)) for key, value in params.items()]

    timestamp = int(time.time() * 1000)
    metrics_arr = [
        Metric(key, value, timestamp, step or 0, epoch or 0)
        for key, value in metrics.items()
    ]

    MlflowClient().log_batch(
        run_id=run_id, metrics=metrics_arr, params=params_arr, tags=tags_arr)


def log_artifact(key, path, step=None):
    """Log a local file or directory as an artifact of the currently active
    run. If no run is active, this method will create a new active run.

    Args:
        key (str): Name of the artifact to upload.
        path (str): Path of the artifact to upload.
        step (None, optional): integer indicating the step at which artifact
                            was generated
    """
    run = _get_or_start_run()
    run_id = run.info.run_id
    experiment_id = run.info.experiment_id
    MlflowClient().log_artifact_lite(
        run_id, experiment_id, key, path, step=step)


def log_image(key, image, tags={}, step=None):
    """logs an image artifact.

    Args:
        key (str): name of the table
        image (np.ndarray, PIL.Image): a numpy array of np.uint8 dtype or a
                                    PIL.Image object
        step (None, optional): integer indicating the step at which artifact
                            was generated
    """

    path = convert_to_imagefile(image)

    run = _get_or_start_run()
    run_id = run.info.run_id
    experiment_id = run.info.experiment_id
    MlflowClient().log_artifact_lite(
        run_id,
        experiment_id,
        key,
        path,
        artifact_type='image',
        step=step,
        tags=tags)


def log_table(key, table, step=None):
    """logs a pandas dataframe/csv file as a table artifact.

    Args:
        key (str): name of the table
        table (pd.Dataframe, str): A pandas dataframe or path to a csv file
        step (None, optional): integer indicating the step at which artifact
                            was generated
    Raises:
        MlflowException: Description
    """

    if not isinstance(table, (str, pd.DataFrame)):
        raise MlflowException(
            'table must be a pandas.DataFrame or a string to a csv file')

    path = table
    if isinstance(table, pd.DataFrame):
        path = os.path.join(gettempdir(), 'temp_table.csv')
        table.to_csv(path, index=False)

    run = _get_or_start_run()
    run_id = run.info.run_id
    experiment_id = run.info.experiment_id
    MlflowClient().log_artifact_lite(
        run_id, experiment_id, key, path, artifact_type='table', step=step)


# def log_bbox_prediction(key,
#                         image,
#                         bbox_pred,
#                         bbox_gt=None,
#                         bbox_type='pascal_voc',
#                         step=None):
#     """logs artifact for object detection.
#
#     Args:
#         key (str): name of the image
#         image (np.ndarray, PIL.Image): a numpy array of np.uint8 dtype or a
#                                     PIL.Image object
#         bbox_pred (np.ndarray, list): a list or a np.ndarray of dimension
#                                     (Nx4). All elemnts will be onverted to
#                                     int32.
#         bbox_gt (None, optional): a list or a np.ndarray of dimension (Nx4).
#                                     All elemnts will be onverted to int32
#         bbox_type (str, optional): can be one of 'yolo', 'pascal_voc', 'coco'
#                                 indicating the format of bbox and prediction.
#         step (None, optional): integer indicating the step at which artifact
#                             was generated.
#
#     Raises:
#         MlflowException: Description
#     """
#
#     if bbox_type not in ['pascal_voc', 'coco', 'yolo']:
#         raise MlflowException(
#             f'bbox_type should be one-of "pascal_voc, coco, yolo" not \
#             {bbox_type}')
#
#     path = convert_to_imagefile(image)
#
#     bbox_pred = np.array(bbox_pred)
#     assert isinstance(
#         bbox_pred, np.ndarray) and bbox_pred.ndim == 2 and bbox_pred.shape[
#             1] == 4, f'bbox_pred should be numpy of dimension (Nx4), got \
#         {bbox_pred.shape}'
#
#     if bbox_type == 'coco':
#         bbox_pred = coco_to_voc_bbox(bbox_pred)
#         if bbox_gt:
#             bbox_gt = coco_to_voc_bbox(bbox_gt)
#     elif bbox_type == 'yolo':
#         bbox_pred = yolo_to_voc_bbox(path, bbox_pred)
#         if bbox_gt:
#             bbox_gt = yolo_to_voc_bbox(path, bbox_gt)
#
#     if bbox_gt is None:
#         bbox_gt = np.array([])
#
#     run = _get_or_start_run()
#     run_id = run.info.run_id
#     experiment_id = run.info.experiment_id
#
#     prediction_struct = Struct()
#     prediction_struct.update({'bbox': bbox_pred.tolist()})
#
#     ground_truth_struct = Struct()
#     ground_truth_struct.update({'bbox': bbox_gt.tolist()})
#
#     MlflowClient().log_artifact_lite(
#         run_id,
#         experiment_id,
#         key,
#         path,
#         prediction=prediction_struct,
#         ground_truth=ground_truth_struct,
#         artifact_type='image',
#         step=step,
#         tags={'image_type': 'bbox_prediction'})


# def log_mask_prediction(key,
#                         image,
#                         pred_mask,
#                         bbox_pred=[],
#                         mask_gt=None,
#                         bbox_gt=None,
#                         bbox_type='pascal_voc',
#                         step=None):
#     """logs artifact for instance/semantic segmentation.
#
#     Args:
#         key (str): name of the image
#         image (np.ndarray, PIL.Image): a numpy array of np.uint8 dtype or a
#                                     PIL.Image object
#         pred_mask (np.ndarray, PIL.Image): a numpy array of np.uint8 dtype or
#                                     a PIL.Image object
#         bbox_pred (None, optional): Description
#         mask_gt (np.ndarray, PIL.Image): a numpy array of np.uint8 dtype or a
#                                     PIL.Image object
#         bbox_gt (None, optional): a list or a np.ndarray of dimension (Nx4).
#                                     All elemnts will be onverted to int32
#         bbox_type (str, optional): Description
#         step (None, optional): integer indicating the step at which artifact
#                                     was generated
#     """
#     if bbox_type not in ['pascal_voc', 'coco', 'yolo']:
#         raise MlflowException(
#             f'bbox_type should be one-of "pascal_voc, coco, yolo" not \
#             {bbox_type}')
#
#     path = convert_to_imagefile(image)
#     pred_mask_path = convert_to_imagefile(pred_mask)
#
#     bbox_pred = np.array(bbox_pred)
#     if bbox_pred.size > 0:
#         print(bbox_pred.size)
#         assert isinstance(
#             bbox_pred, np.ndarray) and bbox_pred.ndim == 2 and bbox_pred.shape[
#                 1] == 4, f'bbox_pred should be numpy of dimension (Nx4), got \
#             {bbox_pred.shape}'
#
#     if bbox_type == 'coco':
#         if bbox_pred.size > 0:
#             bbox_pred = coco_to_voc_bbox(bbox_pred)
#         if bbox_gt:
#             bbox_gt = coco_to_voc_bbox(bbox_gt)
#         else:
#             bbox_gt = np.array([])
#     else:
#         if bbox_pred.size > 0:
#             bbox_pred = yolo_to_voc_bbox(image, bbox_pred)
#         if bbox_gt:
#             bbox_gt = yolo_to_voc_bbox(image, bbox_gt)
#         else:
#             bbox_gt = np.array([])
#
#     run = _get_or_start_run()
#     run_id = run.info.run_id
#     experiment_id = run.info.experiment_id
#
#     prediction_struct = Struct()
#     prediction_struct.update({'bbox': bbox_pred.tolist()})
#
#     ground_truth_struct = Struct()
#     ground_truth_struct.update({'bbox': bbox_gt.tolist()})
#
#     MlflowClient().log_artifact_lite(
#         run_id,
#         experiment_id,
#         key,
#         path,
#         prediction=prediction_struct,
#         ground_truth=ground_truth_struct,
#         artifact_type='image',
#         step=step,
#         tags={'image_type': 'segmentation_mask'})
#
#     log_image(
#         key=key+'_mask',
#         image=pred_mask_path,
#         tags={'mask_parent': key},
#         step=step)


def set_tags(tags):
    """Log a batch of tags for the current run. If no run is active, this
    method will create a new active run.

    Args:
        tags (dict): Dictionary of tag_name: String -> value: (String, but
                    will be string-ified if not
    """
    run_id = _get_or_start_run().info.run_id
    tags_arr = [RunTag(key, str(value)) for key, value in tags.items()]
    MlflowClient().log_batch(
        run_id=run_id, metrics=[], params=[], tags=tags_arr)


def _record_logged_model(mlflow_model):
    """Summary.

    Args:
        mlflow_model (TYPE): Description
    """
    run_id = _get_or_start_run().info.run_id
    MlflowClient()._record_logged_model(run_id, mlflow_model)


def get_experiment(experiment_id):
    """Retrieve an experiment by experiment_id from the backend store.

    Args:
        experiment_id (TYPE): The experiment ID returned from
                            ``create_experiment``.

    Returns:
        TYPE: :py:class:`segmind_track.entities.Experiment`
    """
    return MlflowClient().get_experiment(experiment_id)


def get_experiment_by_name(name):
    """Retrieve an experiment by experiment name from the backend store.

    Args:
        name (TYPE): The experiment name.

    Returns:
        TYPE: :py:class:`segmind_track.entities.Experiment`
    """
    return MlflowClient().get_experiment_by_name(name)


def create_experiment(name, artifact_location=None):
    """Create an experiment.

    Args:
        name (TYPE): The experiment name. Must be unique.
        artifact_location (None, optional): The location to store run
                                        artifacts. If not provided,
                                        the server picks an appropriate
                                        default.

    Returns:
        int: Integer ID of the created experiment.
    """
    return MlflowClient().create_experiment(name, artifact_location)


def delete_experiment(experiment_id):
    """Delete an experiment from the backend store.

    Args:
        experiment_id (int): The experiment ID returned from
                            ``create_experiment``.
    """
    MlflowClient().delete_experiment(experiment_id)


def delete_run(run_id):
    """Deletes a run with the given ID.

    Args:
        run_id (TYPE): Unique identifier for the run to delete.
    """
    MlflowClient().delete_run(run_id)


def search_runs(experiment_ids=None,
                filter_string='',
                run_view_type=ViewType.ACTIVE_ONLY,
                max_results=SEARCH_MAX_RESULTS_PANDAS,
                order_by=None):
    """Get a pandas DataFrame of runs that fit the search criteria.

    Args:
        experiment_ids (None, optional): List of experiment IDs. None will
                                    default to the active experiment.
        filter_string (str, optional): Filter query string, defaults to
                                    searching all runs.
        run_view_type (TYPE, optional): one of enum values ``ACTIVE_ONLY``,
                             ``DELETED_ONLY``, or ``ALL`` runs defined
                            in :py:class:`segmind_track.entities.ViewType`.
        max_results (TYPE, optional): The maximum number of runs to put in the
                            dataframe. Default is 100,000 to avoid causing
                            out-of-memory issues on the user's machine.
        order_by (None, optional): List of columns to order by (e.g.,
                        "metrics.rmse"). The ``order_by`` column can contain
                        an optional ``DESC`` or ``ASC`` value. The default is
                        ``ASC``. The default ordering is to sort by
                        ``start_time DESC``, then ``run_id``.

    Returns:
        TYPE: A pandas.DataFrame of runs, where each metric, parameter, and
         tag are expanded into their own columns named metrics.*, params.*,
         and tags.* respectively. For runs that don't have a particular metric
        , parameter, or tag, their value will be (NumPy) Nan, None, or None
         respectively.
    """
    if not experiment_ids:
        experiment_ids = _get_experiment_id()
    runs = _get_paginated_runs(experiment_ids, filter_string, run_view_type,
                               max_results, order_by)
    info = {
        'run_id': [],
        'experiment_id': [],
        'status': [],
        'artifact_uri': [],
        'start_time': [],
        'end_time': []
    }
    params, metrics, tags = ({}, {}, {})
    PARAM_NULL, METRIC_NULL, TAG_NULL = (None, np.nan, None)
    for i, run in enumerate(runs):
        info['run_id'].append(run.info.run_id)
        info['experiment_id'].append(run.info.experiment_id)
        info['status'].append(run.info.status)
        info['artifact_uri'].append(run.info.artifact_uri)
        info['start_time'].append(
            pd.to_datetime(run.info.start_time, unit='ms', utc=True))
        info['end_time'].append(
            pd.to_datetime(run.info.end_time, unit='ms', utc=True))

        # Params
        param_keys = set(params.keys())
        for key in param_keys:
            if key in run.data.params:
                params[key].append(run.data.params[key])
            else:
                params[key].append(PARAM_NULL)
        new_params = set(run.data.params.keys()) - param_keys
        for p in new_params:
            params[p] = [PARAM_NULL
                         ] * i  # Fill in null values for all previous runs
            params[p].append(run.data.params[p])

        # Metrics
        metric_keys = set(metrics.keys())
        for key in metric_keys:
            if key in run.data.metrics:
                metrics[key].append(run.data.metrics[key])
            else:
                metrics[key].append(METRIC_NULL)
        new_metrics = set(run.data.metrics.keys()) - metric_keys
        for m in new_metrics:
            metrics[m] = [METRIC_NULL] * i
            metrics[m].append(run.data.metrics[m])

        # Tags
        tag_keys = set(tags.keys())
        for key in tag_keys:
            if key in run.data.tags:
                tags[key].append(run.data.tags[key])
            else:
                tags[key].append(TAG_NULL)
        new_tags = set(run.data.tags.keys()) - tag_keys
        for t in new_tags:
            tags[t] = [TAG_NULL] * i
            tags[t].append(run.data.tags[t])

    data = {}
    data.update(info)
    for key in metrics:
        data['metrics.' + key] = metrics[key]
    for key in params:
        data['params.' + key] = params[key]
    for key in tags:
        data['tags.' + key] = tags[key]
    return pd.DataFrame(data)


def _get_paginated_runs(experiment_ids, filter_string, run_view_type,
                        max_results, order_by):
    """Summary.

    Args:
        experiment_ids (TYPE): Description
        filter_string (TYPE): Description
        run_view_type (TYPE): Description
        max_results (TYPE): Description
        order_by (TYPE): Description

    Returns:
        TYPE: Description
    """
    all_runs = []
    next_page_token = None
    while (len(all_runs) < max_results):
        runs_to_get = max_results - len(all_runs)
        if runs_to_get < NUM_RUNS_PER_PAGE_PANDAS:
            runs = MlflowClient().search_runs(experiment_ids, filter_string,
                                              run_view_type, runs_to_get,
                                              order_by, next_page_token)
        else:
            runs = MlflowClient().search_runs(experiment_ids, filter_string,
                                              run_view_type,
                                              NUM_RUNS_PER_PAGE_PANDAS,
                                              order_by, next_page_token)
        all_runs.extend(runs)
        if hasattr(runs,
                   'token') and runs.token != '' and runs.token is not None:
            next_page_token = runs.token
        else:
            break
    return all_runs


def _get_experiment_id_from_env():
    """Summary.

    Returns:
        TYPE: Description
    """
    # experiment_name = env.get_env(_EXPERIMENT_NAME_ENV_VAR)
    # if experiment_name is not None:
    #     exp = MlflowClient().get_experiment_by_name(experiment_name)
    #     return exp.experiment_id if exp else None
    return env.get_env(_EXPERIMENT_ID_ENV_VAR)
