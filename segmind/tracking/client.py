"""Internal package providing a Python CRUD interface to MLflow experiments,
runs, registered models, and model versions.

This is a lower level API than the :py:mod:`segmind_track.tracking.fluent`
module, and is exposed in the :py:mod:`segmind_track.tracking` module.
"""

import os
import time

from segmind.entities import ViewType
from segmind.lite_extensions.client_utils import get_host_uri
from segmind.protos.service_lite_pb2 import (Artifact, ArtifactTag, LogMetric,
                                             LogParam)
from segmind.store.tracking import SEARCH_MAX_RESULTS_DEFAULT
from segmind.store.tracking.rest_store import RestStore

tracking_uri = get_host_uri()
# tracking_uri = 'http://0.0.0.0:5000'


class MlflowClient(object):
    """Client of an MLflow Tracking Server that creates and manages experiments
    and runs, and of an MLflow Registry Server that creates and manages
    registered models and model versions.

    It's a thin wrapper around TrackingServiceClient and RegistryClient so
    there is a unified API but we can keep the implementation of the tracking
    and registry clients independent from each other.
    """

    def __init__(self, tracking_uri=None, registry_uri=None):
        """
        :Args
        tracking_uri: Address of local or remote tracking server. If not
                    provided, defaults to the service set by
                    ``segmind_track.tracking.set_tracking_uri``. See
        registry_uri: Address of local or remote model registry server. If not
                    provided, defaults to the service set by
                    ``segmind_track.tracking.set_tracking_uri``.
        """
        final_tracking_uri = tracking_uri
        self._registry_uri = final_tracking_uri
        # self._tracking_client = TrackingServiceClient(final_tracking_uri)
        self._tracking_client = RestStore()

    def get_run(self, run_id):
        """
        Fetch the run from backend store. The resulting
        :py:class:`Run <segmind_track.entities.Run>` contains a collection of
        run metadata -- :py:class:`RunInfo <segmind_track.entities.RunInfo>`,
        as well as a collection of run parameters, tags, and metrics --
        :py:class:`RunData <segmind_track.entities.RunData>`. In the case
        where multiple metrics with the same key are logged for the run, the
        :py:class:`RunData <segmind_track.entities.RunData>` contains the most
        recently logged value at the largest step for each metric.

        :param run_id: Unique identifier for the run.

        :return: A single :py:class:`segmind_track.entities.Run` object, if
                the run exists. Otherwise, raises an exception.
        """
        return self._tracking_client.get_run(run_id)

    def get_metric_history(self, run_id, key):
        """Return a list of metric objects corresponding to all values logged
        for a given metric.

        :param run_id: Unique identifier for run
        :param key: Metric name within the run

        :return: A list of :py:class:`segmind_track.entities.Metric` entities
        if logged, else empty list
        """
        return self._tracking_client.get_metric_history(run_id, key)

    def create_run(self, experiment_id, start_time=None, tags=None):
        """Create a :py:class:`segmind_track.entities.Run` object that can be
        associated with metrics, parameters, artifacts, etc. Unlike
        :py:func:`segmind_track.projects.run`, creates objects but does not run
        code. Unlike :py:func:`segmind_track.start_run`, does not change the
        "active run" used by :py:func:`segmind_track.log_param`.

        :param experiment_id: The ID of then experiment to create a run in.
        :param start_time: If not provided, use the current timestamp.
        :param tags: A dictionary of key-value pairs that are converted into
                     :py:class:`segmind_track.entities.RunTag` objects.
        :return: :py:class:`segmind_track.entities.Run` that was created.
        """
        return self._tracking_client.create_run(
            experiment_id=experiment_id, start_time=start_time, tags=tags)

    def list_run_infos(self,
                       experiment_id,
                       run_view_type=ViewType.ACTIVE_ONLY):
        """:return: List of :py:class:`segmind_track.entities.RunInfo`"""
        return self._tracking_client.list_run_infos(experiment_id,
                                                    run_view_type)

    def list_experiments(self, view_type=None):
        """
        :return: List of :py:class:`segmind_track.entities.Experiment`
        """
        return self._tracking_client.list_experiments(view_type)

    def get_experiment(self, experiment_id):
        """Retrieve an experiment by experiment_id from the backend store.

        :Args
        experiment_id: The experiment ID returned from``create_experiment``.
        :return: :py:class:`segmind_track.entities.Experiment`
        """
        return self._tracking_client.get_experiment(experiment_id)

    # TO DO: @pk00095
    # Remove this
    def get_experiment_by_name(self, name):
        """Retrieve an experiment by experiment name from the backend store.

        :param name: The experiment name.
        :return: :py:class:`segmind_track.entities.Experiment`
        """
        return self._tracking_client.get_experiment_by_name(name)

    def create_experiment(self, name, artifact_location=None):
        """Create an experiment.

        :Args
        name: The experiment name. Must be unique.
        artifact_location: The location to store run artifacts.
                          If not provided, the server picks an appropriate
                          default.
        :return: Integer ID of the created experiment.
        """
        return self._tracking_client.create_experiment(name, artifact_location)

    def delete_experiment(self, experiment_id):

        # TO DO: @pk00095 # Remove this
        """Delete an experiment from the backend store.

        :Args
        experiment_id: The experiment ID returned from
                        ``create_experiment``.
        """
        self._tracking_client.delete_experiment(experiment_id)

    def restore_experiment(self, experiment_id):
        # TO DO: @pk00095 # Remove this
        """Restore a deleted experiment unless permanently deleted.

        :Args
        experiment_id: The experiment ID returned from ``create_experiment``.
        """
        self._tracking_client.restore_experiment(experiment_id)

    def rename_experiment(self, experiment_id, new_name):
        # TO DO: @pk00095 # Remove this
        """Update an experiment's name. The new name must be unique.

        :Args
        experiment_id: The experiment ID returned from ``create_experiment``.
        """
        self._tracking_client.rename_experiment(experiment_id, new_name)

    def log_metric(
            self,
            run_id,
            key,
            value,
            timestamp=None,
            step=None,
            epoch=None,
            tags=()):
        """Log a metric against the run ID.

        :Args
        run_id: The run id to which the metric should be logged.
        key: Metric name.
        value: Metric value (float). Note that some special values such
              as +/- Infinity may be replaced by other values depending on the
              store. For example, the SQLAlchemy store replaces +/- Inf with
              max / min float values.
        timestamp: Time when this metric was calculated. Defaults to the
                    current system time.
        step: Training step (iteration) at which was the metric calculated.
            Defaults to 0.
        """
        metric = LogMetric(
            key=str(key),
            value=value,
            timestamp=timestamp,
            step=step,
            epoch=epoch,
            tags=tags)
        self._tracking_client.log_metric(run_id, metric)

    def log_param(self, run_id, key, value, tags=()):
        """Log a parameter against the run ID.

        Value is converted to a string.
        """
        param = LogParam(
            key=str(key),
            value=str(value),
            tags=tags)
        self._tracking_client.log_param(run_id, param)

    def log_artifact_lite(self,
                          run_id,
                          experiment_id,
                          key,
                          path,
                          artifact_type='data',
                          timestamp=None,
                          size=None,
                          prediction=None,
                          ground_truth=None,
                          step=None,
                          tags={}):
        """Log an artifact against the run ID.

        :Args
        run_id: The run ID against which the artifact should be logged.
        experiment_id: The experiment ID against the run ID.
        key: Artifact name.
        path: Artifact path. This value should be the absolute path of the
            local artifact.
        """
        timestamp = timestamp or int(time.time() * 1000)
        size = os.path.getsize(path)

        tags_list = []
        for key, value in tags.items():
            tag = ArtifactTag(key=str(key), value=str(value))
            tags_list.append(tag)
        artifact = Artifact(
            key=key,
            path=path,
            artifact_type=artifact_type,
            timestamp=timestamp,
            size=size,
            prediction=prediction,
            ground_truth=ground_truth,
            step=step,
            tags=tags_list)
        self._tracking_client.log_artifact(run_id, experiment_id, artifact)

    def set_project_tag(self, experiment_id, key, value):
        """Set a tag on the experiment with the specified ID. Value is
        converted to a string.

        :param experiment_id: String ID of the experiment.
        :param key: Name of the tag.
        :param value: Tag value (converted to a string).
        """

        self._tracking_client.set_project_tag(experiment_id, key, value)

    def set_tag(self, run_id, key, value):
        """Set a tag on the run with the specified ID. Value is converted to a
        string.

        :param run_id: String ID of the run.
        :param key: Name of the tag.
        :param value: Tag value (converted to a string)
        """
        self._tracking_client.set_tag(run_id, key, value)

    def delete_tag(self, run_id, key):
        """Delete a tag from a run. This is irreversible.

        :param run_id: String ID of the run
        :param key: Name of the tag
        """
        self._tracking_client.delete_tag(run_id, key)

    def log_batch(self, run_id, metrics=(), params=(), tags=()):
        """Log multiple metrics, params, and/or tags.

        :Args
        run_id: String ID of the run
        metrics: If provided, List of Metric(key, value, timestamp) instances.
        params: If provided, List of Param(key, value) instances.
        tags: If provided, List of RunTag(key, value) instances.

        Raises an MlflowException if any errors occur.
        :return: None
        """
        self._tracking_client.log_batch(run_id, metrics, params, tags)

    def log_artifact(self, run_id, local_path, artifact_path=None):
        """Write a local file or directory to the remote ``artifact_uri``.

        :Args
        local_path: Path to the file or directory to write.
        artifact_path: If provided, the directory in ``artifact_uri`` to
                    write to.
        """
        self._tracking_client.log_artifact(run_id, local_path, artifact_path)

    def log_artifacts(self, run_id, local_dir, artifact_path=None):
        """Write a directory of files to the remote ``artifact_uri``.

        :Args
        local_dir: Path to the directory of files to write.
        artifact_path: If provided, the directory in ``artifact_uri`` to write
                        to.
        """
        self._tracking_client.log_artifacts(run_id, local_dir, artifact_path)

    def _record_logged_model(self, run_id, mlflow_model):
        """Record logged model info with the tracking server.

        :Args
        run_id: run_id under which the model has been logged.
        mlflow_model: Model info to be recorded.
        """
        self._tracking_client._record_logged_model(run_id, mlflow_model)

    def list_artifacts(self, run_id, path=None):
        """List the artifacts for a run.

        :Args
        run_id: The run to list artifacts from.
        path: The run's relative artifact path to list from. By default it is
            set to None or the root artifact path.
        :return: List of :py:class:`segmind_track.entities.FileInfo`
        """
        return self._tracking_client.list_artifacts(run_id, path)

    def download_artifacts(self, run_id, path, dst_path=None):
        """Download an artifact file or directory from a run to a local
        directory if applicable, and return a local path for it.

        :Args
        run_id: The run to download artifacts from.
        path: Relative source path to the desired artifact.
        dst_path: Absolute path of the local filesystem destination directory
                to which to download the specified artifacts. This directory
                must already exist. If unspecified, the artifacts will either
                be downloaded to a new uniquely-named directory on the local
                filesystem or will be returned directly in the case of the
                LocalArtifactRepository.
        :return: Local path of desired artifact.
        """
        return self._tracking_client.download_artifacts(run_id, path, dst_path)

    def set_terminated(self, run_id, status=None, end_time=None):
        """Set a run's status to terminated.

        :Args
        status: A string value of :py:class:`segmind_track.entities.RunStatus`
               Defaults to "FINISHED".
        end_time: If not provided, defaults to the current time.
        """
        self._tracking_client.set_terminated(run_id, status, end_time)

    def delete_run(self, run_id):
        """Deletes a run with the given ID."""
        self._tracking_client.delete_run(run_id)

    def restore_run(self, run_id):
        """Restores a deleted run with the given ID."""
        self._tracking_client.restore_run(run_id)

    def search_runs(self,
                    experiment_ids,
                    filter_string='',
                    run_view_type=ViewType.ACTIVE_ONLY,
                    max_results=SEARCH_MAX_RESULTS_DEFAULT,
                    order_by=None,
                    page_token=None):
        """Search experiments that fit the search criteria.

        :Args
        experiment_ids: List of experiment IDs, or a single int or string id.
        filter_string: Filter query string, defaults to searching all runs.
        run_view_type: one of enum values ACTIVE_ONLY, DELETED_ONLY, or ALL
                    runs defined in :py:class:`segmind_track.entities.ViewType`. # noqa
        max_results: Maximum number of runs desired.
        order_by: List of columns to order by (e.g., "metrics.rmse"). The
                ``order_by`` column can contain an optional ``DESC`` or
                ``ASC`` value. The default is ``ASC``. The default ordering is
                to sort by ``start_time DESC``, then ``run_id``.
        page_token: Token specifying the next page of results. It should be
                obtained from a ``search_runs`` call.

        :return: A list of :py:class:`segmind_track.entities.Run` objects that
                satisfy the search expressions. If the underlying tracking
                store supports pagination, the token for the next page may be
                obtained via the ``token`` attribute of the returned object.
        """
        return self._tracking_client.search_runs(experiment_ids, filter_string,
                                                 run_view_type, max_results,
                                                 order_by, page_token)

    def create_registered_model(self, name):
        """Create a new registered model in backend store.

        :Args
        name: Name of the new model. This is expected to be unique in the
            backend store.
        :return: A single object of :py:class:`segmind_track.entities.model_registry.RegisteredModel` # noqa
                 created by backend.
        """
        return self._get_registry_client().create_registered_model(name)

    def update_registered_model(self, name, new_name=None, description=None):
        """Updates metadata for RegisteredModel entity. Either ``new_name`` or
        ``description`` should be non-None. Backend raises exception if a
        registered model with given name does not exist.

        :Args
        name: Name of the registered model to update.
        new_name: (Optional) New proposed name for the registered model.
        description: (Optional) New description.
        :return: A single updated :py:class:`segmind_track.entities.model_registry.RegisteredModel` object. # noqa
        """
        return self._get_registry_client().update_registered_model(
            name, new_name, description)

    def delete_registered_model(self, name):
        """Delete registered model. Backend raises exception if a registered
        model with given name does not exist.

        :param name: Name of the registered model to update.
        """
        self._get_registry_client().delete_registered_model(name)

    def list_registered_models(self):
        """List of all registered models.

        :return: List of :py:class:`segmind_track.entities.model_registry.RegisteredModel` objects. # noqa
        """
        return self._get_registry_client().list_registered_models()

    def get_registered_model_details(self, name):
        """

        :Args
        name: Name of the registered model to update.
        :return: A single :py:class:`segmind_track.entities.model_registry.RegisteredModelDetailed` object. # noqa
        """
        return self._get_registry_client().get_registered_model_details(name)

    def get_latest_versions(self, name, stages=None):
        """Latest version models for each requests stage. If no ``stages``
        provided, returns the latest version for each stage.

        :param name: Name of the registered model to update.
        :param stages: List of desired stages. If input list is None, return latest versions for
                       for ALL_STAGES.
        :return: List of `:py:class:`segmind_track.entities.model_registry.ModelVersionDetailed` objects. # noqa
        """
        return self._get_registry_client().get_latest_versions(name, stages)

    def create_model_version(self, name, source, run_id):
        """Create a new model version from given source or run ID.

        :param name: Name ID for containing registered model.
        :param source: Source path where the MLflow model is stored.
        :param run_id: Run ID from MLflow tracking server that generated the model
        :return: Single :py:class:`segmind_track.entities.model_registry.ModelVersion` object created by # noqa
                 backend.
        """
        return self._get_registry_client().create_model_version(
            name, source, run_id)

    def update_model_version(self,
                             name,
                             version,
                             stage=None,
                             description=None):
        """Update metadata associated with a model version in backend.

        :param name: Name of the containing registered model.
        :param version: Version number of the model version.
        :param stage: New desired stage for this model version.
        :param description: New description.
        """
        self._get_registry_client().update_model_version(
            name, version, stage, description)

    # @experimental
    def delete_model_version(self, name, version):
        """Delete model version in backend.

        :param name: Name of the containing registered model.
        :param version: Version number of the model version.
        """
        self._get_registry_client().delete_model_version(name, version)

    def get_model_version_details(self, name, version):
        """
        :param name: Name of the containing registered model.
        :param version: Version number of the model version.
        :return: A single :py:class:`segmind_track.entities.model_registry.ModelVersionDetailed` object. # noqa
        """
        return self._get_registry_client().get_model_version_details(
            name, version)

    def get_model_version_download_uri(self, name, version):
        """Get the download location in Model Registry for this model version.

        :param name: Name of the containing registered model.
        :param version: Version number of the model version.
        :return: A single URI location that allows reads for downloading.
        """
        return self._get_registry_client().get_model_version_download_uri(
            name, version)

    def search_model_versions(self, filter_string):
        """Search for model versions in backend that satisfy the filter
        criteria.

        :Args
        filter_string: A filter string expression. Currently supports a single
                    filter condition either name of model like
                    ``name = 'model_name'`` or ``run_id = '...'``.
        :return: PagedList of :py:class:`segmind_track.entities.model_registry.ModelVersion` objects. # noqa
        """
        return self._get_registry_client().search_model_versions(filter_string)

    def get_model_version_stages(self, name, version):
        """
        :return: A list of valid stages.
        """
        return self._get_registry_client().get_model_version_stages(
            name, version)
