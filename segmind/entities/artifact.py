from segmind.entities._mlflow_object import _MLflowObject
from segmind.protos.service_lite_pb2 import Artifact as ProtoArtifact
from segmind.protos.service_lite_pb2 import ArtifactTag as ProtoArtifactTag


class Artifact(_MLflowObject):
    """Artifact object."""

    def __init__(self,
                 key,
                 path,
                 artifact_type,
                 timestamp=None,
                 size=None,
                 prediction=None,
                 ground_truth=None,
                 step=None,
                 tags=[]):
        self._key = key
        self._path = path
        self._artifact_type = artifact_type
        self._timestamp = timestamp
        self._size = size
        self._prediction = prediction
        self._ground_truth = ground_truth
        self._step = step
        self._tags = {tag.key: tag.value for tag in (tags or [])}

    @property
    def key(self):
        """String key corresponding to the artifact name."""
        return self._key

    @property
    def path(self):
        """String local path of the artifact."""
        return self._path

    @property
    def artifact_type(self):
        """String value of the artifact type."""
        return self._artifact_type

    @property
    def timestamp(self):
        """Artifact timestamp as an integer (milliseconds since the Unix
        epoch)."""
        return self._timestamp

    @property
    def size(self):
        """Integer value of the artifact size."""
        return self._size

    @property
    def prediction(self):
        """JSON value of the artifact prediction."""
        if not self._prediction:
            return

        return self._prediction.to_dict()

    @property
    def ground_truth(self):
        """JSON value of the artifact ground truth."""
        if not self._ground_truth:
            return

        return self._ground_truth.to_dict()

    @property
    def step(self):
        """Integer value of the artifact step."""
        return self._step

    @property
    def tags(self):
        """The artifact tags, such as type.

        :rtype: :py:class:`mlflow_lite.entities.ArtifactTag`
        """
        return self._tags

    def _add_tag(self, tag):
        self.tags[tag.key] = tag.value

    def to_proto(self):
        artifact = ProtoArtifact()
        artifact.key = self.key
        artifact.path = self.path
        artifact.artifact_type = self.artifact_type
        if self.timestamp:
            artifact.timestamp = self.timestamp
        if self.size:
            artifact.size = self.size
        if self.prediction:
            artifact.prediction = self.prediction
        if self.ground_truth:
            artifact.ground_truth = self.ground_truth
        if self.step:
            artifact.step = self.step
        artifact.tags.extend([
            ProtoArtifactTag(key=key, value=val)
            for key, val in self.tags.items()
        ])

        return artifact

    @classmethod
    def from_proto(cls, proto):
        artifact = cls(proto.name, proto.path, proto.artifact_type,
                       proto.timestamp, proto.size, proto.prediction,
                       proto.ground_truth, proto.step)
        for proto_tag in proto.tags:
            artifact._add_tag(ProtoArtifactTag.from_proto(proto_tag))
        return artifact
