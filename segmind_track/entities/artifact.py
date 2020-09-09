
from segmind_track.entities._mlflow_object import _MLflowObject
from segmind_track.protos.service_lite_pb2 import Artifact as ProtoArtifact


class Artifact(_MLflowObject):
    """Artifact object."""

    def __init__(self, key, path):
        self._key = key
        self._path = path

    @property
    def key(self):
        """String key corresponding to the artifact name."""
        return self._key

    @property
    def path(self):
        """String local path of the artifact."""
        return self._path

    def to_proto(self):
        artifact = ProtoArtifact()
        artifact.key = self.key
        artifact.path = self.path
        return artifact

    @classmethod
    def from_proto(cls, proto):
        return cls(proto.key, proto.path)
