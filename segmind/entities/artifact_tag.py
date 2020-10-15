from mlflow_lite.entities._mlflow_object import _MLflowObject
from mlflow_lite.protos.service_lite_pb2 import ArtifactTag as ProtoArtifactTag


class ArtifactTag(_MLflowObject):
    """ArtifactTag object."""

    def __init__(self, key, value):
        self._key = key
        self._value = value

    def __eq__(self, other):
        if type(other) is type(self):
            return self.__dict__ == other.__dict__
        return False

    @property
    def key(self):
        """String name of the tag."""
        return self._key

    @property
    def value(self):
        """String value of the tag."""
        return self._value

    def to_proto(self):
        artifact_tag = ProtoArtifactTag()
        artifact_tag.key = self.key
        artifact_tag.value = self.value
        return artifact_tag

    @classmethod
    def from_proto(cls, proto):
        return cls(proto.key, proto.value)
