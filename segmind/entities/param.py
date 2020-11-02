# import sys

from segmind.entities._mlflow_object import _MLflowObject
from segmind.protos.service_lite_pb2 import Param as ProtoParam


class Param(_MLflowObject):
    """Parameter object."""

    def __init__(self, key, value, tags=()):
        # if 'pyspark.ml' in sys.modules:
        #     import pyspark.ml.param
        #     if isinstance(key, pyspark.ml.param.Param):
        #         key = key.name
        #         value = str(value)
        self._key = key
        self._value = value
        self._tags = tags

    @property
    def key(self):
        """String key corresponding to the parameter name."""
        return self._key

    @property
    def value(self):
        """String value of the parameter."""
        return self._value

    @property
    def tags(self):
        """String value of the parameter."""
        return self._tags

    def to_proto(self):
        param = ProtoParam()
        param.key = self.key
        param.value = self.value
        param.tags.extend(self.tags)
        return param

    @classmethod
    def from_proto(cls, proto):
        return cls(proto.key, proto.value, proto.tags)
