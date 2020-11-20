from segmind.entities._mlflow_object import _MLflowObject
from segmind.protos.service_lite_pb2 import Metric as ProtoMetric


class Metric(_MLflowObject):
    """Metric object."""

    def __init__(self, key, value, timestamp, step, epoch, tags=()):
        self._key = key
        self._value = value
        self._timestamp = timestamp
        self._step = step
        self._epoch = epoch
        self._tags = tags

    @property
    def key(self):
        """String key corresponding to the metric name."""
        return self._key

    @property
    def value(self):
        """Float value of the metric."""
        return self._value

    @property
    def timestamp(self):
        """Metric timestamp as an integer (milliseconds since the Unix
        epoch)."""
        return self._timestamp

    @property
    def step(self):
        """Integer metric step (x-coordinate)."""
        return self._step

    @property
    def epoch(self):
        """Integer metric epoch (x-coordinate)."""
        return self._epoch

    @property
    def tags(self):
        """Integer metric step (x-coordinate)."""
        return self._tags

    def to_proto(self):
        metric = ProtoMetric()
        metric.key = self.key
        metric.value = self.value
        metric.timestamp = self.timestamp
        metric.step = self.step
        metric.epoch = self.epoch
        metric.tags.extend(self.tags)
        return metric

    @classmethod
    def from_proto(cls, proto):
        return cls(
            proto.key,
            proto.value,
            proto.timestamp,
            proto.step,
            proto.epoch,
            proto.tags)
