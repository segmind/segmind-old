import os

from segmind.exceptions import MlflowliteException
from segmind.protos.errorcodes_pb2 import INVALID_PARAMETER_VALUE


def get_env(variable_name):
    if variable_name not in os.environ:
        raise MlflowliteException('{} not set'.format(variable_name),
                                  INVALID_PARAMETER_VALUE)
    return os.environ.get(variable_name)


def unset_variable(variable_name):
    if variable_name in os.environ:
        del os.environ[variable_name]
