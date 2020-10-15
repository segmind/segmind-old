import json

from segmind.protos.errorcodes_pb2 import (BAD_REQUEST, ENDPOINT_NOT_FOUND,
                                           INTERNAL_ERROR,
                                           INVALID_PARAMETER_VALUE,
                                           INVALID_STATE, PERMISSION_DENIED,
                                           REQUEST_LIMIT_EXCEEDED,
                                           RESOURCE_ALREADY_EXISTS,
                                           RESOURCE_DOES_NOT_EXIST,
                                           TEMPORARILY_UNAVAILABLE, ErrorCode)

ERROR_CODE_TO_HTTP_STATUS = {
    ErrorCode.Name(INTERNAL_ERROR): 500,
    ErrorCode.Name(INVALID_STATE): 500,
    ErrorCode.Name(TEMPORARILY_UNAVAILABLE): 503,
    ErrorCode.Name(REQUEST_LIMIT_EXCEEDED): 429,
    ErrorCode.Name(ENDPOINT_NOT_FOUND): 404,
    ErrorCode.Name(RESOURCE_DOES_NOT_EXIST): 404,
    ErrorCode.Name(PERMISSION_DENIED): 403,
    ErrorCode.Name(BAD_REQUEST): 400,
    ErrorCode.Name(RESOURCE_ALREADY_EXISTS): 400,
    ErrorCode.Name(INVALID_PARAMETER_VALUE): 400
}


class MlflowException(Exception):
    """Generic exception thrown to surface failure information about external-
    facing operations.

    The error message associated with this exception may be exposed to clients
    in HTTP responses for debugging purposes. If the error text is sensitive,
    raise a generic `Exception` object instead.
    """

    def __init__(self, message, error_code=INTERNAL_ERROR, **kwargs):
        """
        Args:
            message: The message describing the error that occured. This will
            be included in the exception's serialized JSON representation.
            error_code: An appropriate error code for the error that occured;
            it will be included in the exception's serialized JSON
            representation. This should be one of the codes listed in the
            `cral.tracking.protos.databricks_pb2` proto.
            kwargs: Additional key-value pairs to include in the serialized
            JSON representation of the MlflowException.
        """
        try:
            self.error_code = ErrorCode.Name(error_code)
        except (ValueError, TypeError):
            self.error_code = ErrorCode.Name(INTERNAL_ERROR)
        self.message = message
        self.json_kwargs = kwargs
        super(MlflowException, self).__init__(message)

    def serialize_as_json(self):
        exception_dict = {
            'error_code': self.error_code,
            'message': self.message
        }
        exception_dict.update(self.json_kwargs)
        return json.dumps(exception_dict)

    def get_http_status_code(self):
        return ERROR_CODE_TO_HTTP_STATUS.get(self.error_code, 500)


class RestException(MlflowException):
    """Exception thrown on non 200-level responses from the REST API."""

    def __init__(self, json):
        error_code = json.get('error_code', ErrorCode.Name(INTERNAL_ERROR))
        message = '%s: %s' % (error_code, json['message'] if 'message' in json
                              else 'Response: ' + str(json))
        super(RestException, self).__init__(
            message, error_code=ErrorCode.Value(error_code))
        self.json = json


class ExecutionException(MlflowException):
    """Exception thrown when executing a project fails."""


class MlflowliteException(Exception):
    """docstring for MlflowliteException."""

    def __init__(self, message, error_code=INTERNAL_ERROR):
        """
        Args:
        message: The message describing the error that occured. This will be
        included in the exception's serialized JSON representation.
        error_code: An appropriate error code for the error that occured; it
        will be included in the exception's serialized JSON representation.
        This should be one of the codes listed in the
        `segmind.tracking.protos.databricks_pb2` proto.
        kwargs: Additional key-value pairs to include in the serialized JSON
        representation of the MlflowException.
        """
        try:
            self.error_code = ErrorCode.Name(error_code)
        except (ValueError, TypeError):
            self.error_code = ErrorCode.Name(INTERNAL_ERROR)
        message = '{}: {}'.format(self.error_code, message)
        # self.json_kwargs = kwargs
        super(MlflowliteException, self).__init__(message)


class MissingConfigException(MlflowException):
    """Exception thrown when expected configuration file/directory not
    found."""
