import json
import logging
import requests
import time

from segmind.exceptions import (MlflowException, MlflowliteException,
                                RestException)
from segmind.lite_extensions.client_utils import (_get_experiment_id,
                                                  get_host_uri, get_token)
from segmind.protos import errorcodes_pb2
from segmind.utils.proto_json_utils import parse_dict
from segmind.utils.string_utils import strip_suffix
from segmind.version import VERSION as __version__

RESOURCE_DOES_NOT_EXIST = 'RESOURCE_DOES_NOT_EXIST'

_logger = logging.getLogger(__name__)

_DEFAULT_HEADERS = {'User-Agent': 'mlflow-python-client/%s' % __version__}


def http_request(endpoint,
                 retries=3,
                 retry_interval=3,
                 max_rate_limit_interval=60,
                 **kwargs):
    """Makes an HTTP request with the specified method to the specified
    hostname/endpoint. Ratelimit error code (429) will be retried with an
    exponential back off (1, 2, 4, ... seconds) for at most
    `max_rate_limit_interval` seconds.  Internal errors (500s) will be retried
    up to `retries` times.

    , waiting `retry_interval` seconds between successive retries. Parses the
    API response (assumed to be JSON) into a Python object and returns it.

    :param host_creds: A :py:class:`segmind_track.rest_utils.MlflowHostCreds`
            object containing hostname and optional authentication.
    :return: Parsed API response
    """

    hostname = get_host_uri()

    token = get_token()
    if endpoint == '/api/2.0/mlflow/experiments/create':
        experiment_id = None
    elif endpoint == '/api/2.0/mlflow/experiments/get':
        experiment_id = kwargs.get("params").get("experiment_id")
    else:
        experiment_id = _get_experiment_id()

    if 'params' in kwargs:
        kwargs['params'].update({
            'token': token,
            'experiment_id': experiment_id
        })
    elif 'json' in kwargs:
        kwargs['json'].update({'token': token, 'experiment_id': experiment_id})
    else:
        raise MlflowliteException(
            "no 'json' or 'params' field found to send in request")

    headers = dict(_DEFAULT_HEADERS)

    verify = True

    def request_with_ratelimit_retries(max_rate_limit_interval, **kwargs):
        response = requests.request(**kwargs)
        time_left = max_rate_limit_interval
        sleep = 1
        while response.status_code == 429 and time_left > 0:
            _logger.warning(
                'API request to {path} returned status code 429 (Rate limit'
                ' exceeded). Retrying in %d seconds. '
                'Will continue to retry 429s for up to %d seconds.', sleep,
                time_left)
            time.sleep(sleep)
            time_left -= sleep
            response = requests.request(**kwargs)
            sleep = min(time_left,
                        sleep * 2)  # sleep for 1, 2, 4, ... seconds;
        return response

    cleaned_hostname = strip_suffix(hostname, '/')
    url = '%s%s' % (cleaned_hostname, endpoint)
    for i in range(retries):
        response = request_with_ratelimit_retries(
            max_rate_limit_interval,
            url=url,
            headers=headers,
            verify=verify,
            **kwargs)
        if response.status_code >= 200 and response.status_code < 500:
            return response
        else:
            _logger.error(f'API request to {url} failed with code \
                {response.status_code} != 200, retrying up to \
                {retries-i-1} more times. API response body: {response.text}')
            time.sleep(retry_interval)
    raise MlflowException(
        'API request to %s failed to return code 200 after %s tries' %
        (url, retries))


def _can_parse_as_json(string):
    try:
        json.loads(string)
        return True
    except Exception:  # pylint: disable=broad-except
        return False


def http_request_safe(host_creds, endpoint, **kwargs):
    """Wrapper around ``http_request`` that also verifies that the request
    succeeds with code 200."""
    response = http_request(endpoint=endpoint, **kwargs)
    return verify_rest_response(response, endpoint)


def verify_rest_response(response, endpoint):
    """Verify the return code and raise exception if the request was not
    successful."""
    if response.status_code != 200:
        if _can_parse_as_json(response.text):
            raise RestException(json.loads(response.text))
        else:
            base_msg = 'API request to endpoint %s failed with error code ' \
                       '%s != 200' % (endpoint, response.status_code)
            raise MlflowException("%s. Response body: '%s'" %
                                  (base_msg, response.text))
    return response


def _get_path(path_prefix, endpoint_path):
    return '{}{}'.format(path_prefix, endpoint_path)


def extract_api_info_for_service(service, path_prefix):
    """Return a dictionary mapping each API method to a tuple (path, HTTP
    method)"""
    service_methods = service.DESCRIPTOR.methods
    res = {}
    for service_method in service_methods:
        endpoints = service_method.GetOptions().Extensions[
            errorcodes_pb2.rpc].endpoints
        endpoint = endpoints[0]
        endpoint_path = _get_path(path_prefix, endpoint.path)
        res[service().GetRequestClass(service_method)] = (endpoint_path,
                                                          endpoint.method)
    return res


def call_endpoint(endpoint, method, json_body, response_proto):
    # Convert json string to json dictionary, to pass to requests
    if json_body:
        json_body = json.loads(json_body)
    if method == 'GET':
        response = http_request(
            endpoint=endpoint, method=method, params=json_body)
    else:
        response = http_request(
            endpoint=endpoint, method=method, json=json_body)
    response = verify_rest_response(response, endpoint)
    js_dict = json.loads(response.text)
    parse_dict(js_dict=js_dict, message=response_proto)
    return response_proto


class MlflowHostCreds(object):
    """Provides a hostname and optional authentication for talking to an MLflow
    tracking server.

    host: Hostname (e.g., http://localhost:5000) to MLflow server. Required.
    username: Username to use with Basic authentication when talking to
        server. If this is specified, password must also be specified.
    password: Password to use with Basic authentication when talking to
        server. If this is specified, username must also be specified.
    token: Token to use with Bearer authentication when talking to server.
        If provided, user/password authentication will be ignored.
    ignore_tls_verification: If true, we will not verify the server's hostname
        or TLS certificate. This is useful for certain testing situations, but
        should never be true in production.
    """

    def __init__(self,
                 host,
                 username=None,
                 password=None,
                 token=None,
                 ignore_tls_verification=False):
        if not host:
            raise MlflowException(
                'host is a required parameter for MlflowHostCreds')
        self.host = host
        self.username = username
        self.password = password
        self.token = token
        self.ignore_tls_verification = ignore_tls_verification
