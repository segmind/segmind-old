import configparser
import os
import requests
import sys
from functools import wraps
from pathlib import Path

from segmind.exceptions import MlflowException
from segmind.lite_extensions.urls import SEGMIND_API_URL, TRACKING_URI
from segmind.utils import env
from .utils import cyan_print

_EXPERIMENT_ID_ENV_VAR = 'TRACK_EXPERIMENT_ID'
_RUN_ID_ENV_VAR = 'TRACK_RUN_ID'
_ACCESS_TOKEN = 'TRACK_ACCESS_TOKEN'
_REFRESH_TOKEN = 'TRACK_REFRESH_TOKEN'

HOME = Path.home()

SEGMIND_FOLDER = os.path.join(HOME, Path('.segmind'))
SECRET_FILE = os.path.join(HOME, Path('.segmind/secret.file'))
TOKENS_FILE = os.path.join(HOME, Path('.segmind/tokens.file'))

os.makedirs(SEGMIND_FOLDER, exist_ok=True)


def create_secret_file(email, password):
    with open(SECRET_FILE, 'w') as file:
        file.write('[secret]\n')
        file.write('email={}\n'.format(email))
        file.write('password={}\n'.format(password))


def create_secret_file_guide():

    message = "couldn't locate your credentials, please configure by typing `cral config` in a terminal"  # noqa
    cyan_print(message)


def set_access_token_guide():
    message = f'set your access-token as env variable by `export {_ACCESS_TOKEN}=your-access-token` in terminal'  # noqa
    cyan_print(message)


class LoginError(Exception):
    pass


def get_host_uri():
    return TRACKING_URI


def get_secret_config():
    config = configparser.ConfigParser()
    if not os.path.isfile(SECRET_FILE):
        create_secret_file_guide()
        cyan_print('Alternatively ..')
        set_access_token_guide()
        sys.exit()
    config.read(SECRET_FILE)
    return config


def fetch_token(email, password):
    payload = {
        'email': email,
        'password': password,
    }
    query = requests.post(f'{SEGMIND_API_URL}/auth/login', data=payload)

    if query.status_code != 200:
        try:
            error_msg = query.json()['message']
        except KeyError:
            error_msg = "couldn't login, please check email-id"
        raise LoginError(error_msg)

    os.environ[_ACCESS_TOKEN] = query.json()['access_token']
    os.environ[_REFRESH_TOKEN] = query.json()['refresh_token']

    return os.environ.get(_ACCESS_TOKEN)


def token_has_expired():
    # ftoken = configparser.ConfigParser()
    # ftoken.read(TOKENS_FILE)
    headers = {'authorization': f'Bearer {os.environ.get(_ACCESS_TOKEN)}'}
    query = requests.get(
        f'{SEGMIND_API_URL}/auth/authenticate', headers=headers)
    if query.status_code != 200:
        return True
    return False


def refresh_token():
    headers = {'authorization': f'Bearer {os.environ.get(_REFRESH_TOKEN)}'}
    query = requests.post(
        f'{SEGMIND_API_URL}/auth/refresh-token', headers=headers)
    if query.status_code != 200:
        raise MlflowException(query.json()['message'])
    else:
        os.environ[_ACCESS_TOKEN] = query.json()['access_token']


def get_token():
    access_token = os.environ.get(_ACCESS_TOKEN)
    if access_token is not None:
        if token_has_expired():
            refresh_token()
            access_token = os.environ.get(_ACCESS_TOKEN)
        return access_token

    config = get_secret_config()
    email = config['secret']['email']
    password = config['secret']['password']
    return fetch_token(email, password)


def catch_mlflowlite_exception(func):

    @wraps(func)
    def wrapper(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except MlflowException as e:
            raise e

    return wrapper


def _get_experiment_id():
    return env.get_env(_EXPERIMENT_ID_ENV_VAR)


@catch_mlflowlite_exception
def _get_run_id():
    return env.get_env(_RUN_ID_ENV_VAR)


def _runid_exists():
    return _RUN_ID_ENV_VAR in os.environ
