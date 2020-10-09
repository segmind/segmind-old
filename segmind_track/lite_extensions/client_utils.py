import configparser
import os
import requests
import sys
from functools import wraps
from pathlib import Path

from segmind_track.exceptions import MlflowException
from segmind_track.lite_extensions.urls import SEGMIND_API_URL, TRACKING_URI
from segmind_track.utils import env
from .utils import cyan_print

_EXPERIMENT_ID_ENV_VAR = 'TRACK_EXPERIMENT_ID'
_RUN_ID_ENV_VAR = 'TRACK_RUN_ID'

HOME = Path.home()

SECRET_FILE = os.path.join(HOME, Path('.segmind/secret.file'))
TOKENS_FILE = os.path.join(HOME, Path('.segmind/tokens.file'))


def create_secret_file_guide():

    message = "couldn't locate your credentials, please configure by typing \
    `cral config` in a terminal"
    cyan_print(message)


class LoginError(Exception):
    pass


def get_host_uri():
    return TRACKING_URI


def get_secret_config():
    config = configparser.ConfigParser()
    if not os.path.isfile(SECRET_FILE):
        create_secret_file_guide()
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
        raise LoginError(query.json()['message'])

    ftoken = configparser.ConfigParser()
    ftoken['TOKENS'] = {
        'access_token': query.json()['access_token'],
        'refresh_token': query.json()['refresh_token'],
    }
    with open(TOKENS_FILE, 'w') as config:
        ftoken.write(config)

    return ftoken['TOKENS']['access_token']


def expired_token():
    ftoken = configparser.ConfigParser()
    ftoken.read(TOKENS_FILE)
    headers = {'authorization': f"Bearer {ftoken['TOKENS']['access_token']}"}
    query = requests.get(
        f'{SEGMIND_API_URL}/auth/authenticate', headers=headers)
    if query.status_code != 200:
        return True
    return False


def get_token():
    config = get_secret_config()
    email = config['secret']['email']
    password = config['secret']['password']

    if not os.path.exists(TOKENS_FILE) or expired_token():
        return fetch_token(email, password)

    ftoken = configparser.ConfigParser()
    ftoken.read(TOKENS_FILE)
    return ftoken['TOKENS']['access_token']


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
