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


def create_token_file_guide():
    message = "couldn't locate your credentials, please configure by typing `segmind config` in a terminal"
    cyan_print(message)


def set_access_token_guide():
    message = f'set your access-token as env variable by `export {_ACCESS_TOKEN}=your-access-token` in terminal'
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


def get_token_config():
    config = configparser.ConfigParser()
    if not os.path.isfile(TOKENS_FILE):
        create_token_file_guide()
        cyan_print('Alternatively ..')
        set_access_token_guide()
        sys.exit()

    config.read(TOKENS_FILE)

    return config


def set_tokens_in_env(user_tokens):
    os.environ[_ACCESS_TOKEN] = user_tokens['access_token']
    os.environ[_REFRESH_TOKEN] = user_tokens['refresh_token']


def get_user_info_with_new_token(access_token):
    os.environ[_ACCESS_TOKEN] = access_token
    user_tokens = get_new_tokens(token=access_token)

    # Set New AccessToken & RefreshToken
    set_tokens_in_env(user_tokens)

    # save token in a file
    save_tokens_in_file(user_tokens)

    return user_tokens


def get_new_tokens(token):
    url = SEGMIND_API_URL + "/auth/profile"

    headers = {
        "Authorization": "Bearer " + str(token),
        "Content-Type": "application/json",
    }
    response = requests.get(url, headers=headers)
    response.raise_for_status()
    response_data = response.json()

    return response_data


def save_tokens_in_file(user_tokens):
    # make sure folder and file are present
    folder_path = os.path.join(os.path.expanduser('~'), '.segmind')
    os.makedirs(folder_path, exist_ok=True)

    ftoken = configparser.ConfigParser()
    ftoken['TOKENS'] = {
        'access_token': user_tokens['access_token'],
        'refresh_token': user_tokens['refresh_token'],
    }
    with open(TOKENS_FILE, 'w') as config:
        ftoken.write(config)


# Deprecated
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


def token_has_expired(access_token=None):
    if not access_token:
        # check token in env first.
        if os.environ.get(_ACCESS_TOKEN):
            access_token = os.environ.get(_ACCESS_TOKEN)
        else:
            # check token in token-file
            tokens = get_token_config()
            access_token = tokens['TOKENS']['access_token']

    headers = {'authorization': f'Bearer {access_token}'}
    query = requests.get(
        f'{SEGMIND_API_URL}/auth/authenticate',
        headers=headers
    )
    if query.status_code != 200:
        return True

    return False


def refresh_token(refresh_token=None):
    if not refresh_token:
        # check token in env first.
        if os.environ.get(_REFRESH_TOKEN):
            refresh_token = os.environ.get(_REFRESH_TOKEN)
        else:
            # check token in token-file
            tokens = get_token_config()
            refresh_token = tokens['TOKENS']['refresh_token']

    headers = {'authorization': f'Bearer {refresh_token}'}
    query = requests.post(
        f'{SEGMIND_API_URL}/auth/refresh-token',
        headers=headers
    )
    if query.status_code != 200:
        raise MlflowException(query.json()['msg'])

    access_token = query.json()['access_token']

    # set new token in env
    os.environ[_ACCESS_TOKEN] = access_token

    # save new token in token-file
    save_tokens_in_file({
        'access_token': access_token,
        'refresh_token': refresh_token,
    })

    return access_token


def get_token():
    access_token = os.environ.get(_ACCESS_TOKEN)
    if not access_token:
        # check token in token-file
        tokens = get_token_config()
        try:
            access_token = tokens['TOKENS']['access_token']
        except:
            # token not found, show guides
            create_token_file_guide()
            cyan_print('Alternatively ..')
            set_access_token_guide()
            sys.exit()

        # token found in token-file, update env
        set_tokens_in_env(dict(tokens['TOKENS']))

    # token found, check if expired
    if token_has_expired(access_token=access_token):

        # token is expired, generate new token
        access_token = refresh_token()

    return access_token


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
