# import configparser
import os

_SEGMIND_FOLDER = os.path.join(os.path.expanduser('~'), '.segmind')
os.makedirs(_SEGMIND_FOLDER, exist_ok=True)

_PROJECTS_FOLDER = os.path.join(
    os.path.expanduser('~'), '.segmind', 'projects')
os.makedirs(_PROJECTS_FOLDER, exist_ok=True)


def red_print(text):
    print('\033[31m{}\033[0m'.format(text))


def cyan_print(text):
    print('\033[36m{}\033[0m'.format(text))


def green_print(text):
    print('\033[32m{}\033[0m'.format(text))


def yellow_print(text):
    print('\033[33m{}\033[0m'.format(text))


def blue_print(text):
    print('\033[34m{}\033[0m'.format(text))


def create_new_project(project_name):

    # assert os.path.isdir(location),f"couldnot locate the folder path\n{location}" # noqa: E501
    # filepath = os.path.join(location, project_name)

    if os.path.isfile(project_name):
        raise ValueError('file already exists {}'.format(project_name))

    with open(project_name, 'w') as f:  # noqa: F841
        pass
