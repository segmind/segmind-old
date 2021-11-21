import getpass
import os
import sys
import time

import click

from segmind.data.public import upload as upload_data
from segmind.lite_extensions.client_utils import (SECRET_FILE, SEGMIND_FOLDER,
                                                  LoginError, fetch_token)
from segmind.utils import cyan_print, green_print, red_print


# import jsonpickle
# from click import UsageError


@click.group()
def cli():
    pass


@cli.command()
def config():
    cyan_print('Please enter your credentials for Segmind:')

    email = input('Enter Email-id :: ')
    password = getpass.getpass('Enter Password :: ')

    try:
        fetch_token(email, password)
    except LoginError:
        red_print('Log-In failed !!! Invalid credentials')
        sys.exit()

    # folder_path = os.path.join(os.path.expanduser('~'), '.segmind')
    os.makedirs(SEGMIND_FOLDER, exist_ok=True)

    file_path = SECRET_FILE

    with open(file_path, 'w') as file:
        file.write('[secret]\n')
        file.write('email={}\n'.format(email))
        file.write('password={}\n'.format(password))

    green_print('Log-In Successful !!!')


@cli.command()
@click.option("-p", "--path",
              required=True,
              type=click.Path(exists=True),
              help="String, path to the file/folder")
@click.option("--destination_path",
              required=False,
              type=str,
              help="String, example README.md OR new_folder/README.md OR old_folder/new_folder/README.md")
@click.option("--datastore_name",
              required=True,
              type=str,
              help="String, name of your datastore, created via Segmind UI from https://cloud.segmind.com/")
def upload(path, destination_path, datastore_name, *args, **kwargs):
    upload_data(
        path=path,
        destination_path=destination_path,
        datastore_name=datastore_name,
        via_cli=True
    )
    green_print(
        "Success! To view your file/folder, login to your account in https://cloud.segmind.com"
    )
