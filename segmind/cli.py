import getpass
import os
import sys
import time

import click

from segmind.data.public import upload as upload_data
from segmind.utils import cyan_print, green_print, red_print
from segmind.lite_extensions.client_utils import get_user_info_with_new_token
from segmind.version import VERSION


@click.group()
def cli():
    pass


@cli.command()
@click.option(
    "-a",
    "--access_token",
    required=False,
    type=click.STRING,
    help="String, Segmind Access Token",
)
def config(access_token):
    cyan_print("Please enter your credentials for Segmind:")

    if not access_token:
        access_token = input("Enter Access token: ")

    try:
        user_info = get_user_info_with_new_token(access_token=access_token)
    except Exception as err:
        red_print(f"Log-In failed !!! Error: {err}")
        sys.exit()

    green_print(f"Welcome {user_info['username']}, Log-In Successful !!!")


@cli.command()
@click.option(
    "-p",
    "--path",
    required=True,
    type=click.Path(exists=True),
    help="String, path to the file/folder",
)
@click.option(
    "--destination_path",
    required=False,
    type=str,
    help="String, example README.md OR new_folder/README.md OR old_folder/new_folder/README.md",
)
@click.option(
    "--datastore_name",
    required=True,
    type=str,
    help="String, name of your datastore, created via Segmind UI from https://cloud.segmind.com/",
)
def upload(path, destination_path, datastore_name, *args, **kwargs):
    upload_data(
        path=path,
        destination_path=destination_path,
        datastore_name=datastore_name,
        via_cli=True,
    )
    green_print(
        "Success! To view your file/folder, login to your account in https://cloud.segmind.com"
    )


@cli.command()
def version(*args, **kwargs):
    green_print(f"Segmind {VERSION}")
