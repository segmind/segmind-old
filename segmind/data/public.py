import json
import os

import boto3
import requests
from tqdm import tqdm

from segmind.lite_extensions.client_utils import get_token
from segmind.lite_extensions.urls import SEGMIND_API_URL, SEGMIND_SPOT_URL
from segmind.utils import red_print, green_print


def get_s3_session(key, secret):
    # You can create a session:
    session = boto3.Session(aws_access_key_id=key, aws_secret_access_key=secret)

    # Let's use Amazon S3
    s3 = session.client("s3")

    return s3


def _get_upload_credentials(token, datastore_name):
    headers = {
        "Authorization": "Bearer " + str(token),
        "Content-Type": "application/json",
    }
    url = SEGMIND_API_URL + "/clusters/s3-creds"
    response = requests.get(url, headers=headers)
    response_data = response.json()
    response.raise_for_status()
    s3 = get_s3_session(
        key=response_data.get("access_key"), secret=response_data.get("secret_key")
    )

    payload = {"name": datastore_name}
    url = SEGMIND_SPOT_URL + "/datastore/details"
    response = requests.get(url, headers=headers, data=json.dumps(payload))
    response_data = response.json()
    response.raise_for_status()
    bucket_name = response_data.get("s3_bucket_name")
    folder_name = response_data.get("folder_name")

    return s3, bucket_name, folder_name


def _get_credentials_str(token, datastore_name):
    headers = {
        "Authorization": "Bearer " + str(token),
        "Content-Type": "application/json",
    }
    url = SEGMIND_API_URL + "/clusters/s3-creds"
    response = requests.get(url, headers=headers)
    response_data = response.json()
    response.raise_for_status()
    sync_str = f"AWS_ACCESS_KEY_ID={response_data.get('access_key')} AWS_SECRET_ACCESS_KEY={response_data.get('secret_key')} aws s3 sync"

    payload = {"name": datastore_name}
    url = SEGMIND_SPOT_URL + "/datastore/details"
    response = requests.get(url, headers=headers, data=json.dumps(payload))
    response_data = response.json()
    response.raise_for_status()
    bucket_name = response_data.get("s3_bucket_name")
    folder_name = response_data.get("folder_name")
    sync_str_destination = f"s3://{bucket_name}/{folder_name}"

    return sync_str + " {source} " + sync_str_destination + "{destination}"


def _get_token(via_cli):
    try:
        # Check user-credentials and fetch the token.
        token = get_token()
    except Exception as err:
        if via_cli:
            red_print(
                "Could not find your secret file, make sure you have already done 'segmind config'"
            )
        raise err

    return token


def walkdir(folder):
    """Walk through every files in a directory"""
    for dirpath, _, files in os.walk(folder):
        for filename in files:
            yield dirpath, filename


def _upload_folder_to_s3(s3, path, bucket_name, s3_folder_name, destination_path):
    """
    boto3 does not support s3 folder upload, os.walk to upload every file
    in the desired directory
    """
    abspath_of_folder = os.path.abspath(path)
    folder_name = abspath_of_folder.split("/")[-1]

    if destination_path:
        s3_path = s3_folder_name + destination_path
    else:
        s3_path = s3_folder_name + "/" + folder_name

    # Precomputing files count
    filescount = 0
    for _ in walkdir(abspath_of_folder):
        filescount += 1

    for dirpath, filepath in tqdm(walkdir(abspath_of_folder), total=filescount):
        filepath_ = dirpath + "/" + filepath
        s3file_ = dirpath.replace(abspath_of_folder, s3_path) + "/" + filepath

        statinfo = os.stat(filepath_)
        with tqdm(
            total=statinfo.st_size,
            desc=f"File: {filepath_}",
            bar_format="{percentage:.1f}%|{bar:25} | {rate_fmt} | {desc}",
            unit="B",
            unit_scale=True,
            unit_divisor=1024,
            leave=False,
        ) as progress_bar_2:
            with open(filepath_, "rb") as data:
                s3.upload_fileobj(
                    Fileobj=data,
                    Bucket=bucket_name,
                    Key=s3file_,
                    Callback=progress_bar_2.update,
                )


def _upload_file_to_s3(s3, path, bucket_name, s3_path):
    # Upload a new file
    statinfo = os.stat(path)
    with tqdm(
        total=statinfo.st_size,
        desc=f"File: {path}",
        bar_format="{percentage:.1f}%|{bar:25} | {rate_fmt} | {desc}",
        unit="B",
        unit_scale=True,
        unit_divisor=1024,
        leave=False,
    ) as progress_bar:
        with open(path, "rb") as data:
            s3.upload_fileobj(
                Fileobj=data,
                Bucket=bucket_name,
                Key=s3_path,
                Callback=progress_bar.update,
            )


def upload(path, datastore_name, destination_path="", via_cli=True):
    token = _get_token(via_cli=via_cli)
    s3, bucket_name, folder_name = _get_upload_credentials(
        token=token, datastore_name=datastore_name
    )

    if destination_path and destination_path in ('.', '/'):
        raise ValueError("destination_path can't be . or / Please specify a file/folder name.")
    if destination_path and not destination_path.startswith('/'):
        destination_path = '/' + destination_path

    if os.path.isdir(path):
        _upload_folder_to_s3(
            s3=s3,
            path=path,
            bucket_name=bucket_name,
            s3_folder_name=folder_name,
            destination_path=destination_path,
        )
    else:
        s3_path = folder_name + "/" + destination_path
        _upload_file_to_s3(
            s3=s3,
            path=path,
            bucket_name=bucket_name,
            s3_path=s3_path,
        )


def sync(path, datastore_name, destination_path="", via_cli=True):
    token = _get_token(via_cli=via_cli)
    sync_str = _get_credentials_str(
        token=token, datastore_name=datastore_name
    )

    abspath_of_folder = os.path.abspath(path)
    folder_name = abspath_of_folder.split("/")[-1]

    if destination_path:
        if destination_path.startswith('/'):
            destination = destination_path
        else:
            destination = "/" + destination_path
    else:
        destination = "/" + folder_name

    os.system(sync_str.format(source=path, destination=destination))
