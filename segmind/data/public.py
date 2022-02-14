import json
import os

import boto3
import requests

from segmind.lite_extensions.client_utils import get_token
from segmind.lite_extensions.urls import SEGMIND_API_URL, SEGMIND_SPOT_URL
from segmind.utils import red_print


def get_s3_session(key, secret):
    # You can create a session:
    session = boto3.Session(
        aws_access_key_id=key,
        aws_secret_access_key=secret
    )

    # Let's use Amazon S3
    s3 = session.resource('s3')

    return s3


def _get_upload_credentials(token, datastore_name):
    headers = {
        "Authorization": 'Bearer ' + str(token),
        'Content-Type': 'application/json'
    }
    url = SEGMIND_API_URL + "/clusters/s3-creds"
    response = requests.get(
        url,
        headers=headers
    )
    response_data = response.json()
    response.raise_for_status()
    s3 = get_s3_session(
        key=response_data.get("access_key"),
        secret=response_data.get("secret_key")
    )

    payload = {
        "name": datastore_name
    }
    url = SEGMIND_SPOT_URL + "/datastore/details"
    response = requests.get(
        url,
        headers=headers,
        data=json.dumps(payload)
    )
    response_data = response.json()
    response.raise_for_status()
    bucket_name = response_data.get("s3_bucket_name")
    folder_name = response_data.get("folder_name")

    return s3, bucket_name, folder_name


def _get_token(via_cli):
    try:
        # Check user-credentials and fetch the token.
        token = get_token()
    except Exception as err:
        if via_cli:
            red_print("Could not find your secret file, make sure you have already done 'segmind config'")
        raise err

    return token


# Print iterations progress
def print_progress_bar(iteration, total, prefix='', suffix='', decimals=1, length=100, fill='█', printEnd="\r"):
    """
    Call in a loop to create terminal progress bar
    @params:
        iteration   - Required  : current iteration (Int)
        total       - Required  : total iterations (Int)
        prefix      - Optional  : prefix string (Str)
        suffix      - Optional  : suffix string (Str)
        decimals    - Optional  : positive number of decimals in percent complete (Int)
        length      - Optional  : character length of bar (Int)
        fill        - Optional  : bar fill character (Str)
        printEnd    - Optional  : end character (e.g. "\r", "\r\n") (Str)
    """
    percent = ("{0:." + str(decimals) + "f}").format(100 * (iteration / float(total)))
    filledLength = int(length * iteration // total)
    bar = fill * filledLength + '-' * (length - filledLength)
    print(f'\r{prefix} |{bar}| {percent}% {suffix}', end=printEnd)
    # Print New Line on Complete
    if iteration == total:
        print()


def _upload_folder_to_s3(s3, path, bucket_name, s3_folder_name, destination_path):
    """
        boto3 does not support s3 folder upload, os.walk to upload every file
        in the desired directory
    """
    abspath_of_folder = os.path.abspath(path)
    folder_name = abspath_of_folder.split('/')[-1]

    if destination_path:
        s3_path = s3_folder_name + "/" + destination_path
    else:
        s3_path = s3_folder_name + "/" + folder_name

     # Initial call to print 0% progress
    length_of_items = len([name for name in os.listdir(path) if os.path.isfile(name)])
    print_progress_bar(0, length_of_items, prefix='Progress:', suffix='Complete')
    for root, _, files in os.walk(path):
        for index, file in enumerate(files):
            __local_file = root + "/" + file
            __s3file = root.replace(abspath_of_folder, s3_path) + "/" + file

            s3.Bucket(bucket_name).upload_file(__local_file, __s3file)

            # Update Progress Bar
            print_progress_bar(index + 1, length_of_items, prefix='Progress:', suffix='Complete')


def _upload_file_to_s3(s3, path, bucket_name, s3_path):
    # Upload a new file
    print_progress_bar(0, 1, prefix='Progress:', suffix='Complete')
    with open(path, 'rb') as data:
        s3.Bucket(bucket_name).put_object(Key=s3_path, Body=data)

        # Update Progress Bar
        print_progress_bar(1, 1, prefix='Progress:', suffix='Complete')


def upload(path, datastore_name, destination_path="", via_cli=True):
    token = _get_token(via_cli=via_cli)
    s3, bucket_name, folder_name = _get_upload_credentials(
        token=token,
        datastore_name=datastore_name
    )

    if os.path.isdir(path):
        _upload_folder_to_s3(
            s3=s3,
            path=path,
            bucket_name=bucket_name,
            s3_folder_name=folder_name,
            destination_path=destination_path
        )
    else:
        s3_path = folder_name + "/" + destination_path
        _upload_file_to_s3(
            s3=s3,
            path=path,
            bucket_name=bucket_name,
            s3_path=s3_path
        )
