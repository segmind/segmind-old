import getpass
import json
import os
import sys
from pprint import pprint

import click
# import jsonpickle
from click import UsageError
# from tensorflow import keras

from segmind_track.lite_extensions.client_utils import LoginError, fetch_token
from segmind_track.utils import cyan_print, green_print, red_print, yellow_print


@click.group()
def cli():
    pass

@cli.command()
def config():
    cyan_print("Please enter your credentials for https://track.segmind.com")

    email = input('Enter Email-id :: ')
    password = getpass.getpass('Enter Password :: ') 

    try:
        fetch_token(email, password)
    except LoginError:
        red_print("Log-In failed !!! Invalid credentials")
        sys.exit()

    folder_path = os.path.join(os.path.expanduser('~'),'.segmind')
    os.makedirs(folder_path, exist_ok=True)

    file_path = os.path.join(folder_path,'secret.file')

    with open(file_path,'w') as file:
        file.write("[secret]\n")
        file.write("email={}\n".format(email))
        file.write("password={}\n".format(password))

    green_print("Log-In Successful !!!")


# _INSPECT_PRINT_TEMPLATE = "This Model was trained for `{}` task,\nUsing `{}` as backbone for feature extraction to build `{}`,\nrelevant configurations are {}"


# @cli.command()
# @click.option("--model-path", 
#     required=True, 
#     type=click.Path(exists=True),
#     help="String or pathlib.Path object, path to the saved model")
# def inspect_model(model_path, *args, **kwargs):
#     model = keras.models.load_model(model_path)

#     try:        
#         location_to_cral_file = model.cral_file.asset_path.numpy()
#         with open(location_to_cral_file) as f:
#             metainfo = json.loads(f.read())
#         #pprint(metainfo)
#         experiment_id = metainfo['exp_id']
#         task_type = metainfo['task_type'].lower()
#         # dataset_format = 
#         if task_type == 'classification':
#             submeta = metainfo['classification_meta']
#         elif task_type == 'object_detection':
#             submeta = metainfo['object_detection_meta']
#         else:
#             pprint(metainfo)
#             return

#         algorithm_name = submeta['architecture']
#         feature_extractor = submeta['feature_extractor']
#         config = vars(jsonpickle.decode(submeta['config']))

#         yellow_print(_INSPECT_PRINT_TEMPLATE.format(task_type, feature_extractor, algorithm_name, config))
#         green_print("For more details login to your account in https://track.segmind.com, \ncheckout experiment-id :: {}".format(experiment_id))


#     except AttributeError:
#         red_print("Couldn't locate any cral config file, probably this model was not trained using cral, or may be corrupted")
