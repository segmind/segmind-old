import requests
from segmind.lite_extensions.client_utils import get_token
from segmind.lite_extensions.urls import SEGMIND_API_URL


def get_user_info(username=True, email=False, token=None):
    if not token:
        token = get_token()

    url = SEGMIND_API_URL + "/auth/profile"
    headers = {
        "Authorization": "Bearer " + str(token),
        "Content-Type": "application/json",
    }
    response = requests.get(url, headers=headers)
    response.raise_for_status()
    response_data = response.json()

    user_info = {}
    if username:
        user_info["username"] = response_data["username"]
    if email:
        user_info["email"] = response_data["email"]

    return user_info
