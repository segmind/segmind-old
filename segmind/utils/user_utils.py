import requests
from segmind.lite_extensions.urls import SEGMIND_API_URL


def get_user_info(
    token, username=True, email=False, access_token=None, refresh_token=None
):
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
    if access_token:
        user_info["access_token"] = response_data["access_token"]
    if refresh_token:
        user_info["refresh_token"] = response_data["refresh_token"]

    return user_info
