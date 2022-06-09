import os
import requests
import json


def stop(instance_id=''):
	if 'TRACK_ACCESS_TOKEN' not in os.environ:
		raise Exception(f"""Please try to setup user access using 
							import segmind
							segmind.config_nb(access_token=...)\n
							or segmind config --access_token=...""")

	if instance_id == '':
		if 'HOSTNAME' not in os.environ:
			raise Exception(f"""Sorry couldn't find any specific instance from environment, Try this command - instance.stop('instance_id')""")
		get_instance_id_from_hostname = str(os.environ['HOSTNAME']).split('-')
		instance_id = get_instance_id_from_hostname[2]

	url = f"https://api.spotprod.segmind.com/notebook/{instance_id}/stop"
	payload = ""
	headers = {
	'Connection': 'keep-alive',
	'sec-ch-ua': '"Chromium";v="92", " Not A;Brand";v="99", "Google Chrome";v="92"',
	'Accept': 'application/json, text/plain, */*',
	'Authorization': f"Bearer {os.environ['TRACK_ACCESS_TOKEN']}"
	}

	response = requests.request("POST", url, headers=headers, data=payload)
	return_response = json.loads(response.text)
	
	if response.status_code == 200:
		return f"Notebook with id {return_response['id']} has been {return_response['status']}"
	
	if not response.status_code == 200:
		return return_response