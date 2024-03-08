# Functions for Pl@ntNet API

import requests
import json
from pprint import pprint
import cv2


def API_Initialize(API_Key, Region="all"):
    api_endpoint = f"https://my-api.plantnet.org/v2/identify/{Region}?api-key={API_Key}"
    return api_endpoint


def load_plant_data(Img_Path, Organ):
    path = Img_Path
    image_data = open(path, "rb")
    data = {"organs": [Organ]}
    files = [
        ("images", (image_data)),
    ]
    return data, files


def Send_API_Request(api_endpoint, files, data):
    req = requests.Request("POST", url=api_endpoint, files=files, data=data)
    prepared = req.prepare()
    s = requests.Session()
    response = s.send(prepared)
    json_result = json.loads(response.text)
    pprint(response)
    return json_result
