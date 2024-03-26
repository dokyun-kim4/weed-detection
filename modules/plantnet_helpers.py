"""
Functions to communicate with the Pl@ntNet API
"""

import requests
import json
from pprint import pprint
import cv2


def API_Initialize(API_Key, Region="all"):
    """
    When provided with a specific API key and Region, will initialize an API endpoint with Pl@ntNet

    Args:

        API_Key (String): The API key being used for the connection.

        Region (String): The region of interest for plant lookup.
    
    Returns:

        api_endpoint (String): A URL to the API endpoint.
    """
    api_endpoint = f"https://my-api.plantnet.org/v2/identify/{Region}?api-key={API_Key}"
    return api_endpoint


def load_plant_data(img_data, Organ):
    """
    Helper function for Send_API_Request.
    Constructs the json representation of required data to be passed to the Pl@ntNet API.

    Args:

        img_data (img_buffer): An image buffer of image data to be send to the API.

        Organ (list of strings): The plant organ visible for each image being processed.
            Options include "leaf" or "flower".
        
    Returns:

        data (dict): A dictionary to inform Pl@ntNet's API about the organs visible in data.

        files (list of tuples): A list of formatted image files to be sent to Pl@ntNet API.
    """
    data = {"organs": Organ}
    files = [
        ("images", (img_data)),
    ]
    return data, files


def Send_API_Request(api_endpoint, files, data):
    """
    Sends API request to Pl@ntNet and receives results on the identified plants.

    Args:
        api_endpoint (String): A URL to the API endpoint.

        files (list of tuples): A list of formatted image files to be sent to Pl@ntNet API.

        data (dict): A dictionary to inform Pl@ntNet's API about the organs visible in data.

    Returns:
        json_result (dict): A dictionary in json format containing the result from Pl@ntNet API.
        Includes data on plant's ID (scientific & common name), as well as confidence in the identification
        result.
    """
    req = requests.Request("POST", url=api_endpoint, files=files, data=data)
    prepared = req.prepare()
    s = requests.Session()
    response = s.send(prepared)
    json_result = json.loads(response.text)
    pprint(response)
    return json_result
