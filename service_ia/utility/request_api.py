import logging
import os

import requests
from dotenv import load_dotenv

logging.basicConfig(level=logging.DEBUG)

# LOAD PROPERTIES
load_dotenv(dotenv_path='../../properties/config.env')
API_SPORTS_KEY = os.environ.get('API_SPORTS_KEY')
API_SPORTS_BASE = os.environ.get('API_SPORTS_BASE')
API_ODDS_KEY = os.environ.get('API_ODDS_KEY')
API_ODDS_BASE = os.environ.get('API_ODDS_BASE')
API_ODDS_ACTUAL_BASE = os.environ.get('API_ODDS_ACTUAL_BASE')
API_ODDS_HISTORIACAL_BASE = os.environ.get('API_ODDS_HISTORIACAL_BASE')


def base_api_odds(type_api=None, path='', params=None):
    if not params:
        params = {}
    params.update({'apiKey': API_ODDS_KEY})

    def split_api():
        match type_api:
            case 'normal':
                return API_ODDS_ACTUAL_BASE
            case 'hist':
                return API_ODDS_HISTORIACAL_BASE
            case _:
                return API_ODDS_BASE

    def check_response(response):
        return response.json() if response.status_code == 200 and response.json() else []

    return check_response(requests.get(url=f'{split_api()}/{path}', params=params))


def base_api_statistics(path='', params=None):
    def check_response(response):
        return response.json().get(
            'response') if response.status_code == 200 and response.json() and response.json().get('response') else []

    return check_response(
        requests.get(url=f'{API_SPORTS_BASE}/{path}', headers={'x-apisports-key': API_SPORTS_KEY}, params=params))
