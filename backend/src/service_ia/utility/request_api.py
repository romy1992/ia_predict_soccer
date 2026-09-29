import logging
import os
import time

import requests
from dotenv import load_dotenv

from src.service_ia.pre_processing.api_sports_provider import ApiSportsProvider

logging.basicConfig(level=logging.DEBUG)

# LOAD PROPERTIES
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
load_dotenv(dotenv_path=os.path.join(PROJECT_ROOT, "properties", "config.env"))
API_SPORTS_KEY = os.environ.get('API_SPORTS_KEY')
API_SPORTS_BASE = os.environ.get('API_SPORTS_BASE')
API_ODDS_KEY = os.environ.get('API_ODDS_KEY')
API_ODDS_BASE = os.environ.get('API_ODDS_BASE')
API_ODDS_ACTUAL_BASE = os.environ.get('API_ODDS_ACTUAL_BASE')
API_ODDS_HISTORIACAL_BASE = os.environ.get('API_ODDS_HISTORIACAL_BASE')
_API_SPORTS_PROVIDER = ApiSportsProvider()


def get_api_sports_provider() -> ApiSportsProvider:
    return _API_SPORTS_PROVIDER


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
        # Check per utilizzo API
        used = int(response.headers.get("X-Requests-Used", 0))
        remaining = float(response.headers.get("X-Requests-Remaining", 0))
        percent_used = (used / (used + remaining)) * 100
        logging.info(f'Percent used {percent_used}%')
        if percent_used > 99:
            raise Exception(f'Stop API : Used {percent_used}%.')

        return response.json() if response.status_code == 200 and response.json() else []

    return check_response(requests.get(url=f'{split_api()}/{path}', params=params))


def base_api_statistics(path='', params=None):
    return get_api_sports_provider().request(path=path, params=params)
