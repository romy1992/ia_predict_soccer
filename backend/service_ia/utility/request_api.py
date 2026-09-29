import logging
import os
import time

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
        # Check per utilizzo API
        used = int(response.headers.get("X-Requests-Used", 0))
        remaining = int(response.headers.get("X-Requests-Remaining", 0))
        percent_used = (used / (used + remaining)) * 100
        logging.info(f'Percent used {percent_used}%')
        if percent_used > 99:
            raise Exception(f'Stop API : Used {percent_used}%.')

        return response.json() if response.status_code == 200 and response.json() else []

    return check_response(requests.get(url=f'{split_api()}/{path}', params=params))


def base_api_statistics(path='', params=None):
    def check_response(response):

        # Check per utilizzo API
        remaining = int(response.headers.get("x-ratelimit-requests-remaining", 0))
        logging.info(f'Remaining {remaining}')
        if remaining == 0:
            raise Exception(f'Stop API : Remaining {remaining} today.')

        # Check per API utilizzate al minuto
        rate_limit = int(response.headers.get("x-ratelimit-limit", 0))
        rate_limit_remaining = int(response.headers.get("x-ratelimit-remaining", 0))
        logging.info(f'Limit minute {rate_limit_remaining}/{rate_limit}')
        if rate_limit_remaining == 0:
            logging.info(f'Sleep process : over {rate_limit}')
            time.sleep(60)

        return response.json().get(
            'response') if response.status_code == 200 and response.json() and response.json().get('response') else []

    return check_response(
        requests.get(url=f'{API_SPORTS_BASE}/{path}', headers={'x-apisports-key': API_SPORTS_KEY}, params=params))
