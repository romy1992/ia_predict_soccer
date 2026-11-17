import logging
import os
import re
import time

import requests
from dotenv import load_dotenv

logging.basicConfig(level=logging.DEBUG)


class _RedactApiKeyFilter(logging.Filter):
    """The Odds API vuole la chiave nella query string (`apiKey=...`) e
    urllib3 logga a DEBUG la riga di richiesta completa, query inclusa
    (`"GET /v4/sports?apiKey=<chiave> HTTP/1.1" 200`). Questo modulo imposta
    il root logger a DEBUG: senza filtro la chiave finisce nei log ad ogni
    chiamata. Il filtro sostituisce il valore con `***` (il resto della riga
    resta)."""

    _PATTERN = re.compile(r"(apiKey=)[^&\s\"']+", re.IGNORECASE)

    def filter(self, record: logging.LogRecord) -> bool:
        try:
            message = record.getMessage()
        except Exception:  # noqa: BLE001 - un record malformato non deve bloccare il log
            return True
        if "apikey=" in message.lower():
            record.msg = self._PATTERN.sub(r"\1***", message)
            record.args = ()
        return True


def _install_api_key_redaction() -> None:
    target = logging.getLogger("urllib3.connectionpool")
    if not any(isinstance(existing, _RedactApiKeyFilter) for existing in target.filters):
        target.addFilter(_RedactApiKeyFilter())


_install_api_key_redaction()

# LOAD PROPERTIES
load_dotenv(dotenv_path='../../properties/config.env')
API_SPORTS_KEY = os.environ.get('API_SPORTS_KEY')
API_SPORTS_BASE = os.environ.get('API_SPORTS_BASE')
API_ODDS_KEY = os.environ.get('API_ODDS_KEY')
API_ODDS_BASE = os.environ.get('API_ODDS_BASE')
API_ODDS_ACTUAL_BASE = os.environ.get('API_ODDS_ACTUAL_BASE')
API_ODDS_HISTORIACAL_BASE = os.environ.get('API_ODDS_HISTORIACAL_BASE')
# Timeout esplicito: senza, una connessione appesa blocca il job a tempo indefinito.
API_ODDS_TIMEOUT_SECONDS = float(os.environ.get('API_ODDS_TIMEOUT_SECONDS') or 30)


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
        # Check per utilizzo API. Gli header possono mancare (errori 4xx/5xx,
        # proxy): senza entrambi non si calcola nessuna percentuale invece di
        # dividere per zero.
        used = float(response.headers.get("X-Requests-Used", 0) or 0)
        remaining = float(response.headers.get("X-Requests-Remaining", 0) or 0)
        total = used + remaining
        if total > 0:
            percent_used = (used / total) * 100
            logging.info(f'Percent used {percent_used}%')
            if percent_used > 99:
                raise Exception(f'Stop API : Used {percent_used}%.')

        if response.status_code != 200:
            return []
        payload = response.json()
        return payload if payload else []

    try:
        response = requests.get(url=f'{split_api()}/{path}', params=params, timeout=API_ODDS_TIMEOUT_SECONDS)
    except requests.RequestException as exc:
        # Il messaggio originale di requests puo' contenere l'URL completo con
        # `apiKey` nella query: si rilancia solo il tipo di errore (come in
        # `src/service_ia/utility/request_api.py`).
        raise requests.RequestException(
            f'The Odds API non raggiungibile su {path!r}: {type(exc).__name__}'
        ) from None
    return check_response(response)


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
