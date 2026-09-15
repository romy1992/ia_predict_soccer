"""
Classe che nasce per l'esigenza di scaricare tutti i prossimi match che ci saranno, con statistiche e quote, e
inserirli a db.
Si punterà a una sola piattaforma, quella di API SPORTS che contiene anche le quote che però durano una settimana.
L'idea sarebbe di attingere quotidianamente a una serie di leghe dell'anno corrente per recuperare tutti i dati
di pre-processing in modo da poter addestrare cin grosse quantità anche in futuro.
"""
import json
import logging
import os
import hashlib
from datetime import datetime, timedelta, timezone
from statistics import mean
from typing import Optional
from uuid import uuid4

import pandas as pd

from src.repository.base.repository_db import SessionLocal
from src.repository.match_repository import MatchRepository
from src.repository.odds_snapshot_repository import OddsSnapshotRepository
from src.service_ia.config.app_config import load_app_config
from src.service_ia.mapper.statistic_mapper import get_attribute_statistics, form_last_5_tot
from src.service_ia.model.match import Match, Statistics, Odds, OddsSnapshot
from src.service_ia.pre_processing.api_sports_provider import (
    ApiSportsProvider,
    ApiSportsQuotaExceededError,
    ApiSportsUnavailableError,
)
from src.service_ia.utility.request_api import base_api_statistics, get_api_sports_provider

logging.basicConfig(level=logging.DEBUG)

# Variabili
"""
135 (Serie A) - 136 (Serie B) - 137 (Coppa Italia) - 138, 942, 943 (Serie C - Girone A,B e C) - 
2 (Champions League) - 3 (Europa League) - 848 (Conference) - 
140 (Liga Spagnola) - 
78 (Bundesliga) - 
39 (Premier League) - 
88 (Olandese) - 492 (Olanda serie B)
94 (Portoghese) -
144 (Belgio) - 
61 (Francia) - 
203 (Turchia)
"""
repo_match = MatchRepository()
repo_snapshot = OddsSnapshotRepository()

BASE_DIR = os.path.dirname(__file__)
BET_FILE = os.path.join(BASE_DIR, '..', 'json', 'bet.json')

with open(os.path.abspath(BET_FILE), 'r', encoding='utf-8') as file:
    BET_BOOKMAKERS = json.load(file)

# FT è partita finita
# NS è partita non disputata
# AET è per partita finita ai supplementari (QUINDI PER COPPE)
# PEN è per partita finita ai rigori (QUINDI PER COPPE)
# ABD è per partita abbandonata
status_list = "FT-AET-PEN-ABD"


def map_base_match(match, id_fix, fixture, league, season):
    teams_home = fixture['teams']['home']
    teams_away = fixture['teams']['away']

    def get_val(team, value):
        return team[value]

    # Punteggio finale (2026-09-10): salvato QUI, indipendentemente da
    # 'statistics', perche' arriva dalla STESSA risposta 'fixtures' che
    # fornisce gia' status/data (sempre disponibile) - a differenza di
    # 'score_ft' su Statistics (vedi map_statistic sotto), che esiste SOLO
    # se l'endpoint dedicato 'fixtures/statistics' ha dati per quella
    # fixture (spesso assente per campionati minori con copertura API
    # limitata: il match risultava "Finita" ma senza alcun punteggio
    # mostrabile in Dashboard, ne' un risultato reale per h2h/goal_no_goal/
    # under_over/dc). Stessa fonte ('score.fulltime.{home,away}') gia'
    # usata da `map_statistic` per 'score_ft', cosi' i due valori restano
    # sempre coerenti quando entrambi disponibili.
    score_fulltime = (fixture.get('score') or {}).get('fulltime') or {}

    return {
        'id_match_fk': match.id_match_fk if match else str(uuid4()),
        'id_fixture': id_fix,
        'name_home': get_val(teams_home, 'name'),
        'id_team_home': get_val(teams_home, 'id'),
        'name_away': get_val(teams_away, 'name'),
        'id_team_away': get_val(teams_away, 'id'),
        'date_match': fixture['fixture']['date'],
        'current_league': league,
        'league_match': fixture['league']['id'],
        'referee': fixture['fixture']['referee'],
        'round': fixture['league']['round'],
        'season': season,
        'status': fixture['fixture']['status']['short'],
        'score_home': score_fulltime.get('home'),
        'score_away': score_fulltime.get('away'),
    }


def map_statistic(match, stat, team, id_fix, fixture):
    """
    Mappa statistiche, forma fisica, comparison e prediction
    :return: nuovo dizionario di statistiche
    """
    id_team = stat['team']['id']
    stat = stat['statistics']

    attribute_stat = get_attribute_statistics(stat)

    last_5 = form_last_5_tot(id_fix, id_team)
    if last_5:
        attribute_stat.update(last_5)

    def search_key_value(index_key):
        return {key: value for key, value in attribute_stat.items() if pd.notna(value) and index_key in key}

    def default_attribute(attribute):
        return attribute_stat.get(attribute) or 0

    id_stat = None
    id_match = None
    if match and match.statistics:
        id_match = match.id_match_fk
        id_stat = [st for st in match.statistics if st.statistics_team_id == id_team]
        if len(id_stat) > 0:
            id_stat = id_stat[0].id_statistics_fk

    return {
        'id_statistics_fk': id_stat,
        'id_match': id_match,
        'statistics_team_id': id_team,
        'score_ht': fixture['score']['halftime'][team],
        'score_ft': fixture['score']['fulltime'][team],
        'shots': {
            'Shots on Goal': default_attribute('Shots on Goal'),
            'Shots off Goal': default_attribute('Shots off Goal'),
            'Total Shots': default_attribute('Total Shots'),
            'Blocked Shots': default_attribute('Blocked Shots'),
            'Shots insidebox': default_attribute('Shots insidebox'),
            'Shots outsidebox': default_attribute('Shots outsidebox')
        },
        'fouls': default_attribute('Fouls'),
        'corners': default_attribute('Corner Kicks'),
        'offside': default_attribute('Offsides'),
        'bass_possession': default_attribute('Ball Possession'),
        'yellow_cards': default_attribute('Yellow Cards'),
        'red_cards': default_attribute('Red Cards'),
        'goal_keeper': default_attribute('Goalkeeper Saves'),
        'passes': search_key_value('asses'),
        'form': search_key_value('form_'),
        'for_': search_key_value('for_'),
        'against': search_key_value('against_'),
        'preview_matches': {
            'wins_home': default_attribute('wins_home'),
            'wins_away': default_attribute('wins_away'),
            'draws_home': default_attribute('draws_home'),
            'draws_away': default_attribute('draws_away'),
            'loses_home': default_attribute('loses_home'),
            'loses_away': default_attribute('loses_away')
        },
        'comparison': search_key_value('comparison_'),
        'predict': search_key_value('predict_'),
        'generic_statistics': {
            'expected_goals': default_attribute('expected_goals'),
            'goals_prevented': default_attribute('goals_prevented'),
            'Assists': default_attribute('Assists'),
            'Counter Attacks': default_attribute('Counter Attacks'),
            'Cross Attacks': default_attribute('Cross Attacks'),
            'Free Kicks': default_attribute('Free Kicks'),
            'Goals': default_attribute('Goals'),
            'Goal Attempts': default_attribute('Goal Attempts'),
            'Substitutions': default_attribute('Substitutions'),
            'Throwins': default_attribute('Throwins'),
            'Medical Treatment': default_attribute('Medical Treatment')
        }
    }


def map_odds(match, id_fix, fixture_bookmakers=None):
    """
    Mappa le quote dei bookmakers
    :return: nuovo dizionario di quote
    """

    def switch_bet(bet, alternate_bet):
        match bet:
            case 'Match Winner':
                return 'h2h'
            case 'Goals Over/Under':
                if alternate_bet in ('Over 1.5', 'Under 1.5'):
                    return 'under_over_1_5'
                elif alternate_bet in ('Over 2.5', 'Under 2.5'):
                    return 'under_over_2_5'
                elif alternate_bet in ('Over 3.5', 'Under 3.5'):
                    return 'under_over_3_5'
                elif alternate_bet in ('Over 4.5', 'Under 4.5'):
                    return 'under_over_4_5'
            case 'Both Teams Score':
                return 'goal_no_goal'
            case 'Corners Over Under':
                return 'corners'
            case 'Cards Over/Under':
                return 'cards'
            case 'Double Chance':
                return 'dc'
        return None

    if fixture_bookmakers is None:
        fixture_bookmakers = base_api_statistics(path='odds', params={'fixture': id_fix})
    if len(fixture_bookmakers) > 0:
        # Inizia a creare il dizionario prima di aggiungere le quote
        bookmakers_filters = [bookmaker for bookmaker in fixture_bookmakers[0]['bookmakers']]

        if len(bookmakers_filters) > 0:
            # Odds base se tutti i nodi esistono altrimenti usa quelli che già ha (sempre se ci sono)
            odd_bet = {
                'id_odds_fk': match.odds[0].id_odds_fk if match and len(match.odds) > 0 else None,
                'id_match': match.id_match_fk if match else None,
                'odds_from': 'sports-api'
            }
            for bookmaker in bookmakers_filters:
                # Crea il dizionario della fixture aggregando tutti gli eventi con le sue quote
                name_book = bookmaker['name']
                filter_bet = [bet for bet in bookmaker['bets'] if
                              bet['id'] in [id_bet['id'] for id_bet in BET_BOOKMAKERS]]  # SOLO QUOTE DI EVENTI AMMESSI
                for filter_bet_name in filter_bet:
                    for value in filter_bet_name['values']:
                        alternate_value = str(value['value']).lower()
                        if alternate_value in ['yes', 'no']:
                            # BUGFIX 2026-09-08: confrontava alternate_value (gia'
                            # lowercased sopra) con 'Yes' (maiuscola) - sempre
                            # falso, quindi sia "Yes" che "No" finivano SEMPRE su
                            # 'no_goal_', con la seconda occorrenza (stessa chiave
                            # f'{alternate_value}_{name_book}') che sovrascriveva
                            # silenziosamente la prima: le quote goal_no_goal
                            # salvate a DB avevano un solo lato (quello processato
                            # per ultimo dall'API) per bookmaker, mai entrambi.
                            alternate_value = 'goal_' if alternate_value == 'yes' else 'no_goal_'
                        elif alternate_value in ['home/draw', 'home/away', 'draw/away']:
                            alternate_value = '1X' if alternate_value == 'home/draw' else '12' if alternate_value == 'home/away' else 'X2'

                        name_bet = switch_bet(bet=filter_bet_name['name'], alternate_bet=value['value'])
                        if name_bet:
                            head_title = f'{alternate_value}_{name_book}'
                            context = {head_title: value['odd']}
                            if context and head_title and value['odd']:
                                odd_bet.get(name_bet).update(context) \
                                    if odd_bet.get(name_bet) else odd_bet.update({name_bet: context})

            return odd_bet  # ritorna odds

    # ritorna quello che già ha se non ci sono nodi delle api chiamate
    return match.odds[0].to_dict() if match and match.odds and len(match.odds) > 0 else None


def _parse_iso_datetime(value: str | None) -> Optional[datetime]:
    if not value:
        return None
    try:
        dt = datetime.fromisoformat(value.replace("Z", "+00:00"))
        return dt if dt.tzinfo else dt.replace(tzinfo=timezone.utc)
    except ValueError:
        return None


def _split_statuses(statuses: str | None) -> list[str]:
    if not statuses:
        return []
    tokens: list[str] = []
    for chunk in statuses.replace(",", "-").split("-"):
        value = chunk.strip().upper()
        if value:
            tokens.append(value)
    return tokens


def _extract_line_value(raw_value: str) -> Optional[str]:
    import re

    matches = re.findall(r"([0-9]+(?:\.[0-9]+)?)", raw_value or "")
    if not matches:
        return None
    return matches[-1]


def _switch_market_name(bet_name: str, alternate_bet: str) -> Optional[str]:
    match bet_name:
        case 'Match Winner':
            return 'h2h'
        case 'Goals Over/Under':
            if alternate_bet in ('Over 1.5', 'Under 1.5'):
                return 'under_over_1_5'
            if alternate_bet in ('Over 2.5', 'Under 2.5'):
                return 'under_over_2_5'
            if alternate_bet in ('Over 3.5', 'Under 3.5'):
                return 'under_over_3_5'
            if alternate_bet in ('Over 4.5', 'Under 4.5'):
                return 'under_over_4_5'
        case 'Both Teams Score':
            return 'goal_no_goal'
        case 'Corners Over Under':
            return 'corners'
        case 'Cards Over/Under':
            return 'cards'
        case 'Double Chance':
            return 'dc'
    return None


def _safe_float(raw_value) -> Optional[float]:
    if raw_value is None:
        return None
    try:
        return float(str(raw_value).replace(',', '.'))
    except Exception:
        return None


def map_odds_snapshots(id_match: str, id_fixture: int, fixture_bookmakers: list[dict]) -> list[OddsSnapshot]:
    if not fixture_bookmakers:
        return []

    source = 'api_sports'
    captured_raw = fixture_bookmakers[0].get('update') if isinstance(fixture_bookmakers[0], dict) else None
    captured_at = _parse_iso_datetime(captured_raw) or datetime.now(timezone.utc)

    allowed_bet_ids = {int(item['id']) for item in BET_BOOKMAKERS if isinstance(item, dict) and item.get('id') is not None}
    snapshots: list[OddsSnapshot] = []

    payload = fixture_bookmakers[0] if isinstance(fixture_bookmakers[0], dict) else {}
    bookmakers = payload.get('bookmakers') or []
    for bookmaker in bookmakers:
        bookmaker_name = bookmaker.get('name') or 'unknown'
        bets = bookmaker.get('bets') or []
        for bet in bets:
            bet_id = bet.get('id')
            try:
                bet_id_value = int(bet_id)
            except Exception:
                continue
            if bet_id_value not in allowed_bet_ids:
                continue

            bet_name = bet.get('name') or ''
            for value in bet.get('values') or []:
                outcome_raw = str(value.get('value') or '').strip()
                market_name = _switch_market_name(bet_name, outcome_raw)
                if not market_name:
                    continue

                odd_value = _safe_float(value.get('odd'))
                if odd_value is None or odd_value <= 0:
                    continue

                line_value = _extract_line_value(outcome_raw) if market_name.startswith('under_over_') or market_name in {'corners', 'cards'} else None
                signature = "|".join(
                    [
                        str(id_fixture),
                        bookmaker_name,
                        market_name,
                        'full_time',
                        line_value or '',
                        outcome_raw,
                        captured_at.isoformat(),
                        source,
                    ]
                )
                snapshot_id = hashlib.sha1(signature.encode('utf-8')).hexdigest()
                snapshots.append(
                    OddsSnapshot(
                        id_snapshot=snapshot_id,
                        id_match=id_match,
                        fixture_id=int(id_fixture),
                        bookmaker=bookmaker_name,
                        market=market_name,
                        period='full_time',
                        line=line_value,
                        outcome=outcome_raw,
                        odd=float(odd_value),
                        captured_at=captured_at,
                        source=source,
                    )
                )

    return snapshots


def download_import_matches(
    seasons=None,
    leagues=None,
    is_next=False,
    current_league: Optional[int] = None,
    from_date: Optional[str] = None,
    to_date: Optional[str] = None,
    fixture_date: Optional[str] = None,
    statuses: Optional[str] = None,
    days_ahead: Optional[int] = None,
    provider: Optional[ApiSportsProvider] = None,
):
    """
    Scarica tutte le partite storiche o successive con odds, statistiche e dettagli vari salvandoli a db
    :param seasons: se valorizzato, scarica solo partite di quella stagione altrimenti prende tutte quelle censite
    :param leagues: se valorizzato, scarica solo partite di quella lega altrimenti prende tutte quelle censite
    :param is_next: modalità legacy per recupero partite future
    :param current_league: stagione corrente opzionale (vincolo legacy)
    :param from_date: filtro data inizio (YYYY-MM-DD)
    :param to_date: filtro data fine (YYYY-MM-DD)
    :param fixture_date: filtro singola data (YYYY-MM-DD)
    :param statuses: stati fixture separati da '-' o ','
    :param days_ahead: se valorizzato costruisce una finestra futura da oggi
    :return: report import con inserted/updated/skipped/failed
    """
    cfg = load_app_config()
    seasons = cfg.seasons if seasons is None else seasons
    leagues = cfg.leagues if leagues is None else leagues
    provider = provider or get_api_sports_provider()

    now_utc = datetime.now(timezone.utc)
    if days_ahead is not None and days_ahead > 0:
        from_date = now_utc.date().isoformat()
        to_date = (now_utc.date() + timedelta(days=days_ahead)).isoformat()
        statuses = statuses or "NS"

    if not fixture_date and not from_date and not to_date:
        from_date = (now_utc - timedelta(days=3)).date().isoformat()
        to_date = now_utc.date().isoformat()

    normalized_statuses = _split_statuses(statuses or ("NS" if is_next else status_list))
    report = {
        "inserted": 0,
        "updated": 0,
        "skipped": 0,
        "failed": 0,
        # Leghe per cui il provider non ha risposto affatto (retry esauriti):
        # distinte da `failed`, che conta le singole fixture. Un valore > 0
        # significa "questo import e' PARZIALE", informazione che prima
        # andava completamente perduta (vedi `ApiSportsUnavailableError`).
        "failed_leagues": 0,
        "fixtures_seen": 0,
        "snapshots_upserted": 0,
        "params": {
            "seasons": seasons,
            "leagues": leagues,
            "from_date": from_date,
            "to_date": to_date,
            "fixture_date": fixture_date,
            "statuses": normalized_statuses,
            "is_next": is_next,
            "days_ahead": days_ahead,
        },
        "errors": [],
        "quota_exceeded": False,
    }

    # 2026-09-15: non esiste piu' una `list_matches` di righe nuove
    # bufferizzate fino alla fine del job (era la causa delle righe
    # duplicate, vedi `upsert_base_by_fixture`): ogni fixture viene
    # persistita nella sua iterazione. Gli snapshot restano bufferizzati -
    # li' il buffer serve a fare un solo bulk upsert invece di migliaia di
    # round-trip, e non c'e' identita' da contendere (la PK `id_snapshot` e'
    # un hash deterministico del contenuto).
    snapshot_buffer: list[OddsSnapshot] = []

    for season in seasons:
        logging.info('<<< Start season %s >>>', season)

        if is_next and current_league and season != current_league:
            continue

        if report["quota_exceeded"]:
            break

        for league in leagues:
            if report["quota_exceeded"]:
                break
            logging.info('<<< Start season %s for league %s >>>', season, league)
            params = {
                'league': league,
                'season': season,
            }
            if fixture_date:
                params['date'] = fixture_date
            else:
                if from_date:
                    params['from'] = from_date
                if to_date:
                    params['to'] = to_date
            if normalized_statuses:
                params['status'] = '-'.join(normalized_statuses)

            try:
                fixtures = provider.get_fixtures(**params)
            except ApiSportsQuotaExceededError as quota_exc:
                # Quota globale (giornaliera/di piano) esaurita: inutile
                # continuare a interrogare le altre leghe, falliranno tutte
                # allo stesso modo. Interrompiamo qui SENZA perdere quanto
                # gia' raccolto per le leghe precedenti in questo stesso giro.
                logging.error('<<< API-Sports quota esaurita, stop import: %s >>>', quota_exc)
                report['quota_exceeded'] = True
                report['errors'].append({
                    'fixture_id': None,
                    'error': f'API-Sports quota esaurita (season={season}, league={league}): {quota_exc}',
                })
                break
            except ApiSportsUnavailableError as unavailable_exc:
                # 2026-09-15: PRIMA `request()` restituiva `[]` a retry
                # esauriti, indistinguibile da "nessuna partita in questa
                # lega/data": la lega veniva saltata e il job chiudeva
                # `success` con `failed=0` e `errors=[]`, senza che nulla
                # dicesse che un intero campionato non era stato importato
                # (vedi `ApiSportsUnavailableError`). Ora il fallimento e'
                # esplicito e finisce nel report - ma NON interrompe il giro
                # come fa la quota esaurita: un 5xx o un rate-limit al minuto
                # riguardano questa richiesta, non tutte le leghe successive.
                logging.error(
                    '<<< API-Sports non disponibile per season=%s league=%s: %s >>>',
                    season, league, unavailable_exc,
                )
                report['failed_leagues'] += 1
                report['errors'].append({
                    'fixture_id': None,
                    'error': f'API-Sports non disponibile (season={season}, league={league}): {unavailable_exc}',
                })
                continue
            report['fixtures_seen'] += len(fixtures)

            for fixture in fixtures:
                try:
                    id_fix = int(fixture['fixture']['id'])
                except Exception:
                    report['skipped'] += 1
                    continue

                try:
                    # Cerco in db se esiste già match
                    match = repo_match.filter_by(dict_search={'id_fixture': id_fix}).first()

                    # Mappa la base del match
                    dict_match = map_base_match(match=match, id_fix=id_fix, fixture=fixture, league=league, season=season)

                    # Recupero statistiche dalla fixture o endpoint dedicato.
                    statistics = fixture.get('statistics') or []
                    status_short = str((fixture.get('fixture') or {}).get('status', {}).get('short') or '').upper()
                    if len(statistics) == 0 and status_short != 'NS':
                        statistics = provider.get_fixture_statistics(id_fix)

                    stats_objs = []
                    if len(statistics) > 0:
                        stats_objs = [
                            Statistics(
                                **map_statistic(
                                    match,
                                    statistic,
                                    'home' if statistic['team']['id'] == fixture['teams']['home']['id'] else 'away',
                                    id_fix,
                                    fixture,
                                )
                            )
                            for statistic in statistics
                        ]

                    fixture_bookmakers = provider.get_fixture_odds(id_fix)
                    odds_map = map_odds(match, id_fix, fixture_bookmakers=fixture_bookmakers)
                    odds_objs = [Odds(**odds_map)] if odds_map else None

                    # Bug fix 2026-09-15 (righe duplicate): la riga base viene
                    # scritta SUBITO con un upsert su `id_fixture`, che
                    # restituisce l'`id_match_fk` definitivo - quello di chi ha
                    # vinto la corsa, se due import girano sovrapposti. PRIMA
                    # le righe nuove restavano in `list_matches` fino alla fine
                    # del job (minuti), invisibili a chiunque altro: due giri
                    # paralleli inserivano due volte la stessa partita e una
                    # delle due copie non veniva piu' aggiornata, restando a
                    # `NS` per sempre (vedi `MatchRepository.upsert_base_by_fixture`).
                    # Va fatto PRIMA di `map_odds_snapshots`, che deve agganciare
                    # gli snapshot all'id definitivo e non a un uuid appena
                    # generato che potrebbe non finire mai a DB.
                    id_match_fk, inserita_ora = repo_match.upsert_base_by_fixture(dict_match)
                    dict_match['id_match_fk'] = id_match_fk

                    snapshots = map_odds_snapshots(
                        id_match=id_match_fk,
                        id_fixture=id_fix,
                        fixture_bookmakers=fixture_bookmakers,
                    )
                    snapshot_buffer.extend(snapshots)

                    dict_match['statistics'] = stats_objs
                    if odds_objs:
                        dict_match['odds'] = odds_objs

                    # L'upsert ha già scritto le colonne base; questo `save`
                    # (merge) serve alle RELAZIONI statistics/odds, che con
                    # `cascade="all, delete-orphan"` vengono sostituite da
                    # quelle appena mappate - esattamente come prima.
                    repo_match.save(Match(**dict_match))
                    if inserita_ora:
                        report['inserted'] += 1
                    else:
                        report['updated'] += 1
                except ApiSportsQuotaExceededError as quota_exc:
                    # BUGFIX 2026-09-06: PRIMA questa eccezione veniva
                    # inghiottita dal blocco `except Exception` generico
                    # sotto (ApiSportsQuotaExceededError e' una RuntimeError,
                    # quindi la ereditava) - il loop CONTINUAVA a tentare
                    # OGNI fixture rimanente (fino a centinaia, es. nella
                    # finestra "prossimi 7 giorni" di "Aggiorna tutto"),
                    # ognuna fallendo allo stesso identico modo, MA senza mai
                    # impostare `quota_exceeded=True` (settato solo se
                    # l'eccezione arriva da `get_fixtures` sopra): il job
                    # restava "in esecuzione" per decine di minuti/ore
                    # inutilmente E il banner "quota esaurita" in UI non
                    # compariva mai. Ora ci fermiamo IMMEDIATAMENTE, qui,
                    # come gia' avviene per l'eccezione sollevata da
                    # `get_fixtures`.
                    logging.error(
                        '<<< API-Sports quota esaurita durante processing fixture %s, stop import: %s >>>',
                        id_fix,
                        quota_exc,
                    )
                    report['quota_exceeded'] = True
                    report['errors'].append({
                        'fixture_id': id_fix,
                        'error': f'API-Sports quota esaurita (fixture={id_fix}): {quota_exc}',
                    })
                    break
                except Exception as fixture_error:
                    report['failed'] += 1
                    report['errors'].append({'fixture_id': id_fix, 'error': str(fixture_error)})
                    continue

            if report["quota_exceeded"]:
                # Il `break` sopra esce solo dal loop `for fixture in
                # fixtures` - usciamo anche da quello corrente `for league`,
                # il loop `for season` lo ricontrollera' alla prossima
                # iterazione (vedi cima del metodo).
                break

    # 2026-09-15: qui c'era il `repo_match.save_all(list_matches)` finale, con
    # dump su `error_save_dict.json` se il bulk insert saltava. Non serve piu':
    # ogni fixture e' ora persistita nella propria iterazione (upsert +
    # merge), quindi non esiste piu' un bulk "tutto o niente" da recuperare a
    # posteriori e un fallimento resta isolato alla singola fixture, contato
    # in `report['failed']` con il dettaglio in `report['errors']` - una
    # granularita' migliore di quella che dava il file. Vedi la nota in
    # `re_processor_error`, che quel file lo leggeva.

    try:
        repo_snapshot.save_many(snapshot_buffer)
        report['snapshots_upserted'] = len(snapshot_buffer)
    except Exception as snapshot_error:
        logging.error('Errore salvataggio odds_snapshot: %s', snapshot_error)
        report['errors'].append({'fixture_id': None, 'error': f'odds_snapshot: {snapshot_error}'})

    if report['errors']:
        report['errors'] = report['errors'][:100]

    # BUGFIX 2026-09-06: `repo_match` (MatchRepository/CrudRepository) usa
    # ora una Session "scoped" per-thread tenuta APERTA per tutta la durata
    # di questo job (vedi crud_repository.py - non piu' chiusa ad ogni
    # singola query, per evitare centinaia di round-trip di rete verso il
    # DB remoto). Va quindi ripulita esplicitamente qui, a fine job: i
    # thread di APScheduler/FastAPI BackgroundTasks vengono RICICLATI per
    # lavori successivi, senza questa `remove()` la prossima esecuzione
    # sullo stesso thread riuserebbe (o troverebbe ancora "aperta") la
    # sessione/transazione di QUESTO job.
    SessionLocal.remove()

    return report


def re_processor_error():
    """
    Riprocessa il file di errori avvenuto durante il download
    :return: prova a salvare tutto a db

    NOTA (2026-09-15): `download_import_matches` non produce piu'
    `error_save_dict.json`. Quel file era il recupero di un bulk insert
    "tutto o niente" a fine job, che non esiste piu' (ogni fixture viene
    persistita singolarmente con upsert+merge, e un fallimento finisce in
    `report['errors']`). La funzione resta invocabile a mano per riprocessare
    un file prodotto da un'esecuzione PRECEDENTE alla modifica, se ne avete
    ancora uno da recuperare.
    """
    # Lettura da file JSON
    with open("error_save_dict.json", "r", encoding="utf-8") as f:
        dict_error = json.load(f)

    for element in dict_error:
        stat = element['statistics']
        odd = element['odds']
        element.pop('statistics')
        element.pop('odds')
        match = Match(**dict(element))
        match.statistics = [Statistics(**dict(s)) for s in stat]
        match.odds = [Odds(**dict(odd[0]))]
        repo_match.save(match)


def calculate_mean(with_season: int = None, force_mean: bool = False, teams: list = None):
    """
    Calcola le medie stagionali di ogni squadra prima del match corrente o in maniera puntuale/massiva
    :param with_season: int -> se valorizzata, calcola solo la stagione indicata
    :param force_mean: forza il calcolo della media ANCHE per match che hanno già la media persistita
    :param teams: Un array di id teams -> se valorizzato, il ricalcolo resta limitato ESATTAMENTE
        a queste squadre (bug fix 2026-09-07: prima veniva esteso anche agli avversari incontrati,
        sovrascrivendone la media con un calcolo parziale, vedi commento piu' sotto)
    :return: Persist in update massive (bulk)
    """
    columns_mean = ['Shots on Goal', 'Shots off Goal', 'Total Shots', 'Blocked Shots', 'Shots insidebox',
                    'expected_goals', 'goals_prevented',
                    'Shots outsidebox', 'Fouls',
                    'Corner Kicks', 'Offsides', 'Ball Possession', 'Yellow Cards', 'Red Cards',
                    'Goalkeeper Saves',
                    'Total passes', 'Passes accurate', 'Passes %']

    # Ricerca per id_fixture valorizzati
    filters = {
        'id_fixture': "not None",
        # 'current_league': 138
    }
    if with_season:
        filters.update({'season': with_season})
    if teams:
        filters.update({"OR": [
            ("id_team_home", teams),
            ("id_team_away", teams)
        ]})

    all_match = repo_match.search_filter(filters=filters)
    seasons = set(match.season for match in all_match if pd.notna(match.season))
    id_teams_home = [match.id_team_home for match in all_match if pd.notna(match.id_team_home)]
    id_teams_away = [match.id_team_away for match in all_match if pd.notna(match.id_team_away)]
    id_teams = set(list(id_teams_home) + list(id_teams_away))
    if teams:
        # Bug fix 2026-09-07 (era il TODO sopra): il filtro OR usato per
        # costruire `all_match` cattura le partite di OGNI squadra che ha
        # incontrato una squadra in `teams` (sia come home che away), quindi
        # `id_teams` finiva per includere anche gli AVVERSARI non richiesti.
        # Per una squadra esplicitamente in `teams`, `all_match` contiene
        # GIA' tutte le sue partite stagionali (ogni sua partita ha
        # home=team O away=team, catturata dal filtro OR sopra) - ma per un
        # avversario aggiunto solo perche' ha giocato CONTRO una squadra in
        # `teams`, `all_match` contiene SOLO le partite contro quelle
        # squadre, non l'intero storico stagionale. Ricalcolare la sua
        # media su questo sottoinsieme parziale la sovrascriveva
        # (`force_mean`/prima media) con un valore SBAGLIATO. Fix: quando
        # `teams` e' specificato, il ricalcolo resta limitato ESATTAMENTE
        # alle squadre richieste.
        id_teams = id_teams & set(teams)
    list_obj = []  # Update in maniera massiva con bulk_update_mappings
    for season in seasons:
        logging.info(f'<<< Start season {season} >>>')
        for id_team in id_teams:
            logging.info(f'<<< Start id_team {id_team} >>>')
            # Filtra partite della squadra in quella stagione
            team_matches = [rec for rec in all_match if
                            (rec.id_team_home == id_team or rec.id_team_away == id_team) and rec.season == season]
            if len(team_matches) > 0:
                # Ordina per data
                team_matches = sorted(team_matches, key=lambda x: x.date_match)
                logging.info(f'<<< Total row match {len(team_matches)} for id_team {id_team} >>>')

                for idx, match in enumerate(team_matches):
                    # Dalla seconda partita stagionale in poi
                    # Se il match NON ha ancora le medie calcolate O la forzatura per il calcolo è True
                    if idx > 0 and (match.mean_statistics is None or force_mean):
                        prev_matches = team_matches[:idx]  # Recupera tutti gli elementi precedenti
                        stat_prev = [s for stat in prev_matches for s in stat.statistics if
                                     s.statistics_team_id == id_team]
                        if len(stat_prev) > 0:
                            mean_rows = {}

                            def create_dict_stat_prev():
                                array_prev = []

                                def get_value(val, des_val=None):
                                    real_attr = getattr(s, val)
                                    if des_val:
                                        raw = real_attr.get(des_val) if real_attr else None
                                    else:
                                        raw = real_attr
                                    # BUGFIX 2026-09-05: l'API Sports a volte restituisce valori
                                    # numerici come stringa (es. "goals_prevented": "-0.45") senza
                                    # il suffisso '%' che `get_attribute_statistics` sa gestire.
                                    # Senza una coercizione robusta QUI (unico punto di accesso ai
                                    # valori), `statistics.mean()` piu' sotto crasha con
                                    # "can't convert type 'str' to numerator/denominator" non
                                    # appena incontra uno di questi valori - mai piu' un campo
                                    # "dimenticato" (come accaduto con 'goals_prevented', che a
                                    # differenza di 'expected_goals' non aveva un float() esplicito).
                                    if raw is None or raw == '':
                                        return 0.0
                                    if isinstance(raw, str):
                                        # Dati legacy pre-esistenti a DB possono ancora avere il
                                        # suffisso '%' non ripulito (es. "Ball Possession": "58%").
                                        raw = raw.strip().rstrip('%')
                                    try:
                                        return float(raw)
                                    except (TypeError, ValueError):
                                        return 0.0

                                for s in stat_prev:
                                    array_prev.append({
                                        'Shots on Goal': get_value('shots', 'Shots on Goal'),
                                        'Shots off Goal': get_value('shots', 'Shots off Goal'),
                                        'Total Shots': get_value('shots', 'Total Shots'),
                                        'Blocked Shots': get_value('shots', 'Blocked Shots'),
                                        'Shots insidebox': get_value('shots', 'Shots insidebox'),
                                        'Shots outsidebox': get_value('shots', 'Shots outsidebox'),

                                        'expected_goals': get_value('generic_statistics', 'expected_goals'),
                                        # quanto la squadra avrebbe dovuto segnare.
                                        'goals_prevented': get_value('generic_statistics', 'goals_prevented'),
                                        # quanto il portiere ha inciso nel prevenire (o subire) gol rispetto alle attese.

                                        'Fouls': get_value('fouls'),
                                        'Corner Kicks': get_value('corners'),
                                        'Offsides': get_value('offside'),
                                        'Ball Possession': get_value('bass_possession'),
                                        'Yellow Cards': get_value('yellow_cards'),
                                        'Red Cards': get_value('red_cards'),
                                        'Goalkeeper Saves': get_value('goal_keeper'),

                                        'Total passes': get_value('passes', 'Total passes'),
                                        'Passes accurate': get_value('passes', 'Passes accurate'),
                                        'Passes %': get_value('passes', 'Passes %')
                                    })

                                return array_prev

                            mean_rows.update({
                                f'mean_{key}': mean(m[key] for m in create_dict_stat_prev())
                                for key in columns_mean
                            })
                            mean_rows.update({'id_team': id_team})

                            # TODO : capire se è un caso possibile
                            # if match.mean_statistics:  # Se il match ha già medie di una delle 2 squadre eseguito
                            #     # Aggiunge la seconda squadra per evitare che cancelli il precedente calcolo
                            #     mean_rows.update(match.mean_statistics)

                            # Controlla se esiste già un elemento con la stessa partita
                            check_l_obj = [l_o for l_o in list_obj if l_o.get('id_match_fk') == match.id_match_fk]
                            # Se esiste nella lista che sto per creare già lo stesso id partita, devo solo aggiornare array con quel dizionario
                            if len(check_l_obj) > 0:
                                old_mean = check_l_obj[0]['mean_statistics']
                                total_mean_match = [old_mean, mean_rows]
                                check_l_obj[0].update(
                                    {'id_match_fk': match.id_match_fk, 'mean_statistics': total_mean_match})
                            else:
                                # Creo le righe che verranno poi aggiornate massivamente
                                list_obj.append({'id_match_fk': match.id_match_fk, 'mean_statistics': mean_rows})
    # Salvataggio massivo
    repo_match.massive_update_bulk(list_obj)


def reload_fixture():
    """
    Ricarica tutte le fixture a db per sistemare eventuali errori
    :return:
    """
    filters = {
        'statistics': "not None",
        # 'id_fixture': "not None",
        # 'status': ['NS'],
        # 'current_league': [2, 3, 848],
        'season': [2020, 2021, 2022, 2023, 2024, 2025]
    }
    all_match = repo_match.search_filter(filters=filters)
    for index, match in enumerate(all_match):
        logging.info(f'<<< Started Match {index}/{len(all_match)} id_fixture {match.id_fixture} >>>')
        manual = False
        for stat in match.statistics:
            shots = stat.shots
            if ((shots.get('Total Shots') is None or shots.get('Total Shots') == 0) and (
                    shots.get('Shots on Goal') > 0 or shots.get('Shots off Goal') > 0)
                    and shots.get('manual_calculate') is not True):
                s_o_goal = shots.get('Shots on Goal') or 0
                s_of_goal = shots.get('Shots off Goal') or 0
                b_shots = shots.get('Blocked Shots') or 0
                total_shots = s_o_goal + s_of_goal + b_shots
                shots['Total Shots'] = total_shots
                manual = True
                shots['manual_calculate'] = manual
        if manual:
            repo_match.save(match)
        logging.info(f'<<< Ended Match {index}/{len(all_match)} and save id_fixture {match.id_fixture} >>>')
        # id_fix = match.id_fixture
        # fixture = base_api_statistics(path='fixtures', params={'id': id_fix})
        # if len(fixture) > 0:
        #     fixture = fixture[0]
        #     dict_match = map_base_match(match=match, id_fix=id_fix, fixture=fixture,
        #                                 league=match.current_league, season=match.season)
        #     new_match = Match(**dict_match)
        #     repo_match.save(new_match)


if __name__ == "__main__":
    download_import_matches(is_next=False)
    # re_processor_error()
    # reload_fixture()
    # calculate_mean(force_mean=True, with_season=2025)
