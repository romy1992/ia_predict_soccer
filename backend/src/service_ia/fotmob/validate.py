"""Validita' strutturale e sportiva, tenute separate.

`rejected` nasce solo da controlli strutturali (conteggio, duplicati, modulo,
titolare anche indisponibile). L'appartenenza alla rosa non rende valido un
JSON rotto, e un JSON ben formato non diventa `validated` senza la rosa.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from src.service_ia.fotmob.mapping import PlayerMap
from src.service_ia.fotmob.parse import ParsedLineup, formation_valid

_ROLES = {"G", "D", "M", "F"}


@dataclass
class Assessment:
    reliability: str
    warnings: list[str] = field(default_factory=list)
    mapped_starters: int = 0
    starter_count: int = 0


def _side_structure(label: str, starters, unavailable, formation: str | None, warnings: list[str]) -> bool:
    rejected = False
    ids = [player.fotmob_id for player in starters]
    if len(starters) != 11:
        rejected = True
        warnings.append(f"{label}:titolari_{len(starters)}")
    known = [item for item in ids if item is not None]
    if len(known) != 11 or len(set(known)) != 11:
        rejected = True
        warnings.append(f"{label}:id_mancante_o_duplicato")
    absent = {player.fotmob_id for player in unavailable if player.fotmob_id is not None}
    if set(known) & absent:
        rejected = True
        warnings.append(f"{label}:titolare_anche_indisponibile")
    if not formation_valid(formation):
        rejected = True
        warnings.append(f"{label}:modulo_non_valido")
    for player in unavailable:
        if player.return_certainty == "uncertain":
            warnings.append(f"{label}:rientro_incerto:{player.name or player.fotmob_id}")
    return rejected


def assess(lineup: ParsedLineup, home_maps: list[PlayerMap], away_maps: list[PlayerMap], *,
           match_method: str, roster_checked: bool) -> Assessment:
    warnings: list[str] = []
    rejected = _side_structure("casa", lineup.home.starters, lineup.home.unavailable, lineup.home.formation, warnings)
    rejected = _side_structure("trasferta", lineup.away.starters, lineup.away.unavailable, lineup.away.formation, warnings) or rejected
    maps = list(home_maps) + list(away_maps)
    mapped = [item for item in maps if item.api_player_id is not None and item.warning != "roster_conflict"]
    api_ids = [item.api_player_id for item in mapped]
    if len(api_ids) != len(set(api_ids)):
        warnings.append("giocatore_api_duplicato")
    for item in maps:
        if item.warning == "ambiguous":
            warnings.append("nome_ambiguo")
        elif item.warning == "unmapped":
            warnings.append("giocatore_non_in_rosa")
        elif item.warning == "roster_conflict":
            warnings.append("giocatore_ultima_squadra_diversa")
        elif item.position not in _ROLES:
            warnings.append("ruolo_storico_mancante")
    if lineup.lineup_type == "unknown":
        warnings.append("tipo_formazione_non_riconosciuto")
    if match_method != "teams_and_kickoff":
        warnings.append(f"abbinamento_{match_method}")
    if not roster_checked:
        warnings.append("rosa_non_verificabile")

    starter_count = len(lineup.home.starters) + len(lineup.away.starters)
    mapped_count = len(mapped)
    structural_ok = not rejected and starter_count == 22
    identity_ok = match_method == "teams_and_kickoff" and lineup.lineup_type in {"predicted", "confirmed"}
    players_ok = (
        roster_checked
        and len(maps) == 22
        and mapped_count == 22
        and len(set(api_ids)) == 22
        and all(item.position in _ROLES for item in mapped)
        and not any(item.warning for item in maps)
    )
    if rejected:
        reliability = "rejected"
    elif not structural_ok or not roster_checked or lineup.lineup_type == "unknown":
        reliability = "unverified"
    elif identity_ok and players_ok:
        reliability = "validated"
    else:
        reliability = "partially_validated"
    # Dedup conservando l'ordine: i warning ripetuti (un per giocatore) restano contati una volta.
    unique: list[str] = []
    for warning in warnings:
        if warning not in unique:
            unique.append(warning)
    return Assessment(reliability=reliability, warnings=unique, mapped_starters=mapped_count, starter_count=starter_count)
