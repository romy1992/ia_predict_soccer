"""Anagrafica giocatori: sincronizzazione da API-Sports, collegamento di FotMob, ricerca e formazioni.

Regole:
- API-Sports e' il contenitore principale: ogni giocatore gia' visto in `player_lineup` ha una identita'.
- Un collegamento esistente si riusa senza rifare il confronto per nome. Un collegamento `manual` o
  `external_project` non viene mai sovrascritto da uno automatico.
- Un nome che corrisponde a piu' persone non si sceglie: si restituiscono i candidati.
- La data di nascita, quando c'e', esclude gli omonimi con data diversa e alza la fiducia.
"""

from __future__ import annotations

import logging
from datetime import datetime, timezone
from typing import Any, Iterable, Optional

from sqlalchemy import func, select
from sqlalchemy.orm import Session

from src.data.player.player_lineup_models import PlayerLineup
from src.service_ia.fotmob.names import name_tokens, normalize_name, player_keys, token_key
from src.service_ia.players.models import PlayerIdentity, PlayerProviderLink

logger = logging.getLogger(__name__)

API = "api_sports"
FOTMOB = "fotmob"
PROVIDERS = (API, FOTMOB)
PROTECTED_METHODS = frozenset({"manual", "external_project"})
METHOD_CONFIDENCE = {
    "provider_id": 1.0, "manual": 1.0, "external_project": 0.95, "exact_name": 0.9, "name_tokens": 0.85,
    "last_name_initial": 0.8, "surname_subset": 0.75, "official_elimination": 0.8, "recency_pick": 0.6,
}


def _now() -> datetime:
    return datetime.now(timezone.utc)


def _clean_date(value: Optional[str]) -> Optional[str]:
    if not value:
        return None
    text = str(value).strip()[:10]
    try:
        datetime.strptime(text, "%Y-%m-%d")
    except ValueError:
        return None
    return text


def get_link(session: Session, provider: str, provider_id: Any) -> Optional[PlayerProviderLink]:
    return session.execute(
        select(PlayerProviderLink).where(
            PlayerProviderLink.provider == provider, PlayerProviderLink.provider_player_id == str(provider_id))
    ).scalars().first()


def _links_of(session: Session, identity_id: int) -> list[PlayerProviderLink]:
    return list(session.execute(
        select(PlayerProviderLink).where(PlayerProviderLink.identity_id == identity_id)
    ).scalars())


def identity_payload(session: Session, identity: PlayerIdentity) -> dict[str, Any]:
    links = _links_of(session, int(identity.id))
    return {
        "identity_id": int(identity.id), "name": identity.canonical_name, "birth_date": identity.birth_date,
        "position": identity.position, "team_api_id": identity.team_api_id,
        "providers": {
            link.provider: {
                "id": link.provider_player_id, "name": link.provider_name, "team": link.provider_team,
                "method": link.method, "confidence": link.confidence, "source": link.source,
            } for link in links
        },
    }


# --- sincronizzazione da API-Sports -------------------------------------------------------------------------

def sync_api_sports_identities(session: Session) -> dict[str, int]:
    """Crea l'identita' di ogni giocatore API-Sports presente in `player_lineup` (idempotente; nessuna chiamata API)."""
    latest = (select(PlayerLineup.player_id, func.max(PlayerLineup.fixture_id).label("fixture_id"))
              .group_by(PlayerLineup.player_id).subquery())
    rows = session.execute(
        select(PlayerLineup.player_id, PlayerLineup.player_name, PlayerLineup.team_id, PlayerLineup.team_name,
               PlayerLineup.position)
        .join(latest, (PlayerLineup.player_id == latest.c.player_id) & (PlayerLineup.fixture_id == latest.c.fixture_id))
    ).all()
    known = {
        int(pid): int(link_id) for pid, link_id in session.execute(
            select(PlayerProviderLink.provider_player_id, PlayerProviderLink.id).where(PlayerProviderLink.provider == API)
        ).all() if str(pid).isdigit()
    }
    created = updated = 0
    for player_id, name, team_id, team_name, position in rows:
        if player_id is None or not name:
            continue
        if int(player_id) in known:
            link = get_link(session, API, player_id)
            identity = session.get(PlayerIdentity, link.identity_id) if link is not None else None
            if identity is not None and (identity.team_api_id != team_id or (position and identity.position != position)):
                identity.team_api_id = team_id
                identity.position = position if position in {"G", "D", "M", "F"} else identity.position
                link.provider_team, link.last_seen = team_name, _now()
                updated += 1
            continue
        ensure_api_identity(session, int(player_id), name, team_id=team_id, team_name=team_name, position=position)
        created += 1
    session.commit()
    return {"identities_created": created, "identities_updated": updated, "api_players_seen": len(rows)}


def ensure_api_identity(session: Session, api_id: int, name: str, *, team_id: Optional[int] = None,
                        team_name: Optional[str] = None, position: Optional[str] = None,
                        birth_date: Optional[str] = None) -> PlayerIdentity:
    link = get_link(session, API, api_id)
    if link is not None:
        identity = session.get(PlayerIdentity, link.identity_id)
        if identity is not None:
            if birth_date and not identity.birth_date:
                identity.birth_date = _clean_date(birth_date)
            return identity
    identity = PlayerIdentity(
        canonical_name=name, birth_date=_clean_date(birth_date), team_api_id=team_id,
        position=position if position in {"G", "D", "M", "F"} else None)
    session.add(identity)
    session.flush()
    session.add(PlayerProviderLink(
        identity_id=int(identity.id), provider=API, provider_player_id=str(api_id), provider_name=name,
        name_norm=normalize_name(name), provider_team=team_name, method="provider_id", confidence=1.0,
        source="lineup_history"))
    session.flush()
    return identity


# --- collegamenti ----------------------------------------------------------------------------------------------

def link_provider(session: Session, *, identity_id: int, provider: str, provider_id: Any, name: Optional[str],
                  method: str, source: str, team_name: Optional[str] = None,
                  confidence: Optional[float] = None) -> tuple[Optional[PlayerProviderLink], str]:
    """Collega `provider_id` all'identita'. Stati: created | unchanged | updated | protected | conflict."""
    if provider not in PROVIDERS:
        raise ValueError(f"provider sconosciuto: {provider}")
    conf = METHOD_CONFIDENCE.get(method, 0.7) if confidence is None else confidence
    existing = get_link(session, provider, provider_id)
    if existing is not None:
        if existing.identity_id == identity_id:
            existing.last_seen = _now()
            if name and not existing.provider_name:
                existing.provider_name, existing.name_norm = name, normalize_name(name)
            return existing, "unchanged"
        if existing.method in PROTECTED_METHODS and method not in PROTECTED_METHODS:
            return existing, "protected"
        if conf <= existing.confidence and method not in PROTECTED_METHODS:
            return existing, "conflict"
        existing.identity_id, existing.method, existing.confidence, existing.source = identity_id, method, conf, source
        existing.last_seen = _now()
        return existing, "updated"
    # un'identita' ha al massimo un collegamento per provider: se ne ha gia' uno diverso, non si sovrascrive a caso
    same_provider = [link for link in _links_of(session, identity_id) if link.provider == provider]
    if same_provider:
        current = same_provider[0]
        if current.method in PROTECTED_METHODS and method not in PROTECTED_METHODS:
            return current, "protected"
        if conf <= current.confidence and method not in PROTECTED_METHODS:
            return current, "conflict"
        session.delete(current)
        session.flush()
    link = PlayerProviderLink(
        identity_id=identity_id, provider=provider, provider_player_id=str(provider_id), provider_name=name,
        name_norm=normalize_name(name), provider_team=team_name, method=method, confidence=conf, source=source)
    session.add(link)
    session.flush()
    return link, "created"


def link_fotmob_to_api(session: Session, *, fotmob_id: Any, fotmob_name: Optional[str], api_player_id: int,
                       method: str, source: str = "fotmob_sync", team_name: Optional[str] = None) -> str:
    """Salva il collegamento FotMob -> API-Sports gia' trovato (per nome o per eliminazione)."""
    if fotmob_id is None:
        return "no_fotmob_id"
    identity = session.get(PlayerIdentity, get_link(session, API, api_player_id).identity_id) \
        if get_link(session, API, api_player_id) is not None else None
    if identity is None:
        identity = ensure_api_identity(session, int(api_player_id), fotmob_name or f"API {api_player_id}")
    _, status = link_provider(session, identity_id=int(identity.id), provider=FOTMOB, provider_id=fotmob_id,
                              name=fotmob_name, method=method, source=source, team_name=team_name)
    return status


def api_id_for_fotmob(session: Session, fotmob_id: Any) -> Optional[int]:
    """ID API-Sports gia' noto per un giocatore FotMob (collegamento salvato), oppure None."""
    if fotmob_id is None:
        return None
    link = get_link(session, FOTMOB, fotmob_id)
    if link is None:
        return None
    api = next((item for item in _links_of(session, int(link.identity_id)) if item.provider == API), None)
    return int(api.provider_player_id) if api is not None and str(api.provider_player_id).isdigit() else None


# --- ricerca ----------------------------------------------------------------------------------------------------

def _name_method(query: str, candidate: Optional[str]) -> Optional[str]:
    """Come `query` corrisponde a `candidate` (dal piu' forte al piu' debole), oppure None."""
    q_full, q_init = player_keys(query)
    c_full, c_init = player_keys(candidate)
    if q_full and q_full == c_full:
        return "exact_name"
    if token_key(query) and token_key(query) == token_key(candidate):
        return "name_tokens"
    if q_init and q_init == c_init:
        return "last_name_initial"
    qt, ct = name_tokens(query), name_tokens(candidate)
    if len(qt) >= 2 and len(ct) >= 2 and qt[0][0] == ct[0][0]:
        qs, cs = set(qt[1:]), set(ct[1:])
        if qs <= cs or cs <= qs:
            return "surname_subset"
    return None


_RANK = {"exact_name": 3, "name_tokens": 2, "last_name_initial": 1, "surname_subset": 0}


def resolve(session: Session, *, provider: Optional[str] = None, provider_id: Any = None, name: Optional[str] = None,
            team_id: Optional[int] = None, birth_date: Optional[str] = None) -> dict[str, Any]:
    """Trova l'identita': per ID del provider (certo) oppure per nome (+ squadra e data di nascita se ci sono).

    Esito: {"status": "found"|"ambiguous"|"not_found", "match": ..., "candidates": [...]}."""
    if provider and provider_id is not None:
        link = get_link(session, provider, provider_id)
        if link is not None:
            identity = session.get(PlayerIdentity, link.identity_id)
            if identity is not None:
                return {"status": "found", "method": "provider_id", "confidence": 1.0,
                        "match": identity_payload(session, identity), "candidates": []}
    if not name:
        return {"status": "not_found", "candidates": []}

    tokens = [token for token in name_tokens(name) if len(token) > 2] or list(name_tokens(name))
    if not tokens:
        return {"status": "not_found", "candidates": []}
    needle = max(tokens, key=len)
    query = select(PlayerProviderLink, PlayerIdentity).join(
        PlayerIdentity, PlayerIdentity.id == PlayerProviderLink.identity_id
    ).where(PlayerProviderLink.name_norm.like(f"%{needle}%"))
    if team_id is not None:
        query = query.where((PlayerIdentity.team_api_id == team_id) | (PlayerIdentity.team_api_id.is_(None)))
    wanted_birth = _clean_date(birth_date)
    scored: dict[int, tuple[int, str, PlayerIdentity]] = {}
    for link, identity in session.execute(query).all():
        if wanted_birth and identity.birth_date and identity.birth_date != wanted_birth:
            continue  # stessa sigla, persona diversa
        method = _name_method(name, link.provider_name)
        if method is None:
            continue
        rank = _RANK[method] + (4 if wanted_birth and identity.birth_date == wanted_birth else 0)
        best = scored.get(int(identity.id))
        if best is None or rank > best[0]:
            scored[int(identity.id)] = (rank, method, identity)
    if not scored:
        return {"status": "not_found", "candidates": []}
    ranked = sorted(scored.values(), key=lambda item: item[0], reverse=True)
    top = [item for item in ranked if item[0] == ranked[0][0]]
    if len(top) != 1:
        return {"status": "ambiguous", "candidates": [identity_payload(session, item[2]) for item in top[:10]]}
    rank, method, identity = top[0]
    dob_match = bool(wanted_birth and identity.birth_date == wanted_birth)
    confidence = min(1.0, METHOD_CONFIDENCE.get(method, 0.7) + (0.1 if dob_match else 0.0))
    return {"status": "found", "method": method, "confidence": round(confidence, 2),
            "birth_date_confirmed": dob_match, "match": identity_payload(session, identity), "candidates": []}


def register_external_link(session: Session, *, api_sports_id: Optional[int] = None, fotmob_id: Optional[int] = None,
                           name: Optional[str] = None, birth_date: Optional[str] = None,
                           source: str = "external_project") -> dict[str, Any]:
    """Collegamento fornito a mano o da un altro progetto: ha la precedenza su quelli automatici."""
    if api_sports_id is None:
        raise ValueError("serve almeno api_sports_id")
    identity = ensure_api_identity(session, int(api_sports_id), name or f"API {api_sports_id}", birth_date=birth_date)
    if birth_date and not identity.birth_date:
        identity.birth_date = _clean_date(birth_date)
    status = "api_only"
    if fotmob_id is not None:
        _, status = link_provider(session, identity_id=int(identity.id), provider=FOTMOB, provider_id=fotmob_id,
                                  name=name, method="external_project" if source == "external_project" else "manual",
                                  source=source)
    session.commit()
    return {"status": status, "identity": identity_payload(session, identity)}
