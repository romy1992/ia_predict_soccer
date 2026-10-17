"""Anagrafica giocatori: ricerca, formazioni con gli ID dei due provider e collegamenti manuali.

Pensato anche per altri progetti: si chiamano con l'intestazione `X-Service-Key` (valore in `PLAYERS_API_KEY`),
senza il login del pannello. Vedi `service_key_ok` e il middleware in `main.py`."""

from __future__ import annotations

import hmac
import os
from typing import Optional

from fastapi import APIRouter, HTTPException, Query
from pydantic import BaseModel, Field

from src.repository.base.repository_db import SessionLocal
from src.service_ia.players import identity, lineups

router = APIRouter(prefix="/players", tags=["players"])


def service_key_ok(header_value: Optional[str]) -> bool:
    expected = os.environ.get("PLAYERS_API_KEY", "")
    return bool(expected) and bool(header_value) and hmac.compare_digest(header_value, expected)


class PlayerLinkRequest(BaseModel):
    api_sports_id: int = Field(..., description="ID API-Sports del giocatore (contenitore principale)")
    fotmob_id: Optional[int] = None
    name: Optional[str] = None
    birth_date: Optional[str] = Field(default=None, description="YYYY-MM-DD")
    source: str = Field(default="external_project", pattern="^(external_project|manual)$")


@router.get("/lookup")
def players_lookup(
    provider: Optional[str] = Query(default=None, pattern="^(api_sports|fotmob)$"),
    provider_id: Optional[str] = None,
    name: Optional[str] = None,
    team_id: Optional[int] = Query(default=None, description="ID squadra API-Sports"),
    birth_date: Optional[str] = Query(default=None, description="YYYY-MM-DD, esclude gli omonimi con data diversa"),
) -> dict:
    """Per ID del provider (certo) oppure per nome. Se il nome e' ambiguo restituisce i candidati, non ne sceglie uno."""
    if not provider_id and not name:
        raise HTTPException(status_code=400, detail="Serve provider+provider_id oppure name")
    with SessionLocal() as session:
        return identity.resolve(session, provider=provider, provider_id=provider_id, name=name,
                                team_id=team_id, birth_date=birth_date)


@router.get("/lineups/{fixture_id}")
def players_lineup(fixture_id: int, prefer: str = Query(default="official", pattern="^(official|predicted)$")) -> dict:
    """Formazione di una partita con gli ID API-Sports e FotMob: l'ufficiale API-Sports se e' completa, altrimenti la
    prevista FotMob (campo `lineup_type`)."""
    with SessionLocal() as session:
        return lineups.lineup_for_fixture(session, fixture_id, prefer=prefer)


@router.get("/lineups")
def players_lineups_bulk(
    fixture_ids: str = Query(..., description="ID API-Sports separati da virgola (massimo 50)"),
    prefer: str = Query(default="official", pattern="^(official|predicted)$"),
) -> dict:
    """Come `/players/lineups/{fixture_id}` per piu' partite in una sola chiamata."""
    try:
        wanted = [int(item) for item in fixture_ids.split(",") if item.strip()]
    except ValueError as exc:
        raise HTTPException(status_code=400, detail="fixture_ids deve contenere solo numeri") from exc
    if not wanted:
        raise HTTPException(status_code=400, detail="fixture_ids vuoto")
    with SessionLocal() as session:
        return lineups.lineups_for_fixtures(session, wanted, prefer=prefer)


@router.post("/links")
def players_link(payload: PlayerLinkRequest) -> dict:
    """Collegamento fornito a mano o da un altro progetto (con data di nascita): ha la precedenza sugli automatici."""
    with SessionLocal() as session:
        try:
            return identity.register_external_link(
                session, api_sports_id=payload.api_sports_id, fotmob_id=payload.fotmob_id, name=payload.name,
                birth_date=payload.birth_date, source=payload.source)
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc


class PlayerLinkItem(BaseModel):
    api_sports_id: int
    fotmob_id: Optional[int] = None
    name: Optional[str] = None
    birth_date: Optional[str] = None


class PlayerLinksBulkRequest(BaseModel):
    links: list[PlayerLinkItem] = Field(..., max_length=500)
    source: str = Field(default="external_project", pattern="^(external_project|manual)$")


@router.post("/links/bulk")
def players_links_bulk(payload: PlayerLinksBulkRequest) -> dict:
    """Fino a 500 collegamenti in una chiamata (stesse regole di `/players/links`)."""
    with SessionLocal() as session:
        return identity.register_external_links(session, [item.model_dump() for item in payload.links], source=payload.source)


@router.post("/sync")
def players_sync() -> dict:
    """Aggiorna l'anagrafica dai giocatori API-Sports gia' a DB e impara dalle formazioni ufficiali recenti."""
    with SessionLocal() as session:
        out = identity.sync_api_sports_identities(session)
        out["learned"] = lineups.learn_recent(session)
        return out
