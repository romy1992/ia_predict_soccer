"""Confronto nomi squadra e giocatore. Nessun ID FotMob viene trattato come ID API-Sports."""

from __future__ import annotations

import re
import unicodedata

_PUNCT = re.compile(r"[^a-z0-9\s]+")
_SPACE = re.compile(r"\s+")
# Token di compagnia, non di identita'. "united" e "city" restano: distinguono due squadre.
_GENERIC = frozenset({"fc", "cf", "ac", "sc", "afc", "as", "ss", "us", "rc", "cd", "club", "calcio", "the", "de"})

# Dopo la normalizzazione. Coppie osservate tra FotMob e API-Sports, non una similarita' libera.
_ALIASES = {
    "bayern munchen": "bayern munich",
    "fc bayern munchen": "bayern munich",
    "bayern munich": "bayern munich",
    "inter milan": "inter",
    "internazionale": "inter",
    "fc internazionale": "inter",
    "psg": "paris saint germain",
    "paris sg": "paris saint germain",
    "man utd": "manchester united",
    "man united": "manchester united",
    "manchester utd": "manchester united",
    "man city": "manchester city",
    "atletico de madrid": "atletico madrid",
    "club atletico de madrid": "atletico madrid",
    "sporting cp": "sporting",
    "sporting lisbon": "sporting",
    "sporting clube de portugal": "sporting",
}


def normalize_name(value: str | None) -> str:
    if not value:
        return ""
    text = unicodedata.normalize("NFKD", value)
    text = "".join(ch for ch in text if not unicodedata.combining(ch))
    text = text.lower().replace("&", " and ")
    text = _PUNCT.sub(" ", text)
    return _SPACE.sub(" ", text).strip()


def _identity(value: str | None) -> str:
    """Forma usata per il confronto: alias esplicito, poi token senza sigle di compagnia."""
    normalized = normalize_name(value)
    if not normalized:
        return ""
    aliased = _ALIASES.get(normalized, normalized)
    tokens = [token for token in aliased.split(" ") if token and token not in _GENERIC]
    folded = " ".join(tokens)
    return _ALIASES.get(folded, folded)


def names_match(left: str | None, right: str | None) -> bool:
    a, b = _identity(left), _identity(right)
    return bool(a) and a == b


def player_keys(name: str | None) -> tuple[str, str | None]:
    """Nome completo normalizzato e, se c'e' un nome di battesimo, `iniziale|cognome`."""
    normalized = normalize_name(name)
    tokens = [token for token in normalized.split(" ") if token not in {"jr", "junior"}]
    full = " ".join(tokens)
    if len(tokens) < 2:
        return full, None
    return full, f"{tokens[0][0]}|{tokens[-1]}"
