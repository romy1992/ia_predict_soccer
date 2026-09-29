"""Livello API per la sezione Model Diagnostics (2026-09-12) - aggiunge
SOLO caching TTL sopra `src.ml.evaluation.model_diagnostics_service` (che
resta la funzione pura, riusata anche dallo script CLI
`scripts/analysis/evaluate_champions_detailed.py`, mai duplicata).

Cache necessaria perche' il walk-forward OOF rifitta il modello registrato
UNA VOLTA PER FOLD (5 fold) per OGNI mercato richiesto - potenzialmente
lento su dataset di migliaia di righe, troppo per un ricalcolo ad ogni
singolo caricamento pagina (stesso principio gia' applicato a
`DashboardService._api_cache`)."""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Optional

from src.ml.evaluation.model_diagnostics_service import evaluate_markets_diagnostics, list_diagnosable_markets
from src.service_ia.training.model_registry import ModelRegistry


class ModelDiagnosticsService:
    # 15 minuti: il walk-forward OOF non deve mai essere invisibile
    # all'operatore per ore (i modelli/dati sottostanti cambiano nel
    # tempo), ma nemmeno rifatto ad ogni refresh della pagina - stesso
    # compromesso "dato quasi fresco invece di ricalcolo costante" gia'
    # scelto per `DashboardService._api_cache` (li' 60s, qui piu' lungo
    # perche' il costo per chiamata e' molto piu' alto).
    _cache: dict[str, tuple[datetime, dict[str, Any]]] = {}
    _cache_ttl_seconds = 900

    def __init__(self):
        self.registry = ModelRegistry()

    def get_diagnostics(self, markets: Optional[list[str]] = None, force_refresh: bool = False) -> dict[str, Any]:
        target_markets = markets if markets else list_diagnosable_markets(registry=self.registry)
        cache_key = ",".join(sorted(target_markets))

        if not force_refresh:
            cached = self._cache_get(cache_key)
            if cached is not None:
                return cached

        results = evaluate_markets_diagnostics(markets=target_markets, registry=self.registry)
        payload = {
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "markets": [r.to_dict() for r in results],
        }
        self._cache_set(cache_key, payload)
        return payload

    @classmethod
    def _cache_get(cls, key: str) -> Optional[dict[str, Any]]:
        item = cls._cache.get(key)
        if not item:
            return None
        ts, payload = item
        age = (datetime.now(timezone.utc) - ts).total_seconds()
        if age > cls._cache_ttl_seconds:
            return None
        return payload

    @classmethod
    def _cache_set(cls, key: str, payload: dict[str, Any]) -> None:
        cls._cache[key] = (datetime.now(timezone.utc), payload)

    @classmethod
    def clear_cache(cls) -> None:
        """SOLO per i test: azzera la cache tra un caso e l'altro."""
        cls._cache.clear()
