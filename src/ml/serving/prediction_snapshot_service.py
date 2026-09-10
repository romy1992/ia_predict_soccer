from __future__ import annotations

import hashlib
import os
from typing import Any, Optional

import numpy as np
import pandas as pd

from src.repository.match_prediction_snapshot_repository import MatchPredictionSnapshotRepository
from src.service_ia.model.match import MatchPredictionSnapshot
from src.service_ia.training.market_service.filter_market_service import FilterMarketService
from src.service_ia.training.model_registry import ModelRegistry

# Stesso insieme di `dashboard_service.FINAL_STATUSES` (fonte di verita' per
# "questa fixture e' conclusa, i suoi dati non cambieranno piu'") - non
# importato da li' per evitare una dipendenza a ritroso di questo package
# (src/ml/serving, usato ANCHE dai job schedulati, src/jobs/) verso il
# layer API (src/api/) - stesso principio di isolamento gia' seguito dalle
# altre 4 copie dello stesso insieme sparse nel progetto (mai state
# consolidate finora, non e' nello scope di questo task farlo).
_FINAL_STATUSES = {"FT", "AET", "PEN", "ABD", "CANC", "PST", "WO"}

_FINGERPRINT_FLOAT_PRECISION = 6


def compute_feature_fingerprint(X: pd.DataFrame, precision: int = _FINGERPRINT_FLOAT_PRECISION) -> str:
    """Hash deterministico del singolo vettore di feature effettivamente
    dato in pasto al modello (`X`, una riga) - usato da
    `PredictionSnapshotService` per decidere se una predizione salvata va
    ancora bene o va ricalcolata: se il fingerprint e' invariato, NESSUNA
    feature e' cambiata (quote, mean_statistics, o qualunque fonte futura
    aggiunta in seguito - generico per costruzione, non lega
    l'invalidazione a colonne specifiche), quindi la predizione salvata
    resta valida.

    I valori numerici sono convertiti esplicitamente a `float` Python
    (mai lasciati come scalari numpy, il cui `repr()` e' cambiato tra
    versioni di numpy) e arrotondati a `precision` decimali PRIMA
    dell'hash, per non invalidare per rumore numerico irrilevante (es.
    una minima differenza di virgola mobile tra due ricalcoli della
    stessa media, a parita' di dati sorgente)."""
    if X is None or X.empty:
        return "empty"

    row = X.iloc[0]
    parts = []
    for column in sorted(row.index):
        value = row[column]
        if pd.isna(value):
            normalized = "nan"
        else:
            try:
                normalized = repr(round(float(value), precision))
            except (TypeError, ValueError):
                normalized = repr(value)
        parts.append(f"{column}={normalized}")

    canonical = "|".join(parts)
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


class PredictionSnapshotService:
    """Risolve la predizione ML per fixture+mercato riusando una riga gia'
    calcolata quando possibile, invece di ricaricare il modello e rifare
    l'inferenza ad ogni chiamata (2026-09-09, richiesto esplicitamente
    dall'operatore: "salvare le predizioni... per le partite di oggi o
    future, solo se cambia una delle feature").

    Due regimi, decisi da `status`:
    - **Partita CONCLUSA** (`status` in `_FINAL_STATUSES`): quote e
      statistiche sono ormai definitive - se esiste gia' una riga salvata
      per fixture+mercato, viene servita DIRETTAMENTE, senza nemmeno
      costruire il frame di feature (zero query extra, zero inferenza).
      Se non esiste ancora (es. mercato promosso dopo la fine di quella
      partita), viene calcolata UNA VOLTA e salvata per sempre - una
      partita conclusa non genera MAI una seconda riga, nemmeno se in
      futuro viene promosso un modello diverso (scelta esplicita
      dell'operatore: la riga storica rappresenta "cosa prediceva il
      modello in quel momento", congelata di proposito).
    - **Partita NON conclusa** (NS/live): il frame di feature viene
      comunque costruito (economico: query + calcolo pandas, nessun
      modello coinvolto) e se ne calcola un fingerprint deterministico
      (`compute_feature_fingerprint`); la riga salvata viene riusata SOLO
      se il fingerprint E il modello in produzione (`model_run_id`) sono
      IDENTICI all'ultimo calcolo - altrimenti si ricalcola (carica il
      modello, esegue l'inferenza) e si appende una riga nuova.

    Il modello caricato (`joblib.load`, potenzialmente decine/centinaia di
    MB) e' cache a LIVELLO DI CLASSE (`_model_cache`, sopravvive tra le
    istanze create ad ogni richiesta Dashboard - stesso principio gia'
    applicato a `DashboardService._api_cache`): senza questo, il costo di
    caricamento da disco si ripeteva ad OGNI singola richiesta, causa
    primaria della lentezza percepita indipendentemente dalla data
    (diagnosticato 2026-09-09)."""

    _model_cache: dict[str, Any] = {}

    def __init__(self):
        self.registry = ModelRegistry()
        self.filter_service = FilterMarketService()
        self.repo = MatchPredictionSnapshotRepository()

    @classmethod
    def clear_model_cache(cls) -> None:
        """SOLO per i test: azzera la cache di classe tra un caso e
        l'altro, cosi' un `model_path` riusato in test diversi con oggetti
        finti diversi non serve mai un valore residuo del test precedente."""
        cls._model_cache.clear()

    @classmethod
    def _load_model(cls, model_path: str):
        if model_path not in cls._model_cache:
            import joblib

            cls._model_cache[model_path] = joblib.load(model_path)
        return cls._model_cache[model_path]

    def _latest_model_for_market(self, market: str) -> Optional[dict[str, Any]]:
        # Letto FRESCO ad ogni chiamata (mai cache a livello di classe qui):
        # e' un file JSONL piccolo, il costo e' trascurabile rispetto al
        # caricamento del modello, e deve riflettere una promozione nuova
        # immediatamente (mai un'istanza "congelata" sulla production di
        # ieri).
        return self.registry.get_production(market=market) or self.registry.get_latest(market=market) or None

    @staticmethod
    def _extract_probability(model: Any, X: pd.DataFrame) -> tuple[int, float]:
        if hasattr(model, "predict_proba"):
            probs = model.predict_proba(X)
            first = np.asarray(probs[0], dtype=float)
            if first.size >= 2:
                p1 = float(first[-1])
                return int(p1 >= 0.5), p1
            if first.size == 1:
                p1 = float(first[0])
                return int(p1 >= 0.5), p1

        pred_raw = model.predict(X)
        pred = int(np.asarray(pred_raw).ravel()[0])
        return pred, float(pred)

    @staticmethod
    def _entry_from_snapshot(snapshot: MatchPredictionSnapshot) -> dict[str, Any]:
        return {
            "prediction": int(snapshot.prediction),
            "probability": float(snapshot.probability),
            "model_name": snapshot.model_name,
            "run_id": snapshot.model_run_id,
        }

    def resolve_predictions(
        self,
        fixture_id: int,
        markets: list[str],
        db_match: Any = None,
        status: Optional[str] = None,
        allow_compute: bool = True,
        force: bool = False,
    ) -> dict[str, dict[str, Any]]:
        """`allow_compute=False` (2026-09-10, richiesto esplicitamente
        dall'operatore: la vista storica della Dashboard deve restare
        SEMPRE veloce, mai "quasi sempre") disabilita qualunque calcolo
        nuovo - serve SOLO cio' che e' gia' salvato, un mercato senza riga
        semplicemente non compare nel payload invece di innescare
        caricamento modello + inferenza. Pensato per le liste di fixture
        storiche (molte fixture insieme, dove un singolo ricalcolo lento si
        moltiplica) - il dettaglio di una singola fixture (un click
        deliberato) resta sempre `allow_compute=True`.

        `force=True` (2026-09-10, punto 4/4 di
        `PROMPT_fast_historical_predictions.md`): ignora QUALUNQUE riga
        esistente - anche una partita conclusa "congelata" - e
        ricalcola+salva SEMPRE una riga nuova per ogni mercato richiesto.
        Uso ESPLICITO e manuale (bottone "Ricalcola previsione" nel
        dettaglio partita), MAI automatico - il regime "congelato" per le
        partite concluse resta l'unico comportamento di default. Ha
        priorita' su `allow_compute` (un `force=True` implica sempre il
        calcolo, indipendentemente dal valore di `allow_compute`)."""
        if not markets:
            return {}

        is_final = (status or "").upper() in _FINAL_STATUSES
        payload: dict[str, dict[str, Any]] = {}

        if force:
            markets_needing_compute = list(markets)
        elif is_final:
            markets_needing_compute = []
            for market in markets:
                snapshot = self.repo.get_latest(fixture_id=fixture_id, market=market)
                if snapshot is not None:
                    payload[market] = self._entry_from_snapshot(snapshot)
                elif allow_compute:
                    markets_needing_compute.append(market)
            if not markets_needing_compute:
                return payload
        elif not allow_compute:
            # Partita non conclusa ma il chiamante ha comunque chiesto di
            # non calcolare (caso raro/difensivo - una data storica non
            # dovrebbe mai avere fixture NS/live): serve solo cio' che e'
            # gia' salvato, mai un calcolo nuovo.
            for market in markets:
                snapshot = self.repo.get_latest(fixture_id=fixture_id, market=market)
                if snapshot is not None:
                    payload[market] = self._entry_from_snapshot(snapshot)
            return payload
        else:
            markets_needing_compute = list(markets)

        if db_match is not None:
            frames = self.filter_service.build_prediction_frames_from_match(db_match, markets_needing_compute)
        else:
            frames = self.filter_service.build_prediction_frames(fixture_id=fixture_id, markets=markets_needing_compute)

        for market in markets_needing_compute:
            model_meta = self._latest_model_for_market(market)
            if not model_meta:
                continue

            model_path = model_meta.get("model_path")
            if not model_path or not os.path.exists(model_path):
                continue

            frame = frames.get(market)
            if frame is None or frame.empty:
                continue

            X = frame.drop(columns=["market", "id_fixture", "season", "league", "prediction_at"], errors="ignore")
            selected_features = model_meta.get("feature_names") or []
            if selected_features:
                for feature_name in selected_features:
                    if feature_name not in X.columns:
                        X[feature_name] = 0.0
                X = X[selected_features]

            model_run_id = model_meta.get("run_id")
            fingerprint = compute_feature_fingerprint(X)

            if not is_final and not force:
                existing = self.repo.get_latest(fixture_id=fixture_id, market=market)
                if (
                    existing is not None
                    and existing.feature_fingerprint == fingerprint
                    and existing.model_run_id == model_run_id
                ):
                    payload[market] = self._entry_from_snapshot(existing)
                    continue

            try:
                model = self._load_model(model_path)
                prediction, probability = self._extract_probability(model=model, X=X)
            except Exception:
                continue

            entry = {
                "prediction": int(prediction),
                "probability": float(probability),
                "model_name": model_meta.get("model_name"),
                "run_id": model_run_id,
            }
            payload[market] = entry

            self.repo.save(
                MatchPredictionSnapshot(
                    fixture_id=int(fixture_id),
                    market=market,
                    prediction=entry["prediction"],
                    probability=entry["probability"],
                    model_name=entry["model_name"],
                    model_run_id=model_run_id,
                    feature_fingerprint=fingerprint,
                )
            )

        return payload
