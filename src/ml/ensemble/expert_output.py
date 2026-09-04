"""Schema di output standardizzato per gli Oracle Experts (ORACLE-01).

Obiettivo (task ORACLE-01, Fase ENSEMBLE): tutti gli esperti gia' esistenti
(EXP-01 TeamStrengthExpert, EXP-02 GoalDistributionExpert, EXP-03
StatisticsExpert, EXP-04 MarketOddsExpert, EXP-05 DirectMarketExpert) e i
market expert con la stessa interfaccia (MARKET-05 CornersExpert, MARKET-06
CardsExpert) devono poter essere "consumati" da un meta-model senza che
quest'ultimo debba conoscere i dettagli interni di ciascuno di essi.

Questo modulo NON modifica nessuno degli esperti esistenti (zero rischio di
regressione sui loro test): si limita a definire lo schema comune di output.
Gli adapter che convertono l'output NATIVO di ciascun esperto in questo
schema sono in `src/ml/ensemble/adapters.py`.

Campi:
- `expert_name`: identificatore stabile dell'esperto/mercato (es.
  'team_strength', 'goal_distribution', 'h2h', 'cards_line_4_5').
- `expert_version`: versione/hash deterministico della logica dell'esperto,
  quando disponibile (EXP-01/02/04 espongono gia' `VERSION`); stringa vuota
  se non applicabile (es. EXP-03, EXP-05 versionano via ModelRegistry).
- `probability_vector`: dict outcome -> probabilita' in [0, 1]. VUOTO per gli
  esperti che non producono direttamente una probabilita' di esito (es.
  TeamStrengthExpert espone SOLO rating/feature riusabili da altri esperti):
  in quel caso il payload va consumato via `raw_output`/`metadata`.
- `model_run_id`: id del run nel ModelRegistry, quando l'esperto e' basato su
  un modello registrato/versionato su disco (EXP-05 e i market expert).
- `stage`: stage del modello nel ModelRegistry ('candidate'/'champion'/
  'production'/'retired'), quando applicabile.
- `feature_timestamp`: timestamp ISO8601 point-in-time delle feature usate
  per produrre l'output (es. 'prediction_at' o 'as_of'), quando disponibile.
- `confidence`: misura scalare in [0, 1] di quanto l'esperto e' "sicuro".
  Se non fornita esplicitamente e `probability_vector` non e' vuoto, viene
  calcolato un default (vedi `_default_confidence`).
- `metadata`: dizionario libero con dettagli aggiuntivi specifici
  dell'esperto (es. thresholds, line, dispersion bookmaker, lambda Poisson).
- `raw_output`: payload originale non trasformato, per debug/consumo
  avanzato da chi conosce gia' l'esperto specifico.
- `created_at`: timestamp ISO8601 di generazione di QUESTO ExpertOutput
  (non del modello: quello e' `feature_timestamp`/`model_run_id`).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Iterable, Optional

_PROBABILITY_TOLERANCE = 1e-9


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


@dataclass
class ExpertOutput:
    """Output comune per qualunque Oracle Expert o market expert (ORACLE-01)."""

    expert_name: str
    expert_version: str = ""
    probability_vector: dict[str, float] = field(default_factory=dict)
    model_run_id: Optional[str] = None
    stage: Optional[str] = None
    feature_timestamp: Optional[str] = None
    confidence: Optional[float] = None
    metadata: dict[str, Any] = field(default_factory=dict)
    raw_output: Any = None
    created_at: str = field(default_factory=_utc_now_iso)

    def __post_init__(self) -> None:
        if not str(self.expert_name or "").strip():
            raise ValueError("expert_name e' obbligatorio")

        normalized: dict[str, float] = {}
        for outcome, value in self.probability_vector.items():
            if value is None:
                continue
            numeric = float(value)
            if numeric < -_PROBABILITY_TOLERANCE or numeric > 1.0 + _PROBABILITY_TOLERANCE:
                raise ValueError(
                    f"probability_vector['{outcome}']={numeric} fuori range [0,1] "
                    f"per l'esperto '{self.expert_name}'"
                )
            normalized[outcome] = min(max(numeric, 0.0), 1.0)
        self.probability_vector = normalized

        if self.confidence is None and self.probability_vector:
            self.confidence = self._default_confidence(self.probability_vector)

    @staticmethod
    def _default_confidence(probability_vector: dict[str, float]) -> float:
        """Fallback quando l'esperto non fornisce una propria confidence:
        quanto la probabilita' piu' estrema si allontana dalla neutralita'
        (0.5), scalato in [0, 1]. Non e' una vera stima di incertezza
        statistica: e' solo un indicatore di default consumabile subito dal
        meta-model finche' un esperto non ne definisce una piu' specifica."""
        most_extreme = max(abs(p - 0.5) for p in probability_vector.values())
        return round(most_extreme * 2.0, 6)

    def as_feature_row(self) -> dict[str, Any]:
        """Appiattisce l'output in una riga di feature numeriche, prefissate
        con `expert_name`, direttamente consumabile da un meta-model
        tabellare (acceptance criteria ORACLE-01: 'Tutti gli esperti
        consumabili da meta-model')."""
        row: dict[str, Any] = {
            f"{self.expert_name}__{outcome}": value for outcome, value in self.probability_vector.items()
        }
        if self.confidence is not None:
            row[f"{self.expert_name}__confidence"] = self.confidence
        return row


def combine_expert_outputs(outputs: Iterable[ExpertOutput]) -> dict[str, Any]:
    """Unisce gli `ExpertOutput` di esperti diversi (stessa fixture/
    osservazione) in un'unica riga di feature per il meta-model.

    Solleva ValueError se due esperti producono la stessa colonna (collisione
    di naming da correggere a monte, es. due esperti con lo stesso
    `expert_name`), per non far sparire silenziosamente una feature.
    """
    combined: dict[str, Any] = {}
    for output in outputs:
        row = output.as_feature_row()
        overlapping = set(row) & set(combined)
        if overlapping:
            raise ValueError(f"Colonne duplicate tra esperti diversi: {sorted(overlapping)}")
        combined.update(row)
    return combined
