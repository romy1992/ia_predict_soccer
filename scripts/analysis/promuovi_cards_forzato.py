"""Promozione FORZATA dei 3 candidati cards (linee 4.5/5.5/6.5) del 19/09/2026.

Il gate ha rifiutato questi 3 candidati perche' il loro `selection_score` e'
piu' basso di quello dei modelli in production del 12/09. Verificato che il
vantaggio dei modelli vecchi NON viene da feature migliori: usano 73 feature,
di cui le "quote" sono in realta' le feature LEGACY aggregate
(odds_mean/min/max/std/slot_1..10, Over e Under mescolati - il pattern che
questo progetto esclude deliberatamente dal training nuovo, vedi
LEGACY_ODDS_FEATURES in filter_market_service.py), addestrati su 8355 righe
IDENTICHE su tutte le 4 linee (segno di zero-riempimento delle righe senza
quota reale). Test locale: aggiungere le 58 feature statistiche squadra ai
candidati nuovi PEGGIORA l'AUC su tutte e 4 le linee (es. 6.5: 0.690 con
sole quote vs 0.661 con statistiche aggiunte).

Decisione esplicita dell'operatore: forzare la promozione per avere lo
stesso principio metodologico (quote separate per esito, niente zero-fill,
niente feature legacy) su tutte e 4 le linee cards, coerentemente con
cards_line_3_5 (gia' promossa senza forzare).

Uso:
    python scripts/analysis/promuovi_cards_forzato.py
"""

from __future__ import annotations

import json
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from src.service_ia.training.model_paths import to_container_path  # noqa: E402
from src.service_ia.training.model_registry import ModelRegistry  # noqa: E402

RUN_IDS = [
    "cards_line_4_5_20260919T064401589205Z",
    "cards_line_5_5_20260919T064559240432Z",
    "cards_line_6_5_20260919T064753521140Z",
]

REASON = (
    "FORZATA (force=True): selection_score piu' basso della production 12/09 su "
    "tutte e 3 le linee (4.5: 0.6864 vs 0.6871, 5.5: 0.6960 vs 0.7241, 6.5: 0.7453 "
    "vs 0.8044), ma il vantaggio dei modelli in production e' un artefatto: usano "
    "73 feature dove le 'quote' sono in realta' le feature legacy aggregate "
    "(odds_mean/min/max/std/slot_1..10, Over/Under mescolati, escluse "
    "deliberatamente dal training nuovo - vedi LEGACY_ODDS_FEATURES) piu' 58 "
    "feature statistiche squadra e 4 arbitro, addestrati su 8355 righe identiche "
    "su tutte le 4 linee (zero-fill delle righe senza quota reale, non filtro "
    "sulla quota vera). Verificato in locale che aggiungere le stesse 58 feature "
    "statistiche ai candidati nuovi peggiora l'AUC su tutte e 4 le linee (es. 6.5: "
    "0.690 solo-quote vs 0.661 con statistiche). Decisione esplicita "
    "dell'operatore: forzare la promozione per avere lo stesso principio "
    "metodologico (quote separate per esito, niente zero-fill, niente feature "
    "legacy) su tutte e 4 le linee cards, coerentemente con cards_line_3_5 (gia' "
    "promossa senza forzare) e con gli altri mercati rifatti nel progetto "
    "(goal_no_goal, under_over_1.5/2.5/3.5)."
)


def riscrivi_percorsi_container() -> None:
    index = os.path.join("best_models", "registry", "index.jsonl")
    if not os.path.exists(index):
        return
    righe = [json.loads(l) for l in open(index, encoding="utf-8")]
    corrette = 0
    for r in righe:
        for campo in ("model_path", "metadata_path"):
            valore = r.get(campo) or ""
            nuovo = to_container_path(valore)
            if valore and nuovo != valore:
                r[campo] = nuovo
                corrette += 1
        cal = (r.get("extra") or {}).get("calibration") or {}
        valore = cal.get("calibrator_path") or ""
        nuovo = to_container_path(valore)
        if valore and nuovo != valore:
            cal["calibrator_path"] = nuovo
            corrette += 1
    if corrette:
        with open(index, "w", encoding="utf-8") as f:
            for r in righe:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        print(f"percorsi riscritti in forma container: {corrette}")


def main() -> int:
    registry = ModelRegistry()
    for run_id in RUN_IDS:
        esito = registry.promote_with_policy(
            run_id=run_id,
            to_stage="production",
            reason=REASON,
            actor="operator_request",
            force=True,
        )
        print(f"{run_id} -> promoted={esito.get('promoted')}")

    riscrivi_percorsi_container()
    return 0


if __name__ == "__main__":
    sys.exit(main())
