"""Addestra e promuove a `production` i 5 mercati "gol" (4 Under/Over +
goal_no_goal) - da eseguire in un ambiente CON accesso DB reale (macchina
locale dell'operatore).

Sostituisce il trasferimento dei file `.pkl` via chat/zip: i modelli
Under/Over arrivano fino a 101MB l'uno (il 4.5, dopo l'integrazione di
SMOTE del 2026-09-08 che ha sostituito i champion di 3.5/4.5 con modelli
piu' grandi) - troppo grandi per un allegato chat, e nel frattempo
superati da retrain successivi. Questo script esegue la STESSA identica
pipeline di produzione (`train_market`/`ModelRegistry.promote_with_policy`,
INVARIATE, nessuna logica duplicata) direttamente sulla macchina che serve
i modelli: i `.pkl` restano sempre generati (e aggiornati) li' dove
servono, mai copiati a mano da un'altra macchina.

Ogni mercato viene addestrato E promosso a production nello stesso giro
(stesso gate di `evaluate_promotion` gia' in uso altrove, nessun bypass):
se il gate blocca un mercato, il messaggio lo dice esplicitamente e si
passa al successivo, nessuna promozione forzata silenziosa.
"""
from __future__ import annotations

from src.service_ia.training.model_registry import ModelRegistry
from src.service_ia.training.train_multi_market import train_market

MARKETS = ["under_over_1_5", "under_over_2_5", "under_over_3_5", "under_over_4_5", "goal_no_goal"]


def main() -> None:
    registry = ModelRegistry()
    for market in MARKETS:
        print(f"\n=== Training {market} ===", flush=True)
        result = train_market(market=market, selection_method="kbest", save_model=True)
        print(
            f"{market}: status={result.status} champion={result.champion} "
            f"selection_score={result.details.get('champion_selection_score')}",
            flush=True,
        )

        if result.status != "trained":
            print(f"  SALTATO (training non riuscito): {market}")
            continue

        run = registry.get_latest(market=market)
        if run is None:
            print(f"  ATTENZIONE: nessun run trovato per {market} dopo il training, promozione saltata")
            continue

        promotion = registry.promote_with_policy(
            run_id=run["run_id"],
            to_stage="production",
            reason="Training + promozione locale (train_and_promote_5_core_markets.py)",
            actor="operator",
        )
        print(
            f"  promoted={promotion['promoted']} allowed={promotion['evaluation']['allowed']} "
            f"blocking={promotion['evaluation']['blocking_reasons']}",
            flush=True,
        )

    print("\nFatto. Verifica con: registry.get_production(market=...) per ciascun mercato.")


if __name__ == "__main__":
    main()
