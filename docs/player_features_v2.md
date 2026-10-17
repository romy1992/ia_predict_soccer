# Marcatori e cartellini giocatori: feature v2 (spente)

## Cosa e'
Le feature v2 aggiungono ai modelli dei giocatori le statistiche per giocatore di `player_match_stats`
(riempita dal backfill). Sono calcolate solo dalle partite PRECEDENTI (point-in-time) e stanno nei dataset
accanto alle v1:

- Marcatori (`scoring_dataset.STATS_FEATURES`): `shots_p90_l10`, `shots_on_p90_l10`, `shots_p90_l5`,
  `key_passes_p90_l10`, `pen_won_l20`, `rating_l5`, `stats_games_l10`.
- Cartellini giocatori (`cards_dataset.STATS_FEATURES`): `fouls_c_p90_l10`, `fouls_d_p90_l10`,
  `fouls_c_p90_l5`, `stats_games_l10`.

Dove la partita non ha statistiche il valore e' vuoto (NaN) e il modello usa la mediana; `stats_games_l10`
dice quante delle ultime 10 partite le avevano.

## Stato: SPENTE di default
L'addestramento usa le feature v1. Per provare v2: variabile d'ambiente `PLAYER_FEATURE_SET=v2` sul servizio
che lancia il job di addestramento. Il bundle salvato ricorda le colonne usate (`feature_columns`), quindi un
modello v1 in produzione continua a funzionare anche con il codice v2; la previsione carica le statistiche solo
se il bundle le usa (`bundle_uses_stats`). Il registro segna `feature_set` v1/v2 in `extra`.

## Perche' non e' attiva (17 ottobre 2026)
Confronto v1 contro v2 su holdout temporale (ultimo 20% delle date, solo titolari, 295.630 righe, 12
competizioni, 4 stagioni), modelli addestrati in cloud, nessuna promozione:

| Mercato | Logloss v1 -> v2 | AUC v1 -> v2 | ECE v1 -> v2 |
|---|---|---|---|
| Marcatori | 0,26718 -> 0,26688 | 0,7424 -> 0,7436 | 0,0027 -> 0,0033 |
| Cartellini giocatori | 0,41671 -> 0,41622 | 0,5896 -> 0,5934 | 0,0042 -> 0,0049 |

Guadagno dentro il rumore, calibrazione leggermente peggiore. Copertura delle statistiche (`stats_games_l10 > 0`):
top5 circa 89-94%, coppe 72-77%, altri campionati 9-11%. Anche i cartellini di squadra (linee 3.5-6.5) con i
dati rigenerati migliorano di soli 0,003-0,008 di logloss.

## Quando riprovare
Quando la copertura dei campionati minori sale, o con feature piu' ricche (assenze, formazione prevista).
Procedura: esportare i dataset con `scripts/build_player_scoring_dataset.py` e
`scripts/build_player_cards_dataset.py` (contengono gia' le colonne v2), valutare v1/v2 con
`model_store.holdout_metrics(df, columns=...)`, e attivare `PLAYER_FEATURE_SET=v2` solo se il guadagno e'
chiaro su tutti i gruppi (top5 / coppe / altri).
