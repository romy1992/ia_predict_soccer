# 06 — Betting Intelligence

Per ogni outcome:
- `p_model`
- `p_market_raw`
- `p_market_fair`
- `odd`
- `prob_edge = p_model - p_market_fair`
- `EV = p_model * odd - 1`
- decision
- policy_version

## Decision
Le soglie non devono essere hardcoded globalmente: vanno versionate e tarate tramite backtest out-of-sample.

## Paper Betting
Ogni prediction deve essere salvata prima del kickoff e poi settled senza alterare il record originale.
