# 04 — ML Architecture

## Experts
- Team Strength Expert
- Goal Distribution Expert
- Statistics Expert
- Market/Odds Expert
- Direct Market Expert

## Meta layer
Gli expert producono probabilità/output standard. Il meta-model combina solo predizioni OOF temporalmente valide.

## Validazione
- walk-forward;
- expanding/rolling windows;
- holdout finale;
- LogLoss, Brier, ECE;
- betting KPI esclusivamente out-of-sample.

## Calibration
Obbligatoria come layer esplicito e versionato.
