# 02 — Target Architecture

```text
API SPORTS / PROVIDER
        |
        v
INGESTION PLATFORM
  historical / today / future / live
        |
        v
CANONICAL DATABASE
        |
        v
POINT-IN-TIME FEATURE LAYER
        |
        +--> Team Strength Expert
        +--> Goal Distribution Expert
        +--> Statistics Expert
        +--> Market/Odds Expert
        +--> Direct Market Expert
                    |
                    v
              META ENSEMBLE
                    |
                    v
                CALIBRATION
                    |
                    v
          ORACLE PROBABILITIES
                    |
                    v
             FAIR ODDS ENGINE
                    |
                    v
           VALUE / EV ENGINE
                    |
                    v
          DECISION POLICY
                    |
                    +--> Match Center
                    +--> Oracle Picks
                    +--> Paper Betting
                    +--> Schedina Oracle
```

## Principi
- dataset point-in-time;
- separazione data jobs / ML jobs;
- probabilità prima di label;
- calibration prima del betting;
- latest model != production model;
- ogni prediction deve essere ricostruibile;
- UI non deve contenere logica scientifica.
