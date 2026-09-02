# 00 — Project Overview

## Obiettivo
Soccer Oracle è una piattaforma React + FastAPI che:
- importa e aggiorna quotidianamente partite storiche, odierne, future e live;
- mantiene un database canonico con fixture, quote, statistiche e snapshot temporali;
- addestra una famiglia di modelli specializzati per diversi mercati;
- combina modelli esperti in un Oracle Ensemble;
- produce probabilità calibrate;
- confronta le probabilità Oracle con il mercato bookmaker;
- calcola fair probability, edge, EV e decisione `PLAY / BORDERLINE / NO BET`;
- costruisce in seguito una Schedina Oracle con gestione correlazioni;
- monitora prestazioni, versioni modello e paper betting.

## Base di codice
La base consigliata è `feature/ml-dashboard-platform`, che è avanti rispetto a `main` e include:
- React frontend;
- FastAPI;
- Docker;
- scheduler;
- dashboard service;
- Model Registry MVP;
- Prediction Logger MVP;
- training multi-market MVP;
- test aggiuntivi.

Il branch `main` resta importante come fonte del codice storico di ingestion, preprocessing e sperimentazione ML.
