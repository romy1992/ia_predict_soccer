# 03 — Data Platform

## Flussi
- Historical Import: intervallo date.
- Today Update: giornata corrente.
- Future Sync: finestra N giorni.
- Settlement: finalizzazione partite concluse.
- Live Sync: pipeline separata futura.

## Principio
L'import deve essere idempotente e parametrico. Nessuna data operativa deve richiedere modifica del codice.

## Odds
Il target V2 usa snapshot normalizzati con timestamp, mantenendo temporaneamente compatibilità con il JSON legacy.
