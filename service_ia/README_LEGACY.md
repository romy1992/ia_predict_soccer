# Legacy folder: `service_ia/`

Questo percorso resta nel repository per storico ed esperimenti, ma non e la struttura runtime ufficiale.

## Regola runtime
- codice applicativo eseguito da API/scheduler/test: `src/`
- namespace ufficiale Python: `src.service_ia.*`

## Note operative
- non rimuovere questo folder senza inventario completo di dataset/script storici;
- evitare nuove feature in `service_ia/`;
- eventuali fix critici vanno implementati in `src/` e solo retroportati se strettamente necessario.
