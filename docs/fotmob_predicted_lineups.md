# Formazioni previste FotMob

FotMob, via RapidAPI, è un provider aggiuntivo delle probabili formazioni. Non sostituisce API-Sports e non cambia i pesi dei modelli marcatori e ammoniti.

## Perché è separato

Il job `scorer_predictions` scarica la formazione ufficiale API-Sports circa 75 minuti prima del calcio d'inizio e calcola le probabilità solo quando ci sono 22 titolari in `player_lineup`. Prima di quell'ora la formazione ufficiale non c'è, quindi non c'è una previsione iniziale.

FotMob ha risposto `lineupType: predicted` su partite di Serie A, La Liga, Bundesliga e Premier League a 24–72 ore dal calcio d'inizio. Quegli ID non sono gli ID API-Sports. Vanno mappati per nome e per data.

Una chiamata live da questo ambiente non è stata possibile: `RAPIDAPI_KEY` non è configurata. Parser e job sono coperti da fixture locali, inclusa una partita di Ligue 1 nello stesso schema. La Ligue 1 reale va verificata al primo giro con la chiave.

## Variabili

| Variabile | Default | Uso |
|---|---|---|
| `RAPIDAPI_KEY` | nessuna | Chiave RapidAPI. Solo ambiente, mai nel repository, nei log o nelle risposte. |
| `FOTMOB_RAPIDAPI_HOST` | `fotmob-api.p.rapidapi.com` | Host del piano. |
| `FOTMOB_LINEUP_HOUR` | `6` | Ora locale Europe/Rome del job. |
| `FOTMOB_LINEUP_MINUTE` | `30` | Minuto. Sta dopo il sync del calendario. |
| `FOTMOB_HORIZON_HOURS` | `72` | Partite `NS` da guardare. |
| `FOTMOB_MAX_REQUESTS_PER_RUN` | `120` | Tetto di chiamate per giro. |
| `FOTMOB_REFRESH_HOURS` | `20` | Non richiama una partita già salvata in questa finestra. |

Su Railway la chiave va aggiunta al servizio `scheduler` (e all'`api` se si usa "Esegui ora"). Questo cambiamento non può impostarla da solo.

## Job

`fotmob_predicted_lineups`, una volta al giorno. Dalle Impostazioni, "Esegui ora" chiama `POST /jobs/fotmob-predicted-lineups`.

Per ogni partita delle prime cinque leghe e delle coppe europee già in tabella `match`:

1. Cerca il `matchId` FotMob in `GET /api/v1/matches/by-date?date=YYYYMMDD` (data UTC del calcio d'inizio).
2. Scarica `GET /api/v1/matches/{matchId}/lineups`.
3. Controlla la struttura e la rosa.
4. Appende uno snapshot. Non cancella gli snapshot precedenti e non scrive `player_lineup`.
5. Se l'affidabilità è `validated` e il tipo è `predicted`, calcola marcatori e ammoniti con il modello già in `production`. La revisione ha `lineup_type=predicted`.

Se manca la chiave, FotMob risponde 401/403, o non c'è formazione, il job non inventa titolari. Le previsioni ufficiali restano quelle di API-Sports.

Il dettaglio partita FotMob non viene chiamato: la data usata per l'abbinamento è quella già presente in `by-date`. Si evita una richiesta in più per partita.

## Affidabilità

Controlli strutturali, da soli, possono solo rifiutare:

- 11 titolari per squadra, ID presenti e non duplicati
- nessun titolare anche nella lista indisponibili
- modulo `N-N-N` i cui numeri sommano a 10

Poi, a parte:

- `rejected`: struttura rotta, oppure le squadre nella formazione non sono quelle della partita
- `unverified`: struttura intera ma rosa storica assente, oppure `lineupType` non è `predicted` né `confirmed`/`official`
- `partially_validated`: struttura intera, ma almeno un titolare non è mappato in modo univoco, il ruolo storico manca, oppure il calcio d'inizio non è entro 3 ore
- `validated`: tutto quanto sopra è passato

`Doubtful`, `Back in training` e `A few weeks` restano testi incerti. Diventano una data solo se il provider manda `expectedReturnDate`.

Non c'è una percentuale di confidenza. Il valore di mercato è salvato sul giocatore e non entra nel modello.

## Previsione iniziale

Si scrive solo con affidabilità `validated` e tipo `predicted`. I `player_id` sono quelli API-Sports, trovati per nome completo univoco nella rosa storica della squadra, oppure per iniziale e cognome se anche quella coppia è univoca. Un nome che punta a due ID non viene scelto.

La formazione ufficiale, quando arriva, aggiunge una revisione `official` anche se gli undici sono gli stessi. Le revisioni previste restano.

Senza modello `production` lo snapshot c'è lo stesso e la previsione no.

Feature possibili per un backtest successivo, non usate ora: titolari abituali assenti, valore di mercato aggregato, stabilità dell'undici, modulo, differenza rispetto all'undici abituale. Vanno misurate contro `scored` / `yellow` prima di qualsiasi peso nuovo.

## Richieste

Un giro completo, a regime, è una chiamata formazione per partita non aggiornata da 20 ore. La lista per data si chiede solo per le partite ancora senza mappa. Nel finesettimana delle cinque leghe e delle coppe sono circa 40–70 formazioni, più qualche lista giornaliera: sotto le 120 richieste del tetto. Al mese, un giro al giorno, circa 1.500–2.200 richieste. Il piano RapidAPI va confrontato con questo numero, e va confermato che il piano permette di conservare gli snapshot.

## Lettura

`GET /fotmob/lineups/{fixture_id}` restituisce l'ultimo snapshot e i giocatori, senza il payload grezzo e senza la chiave.

Le probabilità iniziali compaiono negli endpoint già esistenti `/scorers/fixture/{id}` e `/player-cards/fixture/{id}`, con `lineup_type` `predicted` finché non arriva la revisione `official`.

## Tabelle

- `fotmob_match_map`: fixture API-Sports → match FotMob
- `fotmob_lineup_snapshot`: uno snapshot per recupero, con affidabilità e warning
- `fotmob_lineup_player`: titolari, panchina, indisponibili, allenatore

Migrazione Alembic: `c1d2e3f4a5b6`.
