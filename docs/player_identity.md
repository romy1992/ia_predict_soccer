# Anagrafica giocatori (API-Sports + FotMob)

Una riga per persona in `player_identity`, con un collegamento per provider in `player_provider_link`
(ID del provider, nome come lo scrive il provider, metodo di abbinamento, fiducia, origine). API-Sports è il
contenitore principale: gli ID di modelli e storico restano i suoi. Migrazione Alembic: `d2e3f4a5b6c7`.

## Come si popola
1. **API-Sports**: ogni giocatore già in `player_lineup` ha un'identità (nessuna chiamata API).
2. **FotMob**: il job `fotmob_predicted_lineups` salva ogni abbinamento univoco trovato e, dal giro dopo, usa il
   collegamento salvato senza rifare il confronto per nome.
3. **Formazione ufficiale**: quando arriva, i titolari FotMob rimasti senza abbinamento si abbinano per eliminazione
   (un solo buco per squadra) e quelli già abbinati vengono confermati.
4. **Manuale / altri progetti**: `POST /players/links`. Un collegamento `manual` o `external_project` non viene mai
   sovrascritto da uno automatico.

## API (sola lettura tranne `links` e `sync`)
Accesso: intestazione `X-Service-Key: <PLAYERS_API_KEY>` (variabile d'ambiente del servizio api), valida solo per
`/players/*`. Senza chiave serve il login del pannello.

- `GET /players/lookup?provider=fotmob&provider_id=123` oppure `?name=Lautaro Martinez&team_id=505&birth_date=1997-08-22`
  → `status` = `found` | `ambiguous` (candidati, nessuna scelta) | `not_found`; `method` e `confidence`.
  La data di nascita esclude gli omonimi con data diversa e alza la fiducia.
- `GET /players/lineups/{fixture_id}?prefer=official|predicted` → formazione con `api_sports_id`, `fotmob_id` e
  `identity_id` di ogni giocatore. `lineup_type` = `official` (API-Sports, completa) oppure `predicted` (FotMob).
  L'ufficiale sostituisce la prevista semplicemente perché esiste.
- `GET /players/lineups?fixture_ids=1,2,3&prefer=official|predicted` → stessa risposta per più partite (massimo 50):
  `{"count", "fixtures": [...]}`.
- `POST /players/links/bulk` `{"links": [{api_sports_id, fotmob_id?, name?, birth_date?}], "source": ...}` (massimo 500).
- `POST /players/links` `{api_sports_id, fotmob_id?, name?, birth_date?, source: external_project|manual}`.
- `POST /players/sync` aggiorna l'anagrafica e fa imparare dalle formazioni ufficiali recenti.

## Cosa non fa
Non sceglie tra omonimi senza un criterio (data di nascita, ruolo, recenza di almeno 60 giorni). Un nome non
trovato resta non trovato. Non scambia mai un ID FotMob per un ID API-Sports.

## Campi di ogni giocatore nelle formazioni
`name`, `position` (G/D/M/F, ruolo storico), `api_sports_id` (nullo se non abbinato), `fotmob_id`, `identity_id`,
`status` (`starter` | `bench` | `unavailable`), `mapped` (bool), `method` e `confidence` dell'abbinamento (ufficiale
API-Sports: `provider_id`, 1.0; previsto: nome esatto 0.9, ordine invertito 0.85, iniziale+cognome 0.8, cognome
contenuto 0.75, per eliminazione 0.8, collegamento salvato = la fiducia del collegamento), `shirt_number` (FotMob, solo
nelle previste) e `absence` per gli indisponibili: `{type, expected_return, expected_return_date, certainty}`.
Nelle previste `unmapped_starters` conta i titolari senza ID API-Sports: chi li ha nel proprio database (rosa, maglia,
data di nascita) può abbinarli e rimandarli con `POST /players/links` o `/links/bulk`; da quel momento restano salvati.
