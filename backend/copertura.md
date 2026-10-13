# Copertura dati marcatori (sola lettura)

## Per lega e stagione
lega            stagione  partite  con_eventi  con_formazioni  events_pct  lineups_pct  full_lineup_pct  gol_non_autogol  goal_in_lineup_pct  score_consistent_pct  ruolo_mancante  nome_mancante
--------------  --------  -------  ----------  --------------  ----------  -----------  ---------------  ---------------  ------------------  --------------------  --------------  -------------
Bundesliga      2023      308      308         308             100.0%      100.0%       100.0%           965              100.0%              n.d.                  4               0            
Bundesliga      2024      308      308         308             100.0%      100.0%       100.0%           941              100.0%              n.d.                  2               0            
Bundesliga      2025      305      305         305             100.0%      100.0%       100.0%           964              100.0%              n.d.                  33              0            
Bundesliga      2026      45       45          45              100.0%      100.0%       100.0%           167              100.0%              100.0%                3               0            
La Liga         2023      380      380         380             100.0%      100.0%       100.0%           977              100.0%              n.d.                  0               0            
La Liga         2024      380      380         380             100.0%      100.0%       100.0%           969              100.0%              n.d.                  6               0            
La Liga         2025      371      371         371             100.0%      100.0%       100.0%           974              100.0%              n.d.                  80              0            
La Liga         2026      75       73          73              97.3%       97.3%        100.0%           216              100.0%              100.0%                7               0            
Ligue 1         2023      308      308         308             100.0%      100.0%       100.0%           811              100.0%              n.d.                  23              0            
Ligue 1         2024      308      308         308             100.0%      100.0%       100.0%           891              100.0%              n.d.                  33              0            
Ligue 1         2025      287      286         287             99.7%       100.0%       100.0%           788              100.0%              n.d.                  10              0            
Ligue 1         2026      49       49          49              100.0%      100.0%       95.9%            138              99.3%               100.0%                5               0            
Premier League  2023      380      380         380             100.0%      100.0%       100.0%           1197             100.0%              n.d.                  1               0            
Premier League  2024      380      380         380             100.0%      100.0%       100.0%           1082             100.0%              n.d.                  5               0            
Premier League  2025      375      375         375             100.0%      100.0%       100.0%           995              100.0%              n.d.                  34              0            
Premier League  2026      55       54          54              98.2%       98.2%        100.0%           155              99.4%               100.0%                0               0            
Serie A         2023      380      380         380             100.0%      100.0%       99.7%            968              100.0%              n.d.                  8               0            
Serie A         2024      379      379         379             100.0%      100.0%       100.0%           933              100.0%              n.d.                  0               0            
Serie A         2025      376      376         376             100.0%      100.0%       100.0%           888              100.0%              n.d.                  25              0            
Serie A         2026      56       54          54              96.4%       96.4%        100.0%           153              100.0%              100.0%                4               0            

events_pct / lineups_pct: partite concluse con eventi / formazioni. full_lineup_pct: formazioni con 22 titolari.
goal_in_lineup_pct: gol (non autogol) agganciati a un giocatore in formazione.
score_consistent_pct: gol+autogol negli eventi == punteggio finale.

## Tassi di base per ruolo (titolari che segnano almeno un gol non autogol)
ruolo           ruolo_codice  titolari  segnato  tasso_pct  panchina  panchina_segnato
--------------  ------------  --------  -------  ---------  --------  ----------------
Centrocampista  M             47629     4663     9.8%       34630     697             
Difensore       D             41001     1570     3.8%       32652     164             
Attaccante      F             21321     5127     24.0%      29250     1491            
Portiere        G             10998     0        0.0%       15190     0               
Ruolo mancante  ?             26        4        15.4%      257       2               
TUTTI           -             120975    11364    9.4%       111979    2354            

## Esito
Tutti i controlli sopra soglia.

## Partite anomale (prime)
fixture_id  league  season  problema               
----------  ------  ------  -----------------------
1052572     135     2023    titolari=21 (attesi 22)
1552776     61      2026    titolari=21 (attesi 22)
1552782     61      2026    titolari=21 (attesi 22)

## Note
- Tassi di base calcolati solo sulle fixture che hanno sia formazione sia eventi.
- I panchinari che non sono entrati risultano "non ha segnato": per il training vanno esclusi (minuti giocati derivabili dalle sostituzioni, oggi non salvati).