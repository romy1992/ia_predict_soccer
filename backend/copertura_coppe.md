# Copertura dati marcatori (sola lettura)

## Per lega e stagione
lega  stagione  partite  con_eventi  con_formazioni  events_pct  lineups_pct  full_lineup_pct  gol_non_autogol  goal_in_lineup_pct  score_consistent_pct  ruolo_mancante  nome_mancante
----  --------  -------  ----------  --------------  ----------  -----------  ---------------  ---------------  ------------------  --------------------  --------------  -------------
2     2023      77       77          77              100.0%      100.0%       100.0%           219              100.0%              n.d.                  2               0            
2     2024      95       95          95              100.0%      100.0%       100.0%           271              100.0%              n.d.                  8               0            
2     2025      188      188         188             100.0%      100.0%       100.0%           633              100.0%              n.d.                  33              0            
2     2026      39       39          39              100.0%      100.0%       97.4%            124              100.0%              100.0%                12              0            
3     2023      39       39          39              100.0%      100.0%       100.0%           103              100.0%              n.d.                  2               0            
3     2024      35       35          35              100.0%      100.0%       100.0%           115              100.0%              n.d.                  3               0            
3     2025      187      187         187             100.0%      100.0%       100.0%           495              100.0%              n.d.                  71              0            
3     2026      55       55          55              100.0%      100.0%       100.0%           159              99.4%               100.0%                512             0            
848   2023      12       12          12              100.0%      100.0%       100.0%           38               100.0%              n.d.                  0               0            
848   2024      8        8           8               100.0%      100.0%       100.0%           28               100.0%              n.d.                  0               0            
848   2025      150      150         150             100.0%      100.0%       100.0%           363              100.0%              n.d.                  142             0            
848   2026      78       78          78              100.0%      100.0%       97.4%            237              99.2%               n.d.                  126             0            

events_pct / lineups_pct: partite concluse con eventi / formazioni. full_lineup_pct: formazioni con 22 titolari.
goal_in_lineup_pct: gol (non autogol) agganciati a un giocatore in formazione.
score_consistent_pct: gol+autogol negli eventi == punteggio finale.

## Tassi di base per ruolo (titolari che segnano almeno un gol non autogol)
ruolo           ruolo_codice  titolari  segnato  tasso_pct  panchina  panchina_segnato
--------------  ------------  --------  -------  ---------  --------  ----------------
Centrocampista  M             8068      861      10.7%      6453      129             
Difensore       D             7225      306      4.2%       5636      32              
Attaccante      F             3639      875      24.0%      5031      254             
Portiere        G             1895      2        0.1%       3373      0               
Ruolo mancante  ?             356       37       10.4%      555       5               
TUTTI           -             21183     2081     9.8%       21048     420             

## Esito
Tutti i controlli sopra soglia.

## Partite anomale (prime)
fixture_id  league  season  problema               
----------  ------  ------  -----------------------
1623406     848     2026    titolari=21 (attesi 22)
1607628     848     2026    titolari=21 (attesi 22)
1635708     2       2026    titolari=21 (attesi 22)

## Note
- Tassi di base calcolati solo sulle fixture che hanno sia formazione sia eventi.
- I panchinari che non sono entrati risultano "non ha segnato": per il training vanno esclusi (minuti giocati derivabili dalle sostituzioni, oggi non salvati).