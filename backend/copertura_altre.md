# Copertura dati marcatori (sola lettura)

## Per lega e stagione
lega  stagione  partite  con_eventi  con_formazioni  events_pct  lineups_pct  full_lineup_pct  gol_non_autogol  goal_in_lineup_pct  score_consistent_pct  ruolo_mancante  nome_mancante
----  --------  -------  ----------  --------------  ----------  -----------  ---------------  ---------------  ------------------  --------------------  --------------  -------------
136   2023      390      390         390             100.0%      100.0%       100.0%           965              100.0%              n.d.                  18              0            
136   2024      389      389         389             100.0%      100.0%       100.0%           924              100.0%              n.d.                  37              0            
136   2025      385      385         385             100.0%      100.0%       100.0%           962              100.0%              n.d.                  469             0            
136   2026      55       55          55              100.0%      100.0%       85.5%            146              100.0%              100.0%                19              0            
137   2023      32       32          32              100.0%      100.0%       100.0%           94               100.0%              n.d.                  6               0            
137   2024      40       40          40              100.0%      100.0%       100.0%           117              100.0%              n.d.                  55              0            
137   2025      45       45          45              100.0%      100.0%       100.0%           107              100.0%              n.d.                  196             0            
137   2026      26       26          26              100.0%      100.0%       96.2%            81               98.8%               100.0%                26              0            
138   2025      364      364         364             100.0%      100.0%       97.0%            844              100.0%              n.d.                  16461           0            
138   2026      84       84          75              100.0%      89.3%        100.0%           191              89.0%               100.0%                3424            0            
144   2023      266      266         266             100.0%      100.0%       100.0%           710              100.0%              n.d.                  79              0            
144   2025      315      315         315             100.0%      100.0%       100.0%           822              99.9%               n.d.                  481             0            
144   2026      60       60          60              100.0%      100.0%       100.0%           165              100.0%              100.0%                49              0            
203   2023      380      380         380             100.0%      100.0%       99.7%            1034             100.0%              n.d.                  41              0            
203   2024      342      342         342             100.0%      100.0%       100.0%           975              100.0%              n.d.                  32              0            
203   2025      300      300         300             100.0%      100.0%       100.0%           769              100.0%              n.d.                  153             0            
203   2026      60       60          60              100.0%      100.0%       100.0%           170              100.0%              100.0%                24              0            
492   2025      297      297         297             100.0%      100.0%       100.0%           971              99.1%               100.0%                10603           0            
492   2026      72       72          71              100.0%      98.6%        100.0%           260              96.5%               100.0%                2311            0            
88    2023      321      321         321             100.0%      100.0%       100.0%           1030             100.0%              n.d.                  49              0            
88    2024      321      321         321             100.0%      100.0%       100.0%           925              100.0%              n.d.                  6               0            
88    2025      308      308         307             100.0%      99.7%        100.0%           949              100.0%              n.d.                  79              0            
88    2026      60       60          60              100.0%      100.0%       98.3%            226              100.0%              100.0%                34              0            
94    2023      308      308         308             100.0%      100.0%       100.0%           857              100.0%              n.d.                  0               0            
94    2024      308      308         308             100.0%      100.0%       100.0%           770              100.0%              n.d.                  33              0            
94    2025      300      300         300             100.0%      100.0%       100.0%           768              100.0%              n.d.                  106             0            
94    2026      59       59          59              100.0%      100.0%       100.0%           156              99.4%               100.0%                7               0            
942   2025      343      343         343             100.0%      100.0%       91.3%            753              100.0%              n.d.                  15788           0            
942   2026      86       86          84              100.0%      97.7%        100.0%           212              98.1%               100.0%                3824            0            
943   2025      363      363         363             100.0%      100.0%       100.0%           885              100.0%              n.d.                  16572           0            
943   2026      86       86          76              100.0%      88.4%        100.0%           232              87.1%               100.0%                3429            0            

events_pct / lineups_pct: partite concluse con eventi / formazioni. full_lineup_pct: formazioni con 22 titolari.
goal_in_lineup_pct: gol (non autogol) agganciati a un giocatore in formazione.
score_consistent_pct: gol+autogol negli eventi == punteggio finale.

## Tassi di base per ruolo (titolari che segnano almeno un gol non autogol)
ruolo           ruolo_codice  titolari  segnato  tasso_pct  panchina  panchina_segnato
--------------  ------------  --------  -------  ---------  --------  ----------------
Centrocampista  M             44779     4462     10.0%      30980     558             
Difensore       D             37599     1504     4.0%       28838     147             
Ruolo mancante  ?             37005     3251     8.8%       37406     717             
Attaccante      F             18754     4364     23.3%      29115     1474            
Portiere        G             10133     6        0.1%       13940     0               
TUTTI           -             148270    13587    9.2%       140279    2896            

## Esito
ATTENZIONE:
- 136 2026: full_lineup_pct 85.5% < 95%
- 138 2026: lineups_pct 89.3% < 95%
- 138 2026: goal_in_lineup_pct 89.0% < 97%
- 492 2026: goal_in_lineup_pct 96.5% < 97%
- 942 2025: full_lineup_pct 91.3% < 95%
- 943 2026: lineups_pct 88.4% < 95%
- 943 2026: goal_in_lineup_pct 87.1% < 97%

## Partite anomale (prime)
fixture_id  league  season  problema               
----------  ------  ------  -----------------------
1439697     942     2025    titolari=21 (attesi 22)
1439487     942     2025    titolari=21 (attesi 22)
1439605     942     2025    titolari=21 (attesi 22)
1438979     138     2025    titolari=21 (attesi 22)
1438995     138     2025    titolari=21 (attesi 22)
1439364     942     2025    titolari=21 (attesi 22)
1439376     942     2025    titolari=21 (attesi 22)
1439380     942     2025    titolari=21 (attesi 22)
1439392     942     2025    titolari=21 (attesi 22)
1439404     942     2025    titolari=21 (attesi 22)
1439417     942     2025    titolari=21 (attesi 22)
1439556     942     2025    titolari=21 (attesi 22)
1439534     942     2025    titolari=21 (attesi 22)
1439101     138     2025    titolari=21 (attesi 22)
1439668     942     2025    titolari=21 (attesi 22)

## Note
- Tassi di base calcolati solo sulle fixture che hanno sia formazione sia eventi.
- I panchinari che non sono entrati risultano "non ha segnato": per il training vanno esclusi (minuti giocati derivabili dalle sostituzioni, oggi non salvati).