SELECT * from public.match m
--inner join public.statistics s on s.id_match = m.id_match_fk
inner join public.odds o on o.id_match = m.id_match_fk
where
--m.current_league=88
--and m.season !=2025
--and
m.id_fixture is null;

SELECT m.*
FROM public.match m
WHERE m.id_fixture IS NULL
  AND EXISTS (
    SELECT 1
    FROM public.match x
    WHERE x.id_fixture IS NOT NULL
      AND (
          (LOWER(x.name_home) = LOWER(m.name_home)
       AND  LOWER(x.name_away) = LOWER(m.name_away))
       OR (LOWER(x.name_home) = LOWER(m.name_away)  -- se vuoi considerare home/away invertiti
       AND  LOWER(x.name_away) = LOWER(m.name_home))
      )
	  AND x.current_league=88
  );



SELECT m.*
FROM public.match m
WHERE m.id_fixture IS NULL
  AND EXISTS (
    SELECT 1
    FROM public.match x
    WHERE x.id_fixture IS NOT NULL
      AND (
		   (similarity(x.name_home, m.name_home) >= 0.65
		AND similarity(x.name_away, m.name_away) >= 0.65)
		OR (similarity(x.name_home, m.name_away) >= 0.65
		AND similarity(x.name_away, m.name_home) >= 0.65)
		)
	  AND x.current_league=88
  );


SELECT
  m.id_fixture           AS id_null_fixture,
  m.name_home    AS m_home,
  m.name_away    AS m_away,
  x.id_fixture   AS matched_fixture,
  x.name_home    AS x_home,
  x.name_away    AS x_away,
  similarity(x.name_home, m.name_home) AS sim_home_direct,
  similarity(x.name_away, m.name_away) AS sim_away_direct
FROM public.match m
JOIN public.match x
  ON x.id_fixture IS NOT NULL
 AND (
      (x.name_home % m.name_home AND x.name_away % m.name_away)
   OR (x.name_home % m.name_away AND x.name_away % m.name_home)
 )
 AND x.current_league=88
WHERE m.id_fixture IS NULL
ORDER BY GREATEST(
  similarity(x.name_home, m.name_home) + similarity(x.name_away, m.name_away),
  similarity(x.name_home, m.name_away) + similarity(x.name_away, m.name_home)
) DESC;

CREATE EXTENSION IF NOT EXISTS pg_trgm;       -- abilita trigrammi
SELECT set_limit(0.6);                        -- soglia di similarità (0..1), regola a piacere
