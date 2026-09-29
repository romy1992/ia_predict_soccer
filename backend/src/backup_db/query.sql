--Query odds join match by date
SELECT *
FROM public.odds o
INNER JOIN public.match m ON o.id_match = m.id_match_fk
WHERE m.date_match >= '2026-01-04 00:00:00'
AND m.date_match < '2026-01-05 00:00:00'
ORDER BY m.current_league ASC;

--Query statistics join match by date
SELECT *
FROM public.statistics s
INNER JOIN public.match m ON s.id_match = m.id_match_fk
WHERE m.date_match >= '2026-01-04 00:00:00'
AND m.date_match < '2026-01-05 00:00:00'
ORDER BY m.current_league ASC;

--Query statistics and odds join match by date
SELECT *
FROM public.odds o
INNER JOIN public.match m ON o.id_match = m.id_match_fk
INNER JOIN public.statistics s ON s.id_match = m.id_match_fk
WHERE m.date_match >= '2026-01-04 00:00:00'
AND m.date_match < '2026-01-05 00:00:00'
ORDER BY m.current_league ASC;

-- Check per controllare se alcuni json delle statistics hanno dei valori particolari, in questo caso '^form'
SELECT
  s.for_            AS for_before,
  (
    SELECT COALESCE(jsonb_object_agg(k, v), '{}'::jsonb)
    FROM jsonb_each(COALESCE(for_::jsonb, '{}'::jsonb)) AS e(k, v)
    WHERE k !~ '^form'
  )               AS for_after,
  s.against         AS against_before,
  (
    SELECT COALESCE(jsonb_object_agg(k, v), '{}'::jsonb)
    FROM jsonb_each(COALESCE(against::jsonb, '{}'::jsonb)) AS e(k, v)
    WHERE k !~ '^form'
  )               AS against_after
FROM public.statistics s
WHERE
  EXISTS (SELECT 1 FROM jsonb_object_keys(COALESCE(for_::jsonb, '{}'::jsonb)) AS k WHERE k ~ '^form')
  OR
  EXISTS (SELECT 1 FROM jsonb_object_keys(COALESCE(against::jsonb, '{}'::jsonb)) AS k WHERE k ~ '^form');

-- Update json delle statistics che hanno dei valori particolari, in questo caso '^form'
UPDATE statistics
SET
  for_ = (
    SELECT COALESCE(jsonb_object_agg(k, v), '{}'::jsonb)
    FROM jsonb_each(COALESCE(for_::jsonb, '{}'::jsonb)) AS e(k, v)
    WHERE k !~ '^form'
  ),
  against = (
    SELECT COALESCE(jsonb_object_agg(k, v), '{}'::jsonb)
    FROM jsonb_each(COALESCE(against::jsonb, '{}'::jsonb)) AS e(k, v)
    WHERE k !~ '^form'
  )
WHERE
  EXISTS (SELECT 1 FROM jsonb_object_keys(COALESCE(for_::jsonb, '{}'::jsonb)) AS k WHERE k ~ '^form')
  OR
  EXISTS (SELECT 1 FROM jsonb_object_keys(COALESCE(against::jsonb, '{}'::jsonb)) AS k WHERE k ~ '^form');

-- Inner join status
SELECT * FROM public.odds o
inner join public.match m on m.id_match_fk = o.id_match
where m.status='NS'

SELECT * FROM public.statistics s
inner join public.match m on m.id_match_fk = s.id_match
where m.status='NS'

--Query per cambiare/aggiungere valori nei json columns
UPDATE statistics
SET shots = (
  jsonb_set(
    shots::jsonb,
    '{Total Shots}',
    to_jsonb(
      (ROUND(
        COALESCE( CASE WHEN (shots->>'Shots insidebox')  ~ '^\s*-?\d+(\.\d+)?\s*$'
                       THEN (shots->>'Shots insidebox')::numeric ELSE 0 END, 0)
      + COALESCE( CASE WHEN (shots->>'Shots outsidebox') ~ '^\s*-?\d+(\.\d+)?\s*$'
                       THEN (shots->>'Shots outsidebox')::numeric ELSE 0 END, 0)
      ))::int
    ),
    true
  )
)::json
WHERE (shots::jsonb ? 'Shots insidebox' OR shots::jsonb ? 'Shots outsidebox')
RETURNING shots->>'Shots insidebox' AS insidebox,
          shots->>'Shots outsidebox' AS outsidebox,
          shots->>'Total Shots'      AS total_after;

-- Query che seleziona solo determinati json
SELECT *
FROM public.statistics s
WHERE (s.generic_statistics->>'expected_goals') IS NOT NULL
  AND (s.generic_statistics->>'expected_goals')::numeric > 0;

  -- Controllare dentro i Json
SELECT *
FROM public.match m
WHERE (
  (json_typeof(m.mean_statistics) = 'object'
   AND (m.mean_statistics->>'mean_expected_goals')::numeric > 0)
  OR
  (json_typeof(m.mean_statistics) = 'array'
   AND EXISTS (
     SELECT 1
     FROM json_array_elements(m.mean_statistics) AS elem
     WHERE (elem->>'mean_expected_goals')::numeric > 0
   ))
)
ORDER BY m.date_match ASC;



