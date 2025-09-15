--Query odds join match by date
SELECT *
FROM public.odds o
INNER JOIN public.match m ON o.id_match = m.id_match_fk
WHERE m.date_match >= '2025-09-13 00:00:00'
  AND m.date_match < '2025-09-15 00:00:00'
ORDER BY m.current_league ASC;

--Query statistics join match by date
SELECT *
FROM public.statistics s
INNER JOIN public.match m ON s.id_match = m.id_match_fk
WHERE m.date_match >= '2025-09-13 00:00:00'
  AND m.date_match < '2025-09-15 00:00:00'
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
