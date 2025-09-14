--Query odds join match by date
SELECT *
FROM public.odds o
INNER JOIN public.match m ON o.id_match = m.id_match_fk
WHERE m.date_match >= '2025-09-13 00:00:00'
  AND m.date_match < '2025-09-14 00:00:00'
ORDER BY m.id_match_fk ASC;

--Query statistics join match by date
SELECT *
FROM public.statistics s
INNER JOIN public.match m ON s.id_match = m.id_match_fk
WHERE m.date_match >= '2025-09-14 00:00:00'
  AND m.date_match < '2025-09-15 00:00:00'
ORDER BY m.id_match_fk ASC;