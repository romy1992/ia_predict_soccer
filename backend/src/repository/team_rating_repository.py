from __future__ import annotations

from datetime import datetime, timezone

from src.repository.base.repository_db import SessionLocal
from src.service_ia.model.match import TeamRating


class TeamRatingRepository:
    """Persistenza dei rating CORRENTI per squadra (una riga per team_id,
    UPSERT dal job schedulato `run_team_rating_refresh`).

    Stesso pattern "semplice" gia' in uso nelle altre repository del
    progetto: una sessione per operazione, nessuno stato condiviso tra
    chiamate."""

    def get_for_teams(self, team_ids: list[int]) -> dict[int, TeamRating]:
        if not team_ids:
            return {}
        with SessionLocal() as session:
            rows = (
                session.query(TeamRating)
                .filter(TeamRating.team_id.in_([int(value) for value in team_ids]))
                .all()
            )
            return {row.team_id: row for row in rows}

    def upsert_many(self, ratings: dict[int, dict[str, float]], rating_version: str) -> int:
        """Sovrascrive lo stato corrente per ogni `team_id` in `ratings`
        (dict nella stessa forma di `TeamStrengthExpert.current_ratings`,
        prefisso `team_`). `session.merge` fa UPSERT-per-PK: nessuna riga
        vecchia orfana, nessun bisogno di un DELETE preventivo."""
        if not ratings:
            return 0

        now = datetime.now(timezone.utc)
        with SessionLocal() as session:
            for team_id, payload in ratings.items():
                session.merge(
                    TeamRating(
                        team_id=int(team_id),
                        rating_version=rating_version,
                        matches_played=int(payload.get("team_matches_played") or 0),
                        attack_rating=float(payload.get("team_attack_rating") or 0.0),
                        defense_rating=float(payload.get("team_defense_rating") or 0.0),
                        home_attack_rating=float(payload.get("team_home_attack_rating") or 0.0),
                        home_defense_rating=float(payload.get("team_home_defense_rating") or 0.0),
                        away_attack_rating=float(payload.get("team_away_attack_rating") or 0.0),
                        away_defense_rating=float(payload.get("team_away_defense_rating") or 0.0),
                        home_advantage=float(payload.get("team_home_advantage") or 0.0),
                        rolling_form=float(payload.get("team_rolling_form") or 0.5),
                        rolling_goal_diff=float(payload.get("team_rolling_goal_diff") or 0.0),
                        computed_at=now,
                    )
                )
            session.commit()
        return len(ratings)
