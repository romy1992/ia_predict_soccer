from __future__ import annotations

import argparse
import json

from src.ml.datasets.point_in_time_builder import PointInTimeDatasetBuilder


def _parse_seasons(raw: str | None) -> list[int] | None:
    if not raw:
        return None
    values = []
    for token in raw.split(","):
        token = token.strip()
        if not token:
            continue
        values.append(int(token))
    return values or None


def main() -> None:
    parser = argparse.ArgumentParser(description="Build point-in-time dataset for a market")
    parser.add_argument("--market", required=True, help="Market key (es. under_over_2_5)")
    parser.add_argument("--seasons", default=None, help="Comma separated seasons (es. 2024,2025)")
    parser.add_argument("--period", default="full_time", help="Market period")
    parser.add_argument("--line", default=None, help="Optional line (es. 2.5)")
    parser.add_argument("--outcome", default=None, help="Optional outcome scope")
    parser.add_argument("--save-snapshot", action="store_true", help="Persist CSV + metadata in best_models/datasets")
    args = parser.parse_args()

    builder = PointInTimeDatasetBuilder()
    dataset = builder.build_from_db(
        market=args.market,
        seasons=_parse_seasons(args.seasons),
        period=args.period,
        line=args.line,
        outcome=args.outcome,
        save_snapshot=args.save_snapshot,
    )

    print(
        json.dumps(
            {
                "market": dataset.market,
                "version": dataset.version,
                "rows": len(dataset.frame),
                "no_leakage": builder.assert_no_leakage(dataset.frame),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()

