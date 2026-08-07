"""Select the global Top-5 UCB configurations from the merged grid-search CSV."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


DEFAULT_INPUT = Path(__file__).with_name("output.csv")
DEFAULT_OUTPUT = Path(__file__).with_name("top5_ucb_configs.csv")

REQUIRED_COLUMNS = {
    "config_id",
    "global_index",
    "beta",
    "beta_decay",
    "delta",
    "sharpe_mean",
    "profit_mean",
    "profit_std",
    "max_drawdown_mean",
}

NUMERIC_COLUMNS = (
    "global_index",
    "beta",
    "beta_decay",
    "delta",
    "sharpe_mean",
    "sharpe_std",
    "profit_mean",
    "profit_std",
    "arr_mean",
    "arr_std",
    "max_drawdown_mean",
    "max_drawdown_std",
    "score",
)

OUTPUT_COLUMNS = (
    "rank",
    "config_id",
    "global_index",
    "gamma",
    "beta",
    "beta_decay",
    "delta",
    "seed_list",
    "seeds_completed",
    "sharpe_mean",
    "sharpe_std",
    "profit_mean",
    "profit_std",
    "arr_mean",
    "arr_std",
    "max_drawdown_mean",
    "max_drawdown_std",
    "score",
)


def select_top5(
    input_csv: Path,
    max_drawdown_percent: float = 20.0,
    gamma: float = 0.95,
) -> pd.DataFrame:
    """Load, validate, filter, rank, and return the five best configurations."""
    frame = pd.read_csv(input_csv)
    missing = sorted(REQUIRED_COLUMNS - set(frame.columns))
    if missing:
        raise ValueError(f"Input CSV is missing required columns: {missing}")

    frame = frame.copy()
    for column in NUMERIC_COLUMNS:
        if column in frame.columns:
            frame[column] = pd.to_numeric(frame[column], errors="raise")

    filtered = frame.loc[
        frame["max_drawdown_mean"] < float(max_drawdown_percent)
    ].copy()
    if filtered.empty:
        raise ValueError(
            f"No configuration has max_drawdown_mean < {max_drawdown_percent}."
        )

    # Primary criterion: higher mean Sharpe. Profit std only breaks Sharpe ties.
    ranked = filtered.sort_values(
        by=["sharpe_mean", "profit_std", "config_id"],
        ascending=[False, True, True],
        kind="mergesort",
    ).head(5).reset_index(drop=True)

    ranked.insert(0, "rank", range(1, len(ranked) + 1))
    ranked.insert(ranked.columns.get_loc("beta"), "gamma", float(gamma))

    available_output_columns = [
        column for column in OUTPUT_COLUMNS if column in ranked.columns
    ]
    return ranked.loc[:, available_output_columns]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Select Top-5 UCB configurations for 20-seed validation."
    )
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--max-drawdown", type=float, default=20.0)
    parser.add_argument("--gamma", type=float, default=0.95)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    top5 = select_top5(
        input_csv=args.input,
        max_drawdown_percent=args.max_drawdown,
        gamma=args.gamma,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    top5.to_csv(args.output, index=False)

    print("=== TOP 5 CONFIGURATIONS FOR 20-SEED VALIDATION ===")
    print(
        top5[
            [
                "rank",
                "config_id",
                "gamma",
                "beta",
                "beta_decay",
                "delta",
                "sharpe_mean",
                "profit_mean",
                "profit_std",
                "max_drawdown_mean",
            ]
        ].to_string(index=False)
    )
    print(f"\nSaved: {args.output.resolve()}")


if __name__ == "__main__":
    main()
