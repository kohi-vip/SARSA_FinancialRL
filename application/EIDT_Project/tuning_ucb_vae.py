"""Three-stage hyperparameter tuning for the notebook UCB-VAE agent.

The module deliberately receives the notebook's training callbacks instead of
duplicating the model implementation. This keeps tuning and final training on
the same code path.
"""

from __future__ import annotations

import copy
import gc
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import torch

try:
    import optuna
except ImportError:  # The notebook installation cell installs it on Kaggle.
    optuna = None


Fold = Tuple[str, pd.DataFrame, pd.DataFrame]


@dataclass(frozen=True)
class TuningBudget:
    """Compute budget used by all three stages."""

    stage_a_trials: int = 15
    stage_b_trials: int = 20
    stage_c_trials: int = 15
    episodes_per_trial: int = 15
    seeds: Tuple[int, ...] = (42, 43)
    stability_penalty: float = 0.25
    timeout_per_stage: Optional[int] = None


def build_fixed_walk_forward_folds(price_history: pd.DataFrame) -> List[Fold]:
    """Build validation folds wholly inside 2013-2018.

    Consequently, the notebook's GOOD test (2019-2021) and BAD test
    (2022-2023) remain untouched for the final comparison.
    """

    data = price_history.copy()
    data["time"] = pd.to_datetime(data["time"])
    specs = [
        ("fold_2016", "2013-01-01", "2015-12-31", "2016-01-01", "2016-12-31"),
        ("fold_2017", "2013-01-01", "2016-12-31", "2017-01-01", "2017-12-31"),
        ("fold_2018", "2013-01-01", "2017-12-31", "2018-01-01", "2018-12-31"),
    ]

    folds: List[Fold] = []
    for name, train_start, train_end, val_start, val_end in specs:
        train = data[data["time"].between(train_start, train_end)].reset_index(drop=True)
        validation = data[data["time"].between(val_start, val_end)].reset_index(drop=True)
        if train.empty or validation.empty:
            raise ValueError(
                f"{name} is empty: train={len(train)}, validation={len(validation)}."
            )
        folds.append((name, train, validation))
    return folds


def _portfolio_metrics(portfolio: Sequence[float], dates: Sequence[Any]) -> Dict[str, float]:
    values = np.asarray(portfolio, dtype=np.float64)
    if values.size < 2 or not np.all(np.isfinite(values)):
        raise ValueError("Portfolio history must contain at least two finite values.")

    denominators = np.maximum(np.abs(values[:-1]), 1e-8)
    returns = np.diff(values) / denominators
    volatility = float(np.std(returns) * np.sqrt(252.0) * 100.0)
    annual_return = float(np.mean(returns) * 252.0 * 100.0)
    sharpe = 0.0 if volatility <= 1e-8 else (annual_return - 2.0) / volatility

    peak = np.maximum.accumulate(values)
    max_drawdown = float(abs(np.min((values - peak) / np.maximum(np.abs(peak), 1e-8))) * 100.0)

    start, end = pd.to_datetime(dates[0]), pd.to_datetime(dates[-1])
    years = max((end - start).days / 365.25, 1.0 / 365.25)
    initial_value, final_value = float(values[0]), float(values[-1])
    growth = final_value / max(abs(initial_value), 1e-8)
    cagr = -100.0 if growth <= 0 else float((growth ** (1.0 / years) - 1.0) * 100.0)

    return {
        "sharpe": float(sharpe),
        "annual_return": annual_return,
        "cagr": cagr,
        "max_drawdown": max_drawdown,
    }


def _score_metrics(metrics: Dict[str, float]) -> float:
    """Risk-aware scalar objective used consistently in all stages."""

    return float(
        metrics["sharpe"]
        + 0.002 * metrics["cagr"]
        - 0.01 * metrics["max_drawdown"]
    )


def _suggest_stage_parameters(trial: Any, stage: str) -> Dict[str, Any]:
    if stage == "A":
        return {
            "gamma": trial.suggest_float("gamma", 0.90, 0.995),
            "alpha": trial.suggest_float("alpha", 0.10, 1.0),
            "nn_lr": trial.suggest_float("nn_lr", 1e-5, 1e-3, log=True),
            "nn_epochs": trial.suggest_int("nn_epochs", 1, 4),
            "batch_size": trial.suggest_categorical("batch_size", [32, 64, 128, 256]),
        }
    if stage == "B":
        return {
            "vae_latent_dim": trial.suggest_categorical("vae_latent_dim", [4, 8, 16, 32]),
            "vae_lr": trial.suggest_float("vae_lr", 1e-5, 3e-3, log=True),
            "vae_beta_kl": trial.suggest_float("vae_beta_kl", 1e-5, 1e-1, log=True),
            "vae_batch_size": trial.suggest_categorical("vae_batch_size", [32, 64, 128, 256]),
            "vae_updates_per_q_batch": trial.suggest_int("vae_updates_per_q_batch", 1, 3),
            "vae_replay_capacity": trial.suggest_categorical(
                "vae_replay_capacity", [5_000, 10_000, 25_000, 50_000]
            ),
            "bootstrap_trajectories": trial.suggest_int("bootstrap_trajectories", 2, 8),
            "bootstrap_vae_updates": trial.suggest_categorical(
                "bootstrap_vae_updates", [20, 50, 100]
            ),
        }
    if stage == "C":
        return {
            "beta": trial.suggest_float("beta", 0.01, 1.0, log=True),
            "beta_decay": trial.suggest_float("beta_decay", 0.90, 0.999),
            "beta_min": trial.suggest_float("beta_min", 0.001, 0.10, log=True),
            "delta": 1.0,
            "lam": trial.suggest_float("lam", 0.1, 10.0, log=True),
        }
    raise ValueError(f"Unknown tuning stage: {stage}")


def _evaluate_config(
    trial: Any,
    config: Dict[str, Any],
    folds: Sequence[Fold],
    seeds: Sequence[int],
    stability_penalty: float,
    run_single_ucb: Callable[[pd.DataFrame, pd.DataFrame, Any, Dict[str, Any]], Dict[str, Any]],
    mdp_factory: Callable[[Dict[str, Any]], Any],
    set_seed: Callable[[int], None],
) -> float:
    run_scores: List[float] = []
    metric_rows: List[Dict[str, float]] = []
    step = 0

    for fold_name, train, validation in folds:
        for seed in seeds:
            set_seed(int(seed))
            mdp = mdp_factory(config)
            result = run_single_ucb(train, validation, mdp, config)
            metrics = _portfolio_metrics(result["portfolio_history"], validation["time"])
            score = _score_metrics(metrics)
            if not np.isfinite(score):
                raise ValueError(f"Non-finite score in {fold_name}, seed={seed}: {score}")

            run_scores.append(score)
            metric_rows.append(metrics)
            step += 1
            interim = float(np.mean(run_scores) - stability_penalty * np.std(run_scores))
            trial.report(interim, step=step)
            if trial.should_prune():
                raise optuna.TrialPruned()

            del result, mdp
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    robust_score = float(np.mean(run_scores) - stability_penalty * np.std(run_scores))
    trial.set_user_attr("mean_score", float(np.mean(run_scores)))
    trial.set_user_attr("std_score", float(np.std(run_scores)))
    for key in ("sharpe", "cagr", "max_drawdown"):
        trial.set_user_attr(f"mean_{key}", float(np.mean([row[key] for row in metric_rows])))
    return robust_score


def _run_stage(
    stage: str,
    n_trials: int,
    base_config: Dict[str, Any],
    folds: Sequence[Fold],
    budget: TuningBudget,
    run_single_ucb: Callable[[pd.DataFrame, pd.DataFrame, Any, Dict[str, Any]], Dict[str, Any]],
    mdp_factory: Callable[[Dict[str, Any]], Any],
    set_seed: Callable[[int], None],
    output_dir: Path,
) -> Tuple[Any, Dict[str, Any]]:
    db_path = (output_dir / "ucb_vae_tuning.sqlite3").resolve().as_posix()
    sampler = optuna.samplers.TPESampler(seed=42, multivariate=True)
    pruner = optuna.pruners.MedianPruner(n_startup_trials=5, n_warmup_steps=len(budget.seeds))
    study = optuna.create_study(
        study_name=f"ucb_vae_stage_{stage.lower()}",
        direction="maximize",
        sampler=sampler,
        pruner=pruner,
        storage=f"sqlite:///{db_path}",
        load_if_exists=True,
    )

    def objective(trial: Any) -> float:
        config = copy.deepcopy(base_config)
        config.update(_suggest_stage_parameters(trial, stage))
        config["episodes"] = int(budget.episodes_per_trial)
        # Validation is scored once after training, not after every episode.
        config["evaluate_each_episode"] = False
        trial.set_user_attr("resolved_config", config)
        return _evaluate_config(
            trial, config, folds, budget.seeds, budget.stability_penalty,
            run_single_ucb, mdp_factory, set_seed,
        )

    study.optimize(
        objective,
        n_trials=int(n_trials),
        timeout=budget.timeout_per_stage,
        gc_after_trial=True,
        show_progress_bar=True,
    )
    best_config = copy.deepcopy(study.best_trial.user_attrs["resolved_config"])
    study.trials_dataframe().to_csv(output_dir / f"stage_{stage.lower()}_trials.csv", index=False)
    return study, best_config


def run_three_stage_tuning(
    price_history: pd.DataFrame,
    base_config: Dict[str, Any],
    run_single_ucb: Callable[[pd.DataFrame, pd.DataFrame, Any, Dict[str, Any]], Dict[str, Any]],
    mdp_factory: Callable[[Dict[str, Any]], Any],
    set_seed: Callable[[int], None],
    budget: Optional[TuningBudget] = None,
    output_dir: str = "./tuning_results/ucb_vae",
) -> Dict[str, Any]:
    """Run SARSA, VAE, then UCB tuning while freezing earlier best values."""

    if optuna is None:
        raise ImportError("Optuna is missing. Run: pip install optuna")

    budget = budget or TuningBudget()
    target_dir = Path(output_dir)
    target_dir.mkdir(parents=True, exist_ok=True)
    folds = build_fixed_walk_forward_folds(price_history)

    study_a, config_a = _run_stage(
        "A", budget.stage_a_trials, base_config, folds, budget,
        run_single_ucb, mdp_factory, set_seed, target_dir,
    )
    study_b, config_b = _run_stage(
        "B", budget.stage_b_trials, config_a, folds, budget,
        run_single_ucb, mdp_factory, set_seed, target_dir,
    )
    study_c, config_c = _run_stage(
        "C", budget.stage_c_trials, config_b, folds, budget,
        run_single_ucb, mdp_factory, set_seed, target_dir,
    )

    # Final training should use the notebook's normal episode count and may
    # collect a learning curve again.
    final_config = copy.deepcopy(config_c)
    final_config["episodes"] = int(base_config["episodes"])
    final_config["evaluate_each_episode"] = True

    summary = {
        "budget": asdict(budget),
        "folds": [name for name, _, _ in folds],
        "stage_a_best_value": float(study_a.best_value),
        "stage_b_best_value": float(study_b.best_value),
        "stage_c_best_value": float(study_c.best_value),
        "stage_a_config": config_a,
        "stage_b_config": config_b,
        "best_config": final_config,
    }
    with (target_dir / "best_config.json").open("w", encoding="utf-8") as file:
        json.dump(summary, file, ensure_ascii=False, indent=2)

    return {
        "studies": {"A": study_a, "B": study_b, "C": study_c},
        "summary": summary,
        "best_config": final_config,
        "folds": folds,
    }

