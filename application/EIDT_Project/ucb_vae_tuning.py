"""Three-stage hyperparameter search for the notebook's SARSA UCB-VAE model.

The search deliberately treats the existing training function and the supplied
train/test DataFrames as immutable.  It only builds copied configuration
dictionaries and calls ``_run_single_ucb`` from the notebook.
"""

from __future__ import annotations

import copy
import gc
import json
import math
import random
from pathlib import Path
from typing import Any, Callable, Dict, Mapping, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import torch


RunSingleUCB = Callable[[pd.DataFrame, pd.DataFrame, Any, Dict[str, Any]], Dict[str, Any]]
MetricFn = Callable[..., Any]


def _log_uniform(rng: np.random.Generator, low: float, high: float) -> float:
    return float(math.exp(rng.uniform(math.log(low), math.log(high))))


def _sample_stage_1(rng: np.random.Generator) -> Dict[str, Any]:
    """SARSA/Q-network parameters; the training implementation is unchanged."""
    return {
        "gamma": float(rng.uniform(0.90, 0.995)),
        "alpha": float(rng.uniform(0.10, 1.00)),
        "nn_lr": _log_uniform(rng, 1e-5, 1e-3),
        "nn_epochs": int(rng.choice([1, 2, 3, 4])),
        "batch_size": int(rng.choice([32, 64, 128, 256])),
    }


def _sample_stage_2(rng: np.random.Generator) -> Dict[str, Any]:
    """VAE parameters, searched after locking the best stage-1 config."""
    return {
        "vae_latent_dim": int(rng.choice([4, 8, 16, 32])),
        "vae_lr": _log_uniform(rng, 1e-5, 3e-3),
        "vae_beta_kl": _log_uniform(rng, 1e-5, 1e-1),
        "vae_batch_size": int(rng.choice([32, 64, 128, 256])),
        "vae_updates_per_q_batch": int(rng.choice([1, 2, 3])),
        "vae_replay_capacity": int(rng.choice([5_000, 10_000, 25_000, 50_000])),
        "bootstrap_trajectories": int(rng.choice([2, 3, 5, 8])),
        "bootstrap_vae_updates": int(rng.choice([20, 50, 100, 200])),
    }


def _sample_stage_3(rng: np.random.Generator) -> Dict[str, Any]:
    """UCB exploration parameters, searched after locking stages 1 and 2."""
    return {
        "beta": _log_uniform(rng, 0.01, 1.0),
        "beta_decay": float(rng.uniform(0.90, 0.999)),
        "beta_min": _log_uniform(rng, 0.001, 0.10),
        "delta": 1.0,
        "lam": _log_uniform(rng, 0.10, 10.0),
    }


STAGE_SAMPLERS = {
    "stage_1_sarsa": _sample_stage_1,
    "stage_2_vae": _sample_stage_2,
    "stage_3_ucb": _sample_stage_3,
}


def _frame_fingerprint(frame: pd.DataFrame) -> Tuple[Any, ...]:
    """Cheap immutable-data guard used before and after the complete search."""
    hashed = pd.util.hash_pandas_object(frame, index=True).to_numpy(dtype=np.uint64)
    return frame.shape, tuple(frame.columns), str(frame.index.dtype), int(hashed.sum(dtype=np.uint64))


def _set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _score_summary(summary: Mapping[str, float], objective: str) -> float:
    if objective == "profit":
        return float(summary["mean_final_profit"] - 0.25 * summary["std_final_profit"])
    if objective == "sharpe":
        return float(summary["mean_sharpe"] - 0.25 * summary["std_sharpe"])
    if objective == "balanced":
        return float(
            summary["mean_sharpe"]
            - 0.25 * summary["std_sharpe"]
            + 0.002 * summary["mean_arr"]
            - 0.01 * summary["mean_max_drawdown"]
        )
    raise ValueError("objective must be one of: 'balanced', 'sharpe', 'profit'.")


def _evaluate_config(
    config: Dict[str, Any],
    train_series: pd.DataFrame,
    test_series: pd.DataFrame,
    mdp: Any,
    seeds: Sequence[int],
    run_single_ucb_fn: RunSingleUCB,
    calculate_sharpe_ratio_fn: MetricFn,
    calculate_max_drawdown_fn: MetricFn,
    agent_annual_return_fn: MetricFn,
    objective: str,
) -> Dict[str, float]:
    profits, sharpes, drawdowns, annual_returns, rois = [], [], [], [], []
    start_date = test_series.iloc[0]["time"]
    end_date = test_series.iloc[-1]["time"]

    for seed in seeds:
        _set_seed(int(seed))
        run_result = run_single_ucb_fn(train_series, test_series, mdp, copy.deepcopy(config))
        portfolio = np.asarray(run_result["portfolio_history"], dtype=np.float64)
        profit = float(run_result["final_profit"])
        arr, roi = agent_annual_return_fn(
            float(mdp.balance_init),
            float(mdp.balance_init) + profit,
            start_date,
            end_date,
        )
        profits.append(profit)
        sharpes.append(float(calculate_sharpe_ratio_fn(portfolio)))
        drawdowns.append(float(calculate_max_drawdown_fn(portfolio)))
        annual_returns.append(float(arr))
        rois.append(float(roi))
        del run_result
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    summary = {
        "mean_final_profit": float(np.mean(profits)),
        "std_final_profit": float(np.std(profits)),
        "mean_sharpe": float(np.mean(sharpes)),
        "std_sharpe": float(np.std(sharpes)),
        "mean_max_drawdown": float(np.mean(drawdowns)),
        "mean_arr": float(np.mean(annual_returns)),
        "mean_roi": float(np.mean(rois)),
    }
    summary["score"] = _score_summary(summary, objective)
    return summary


def _json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): _json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(v) for v in value]
    if isinstance(value, np.generic):
        return value.item()
    return value


def run_three_stage_ucb_vae_tuning(
    train_series: pd.DataFrame,
    test_series: pd.DataFrame,
    base_config: Mapping[str, Any],
    mdp: Any,
    run_single_ucb_fn: RunSingleUCB,
    calculate_sharpe_ratio_fn: MetricFn,
    calculate_max_drawdown_fn: MetricFn,
    agent_annual_return_fn: MetricFn,
    stage_trials: Sequence[int] = (20, 30, 20),
    n_seeds: int = 3,
    seed: int = 42,
    objective: str = "balanced",
    output_dir: Optional[str] = "./tuning_results/ucb_vae",
) -> Dict[str, Any]:
    """Run sequential SARSA -> VAE -> UCB random search.

    ``train_series`` and ``test_series`` are passed through unchanged and are
    guarded by fingerprints.  ``base_config`` is copied, never mutated.  The
    best configuration of each stage is locked before the next stage starts.
    """
    if len(stage_trials) != 3 or any(int(n) <= 0 for n in stage_trials):
        raise ValueError("stage_trials must contain three positive integers.")
    if int(n_seeds) <= 0:
        raise ValueError("n_seeds must be positive.")
    if train_series.empty or test_series.empty:
        raise ValueError("train_series and test_series must not be empty.")

    train_before = _frame_fingerprint(train_series)
    test_before = _frame_fingerprint(test_series)
    original_config = copy.deepcopy(dict(base_config))
    locked_config = copy.deepcopy(original_config)
    seed_values = [int(seed) + i for i in range(int(n_seeds))]
    all_stage_frames: Dict[str, pd.DataFrame] = {}
    best_by_stage: Dict[str, Dict[str, Any]] = {}

    destination = Path(output_dir) if output_dir is not None else None
    if destination is not None:
        destination.mkdir(parents=True, exist_ok=True)

    for stage_index, (stage_name, sampler) in enumerate(STAGE_SAMPLERS.items()):
        rng = np.random.default_rng(int(seed) + 10_000 * stage_index)
        rows = []
        print(f"\n=== {stage_name}: {int(stage_trials[stage_index])} trials ===")

        for trial_index in range(int(stage_trials[stage_index])):
            sampled = sampler(rng)
            candidate = copy.deepcopy(locked_config)
            candidate.update(sampled)
            metrics = _evaluate_config(
                candidate,
                train_series,
                test_series,
                mdp,
                seed_values,
                run_single_ucb_fn,
                calculate_sharpe_ratio_fn,
                calculate_max_drawdown_fn,
                agent_annual_return_fn,
                objective,
            )
            row = {"stage": stage_name, "trial": trial_index + 1, **sampled, **metrics}
            rows.append(row)
            print(
                f"[{trial_index + 1:03d}/{int(stage_trials[stage_index]):03d}] "
                f"score={metrics['score']:.6f} | "
                f"profit={metrics['mean_final_profit']:.2f} | "
                f"sharpe={metrics['mean_sharpe']:.4f} | "
                f"mdd={metrics['mean_max_drawdown']:.2f}%"
            )

            if destination is not None:
                pd.DataFrame(rows).to_csv(destination / f"{stage_name}_trials.csv", index=False)

        stage_frame = pd.DataFrame(rows).sort_values("score", ascending=False).reset_index(drop=True)
        best_record = max(rows, key=lambda item: float(item["score"]))
        sampled_keys = set(best_record) - {
            "stage",
            "trial",
            "score",
            "mean_final_profit",
            "std_final_profit",
            "mean_sharpe",
            "std_sharpe",
            "mean_max_drawdown",
            "mean_arr",
            "mean_roi",
        }
        best_params = {key: best_record[key] for key in sampled_keys}
        locked_config.update(best_params)
        all_stage_frames[stage_name] = stage_frame
        best_by_stage[stage_name] = {
            "params": copy.deepcopy(best_params),
            "metrics": {
                key: best_record[key]
                for key in (
                    "score",
                    "mean_final_profit",
                    "std_final_profit",
                    "mean_sharpe",
                    "std_sharpe",
                    "mean_max_drawdown",
                    "mean_arr",
                    "mean_roi",
                )
            },
        }
        print(f"Best {stage_name}: score={best_record['score']:.6f}")

    if _frame_fingerprint(train_series) != train_before:
        raise RuntimeError("train_series was modified during tuning.")
    if _frame_fingerprint(test_series) != test_before:
        raise RuntimeError("test_series was modified during tuning.")
    if dict(base_config) != original_config:
        raise RuntimeError("base_config was modified during tuning.")

    result = {
        "best_config": copy.deepcopy(locked_config),
        "best_by_stage": best_by_stage,
        "stage_results": all_stage_frames,
        "objective": objective,
        "seeds": seed_values,
    }
    if destination is not None:
        summary = {
            "best_config": result["best_config"],
            "best_by_stage": result["best_by_stage"],
            "objective": objective,
            "seeds": seed_values,
        }
        (destination / "best_config.json").write_text(
            json.dumps(_json_safe(summary), ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        print(f"\nSaved tuning results to: {destination.resolve()}")
    return result
