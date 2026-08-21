import json
import math
import unittest
from pathlib import Path

import numpy as np
import pandas as pd


NOTEBOOK_PATH = Path(__file__).resolve().parents[1] / "Test_Three_Strategy_6Stocks.ipynb"


def load_notebook_namespace():
    notebook = json.loads(NOTEBOOK_PATH.read_text(encoding="utf-8"))
    namespace = {"__name__": "training_notebook_under_test"}
    wanted = ("# CELL 1", "# CELL 2", "# CELL 3", "# CELL 4", "# CELL 5")
    selected = {}
    for cell in notebook["cells"]:
        source = "".join(cell.get("source", []))
        for marker in wanted:
            if source.startswith(marker):
                selected[marker] = source
    missing = set(wanted) - set(selected)
    if missing:
        raise AssertionError(f"Missing notebook cells: {sorted(missing)}")
    for marker in wanted:
        if marker == "# CELL 5":
            selected[marker] = selected[marker].split("\nif RUN_TRAINING:", 1)[0]
        exec(compile(selected[marker], str(NOTEBOOK_PATH), "exec"), namespace)
    return namespace


NS = load_notebook_namespace()
TradingEnv = NS["TradingEnv"]
ACTION_VALUES = NS["ACTION_VALUES"]
TRANSACTION_FEE = NS["TRANSACTION_FEE"]
W_STABILITY = NS["W_STABILITY"]
period_metrics = NS["period_metrics"]
evaluation_dates = NS["evaluation_dates"]
epsilon_action = NS["epsilon_action"]
action_scores = NS["action_scores"]
collect_epsilon_episode = NS["collect_epsilon_episode"]
collect_ucb_episode = NS["collect_ucb_episode"]
ucb_training_controls = NS["ucb_training_controls"]
torch = NS["torch"]


class IdentityScaler:
    @staticmethod
    def transform(values):
        return np.asarray(values, dtype=np.float32)


class FixedQ(torch.nn.Module):
    def forward(self, states):
        descending = torch.arange(11, 0, -1, dtype=torch.float32, device=states.device)
        return descending.repeat(states.shape[0], 1)


class ZeroVAE(torch.nn.Module):
    def compute_u_ep(self, states, actions, reduction):
        return torch.zeros(states.shape[0], dtype=torch.float32, device=states.device)


class ZeroCost(torch.nn.Module):
    def forward(self, states, actions):
        return torch.zeros(states.shape[0], dtype=torch.float32, device=states.device)


def market_frame(prices):
    size = len(prices)
    return pd.DataFrame(
        {
            "time": pd.date_range("2024-01-01", periods=size, freq="D"),
            "close": prices,
            "MACD": np.zeros(size),
            "RSI": np.full(size, 50.0),
            "CCI": np.zeros(size),
            "ADX": np.full(size, 20.0),
        }
    )


class TradingEnvironmentTests(unittest.TestCase):
    def test_initial_mask_rejects_sell_and_keeps_hold(self):
        env = TradingEnv(market_frame([100.0, 101.0]), reward_shaping=False)
        mask = env.valid_action_mask()
        self.assertTrue(mask[ACTION_VALUES == 0].item())
        self.assertFalse(mask[ACTION_VALUES < 0].any())
        self.assertTrue(mask[ACTION_VALUES > 0].all())

    def test_insufficient_cash_is_masked(self):
        env = TradingEnv(market_frame([100.0, 101.0]), reward_shaping=False, initial_cash=150.0)
        mask = env.valid_action_mask()
        self.assertTrue(mask[ACTION_VALUES == 1].item())
        self.assertFalse(mask[ACTION_VALUES > 1].any())

    def test_both_policy_selectors_respect_environment_mask(self):
        env = TradingEnv(market_frame([100.0, 101.0]), reward_shaping=False)
        state = env.reset()
        mask = env.valid_action_mask()
        epsilon_selected, _ = epsilon_action(FixedQ(), state, 0.0, False, IdentityScaler(), mask)
        ucb_selected, _, _ = action_scores(
            FixedQ(), ZeroVAE(), ZeroCost(), state, 0.03,
            {"kl_reduction": "sum", "q_scale_floor": 1e-3}, IdentityScaler(), mask,
        )
        self.assertEqual(epsilon_selected, 0)
        self.assertEqual(ucb_selected, 0)

    def test_episode_collectors_do_not_generate_invalid_actions(self):
        frame = market_frame([100.0, 101.0, 99.0, 102.0])
        epsilon_env = TradingEnv(frame, reward_shaping=True)
        epsilon_trajectory = collect_epsilon_episode(
            epsilon_env, FixedQ(), epsilon=0.0, scaler=IdentityScaler()
        )
        ucb_env = TradingEnv(frame, reward_shaping=True)
        ucb_trajectory = collect_ucb_episode(
            ucb_env, FixedQ(), ZeroVAE(), ZeroCost(), beta=0.03,
            epsilon_ucb=1.0, config={"kl_reduction": "sum", "q_scale_floor": 1e-3},
            scaler=IdentityScaler(),
        )
        self.assertEqual(epsilon_env.invalid_action_count, 0)
        self.assertEqual(ucb_env.invalid_action_count, 0)
        self.assertTrue(all(ACTION_VALUES[index] >= 0 for index in epsilon_trajectory[3]))

    def test_beta_waits_for_coverage_and_warmup_epsilon_decays(self):
        config = NS["MODEL_CONFIGS"]["UNCERTAINTY_AWARE_UCB_009"]
        beta_without_coverage, epsilon_start, coverage = ucb_training_controls(
            episode=0, total_transitions=10_000, visited_actions=set(), config=config
        )
        beta_with_coverage, epsilon_after_warmup, covered = ucb_training_controls(
            episode=15, total_transitions=10_000, visited_actions=set(range(8)), config=config
        )
        self.assertEqual(beta_without_coverage, config["beta_0"])
        self.assertEqual(epsilon_start, config["ucb_epsilon_init"])
        self.assertEqual(coverage, 0.0)
        self.assertLess(beta_with_coverage, config["beta_0"])
        self.assertEqual(epsilon_after_warmup, config["ucb_epsilon_min"])
        self.assertGreaterEqual(covered, config["beta_min_action_coverage"])

    def test_requested_and_executed_actions_are_reported(self):
        env = TradingEnv(market_frame([100.0, 101.0]), reward_shaping=False)
        _, _, _, info = env.step(-5)
        self.assertEqual(info["requested_action"], -5)
        self.assertEqual(info["executed_action"], 0)
        self.assertTrue(info["invalid_action"])
        self.assertEqual(env.diagnostics()["invalid_action_count"], 1)

    def test_valid_trade_preserves_cash_position_and_fee_accounting(self):
        env = TradingEnv(market_frame([100.0, 110.0]), reward_shaping=False)
        _, _, _, info = env.step(2)
        expected_cash = 1_000.0 - 2 * 100.0 * (1.0 + TRANSACTION_FEE)
        self.assertAlmostEqual(info["cash"], expected_cash)
        self.assertEqual(info["position"], 2)
        self.assertEqual(info["requested_action"], info["executed_action"])
        self.assertFalse(info["invalid_action"])

    def test_stability_penalty_is_zero_without_exposure(self):
        env = TradingEnv(market_frame([100.0, 110.0]), reward_shaping=True)
        _, reward, _, info = env.step(0)
        self.assertEqual(info["exposure"], 0.0)
        self.assertEqual(info["stability_penalty"], 0.0)
        self.assertEqual(reward, 0.0)

    def test_stability_penalty_scales_with_exposure(self):
        env = TradingEnv(market_frame([100.0, 110.0]), reward_shaping=True)
        _, reward, _, info = env.step(1)
        expected_penalty = 0.1 * 0.1
        self.assertAlmostEqual(info["stability_penalty"], expected_penalty)
        self.assertAlmostEqual(reward, info["portfolio_return"] - W_STABILITY * expected_penalty)


class MetricTests(unittest.TestCase):
    def test_dates_include_train_test_boundary(self):
        train = market_frame([90.0, 100.0])
        test = market_frame([101.0, 102.0, 103.0])
        dates = evaluation_dates(test, train.iloc[-1])
        self.assertEqual(len(dates), len(test) + 1)
        self.assertEqual(dates[0], pd.Timestamp(train.iloc[-1]["time"]))

    def test_metric_rejects_portfolio_date_mismatch(self):
        with self.assertRaisesRegex(ValueError, "length mismatch"):
            period_metrics(np.asarray([1_000.0, 1_001.0]), [pd.Timestamp("2024-01-01")])

    def test_no_trade_is_collapse_and_sharpe_is_nan(self):
        metrics = period_metrics(
            np.asarray([1_000.0, 1_000.0, 1_000.0]),
            pd.date_range("2024-01-01", periods=3),
            diagnostics={"trade_count": 0},
        )
        self.assertTrue(metrics["no_trade"])
        self.assertTrue(metrics["collapse"])
        self.assertTrue(math.isnan(metrics["sharpe"]))

    def test_nearly_flat_portfolio_is_collapse(self):
        portfolio = np.full(21, 1_000.0)
        portfolio[-1] = 1_000.01
        metrics = period_metrics(
            portfolio,
            pd.date_range("2024-01-01", periods=len(portfolio)),
            diagnostics={"trade_count": 1},
        )
        self.assertGreaterEqual(metrics["flat_step_ratio"], 0.95)
        self.assertTrue(metrics["collapse"])
        self.assertTrue(math.isnan(metrics["sharpe"]))


if __name__ == "__main__":
    unittest.main()
