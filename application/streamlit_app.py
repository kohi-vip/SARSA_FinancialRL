from __future__ import annotations

import hashlib
import re
import sys
from dataclasses import dataclass
from io import BytesIO
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import streamlit as st
import torch

try:
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
except ImportError:  # pragma: no cover - optional dependency
    go = None
    make_subplots = None

try:
    import matplotlib.pyplot as plt
except ImportError:  # pragma: no cover - optional dependency
    plt = None

try:
    import shap
except ImportError:  # pragma: no cover - optional dependency
    shap = None


def discover_project_root() -> Path:
    """Tìm thư mục gốc dự án dù file nằm trong /application hay thư mục gốc."""
    current_file = Path(__file__).resolve()
    candidates = [
        current_file.parent,
        current_file.parent.parent,
        Path.cwd().resolve(),
        Path.cwd().resolve().parent,
    ]
    for candidate in candidates:
        if (candidate / "agents").exists() and (candidate / "environments").exists():
            return candidate
    # Tương thích cấu trúc cũ: application/streamlit_app.py
    return current_file.parent.parent


ROOT_DIR = discover_project_root()
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from agents.d_sarsa.d_sarsa import Qsa  # noqa: E402
from environments.stock_trading_env.mdp import StockTradingMDP  # noqa: E402


MODELS_DIR = ROOT_DIR / "models"
TEST_DATA_DIR = ROOT_DIR / "data" / "data_storer" / "data_research" / "test"
REQUIRED_COLUMNS = ["time", "open", "high", "low", "close", "volume", "MACD", "RSI", "CCI", "ADX"]
STATE_COLUMNS = ["close", "balance", "shares", "MACD", "RSI", "CCI", "ADX"]
FEATURE_NAMES = ["Close", "Balance", "Position", "MACD", "RSI", "CCI", "ADX"]
RDX_COMPONENTS = ["Profit", "Risk", "Trend", "Stability"]
PHASE_PERIODS = {"phase_1": "2013–2017", "phase_2": "2015–2019", "phase_3": "2017–2021"}


@dataclass
class DemoResult:
    # frame chứa chuỗi trạng thái s_0 ... s_T do môi trường trả về.
    frame: pd.DataFrame
    # market_frame giữ nguyên thứ tự dòng của CSV giống notebook.
    market_frame: pd.DataFrame
    # transition_frame phân biệt rõ lệnh policy yêu cầu và lượng thực sự khớp.
    transition_frame: pd.DataFrame
    metrics: dict[str, float | int | str]
    model_name: str
    dataset_name: str
    # actions là hành động policy/greedy mà model yêu cầu; XAI giải thích mảng này.
    actions: np.ndarray
    # executed_actions là thay đổi vị thế thực tế, suy ra từ shares(t+1)-shares(t).
    executed_actions: np.ndarray
    rewards: np.ndarray


@dataclass
class RDXResult:
    q_values: np.ndarray
    q_components: np.ndarray
    critical_points: dict[str, list[int]]
    critical_indices: list[int]
    critical_table: pd.DataFrame
    weights: tuple[float, float, float, float]
    alpha: float


@dataclass
class SHAPResult:
    values: np.ndarray
    data: np.ndarray
    base_value: float
    sample_indices: np.ndarray
    dates: list[str]
    background_size: int
    nsamples: int
    sampling_mode: str


def inject_css() -> None:
    st.markdown(
        """
        <style>
            .stApp {
                background: radial-gradient(circle at top left, rgba(10, 25, 47, 0.95), rgba(5, 8, 20, 1) 70%);
                color: #e5eef9;
            }
            .hero-card {
                padding: 1.4rem 1.5rem;
                border-radius: 22px;
                background: linear-gradient(135deg, rgba(18, 33, 60, 0.95), rgba(9, 15, 31, 0.92));
                border: 1px solid rgba(143, 178, 221, 0.20);
                box-shadow: 0 18px 60px rgba(0, 0, 0, 0.30);
            }
            .hero-title {
                font-size: 2.1rem;
                font-weight: 800;
                letter-spacing: 0.2px;
                margin-bottom: 0.35rem;
            }
            .hero-subtitle {
                color: #a9bdd8;
                font-size: 0.98rem;
                line-height: 1.55;
            }
            .badge-row {
                display: flex;
                gap: 0.65rem;
                flex-wrap: wrap;
                margin-top: 0.9rem;
            }
            .badge {
                display: inline-block;
                padding: 0.38rem 0.72rem;
                border-radius: 999px;
                background: rgba(74, 110, 165, 0.18);
                border: 1px solid rgba(74, 110, 165, 0.30);
                color: #d7e7ff;
                font-size: 0.82rem;
                font-weight: 600;
            }
            .explain-card {
                padding: 1rem 1.1rem;
                border-radius: 16px;
                background: rgba(255,255,255,0.04);
                border: 1px solid rgba(255,255,255,0.08);
                margin: 0.3rem 0 0.9rem 0;
            }
            div[data-testid="metric-container"] {
                background: rgba(255, 255, 255, 0.04);
                border: 1px solid rgba(255, 255, 255, 0.08);
                padding: 0.8rem 0.9rem;
                border-radius: 16px;
            }
            section[data-testid="stSidebar"] {
                background: rgba(6, 10, 20, 0.92);
                border-right: 1px solid rgba(255, 255, 255, 0.08);
            }
            .block-container {
                padding-top: 1.25rem;
                padding-bottom: 2rem;
            }
            h1, h2, h3, h4, h5, h6, p, li, label, span {
                color: #e7eefb;
            }
            .stTabs [data-baseweb="tab-list"] { gap: 0.35rem; }
            .stTabs [data-baseweb="tab"] {
                background: rgba(255, 255, 255, 0.04);
                border-radius: 12px 12px 0 0;
                padding: 0.6rem 0.9rem;
            }
        </style>
        """,
        unsafe_allow_html=True,
    )


@st.cache_data(show_spinner=False)
def list_checkpoints() -> list[str]:
    if not MODELS_DIR.exists():
        return []
    return sorted(p.name for p in MODELS_DIR.glob("*.pth"))


@st.cache_data(show_spinner=False)
def list_sarsa_checkpoints() -> list[str]:
    return [name for name in list_checkpoints() if name.lower().startswith("sarsa_")]


@st.cache_data(show_spinner=False)
def list_datasets() -> list[str]:
    if not TEST_DATA_DIR.exists():
        return []
    return sorted(p.name for p in TEST_DATA_DIR.glob("*.csv"))


def _read_csv_bytes(uploaded_csv: bytes) -> pd.DataFrame:
    return pd.read_csv(BytesIO(uploaded_csv))


@st.cache_data(show_spinner=False)
def load_market_data(source_name: str | None = None, uploaded_csv: bytes | None = None) -> pd.DataFrame:
    """Đọc dữ liệu theo đúng thứ tự CSV để khớp notebook.

    Phiên bản trước tự động sort theo ngày và drop các dòng lỗi. Hai thao tác đó có
    thể làm thay đổi chuỗi state mà model quan sát, dẫn tới action/PnL khác notebook.
    Ở đây dữ liệu không bị sắp xếp lại hay loại bỏ âm thầm; dữ liệu không hợp lệ sẽ
    được báo lỗi rõ ràng.
    """
    if uploaded_csv is not None:
        df = _read_csv_bytes(uploaded_csv)
    elif source_name:
        df = pd.read_csv(TEST_DATA_DIR / source_name)
    else:
        raise ValueError("Chưa có nguồn dữ liệu đầu vào.")

    if "date" in df.columns and "time" not in df.columns:
        df = df.rename(columns={"date": "time"})

    missing = [column for column in REQUIRED_COLUMNS if column not in df.columns]
    if missing:
        raise ValueError(f"Dataset thiếu các cột bắt buộc: {', '.join(missing)}")

    df = df.copy().reset_index(drop=True)
    source_rows = np.arange(len(df), dtype=int)

    parsed_time = pd.to_datetime(df["time"], errors="coerce")
    if parsed_time.isna().any():
        bad_rows = source_rows[parsed_time.isna().to_numpy()][:10].tolist()
        raise ValueError(f"Cột time không đọc được tại các dòng CSV: {bad_rows}")

    numeric_columns = ["open", "high", "low", "close", "volume", "MACD", "RSI", "CCI", "ADX"]
    for column in numeric_columns:
        df[column] = pd.to_numeric(df[column], errors="coerce")

    state_columns = ["close", "MACD", "RSI", "CCI", "ADX"]
    invalid_mask = df[state_columns].isna().any(axis=1)
    if invalid_mask.any():
        bad_rows = source_rows[invalid_mask.to_numpy()][:10].tolist()
        raise ValueError(
            "Dataset có NaN/giá trị không phải số trong state tại các dòng CSV "
            f"{bad_rows}. Streamlit không tự drop để tránh làm lệch kết quả notebook."
        )

    # Chỉ chuẩn hóa cách hiển thị ngày; tuyệt đối giữ nguyên thứ tự dòng đầu vào.
    df["time"] = parsed_time.dt.strftime("%Y-%m-%d")
    return df


@st.cache_resource(show_spinner=False)
def load_qsa_model(model_name: str) -> Qsa:
    model_path = MODELS_DIR / model_name
    checkpoint = torch.load(model_path, map_location="cpu")
    state_dict = checkpoint.get("q_network", checkpoint) if isinstance(checkpoint, dict) else checkpoint

    model = Qsa(input_size=7, num_classes=11)
    model.load_state_dict(state_dict)
    model.eval()
    return model


def action_label(action: int | float) -> str:
    if pd.isna(action):
        return "Khởi tạo"
    action_int = int(action)
    if action_int == 0:
        return "Giữ"
    if action_int > 0:
        return f"Mua {action_int}"
    return f"Bán {abs(action_int)}"


def action_group(action: int) -> str:
    if action > 0:
        return "Mua"
    if action < 0:
        return "Bán"
    return "Giữ"


def infer_experiment_metadata(filename: str) -> tuple[str | None, str | None, str | None]:
    lower_name = Path(filename).stem.lower()
    ticker_match = re.search(r"(?:sarsa|test)_([a-z0-9]+)_phase_[123]", lower_name)
    phase_match = re.search(r"phase_([123])", lower_name)
    ticker = ticker_match.group(1).upper() if ticker_match else None
    phase = f"phase_{phase_match.group(1)}" if phase_match else None
    return ticker, phase, PHASE_PERIODS.get(phase) if phase else None


def validate_model_dataset_pair(model_name: str, dataset_name: str) -> list[str]:
    model_ticker, model_phase, _ = infer_experiment_metadata(model_name)
    data_ticker, data_phase, _ = infer_experiment_metadata(dataset_name)
    warnings: list[str] = []
    if model_ticker and data_ticker and model_ticker != data_ticker:
        warnings.append(f"Checkpoint {model_ticker} đang chạy trên dữ liệu {data_ticker}.")
    if model_phase and data_phase and model_phase != data_phase:
        warnings.append(f"Checkpoint {model_phase} không cùng giai đoạn với dataset {data_phase}.")
    return warnings


NOTEBOOK_REFERENCE = {
    ("ACB", "phase_1"): (1155.25, 155.25),
    ("FPT", "phase_1"): (1293.63, 293.63),
    ("GAS", "phase_1"): (1148.72, 148.72),
    ("HPG", "phase_1"): (1517.91, 517.91),
    ("SSI", "phase_1"): (1300.03, 300.03),
    ("VCB", "phase_1"): (1456.67, 456.67),
    ("ACB", "phase_2"): (988.76, -11.24),
    ("FPT", "phase_2"): (1150.65, 150.65),
    ("GAS", "phase_2"): (1115.27, 115.27),
    ("HPG", "phase_2"): (1003.05, 3.05),
    ("SSI", "phase_2"): (644.19, -355.81),
    ("VCB", "phase_2"): (1003.93, 3.93),
    ("ACB", "phase_3"): (1113.24, 113.24),
    ("FPT", "phase_3"): (1609.94, 609.94),
    ("GAS", "phase_3"): (1018.76, 18.76),
    ("HPG", "phase_3"): (1427.47, 427.47),
    ("SSI", "phase_3"): (1893.87, 893.87),
    ("VCB", "phase_3"): (1091.37, 91.37),
}


def _fingerprint_array(values: np.ndarray) -> str:
    arr = np.ascontiguousarray(values, dtype=np.float64)
    return hashlib.sha256(arr.tobytes()).hexdigest()[:12]


def _build_transition_frame(
    states: list[list[float]],
    actions: list[int],
    rewards: list[float],
    df: pd.DataFrame,
    state_init: list[float],
) -> tuple[pd.DataFrame, np.ndarray]:
    """Ghép (s_t, a_t, r_t, s_t+1) và suy ra giao dịch thực sự.

    StockTradingMDP trong notebook trả 250 states và 249 actions. Vì vậy action[t]
    là lệnh policy ở s_t, còn thay đổi vị thế được quan sát giữa states[t] và
    states[t+1]. Marker Mua/Bán phải đặt ở ngày của s_t+1 và chỉ được hiển thị
    khi shares thực sự thay đổi.
    """
    state_arr = np.asarray(states, dtype=float)
    requested = np.asarray(actions, dtype=int)
    reward_arr = np.asarray(rewards, dtype=float)
    if state_arr.ndim != 2 or state_arr.shape[1] != len(STATE_COLUMNS):
        raise ValueError(f"States có shape không hợp lệ: {state_arr.shape}")

    # Chuẩn notebook: states[0] chính là state_init và len(states)=len(actions)+1.
    first_is_initial = len(state_arr) > 0 and np.allclose(
        state_arr[0], np.asarray(state_init, dtype=float), rtol=1e-5, atol=1e-6, equal_nan=False
    )
    if first_is_initial:
        transition_count = min(len(requested), len(state_arr) - 1, len(df) - 1)
        pre_states = state_arr[:transition_count]
        post_states = state_arr[1 : transition_count + 1]
        decision_rows = np.arange(transition_count)
        execution_rows = np.arange(1, transition_count + 1)
    else:
        # Fallback cho biến thể environment chỉ trả post-state.
        transition_count = min(len(requested), len(state_arr), len(df) - 1)
        pre_states = np.vstack([np.asarray(state_init, dtype=float), state_arr[: max(0, transition_count - 1)]])
        post_states = state_arr[:transition_count]
        decision_rows = np.arange(transition_count)
        execution_rows = np.arange(1, transition_count + 1)

    requested = requested[:transition_count]
    reward_used = np.full(transition_count, np.nan, dtype=float)
    reward_used[: min(len(reward_arr), transition_count)] = reward_arr[:transition_count]

    pre_shares = pre_states[:, 2]
    post_shares = post_states[:, 2]
    executed = post_shares - pre_shares
    # Environment dùng lượng cổ phiếu rời rạc; làm sạch sai số float rất nhỏ.
    executed[np.isclose(executed, 0.0, atol=1e-9)] = 0.0

    pre_balance = pre_states[:, 1]
    post_balance = post_states[:, 1]
    balance_delta = post_balance - pre_balance
    pre_portfolio = pre_balance + pre_states[:, 0] * pre_shares
    post_portfolio = post_balance + post_states[:, 0] * post_shares
    portfolio_delta = post_portfolio - pre_portfolio

    status = np.where(executed > 0, "Mua khớp", np.where(executed < 0, "Bán khớp", "Không giao dịch"))
    reason: list[str] = []
    for req, qty, shares_before in zip(requested, executed, pre_shares):
        if req < 0 and np.isclose(qty, 0.0):
            reason.append("Lệnh bán không khớp vì vị thế hiện tại bằng 0")
        elif req > 0 and np.isclose(qty, 0.0):
            reason.append("Lệnh mua không làm thay đổi vị thế (kiểm tra số dư/min_balance)")
        elif req == 0 and np.isclose(qty, 0.0):
            reason.append("Policy chọn Giữ")
        elif np.sign(req) != np.sign(qty):
            reason.append("Hướng khớp khác lệnh policy — cần kiểm tra MDP")
        elif not np.isclose(abs(req), abs(qty)):
            reason.append("Khớp một phần do ràng buộc môi trường")
        else:
            reason.append("Khớp đủ")

    implied_price = np.full(transition_count, np.nan, dtype=float)
    nonzero = ~np.isclose(executed, 0.0)
    implied_price[nonzero] = np.abs(balance_delta[nonzero] / executed[nonzero])

    transition = pd.DataFrame(
        {
            "step": np.arange(transition_count, dtype=int),
            "decision_row": decision_rows,
            "execution_row": execution_rows,
            "decision_time": df["time"].iloc[decision_rows].to_numpy(),
            "execution_time": df["time"].iloc[execution_rows].to_numpy(),
            "requested_action": requested,
            "requested_label": [action_label(v) for v in requested],
            "executed_qty": executed,
            "execution_status": status,
            "execution_note": reason,
            "price_before": pre_states[:, 0],
            "execution_price": post_states[:, 0],
            "implied_cash_price": implied_price,
            "shares_before": pre_shares,
            "shares_after": post_shares,
            "balance_before": pre_balance,
            "balance_after": post_balance,
            "balance_delta": balance_delta,
            "portfolio_before": pre_portfolio,
            "portfolio_after": post_portfolio,
            "portfolio_delta": portfolio_delta,
            "reward": reward_used,
        }
    )
    return transition, executed


def simulate_on_dataframe(
    model_name: str,
    data_name: str,
    df: pd.DataFrame,
    balance_init: int,
    min_balance: int,
) -> DemoResult:
    if len(df) < 2:
        raise ValueError("Dataset cần ít nhất 2 dòng để mô phỏng.")

    model = load_qsa_model(model_name)
    mdp = StockTradingMDP(balance_init=balance_init, k=5, min_balance=min_balance)

    first_row = df.iloc[0]
    state_init = [
        float(first_row["close"]),
        float(balance_init),
        0.0,
        float(first_row["MACD"]),
        float(first_row["RSI"]),
        float(first_row["CCI"]),
        float(first_row["ADX"]),
    ]

    def policy_fn(state: list[float], greedy: bool = True, eps: float = 0.0) -> int:
        with torch.no_grad():
            logits = model(torch.tensor(state, dtype=torch.float32)).squeeze()
            return int(torch.argmax(logits).item()) - 5

    states, rewards, actions = mdp.simulate(
        df.iloc[1:].reset_index(drop=True), state_init, policy_fn, True, eps=0.0
    )
    states = list(states)
    rewards = list(rewards)
    actions = list(actions)
    if not states:
        raise ValueError("Môi trường không trả về trạng thái mô phỏng nào.")

    transition_frame, executed_actions = _build_transition_frame(states, actions, rewards, df, state_init)
    action_count = len(transition_frame)
    aligned_actions = np.asarray(actions[:action_count], dtype=int)
    aligned_rewards = transition_frame["reward"].to_numpy(dtype=float)

    state_df = pd.DataFrame(states, columns=STATE_COLUMNS)
    available_states = min(len(state_df), len(df))
    state_df = state_df.iloc[:available_states].copy().reset_index(drop=True)
    state_df.insert(0, "time", df["time"].iloc[:available_states].to_numpy())
    state_df["portfolio"] = state_df["balance"] + state_df["close"] * state_df["shares"]

    # Policy action thuộc s_t; reward và lượng khớp thuộc transition dẫn đến s_t+1.
    state_df["policy_action"] = np.nan
    state_df["executed_qty"] = np.nan
    state_df["reward"] = np.nan
    if action_count:
        state_df.loc[: action_count - 1, "policy_action"] = aligned_actions
        state_df.loc[1:action_count, "executed_qty"] = executed_actions
        state_df.loc[1:action_count, "reward"] = aligned_rewards
    state_df.loc[0, "reward"] = 0.0
    state_df["policy_action_label"] = [action_label(v) for v in state_df["policy_action"]]
    state_df["execution_label"] = np.where(
        state_df["executed_qty"] > 0,
        "Mua khớp",
        np.where(state_df["executed_qty"] < 0, "Bán khớp", "Không giao dịch"),
    )
    # Tương thích các phần XAI cũ: action vẫn là policy action.
    state_df["action"] = state_df["policy_action"]
    state_df["action_label"] = state_df["policy_action_label"]

    portfolio = state_df["portfolio"].to_numpy(dtype=float)
    peak = np.maximum.accumulate(portfolio)
    drawdown = np.where(peak > 0, (peak - portfolio) / peak, 0.0)
    expected_decisions = max(0, len(df) - 1)
    coverage_pct = 100.0 if expected_decisions == 0 else min(100.0, action_count / expected_decisions * 100.0)

    requested_buy = int(np.sum(aligned_actions > 0))
    requested_sell = int(np.sum(aligned_actions < 0))
    requested_hold = int(np.sum(aligned_actions == 0))
    executed_buy = int(np.sum(executed_actions > 0))
    executed_sell = int(np.sum(executed_actions < 0))
    no_trade = int(np.sum(np.isclose(executed_actions, 0.0)))
    rejected = int(np.sum((aligned_actions != 0) & np.isclose(executed_actions, 0.0)))

    total_reward = float(np.nansum(aligned_rewards))
    portfolio_profit = float(portfolio[-1] - portfolio[0])
    reward_gap = float(portfolio_profit - total_reward)
    ticker, phase, _ = infer_experiment_metadata(model_name)
    data_ticker, data_phase, _ = infer_experiment_metadata(data_name)
    pair_matches = (ticker == data_ticker) and (phase == data_phase)
    ref = NOTEBOOK_REFERENCE.get((ticker, phase)) if ticker and phase and pair_matches else None

    metrics: dict[str, float | int | str] = {
        "steps": int(action_count),
        "dataset_rows": int(len(df)),
        "expected_decisions": int(expected_decisions),
        "simulation_states": int(len(state_df)),
        "coverage_pct": float(coverage_pct),
        "ended_early": bool(action_count < expected_decisions),
        "final_portfolio": float(portfolio[-1]),
        "profit": portfolio_profit,
        "total_reward": total_reward,
        "reward_portfolio_gap": reward_gap,
        "max_portfolio": float(np.nanmax(portfolio)),
        "min_portfolio": float(np.nanmin(portfolio)),
        "max_drawdown_pct": float(np.nanmax(drawdown) * 100.0),
        "buy_count": requested_buy,
        "sell_count": requested_sell,
        "hold_count": requested_hold,
        "executed_buy_count": executed_buy,
        "executed_sell_count": executed_sell,
        "no_trade_count": no_trade,
        "rejected_count": rejected,
        "data_fingerprint": _fingerprint_array(df[["close", "MACD", "RSI", "CCI", "ADX"]].to_numpy()),
        "state_fingerprint": _fingerprint_array(state_df[STATE_COLUMNS].to_numpy()),
        "time_monotonic": bool(pd.to_datetime(df["time"], errors="coerce").is_monotonic_increasing),
        "balance_init": int(balance_init),
        "min_balance": int(min_balance),
        "notebook_expected_final": float(ref[0]) if ref else np.nan,
        "notebook_expected_reward": float(ref[1]) if ref else np.nan,
        "notebook_final_gap": float(portfolio[-1] - ref[0]) if ref else np.nan,
        "notebook_reward_gap": float(total_reward - ref[1]) if ref else np.nan,
    }

    return DemoResult(
        frame=state_df,
        market_frame=df.copy().reset_index(drop=True),
        transition_frame=transition_frame,
        metrics=metrics,
        model_name=model_name,
        dataset_name=data_name,
        actions=aligned_actions,
        executed_actions=np.asarray(executed_actions, dtype=float),
        rewards=aligned_rewards,
    )



def build_chart(result: DemoResult):
    """Vẽ giá, giao dịch THỰC SỰ KHỚP và PnL/danh mục.

    Không dùng trực tiếp action policy làm marker Mua/Bán, vì model có thể yêu cầu
    bán khi shares=0 hoặc mua khi không đủ tiền. Các marker được suy ra từ Δshares.
    """
    if go is None or make_subplots is None:
        return None

    market = result.market_frame.copy()
    simulation = result.frame.copy()
    transitions = result.transition_frame.copy()
    market_dates = pd.to_datetime(market["time"], errors="coerce")
    simulation_dates = pd.to_datetime(simulation["time"], errors="coerce")
    transitions["execution_date"] = pd.to_datetime(transitions["execution_time"], errors="coerce")

    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.08, row_heights=[0.62, 0.38])
    fig.add_trace(
        go.Scatter(
            x=market_dates,
            y=market["close"],
            name="Giá",
            mode="lines",
            line=dict(color="#7dd3fc", width=2.2),
            hovertemplate="Ngày: %{x|%Y-%m-%d}<br>Giá: %{y:.4f}<extra></extra>",
        ), row=1, col=1,
    )

    buy = transitions["executed_qty"] > 0
    sell = transitions["executed_qty"] < 0
    rejected = (transitions["requested_action"] != 0) & np.isclose(transitions["executed_qty"], 0.0)

    if buy.any():
        fig.add_trace(
            go.Scatter(
                x=transitions.loc[buy, "execution_date"],
                y=transitions.loc[buy, "execution_price"],
                mode="markers", name="Mua khớp",
                marker=dict(symbol="triangle-up", size=11, color="#22c55e"),
                customdata=np.column_stack([
                    transitions.loc[buy, "requested_action"],
                    transitions.loc[buy, "executed_qty"],
                    transitions.loc[buy, "shares_after"],
                ]),
                hovertemplate=(
                    "Ngày khớp: %{x|%Y-%m-%d}<br>Giá: %{y:.4f}"
                    "<br>Policy yêu cầu: %{customdata[0]}"
                    "<br>Khớp thực tế: +%{customdata[1]}"
                    "<br>Vị thế sau lệnh: %{customdata[2]}<extra></extra>"
                ),
            ), row=1, col=1,
        )
    if sell.any():
        fig.add_trace(
            go.Scatter(
                x=transitions.loc[sell, "execution_date"],
                y=transitions.loc[sell, "execution_price"],
                mode="markers", name="Bán khớp",
                marker=dict(symbol="triangle-down", size=11, color="#ef4444"),
                customdata=np.column_stack([
                    transitions.loc[sell, "requested_action"],
                    transitions.loc[sell, "executed_qty"],
                    transitions.loc[sell, "shares_after"],
                ]),
                hovertemplate=(
                    "Ngày khớp: %{x|%Y-%m-%d}<br>Giá: %{y:.4f}"
                    "<br>Policy yêu cầu: %{customdata[0]}"
                    "<br>Khớp thực tế: %{customdata[1]}"
                    "<br>Vị thế sau lệnh: %{customdata[2]}<extra></extra>"
                ),
            ), row=1, col=1,
        )
    if rejected.any():
        fig.add_trace(
            go.Scatter(
                x=transitions.loc[rejected, "execution_date"],
                y=transitions.loc[rejected, "execution_price"],
                mode="markers", name="Lệnh không khớp",
                visible="legendonly",
                marker=dict(symbol="x", size=8, color="#94a3b8"),
                text=transitions.loc[rejected, "execution_note"],
                hovertemplate="Ngày: %{x|%Y-%m-%d}<br>Giá: %{y:.4f}<br>%{text}<extra></extra>",
            ), row=1, col=1,
        )

    fig.add_trace(
        go.Scatter(
            x=simulation_dates, y=simulation["portfolio"], name="Danh mục / PnL",
            mode="lines", line=dict(color="#f59e0b", width=2.5),
            hovertemplate="Ngày: %{x|%Y-%m-%d}<br>Danh mục: %{y:.2f}<extra></extra>",
        ), row=2, col=1,
    )
    fig.add_hline(y=float(simulation["portfolio"].iloc[0]), line_dash="dash", line_color="#94a3b8", row=2, col=1)

    fig.update_layout(
        template="plotly_dark", height=720, margin=dict(l=10, r=10, t=45, b=10),
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        hovermode="x unified", uirevision=f"{result.model_name}-{result.dataset_name}-executed",
    )
    fig.update_yaxes(title_text="Giá", row=1, col=1)
    fig.update_yaxes(title_text="Giá trị danh mục", row=2, col=1)
    fig.update_xaxes(type="date", showgrid=True, row=1, col=1)
    fig.update_xaxes(type="date", title_text="Ngày", showgrid=True, rangeslider=dict(visible=True, thickness=0.08), row=2, col=1)
    return fig


def build_action_count_chart(result: DemoResult):
    """So sánh số lệnh policy yêu cầu với giao dịch thực sự khớp."""
    if go is None:
        return None
    requested = {
        "Mua": int(np.sum(result.actions > 0)),
        "Bán": int(np.sum(result.actions < 0)),
        "Giữ": int(np.sum(result.actions == 0)),
    }
    executed = {
        "Mua": int(np.sum(result.executed_actions > 0)),
        "Bán": int(np.sum(result.executed_actions < 0)),
        "Giữ": int(np.sum(np.isclose(result.executed_actions, 0.0))),
    }
    categories = ["Mua", "Bán", "Giữ/Không giao dịch"]
    fig = go.Figure()
    fig.add_trace(go.Bar(name="Policy yêu cầu", x=categories, y=[requested["Mua"], requested["Bán"], requested["Giữ"]], text=[requested["Mua"], requested["Bán"], requested["Giữ"]], textposition="outside"))
    fig.add_trace(go.Bar(name="Thực tế khớp", x=categories, y=[executed["Mua"], executed["Bán"], executed["Giữ"]], text=[executed["Mua"], executed["Bán"], executed["Giữ"]], textposition="outside"))
    fig.update_layout(
        template="plotly_dark", barmode="group", height=360,
        title="Policy action và giao dịch thực sự",
        xaxis_title="Nhóm hành động", yaxis_title="Số bước",
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        margin=dict(l=10, r=10, t=60, b=10),
    )
    return fig



def batched_q_values(model: Qsa, states: np.ndarray, batch_size: int = 1024) -> np.ndarray:
    outputs: list[np.ndarray] = []
    with torch.no_grad():
        for start in range(0, len(states), batch_size):
            batch = torch.tensor(states[start : start + batch_size], dtype=torch.float32)
            outputs.append(model(batch).detach().cpu().numpy())
    return np.vstack(outputs).astype(np.float64)


def _moving_average_same(values: np.ndarray, window: int = 10) -> np.ndarray:
    if len(values) >= window:
        return np.convolve(values, np.ones(window) / window, mode="same")[: len(values)]
    return pd.Series(values).rolling(window=window, min_periods=1, center=True).mean().to_numpy()


def identify_critical_points(
    prices: np.ndarray,
    actions: np.ndarray,
    action_change_threshold: int = 3,
    trend_window: int = 10,
    top_k: int = 15,
) -> tuple[dict[str, list[int]], list[int]]:
    """Phiên bản Streamlit của bộ lọc critical points trong notebook RDX/MSX."""
    prices = np.asarray(prices, dtype=float)
    actions = np.asarray(actions, dtype=int)
    n = min(len(prices), len(actions))
    prices = prices[:n]
    actions = actions[:n]
    if n == 0:
        return {"lowest_price": [], "trend_reversal": [], "action_shift": []}, []

    global_min_idx = int(np.argmin(prices))
    running_max = np.maximum.accumulate(prices)
    safe_running_max = np.where(np.abs(running_max) < 1e-12, 1e-12, running_max)
    drawdowns = (prices - running_max) / safe_running_max
    max_dd_idx = int(np.argmin(drawdowns))
    bottoms = [global_min_idx]
    if max_dd_idx != global_min_idx:
        bottoms.append(max_dd_idx)

    order = max(1, min(int(trend_window), max(1, (n - 1) // 2)))
    reversals: list[int] = []
    for index in range(order, n - order):
        local = prices[index - order : index + order + 1]
        if prices[index] >= np.max(local) or prices[index] <= np.min(local):
            reversals.append(index)
    median_price = float(np.median(prices))
    reversals = sorted(set(reversals), key=lambda idx: abs(prices[idx] - median_price), reverse=True)

    action_diffs = np.abs(np.diff(actions))
    shifts = (np.where(action_diffs >= action_change_threshold)[0] + 1).tolist()
    shifts = sorted(shifts, key=lambda idx: action_diffs[idx - 1], reverse=True)

    final_indices: list[int] = []
    for group in (bottoms, reversals, shifts):
        for index in group:
            if index not in final_indices and len(final_indices) < top_k:
                final_indices.append(int(index))

    # Khi dữ liệu ít đảo chiều/hành động không đổi, bổ sung các ngày biến động giá mạnh nhất.
    if len(final_indices) < min(top_k, n):
        abs_returns = np.zeros(n)
        if n > 1:
            denominator = np.where(np.abs(prices[:-1]) < 1e-12, 1e-12, prices[:-1])
            abs_returns[1:] = np.abs(np.diff(prices) / denominator)
        for index in np.argsort(abs_returns)[::-1]:
            if int(index) not in final_indices:
                final_indices.append(int(index))
            if len(final_indices) >= min(top_k, n):
                break

    final_indices = sorted(final_indices)
    bottom_set, reversal_set, shift_set = set(bottoms), set(reversals), set(shifts)
    critical_points = {
        "lowest_price": [idx for idx in final_indices if idx in bottom_set],
        "trend_reversal": [idx for idx in final_indices if idx in reversal_set],
        "action_shift": [idx for idx in final_indices if idx in shift_set],
    }
    return critical_points, final_indices


def analyze_msx(
    selected_vector: np.ndarray,
    compared_vector: np.ndarray,
    component_names: list[str] = RDX_COMPONENTS,
) -> dict[str, Any]:
    """Minimal Sufficient Explanation (MSX+) theo logic trong notebook."""
    delta = np.asarray(selected_vector, dtype=float) - np.asarray(compared_vector, dtype=float)
    pros: list[tuple[int, float]] = []
    disadvantage_sum = 0.0

    for index, value in enumerate(delta):
        if value > 0:
            pros.append((index, float(value)))
        else:
            disadvantage_sum += abs(float(value))

    pros.sort(key=lambda item: item[1], reverse=True)
    msx_plus: list[dict[str, float | str]] = []
    current_sum = 0.0
    is_sufficient = False
    for index, value in pros:
        current_sum += value
        msx_plus.append(
            {
                "component": component_names[index],
                "value": value,
                "contribution_percent": 0.0,
            }
        )
        if current_sum > disadvantage_sum:
            is_sufficient = True
            break

    if current_sum > 0:
        for item in msx_plus:
            item["contribution_percent"] = float(item["value"]) / current_sum * 100.0

    cons = [
        {"component": component_names[index], "value": float(value)}
        for index, value in enumerate(delta)
        if value < 0
    ]
    return {
        "is_dominated": disadvantage_sum == 0,
        "is_sufficient": is_sufficient,
        "msx_plus": msx_plus,
        "advantage": current_sum,
        "disadvantage": disadvantage_sum,
        "cons_details": cons,
        "full_delta": delta,
    }


def _critical_reason(index: int, critical_points: dict[str, list[int]]) -> str:
    reasons: list[str] = []
    if index in critical_points.get("lowest_price", []):
        reasons.append("Đáy/Drawdown")
    if index in critical_points.get("trend_reversal", []):
        reasons.append("Đảo chiều")
    if index in critical_points.get("action_shift", []):
        reasons.append("Đổi hành động")
    return ", ".join(reasons) if reasons else "Biến động mạnh"


def compute_rdx_msx(
    model: Qsa,
    result: DemoResult,
    weights: tuple[float, float, float, float] = (1.0, 0.5, 1.0, 0.1),
    alpha: float = 0.005,
    top_k: int = 15,
    action_change_threshold: int = 3,
    trend_window: int = 10,
) -> RDXResult:
    """Phân rã Q-value theo công thức Balanced RDX của các notebook."""
    decision_frame = result.frame.iloc[: len(result.actions)].copy().reset_index(drop=True)
    states = decision_frame[STATE_COLUMNS].to_numpy(dtype=np.float32)
    actions = result.actions[: len(states)]
    if len(states) == 0:
        raise ValueError("Không có trạng thái quyết định để phân tích XAI.")

    q_values = batched_q_values(model, states)
    prices = decision_frame["close"].to_numpy(dtype=float)
    portfolios = decision_frame["portfolio"].to_numpy(dtype=float)
    portfolios = np.where(np.abs(portfolios) < 10.0, 1000.0, portfolios)

    returns = np.zeros_like(prices, dtype=float)
    if len(prices) > 1:
        safe_prices = np.where(np.abs(prices[:-1]) < 1e-12, 1e-12, prices[:-1])
        returns[1:] = np.diff(prices) / safe_prices
    running_max = np.maximum.accumulate(prices)
    safe_max = np.where(np.abs(running_max) < 1e-12, 1e-12, running_max)
    drawdowns = (prices - running_max) / safe_max
    moving_average = _moving_average_same(prices, window=10)
    trend_sign = np.sign(prices - moving_average)

    _w_profit, w_risk, w_trend, w_stability = weights
    q_normalized = q_values / portfolios[:, None]
    q_components = np.zeros((len(states), 11, 4), dtype=np.float64)

    risk_component = w_risk * -np.abs(drawdowns)
    stability_component = w_stability * -np.abs(returns)
    for action_index, action_value in enumerate(range(-5, 6)):
        if action_value == 0:
            trend_component = np.zeros(len(states), dtype=float)
        else:
            same_direction = np.sign(action_value) == trend_sign
            trend_component = np.where(same_direction, w_trend * alpha, -w_trend * alpha)

        profit_component = q_normalized[:, action_index] - (
            risk_component + trend_component + stability_component
        )
        q_components[:, action_index, 0] = profit_component
        q_components[:, action_index, 1] = risk_component
        q_components[:, action_index, 2] = trend_component
        q_components[:, action_index, 3] = stability_component

    critical_points, critical_indices = identify_critical_points(
        prices,
        actions,
        action_change_threshold=action_change_threshold,
        trend_window=trend_window,
        top_k=top_k,
    )

    rows: list[dict[str, Any]] = []
    for index in critical_indices:
        selected_action = int(actions[index])
        selected_idx = selected_action + 5
        alternatives = np.argsort(q_values[index])[::-1]
        alternative_idx = int(next(idx for idx in alternatives if idx != selected_idx))
        alternative_action = alternative_idx - 5
        msx = analyze_msx(q_components[index, selected_idx], q_components[index, alternative_idx])
        rows.append(
            {
                "index": index,
                "time": decision_frame.iloc[index]["time"],
                "close": float(prices[index]),
                "portfolio": float(portfolios[index]),
                "reason": _critical_reason(index, critical_points),
                "selected_action": selected_action,
                "selected_label": action_label(selected_action),
                "alternative_action": alternative_action,
                "alternative_label": action_label(alternative_action),
                "q_selected": float(q_values[index, selected_idx]),
                "q_alternative": float(q_values[index, alternative_idx]),
                "q_margin": float(q_values[index, selected_idx] - q_values[index, alternative_idx]),
                "msx_plus": ", ".join(str(item["component"]) for item in msx["msx_plus"]) or "Không có",
                "advantage": float(msx["advantage"]),
                "disadvantage": float(msx["disadvantage"]),
                "is_sufficient": bool(msx["is_sufficient"] or msx["is_dominated"]),
            }
        )

    return RDXResult(
        q_values=q_values,
        q_components=q_components,
        critical_points=critical_points,
        critical_indices=critical_indices,
        critical_table=pd.DataFrame(rows),
        weights=weights,
        alpha=alpha,
    )


def build_critical_timeline(result: DemoResult, rdx_result: RDXResult):
    if go is None:
        return None
    decision_frame = result.frame.iloc[: len(result.actions)].reset_index(drop=True)
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=decision_frame["time"],
            y=decision_frame["close"],
            mode="lines",
            name="Giá đóng cửa",
            line=dict(color="#7dd3fc", width=2),
        )
    )

    style_map = {
        "lowest_price": ("Đáy/Drawdown", "diamond", "#ef4444"),
        "trend_reversal": ("Đảo chiều", "circle", "#f59e0b"),
        "action_shift": ("Đổi hành động", "x", "#a78bfa"),
    }
    for key, (name, symbol, color) in style_map.items():
        indices = rdx_result.critical_points.get(key, [])
        if not indices:
            continue
        hover = [
            f"{decision_frame.iloc[idx]['time']}<br>{action_label(result.actions[idx])}<br>Index: {idx}"
            for idx in indices
        ]
        fig.add_trace(
            go.Scatter(
                x=[decision_frame.iloc[idx]["time"] for idx in indices],
                y=[decision_frame.iloc[idx]["close"] for idx in indices],
                mode="markers",
                name=name,
                text=hover,
                hovertemplate="%{text}<extra></extra>",
                marker=dict(symbol=symbol, size=12, color=color, line=dict(width=1, color="#111827")),
            )
        )

    fig.update_layout(
        template="plotly_dark",
        height=440,
        title="Các điểm quyết định quan trọng dùng cho MSX + RDX",
        xaxis_title="Ngày",
        yaxis_title="Giá đóng cửa",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        legend=dict(orientation="h", y=1.12),
        margin=dict(l=10, r=10, t=80, b=10),
    )
    return fig


def best_alternative_index(q_values_at_t: np.ndarray, selected_idx: int) -> int:
    return int(next(idx for idx in np.argsort(q_values_at_t)[::-1] if int(idx) != selected_idx))


def build_rdx_local_chart(result: DemoResult, rdx_result: RDXResult, time_index: int):
    if go is None or make_subplots is None:
        return None
    action = int(result.actions[time_index])
    selected_idx = action + 5
    alternative_idx = best_alternative_index(rdx_result.q_values[time_index], selected_idx)
    alternative_action = alternative_idx - 5

    components = rdx_result.q_components[time_index]
    totals = components.sum(axis=1)
    selected_vector = components[selected_idx]
    alternative_vector = components[alternative_idx]

    fig = make_subplots(
        rows=2,
        cols=1,
        vertical_spacing=0.18,
        subplot_titles=("Q-value chuẩn hóa của 11 hành động", "Phân rã RDX: lựa chọn và phương án tốt thứ hai"),
    )
    action_values = list(range(-5, 6))
    bar_colors = ["#60a5fa"] * 11
    bar_colors[selected_idx] = "#22c55e"
    bar_colors[alternative_idx] = "#f59e0b"
    fig.add_trace(
        go.Bar(
            x=[action_label(value) for value in action_values],
            y=totals,
            marker_color=bar_colors,
            text=[f"{value:.4f}" for value in totals],
            textposition="outside",
            name="Q chuẩn hóa",
        ),
        row=1,
        col=1,
    )

    colors = ["#2ecc71", "#e74c3c", "#3498db", "#f39c12"]
    for component_index, component_name in enumerate(RDX_COMPONENTS):
        fig.add_trace(
            go.Bar(
                x=[action_label(action), action_label(alternative_action)],
                y=[selected_vector[component_index], alternative_vector[component_index]],
                name=component_name,
                marker_color=colors[component_index],
                text=[f"{selected_vector[component_index]:.4f}", f"{alternative_vector[component_index]:.4f}"],
                textposition="outside",
            ),
            row=2,
            col=1,
        )

    fig.update_layout(
        template="plotly_dark",
        height=760,
        barmode="relative",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        legend=dict(orientation="h", y=1.04),
        margin=dict(l=10, r=10, t=90, b=10),
    )
    fig.update_yaxes(title_text="Q / Portfolio", row=1, col=1)
    fig.update_yaxes(title_text="Đóng góp", zeroline=True, zerolinecolor="#e5e7eb", row=2, col=1)
    return fig


def build_msx_delta_chart(msx_result: dict[str, Any]):
    if go is None:
        return None
    delta = np.asarray(msx_result["full_delta"], dtype=float)
    msx_names = {str(item["component"]) for item in msx_result["msx_plus"]}
    colors: list[str] = []
    text: list[str] = []
    for component, value in zip(RDX_COMPONENTS, delta):
        if component in msx_names:
            colors.append("#22c55e")
            text.append("MSX+")
        elif value >= 0:
            colors.append("#60a5fa")
            text.append("Lợi thế bổ sung")
        else:
            colors.append("#ef4444")
            text.append("Bất lợi")

    fig = go.Figure(
        go.Bar(
            x=RDX_COMPONENTS,
            y=delta,
            marker_color=colors,
            text=[f"{label}<br>{value:.5f}" for label, value in zip(text, delta)],
            textposition="outside",
        )
    )
    fig.add_hline(y=0, line_color="#e5e7eb")
    fig.update_layout(
        template="plotly_dark",
        height=430,
        title="RDX difference và tập giải thích tối thiểu MSX+",
        xaxis_title="Thành phần",
        yaxis_title="Selected − Alternative",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        margin=dict(l=10, r=10, t=70, b=10),
    )
    return fig


def build_rdx_heatmap(result: DemoResult, rdx_result: RDXResult):
    if go is None or not rdx_result.critical_indices:
        return None
    z_values: list[list[float]] = []
    labels: list[str] = []
    for index in rdx_result.critical_indices:
        action = int(result.actions[index])
        z_values.append(rdx_result.q_components[index, action + 5].tolist())
        labels.append(f"{result.frame.iloc[index]['time']} | {action_label(action)}")

    fig = go.Figure(
        data=go.Heatmap(
            z=np.asarray(z_values),
            x=RDX_COMPONENTS,
            y=labels,
            colorscale="RdBu",
            zmid=0,
            colorbar=dict(title="Đóng góp"),
            hovertemplate="%{y}<br>%{x}: %{z:.5f}<extra></extra>",
        )
    )
    fig.update_layout(
        template="plotly_dark",
        height=max(460, len(labels) * 34),
        title="RDX của hành động được chọn tại các critical points",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        margin=dict(l=10, r=10, t=70, b=10),
    )
    return fig


def choose_sample_indices(total: int, sample_size: int, mode: str, seed: int = 42) -> np.ndarray:
    sample_size = min(max(1, sample_size), total)
    if mode == "Các bước gần nhất":
        return np.arange(total - sample_size, total, dtype=int)
    if mode == "Ngẫu nhiên":
        rng = np.random.default_rng(seed)
        return np.sort(rng.choice(total, size=sample_size, replace=False)).astype(int)
    return np.unique(np.linspace(0, total - 1, sample_size, dtype=int))


def compute_shap_explanations(
    model: Qsa,
    result: DemoResult,
    background_size: int = 40,
    sample_size: int = 30,
    nsamples: int = 120,
    sampling_mode: str = "Phân bố đều theo thời gian",
) -> SHAPResult:
    if shap is None:
        raise ImportError("Chưa cài thư viện shap. Chạy: pip install shap")

    decision_frame = result.frame.iloc[: len(result.actions)].reset_index(drop=True)
    all_states = decision_frame[STATE_COLUMNS].to_numpy(dtype=np.float32)
    if len(all_states) == 0:
        raise ValueError("Không có trạng thái để phân tích SHAP.")

    rng = np.random.default_rng(42)
    background_size = min(max(2, background_size), len(all_states))
    background_indices = np.sort(rng.choice(len(all_states), size=background_size, replace=False))
    sample_indices = choose_sample_indices(len(all_states), sample_size, sampling_mode)
    background = all_states[background_indices]
    samples = all_states[sample_indices]

    def predict_q_greedy(x_batch: np.ndarray) -> np.ndarray:
        x_tensor = torch.tensor(np.asarray(x_batch), dtype=torch.float32)
        with torch.no_grad():
            q_values = model(x_tensor)
            greedy_indices = torch.argmax(q_values, dim=1)
            selected_q = q_values.gather(1, greedy_indices.reshape(-1, 1)).squeeze(1)
        return selected_q.detach().cpu().numpy().astype(np.float32)

    explainer = shap.KernelExplainer(predict_q_greedy, background)
    try:
        raw_values = explainer.shap_values(samples, nsamples=nsamples, silent=True)
    except TypeError:  # Tương thích SHAP phiên bản cũ
        raw_values = explainer.shap_values(samples, nsamples=nsamples)

    if isinstance(raw_values, list):
        values = np.asarray(raw_values[0], dtype=float)
    else:
        values = np.asarray(raw_values, dtype=float)
    values = np.squeeze(values)
    if values.ndim == 1:
        values = values.reshape(1, -1)
    if values.shape[0] != len(samples) and values.shape[1] == len(samples):
        values = values.T

    expected_value = np.asarray(explainer.expected_value).reshape(-1)
    base_value = float(expected_value[0])
    dates = decision_frame.iloc[sample_indices]["time"].astype(str).tolist()
    return SHAPResult(
        values=values,
        data=samples,
        base_value=base_value,
        sample_indices=sample_indices,
        dates=dates,
        background_size=background_size,
        nsamples=nsamples,
        sampling_mode=sampling_mode,
    )


def build_shap_importance_chart(shap_result: SHAPResult):
    if go is None:
        return None
    importance = np.mean(np.abs(shap_result.values), axis=0)
    order = np.argsort(importance)
    fig = go.Figure(
        go.Bar(
            x=importance[order],
            y=[FEATURE_NAMES[index] for index in order],
            orientation="h",
            marker_color="#60a5fa",
            text=[f"{importance[index]:.4f}" for index in order],
            textposition="outside",
        )
    )
    fig.update_layout(
        template="plotly_dark",
        height=430,
        title="Global SHAP importance — mean(|SHAP value|)",
        xaxis_title="Mức ảnh hưởng trung bình lên greedy Q-value",
        yaxis_title="Biến đầu vào",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        margin=dict(l=10, r=10, t=70, b=10),
    )
    return fig


def build_shap_temporal_chart(shap_result: SHAPResult, top_n: int = 4):
    if go is None:
        return None
    importance = np.mean(np.abs(shap_result.values), axis=0)
    top_indices = np.argsort(importance)[::-1][:top_n]
    fig = go.Figure()
    for feature_index in top_indices:
        fig.add_trace(
            go.Scatter(
                x=shap_result.dates,
                y=shap_result.values[:, feature_index],
                mode="lines+markers",
                name=FEATURE_NAMES[feature_index],
            )
        )
    fig.add_hline(y=0, line_dash="dash", line_color="#94a3b8")
    fig.update_layout(
        template="plotly_dark",
        height=450,
        title=f"SHAP theo thời gian — Top {len(top_indices)} biến",
        xaxis_title="Ngày mẫu",
        yaxis_title="SHAP value",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        legend=dict(orientation="h", y=1.12),
        margin=dict(l=10, r=10, t=80, b=10),
    )
    return fig


def create_shap_summary_figure(shap_result: SHAPResult):
    if shap is None or plt is None:
        return None
    plt.figure(figsize=(10, 6))
    shap.summary_plot(
        shap_result.values,
        shap_result.data,
        feature_names=FEATURE_NAMES,
        max_display=7,
        show=False,
        plot_size=None,
        sort=True,
        plot_type="dot",
    )
    figure = plt.gcf()
    figure.suptitle("SHAP beeswarm — tác động lên greedy Q-value", fontsize=14, fontweight="bold")
    figure.tight_layout()
    return figure


def create_shap_local_figure(shap_result: SHAPResult, sample_position: int):
    if shap is None or plt is None:
        return None
    explanation = shap.Explanation(
        values=shap_result.values[sample_position],
        base_values=shap_result.base_value,
        data=shap_result.data[sample_position],
        feature_names=FEATURE_NAMES,
    )
    plt.figure(figsize=(10, 6))
    shap.plots.waterfall(explanation, max_display=7, show=False)
    figure = plt.gcf()
    figure.suptitle(
        f"Local SHAP — {shap_result.dates[sample_position]}",
        fontsize=14,
        fontweight="bold",
    )
    figure.tight_layout()
    return figure


def render_demo_result(result: DemoResult, tail_rows: int) -> None:
    metrics = result.metrics
    market_frame = result.market_frame
    model_ticker, model_phase, period = infer_experiment_metadata(result.model_name)
    caption_parts = [part for part in [model_ticker, model_phase, period] if part]
    if caption_parts:
        st.caption(" | ".join(caption_parts))

    metric_cols = st.columns(4)
    metric_cols[0].metric("Danh mục cuối", f"${metrics['final_portfolio']:,.2f}", f"{metrics['profit']:,.2f}")
    metric_cols[1].metric("Tổng reward", f"{metrics['total_reward']:,.2f}")
    metric_cols[2].metric("Max drawdown", f"{metrics['max_drawdown_pct']:.2f}%")
    metric_cols[3].metric("Độ phủ bộ test", f"{metrics.get('coverage_pct', 100.0):.1f}%")

    policy_cols = st.columns(4)
    policy_cols[0].metric("Policy Mua / Bán / Giữ", f"{metrics['buy_count']} / {metrics['sell_count']} / {metrics['hold_count']}")
    policy_cols[1].metric("Khớp Mua / Bán", f"{metrics['executed_buy_count']} / {metrics['executed_sell_count']}")
    policy_cols[2].metric("Lệnh không khớp", f"{metrics['rejected_count']}")
    policy_cols[3].metric("Số transition", f"{metrics['steps']:,}")

    if not bool(metrics.get("time_monotonic", True)):
        st.warning("Thứ tự ngày trong CSV không tăng dần. Ứng dụng vẫn giữ nguyên thứ tự để khớp notebook; hãy kiểm tra chính file CSV.")

    pnl_gap = float(metrics.get("reward_portfolio_gap", np.nan))
    if np.isfinite(pnl_gap) and abs(pnl_gap) <= 1e-4:
        st.success(f"Kiểm tra PnL đạt: Final portfolio − Initial = Σreward, sai lệch {pnl_gap:.6f}.")
    else:
        st.warning(f"PnL và reward chưa khớp hoàn toàn, sai lệch = {pnl_gap:.6f}. Cần kiểm tra công thức reward/transaction fee trong MDP.")

    expected_final = float(metrics.get("notebook_expected_final", np.nan))
    parity_config = (
        int(metrics.get("balance_init", -1)) == 1000
        and int(metrics.get("min_balance", 999999)) == -100
        and int(metrics.get("dataset_rows", 0)) == 250
    )
    if np.isfinite(expected_final) and parity_config:
        final_gap = float(metrics.get("notebook_final_gap", np.nan))
        reward_gap = float(metrics.get("notebook_reward_gap", np.nan))
        if abs(final_gap) <= 0.02 and abs(reward_gap) <= 0.02:
            st.success(
                f"Khớp notebook {model_ticker} {model_phase}: final ${expected_final:,.2f}; "
                f"chênh final {final_gap:+.4f}, chênh reward {reward_gap:+.4f}."
            )
        else:
            st.error(
                f"Chưa khớp notebook {model_ticker} {model_phase}. Notebook: final ${expected_final:,.2f}, "
                f"reward {metrics['notebook_expected_reward']:,.2f}; Streamlit lệch final {final_gap:+.4f}, "
                f"reward {reward_gap:+.4f}. Kiểm tra đúng checkpoint, CSV, balance=1000, min_balance=-100 và thứ tự dòng."
            )
    elif np.isfinite(expected_final):
        st.info("Đối chiếu notebook chỉ bật khi balance=1000, min_balance=-100 và dataset có 250 dòng như notebook mẫu.")

    st.caption(
        f"Data fingerprint: {metrics.get('data_fingerprint')} | State fingerprint: {metrics.get('state_fingerprint')} | "
        "XAI giải thích policy action; biểu đồ giá chỉ đánh dấu giao dịch thực sự làm thay đổi vị thế."
    )

    chart = build_chart(result)
    if chart is not None:
        st.plotly_chart(chart, use_container_width=True, key="live_timeseries_chart_executed")
        action_chart = build_action_count_chart(result)
        if action_chart is not None:
            st.plotly_chart(action_chart, use_container_width=True, key="live_action_count_chart_executed")
    else:
        st.info("Cài plotly để xem biểu đồ tương tác: pip install plotly")

    st.subheader("Nhật ký transition và khớp lệnh")
    st.caption(
        "requested_action là quyết định greedy của model. executed_qty = shares(t+1) − shares(t) mới là lượng giao dịch thực tế. "
        "Do đó, lệnh Bán khi chưa có cổ phiếu sẽ xuất hiện là lệnh không khớp, không phải một giao dịch Bán trên đồ thị."
    )
    transition_view = result.transition_frame.tail(tail_rows).copy()
    numeric_round = ["execution_price", "implied_cash_price", "balance_before", "balance_after", "portfolio_before", "portfolio_after", "portfolio_delta", "reward"]
    for col in numeric_round:
        if col in transition_view:
            transition_view[col] = transition_view[col].round(6)
    st.dataframe(transition_view, use_container_width=True, hide_index=True)

    d1, d2 = st.columns(2)
    d1.download_button(
        label="Tải CSV trạng thái",
        data=result.frame.to_csv(index=False).encode("utf-8"),
        file_name=f"{Path(result.model_name).stem}_{Path(result.dataset_name).stem}_states.csv",
        mime="text/csv",
        key="download_states_csv",
    )
    d2.download_button(
        label="Tải CSV transition/khớp lệnh",
        data=result.transition_frame.to_csv(index=False).encode("utf-8"),
        file_name=f"{Path(result.model_name).stem}_{Path(result.dataset_name).stem}_transitions.csv",
        mime="text/csv",
        key="download_transitions_csv",
    )



def main() -> None:
    st.set_page_config(page_title="SARSA Financial RL + XAI", page_icon="📈", layout="wide")
    inject_css()

    checkpoints = list_checkpoints()
    sarsa_checkpoints = list_sarsa_checkpoints()
    datasets = list_datasets()
    preferred_checkpoint = next(
        (name for name in sarsa_checkpoints if "phase_3" in name.lower()),
        sarsa_checkpoints[0] if sarsa_checkpoints else "",
    )
    preferred_dataset = next(
        (name for name in datasets if name.lower() == "test_acb_phase_3.csv"),
        datasets[0] if datasets else "",
    )

    with st.sidebar:
        st.markdown("### Điều khiển demo")
        st.caption("Checkpoint và bộ test được đọc trực tiếp từ workspace.")

        if sarsa_checkpoints:
            model_name = st.selectbox(
                "SARSA checkpoint",
                sarsa_checkpoints,
                index=sarsa_checkpoints.index(preferred_checkpoint) if preferred_checkpoint in sarsa_checkpoints else 0,
            )
        else:
            st.warning("Không tìm thấy checkpoint SARSA trong /models")
            model_name = ""

        if datasets:
            dataset_name = st.selectbox(
                "Test dataset",
                datasets,
                index=datasets.index(preferred_dataset) if preferred_dataset in datasets else 0,
            )
        else:
            st.warning("Không tìm thấy CSV trong /data/data_storer/data_research/test")
            dataset_name = ""

        uploaded_file = st.file_uploader("Hoặc tải CSV riêng", type=["csv"])
        balance_init = st.slider("Số dư ban đầu", min_value=500, max_value=5000, value=1000, step=100)
        min_balance = st.slider("Ngưỡng số dư tối thiểu", min_value=-2000, max_value=0, value=-100, step=50)
        tail_rows = st.slider("Số dòng hiển thị", min_value=40, max_value=250, value=120, step=10)

        st.markdown("---")
        st.write(f"Tổng checkpoint: {len(checkpoints)}")
        st.write(f"Checkpoint SARSA: {len(sarsa_checkpoints)}")
        st.write(f"Bộ test CSV: {len(datasets)}")

    st.markdown(
        """
        <div class="hero-card">
            <div class="hero-title">SARSA Financial RL + XAI Dashboard</div>
            <div class="hero-subtitle">
                Chạy mô phỏng giao dịch, sau đó giải thích quyết định bằng hai hướng độc lập:
                MSX + RDX cho cấu trúc lý do theo reward/Q-value và SHAP cho mức đóng góp của 7 biến trạng thái.
            </div>
            <div class="badge-row">
                <span class="badge">Live SARSA simulation</span>
                <span class="badge">RDX decomposition</span>
                <span class="badge">MSX+ minimal reasons</span>
                <span class="badge">Kernel SHAP</span>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    pages = ["Tổng quan", "Live Demo", "MSX + RDX", "SHAP", "Cách chạy"]
    page = st.radio(
        "Điều hướng dashboard",
        pages,
        horizontal=True,
        key="main_page",
        label_visibility="collapsed",
    )

    if page == "Tổng quan":
        col1, col2, col3 = st.columns([1.15, 1, 1.1])
        with col1:
            st.subheader("Model Library")
            model_df = (
                pd.DataFrame(
                    [
                        {
                            "file": path.name,
                            "size_kb": round(path.stat().st_size / 1024, 1),
                            "family": path.stem.split("_")[0].upper(),
                            "period": infer_experiment_metadata(path.name)[2] or "—",
                        }
                        for path in sorted(MODELS_DIR.glob("*.pth"))
                    ]
                )
                if MODELS_DIR.exists()
                else pd.DataFrame(columns=["file", "size_kb", "family", "period"])
            )
            st.dataframe(model_df, use_container_width=True, hide_index=True)

        with col2:
            st.subheader("Data Library")
            data_df = (
                pd.DataFrame(
                    [
                        {
                            "file": path.name,
                            "size_kb": round(path.stat().st_size / 1024, 1),
                            "period": infer_experiment_metadata(path.name)[2] or "—",
                        }
                        for path in sorted(TEST_DATA_DIR.glob("*.csv"))
                    ]
                )
                if TEST_DATA_DIR.exists()
                else pd.DataFrame(columns=["file", "size_kb", "period"])
            )
            st.dataframe(data_df, use_container_width=True, hide_index=True)

        with col3:
            st.subheader("Luồng phân tích")
            st.markdown(
                """
                <div class="explain-card">
                <b>1. Live Demo</b><br>
                Tạo chuỗi state, action, reward và portfolio.<br><br>
                <b>2. RDX</b><br>
                Chuẩn hóa Q theo portfolio rồi phân rã thành Profit, Risk, Trend, Stability.<br><br>
                <b>3. MSX+</b><br>
                So sánh hành động được chọn với phương án tốt thứ hai và tìm tập lý do dương nhỏ nhất đủ thắng bất lợi.<br><br>
                <b>4. SHAP</b><br>
                KernelExplainer đo đóng góp của Close, Balance, Position, MACD, RSI, CCI và ADX lên greedy Q-value.
                </div>
                """,
                unsafe_allow_html=True,
            )
            st.info("Hãy chạy Live Demo trước. Kết quả được giữ trong session để dùng tiếp ở hai tab XAI.")

    elif page == "Live Demo":
        run_clicked = st.button("Chạy mô phỏng", type="primary", key="run_simulation")
        if run_clicked:
            try:
                if uploaded_file is not None:
                    demo_df = load_market_data(uploaded_csv=uploaded_file.getvalue())
                    data_name = uploaded_file.name
                else:
                    if not dataset_name:
                        raise ValueError("Không có dataset tích hợp.")
                    demo_df = load_market_data(source_name=dataset_name)
                    data_name = dataset_name

                if not model_name:
                    raise ValueError("Không có checkpoint để chạy.")

                for warning in validate_model_dataset_pair(model_name, data_name):
                    st.warning(warning)

                with st.spinner("Đang chạy SARSA simulation..."):
                    result = simulate_on_dataframe(model_name, data_name, demo_df, balance_init, min_balance)
                st.session_state["demo_result"] = result
                st.session_state.pop("rdx_result", None)
                st.session_state.pop("shap_result", None)
                st.session_state.pop("rdx_critical_point", None)
                st.session_state.pop("shap_local_position", None)
                st.success("Mô phỏng hoàn tất. Hai tab XAI đã sẵn sàng.")
            except Exception as exc:  # pragma: no cover - UI feedback
                st.error(f"Mô phỏng thất bại: {exc}")
                st.caption("Kiểm tra kiến trúc checkpoint và các cột bắt buộc trong CSV.")

        stored_result: DemoResult | None = st.session_state.get("demo_result")
        if stored_result is not None:
            render_demo_result(stored_result, tail_rows)
        else:
            st.info("Chọn checkpoint và dataset, sau đó nhấn Chạy mô phỏng.")

    elif page == "MSX + RDX":
        result = st.session_state.get("demo_result")
        if result is None:
            st.info("Cần chạy Live Demo trước khi phân tích MSX + RDX.")
        else:
            st.markdown(
                """
                <div class="explain-card">
                <b>RDX</b> trả lời: mỗi thành phần Profit, Risk, Trend và Stability đóng góp bao nhiêu vào Q-value?<br>
                <b>MSX+</b> trả lời: tập lý do dương nhỏ nhất nào đủ giải thích vì sao hành động đã chọn thắng phương án tốt thứ hai?
                </div>
                """,
                unsafe_allow_html=True,
            )
            with st.expander("Tham số RDX/MSX", expanded=False):
                c1, c2, c3, c4 = st.columns(4)
                w_profit = c1.number_input("w Profit (tham chiếu)", 0.0, 5.0, 1.0, 0.1)
                w_risk = c2.number_input("w Risk", 0.0, 5.0, 0.5, 0.1)
                w_trend = c3.number_input("w Trend", 0.0, 5.0, 1.0, 0.1)
                w_stability = c4.number_input("w Stability", 0.0, 5.0, 0.1, 0.1)
                c5, c6, c7, c8 = st.columns(4)
                alpha = c5.number_input("Trend alpha", 0.0001, 0.1, 0.005, 0.001, format="%.4f")
                top_k = c6.slider("Critical points", 5, 30, 15)
                action_threshold = c7.slider("Ngưỡng đổi action", 1, 10, 3)
                trend_window = c8.slider("Cửa sổ đảo chiều", 2, 30, 10)
                st.caption("Giống notebook: Profit là phần dư để tổng 4 thành phần bằng Q/Portfolio; w Profit được giữ để tương thích cấu hình.")

            if st.button("Chạy phân tích MSX + RDX", type="primary", key="run_rdx"):
                try:
                    model = load_qsa_model(result.model_name)
                    with st.spinner("Đang phân rã Q-value và tìm critical points..."):
                        rdx_result = compute_rdx_msx(
                            model,
                            result,
                            weights=(float(w_profit), float(w_risk), float(w_trend), float(w_stability)),
                            alpha=float(alpha),
                            top_k=int(top_k),
                            action_change_threshold=int(action_threshold),
                            trend_window=int(trend_window),
                        )
                    st.session_state["rdx_result"] = rdx_result
                    st.session_state.pop("rdx_critical_point", None)
                    st.success("Phân tích MSX + RDX hoàn tất.")
                except Exception as exc:
                    st.error(f"Không thể chạy MSX + RDX: {exc}")

            rdx_result: RDXResult | None = st.session_state.get("rdx_result")
            if rdx_result is not None:
                count_cols = st.columns(4)
                count_cols[0].metric("Critical points", len(rdx_result.critical_indices))
                count_cols[1].metric("Đáy/Drawdown", len(rdx_result.critical_points["lowest_price"]))
                count_cols[2].metric("Đảo chiều", len(rdx_result.critical_points["trend_reversal"]))
                count_cols[3].metric("Đổi action", len(rdx_result.critical_points["action_shift"]))

                timeline = build_critical_timeline(result, rdx_result)
                if timeline is not None:
                    st.plotly_chart(timeline, use_container_width=True, key="rdx_timeline_chart")

                heatmap = build_rdx_heatmap(result, rdx_result)
                if heatmap is not None:
                    st.plotly_chart(heatmap, use_container_width=True, key="rdx_heatmap_chart")

                st.subheader("Bảng MSX+ tại các điểm quan trọng")
                display_columns = [
                    "time",
                    "close",
                    "reason",
                    "selected_label",
                    "alternative_label",
                    "q_margin",
                    "msx_plus",
                    "advantage",
                    "disadvantage",
                    "is_sufficient",
                ]
                table = rdx_result.critical_table[display_columns].copy()
                for column in ["close", "q_margin", "advantage", "disadvantage"]:
                    table[column] = table[column].round(6)
                st.dataframe(table, use_container_width=True, hide_index=True)

                st.download_button(
                    "Tải kết quả MSX + RDX",
                    data=rdx_result.critical_table.to_csv(index=False).encode("utf-8"),
                    file_name=f"{Path(result.model_name).stem}_{Path(result.dataset_name).stem}_msx_rdx.csv",
                    mime="text/csv",
                )

                if rdx_result.critical_indices:
                    options = rdx_result.critical_indices
                    selected_time = st.selectbox(
                        "Chọn critical point để xem giải thích cục bộ",
                        options,
                        key="rdx_critical_point",
                        format_func=lambda idx: (
                            f"#{idx} | {result.frame.iloc[idx]['time']} | "
                            f"{action_label(result.actions[idx])} | {_critical_reason(idx, rdx_result.critical_points)}"
                        ),
                    )
                    selected_action = int(result.actions[selected_time])
                    selected_idx = selected_action + 5
                    alternative_idx = best_alternative_index(rdx_result.q_values[selected_time], selected_idx)
                    alternative_action = alternative_idx - 5
                    msx_result = analyze_msx(
                        rdx_result.q_components[selected_time, selected_idx],
                        rdx_result.q_components[selected_time, alternative_idx],
                    )

                    local_cols = st.columns(5)
                    local_cols[0].metric("Policy action", action_label(selected_action))
                    local_cols[1].metric("Khớp thực tế", f"{result.executed_actions[selected_time]:+.0f}" if selected_time < len(result.executed_actions) else "—")
                    local_cols[2].metric("Phương án so sánh", action_label(alternative_action))
                    local_cols[3].metric("Tổng lợi thế MSX+", f"{msx_result['advantage']:.5f}")
                    local_cols[4].metric("Tổng bất lợi", f"{msx_result['disadvantage']:.5f}")

                    msx_items = msx_result["msx_plus"]
                    if msx_items:
                        reason_text = ", ".join(
                            f"{item['component']} ({item['contribution_percent']:.1f}%)" for item in msx_items
                        )
                        st.success(f"MSX+ tối thiểu: {reason_text}")
                    elif msx_result["is_dominated"]:
                        st.success("Hành động được chọn trội hơn ở tất cả thành phần; không có bất lợi cần bù.")
                    else:
                        st.warning("Không tìm được tập lý do dương đủ thắng tổng bất lợi tại điểm này.")

                    local_chart = build_rdx_local_chart(result, rdx_result, selected_time)
                    if local_chart is not None:
                        st.plotly_chart(local_chart, use_container_width=True, key="rdx_local_chart")
                    delta_chart = build_msx_delta_chart(msx_result)
                    if delta_chart is not None:
                        st.plotly_chart(delta_chart, use_container_width=True, key="msx_delta_chart")

    elif page == "SHAP":
        result = st.session_state.get("demo_result")
        if result is None:
            st.info("Cần chạy Live Demo trước khi phân tích SHAP.")
        elif shap is None:
            st.error("Chưa cài SHAP. Chạy: pip install shap matplotlib")
        else:
            st.markdown(
                """
                <div class="explain-card">
                SHAP ở đây bám theo notebook: hàm dự đoán trả về <b>Q-value của hành động greedy</b> cho từng trạng thái.
                KernelExplainer dùng một background nhỏ và một tập sample giới hạn để dashboard phản hồi nhanh hơn.
                </div>
                """,
                unsafe_allow_html=True,
            )
            c1, c2, c3, c4 = st.columns(4)
            max_decisions = max(2, len(result.actions))
            background_size = c1.slider("Background states", 2, min(200, max_decisions), min(40, max_decisions))
            sample_size = c2.slider("States cần giải thích", 1, min(100, max_decisions), min(30, max_decisions))
            nsamples = c3.slider("Kernel evaluations", 30, 500, 120, step=10)
            sampling_mode = c4.selectbox(
                "Cách chọn sample",
                ["Phân bố đều theo thời gian", "Các bước gần nhất", "Ngẫu nhiên"],
            )

            estimated_calls = background_size + sample_size * nsamples
            st.caption(f"Cấu hình hiện tại có quy mô xấp xỉ {estimated_calls:,} lượt đánh giá mẫu; giảm sample/nsamples nếu máy yếu.")

            if st.button("Chạy SHAP", type="primary", key="run_shap"):
                try:
                    model = load_qsa_model(result.model_name)
                    with st.spinner("Đang chạy Kernel SHAP..."):
                        shap_result = compute_shap_explanations(
                            model,
                            result,
                            background_size=int(background_size),
                            sample_size=int(sample_size),
                            nsamples=int(nsamples),
                            sampling_mode=sampling_mode,
                        )
                    st.session_state["shap_result"] = shap_result
                    st.success("SHAP hoàn tất.")
                except Exception as exc:
                    st.error(f"Không thể chạy SHAP: {exc}")

            shap_result: SHAPResult | None = st.session_state.get("shap_result")
            if shap_result is not None:
                importance = np.mean(np.abs(shap_result.values), axis=0)
                top_feature_index = int(np.argmax(importance))
                metric_cols = st.columns(4)
                metric_cols[0].metric("Số mẫu SHAP", len(shap_result.data))
                metric_cols[1].metric("Background", shap_result.background_size)
                metric_cols[2].metric("Biến ảnh hưởng nhất", FEATURE_NAMES[top_feature_index])
                metric_cols[3].metric("Mean |SHAP|", f"{importance[top_feature_index]:.4f}")

                importance_chart = build_shap_importance_chart(shap_result)
                if importance_chart is not None:
                    st.plotly_chart(importance_chart, use_container_width=True, key="shap_importance_chart")

                summary_figure = create_shap_summary_figure(shap_result)
                if summary_figure is not None:
                    st.subheader("SHAP beeswarm — Global interpretation")
                    st.pyplot(summary_figure, use_container_width=True)
                    plt.close(summary_figure)

                temporal_chart = build_shap_temporal_chart(shap_result, top_n=4)
                if temporal_chart is not None:
                    st.plotly_chart(temporal_chart, use_container_width=True, key="shap_temporal_chart")

                local_position = st.selectbox(
                    "Chọn mẫu để xem Local SHAP",
                    list(range(len(shap_result.dates))),
                    key="shap_local_position",
                    format_func=lambda pos: (
                        f"{shap_result.dates[pos]} | simulation index {int(shap_result.sample_indices[pos])}"
                    ),
                )
                local_figure = create_shap_local_figure(shap_result, int(local_position))
                if local_figure is not None:
                    st.pyplot(local_figure, use_container_width=True)
                    plt.close(local_figure)

                shap_table = pd.DataFrame(shap_result.values, columns=[f"SHAP_{name}" for name in FEATURE_NAMES])
                shap_table.insert(0, "simulation_index", shap_result.sample_indices)
                shap_table.insert(1, "time", shap_result.dates)
                feature_table = pd.DataFrame(shap_result.data, columns=FEATURE_NAMES)
                export_table = pd.concat([shap_table, feature_table], axis=1)
                st.subheader("Dữ liệu SHAP")
                st.dataframe(export_table.round(6), use_container_width=True, hide_index=True)
                st.download_button(
                    "Tải SHAP CSV",
                    data=export_table.to_csv(index=False).encode("utf-8"),
                    file_name=f"{Path(result.model_name).stem}_{Path(result.dataset_name).stem}_shap.csv",
                    mime="text/csv",
                )

    elif page == "Cách chạy":
        st.subheader("Cài thư viện")
        st.code("pip install streamlit plotly shap matplotlib scipy", language="bash")
        st.subheader("Chạy ứng dụng")
        st.code("streamlit run application/streamlit_app.py --server.port 8501", language="bash")
        st.write(
            "Đặt file này tại application/streamlit_app.py trong dự án để ROOT_DIR tự nhận diện đúng thư mục models, agents và data. Điều hướng ngang dùng session state nên không bị reset khi đổi critical point."
        )
        st.subheader("Thứ tự demo đề xuất")
        st.write("1. Chọn checkpoint và test dataset cùng ticker/cùng phase.")
        st.write("2. Chạy Live Demo và kiểm tra phân bố Mua/Bán/Giữ.")
        st.write("3. Chạy MSX + RDX để xem critical points, heatmap và giải thích cục bộ.")
        st.write("4. Chạy SHAP với 20–40 samples trước; tăng dần khi cần hình tổng quan ổn định hơn.")
        st.warning(
            "Không dùng chuỗi tiếng Việt làm đầu vào model. Giao diện chỉ dịch nhãn hiển thị; action nội bộ luôn giữ số nguyên từ -5 đến +5."
        )


if __name__ == "__main__":
    main()
