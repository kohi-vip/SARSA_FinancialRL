from __future__ import annotations

import sys
from dataclasses import dataclass
from io import BytesIO
from pathlib import Path

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


ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from agents.d_sarsa.d_sarsa import Qsa  # noqa: E402
from environments.stock_trading_env.mdp import StockTradingMDP  # noqa: E402


MODELS_DIR = ROOT_DIR / "models"
TEST_DATA_DIR = ROOT_DIR / "data" / "data_storer" / "data_research" / "test"
REQUIRED_COLUMNS = ["time", "open", "high", "low", "close", "volume", "MACD", "RSI", "CCI", "ADX"]
STATE_COLUMNS = ["close", "balance", "shares", "MACD", "RSI", "CCI", "ADX"]


@dataclass
class DemoResult:
    frame: pd.DataFrame
    metrics: dict[str, float | int | str]
    model_name: str
    dataset_name: str


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
            .stTabs [data-baseweb="tab-list"] {
                gap: 0.35rem;
            }
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
    return [name for name in list_checkpoints() if name.startswith("sarsa_")]


@st.cache_data(show_spinner=False)
def list_datasets() -> list[str]:
    if not TEST_DATA_DIR.exists():
        return []
    return sorted(p.name for p in TEST_DATA_DIR.glob("*.csv"))


def _read_csv_bytes(uploaded_csv: bytes) -> pd.DataFrame:
    return pd.read_csv(BytesIO(uploaded_csv))


@st.cache_data(show_spinner=False)
def load_market_data(source_name: str | None = None, uploaded_csv: bytes | None = None) -> pd.DataFrame:
    if uploaded_csv is not None:
        df = _read_csv_bytes(uploaded_csv)
    elif source_name:
        df = pd.read_csv(TEST_DATA_DIR / source_name)
    else:
        raise ValueError("No data source was provided")

    if "date" in df.columns and "time" not in df.columns:
        df = df.rename(columns={"date": "time"})

    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"Dataset is missing required columns: {', '.join(missing)}")

    df = df.copy()
    df["time"] = pd.to_datetime(df["time"], errors="coerce")
    df = df.dropna(subset=["time"]).sort_values("time").reset_index(drop=True)

    for col in ["open", "high", "low", "close", "volume", "MACD", "RSI", "CCI", "ADX"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df = df.dropna(subset=["close", "MACD", "RSI", "CCI", "ADX"]).reset_index(drop=True)
    df["time"] = df["time"].dt.strftime("%Y-%m-%d")
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


def action_label(action: int) -> str:
    if action == 0:
        return "Hold"
    if action > 0:
        return f"Buy {action}"
    return f"Sell {abs(action)}"


def simulate_on_dataframe(model_name: str, data_name: str, df: pd.DataFrame, balance_init: int, min_balance: int) -> DemoResult:
    if len(df) < 2:
        raise ValueError("The selected dataset needs at least 2 rows to run a simulation")

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

    def policy_fn(state, greedy: bool = True, eps: float = 0.0):
        with torch.no_grad():
            logits = model(torch.tensor(state, dtype=torch.float32)).squeeze()
            return int(torch.argmax(logits).item()) - 5

    states, rewards, actions = mdp.simulate(df.iloc[1:].reset_index(drop=True), state_init, policy_fn, True, eps=0.0)

    state_df = pd.DataFrame(states, columns=STATE_COLUMNS)
    state_df.insert(0, "time", df["time"].iloc[: len(state_df)].values)
    state_df["portfolio"] = state_df["balance"] + state_df["close"] * state_df["shares"]
    state_df["reward"] = [0.0] + rewards
    state_df["action"] = [np.nan] + actions
    state_df["action_label"] = ["INIT"] + [action_label(a) for a in actions]

    portfolio = state_df["portfolio"].to_numpy(dtype=float)
    peak = np.maximum.accumulate(portfolio)
    drawdown = np.where(peak > 0, (peak - portfolio) / peak, 0.0)

    metrics = {
        "steps": int(len(actions)),
        "final_portfolio": float(portfolio[-1]),
        "profit": float(portfolio[-1] - balance_init),
        "total_reward": float(np.sum(rewards)),
        "max_portfolio": float(np.max(portfolio)),
        "min_portfolio": float(np.min(portfolio)),
        "max_drawdown_pct": float(np.max(drawdown) * 100.0),
        "buy_count": int(np.sum(np.array(actions) > 0)),
        "sell_count": int(np.sum(np.array(actions) < 0)),
        "hold_count": int(np.sum(np.array(actions) == 0)),
    }

    return DemoResult(
        frame=state_df,
        metrics=metrics,
        model_name=model_name,
        dataset_name=data_name,
    )


def build_chart(result: DemoResult):
    frame = result.frame
    if go is None or make_subplots is None:
        return None

    fig = make_subplots(
        rows=3,
        cols=1,
        shared_xaxes=True,
        vertical_spacing=0.06,
        row_heights=[0.45, 0.27, 0.28],
        specs=[[{}], [{}], [{}]],
    )

    fig.add_trace(
        go.Scatter(x=frame["time"], y=frame["close"], name="Price", line=dict(color="#7dd3fc", width=2.5)),
        row=1,
        col=1,
    )

    buy_mask = frame["action"].fillna(0) > 0
    sell_mask = frame["action"].fillna(0) < 0
    if buy_mask.any():
        fig.add_trace(
            go.Scatter(
                x=frame.loc[buy_mask, "time"],
                y=frame.loc[buy_mask, "close"],
                mode="markers",
                name="Buy",
                marker=dict(symbol="triangle-up", size=12, color="#22c55e", line=dict(width=0)),
            ),
            row=1,
            col=1,
        )
    if sell_mask.any():
        fig.add_trace(
            go.Scatter(
                x=frame.loc[sell_mask, "time"],
                y=frame.loc[sell_mask, "close"],
                mode="markers",
                name="Sell",
                marker=dict(symbol="triangle-down", size=12, color="#ef4444", line=dict(width=0)),
            ),
            row=1,
            col=1,
        )

    fig.add_trace(
        go.Scatter(x=frame["time"], y=frame["portfolio"], name="Portfolio", line=dict(color="#f59e0b", width=2.5)),
        row=2,
        col=1,
    )
    fig.add_hline(y=float(frame["portfolio"].iloc[0]), line_dash="dash", line_color="#94a3b8", row=2, col=1)

    action_counts = (
        frame["action_label"].value_counts()
        .reindex(["Hold"] + [f"Buy {i}" for i in range(1, 6)] + [f"Sell {i}" for i in range(1, 6)], fill_value=0)
        .reset_index()
    )
    action_counts.columns = ["action", "count"]
    fig.add_trace(
        go.Bar(x=action_counts["action"], y=action_counts["count"], name="Action Count", marker_color="#60a5fa"),
        row=3,
        col=1,
    )

    fig.update_layout(
        template="plotly_dark",
        height=860,
        margin=dict(l=10, r=10, t=30, b=10),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    )
    fig.update_yaxes(title_text="Price", row=1, col=1)
    fig.update_yaxes(title_text="Portfolio", row=2, col=1)
    fig.update_yaxes(title_text="Count", row=3, col=1)
    fig.update_xaxes(title_text="Date", row=3, col=1)
    return fig


def main() -> None:
    st.set_page_config(page_title="SARSA Financial RL Demo", page_icon="📈", layout="wide")
    inject_css()

    checkpoints = list_checkpoints()
    sarsa_checkpoints = list_sarsa_checkpoints()
    datasets = list_datasets()
    preferred_checkpoint = next((name for name in sarsa_checkpoints if "phase_3" in name), sarsa_checkpoints[0] if sarsa_checkpoints else "")
    preferred_dataset = next((name for name in datasets if name == "test_ACB_phase_3.csv"), datasets[0] if datasets else "")

    with st.sidebar:
        st.markdown("### Demo Controls")
        st.caption("Checkpoint and test data come from the local workspace.")

        if sarsa_checkpoints:
            model_name = st.selectbox(
                "SARSA checkpoint",
                sarsa_checkpoints,
                index=sarsa_checkpoints.index(preferred_checkpoint) if preferred_checkpoint in sarsa_checkpoints else 0,
            )
        else:
            st.warning("No SARSA checkpoint files were found in /models")
            model_name = ""

        if datasets:
            dataset_name = st.selectbox(
                "Test dataset",
                datasets,
                index=datasets.index(preferred_dataset) if preferred_dataset in datasets else 0,
            )
        else:
            st.warning("No CSV files were found in /data/data_storer/data_research/test")
            dataset_name = ""

        uploaded_file = st.file_uploader("Or upload your own CSV", type=["csv"])

        balance_init = st.slider("Initial cash balance", min_value=500, max_value=5000, value=1000, step=100)
        min_balance = st.slider("Minimum balance threshold", min_value=-2000, max_value=0, value=-100, step=50)
        tail_rows = st.slider("Rows shown in table", min_value=40, max_value=250, value=120, step=10)

        st.markdown("---")
        st.write(f"Available checkpoints: {len(checkpoints)}")
        st.write(f"SARSA-compatible checkpoints: {len(sarsa_checkpoints)}")
        st.write(f"Available datasets: {len(datasets)}")

    st.markdown(
        """
        <div class="hero-card">
            <div class="hero-title">SARSA Financial RL Streamlit Demo</div>
            <div class="hero-subtitle">
                Run the trading model directly in your browser, inspect the simulated portfolio, and compare
                checkpoints without opening a notebook. The demo loads the local .pth checkpoint, executes the
                MDP simulation, and renders the result with a finance-style dashboard.
            </div>
            <div class="badge-row">
                <span class="badge">Localhost friendly</span>
                <span class="badge">PyTorch checkpoints</span>
                <span class="badge">Interactive portfolio chart</span>
                <span class="badge">Built for demo sessions</span>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    tabs = st.tabs(["Overview", "Live Demo", "How to Run"])

    with tabs[0]:
        col1, col2, col3 = st.columns([1.2, 1, 1])
        with col1:
            st.subheader("Model Library")
            model_df = (
                pd.DataFrame(
                    [
                        {
                            "file": p.name,
                            "size_kb": round(p.stat().st_size / 1024, 1),
                            "family": p.stem.split("_")[0].upper(),
                        }
                        for p in sorted(MODELS_DIR.glob("*.pth"))
                    ]
                )
                if MODELS_DIR.exists()
                else pd.DataFrame(columns=["file", "size_kb", "family"])
            )
            st.dataframe(model_df, use_container_width=True, hide_index=True)

        with col2:
            st.subheader("Data Library")
            data_df = (
                pd.DataFrame(
                    [
                        {
                            "file": p.name,
                            "size_kb": round(p.stat().st_size / 1024, 1),
                        }
                        for p in sorted(TEST_DATA_DIR.glob("*.csv"))
                    ]
                )
                if TEST_DATA_DIR.exists()
                else pd.DataFrame(columns=["file", "size_kb"])
            )
            st.dataframe(data_df, use_container_width=True, hide_index=True)

        with col3:
            st.subheader("Notes")
            st.write("- The live demo is wired to SARSA/Qsa checkpoints only.")
            st.write("- DQN and policy-gradient checkpoints are listed in the catalog for later extension.")
            st.write("- The SARSA network expects 7 state inputs and 11 actions.")
            st.write("- The environment uses k=5, so actions range from -5 to +5.")
            st.write("- You can swap datasets or upload a custom CSV to demo new periods.")

    with tabs[1]:
        run_clicked = st.button("Run simulation", type="primary")
        if run_clicked:
            try:
                if uploaded_file is not None:
                    demo_df = load_market_data(uploaded_csv=uploaded_file.getvalue())
                    data_name = uploaded_file.name
                else:
                    if not dataset_name:
                        raise ValueError("No built-in dataset is available")
                    demo_df = load_market_data(source_name=dataset_name)
                    data_name = dataset_name

                if not model_name:
                    raise ValueError("No checkpoint is available")

                result = simulate_on_dataframe(model_name, data_name, demo_df, balance_init, min_balance)
                metrics = result.metrics
                st.success("Simulation completed successfully")

                metric_cols = st.columns(4)
                metric_cols[0].metric("Final Portfolio", f"${metrics['final_portfolio']:,.2f}", f"{metrics['profit']:,.2f}")
                metric_cols[1].metric("Total Reward", f"{metrics['total_reward']:,.2f}")
                metric_cols[2].metric("Max Drawdown", f"{metrics['max_drawdown_pct']:.2f}%")
                metric_cols[3].metric("Actions", f"{metrics['steps']:,}")

                metric_cols_2 = st.columns(3)
                metric_cols_2[0].metric("Buy / Sell / Hold", f"{metrics['buy_count']} / {metrics['sell_count']} / {metrics['hold_count']}")
                metric_cols_2[1].metric("Best Portfolio", f"${metrics['max_portfolio']:,.2f}")
                metric_cols_2[2].metric("Worst Portfolio", f"${metrics['min_portfolio']:,.2f}")

                chart = build_chart(result)
                if chart is not None:
                    st.plotly_chart(chart, use_container_width=True)
                else:
                    st.info("Plotly is not installed. Install it for the richer chart experience.")

                st.subheader("Simulation Table")
                view_df = result.frame.tail(tail_rows).copy()
                view_df["portfolio"] = view_df["portfolio"].round(2)
                view_df["reward"] = view_df["reward"].round(4)
                st.dataframe(view_df, use_container_width=True, hide_index=True)

                st.download_button(
                    label="Download simulation CSV",
                    data=result.frame.to_csv(index=False).encode("utf-8"),
                    file_name=f"{Path(model_name).stem}_{Path(data_name).stem}_simulation.csv",
                    mime="text/csv",
                )
            except Exception as exc:  # pragma: no cover - UI feedback
                st.error(f"Simulation failed: {exc}")
                st.caption("Check whether the selected checkpoint matches the Qsa architecture and the CSV has the required columns.")
        else:
            st.info("Choose a checkpoint and dataset, then press Run simulation.")

    with tabs[2]:
        st.subheader("How to run locally")
        st.code("streamlit run application/streamlit_app.py --server.port 8501", language="bash")
        st.write("Recommended workflow:")
        st.write("1. Activate your Python environment.")
        st.write("2. Install Streamlit if it is missing: pip install streamlit plotly.")
        st.write("3. Run the command above from the project root.")
        st.write("4. Open the localhost URL shown by Streamlit in your browser.")

        st.subheader("UI tuning ideas")
        st.write("- Keep the dashboard on layout='wide' for side-by-side metrics and charts.")
        st.write("- Use Plotly for dark finance visuals and hover inspection.")
        st.write("- Keep the file uploader in the sidebar so you can demo new market periods quickly.")
        st.write("- Keep the action range fixed to the checkpoint's k=5 so the policy mapping stays valid.")


if __name__ == "__main__":
    main()