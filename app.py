"""
app.py

Final Business Dashboard — Autonomous Pricing Agent via Reinforcement Learning
Travel & Hospitality | Infotact Solutions & Co.

Run with:
    streamlit run app.py

Place this file in the project ROOT directory (same level as src/, notebooks/,
outputs/, README.md) so imports and asset paths resolve correctly.

This dashboard tries to load REAL trained artifacts if present:
  - Q-Learning: outputs/trained_qtable_best.npy (falls back to trained_qtable.npy)
  - DQN checkpoint: models/dqn_best.pt, outputs/dqn_checkpoints/dqn_best.pt,
    or the highest-numbered outputs/dqn_checkpoints/dqn_ep*.pt
If an artifact isn't found, that agent is skipped gracefully and the
pre-generated result images in outputs/ are shown instead.

For best visual results, also copy the accompanying `.streamlit/config.toml`
into a `.streamlit/` folder in the project root.
"""

import sys
import os
import glob
import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go
import plotly.express as px

# ----------------------------------------------------------------------------
# PATH SETUP
# ----------------------------------------------------------------------------
ROOT_PATH = os.path.dirname(os.path.abspath(__file__))
SRC_PATH = os.path.join(ROOT_PATH, "src")
OUTPUTS_PATH = os.path.join(ROOT_PATH, "outputs")
if SRC_PATH not in sys.path:
    sys.path.append(SRC_PATH)

# ----------------------------------------------------------------------------
# DESIGN TOKENS — "Runway" palette: cool white base, vivid multi-color accents
# ----------------------------------------------------------------------------
BG          = "#F5F8FD"
SURFACE     = "#FFFFFF"
SURFACE_2   = "#EEF3FC"
BORDER      = "#E2E9F7"
SKY         = "#2563EB"
SKY_DIM     = "#DDE8FD"
CORAL       = "#FF5D6C"
CORAL_DIM   = "#FFE1E4"
EMERALD     = "#0EA96B"
EMERALD_DIM = "#DCF6EA"
VIOLET      = "#7C5CFC"
VIOLET_DIM  = "#EAE3FF"
AMBER       = "#F5A524"
AMBER_DIM   = "#FDECD0"
TEXT        = "#121933"
MUTED       = "#5E6B8C"
FONT_DISPLAY = "'Space Grotesk', sans-serif"
FONT_BODY    = "'Inter', sans-serif"

PLOTLY_TEMPLATE = "plotly_white"
CHART_COLORWAY = [SKY, CORAL, EMERALD, VIOLET, AMBER]

AGENT_COLOR = {
    "FixedPrice": SKY,
    "TimeBasedDiscount": CORAL,
    "DemandBased": EMERALD,
    "Q-Learning": VIOLET,
    "DQN": AMBER,
}

# ----------------------------------------------------------------------------
# PAGE CONFIG
# ----------------------------------------------------------------------------
st.set_page_config(
    page_title="Dynamic Pricing RL — Runway Dashboard",
    page_icon="🛫",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ----------------------------------------------------------------------------
# GLOBAL CSS
# ----------------------------------------------------------------------------
st.markdown(
    f"""
    <link rel="preconnect" href="https://fonts.googleapis.com">
    <link href="https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@500;600;700&family=Inter:wght@400;500;600;700&family=JetBrains+Mono:wght@500&display=swap" rel="stylesheet">
    <style>
        html, body, [class*="css"] {{ font-family: {FONT_BODY}; }}
        .stApp {{
            background:
                radial-gradient(circle at 6% -10%, rgba(37,99,235,0.08), transparent 38%),
                radial-gradient(circle at 96% 0%, rgba(255,93,108,0.08), transparent 36%),
                radial-gradient(circle at 45% 105%, rgba(14,169,107,0.06), transparent 42%),
                {BG};
        }}
        h1, h2, h3, h4 {{
            font-family: {FONT_DISPLAY} !important;
            color: {TEXT} !important;
            letter-spacing: -0.01em;
        }}
        p, li, span, label, div {{ color: {TEXT}; }}

        section[data-testid="stSidebar"] {{
            background: {SURFACE};
            border-right: 1px solid {BORDER};
        }}
        section[data-testid="stSidebar"] * {{ color: {TEXT}; }}

        .hero {{
            background: linear-gradient(120deg, #FFFFFF 0%, {SKY_DIM} 150%);
            border: 1px solid {BORDER};
            border-radius: 22px;
            padding: 44px 48px;
            margin-bottom: 26px;
            position: relative;
            overflow: hidden;
            box-shadow: 0 10px 34px rgba(37,99,235,0.10);
        }}
        .hero::before {{
            content: ""; position: absolute; top: -70px; right: -70px;
            width: 280px; height: 280px; border-radius: 50%;
            background: radial-gradient(circle, rgba(255,93,108,0.22), transparent 70%);
        }}
        .hero::after {{
            content: ""; position: absolute; bottom: -90px; left: 28%;
            width: 240px; height: 240px; border-radius: 50%;
            background: radial-gradient(circle, rgba(14,169,107,0.16), transparent 70%);
        }}
        .hero-eyebrow {{
            display: inline-flex; align-items: center; gap: 8px;
            font-size: 0.78rem; font-weight: 700; letter-spacing: 0.14em;
            text-transform: uppercase; color: {SKY}; background: {SKY_DIM};
            border: 1px solid rgba(37,99,235,0.28);
            padding: 6px 14px; border-radius: 999px; margin-bottom: 18px;
        }}
        .hero-title {{
            font-family: {FONT_DISPLAY}; font-weight: 700; font-size: 2.7rem;
            line-height: 1.08; margin: 0 0 12px 0; color: {TEXT};
            position: relative; z-index: 1;
        }}
        .hero-title .accent {{ color: {SKY}; }}
        .hero-sub {{ color: {MUTED}; font-size: 1.08rem; max-width: 660px; position: relative; z-index: 1; }}

        .bp-row {{ display: flex; gap: 16px; flex-wrap: wrap; margin: 18px 0 6px 0; }}
        .bp-card {{
            flex: 1; min-width: 150px; background: {SURFACE};
            border: 1px solid {BORDER}; border-top: 4px solid var(--accent, {SKY});
            border-radius: 14px; padding: 18px 20px;
            box-shadow: 0 3px 16px rgba(18,25,51,0.05);
            transition: transform 0.15s ease;
        }}
        .bp-card:hover {{ transform: translateY(-2px); }}
        .bp-label {{
            font-size: 0.72rem; letter-spacing: 0.10em; text-transform: uppercase;
            color: {MUTED}; margin-bottom: 6px; font-weight: 700;
        }}
        .bp-value {{ font-family: {FONT_DISPLAY}; font-size: 1.9rem; font-weight: 700; color: {TEXT}; }}

        .section-eyebrow {{
            font-size: 0.75rem; font-weight: 700; letter-spacing: 0.14em;
            text-transform: uppercase; color: {SKY}; margin-bottom: 4px;
        }}

        .panel {{
            background: {SURFACE}; border: 1px solid {BORDER}; border-radius: 14px;
            padding: 22px 24px; margin-bottom: 18px;
            box-shadow: 0 2px 12px rgba(18,25,51,0.04);
        }}
        .panel-sky     {{ border-left: 4px solid {SKY}; }}
        .panel-coral   {{ border-left: 4px solid {CORAL}; }}
        .panel-emerald {{ border-left: 4px solid {EMERALD}; }}
        .panel-violet  {{ border-left: 4px solid {VIOLET}; }}
        .panel-amber   {{ border-left: 4px solid {AMBER}; }}

        code, .stCodeBlock, pre {{ font-family: 'JetBrains Mono', monospace !important; }}
        code {{ background: {SURFACE_2} !important; color: {SKY} !important; padding: 2px 6px; border-radius: 5px; }}

        [data-testid="stMetric"] {{
            background: {SURFACE}; border: 1px solid {BORDER}; border-radius: 12px;
            padding: 14px 16px; box-shadow: 0 2px 10px rgba(18,25,51,0.04);
        }}
        [data-testid="stMetricLabel"] {{ color: {MUTED} !important; }}
        [data-testid="stMetricValue"] {{ color: {SKY} !important; font-family: {FONT_DISPLAY}; }}

        .stTabs [data-baseweb="tab-list"] {{ gap: 6px; border-bottom: 2px solid {BORDER}; }}
        .stTabs [data-baseweb="tab"] {{
            background: transparent; border-radius: 8px 8px 0 0; padding: 10px 18px;
            color: {MUTED}; font-weight: 700;
        }}
        .stTabs [aria-selected="true"] {{
            background: {SKY_DIM} !important; color: {SKY} !important;
            border-bottom: 3px solid {SKY};
        }}

        .stButton > button {{
            background: linear-gradient(120deg, {SKY}, #1D4ED8);
            color: #FFFFFF; font-weight: 700; border: none; border-radius: 10px;
            padding: 8px 18px; box-shadow: 0 4px 14px rgba(37,99,235,0.28);
        }}
        .stButton > button:hover {{ filter: brightness(1.06); }}

        .route {{ display: flex; align-items: flex-start; margin: 28px 0 10px 0; }}
        .route-stop {{ text-align: center; flex: 1; position: relative; }}
        .route-dot {{
            width: 18px; height: 18px; border-radius: 50%; margin: 0 auto 12px auto;
            border: 3px solid #fff; box-shadow: 0 0 0 3px var(--dot, {SKY});
            background: var(--dot, {SKY});
        }}
        .route-line {{
            position: absolute; top: 8px; left: -50%; width: 100%; height: 3px;
            background: repeating-linear-gradient(90deg, {BORDER} 0 8px, transparent 8px 16px);
            z-index: -1;
        }}
        .route-stop:first-child .route-line {{ display: none; }}
        .route-week {{ font-size: 0.74rem; letter-spacing: 0.08em; font-weight: 800; text-transform: uppercase; color: var(--dot, {SKY}); }}
        .route-focus {{ font-size: 0.85rem; color: {MUTED}; margin-top: 5px; padding: 0 8px; }}

        .footer-note {{
            color: {MUTED}; font-size: 0.82em; text-align: center; margin-top: 48px;
            padding-top: 18px; border-top: 1px solid {BORDER};
            font-family: 'JetBrains Mono', monospace;
        }}

        .badge {{ display: inline-block; padding: 4px 12px; border-radius: 999px; font-size: 0.72rem; font-weight: 700; letter-spacing: 0.04em; margin: 2px 4px 2px 0; }}
        .badge-live    {{ background: {EMERALD_DIM}; color: {EMERALD}; border: 1px solid rgba(14,169,107,0.3); }}
        .badge-pending {{ background: {SURFACE_2}; color: {MUTED}; border: 1px solid {BORDER}; }}
        .chip {{
            display: inline-block; margin: 4px 6px 0 0; padding: 6px 14px;
            border-radius: 999px; font-size: 0.8rem; font-weight: 700;
            background: {SURFACE_2}; border: 1px solid {BORDER}; color: {SKY};
        }}
        .gallery-caption {{ text-align: center; color: {MUTED}; font-size: 0.85rem; margin-top: 4px; }}
    </style>
    """,
    unsafe_allow_html=True,
)


def plotly_theme(fig, height=420):
    fig.update_layout(
        template=PLOTLY_TEMPLATE, paper_bgcolor=SURFACE, plot_bgcolor=SURFACE,
        font=dict(family="Inter, sans-serif", color=TEXT, size=13),
        colorway=CHART_COLORWAY, height=height, margin=dict(l=10, r=10, t=40, b=10),
        legend=dict(bgcolor="rgba(0,0,0,0)"),
        title_font=dict(family="Space Grotesk, sans-serif", size=16, color=TEXT),
    )
    fig.update_xaxes(gridcolor=BORDER, zerolinecolor=BORDER)
    fig.update_yaxes(gridcolor=BORDER, zerolinecolor=BORDER)
    return fig


def panel(html, variant="sky"):
    st.markdown(f'<div class="panel panel-{variant}">{html}</div>', unsafe_allow_html=True)


def show_image_if_exists(filename, caption):
    path = os.path.join(OUTPUTS_PATH, filename)
    if os.path.exists(path):
        st.image(path, use_container_width=True)
        st.markdown(f'<div class="gallery-caption">{caption}</div>', unsafe_allow_html=True)
        return True
    return False


# ----------------------------------------------------------------------------
# TRY TO IMPORT PROJECT MODULES (fail gracefully if not present yet)
# ----------------------------------------------------------------------------
IMPORT_ERROR = None
try:
    from pricing_env import PricingEnv
    from baseline_agents import FixedPriceAgent, TimeBasedDiscountAgent, DemandBasedAgent
except Exception as e:
    IMPORT_ERROR = str(e)

QLEARNING_AVAILABLE = False
try:
    from q_learning_agent import QLearningAgent
    QLEARNING_AVAILABLE = True
except Exception:
    pass

DQN_AVAILABLE = False
try:
    import torch
    from dqn_agent import DQNAgent
    DQN_AVAILABLE = True
except Exception:
    pass


@st.cache_resource(show_spinner=False)
def load_qlearning_agent():
    """Load the trained Q-table if present. Returns None if unavailable."""
    if not QLEARNING_AVAILABLE:
        return None
    for fname in ["trained_qtable_best.npy", "trained_qtable.npy"]:
        path = os.path.join(OUTPUTS_PATH, fname)
        if os.path.exists(path):
            agent = QLearningAgent()
            agent.load(path)
            agent.epsilon = 0.0  # greedy evaluation, no exploration
            return agent
    return None


@st.cache_resource(show_spinner=False)
def load_dqn_agent():
    """Try a few common checkpoint locations. Returns None if unavailable."""
    if not DQN_AVAILABLE:
        return None
    candidates = [
        os.path.join(ROOT_PATH, "models", "dqn_best.pt"),
        os.path.join(OUTPUTS_PATH, "dqn_checkpoints", "dqn_best.pt"),
        os.path.join(OUTPUTS_PATH, "dqn_best.pt"),
    ]
    ep_ckpts = sorted(
        glob.glob(os.path.join(OUTPUTS_PATH, "dqn_checkpoints", "dqn_ep*.pt"))
    )
    if ep_ckpts:
        candidates.append(ep_ckpts[-1])

    for path in candidates:
        if path and os.path.exists(path):
            agent = DQNAgent()
            agent.policy_net.load_state_dict(torch.load(path, map_location="cpu"))
            agent.policy_net.eval()
            return agent
    return None


# ----------------------------------------------------------------------------
# HELPER: run any agent for n episodes and collect stats + a sample trajectory
# ----------------------------------------------------------------------------
@st.cache_data(show_spinner=False)
def run_agent_evaluation(agent_name, n_episodes=200, seed=42):
    np.random.seed(seed)
    env = PricingEnv()

    q_agent = load_qlearning_agent()
    dqn_agent = load_dqn_agent()

    agent_map = {
        "FixedPrice": (FixedPriceAgent(), False),
        "TimeBasedDiscount": (TimeBasedDiscountAgent(), True),
        "DemandBased": (DemandBasedAgent(), False),
    }
    if q_agent is not None:
        agent_map["Q-Learning"] = (q_agent, False)
    if dqn_agent is not None:
        agent_map["DQN"] = (dqn_agent, False)

    agent, has_reset = agent_map[agent_name]
    use_greedy = agent_name == "DQN"

    revenues, sell_through = [], []
    sample_prices = []

    for ep in range(n_episodes):
        obs, info = env.reset()
        if has_reset:
            agent.reset()
        total_reward = 0
        initial_inventory = env.max_inventory
        done = False
        while not done:
            action = agent.act_greedy(obs) if use_greedy else agent.act(obs)
            if ep == 0:
                sample_prices.append(action)
            obs, reward, terminated, truncated, info = env.step(action)
            total_reward += reward
            done = terminated or truncated
        revenues.append(total_reward)
        sell_through.append((initial_inventory - obs[0]) / initial_inventory)

    return {
        "revenues": revenues,
        "sell_through": sell_through,
        "mean_revenue": float(np.mean(revenues)),
        "std_revenue": float(np.std(revenues)),
        "mean_sell_through": float(np.mean(sell_through)),
        "sample_prices": sample_prices,
    }


def available_agents():
    names = ["FixedPrice", "TimeBasedDiscount", "DemandBased"]
    if load_qlearning_agent() is not None:
        names.append("Q-Learning")
    if load_dqn_agent() is not None:
        names.append("DQN")
    return names


# ----------------------------------------------------------------------------
# SIDEBAR
# ----------------------------------------------------------------------------
with st.sidebar:
    st.markdown(
        f"""
        <div style="display:flex;align-items:center;gap:10px;margin-bottom:2px;">
            <div style="font-size:1.7rem;">🛫</div>
            <div style="font-family:{FONT_DISPLAY};font-weight:700;font-size:1.2rem;color:{TEXT};">
                Runway Dashboard
            </div>
        </div>
        <div style="color:{MUTED};font-size:0.82rem;margin-bottom:18px;">
            Dynamic Pricing RL — Travel &amp; Hospitality
        </div>
        """,
        unsafe_allow_html=True,
    )

    page = st.radio(
        "Navigate",
        [
            "🏠  Overview",
            "🧩  Environment Design",
            "📊  Agent Comparison",
            "🖼️  Results Gallery",
            "📈  Policy Analysis",
            "💼  Business Recommendations",
            "👥  Team & Project Info",
        ],
        label_visibility="collapsed",
    )

    st.markdown(f"<hr style='border-color:{BORDER};'>", unsafe_allow_html=True)
    n_episodes = st.slider("Episodes per agent", 50, 1000, 200, step=50)
    st.caption("Higher = more accurate, slower to load.")

    st.markdown(f"<hr style='border-color:{BORDER};'>", unsafe_allow_html=True)
    st.markdown("**Live Agent Status**")
    if IMPORT_ERROR:
        st.error("⚠️ Couldn't import src/ modules. Run from project root.")
    else:
        st.markdown('<span class="badge badge-live">● Environment</span>', unsafe_allow_html=True)
        q_ok = load_qlearning_agent() is not None
        d_ok = load_dqn_agent() is not None
        st.markdown(
            f'<span class="badge {"badge-live" if q_ok else "badge-pending"}">'
            f'{"● Q-Learning" if q_ok else "○ Q-Learning (not found)"}</span>',
            unsafe_allow_html=True,
        )
        st.markdown(
            f'<span class="badge {"badge-live" if d_ok else "badge-pending"}">'
            f'{"● DQN" if d_ok else "○ DQN (checkpoint not found)"}</span>',
            unsafe_allow_html=True,
        )

    st.markdown(f"<hr style='border-color:{BORDER};'>", unsafe_allow_html=True)
    st.caption("Infotact Solutions & Co.\nBengaluru, Karnataka")

page = page.split("  ", 1)[1]

# ============================================================================
# PAGE 1 — OVERVIEW
# ============================================================================
if page == "Overview":
    st.markdown(
        f"""
        <div class="hero">
            <span class="hero-eyebrow">✈ Reinforcement Learning · Dynamic Pricing</span>
            <div class="hero-title">Autonomous Pricing Agent for<br><span class="accent">Travel &amp; Hospitality</span></div>
            <div class="hero-sub">
                A DQN agent trained to set booking prices in real time — benchmarked against
                heuristic and tabular Q-Learning baselines across 1,000 simulated booking seasons.
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown(
        f"""
        <div class="bp-row">
            <div class="bp-card" style="--accent:{SKY};"><div class="bp-label">Sprint Duration</div><div class="bp-value">4 <span style="font-size:1rem;color:{MUTED};">weeks</span></div></div>
            <div class="bp-card" style="--accent:{CORAL};"><div class="bp-label">Team Size</div><div class="bp-value">4</div></div>
            <div class="bp-card" style="--accent:{EMERALD};"><div class="bp-label">Agents Live</div><div class="bp-value">{len(available_agents())}</div></div>
            <div class="bp-card" style="--accent:{VIOLET};"><div class="bp-label">Simulated Seasons</div><div class="bp-value">1,000</div></div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    col1, col2 = st.columns([1.3, 1])
    with col1:
        st.markdown('<div class="section-eyebrow">Mission Brief</div>', unsafe_allow_html=True)
        panel(
            f"""Train a <b>Deep Q-Network (DQN)</b> agent to autonomously set booking prices that
            <b style="color:{SKY};">outperform heuristic baselines</b> — fixed, time-based discount,
            and demand-based pricing — as well as a tabular Q-Learning baseline, in mean episodic
            revenue across 1,000 simulated seasons.""",
            "sky",
        )

        st.markdown('<div class="section-eyebrow" style="margin-top:6px;">Flight Path — 4 Week Sprint</div>', unsafe_allow_html=True)
        weeks = [
            ("Week 1", "MDP formulation &<br>custom Gym environment", SKY),
            ("Week 2", "Heuristic baselines &<br>tabular Q-Learning", CORAL),
            ("Week 3", "Deep Q-Network<br>(DQN) implementation", EMERALD),
            ("Week 4", "Large-scale evaluation &<br>business dashboard", VIOLET),
        ]
        stops_html = "".join(
            f"""<div class="route-stop" style="--dot:{c};"><div class="route-line"></div>
                <div class="route-dot"></div>
                <div class="route-week">{w}</div>
                <div class="route-focus">{f}</div></div>"""
            for w, f, c in weeks
        )
        st.markdown(f'<div class="route">{stops_html}</div>', unsafe_allow_html=True)

    with col2:
        st.markdown('<div class="section-eyebrow">Tech Stack</div>', unsafe_allow_html=True)
        stack = ["Python", "Gymnasium", "PyTorch", "NumPy", "Pandas", "Streamlit", "Plotly"]
        chips = "".join(f'<span class="chip">{s}</span>' for s in stack)
        panel(chips, "violet")

        st.markdown('<div class="section-eyebrow">Target KPI</div>', unsafe_allow_html=True)
        panel(
            f"""Trained DQN agent must <b style="color:{EMERALD};">outperform all baselines</b>
            (heuristic and tabular Q-Learning) in mean episodic revenue across 1,000 simulated
            seasons.""",
            "emerald",
        )

# ============================================================================
# PAGE 2 — ENVIRONMENT DESIGN
# ============================================================================
elif page == "Environment Design":
    st.markdown('<div class="section-eyebrow">Module 01</div>', unsafe_allow_html=True)
    st.markdown("## 🧩 Environment Design")
    st.caption("Custom Gymnasium environment — `PricingEnv`")

    col1, col2 = st.columns(2)
    with col1:
        panel(
            f"""<b>State Space</b><br>
            <code>Box([remaining_inventory, days_until_departure])</code>
            <ul style="color:{MUTED};margin-top:8px;">
                <li><code>remaining_inventory</code>: 0 – 100 units</li>
                <li><code>days_until_departure</code>: 0 – 30 days</li>
            </ul>""",
            "sky",
        )
        panel('<b>Action Space</b><br><code>Discrete(10)</code> — 10 price level bins', "coral")
    with col2:
        panel('<b>Reward Function</b><br><code>reward = price_level × units_sold</code>', "emerald")
        panel(
            f"""<b>Episode Termination</b>
            <ul style="color:{MUTED};margin-top:8px;">
                <li>Inventory reaches 0 (sold out), <b>or</b></li>
                <li><code>days_until_departure</code> reaches 0</li>
            </ul>""",
            "violet",
        )

    panel(
        f"""<b>Demand Model</b><br>
        A stochastic demand function using a <b style="color:{SKY};">logistic curve</b> determines
        units sold each step. Purchase probability decreases as price increases, and increases as
        the deadline approaches (urgency effect).""",
        "amber",
    )

    st.markdown("---")

    if not IMPORT_ERROR:
        st.markdown('<div class="section-eyebrow">Interactive</div>', unsafe_allow_html=True)
        st.markdown("### 🔬 Live Environment Demo")
        st.caption("Step through the environment manually to see how state changes.")

        if "demo_env" not in st.session_state:
            st.session_state.demo_env = PricingEnv()
            st.session_state.demo_obs, _ = st.session_state.demo_env.reset()
            st.session_state.demo_log = []

        c1, c2 = st.columns(2)
        c1.metric("Remaining Inventory", int(st.session_state.demo_obs[0]))
        c2.metric("Days Until Departure", int(st.session_state.demo_obs[1]))

        action = st.slider("Choose a price level (0 = lowest, 9 = highest)", 0, 9, 5)
        bc1, bc2 = st.columns([1, 1])
        with bc1:
            step_clicked = st.button("✈ Step Environment", use_container_width=True)
        with bc2:
            reset_clicked = st.button("↺ Reset Episode", use_container_width=True)

        if step_clicked:
            obs, reward, terminated, truncated, info = st.session_state.demo_env.step(action)
            st.session_state.demo_obs = obs
            st.session_state.demo_log.append(
                {"Price Level": action, "Units Sold": info["units_sold"],
                 "Reward": round(reward, 2), "Inventory": obs[0], "Days Left": obs[1]}
            )
            if terminated:
                st.warning("Episode ended — resetting environment.")
                st.session_state.demo_obs, _ = st.session_state.demo_env.reset()
            st.rerun()

        if reset_clicked:
            st.session_state.demo_obs, _ = st.session_state.demo_env.reset()
            st.session_state.demo_log = []
            st.rerun()

        if st.session_state.demo_log:
            log_df = pd.DataFrame(st.session_state.demo_log)
            st.dataframe(log_df, use_container_width=True, hide_index=True)

            fig = go.Figure()
            fig.add_trace(go.Scatter(y=log_df["Inventory"], name="Inventory", line=dict(color=SKY, width=3)))
            fig.add_trace(go.Bar(y=log_df["Reward"], name="Reward", marker_color=CORAL, opacity=0.65, yaxis="y2"))
            fig.update_layout(
                yaxis=dict(title="Inventory"),
                yaxis2=dict(title="Reward", overlaying="y", side="right"),
                title="Live Trajectory",
            )
            st.plotly_chart(plotly_theme(fig, height=340), use_container_width=True)
    else:
        st.error(f"Environment module not found: {IMPORT_ERROR}")

# ============================================================================
# PAGE 3 — AGENT COMPARISON
# ============================================================================
elif page == "Agent Comparison":
    st.markdown('<div class="section-eyebrow">Module 02</div>', unsafe_allow_html=True)
    st.markdown("## 📊 Agent Comparison")
    agents_now = available_agents()
    st.caption(" · ".join(agents_now) + "  (agents load automatically when their trained artifacts are found)")

    if IMPORT_ERROR:
        st.error(f"Cannot run evaluation — module import failed: {IMPORT_ERROR}")
    else:
        with st.spinner(f"Running {n_episodes} episodes across {len(agents_now)} agents..."):
            results = {name: run_agent_evaluation(name, n_episodes) for name in agents_now}

        best_agent = max(results, key=lambda k: results[k]["mean_revenue"])

        kpi_html = "".join(
            f"""<div class="bp-card" style="--accent:{AGENT_COLOR.get(name, SKY)};">
                    <div class="bp-label">{name}{" 🏆" if name == best_agent else ""}</div>
                    <div class="bp-value" style="color:{AGENT_COLOR.get(name, SKY)};">{res['mean_revenue']:.0f}</div>
                    <div style="color:{MUTED};font-size:0.78rem;margin-top:2px;">
                        σ {res['std_revenue']:.0f} · {res['mean_sell_through']*100:.0f}% sold
                    </div>
                </div>"""
            for name, res in results.items()
        )
        st.markdown(f'<div class="bp-row">{kpi_html}</div>', unsafe_allow_html=True)
        st.success(f"🏆 Best performing agent: **{best_agent}**")

        if "DQN" not in agents_now or "Q-Learning" not in agents_now:
            missing = [a for a in ["Q-Learning", "DQN"] if a not in agents_now]
            st.info(
                f"{' and '.join(missing)} not shown here because the trained artifact wasn't "
                f"found locally. Pre-generated results for these are available in the "
                f"**Results Gallery** tab."
            )

        tab1, tab2, tab3, tab4 = st.tabs(
            ["📋 Summary Table", "📉 Revenue Distribution", "🛫 Price Trajectory", "🧪 Significance Test"]
        )

        with tab1:
            summary_rows = [
                {
                    "Agent": name,
                    "Episodes": n_episodes,
                    "Mean Revenue": round(res["mean_revenue"], 2),
                    "Std Dev": round(res["std_revenue"], 2),
                    "Sell-Through Rate": f"{res['mean_sell_through']*100:.1f}%",
                    "Status": "✅ Live",
                }
                for name, res in results.items()
            ]
            for rl_agent in ["Q-Learning", "DQN"]:
                if rl_agent not in agents_now:
                    summary_rows.append(
                        {"Agent": rl_agent, "Episodes": "—", "Mean Revenue": "See Gallery",
                         "Std Dev": "—", "Sell-Through Rate": "—", "Status": "🕓 Not found"}
                    )
            st.dataframe(pd.DataFrame(summary_rows), use_container_width=True, hide_index=True)

            bar_fig = px.bar(
                x=list(results.keys()), y=[results[k]["mean_revenue"] for k in results],
                labels={"x": "Agent", "y": "Mean Revenue"}, color=list(results.keys()),
                color_discrete_map=AGENT_COLOR, text_auto=".2f",
            )
            bar_fig.update_layout(showlegend=False, title="Mean Revenue by Agent")
            st.plotly_chart(plotly_theme(bar_fig), use_container_width=True)

        with tab2:
            box_data = [
                {"Agent": name, "Revenue": r}
                for name, res in results.items() for r in res["revenues"]
            ]
            box_df = pd.DataFrame(box_data)
            box_fig = px.box(box_df, x="Agent", y="Revenue", color="Agent", color_discrete_map=AGENT_COLOR)
            box_fig.update_layout(showlegend=False, title="Revenue Distribution")
            st.plotly_chart(plotly_theme(box_fig), use_container_width=True)

            violin_fig = px.violin(box_df, x="Agent", y="Revenue", color="Agent",
                                    color_discrete_map=AGENT_COLOR, box=True, points=False)
            violin_fig.update_layout(showlegend=False, title="Revenue Density")
            st.plotly_chart(plotly_theme(violin_fig), use_container_width=True)

        with tab3:
            traj_fig = go.Figure()
            for name, res in results.items():
                traj_fig.add_trace(
                    go.Scatter(y=res["sample_prices"], mode="lines+markers", name=name,
                               line=dict(color=AGENT_COLOR.get(name, SKY), width=3))
                )
            traj_fig.update_layout(
                xaxis_title="Step (Day)", yaxis_title="Price Level (0–9)",
                title="Sample Episode — Price Trajectory",
            )
            st.plotly_chart(plotly_theme(traj_fig), use_container_width=True)

        with tab4:
            try:
                from scipy import stats
                names = list(results.keys())
                c1, c2 = st.columns(2)
                agent_a = c1.selectbox("Agent A", names, index=0)
                agent_b = c2.selectbox("Agent B", names, index=min(1, len(names) - 1))
                if agent_a != agent_b:
                    t_stat, p_value = stats.ttest_rel(
                        results[agent_a]["revenues"], results[agent_b]["revenues"]
                    )
                    m1, m2 = st.columns(2)
                    m1.metric("t-statistic", f"{t_stat:.4f}")
                    m2.metric("p-value", f"{p_value:.6f}")
                    if p_value < 0.05:
                        st.success("✅ Statistically significant difference (p < 0.05)")
                    else:
                        st.warning("⚠️ Not statistically significant (p ≥ 0.05)")
                else:
                    st.info("Choose two different agents to compare.")
            except ImportError:
                st.info("Install `scipy` (`pip install scipy`) to enable significance testing.")

# ============================================================================
# PAGE 4 — RESULTS GALLERY (pre-generated artifacts from outputs/)
# ============================================================================
elif page == "Results Gallery":
    st.markdown('<div class="section-eyebrow">Module 03</div>', unsafe_allow_html=True)
    st.markdown("## 🖼️ Results Gallery")
    st.caption("Pre-generated plots produced during training and evaluation — pulled directly from `outputs/`.")

    if not os.path.isdir(OUTPUTS_PATH):
        st.warning("`outputs/` folder not found next to app.py.")
    else:
        gtab1, gtab2, gtab3 = st.tabs(["🎓 Q-Learning", "🧠 DQN", "⚖️ Cross-Agent Comparison"])

        with gtab1:
            c1, c2 = st.columns(2)
            with c1:
                if not show_image_if_exists("qlearning_training_curve.png", "Q-Learning training curve"):
                    st.info("qlearning_training_curve.png not found.")
            with c2:
                if not show_image_if_exists("qlearning_policy_behavior.png", "Learned pricing policy behavior"):
                    st.info("qlearning_policy_behavior.png not found.")
            if not show_image_if_exists("qlearning_sample_trajectories.png", "Sample Q-Learning episode trajectories"):
                st.info("qlearning_sample_trajectories.png not found.")

        with gtab2:
            c1, c2 = st.columns(2)
            with c1:
                if not show_image_if_exists("dqn_convergence_curves.png", "DQN convergence across random seeds"):
                    st.info("dqn_convergence_curves.png not found.")
            with c2:
                if not show_image_if_exists("dqn_policy_analysis.png", "DQN learned policy analysis"):
                    st.info("dqn_policy_analysis.png not found.")
            if not show_image_if_exists("dqn_multi_season_trajectories.png", "DQN price trajectories across seasons"):
                st.info("dqn_multi_season_trajectories.png not found.")

        with gtab3:
            c1, c2 = st.columns(2)
            with c1:
                if not show_image_if_exists("baseline_comparison_boxplot.png", "Heuristic baseline revenue comparison"):
                    st.info("baseline_comparison_boxplot.png not found.")
            with c2:
                if not show_image_if_exists("all_agents_violin_comparison.png", "All agents — revenue density comparison"):
                    st.info("all_agents_violin_comparison.png not found.")
            c3, c4 = st.columns(2)
            with c3:
                if not show_image_if_exists("random_agent_revenue_histogram.png", "Random-agent baseline distribution"):
                    st.info("random_agent_revenue_histogram.png not found.")
            with c4:
                if not show_image_if_exists("inventory_depletion.png", "Inventory depletion over a season"):
                    st.info("inventory_depletion.png not found.")
            if not show_image_if_exists("sample_episode_trajectory.png", "Sample full-season episode trajectory"):
                st.info("sample_episode_trajectory.png not found.")

# ============================================================================
# PAGE 5 — POLICY ANALYSIS
# ============================================================================
elif page == "Policy Analysis":
    st.markdown('<div class="section-eyebrow">Module 04</div>', unsafe_allow_html=True)
    st.markdown("## 📈 Policy Analysis")

    panel(
        f"""<b>Time-Based Discount Agent</b><br>
        <span style="color:{MUTED};">Prices decay ~10% per day, resulting in aggressive
        last-minute discounting <i>regardless</i> of actual demand.</span>""",
        "coral",
    )
    panel(
        f"""<b>Demand-Based Agent</b><br>
        <span style="color:{MUTED};">Prices adjust dynamically based on the inventory-to-time
        ratio — raising prices when inventory sells fast relative to time remaining, and
        lowering prices when inventory is high relative to time left.</span>""",
        "emerald",
    )
    panel(
        f"""<b>Q-Learning / DQN (Learned Agents)</b><br>
        <span style="color:{MUTED};">Combine both effects — discounting near the deadline
        <i>only when</i> inventory-clearing risk is high, since the reward signal directly
        penalizes underpricing when demand is otherwise strong. See the policy heatmaps in the
        Results Gallery for the actual learned behavior.</span>""",
        "violet",
    )
    panel(
        f"""⚠️ <b>Edge Cases to Monitor</b><br>
        <span style="color:{MUTED};">Learned policies risk discovering degenerate behavior,
        such as pricing near zero purely to guarantee a sale late in the season. This motivates
        the safety bounds in the Business Recommendations tab.</span>""",
        "amber",
    )

    if show_image_if_exists("dqn_policy_analysis.png", "DQN policy analysis — price level by state"):
        pass
    elif show_image_if_exists("qlearning_policy_behavior.png", "Q-Learning policy behavior — price level by state"):
        pass
    else:
        st.info("Policy heatmap image not found in outputs/ — see the Results Gallery tab.")

# ============================================================================
# PAGE 6 — BUSINESS RECOMMENDATIONS
# ============================================================================
elif page == "Business Recommendations":
    st.markdown('<div class="section-eyebrow">Module 05</div>', unsafe_allow_html=True)
    st.markdown("## 💼 Business Recommendations")

    col1, col2 = st.columns(2)
    with col1:
        st.markdown('<div class="section-eyebrow">Safety Bounds</div>', unsafe_allow_html=True)
        panel(
            f"""<ul style="color:{MUTED};margin:0;">
                <li><b style="color:{TEXT};">Minimum price bound</b> — prevents excessive
                    discounting that erodes margin</li>
                <li><b style="color:{TEXT};">Maximum price bound</b> — prevents pricing so high
                    that sell-through fails entirely</li>
            </ul>""",
            "sky",
        )
        min_bound = st.slider("Minimum price level", 0, 9, 2)
        max_bound = st.slider("Maximum price level", 0, 9, 8)
        st.caption(f"Configured safe range: price levels **{min_bound}–{max_bound}**")

    with col2:
        st.markdown('<div class="section-eyebrow">Monitoring Guidance</div>', unsafe_allow_html=True)
        panel(
            f"""<ul style="color:{MUTED};margin:0;">
                <li>Sudden drops in sell-through rate → pricing may be too high</li>
                <li>Excessive last-minute discounting → reward function may encourage
                    inventory dumping</li>
                <li>Periodic re-evaluation against updated heuristic baselines as market
                    conditions change</li>
            </ul>""",
            "emerald",
        )

    st.markdown('<div class="section-eyebrow">Deployment Readiness</div>', unsafe_allow_html=True)
    st.success(
        "The environment and evaluation framework have been validated across 1,000 simulated "
        "seasons per agent, with statistically significant performance differences confirmed "
        "via paired t-testing. Adaptive and learned strategies outperform static pricing in "
        "both mean revenue and sell-through rate."
    )

# ============================================================================
# PAGE 7 — TEAM & PROJECT INFO
# ============================================================================
elif page == "Team & Project Info":
    st.markdown('<div class="section-eyebrow">Crew Manifest</div>', unsafe_allow_html=True)
    st.markdown("## 👥 Team & Project Info")

    team = [
        ("01", "Tarun Saxena", "Environment & Simulation Engineer", SKY),
        ("02", "Vaibhav Gautam", "RL Algorithm Engineer", CORAL),
        ("03", "Vaibhav Gautam", "Analysis & Policy Evaluation", EMERALD),
        ("04", "Tarun Saxena", "Eval & Deploy Lead", VIOLET),
    ]
    cols = st.columns(4)
    for c, (num, name, role, color) in zip(cols, team):
        c.markdown(
            f"""
            <div class="panel" style="border-left:4px solid {color};text-align:center;">
                <div style="color:{color};font-family:{FONT_DISPLAY};font-weight:700;font-size:1.5rem;">{num}</div>
                <div style="font-weight:700;margin-top:6px;">{name}</div>
                <div style="color:{MUTED};font-size:0.82rem;margin-top:2px;">{role}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    col1, col2 = st.columns([1, 1])
    with col1:
        st.markdown('<div class="section-eyebrow">Organization</div>', unsafe_allow_html=True)
        panel(
            f"""<b>Infotact Solutions &amp; Co.</b>, Bengaluru, Karnataka<br>
            <span style="color:{MUTED};">Domain: Data Science &amp; Machine Learning</span><br>
            <span style="color:{MUTED};">Internship: 25 May 2026 – 25 Aug 2026</span>""",
            "sky",
        )
    with col2:
        st.markdown('<div class="section-eyebrow">Repository Structure</div>', unsafe_allow_html=True)
        st.code(
            """project-root/
├── app.py
├── requirements.txt
├── README.md
├── outputs/        (generated plots + trained_qtable.npy)
├── notebooks/
└── src/
    ├── pricing_env.py
    ├── baseline_agents.py
    ├── q_learning_agent.py
    ├── dqn_agent.py
    └── train_dqn.py""",
            language="text",
        )

# ----------------------------------------------------------------------------
# FOOTER
# ----------------------------------------------------------------------------
st.markdown(
    '<div class="footer-note">INFOTACT SOLUTIONS &amp; CO. · BENGALURU, KARNATAKA · '
    "AUTONOMOUS PRICING AGENT VIA REINFORCEMENT LEARNING · v1.0.0</div>",
    unsafe_allow_html=True,
)
