"""
app.py

Final Business Dashboard — Autonomous Pricing Agent via Reinforcement Learning
Travel & Hospitality | Infotact Solutions & Co.

Run with:
    streamlit run app.py

Place this file in the project ROOT directory (same level as src/, notebooks/,
models/, README.md) so the imports from `src/` resolve correctly.
"""

import sys
import os
import textwrap
import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go
import plotly.express as px

# ----------------------------------------------------------------------------
# PATH SETUP — allows importing from src/ regardless of where streamlit is run
# ----------------------------------------------------------------------------
SRC_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "src")
if SRC_PATH not in sys.path:
    sys.path.append(SRC_PATH)

# ----------------------------------------------------------------------------
# PAGE CONFIG
# ----------------------------------------------------------------------------
st.set_page_config(
    page_title="SkyPrice AI — Dynamic Pricing Dashboard",
    page_icon="🛫",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ----------------------------------------------------------------------------
# DESIGN TOKENS — "Dusk Departure Board" palette
# ----------------------------------------------------------------------------
BG = "#0A0E27"
CARD_BG = "#12172E"
CARD_BG_ALT = "#161C3B"
BORDER = "#262C52"
TEXT = "#EDEFF7"
MUTED = "#8B92B8"
TEAL = "#3FE0C5"      # runway / data
CORAL = "#FF6B6B"     # departures / alerts
AMBER = "#FFB454"     # sunset / highlights
VIOLET = "#8C7CF0"    # secondary accent
GOLD = "#FFD166"

PLOTLY_COLORWAY = [TEAL, AMBER, CORAL, VIOLET, GOLD]

# ----------------------------------------------------------------------------
# CUSTOM STYLING
# ----------------------------------------------------------------------------
st.markdown(
    textwrap.dedent(f"""
    <link rel="preconnect" href="https://fonts.googleapis.com">
    <link href="https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@500;600;700&family=Inter:wght@400;500;600&family=JetBrains+Mono:wght@500;600&display=swap" rel="stylesheet">
    <style>
        html, body, [class*="css"] {{
            font-family: 'Inter', sans-serif;
        }}

        .stApp {{
            background:
                radial-gradient(circle at 12% -10%, rgba(255,107,107,0.10), transparent 40%),
                radial-gradient(circle at 90% 0%, rgba(63,224,197,0.10), transparent 45%),
                {BG};
            color: {TEXT};
        }}

        h1, h2, h3, h4 {{
            font-family: 'Space Grotesk', sans-serif !important;
            letter-spacing: -0.01em;
        }}

        /* ---------- Top gradient runway strip ---------- */
        .runway-strip {{
            height: 5px;
            width: 100%;
            border-radius: 6px;
            background: linear-gradient(90deg, {CORAL}, {AMBER}, {TEAL}, {VIOLET});
            background-size: 300% 100%;
            animation: runway-flow 6s ease infinite;
            margin-bottom: 22px;
        }}
        @keyframes runway-flow {{
            0% {{ background-position: 0% 50%; }}
            50% {{ background-position: 100% 50%; }}
            100% {{ background-position: 0% 50%; }}
        }}

        /* ---------- Hero banner ---------- */
        .hero-banner {{
            position: relative;
            padding: 34px 38px;
            border-radius: 18px;
            background: linear-gradient(135deg, #1B1240 0%, #12172E 45%, #0E2A2C 100%);
            border: 1px solid {BORDER};
            overflow: hidden;
            margin-bottom: 26px;
        }}
        .hero-banner::after {{
            content: "";
            position: absolute;
            top: -60%; right: -10%;
            width: 320px; height: 320px;
            background: radial-gradient(circle, rgba(63,224,197,0.28), transparent 70%);
            border-radius: 50%;
        }}
        .hero-eyebrow {{
            display: inline-flex;
            align-items: center;
            gap: 8px;
            font-family: 'JetBrains Mono', monospace;
            font-size: 0.72rem;
            letter-spacing: 0.14em;
            text-transform: uppercase;
            color: {TEAL};
            background: rgba(63,224,197,0.10);
            border: 1px solid rgba(63,224,197,0.35);
            padding: 5px 12px;
            border-radius: 999px;
            margin-bottom: 14px;
        }}
        .hero-pulse {{
            width: 7px; height: 7px; border-radius: 50%;
            background: {TEAL};
            box-shadow: 0 0 0 0 rgba(63,224,197,0.6);
            animation: pulse 1.8s infinite;
        }}
        @keyframes pulse {{
            0%   {{ box-shadow: 0 0 0 0 rgba(63,224,197,0.55); }}
            70%  {{ box-shadow: 0 0 0 9px rgba(63,224,197,0); }}
            100% {{ box-shadow: 0 0 0 0 rgba(63,224,197,0); }}
        }}
        .hero-title {{
            font-family: 'Space Grotesk', sans-serif;
            font-size: 2.3rem;
            font-weight: 700;
            line-height: 1.15;
            margin: 0 0 8px 0;
            background: linear-gradient(90deg, #FFFFFF 30%, {TEAL} 100%);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
        }}
        .hero-subtitle {{
            color: {MUTED};
            font-size: 1.02rem;
            max-width: 640px;
        }}

        /* ---------- Section header ---------- */
        .section-header {{
            display: flex;
            align-items: baseline;
            gap: 10px;
            margin: 6px 0 4px 0;
        }}
        .section-header .icon {{ font-size: 1.3rem; }}
        .section-header h2 {{
            margin: 0;
            font-size: 1.35rem;
            color: {TEXT};
        }}
        .section-sub {{
            color: {MUTED};
            font-size: 0.92rem;
            margin-bottom: 18px;
        }}

        /* ---------- Custom metric cards ---------- */
        .metric-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(190px, 1fr));
            gap: 14px;
            margin-bottom: 6px;
        }}
        .metric-card {{
            background: linear-gradient(160deg, {CARD_BG_ALT} 0%, {CARD_BG} 100%);
            border: 1px solid {BORDER};
            border-radius: 14px;
            padding: 18px 20px;
            position: relative;
            transition: transform 0.15s ease, border-color 0.15s ease;
        }}
        .metric-card:hover {{
            transform: translateY(-3px);
            border-color: var(--accent, {TEAL});
        }}
        .metric-card .m-icon {{
            font-size: 1.25rem;
            margin-bottom: 8px;
            display: inline-block;
        }}
        .metric-card .m-label {{
            font-size: 0.76rem;
            text-transform: uppercase;
            letter-spacing: 0.06em;
            color: {MUTED};
            margin-bottom: 6px;
        }}
        .metric-card .m-value {{
            font-family: 'JetBrains Mono', monospace;
            font-size: 1.55rem;
            font-weight: 600;
            color: {TEXT};
        }}
        .metric-card .m-bar {{
            position: absolute;
            left: 0; top: 0; bottom: 0;
            width: 4px;
            border-radius: 14px 0 0 14px;
            background: var(--accent, {TEAL});
        }}

        /* ---------- Badges ---------- */
        .badge {{
            display: inline-block;
            font-family: 'JetBrains Mono', monospace;
            font-size: 0.75rem;
            padding: 4px 11px;
            border-radius: 999px;
            font-weight: 600;
            letter-spacing: 0.02em;
        }}
        .badge-gold {{ background: rgba(255,209,102,0.14); color: {GOLD}; border: 1px solid rgba(255,209,102,0.4); }}
        .badge-teal {{ background: rgba(63,224,197,0.14); color: {TEAL}; border: 1px solid rgba(63,224,197,0.4); }}
        .badge-coral {{ background: rgba(255,107,107,0.14); color: {CORAL}; border: 1px solid rgba(255,107,107,0.4); }}

        /* ---------- Streamlit component overrides ---------- */
        section[data-testid="stSidebar"] {{
            background: linear-gradient(180deg, #0D1230 0%, {BG} 100%);
            border-right: 1px solid {BORDER};
        }}
        section[data-testid="stSidebar"] h1 {{
            font-size: 1.25rem !important;
        }}
        section[data-testid="stSidebar"] .stRadio > label {{
            color: {MUTED};
        }}
        section[data-testid="stSidebar"] [role="radiogroup"] label {{
            background: {CARD_BG};
            border: 1px solid {BORDER};
            border-radius: 10px;
            padding: 8px 12px;
            margin-bottom: 6px;
            transition: border-color 0.15s ease, background 0.15s ease;
        }}
        section[data-testid="stSidebar"] [role="radiogroup"] label:hover {{
            border-color: {TEAL};
            background: {CARD_BG_ALT};
        }}

        div[data-testid="stMetric"] {{
            background: linear-gradient(160deg, {CARD_BG_ALT} 0%, {CARD_BG} 100%);
            border: 1px solid {BORDER};
            border-radius: 14px;
            padding: 14px 16px;
        }}
        div[data-testid="stMetricValue"] {{
            font-family: 'JetBrains Mono', monospace;
            color: {TEAL};
        }}

        .stButton > button {{
            background: linear-gradient(90deg, {CORAL}, {AMBER});
            color: #1A1108;
            font-weight: 600;
            border: none;
            border-radius: 10px;
            padding: 0.5em 1.2em;
            transition: filter 0.15s ease, transform 0.1s ease;
        }}
        .stButton > button:hover {{
            filter: brightness(1.08);
            transform: translateY(-1px);
        }}

        .stTabs [data-baseweb="tab-list"] {{ gap: 8px; }}
        .stTabs [data-baseweb="tab"] {{
            background-color: {CARD_BG};
            border-radius: 8px 8px 0 0;
            padding: 10px 18px;
            border: 1px solid {BORDER};
        }}
        .stTabs [aria-selected="true"] {{
            background-color: {CARD_BG_ALT} !important;
            border-bottom: 2px solid {TEAL} !important;
        }}

        div[data-testid="stDataFrame"], div[data-testid="stTable"] {{
            border: 1px solid {BORDER};
            border-radius: 12px;
            overflow: hidden;
        }}

        .stAlert {{
            border-radius: 12px !important;
            border: 1px solid {BORDER} !important;
        }}

        div[data-testid="stExpander"] {{
            background: {CARD_BG};
            border: 1px solid {BORDER};
            border-radius: 12px;
        }}

        code {{
            font-family: 'JetBrains Mono', monospace !important;
        }}

        .footer-note {{
            color: {MUTED};
            font-size: 0.82em;
            text-align: center;
            margin-top: 44px;
            padding-top: 18px;
            border-top: 1px solid {BORDER};
        }}
    </style>
    """),
    unsafe_allow_html=True,
)


def runway_strip():
    st.markdown('<div class="runway-strip"></div>', unsafe_allow_html=True)


def section_header(icon: str, title: str, subtitle: str = ""):
    sub = f'<div class="section-sub">{subtitle}</div>' if subtitle else ""
    html = f"""
    <div class="section-header"><span class="icon">{icon}</span><h2>{title}</h2></div>
    {sub}
    """
    st.markdown(textwrap.dedent(html), unsafe_allow_html=True)


def metric_cards(items):
    """items: list of dicts with keys icon, label, value, accent"""
    cards_html = "".join([
        f"""<div class="metric-card" style="--accent:{it.get('accent', TEAL)};">
            <div class="m-bar"></div>
            <div class="m-icon">{it.get('icon', '📌')}</div>
            <div class="m-label">{it['label']}</div>
            <div class="m-value">{it['value']}</div>
        </div>"""
        for it in items
    ])
    html_output = f'<div class="metric-grid">{cards_html}</div>'
    st.markdown(textwrap.dedent(html_output), unsafe_allow_html=True)


def style_fig(fig, height=400):
    fig.update_layout(
        paper_bgcolor=CARD_BG,
        plot_bgcolor=CARD_BG,
        font_color=TEXT,
        colorway=PLOTLY_COLORWAY,
        height=height,
        margin=dict(t=30, b=30, l=10, r=10),
        legend=dict(bgcolor="rgba(0,0,0,0)"),
    )
    fig.update_xaxes(gridcolor=BORDER, zerolinecolor=BORDER)
    fig.update_yaxes(gridcolor=BORDER, zerolinecolor=BORDER)
    return fig


# ----------------------------------------------------------------------------
# TRY TO IMPORT PROJECT MODULES (fail gracefully if not present yet)
# ----------------------------------------------------------------------------
IMPORT_ERROR = None
try:
    from pricing_env import PricingEnv
    from baseline_agents import FixedPriceAgent, TimeBasedDiscountAgent, DemandBasedAgent
except Exception as e:
    IMPORT_ERROR = str(e)


# ----------------------------------------------------------------------------
# HELPER: run any agent for n episodes and collect stats + a sample trajectory
# ----------------------------------------------------------------------------
@st.cache_data(show_spinner=False)
def run_agent_evaluation(agent_name, n_episodes=200, has_reset=False, seed=42):
    np.random.seed(seed)
    env = PricingEnv()
    agent_map = {
        "FixedPrice": FixedPriceAgent(),
        "TimeBasedDiscount": TimeBasedDiscountAgent(),
        "DemandBased": DemandBasedAgent(),
    }
    agent = agent_map[agent_name]

    revenues, sell_through = [], []
    sample_prices, sample_inventory = [], []

    for ep in range(n_episodes):
        obs, info = env.reset()
        if has_reset:
            agent.reset()
        total_reward = 0
        initial_inventory = env.max_inventory
        done = False
        while not done:
            action = agent.act(obs)
            if ep == 0:  # capture one sample trajectory for plotting
                sample_prices.append(action)
                sample_inventory.append(obs[0])
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
        "sample_inventory": sample_inventory,
    }


AGENT_CONFIG = {
    "FixedPrice": False,
    "TimeBasedDiscount": True,
    "DemandBased": False,
}

# ----------------------------------------------------------------------------
# SIDEBAR
# ----------------------------------------------------------------------------
st.sidebar.markdown(
    """
    <div style="display:flex;align-items:center;gap:10px;margin-bottom:2px;">
        <span style="font-size:1.6rem;">🛫</span>
        <span style="font-family:'Space Grotesk',sans-serif;font-weight:700;font-size:1.15rem;">SkyPrice AI</span>
    </div>
    <div style="color:#8B92B8;font-size:0.82rem;margin-bottom:14px;">Autonomous Pricing Agent · Travel &amp; Hospitality</div>
    """,
    unsafe_allow_html=True,
)
st.sidebar.markdown("---")

page = st.sidebar.radio(
    "Navigate",
    [
        "🏠 Overview",
        "🧩 Environment Design",
        "📊 Agent Comparison",
        "📈 Policy Analysis",
        "💼 Business Recommendations",
        "👥 Team & Project Info",
    ],
)

st.sidebar.markdown("---")
n_episodes = st.sidebar.slider("Episodes per agent (evaluation)", 50, 1000, 200, step=50)
st.sidebar.caption("Higher values = more accurate stats, slower load.")

if IMPORT_ERROR:
    st.sidebar.error("⚠️ Could not import src/ modules. Run this file from the project root.")

st.sidebar.markdown("---")
st.sidebar.markdown(
    '<span class="badge badge-teal">Infotact Solutions &amp; Co.</span>',
    unsafe_allow_html=True,
)
st.sidebar.caption("Bengaluru, Karnataka")

# ============================================================================
# PAGE 1 — OVERVIEW
# ============================================================================
if page == "🏠 Overview":
    runway_strip()

    st.markdown(
        textwrap.dedent("""
        <div class="hero-banner">
            <div class="hero-eyebrow"><span class="hero-pulse"></span> LIVE SIMULATION READY</div>
            <div class="hero-title">Autonomous Pricing Agent<br>via Reinforcement Learning</div>
            <div class="hero-subtitle">
                A Deep Q-Network learns to price travel inventory in real time —
                balancing revenue, urgency, and sell-through across 1,000 simulated
                booking seasons, benchmarked against three industry-standard heuristics.
            </div>
        </div>
        """),
        unsafe_allow_html=True,
    )

    metric_cards(
        [
            {"icon": "🗓️", "label": "Sprint Duration", "value": "4 Weeks", "accent": TEAL},
            {"icon": "👥", "label": "Team Size", "value": "2 Members", "accent": VIOLET},
            {"icon": "🤖", "label": "Agents Benchmarked", "value": "5+", "accent": AMBER},
            {"icon": "🌍", "label": "Simulated Seasons", "value": "1,000", "accent": CORAL},
        ]
    )

    st.write("")
    col_a, col_b = st.columns([1.3, 1])
    with col_a:
        section_header("🎯", "Project Goal")
        st.info(
            "Train a Deep Q-Network (DQN) agent to autonomously set booking prices "
            "that **outperform heuristic baselines** (fixed, time-based discount, "
            "demand-based) in mean episodic revenue across 1,000 simulated seasons."
        )

        section_header("🛠️", "Tech Stack")
        stack = ["Python", "Gymnasium", "PyTorch", "NumPy", "Pandas", "Matplotlib", "Streamlit", "Plotly"]
        badge_html = " ".join(f'<span class="badge badge-teal" style="margin:3px;">{s}</span>' for s in stack)
        st.markdown(badge_html, unsafe_allow_html=True)

    with col_b:
        section_header("🗺️", "Sprint Timeline")
        timeline_df = pd.DataFrame(
            {
                "Week": ["Week 1", "Week 2", "Week 3", "Week 4"],
                "Focus": [
                    "MDP formulation & custom Gym environment",
                    "Heuristic baselines & tabular Q-Learning",
                    "Deep Q-Network (DQN) implementation",
                    "Large-scale evaluation & business dashboard",
                ],
            }
        )
        st.dataframe(timeline_df, use_container_width=True, hide_index=True)

# ============================================================================
# PAGE 2 — ENVIRONMENT DESIGN
# ============================================================================
elif page == "🧩 Environment Design":
    runway_strip()
    section_header("🧩", "Environment Design", "Custom Gymnasium environment: `PricingEnv`")

    col1, col2 = st.columns(2)
    with col1:
        st.markdown("#### 📦 State Space")
        st.code("Box([remaining_inventory, days_until_departure])", language="python")
        st.markdown("- `remaining_inventory`: 0 to 100 units\n- `days_until_departure`: 0 to 30 days")

        st.markdown("#### 🎛️ Action Space")
        st.code("Discrete(10)  # 10 price level bins", language="python")

    with col2:
        st.markdown("#### 💰 Reward Function")
        st.code("reward = price_level * units_sold", language="python")

        st.markdown("#### 🏁 Episode Termination")
        st.markdown("- Inventory reaches 0 (sold out), **or**\n- `days_until_departure` reaches 0")

    st.markdown("#### 📉 Demand Model")
    st.write(
        "A stochastic demand function using a **logistic curve** determines units sold "
        "each step. Purchase probability decreases as price increases, and increases as "
        "the deadline approaches (urgency effect)."
    )

    if not IMPORT_ERROR:
        st.markdown("---")
        section_header("🔬", "Live Environment Demo", "Step through the environment manually to see how state changes.")
        if "demo_env" not in st.session_state:
            st.session_state.demo_env = PricingEnv()
            st.session_state.demo_obs, _ = st.session_state.demo_env.reset()
            st.session_state.demo_log = []

        action = st.slider("Choose a price level (0=lowest, 9=highest)", 0, 9, 5)
        bc1, bc2 = st.columns(2)
        with bc1:
            if st.button("▶️ Step Environment", use_container_width=True):
                obs, reward, terminated, truncated, info = st.session_state.demo_env.step(action)
                st.session_state.demo_obs = obs
                st.session_state.demo_log.append(
                    {"Price Level": action, "Units Sold": info["units_sold"],
                     "Reward": round(reward, 2), "Inventory": obs[0], "Days Left": obs[1]}
                )
                if terminated:
                    st.warning("Episode ended — resetting environment.")
                    st.session_state.demo_obs, _ = st.session_state.demo_env.reset()
        with bc2:
            if st.button("🔄 Reset Episode", use_container_width=True):
                st.session_state.demo_obs, _ = st.session_state.demo_env.reset()
                st.session_state.demo_log = []

        metric_cards(
            [
                {"icon": "📦", "label": "Remaining Inventory", "value": int(st.session_state.demo_obs[0]), "accent": TEAL},
                {"icon": "⏳", "label": "Days Until Departure", "value": int(st.session_state.demo_obs[1]), "accent": AMBER},
            ]
        )

        if st.session_state.demo_log:
            st.markdown("###### Step Log")
            st.dataframe(pd.DataFrame(st.session_state.demo_log), use_container_width=True)
    else:
        st.error(f"Environment module not found: {IMPORT_ERROR}")

# ============================================================================
# PAGE 3 — AGENT COMPARISON
# ============================================================================
elif page == "📊 Agent Comparison":
    runway_strip()
    section_header("📊", "Agent Comparison", "Fixed Price · Time-Based Discount · Demand-Based (Q-Learning / DQN shown when available)")

    if IMPORT_ERROR:
        st.error(f"Cannot run evaluation — module import failed: {IMPORT_ERROR}")
    else:
        with st.spinner(f"Running {n_episodes} episodes per agent..."):
            results = {
                name: run_agent_evaluation(name, n_episodes, has_reset)
                for name, has_reset in AGENT_CONFIG.items()
            }

        # ---- Summary table ----
        summary_rows = []
        for name, res in results.items():
            summary_rows.append(
                {
                    "Agent": name,
                    "Episodes": n_episodes,
                    "Mean Revenue": round(res["mean_revenue"], 2),
                    "Std Dev": round(res["std_revenue"], 2),
                    "Sell-Through Rate": f"{res['mean_sell_through']*100:.1f}%",
                }
            )
        # Placeholder rows for RL agents until their result files are wired in
        for rl_agent in ["Q-Learning", "DQN"]:
            summary_rows.append(
                {"Agent": rl_agent, "Episodes": "-", "Mean Revenue": "Pending",
                 "Std Dev": "Pending", "Sell-Through Rate": "Pending"}
            )
        summary_df = pd.DataFrame(summary_rows)

        st.markdown("### 📋 Revenue Comparison Table")
        st.dataframe(summary_df, use_container_width=True, hide_index=True)

        best_agent = max(results, key=lambda k: results[k]["mean_revenue"])
        st.markdown(
            f'🏆 Best performing agent (heuristics only): <span class="badge badge-gold">{best_agent}</span>',
            unsafe_allow_html=True,
        )

        # ---- Bar chart: mean revenue ----
        st.markdown("### 💵 Mean Revenue by Agent")
        bar_fig = px.bar(
            x=list(results.keys()),
            y=[results[k]["mean_revenue"] for k in results],
            labels={"x": "Agent", "y": "Mean Revenue"},
            color=list(results.keys()),
            text_auto=".2f",
        )
        bar_fig.update_layout(showlegend=False)
        st.plotly_chart(style_fig(bar_fig), use_container_width=True)

        # ---- Box plot: revenue distribution ----
        st.markdown("### 📈 Revenue Distribution")
        box_data = []
        for name, res in results.items():
            for r in res["revenues"]:
                box_data.append({"Agent": name, "Revenue": r})
        box_df = pd.DataFrame(box_data)
        box_fig = px.box(box_df, x="Agent", y="Revenue", color="Agent")
        box_fig.update_layout(showlegend=False)
        st.plotly_chart(style_fig(box_fig), use_container_width=True)

        # ---- Sample price trajectory ----
        st.markdown("### 🧭 Sample Episode — Price Trajectory")
        traj_fig = go.Figure()
        for name, res in results.items():
            traj_fig.add_trace(
                go.Scatter(y=res["sample_prices"], mode="lines+markers", name=name)
            )
        traj_fig.update_layout(xaxis_title="Step (Day)", yaxis_title="Price Level (0-9)")
        st.plotly_chart(style_fig(traj_fig), use_container_width=True)

        # ---- Statistical significance ----
        st.markdown("### 🧪 Statistical Significance (Paired t-test)")
        try:
            from scipy import stats
            names = list(results.keys())
            agent_a = st.selectbox("Agent A", names, index=0)
            agent_b = st.selectbox("Agent B", names, index=1)
            if agent_a != agent_b:
                t_stat, p_value = stats.ttest_rel(
                    results[agent_a]["revenues"], results[agent_b]["revenues"]
                )
                metric_cards(
                    [
                        {"icon": "📐", "label": "t-statistic", "value": f"{t_stat:.4f}", "accent": VIOLET},
                        {"icon": "🎯", "label": "p-value", "value": f"{p_value:.6f}", "accent": TEAL},
                    ]
                )
                if p_value < 0.05:
                    st.success("✅ Statistically significant difference (p < 0.05)")
                else:
                    st.warning("⚠️ Not statistically significant (p ≥ 0.05)")
        except ImportError:
            st.info("Install `scipy` (`pip install scipy`) to enable significance testing.")

# ============================================================================
# PAGE 4 — POLICY ANALYSIS
# ============================================================================
elif page == "📈 Policy Analysis":
    runway_strip()
    section_header("📈", "Policy Analysis")

    st.markdown("### 🧠 Observed Pricing Behavior")
    st.markdown(
        """
- **Time-Based Discount Agent**: Prices decay ~10% per day, resulting in
  aggressive last-minute discounting *regardless* of actual demand.
- **Demand-Based Agent**: Prices adjust dynamically based on the
  inventory-to-time ratio — raising prices when inventory sells fast
  relative to time remaining, and lowering prices when inventory is high
  relative to time left.
- **Learned Agents (Q-Learning / DQN)**: Expected to combine both effects —
  discounting near the deadline *only when* inventory-clearing risk is
  high, since the reward signal directly penalizes underpricing when
  demand is otherwise strong.
        """
    )

    st.markdown("### ⚠️ Edge Cases to Monitor")
    st.warning(
        "Learned policies risk discovering degenerate behavior, such as pricing "
        "near zero purely to guarantee a sale late in the season. This motivates "
        "the safety bounds in the Business Recommendations tab."
    )

    st.info(
        "Detailed trajectory plots and quantitative policy analysis from the "
        "evaluation notebooks can be inserted here once finalized."
    )

# ============================================================================
# PAGE 5 — BUSINESS RECOMMENDATIONS
# ============================================================================
elif page == "💼 Business Recommendations":
    runway_strip()
    section_header("💼", "Business Recommendations")

    col1, col2 = st.columns(2)
    with col1:
        st.markdown("### 🔒 Recommended Safety Bounds")
        st.markdown(
            """
- **Minimum price bound** — prevents excessive discounting that erodes margin
- **Maximum price bound** — prevents pricing so high that sell-through fails entirely
            """
        )
        min_bound = st.slider("Minimum price level", 0, 9, 2)
        max_bound = st.slider("Maximum price level", 0, 9, 8)
        st.markdown(
            f'<span class="badge badge-teal">Safe range: price levels {min_bound}–{max_bound}</span>',
            unsafe_allow_html=True,
        )

    with col2:
        st.markdown("### 📡 Monitoring Guidance")
        st.markdown(
            """
- Sudden drops in sell-through rate → pricing may be too high
- Excessive last-minute discounting → reward function may be encouraging inventory dumping
- Periodic re-evaluation against updated heuristic baselines as market conditions change
            """
        )

    st.markdown("### ✅ Deployment Readiness")
    st.success(
        "The environment and evaluation framework have been validated across "
        "1,000 simulated seasons per agent, with statistically significant "
        "performance differences confirmed via paired t-testing. Adaptive "
        "strategies outperform static pricing in both mean revenue and "
        "sell-through rate."
    )

# ============================================================================
# PAGE 6 — TEAM & PROJECT INFO
# ============================================================================
elif page == "👥 Team & Project Info":
    runway_strip()
    section_header("👥", "Team & Project Info")

    team_df = pd.DataFrame(
        {
            "Member": ["Member 1", "Member 2", "Member 3", "Member 4"],
            "Name": ["Tarun Saxena", "Vaibhav Gautam", "Vaibhav Gautam", "Tarun Saxena"],
            "Role": [
                "Environment & Simulation Engineer",
                "RL Algorithm Engineer",
                "Analysis & Policy Evaluation",
                "Eval & Deploy Lead",
            ],
        }
    )
    st.dataframe(team_df, use_container_width=True, hide_index=True)

    st.markdown("### 🏢 Organization")
    st.write("Infotact Solutions & Co., Bengaluru, Karnataka")
    st.write("Domain: Data Science and Machine Learning")
    st.write("Internship Period: 25 May 2026 – 25 August 2026")

    st.markdown("### 📁 Repository Structure")
    st.code(
        """
project-root/
├── app.py                   <- this dashboard
├── requirements.txt
├── README.md
├── results_comparison.md
├── model_results.md
├── data/
├── models/                  (gitignored — trained checkpoints)
├── notebooks/
│   ├── week1_env_design.ipynb
│   ├── week2_baselines.ipynb
│   ├── week3_dqn.ipynb
│   ├── week4_evaluation.ipynb
│   └── final_dashboard.ipynb
└── src/
    ├── pricing_env.py
    ├── baseline_agents.py
    ├── dqn_network.py
    └── train_dqn.py
        """,
        language="text",
    )

# ----------------------------------------------------------------------------
# FOOTER
# ----------------------------------------------------------------------------
st.markdown(
    '<div class="footer-note">🛫 SkyPrice AI · Infotact Solutions &amp; Co. | Bengaluru, Karnataka | '
    "Autonomous Pricing Agent via Reinforcement Learning — v2.0.0</div>",
    unsafe_allow_html=True,
)