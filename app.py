"""
AI Risk & Finance Dashboard
===========================

Foundational Streamlit architecture for tracking multi-platform wealth and
macroeconomic exposure.

Structure
---------
1. Configuration & styling ......... page config + lightweight CSS polish
2. Data layer ...................... mock data providers (swap for live APIs later)
3. Gemini integration .............. API-key resolution, prompt building, and
                                     the google-genai client call
4. UI components ................... reusable render functions (KPIs, sidebar,
                                     holdings, stress test, AI feed, AI advisor)
5. Main ............................ page assembly

All data-facing functions are defensive: they return safe defaults when data
is missing or malformed so the UI never crashes on incomplete inputs.
"""

from __future__ import annotations

import os
from typing import Any

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

# ---------------------------------------------------------------------------
# 1. CONFIGURATION & STYLING
# ---------------------------------------------------------------------------

st.set_page_config(
    page_title="AI Risk & Finance Dashboard",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Minimal CSS polish: tightens metric cards, styles the macro rate cards and
# the AI alert feed without fighting Streamlit's native theme.
CUSTOM_CSS = """
<style>
    /* KPI metric cards */
    div[data-testid="stMetric"] {
        background: rgba(28, 131, 225, 0.06);
        border: 1px solid rgba(28, 131, 225, 0.15);
        border-radius: 12px;
        padding: 14px 18px;
    }

    /* Sidebar macro rate cards */
    .macro-card {
        background: rgba(255, 255, 255, 0.04);
        border: 1px solid rgba(128, 128, 128, 0.25);
        border-radius: 10px;
        padding: 10px 14px;
        margin-bottom: 8px;
    }
    .macro-card .label { font-size: 0.78rem; opacity: 0.7; }
    .macro-card .value { font-size: 1.15rem; font-weight: 700; }
    .macro-card .delta { font-size: 0.75rem; }
    .delta-up   { color: #e45756; }   /* rising rates/inflation = risk tone */
    .delta-down { color: #54a24b; }

    /* AI recommendation feed */
    .ai-alert {
        border-left: 4px solid;
        border-radius: 6px;
        padding: 10px 14px;
        margin-bottom: 10px;
        background: rgba(255, 255, 255, 0.03);
    }
    .ai-alert.high   { border-color: #e45756; }
    .ai-alert.medium { border-color: #f2a93b; }
    .ai-alert.low    { border-color: #54a24b; }
    .ai-alert .severity { font-size: 0.72rem; font-weight: 700;
                          text-transform: uppercase; letter-spacing: 0.05em; }
    .ai-alert .msg { font-size: 0.88rem; margin-top: 2px; }

    /* Gemini AI Risk Advisor response card */
    .advisor-card {
        background: linear-gradient(135deg, rgba(66, 133, 244, 0.08),
                                            rgba(156, 39, 176, 0.06));
        border: 1px solid rgba(66, 133, 244, 0.30);
        border-radius: 12px;
        padding: 18px 22px;
        margin-top: 8px;
    }
    .advisor-card .advisor-header {
        font-size: 0.75rem; font-weight: 700; text-transform: uppercase;
        letter-spacing: 0.06em; opacity: 0.7; margin-bottom: 8px;
    }
</style>
"""
st.markdown(CUSTOM_CSS, unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# 2. DATA LAYER (mock providers — replace with live API calls later)
# ---------------------------------------------------------------------------

@st.cache_data(ttl=300)
def get_holdings() -> pd.DataFrame:
    """Return the sample multi-platform portfolio.

    In production this would aggregate broker/exchange/bank APIs. Returns an
    empty DataFrame with the expected schema if anything goes wrong, so
    downstream renderers can rely on the columns existing.
    """
    columns = ["Asset", "Ticker", "Platform", "Asset Class",
               "Allocation %", "Current Value USD"]
    try:
        rows = [
            ("S&P 500 ETF",           "VOO",     "Vanguard",         "Equities",     28.0, 140_000),
            ("Nasdaq 100 ETF",        "QQQ",     "Interactive Brokers", "Equities",  15.0,  75_000),
            ("MSCI World ex-US",      "VEU",     "Vanguard",         "Equities",      7.0,  35_000),
            ("US Treasury 2Y Ladder", "SHY",     "Fidelity",         "Fixed Income", 14.0,  70_000),
            ("UK Gilts ETF",          "IGLT.L",  "Hargreaves Lansdown", "Fixed Income", 6.0, 30_000),
            ("Corp IG Bond Fund",     "LQD",     "Fidelity",         "Fixed Income",  6.0,  30_000),
            ("Bitcoin",               "BTC",     "Coinbase",         "Crypto",        8.0,  40_000),
            ("Ethereum",              "ETH",     "Kraken",           "Crypto",        4.0,  20_000),
            ("USD Money Market",      "VMFXX",   "Vanguard",         "Cash",          8.0,  40_000),
            ("GBP Instant Access",    "—",       "Barclays",         "Cash",          4.0,  20_000),
        ]
        return pd.DataFrame(rows, columns=columns)
    except Exception:
        return pd.DataFrame(columns=columns)


@st.cache_data(ttl=300)
def get_macro_rates() -> dict[str, list[dict[str, Any]]]:
    """Return placeholder macroeconomic indicators grouped by region.

    Each entry: name, value (display string), delta (bps/pp change) and
    direction ("up"/"down"/"flat"). Swap for FRED / BoE / ECB API calls later.
    """
    return {
        "🇺🇸 United States": [
            {"name": "SOFR",            "value": "5.31%", "delta": "+1bp MoM",  "direction": "up"},
            {"name": "Fed Funds Rate",  "value": "5.25–5.50%", "delta": "unchanged", "direction": "flat"},
        ],
        "🇬🇧 United Kingdom": [
            {"name": "BoE Base Rate",   "value": "5.00%", "delta": "-25bps last MPC", "direction": "down"},
            {"name": "CPI (YoY)",       "value": "2.2%",  "delta": "+0.2pp MoM", "direction": "up"},
        ],
        "🇪🇺 Euro Area": [
            {"name": "ECB Deposit Rate", "value": "3.75%", "delta": "-25bps last meeting", "direction": "down"},
            {"name": "HICP (YoY)",       "value": "2.6%",  "delta": "+0.1pp MoM", "direction": "up"},
        ],
    }


@st.cache_data(ttl=300)
def get_ai_recommendations() -> list[dict[str, str]]:
    """Return mock AI-generated risk alerts (severity: high/medium/low)."""
    return [
        {"severity": "high",
         "title": "Crypto concentration",
         "message": "Crypto allocation (12%) exceeds your 10% risk budget. "
                    "Consider trimming BTC into short-duration Treasuries."},
        {"severity": "medium",
         "title": "Cash drag under inflation",
         "message": "Cash allocation exceeds 15% under high inflation: "
                    "consider short-duration yields (e.g. 3–6M T-bills at ~5.3%)."},
        {"severity": "medium",
         "title": "Rate-cut sensitivity",
         "message": "BoE and ECB are in easing cycles. GBP/EUR cash yields may "
                    "compress — locking duration in gilts could preserve carry."},
        {"severity": "low",
         "title": "Rebalance drift",
         "message": "Equities drifted +2.1% above target since last rebalance. "
                    "Next scheduled rebalance window: end of quarter."},
    ]


# --- Portfolio math helpers -------------------------------------------------

def compute_kpis(holdings: pd.DataFrame) -> dict[str, float]:
    """Derive headline KPIs from holdings; safe against empty/missing data."""
    kpis = {"total_value": 0.0, "cash_pct": 0.0, "pnl_24h": 0.0, "pnl_24h_pct": 0.0}
    if holdings is None or holdings.empty:
        return kpis

    values = pd.to_numeric(holdings.get("Current Value USD"), errors="coerce").fillna(0)
    total = float(values.sum())
    kpis["total_value"] = total

    if total > 0 and "Asset Class" in holdings.columns:
        cash_value = float(values[holdings["Asset Class"] == "Cash"].sum())
        kpis["cash_pct"] = 100.0 * cash_value / total

    # Mock 24h P&L (would come from position-level price deltas in production)
    kpis["pnl_24h"] = 3_240.0
    kpis["pnl_24h_pct"] = 100.0 * kpis["pnl_24h"] / total if total else 0.0
    return kpis


def run_stress_test(holdings: pd.DataFrame,
                    equity_shock: float = -0.40,
                    rate_shock_bps: float = -200.0) -> pd.DataFrame:
    """Apply a simple scenario shock per asset class and return before/after.

    Assumptions (illustrative only):
      - Equities move by `equity_shock`.
      - Fixed income gains ~duration × rate move (avg duration ≈ 4y assumed).
      - Crypto behaves as high-beta equity (1.5× equity shock, floored at -95%).
      - Cash is unaffected in nominal terms.
    """
    schema = ["Asset Class", "Current Value", "Stressed Value", "Change"]
    if holdings is None or holdings.empty:
        return pd.DataFrame(columns=schema)

    avg_duration = 4.0
    fi_gain = avg_duration * (-rate_shock_bps / 10_000.0)   # -200bps -> +8%
    shock_map = {
        "Equities": equity_shock,
        "Fixed Income": fi_gain,
        "Crypto": max(equity_shock * 1.5, -0.95),
        "Cash": 0.0,
    }

    df = holdings.copy()
    df["Current Value USD"] = pd.to_numeric(df["Current Value USD"], errors="coerce").fillna(0)
    grouped = df.groupby("Asset Class", sort=False)["Current Value USD"].sum().reset_index()
    grouped["Shock"] = grouped["Asset Class"].map(shock_map).fillna(0.0)
    grouped["Stressed Value"] = grouped["Current Value USD"] * (1 + grouped["Shock"])
    grouped["Change"] = grouped["Stressed Value"] - grouped["Current Value USD"]
    return grouped.rename(columns={"Current Value USD": "Current Value"})[schema]


# ---------------------------------------------------------------------------
# 3. GEMINI INTEGRATION (AI Risk Advisor)
# ---------------------------------------------------------------------------

# Primary model first; each subsequent entry is tried if the previous fails
# (e.g. model not available on the key's tier, or transient API error).
GEMINI_MODELS = ("gemini-3.6-flash", "gemini-2.5-flash", "gemini-2.5-pro")
GEMINI_MODEL = GEMINI_MODELS[0]


def resolve_gemini_api_key() -> str:
    """Resolve the Gemini API key from (in priority order):

    1. `GEMINI_API_KEY` environment variable (works with .env via `dotenv`
       or an exported shell variable),
    2. Streamlit secrets (`.streamlit/secrets.toml`),
    3. the session value entered in the sidebar password field.

    Returns an empty string when no key is configured anywhere.
    """
    env_key = os.environ.get("GEMINI_API_KEY", "").strip()
    if env_key:
        return env_key

    try:  # st.secrets raises if no secrets file exists at all
        secret_key = str(st.secrets.get("GEMINI_API_KEY", "")).strip()
        if secret_key:
            return secret_key
    except Exception:
        pass

    return str(st.session_state.get("gemini_api_key_input", "")).strip()


def build_risk_prompt(holdings: pd.DataFrame,
                      kpis: dict[str, float],
                      macro: dict[str, list[dict[str, Any]]]) -> str:
    """Serialize portfolio + macro context into a structured Gemini prompt.

    The model is instructed to respond in fixed markdown sections so the UI
    can render the output predictably.
    """
    holdings_block = (
        holdings.to_markdown(index=False)
        if holdings is not None and not holdings.empty
        else "(no holdings data available)"
    )

    macro_lines: list[str] = []
    for region, indicators in (macro or {}).items():
        for ind in indicators:
            macro_lines.append(
                f"- {region} {ind.get('name', '?')}: {ind.get('value', '?')} "
                f"({ind.get('delta', 'n/a')})"
            )
    macro_block = "\n".join(macro_lines) or "(no macro data available)"

    return f"""You are a senior portfolio risk manager at a global asset manager.
Analyze the client portfolio below in the context of the current macroeconomic
rate environment.

## Portfolio holdings
{holdings_block}

Total portfolio value: ${kpis.get('total_value', 0):,.0f}
Cash allocation: {kpis.get('cash_pct', 0):.1f}%

## Current macro rates
{macro_block}

## Your task
Respond in **exactly** this markdown structure (keep the headings verbatim):

### 🔻 3 Key Vulnerabilities
1. ... (specific to this allocation given current interest rates)
2. ...
3. ...

### 🔄 2 Concrete Rebalancing Actions
1. ... (name specific assets/tickers from the table and target percentages)
2. ...

### 🩺 Portfolio Health Score
**Score: NN/100** — one-sentence justification.

Be concise, quantitative where possible, and reference the actual holdings
and rates provided. Do not add any sections beyond the three above."""


def get_gemini_analysis(api_key: str, prompt: str) -> tuple[str, str]:
    """Call Gemini and return ``(analysis_text, error_message)``.

    Exactly one of the two is non-empty. Tries the models in ``GEMINI_MODELS``
    order (`gemini-3.6-flash` first, then 2.5 fallbacks) until one succeeds.
    Import is deferred so the dashboard still runs when `google-genai` is not
    installed.
    """
    if not api_key:
        return "", "No API key provided."

    try:
        from google import genai
    except ImportError:
        return "", ("The `google-genai` package is not installed. "
                    "Run `pip install -r requirements.txt` and restart the app.")

    client = genai.Client(api_key=api_key)

    errors: list[str] = []
    for model in GEMINI_MODELS:
        try:
            response = client.models.generate_content(
                model=model,
                contents=prompt,
            )
            text = (response.text or "").strip()
            if text:
                return text, ""
            errors.append(f"{model}: returned an empty response")
        except Exception as exc:  # API/auth/quota errors — try next model
            errors.append(f"{model}: {exc}")

    return "", "Gemini API error — all models failed. " + " | ".join(errors)


# ---------------------------------------------------------------------------
# 4. UI COMPONENTS
# ---------------------------------------------------------------------------

def render_header() -> None:
    st.title("📊 AI Risk & Finance Dashboard")
    st.caption("Tracking multi-platform wealth & macroeconomic exposure — "
               "equities, fixed income, crypto and cash in one view.")
    st.divider()


def render_kpis(kpis: dict[str, float]) -> None:
    """Top-line KPI metric row."""
    col1, col2, col3 = st.columns(3)
    col1.metric(
        "Total Portfolio Value",
        f"${kpis['total_value']:,.0f}",
        help="Aggregated across all connected platforms.",
    )
    col2.metric(
        "Cash Allocation",
        f"{kpis['cash_pct']:.1f}%",
        help="Money market + bank deposits as % of total.",
    )
    col3.metric(
        "24h P&L",
        f"${kpis['pnl_24h']:+,.0f}",
        delta=f"{kpis['pnl_24h_pct']:+.2f}%",
        help="Mark-to-market change over the past 24 hours (mock).",
    )


def render_sidebar(macro: dict[str, list[dict[str, Any]]]) -> None:
    """Sidebar with macroeconomic tracker rate cards, grouped by region."""
    with st.sidebar:
        st.header("🌍 Macro Trackers")
        st.caption("Placeholder values — wire to FRED / BoE / ECB feeds.")

        if not macro:
            st.info("Macro data unavailable.")
            return

        for region, indicators in macro.items():
            st.subheader(region)
            for ind in indicators:
                direction = ind.get("direction", "flat")
                delta_cls = {"up": "delta-up", "down": "delta-down"}.get(direction, "")
                arrow = {"up": "▲", "down": "▼"}.get(direction, "•")
                st.markdown(
                    f"""
                    <div class="macro-card">
                        <div class="label">{ind.get("name", "—")}</div>
                        <div class="value">{ind.get("value", "—")}</div>
                        <div class="delta {delta_cls}">{arrow} {ind.get("delta", "")}</div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

        st.divider()

        # --- Gemini API key (only ask if not already set via env/secrets) ---
        st.header("🔑 Gemini API")
        if os.environ.get("GEMINI_API_KEY", "").strip():
            st.success("API key loaded from environment.", icon="✅")
        else:
            st.text_input(
                "GEMINI_API_KEY",
                type="password",
                key="gemini_api_key_input",
                help="Stored only in this session. Alternatively set the "
                     "GEMINI_API_KEY environment variable or add it to "
                     ".streamlit/secrets.toml.",
                placeholder="Paste your Gemini API key",
            )

        st.divider()
        st.caption("Data refresh: every 5 min (cached) · All figures indicative.")


def render_holdings(holdings: pd.DataFrame) -> None:
    """Holdings table with formatted currency/percentage columns."""
    st.subheader("💼 Holdings")
    if holdings is None or holdings.empty:
        st.info("No holdings data available. Connect a platform to begin.")
        return

    st.dataframe(
        holdings,
        width="stretch",
        hide_index=True,
        column_config={
            "Allocation %": st.column_config.ProgressColumn(
                "Allocation %", format="%.1f%%", min_value=0, max_value=100,
            ),
            "Current Value USD": st.column_config.NumberColumn(
                "Current Value (USD)", format="$%,.0f",
            ),
        },
    )


def render_stress_test(holdings: pd.DataFrame) -> None:
    """Interactive stress-test preview card for the 2008-crisis scenario."""
    st.subheader("⚠️ Stress Test Preview")

    scenario = st.selectbox(
        "Scenario",
        ["2008 Financial Crisis (-40% Equities / -200bps Rates)"],
        help="More scenarios (COVID-2020, 1970s stagflation, custom) coming soon.",
    )
    st.caption(f"Simulating: **{scenario}**")

    result = run_stress_test(holdings, equity_shock=-0.40, rate_shock_bps=-200)
    if result.empty:
        st.info("Stress test unavailable — no portfolio data.")
        return

    current_total = float(result["Current Value"].sum())
    stressed_total = float(result["Stressed Value"].sum())
    change = stressed_total - current_total
    change_pct = 100.0 * change / current_total if current_total else 0.0

    m1, m2 = st.columns(2)
    m1.metric("Portfolio (stressed)", f"${stressed_total:,.0f}",
              delta=f"{change_pct:+.1f}%")
    m2.metric("Impact", f"${change:+,.0f}")

    # Grouped bar: current vs stressed value by asset class
    fig = go.Figure(data=[
        go.Bar(name="Current", x=result["Asset Class"], y=result["Current Value"],
               marker_color="#1c83e1"),
        go.Bar(name="Stressed", x=result["Asset Class"], y=result["Stressed Value"],
               marker_color="#e45756"),
    ])
    fig.update_layout(
        barmode="group", height=300,
        margin=dict(l=10, r=10, t=10, b=10),
        yaxis_title="Value (USD)", legend=dict(orientation="h", y=1.1),
    )
    st.plotly_chart(fig, width="stretch")


def render_ai_feed(alerts: list[dict[str, str]]) -> None:
    """Mock AI recommendation feed with severity-coded alert cards."""
    st.subheader("🤖 AI Recommendation Feed")
    if not alerts:
        st.info("No active risk alerts. Portfolio within all risk budgets.")
        return

    for alert in alerts:
        severity = alert.get("severity", "low")
        st.markdown(
            f"""
            <div class="ai-alert {severity}">
                <div class="severity">{severity} · {alert.get("title", "Alert")}</div>
                <div class="msg">{alert.get("message", "")}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    st.caption("Alerts are illustrative — generated by mock rules engine.")


def render_ai_advisor(holdings: pd.DataFrame,
                      kpis: dict[str, float],
                      macro: dict[str, list[dict[str, Any]]]) -> None:
    """Gemini-powered AI Risk Advisor panel.

    Sends holdings, total value, and macro rates to Gemini on demand and
    renders the structured analysis (vulnerabilities, rebalancing actions,
    health score) in a styled card. The last result is kept in session state
    so it survives unrelated widget reruns.
    """
    st.subheader("✨ AI Risk Advisor · Gemini")
    st.caption(f"Model: `{GEMINI_MODEL}` — acts as a senior risk manager over "
               "your live holdings and macro context.")

    api_key = resolve_gemini_api_key()

    clicked = st.button(
        "🔎 Analyze Portfolio & Macro Risk with Gemini",
        type="primary",
        disabled=not api_key,
        help=None if api_key else "Enter your GEMINI_API_KEY in the sidebar first.",
    )
    if not api_key:
        st.info("Add your Gemini API key in the sidebar (🔑 Gemini API) to "
                "enable the advisor.", icon="🔑")

    if clicked:
        prompt = build_risk_prompt(holdings, kpis, macro)
        with st.spinner("Gemini is analyzing your portfolio and macro exposure…"):
            analysis, error = get_gemini_analysis(api_key, prompt)
        st.session_state["advisor_analysis"] = analysis
        st.session_state["advisor_error"] = error

    error = st.session_state.get("advisor_error", "")
    analysis = st.session_state.get("advisor_analysis", "")

    if error:
        st.error(error, icon="⚠️")
    elif analysis:
        st.markdown('<div class="advisor-card">'
                    '<div class="advisor-header">Gemini Risk Analysis</div>'
                    '</div>', unsafe_allow_html=True)
        with st.container(border=True):
            st.markdown(analysis)
        st.caption("AI-generated analysis — informational only, not financial advice.")


# ---------------------------------------------------------------------------
# 5. MAIN
# ---------------------------------------------------------------------------

def main() -> None:
    holdings = get_holdings()
    macro = get_macro_rates()
    alerts = get_ai_recommendations()
    kpis = compute_kpis(holdings)

    render_header()
    render_sidebar(macro)
    render_kpis(kpis)
    st.divider()

    render_holdings(holdings)
    st.divider()

    # Stress test and AI feed side by side
    left, right = st.columns([3, 2], gap="large")
    with left:
        render_stress_test(holdings)
    with right:
        render_ai_feed(alerts)

    st.divider()
    render_ai_advisor(holdings, kpis, macro)


if __name__ == "__main__":
    main()
