"""
AI Risk & Finance Dashboard
===========================

Foundational Streamlit architecture for tracking multi-platform wealth and
macroeconomic exposure.

Structure
---------
1. Configuration & styling ......... page config + lightweight CSS polish
2. Data layer ...................... editable holdings input + yfinance live
                                     price engine + portfolio math
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
import time
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

# Minimal CSS polish: tightens metric cards and styles the AI alert feed
# without fighting Streamlit's native theme.
CUSTOM_CSS = """
<style>
    /* KPI metric cards */
    div[data-testid="stMetric"] {
        background: rgba(28, 131, 225, 0.06);
        border: 1px solid rgba(28, 131, 225, 0.15);
        border-radius: 12px;
        padding: 14px 18px;
    }

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
# 2. DATA LAYER (user-editable holdings + live price engine)
# ---------------------------------------------------------------------------

# The 5 fields the user enters directly; everything else is computed.
INPUT_COLUMNS = ["Asset", "Ticker", "Platform", "Asset Class", "Units Owned"]
ASSET_CLASSES = ["Equities", "Fixed Income", "Crypto", "Cash", "Commodities"]
PLATFORMS = ["Vanguard", "Trading 212", "Interactive Brokers", "Fidelity",
             "Hargreaves Lansdown", "Cash/Bank"]
CASH_TICKER = "CASH"
HOLDINGS_STATE_KEY = "holdings_input"


def default_holdings_input() -> pd.DataFrame:
    """Seed portfolio for first load — Yahoo-Finance-valid tickers so live
    pricing works out of the box. USD assets are auto-converted to GBP; the
    `.L` ticker demonstrates native pence (GBp) handling. Cash units = GBP.
    """
    rows = [
        ("S&P 500 ETF",        "VOO",     "Vanguard",            "Equities",     120.0),
        ("Nasdaq 100 ETF",     "QQQ",     "Interactive Brokers", "Equities",      80.0),
        ("Apple",              "AAPL",    "Interactive Brokers", "Equities",      50.0),
        ("US Treasury 1-3Y",   "SHY",     "Fidelity",            "Fixed Income", 400.0),
        ("UK Gilts ETF",       "IGLT.L",  "Hargreaves Lansdown", "Fixed Income", 200.0),
        ("Bitcoin",            "BTC-USD", "Trading 212",         "Crypto",         0.5),
        ("Ethereum",           "ETH-USD", "Trading 212",         "Crypto",         4.0),
        ("GBP Cash",           "CASH",    "Cash/Bank",           "Cash",       30_000.0),
    ]
    return pd.DataFrame(rows, columns=INPUT_COLUMNS)


@st.cache_data(ttl=60, show_spinner=False)
def fetch_live_price(ticker: str) -> dict[str, Any]:
    """Fetch the live price (and previous close) for one ticker via yfinance.

    Returns ``{"price": float, "prev_close": float, "currency": str, "ok": bool}``
    in the asset's *native* quote currency (currency conversion happens in
    :func:`enrich_holdings`). Cached for 60s so table edits and reruns stay
    fast. Any failure (bad ticker, network down, delisted symbol) degrades to
    price 0.0 with ``ok=False`` — callers flag the row instead of crashing.
    """
    result = {"price": 0.0, "prev_close": 0.0, "currency": "USD", "ok": False}
    ticker = (ticker or "").strip().upper()
    if not ticker:
        return result

    try:
        import yfinance as yf

        asset = yf.Ticker(ticker)
        price = prev_close = 0.0

        # Fast path: fast_info avoids a full history download.
        try:
            fast = asset.fast_info
            price = float(fast["last_price"] or 0.0)
            prev_close = float(fast["previous_close"] or 0.0)
            # NB: keep the exact case — Yahoo uses "GBp" for pence quotes,
            # which must not be confused with "GBP".
            result["currency"] = str(fast["currency"] or "USD")
        except Exception:
            pass

        # Fallback: latest daily closes (also fills a missing prev_close).
        if price <= 0 or price != price or prev_close <= 0:
            closes = asset.history(period="5d")["Close"].dropna()
            if not closes.empty:
                if price <= 0 or price != price:
                    price = float(closes.iloc[-1])
                if prev_close <= 0 and len(closes) > 1:
                    prev_close = float(closes.iloc[-2])

        if price > 0:
            result.update(price=price,
                          prev_close=prev_close if prev_close > 0 else price,
                          ok=True)
    except Exception:
        pass  # keep the safe default

    return result


# Used only if the live GBP/USD quote cannot be fetched (approximate rate).
FALLBACK_USD_TO_GBP = 0.79


@st.cache_data(ttl=60, show_spinner=False)
def get_usd_to_gbp() -> dict[str, Any]:
    """Return ``{"rate": USD→GBP multiplier, "ok": bool}``.

    Derived from the live ``GBPUSD=X`` quote (USD per 1 GBP). Falls back to
    an approximate constant when the FX feed is unavailable so the dashboard
    keeps working — callers surface a warning when ``ok`` is False.
    """
    quote = fetch_live_price("GBPUSD=X")
    if quote["ok"] and quote["price"] > 0:
        return {"rate": 1.0 / quote["price"], "ok": True}
    return {"rate": FALLBACK_USD_TO_GBP, "ok": False}


def convert_to_gbp(amount: float, currency: str, usd_to_gbp: float) -> float:
    """Convert an amount from its native quote currency into GBP (£).

    - ``GBp`` / ``GBX`` (LSE pence quotes) → divide by 100.
    - ``GBP``                              → unchanged.
    - ``USD`` and anything else            → multiply by the USD→GBP rate
      (non-USD majors are approximated via the USD leg for simplicity).
    """
    raw = str(currency or "USD")
    if raw == "GBp" or raw.upper() == "GBX":
        return amount / 100.0
    if raw.upper() == "GBP":
        return amount
    return amount * usd_to_gbp


def is_cash_row(ticker: str, asset_class: str) -> bool:
    """Cash rows are priced at a locked £1.00 (units == GBP)."""
    return (str(ticker).strip().upper() == CASH_TICKER
            or str(asset_class).strip() == "Cash")


def normalize_input(input_df: pd.DataFrame) -> pd.DataFrame:
    """Canonicalize a raw 5-column holdings input frame.

    Coerces units to float, strips text fields, and drops blank rows (added
    in the editor but not yet filled). Used both by the pricing pipeline and
    by the editor's change detection, so the two always compare like-for-like
    (identical dtypes and row order) — this is what prevents rerun loops.
    """
    if input_df is None or input_df.empty:
        return pd.DataFrame(columns=INPUT_COLUMNS)

    df = input_df.copy()
    for col in INPUT_COLUMNS:
        if col not in df.columns:
            df[col] = None

    df["Units Owned"] = pd.to_numeric(df["Units Owned"], errors="coerce").fillna(0.0)
    for col in ("Asset", "Ticker", "Platform", "Asset Class"):
        df[col] = df[col].fillna("").astype(str).str.strip()

    df = df[(df["Ticker"] != "") | (df["Asset"] != "")].reset_index(drop=True)
    return df[INPUT_COLUMNS]


def enrich_holdings(input_df: pd.DataFrame) -> pd.DataFrame:
    """Turn the 5-column user input into a fully-priced GBP holdings table.

    Native quotes are converted to £ (USD via the live GBP/USD rate; LSE
    pence quotes divided by 100). Adds: Live Price (£), Current Value GBP
    (= units × price), Allocation %, and a Price OK flag for rows whose
    ticker failed to price. Blank rows (added but not yet filled in the
    editor) are dropped. Never raises.
    """
    computed_cols = INPUT_COLUMNS + ["Live Price", "Current Value GBP",
                                     "Allocation %", "Price OK"]
    df = normalize_input(input_df)
    if df.empty:
        return pd.DataFrame(columns=computed_cols)

    usd_to_gbp = get_usd_to_gbp()["rate"]

    prices, prev_closes, ok_flags = [], [], []
    for _, row in df.iterrows():
        if is_cash_row(row["Ticker"], row["Asset Class"]):
            prices.append(1.0)
            prev_closes.append(1.0)
            ok_flags.append(True)
        else:
            quote = fetch_live_price(row["Ticker"])
            prices.append(convert_to_gbp(quote["price"], quote["currency"], usd_to_gbp))
            prev_closes.append(convert_to_gbp(quote["prev_close"], quote["currency"], usd_to_gbp))
            ok_flags.append(quote["ok"])

    df["Live Price"] = prices
    df["Prev Close"] = prev_closes
    df["Price OK"] = ok_flags
    df["Current Value GBP"] = df["Units Owned"] * df["Live Price"]

    total = float(df["Current Value GBP"].sum())
    df["Allocation %"] = (100.0 * df["Current Value GBP"] / total) if total > 0 else 0.0
    return df


# Benchmark macro defaults (editable in the sidebar). Each indicator has a
# stable session-state key so manual adjustments persist across reruns.
MACRO_DEFAULTS: dict[str, list[dict[str, Any]]] = {
    "🇬🇧 United Kingdom": [
        {"name": "BoE Base Rate",             "key": "macro_uk_boe",  "value": 3.75},
        {"name": "UK CPI (YoY)",              "key": "macro_uk_cpi",  "value": 2.60},
    ],
    "🇺🇸 United States": [
        {"name": "SOFR",                      "key": "macro_us_sofr", "value": 3.62},
        {"name": "Fed Funds Rate (upper)",    "key": "macro_us_ffr",  "value": 3.75},
    ],
    "🇪🇺 Euro Area": [
        {"name": "ECB Deposit Facility Rate", "key": "macro_eu_ecb",  "value": 2.25},
        {"name": "Eurozone HICP (YoY)",       "key": "macro_eu_hicp", "value": 2.20},
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
    """Derive headline KPIs from enriched holdings; safe against missing data.

    24h P&L is computed from real price deltas: units × (live price − previous
    close), summed over rows that priced successfully.
    """
    kpis = {"total_value": 0.0, "cash_pct": 0.0, "pnl_24h": 0.0,
            "pnl_24h_pct": 0.0, "num_holdings": 0}
    if holdings is None or holdings.empty:
        return kpis

    kpis["num_holdings"] = int(len(holdings))
    values = pd.to_numeric(holdings.get("Current Value GBP"), errors="coerce").fillna(0)
    total = float(values.sum())
    kpis["total_value"] = total

    if total > 0 and "Asset Class" in holdings.columns:
        cash_value = float(values[holdings["Asset Class"] == "Cash"].sum())
        kpis["cash_pct"] = 100.0 * cash_value / total

    if {"Units Owned", "Live Price", "Prev Close", "Price OK"}.issubset(holdings.columns):
        priced = holdings[holdings["Price OK"].astype(bool)]
        units = pd.to_numeric(priced["Units Owned"], errors="coerce").fillna(0)
        live = pd.to_numeric(priced["Live Price"], errors="coerce").fillna(0)
        prev = pd.to_numeric(priced["Prev Close"], errors="coerce").fillna(0)
        kpis["pnl_24h"] = float((units * (live - prev)).sum())

    prev_total = total - kpis["pnl_24h"]
    kpis["pnl_24h_pct"] = 100.0 * kpis["pnl_24h"] / prev_total if prev_total else 0.0
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
        "Commodities": equity_shock * 0.5,  # commodities sold off ~half as hard in 2008
    }

    df = holdings.copy()
    df["Current Value GBP"] = pd.to_numeric(df["Current Value GBP"], errors="coerce").fillna(0)
    grouped = df.groupby("Asset Class", sort=False)["Current Value GBP"].sum().reset_index()
    grouped["Shock"] = grouped["Asset Class"].map(shock_map).fillna(0.0)
    grouped["Stressed Value"] = grouped["Current Value GBP"] * (1 + grouped["Shock"])
    grouped["Change"] = grouped["Stressed Value"] - grouped["Current Value GBP"]
    return grouped.rename(columns={"Current Value GBP": "Current Value"})[schema]


# ---------------------------------------------------------------------------
# 3. GEMINI INTEGRATION (AI Risk Advisor)
# ---------------------------------------------------------------------------

# Active free-tier Gemini endpoints, tried in order by the self-healing
# fallback loop. On 503 (high demand) or 429 (rate limit) the engine waits
# 2 seconds before moving to the next model; any other API error skips ahead
# immediately. The first model to respond wins.
MODELS_TO_TRY = [
    "gemini-2.5-flash",
    "gemini-3.5-flash",
    "gemini-3.7-flash",
    "gemini-3.5-flash-lite",
]

# Errors that indicate transient load/quota pressure — worth a short pause
# before hitting the next endpoint.
RETRYABLE_MARKERS = ("503", "UNAVAILABLE", "429", "RESOURCE_EXHAUSTED")


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
            try:
                value_str = f"{float(ind.get('value', 0)):.2f}%"
            except (TypeError, ValueError):
                value_str = str(ind.get("value", "?"))
            macro_lines.append(f"- {region} {ind.get('name', '?')}: {value_str}")
    macro_block = "\n".join(macro_lines) or "(no macro data available)"

    return f"""You are a senior portfolio risk manager at a global asset manager.
Analyze the client portfolio below in the context of the current macroeconomic
rate environment. All monetary values are in British Pounds (GBP, £).

## Portfolio holdings
{holdings_block}

Total portfolio value: £{kpis.get('total_value', 0):,.2f}
Cash allocation: {kpis.get('cash_pct', 0):.1f}%

## Current macro rates (user-adjusted, live policy environment)
{macro_block}

## Your task
Respond in **exactly** this markdown structure (keep the headings verbatim):

### 🩺 Portfolio Health Score
**Score: NN/100** — one-sentence justification.

### 🔻 3 Rate & Allocation Vulnerabilities
1. ... (specific to this allocation given the interest rates above; quote
   affected amounts in GBP, £)
2. ...
3. ...

### 🔄 2 Immediate Rebalancing Recommendations
1. ... (name specific assets/tickers from the table and target percentages)
2. ...

Be concise, quantitative where possible, and reference the actual holdings
and rates provided. Do not add any sections beyond the three above."""


def get_gemini_analysis(api_key: str, prompt: str) -> tuple[str, str]:
    """Self-healing Gemini call. Returns ``(analysis_text, model_used, error)``.

    On success, ``analysis_text`` and ``model_used`` are set and ``error`` is
    empty. Iterates ``MODELS_TO_TRY`` in order: the first successful
    generation wins. On 503 (high demand) or 429 (rate limit) the loop sleeps
    2 seconds before trying the next model; any other API exception moves on
    immediately. If every model fails, ``error`` carries the combined
    diagnostics. Import is deferred so the dashboard still runs when
    `google-genai` is not installed.
    """
    if not api_key:
        return "", "", "No API key provided."

    try:
        from google import genai
    except ImportError:
        return "", "", ("The `google-genai` package is not installed. "
                        "Run `pip install -r requirements.txt` and restart the app.")

    client = genai.Client(api_key=api_key.strip())

    failures: list[str] = []
    for index, model in enumerate(MODELS_TO_TRY):
        try:
            response = client.models.generate_content(model=model, contents=prompt)
            text = (response.text or "").strip()
            if text:
                return text, model, ""  # success — caller renders the markdown
            raise RuntimeError("model returned an empty response")
        except Exception as exc:  # 503 / 429 / 404 / anything else
            failures.append(f"{model}: {exc}")
            next_model = (MODELS_TO_TRY[index + 1]
                          if index + 1 < len(MODELS_TO_TRY) else None)
            if next_model:
                if any(marker in str(exc) for marker in RETRYABLE_MARKERS):
                    st.warning(f"`{model}` under high demand / rate-limited — "
                               f"retrying with `{next_model}` in 2s…", icon="⏳")
                    time.sleep(2)
                else:
                    st.warning(f"`{model}` unavailable — trying `{next_model}`…",
                               icon="🔀")

    return "", "", ("Gemini API error — all models failed. "
                    + " | ".join(failures))


# ---------------------------------------------------------------------------
# 4. UI COMPONENTS
# ---------------------------------------------------------------------------

def render_header() -> None:
    st.title("📊 AI Risk & Finance Dashboard")
    st.caption("Tracking multi-platform wealth & macroeconomic exposure in "
               "GBP (£) — equities, fixed income, crypto and cash in one view.")
    st.divider()


def render_kpis(kpis: dict[str, float]) -> None:
    """Top-line visual overview cards."""
    col1, col2, col3, col4 = st.columns(4)
    col1.metric(
        "💷 Total Portfolio Value",
        f"£{kpis['total_value']:,.2f}",
        help="Aggregated across all platforms, in GBP.",
    )
    col2.metric(
        "📈 Total Day Change",
        f"£{kpis['pnl_24h']:+,.2f}",
        delta=f"{kpis['pnl_24h_pct']:+.2f}%",
        help="Units × (live price − previous close) in GBP, summed over priced rows.",
    )
    col3.metric(
        "🧾 Number of Holdings",
        f"{kpis['num_holdings']:d}",
        help="Positions currently tracked in the portfolio.",
    )
    col4.metric(
        "💰 Cash Allocation",
        f"{kpis['cash_pct']:.1f}%",
        help="Cash balances as % of total portfolio value.",
    )


def render_sidebar() -> dict[str, list[dict[str, Any]]]:
    """Sidebar with editable macroeconomic trackers, grouped by region.

    Each indicator is a number input seeded with the benchmark default and
    adjustable at any time (persisted in session state). Returns the current
    values as ``{region: [{"name": ..., "value": float}, ...]}`` so they can
    be fed into the Gemini risk prompt.
    """
    macro: dict[str, list[dict[str, Any]]] = {}

    with st.sidebar:
        st.header("🌍 Macro Trackers")
        st.caption("Benchmark policy rates & inflation — adjust anytime; the "
                   "AI Risk Advisor uses your values.")

        for region, indicators in MACRO_DEFAULTS.items():
            st.subheader(region)
            current: list[dict[str, Any]] = []
            for ind in indicators:
                value = st.number_input(
                    f"{ind['name']} (%)",
                    min_value=-5.0,
                    max_value=50.0,
                    value=float(ind["value"]),
                    step=0.05,
                    format="%.2f",
                    key=ind["key"],
                )
                current.append({"name": ind["name"], "value": float(value)})
            macro[region] = current

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
        st.caption("Prices cached 60s · FX via GBPUSD=X · All figures indicative.")

    return macro


def render_add_holding_form() -> None:
    """Quick 'Add New Stock / Share' form inside a collapsed expander.

    Appends the new asset to session state on submit; the caller re-enriches
    afterwards, so valuations recalculate automatically in the same rerun.
    """
    with st.expander("➕ Add New Stock / Share", expanded=False):
        with st.form("add_holding_form", clear_on_submit=True, border=False):
            c1, c2, c3 = st.columns(3)
            asset_name = c1.text_input(
                "Asset Name", placeholder="e.g. Apple, Vanguard S&P 500")
            ticker = c2.text_input(
                "Ticker", placeholder="e.g. AAPL, VUAG.L, BTC-USD, CASH")
            platform = c3.selectbox("Platform", PLATFORMS)

            c4, c5 = st.columns(2)
            asset_class = c4.selectbox("Asset Class", ASSET_CLASSES)
            units = c5.number_input(
                "Units / Shares Owned", min_value=0.0, value=0.0, step=1.0,
                format="%.4f", help="For CASH rows, enter the GBP amount.")

            submitted = st.form_submit_button(
                "✨ Add to Portfolio", type="primary", width="stretch")

        if submitted:
            if not asset_name.strip() or not ticker.strip():
                st.warning("Please provide both an Asset Name and a Ticker.",
                           icon="✏️")
            elif units <= 0:
                st.warning("Units / Shares Owned must be greater than zero.",
                           icon="🔢")
            else:
                new_row = pd.DataFrame(
                    [[asset_name.strip(), ticker.strip().upper(),
                      platform, asset_class, float(units)]],
                    columns=INPUT_COLUMNS)
                st.session_state[HOLDINGS_STATE_KEY] = pd.concat(
                    [normalize_input(st.session_state[HOLDINGS_STATE_KEY]),
                     new_row], ignore_index=True)
                st.success(f"Added **{asset_name.strip()}** ({ticker.strip().upper()}) "
                           "to your portfolio.", icon="✅")


def render_analytics(holdings: pd.DataFrame) -> None:
    """Visual analytics tab: asset-class donut + platform allocation bars."""
    if holdings is None or holdings.empty:
        st.info("No holdings yet — add your first asset above to see charts.")
        return

    values = pd.to_numeric(holdings["Current Value GBP"], errors="coerce").fillna(0)
    plot_df = holdings.assign(_value=values)
    plot_df = plot_df[plot_df["_value"] > 0]
    if plot_df.empty:
        st.info("Nothing to chart yet — holdings have no market value.")
        return

    col_donut, col_bar = st.columns(2)

    with col_donut:
        st.markdown("**Asset Class Breakdown**")
        by_class = plot_df.groupby("Asset Class", sort=False)["_value"].sum()
        donut = go.Figure(go.Pie(
            labels=by_class.index.tolist(),
            values=by_class.values.tolist(),
            hole=0.55,
            textinfo="label+percent",
            hovertemplate="%{label}: £%{value:,.2f} (%{percent})<extra></extra>",
        ))
        donut.update_layout(height=340, margin=dict(l=10, r=10, t=10, b=10),
                            showlegend=False)
        st.plotly_chart(donut, width="stretch")

    with col_bar:
        st.markdown("**Platform Allocation**")
        by_platform = (plot_df.groupby("Platform", sort=False)["_value"]
                       .sum().sort_values())
        bars = go.Figure(go.Bar(
            x=by_platform.values.tolist(),
            y=by_platform.index.tolist(),
            orientation="h",
            marker_color="#1c83e1",
            hovertemplate="%{y}: £%{x:,.2f}<extra></extra>",
        ))
        bars.update_layout(height=340, margin=dict(l=10, r=10, t=10, b=10),
                           xaxis_title="Value (£ GBP)")
        st.plotly_chart(bars, width="stretch")


def render_holdings_table(holdings: pd.DataFrame) -> None:
    """Single interactive grid: in-line editing + live computed GBP columns.

    The 5 input columns are editable (rows can be added or deleted); Live
    Price, Current Value and Weight are computed and locked. Any change is
    saved back to session state and immediately re-priced via ``st.rerun``.
    """
    if not get_usd_to_gbp()["ok"]:
        st.warning("Live GBP/USD rate unavailable — using an approximate "
                   f"fallback of {FALLBACK_USD_TO_GBP:.2f}. USD-quoted values "
                   "may be slightly off.", icon="💱")

    if holdings is not None and not holdings.empty:
        failed = holdings.loc[~holdings["Price OK"].astype(bool), "Ticker"].tolist()
        if failed:
            st.warning(f"Could not fetch prices for: {', '.join(failed)} — "
                       "these rows are valued at £0.00. Check the ticker "
                       "symbols.", icon="⚠️")

    display_cols = INPUT_COLUMNS + ["Live Price", "Current Value GBP", "Allocation %"]
    display = (holdings[display_cols] if holdings is not None and not holdings.empty
               else pd.DataFrame(columns=display_cols))

    # Include any legacy platform/class values so old sessions don't break
    # the selectbox columns.
    platform_options = list(dict.fromkeys(
        PLATFORMS + [p for p in display["Platform"].tolist() if p]))
    class_options = list(dict.fromkeys(
        ASSET_CLASSES + [c for c in display["Asset Class"].tolist() if c]))

    edited = st.data_editor(
        display,
        num_rows="dynamic",          # allow adding/deleting rows in the UI
        width="stretch",
        hide_index=True,
        key="holdings_editor",
        disabled=["Live Price", "Current Value GBP", "Allocation %"],
        column_config={
            "Asset": st.column_config.TextColumn(
                "Asset Name", help="e.g. Apple, Vanguard S&P 500",
                required=True),
            "Ticker": st.column_config.TextColumn(
                "Ticker", help="Yahoo Finance symbol, e.g. AAPL, VUAG.L, "
                "BTC-USD — or CASH for cash", required=True),
            "Platform": st.column_config.SelectboxColumn(
                "Platform", options=platform_options),
            "Asset Class": st.column_config.SelectboxColumn(
                "Asset Class", options=class_options, required=True),
            "Units Owned": st.column_config.NumberColumn(
                "Units / Shares", min_value=0.0, format="%.4f",
                help="Shares / coins held. For CASH rows: GBP amount."),
            "Live Price": st.column_config.NumberColumn(
                "Live Price (£)", format="£%.2f"),
            "Current Value GBP": st.column_config.NumberColumn(
                "Market Value (£)", format="£%,.2f"),
            "Allocation %": st.column_config.ProgressColumn(
                "Weight (%)", format="%.1f%%", min_value=0, max_value=100),
        },
    )

    # Persist in-line edits/deletions and re-price immediately. Both sides
    # are normalized identically, so this comparison is stable and cannot
    # trigger a rerun loop.
    new_input = normalize_input(edited[INPUT_COLUMNS])
    old_input = normalize_input(st.session_state[HOLDINGS_STATE_KEY])
    if not new_input.equals(old_input):
        st.session_state[HOLDINGS_STATE_KEY] = new_input
        st.rerun()


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
    m1.metric("Portfolio (stressed)", f"£{stressed_total:,.2f}",
              delta=f"{change_pct:+.1f}%")
    m2.metric("Impact", f"£{change:+,.2f}")

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
        yaxis_title="Value (£ GBP)", legend=dict(orientation="h", y=1.1),
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
    st.caption(f"Models: `{'` → `'.join(MODELS_TO_TRY)}` (auto-fallback) — "
               "acts as a senior risk manager over your live £ holdings and "
               "macro context.")

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
            analysis, model_used, error = get_gemini_analysis(api_key, prompt)
        st.session_state["advisor_analysis"] = analysis
        st.session_state["advisor_model"] = model_used
        st.session_state["advisor_error"] = error

    error = st.session_state.get("advisor_error", "")
    analysis = st.session_state.get("advisor_analysis", "")
    model_used = st.session_state.get("advisor_model", "")

    if error:
        st.error(error, icon="⚠️")
    elif analysis:
        st.markdown('<div class="advisor-card">'
                    '<div class="advisor-header">Gemini Risk Analysis</div>'
                    '</div>', unsafe_allow_html=True)
        with st.container(border=True):
            st.markdown(analysis)
        if model_used:
            st.caption(f"Generated via {model_used}")
        st.caption("AI-generated analysis — informational only, not financial advice.")


# ---------------------------------------------------------------------------
# 5. MAIN
# ---------------------------------------------------------------------------

def main() -> None:
    alerts = get_ai_recommendations()

    if HOLDINGS_STATE_KEY not in st.session_state:
        st.session_state[HOLDINGS_STATE_KEY] = default_holdings_input()

    render_header()
    macro = render_sidebar()  # editable macro rates, fed to the Gemini prompt

    # KPIs must appear above the holdings section but depend on its edits, so
    # reserve the slot now and fill it after the editor has run this rerun.
    kpi_slot = st.container()
    st.divider()

    st.subheader("💼 Holdings")
    st.caption("Add stocks & shares below — prices are fetched live from "
               "Yahoo Finance and converted to GBP (£). Use ticker `CASH` "
               "(or asset class *Cash*) for cash balances: 1 unit = £1.00.")

    render_add_holding_form()

    if st.button("🔄 Refresh Market Prices",
                 help="Clears the 60s price cache and refetches all tickers."):
        fetch_live_price.clear()

    with st.spinner("Fetching live market prices…"):
        holdings = enrich_holdings(st.session_state[HOLDINGS_STATE_KEY])

    tab_analytics, tab_table = st.tabs(["📊 Visual Analytics", "📋 Holdings Table"])
    with tab_analytics:
        render_analytics(holdings)
    with tab_table:
        render_holdings_table(holdings)

    kpis = compute_kpis(holdings)
    with kpi_slot:
        render_kpis(kpis)
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
