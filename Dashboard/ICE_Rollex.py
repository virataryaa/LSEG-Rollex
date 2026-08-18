"""
ICE_Rollex.py — Roll-Adjusted Commodity Price Series (ICE Connect)
Run: streamlit run ICE_Rollex.py
"""

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st
from pathlib import Path
import scipy.stats as stats

# ── Data loading ──────────────────────────────────────────────────────────────
DB_DIR = Path(__file__).resolve().parent.parent / "Database"

AVAILABLE = ["KC", "RC", "CC", "LCC", "SB", "CT", "LSU"]

def load_rollex(comm: str) -> pd.DataFrame:
    path = DB_DIR / f"rollex_{comm}.parquet"
    df = pd.read_parquet(path)
    df.index = pd.to_datetime(df.index)
    df.index.name = "Date"
    return df.sort_index()

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(page_title="Rollex Dashboard", layout="wide",
                   initial_sidebar_state="expanded")

st.markdown("""<style>
  [data-testid="stAppViewContainer"],[data-testid="stMain"],.main{background:#fafafa!important}
  [data-testid="stHeader"]{background:transparent!important}
  .block-container{padding-top:1.5rem!important;padding-bottom:1.5rem;max-width:1440px}
  hr{border:none!important;border-top:1px solid #e8e8ed!important;margin:.3rem 0!important}
  .stDataFrame{font-size:.75rem}
  [data-testid="stSidebar"]{background:#f0f2f8!important}
  .stTabs [data-baseweb="tab-list"]{gap:4px}
  .stTabs [data-baseweb="tab"]{background:#f0f2f8;border-radius:6px 6px 0 0;
    padding:6px 18px;font-size:.8rem;font-weight:600;color:#444}
  .stTabs [aria-selected="true"]{background:#0a2463!important;color:#fff!important}
</style>""", unsafe_allow_html=True)

# ── Constants ─────────────────────────────────────────────────────────────────
NAVY  = "#0a2463"
RED   = "#8b1a00"
GREEN = "#1a7a1a"
DRED  = "#c0392b"
AMBER = "#e8a020"

COMM_COLORS = {
    "KC":  "#0a2463",
    "RC":  "#8b1a00",
    "CC":  "#e8a020",
    "LCC": "#4a7fb5",
    "SB":  "#1a7a1a",
    "CT":  "#7b2d8b",
    "LSU": "#c0553a",
}

COMM_NAMES = {
    "KC":  "KC — Arabica",
    "RC":  "RC — Robusta",
    "CC":  "CC — Cocoa (ICE)",
    "LCC": "LCC — Cocoa (Liffe)",
    "SB":  "SB — Sugar #11",
    "CT":  "CT — Cotton",
    "LSU": "LSU — White Sugar",
}

_D = dict(
    template="plotly_white",
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(0,0,0,0)",
    font=dict(family="-apple-system,Helvetica Neue,sans-serif", color="#1d1d1f", size=10)
)

MONTH_LABELS = ["Jan","Feb","Mar","Apr","May","Jun","Jul","Aug","Sep","Oct","Nov","Dec"]
ROW_H = 24


def lbl(t, color=NAVY):
    return (f"<div style='background:{color};padding:5px 13px;border-radius:5px;"
            f"margin:0 0 6px 0;text-align:center'><span style='font-size:.78rem;"
            f"font-weight:500;letter-spacing:.07em;text-transform:uppercase;"
            f"color:#dde4f0'>{t}</span></div>")


def kpi(label, val, sub=None, color=NAVY):
    sub_html = f"<span style='font-size:.7rem;color:#888;margin-left:6px'>{sub}</span>" if sub else ""
    return (f"<div style='display:inline-flex;flex-direction:column;background:#f0f2f8;"
            f"border-radius:8px;padding:7px 14px;margin:3px;min-width:110px'>"
            f"<span style='font-size:.58rem;color:#6e6e73;text-transform:uppercase;"
            f"letter-spacing:.1em'>{label}</span>"
            f"<span style='font-size:.95rem;font-weight:700;color:{color}'>{val}{sub_html}</span>"
            f"</div>")


@st.cache_data(ttl=600)
def get_data(comm: str) -> pd.DataFrame:
    return load_rollex(comm)


@st.cache_data(ttl=600)
def get_all_data() -> dict:
    out = {}
    for c in AVAILABLE:
        try:
            out[c] = load_rollex(c)
        except Exception:
            pass
    return out


# ── Sidebar — Filters ─────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown(
        "<div style='font-size:.9rem;font-weight:700;color:#0a2463;"
        "margin-bottom:12px;letter-spacing:.05em'>ROLLEX FILTERS</div>",
        unsafe_allow_html=True)

    sel_comm = st.selectbox(
        "Commodity",
        options=AVAILABLE,
        format_func=lambda x: COMM_NAMES[x],
        index=0)

    df_raw = get_data(sel_comm)
    min_d  = df_raw.index.min().date()
    max_d  = df_raw.index.max().date()

    date_range = st.slider(
        "Date range",
        min_value=min_d, max_value=max_d,
        value=(pd.Timestamp("2015-01-01").date(), max_d),
        format="YYYY-MM-DD")

    st.markdown("<hr>", unsafe_allow_html=True)
    st.caption(
        f"Roll offset: 20 trading days\n\n"
        f"Source: LSEG (interim)\n\n"
        f"Data as of {max_d.strftime('%d/%m/%Y')}")


# ── Filtered data ─────────────────────────────────────────────────────────────
df = df_raw.loc[str(date_range[0]):str(date_range[1])].copy()
df = df.dropna(subset=["rollex_px"])

# shared stats used across tabs
rets      = df["rollex_ret"].dropna()
latest_ret = df["rollex_ret"].iloc[-1]
z_now      = (latest_ret - rets.mean()) / rets.std() if rets.std() > 0 else 0

# Load all commodity data once (cached) before tabs render
all_data = get_all_data()

# ── Tabs ──────────────────────────────────────────────────────────────────────
tab_season, tab_corr, tab_idx, tab_pv, tab_dist = st.tabs(
    ["Seasonality", "Correlation", "Indexed Performance", "Price & Vol", "Return Distribution"])


# =============================================================================
# SHARED HEATMAP HELPER
# =============================================================================
RET_CS = [
    [0.0,  "#c0392b"],
    [0.45, "rgba(255,200,200,0.4)"],
    [0.5,  "#f8f8f8"],
    [0.55, "rgba(200,235,200,0.4)"],
    [1.0,  "#1a7a1a"],
]
VOL_CS = [
    [0.0,  "#f0f0f0"],
    [0.01, "#1a7a1a"],
    [0.5,  "#f8f8f8"],
    [1.0,  "#c0392b"],
]


def _simple_heatmap(year_pivot, fmt_val, fmt_stat, colorscale,
                    zmin, zmax, zmid, hover_suffix):
    piv = year_pivot.sort_index(ascending=True).copy()
    avg_row = piv.mean(skipna=True)
    std_row = piv.std(skipna=True)
    icv_row = avg_row / std_row

    y_labels = [str(y) for y in piv.index] + [""] + ["Avg", "Std", "ICV"]
    z_all, t_all = [], []

    for _, row in piv.iterrows():
        z_all.append([float(v) if pd.notna(v) else None for v in row])
        t_all.append([fmt_val(v) if pd.notna(v) else "" for v in row])

    z_all.append([None] * len(MONTH_LABELS))
    t_all.append([""] * len(MONTH_LABELS))

    for lbl_s, row in zip(["Avg", "Std", "ICV"], [avg_row, std_row, icv_row]):
        z_all.append([0.0] * len(MONTH_LABELS))
        if lbl_s == "ICV":
            t_all.append([f"<b>{fmt_stat(lbl_s, v)}</b>" if pd.notna(v) else "" for v in row])
        else:
            t_all.append([fmt_stat(lbl_s, v) if pd.notna(v) else "" for v in row])

    fig = go.Figure(go.Heatmap(
        z=z_all, x=MONTH_LABELS, y=y_labels,
        colorscale=colorscale,
        zmid=zmid, zmin=zmin, zmax=zmax,
        text=t_all, texttemplate="%{text}",
        textfont=dict(size=8.5),
        hovertemplate="%{y}  %{x}: <b>%{z:.2f}" + hover_suffix + "</b><extra></extra>",
        showscale=False, xgap=1, ygap=1))

    n_rows = len(y_labels)
    fig.update_layout(
        height=max(300, n_rows * ROW_H + 80),
        margin=dict(t=35, b=8, l=50, r=4),
        xaxis=dict(tickfont=dict(size=9), side="top", showgrid=False, showticklabels=False),
        yaxis=dict(tickfont=dict(size=9), showgrid=False, autorange=True),
        **_D)

    for month in MONTH_LABELS:
        fig.add_annotation(
            x=month, y=1.02, xref="x", yref="paper",
            text=f"<b>{month}</b>", showarrow=False,
            font=dict(size=9, color="#1d1d1f"), align="center")

    return fig


# =============================================================================
# TAB 1: Seasonality
# =============================================================================
with tab_season:
    # Monthly Returns Heatmap
    st.markdown(lbl(f"{COMM_NAMES[sel_comm]} — Monthly Returns Heatmap"),
                unsafe_allow_html=True)

    monthly_s = df["rollex_px"].resample("ME").last().pct_change() * 100
    monthly_s.index = monthly_s.index.to_period("M")
    m_df = monthly_s.reset_index()
    m_df.columns = ["Period", "Return"]
    m_df["Year"]  = m_df["Period"].dt.year
    m_df["Month"] = m_df["Period"].dt.month

    ret_pivot = m_df.pivot_table(index="Year", columns="Month", values="Return")
    ret_pivot.columns = [MONTH_LABELS[m - 1] for m in ret_pivot.columns]
    ret_pivot = ret_pivot.reindex(columns=MONTH_LABELS)
    ret_pivot = ret_pivot.dropna(how="all")
    abs_max   = ret_pivot.abs().max().max()

    fig_heat = _simple_heatmap(
        ret_pivot,
        fmt_val=lambda v: f"{v:.1f}%",
        fmt_stat=lambda lbl_s, v: f"{v:.1f}%" if lbl_s in ("Avg","Std") else f"{v:.1f}",
        colorscale=RET_CS, zmin=-abs_max, zmax=abs_max, zmid=0,
        hover_suffix="%")
    st.plotly_chart(fig_heat, use_container_width=True)

    st.markdown("<hr>", unsafe_allow_html=True)

    # Realized Volatility Heatmap
    st.markdown(lbl(f"{COMM_NAMES[sel_comm]} — Monthly Realized Volatility Heatmap"),
                unsafe_allow_html=True)

    vol_radio_col, _ = st.columns([1, 5])
    with vol_radio_col:
        vol_window = st.radio("Window", ["20d", "60d", "120d"], horizontal=True, key="vol_win")

    win_map = {"20d": 20, "60d": 60, "120d": 120}
    win     = win_map[vol_window]

    df["rv"]     = df["rollex_ret"].rolling(win).std() * np.sqrt(252) * 100
    rv_monthly   = df["rv"].resample("ME").last()
    rv_monthly.index = rv_monthly.index.to_period("M")
    rv_df         = rv_monthly.reset_index()
    rv_df.columns = ["Period", "RV"]
    rv_df["Year"]  = rv_df["Period"].dt.year
    rv_df["Month"] = rv_df["Period"].dt.month

    rv_pivot = rv_df.pivot_table(index="Year", columns="Month", values="RV")
    rv_pivot.columns = [MONTH_LABELS[m - 1] for m in rv_pivot.columns]
    rv_pivot = rv_pivot.reindex(columns=MONTH_LABELS)
    rv_pivot = rv_pivot.dropna(how="all")
    rv_max   = float(rv_pivot.max().max())
    rv_min   = float(rv_pivot.min().min())

    fig_rv = _simple_heatmap(
        rv_pivot,
        fmt_val=lambda v: f"{v:.1f}",
        fmt_stat=lambda lbl_s, v: f"{v:.1f}",
        colorscale=VOL_CS, zmin=rv_min, zmax=rv_max, zmid=None,
        hover_suffix="%")
    st.plotly_chart(fig_rv, use_container_width=True)


# =============================================================================
# TAB 2: Pairwise Correlation
# =============================================================================
with tab_corr:
    st.markdown(lbl("Pairwise Correlation"), unsafe_allow_html=True)

    corr_cols = st.columns([1, 1, 4])
    with corr_cols[0]:
        pair_a = st.selectbox("Commodity A", AVAILABLE,
                              index=0, format_func=lambda x: COMM_NAMES[x], key="pair_a")
    with corr_cols[1]:
        pair_b = st.selectbox("Commodity B", AVAILABLE,
                              index=1, format_func=lambda x: COMM_NAMES[x], key="pair_b")

    if pair_a != pair_b and pair_a in all_data and pair_b in all_data:
        da = all_data[pair_a].loc[str(date_range[0]):str(date_range[1])]
        db = all_data[pair_b].loc[str(date_range[0]):str(date_range[1])]

        ret_a = da["rollex_ret"].dropna() * 100
        ret_b = db["rollex_ret"].dropna() * 100
        common_idx = ret_a.index.intersection(ret_b.index)
        ret_a = ret_a.loc[common_idx]
        ret_b = ret_b.loc[common_idx]

        px_a = da["rollex_px"].dropna()
        px_b = db["rollex_px"].dropna()
        common_px = px_a.index.intersection(px_b.index)
        px_a = px_a.loc[common_px]
        px_b = px_b.loc[common_px]

        if len(common_idx) > 10:
            corr_ret  = ret_a.corr(ret_b)
            corr_px   = px_a.corr(px_b)
            r2_ret    = corr_ret ** 2
            r2_px     = corr_px ** 2
            m_ret, b_ret = np.polyfit(ret_a, ret_b, 1)
            x_line = np.linspace(ret_a.min(), ret_a.max(), 100)
            latest_common = common_idx[-1]
            latest_a_ret  = ret_a.loc[latest_common]
            latest_b_ret  = ret_b.loc[latest_common]

            cc1, cc2 = st.columns(2)

            with cc1:
                st.markdown(lbl(f"% Move Correlation  |  r = {corr_ret:.3f}  |  R² = {r2_ret:.3f}", NAVY),
                            unsafe_allow_html=True)
                fig_sc1 = go.Figure()
                fig_sc1.add_trace(go.Scatter(
                    x=ret_a, y=ret_b, mode="markers",
                    marker=dict(color=NAVY, size=4, opacity=0.35),
                    hovertemplate=f"{COMM_NAMES[pair_a]}: %{{x:.2f}}%<br>"
                                  f"{COMM_NAMES[pair_b]}: %{{y:.2f}}%<extra></extra>",
                    name="Daily returns"))
                fig_sc1.add_trace(go.Scatter(
                    x=x_line, y=m_ret * x_line + b_ret,
                    mode="lines", name=f"beta={m_ret:.2f}",
                    line=dict(color=DRED, width=1.8)))
                fig_sc1.add_trace(go.Scatter(
                    x=[latest_a_ret], y=[latest_b_ret], mode="markers+text",
                    marker=dict(color="#FFD700", size=14, symbol="star",
                                line=dict(color="#333", width=1)),
                    text=[f"Latest ({latest_common.strftime('%d/%m/%y')})"],
                    textposition="top right", textfont=dict(size=8, color="#FFD700"),
                    name="Latest"))
                fig_sc1.update_layout(height=560,
                    showlegend=False,
                    margin=dict(t=10, b=8, l=4, r=4),
                    xaxis=dict(showgrid=True, gridcolor="#f0f0f0",
                               tickfont=dict(size=9), title=f"{pair_a} ret %",
                               zeroline=True, zerolinecolor="#ddd"),
                    yaxis=dict(showgrid=True, gridcolor="#f0f0f0",
                               tickfont=dict(size=9), title=f"{pair_b} ret %",
                               zeroline=True, zerolinecolor="#ddd"),
                    **_D)
                st.plotly_chart(fig_sc1, use_container_width=True)

            with cc2:
                st.markdown(lbl(f"Flat Price Correlation  |  r = {corr_px:.3f}  |  R² = {r2_px:.3f}", RED),
                            unsafe_allow_html=True)
                m_px, b_px = np.polyfit(px_a, px_b, 1)
                x_px = np.linspace(px_a.min(), px_a.max(), 100)
                latest_common_px = common_px[-1]
                latest_a_px = px_a.loc[latest_common_px]
                latest_b_px = px_b.loc[latest_common_px]

                fig_sc2 = go.Figure()
                fig_sc2.add_trace(go.Scatter(
                    x=px_a, y=px_b, mode="markers",
                    marker=dict(color=RED, size=4, opacity=0.35),
                    hovertemplate=f"{pair_a}: %{{x:.2f}}<br>{pair_b}: %{{y:.2f}}<extra></extra>",
                    name="Flat prices"))
                fig_sc2.add_trace(go.Scatter(
                    x=x_px, y=m_px * x_px + b_px,
                    mode="lines", name=f"beta={m_px:.2f}",
                    line=dict(color=NAVY, width=1.8)))
                fig_sc2.add_trace(go.Scatter(
                    x=[latest_a_px], y=[latest_b_px], mode="markers+text",
                    marker=dict(color="#FFD700", size=14, symbol="star",
                                line=dict(color="#333", width=1)),
                    text=[f"Latest ({latest_common_px.strftime('%d/%m/%y')})"],
                    textposition="top right", textfont=dict(size=8, color="#FFD700"),
                    name="Latest"))
                fig_sc2.update_layout(height=560,
                    showlegend=False,
                    margin=dict(t=10, b=8, l=4, r=4),
                    xaxis=dict(showgrid=True, gridcolor="#f0f0f0",
                               tickfont=dict(size=9), title=f"{pair_a} price"),
                    yaxis=dict(showgrid=True, gridcolor="#f0f0f0",
                               tickfont=dict(size=9), title=f"{pair_b} price"),
                    **_D)
                st.plotly_chart(fig_sc2, use_container_width=True)
        else:
            st.info("Not enough common dates for the selected pair and date range.")
    else:
        st.info("Select two different commodities above to see the correlation.")

    # ── Full correlation matrix ───────────────────────────────────────────────
    st.markdown("<hr>", unsafe_allow_html=True)
    st.markdown(lbl("Return Correlation Matrix — All Commodities", NAVY),
                unsafe_allow_html=True)

    avail_comms = [c for c in AVAILABLE if c in all_data]
    ret_matrix  = pd.DataFrame({
        c: all_data[c].loc[str(date_range[0]):str(date_range[1])]["rollex_ret"]
        for c in avail_comms
    }).dropna()

    corr_matrix = ret_matrix.corr()
    labels      = [COMM_NAMES[c].split("—")[0].strip() for c in avail_comms]

    # Mask diagonal — set to None so cells render blank
    arr = corr_matrix.to_numpy(dtype=float, copy=True)
    np.fill_diagonal(arr, np.nan)
    z_mat = [[None if np.isnan(v) else v for v in row] for row in arr]
    t_mat = [["" if v is None else f"{v:.2f}" for v in row] for row in z_mat]

    CORR_CS = [
        [0.0,  "#c0392b"],
        [0.45, "rgba(255,200,200,0.4)"],
        [0.5,  "#f8f8f8"],
        [0.55, "rgba(200,235,200,0.4)"],
        [1.0,  "#0a2463"],
    ]

    fig_mat = go.Figure(go.Heatmap(
        z=z_mat, x=labels, y=labels,
        colorscale=CORR_CS, zmin=-1, zmax=1, zmid=0,
        text=t_mat, texttemplate="%{text}",
        textfont=dict(size=11),
        hovertemplate="%{y} / %{x}: <b>%{z:.3f}</b><extra></extra>",
        showscale=True,
        colorbar=dict(thickness=12, len=0.8, tickfont=dict(size=9)),
        xgap=2, ygap=2))

    fig_mat.update_layout(
        height=380,
        margin=dict(t=10, b=8, l=4, r=4),
        xaxis=dict(tickfont=dict(size=10), side="bottom", showgrid=False),
        yaxis=dict(tickfont=dict(size=10), showgrid=False, autorange="reversed"),
        **_D)
    st.plotly_chart(fig_mat, use_container_width=True)


# =============================================================================
# TAB 3: Indexed Performance
# =============================================================================
with tab_idx:
    st.markdown(lbl("Indexed Performance — Base 100 at Start of Period"),
                unsafe_allow_html=True)

    idx_c1, idx_c2 = st.columns([1, 5])
    with idx_c1:
        idx_comms = st.multiselect(
            "Select commodities:",
            options=AVAILABLE,
            default=AVAILABLE,
            format_func=lambda x: COMM_NAMES[x],
            key="idx_sel")

    fig_idx = go.Figure()
    for comm in idx_comms:
        if comm not in all_data:
            continue
        d = all_data[comm].loc[str(date_range[0]):str(date_range[1])]["rollex_px"].dropna()
        if len(d) == 0:
            continue
        indexed = d / d.iloc[0] * 100
        fig_idx.add_trace(go.Scatter(
            x=indexed.index, y=indexed,
            name=COMM_NAMES[comm], mode="lines",
            line=dict(color=COMM_COLORS.get(comm, "#aaa"), width=1.8),
            hovertemplate=f"{COMM_NAMES[comm]}<br>%{{x|%d/%m/%Y}}<br><b>%{{y:.1f}}</b><extra></extra>"))

    fig_idx.add_hline(y=100, line_color="#cccccc", line_width=1)
    fig_idx.update_layout(height=460,
        legend=dict(orientation="h", y=1.02, x=0, font=dict(size=8)),
        margin=dict(t=10, b=8, l=4, r=4),
        xaxis=dict(showgrid=False, tickfont=dict(size=9)),
        yaxis=dict(showgrid=True, gridcolor="#f0f0f0", tickfont=dict(size=9),
                   title="Indexed (Base=100)"),
        **_D)
    st.plotly_chart(fig_idx, use_container_width=True)


# =============================================================================
# TAB 4: Price & Vol
# =============================================================================
with tab_pv:
    st.markdown(lbl(f"{COMM_NAMES[sel_comm]} — Rollex Price & Rolling Volatility"),
                unsafe_allow_html=True)

    df["vol20"] = df["rollex_ret"].rolling(20).std() * np.sqrt(252) * 100
    df["vol60"] = df["rollex_ret"].rolling(60).std() * np.sqrt(252) * 100

    # Build active_label customdata — show contract on every price point
    active_labels = df["active_label"].values if "active_label" in df.columns else [""] * len(df)

    fig_px = make_subplots(specs=[[{"secondary_y": True}]])
    fig_px.add_trace(go.Scatter(
        x=df.index, y=df["rollex_px"],
        name="Rollex Px", mode="lines",
        line=dict(color=COMM_COLORS.get(sel_comm, NAVY), width=2),
        fill="tozeroy", fillcolor="rgba(10,36,99,0.07)",
        customdata=active_labels,
        hovertemplate=(
            "%{x|%d/%m/%Y}<br>"
            "<b>%{y:.2f}</b><br>"
            "Contract: %{customdata}<extra></extra>")),
        secondary_y=False)
    fig_px.add_trace(go.Scatter(
        x=df.index, y=df["vol20"],
        name="Vol 20d", mode="lines",
        line=dict(color=AMBER, width=1.5, dash="solid"),
        hovertemplate="%{x|%d/%m/%Y}<br>20d Vol: <b>%{y:.1f}%</b><extra></extra>"),
        secondary_y=True)
    fig_px.add_trace(go.Scatter(
        x=df.index, y=df["vol60"],
        name="Vol 60d", mode="lines",
        line=dict(color="#888", width=1.2, dash="dot"),
        hovertemplate="%{x|%d/%m/%Y}<br>60d Vol: <b>%{y:.1f}%</b><extra></extra>"),
        secondary_y=True)

    fig_px.update_layout(height=420,
        legend=dict(orientation="h", y=1.02, x=0, font=dict(size=8)),
        margin=dict(t=10, b=8, l=4, r=4), **_D)
    fig_px.update_yaxes(title_text="Price", secondary_y=False,
        showgrid=True, gridcolor="#f0f0f0", tickfont=dict(size=9))
    fig_px.update_yaxes(title_text="Ann. Vol %", secondary_y=True,
        showgrid=False, tickfont=dict(size=9))
    fig_px.update_xaxes(showgrid=False, tickfont=dict(size=9))
    st.plotly_chart(fig_px, use_container_width=True)


# =============================================================================
# TAB 5: Return Distribution
# =============================================================================
with tab_dist:
    st.markdown(lbl(f"{COMM_NAMES[sel_comm]} — Log Return Distribution & Z-Score"),
                unsafe_allow_html=True)

    log_rets  = np.log1p(df["rollex_ret"]).dropna() * 100   # log returns in %
    mu        = log_rets.mean()
    sigma     = log_rets.std()
    latest_lr = np.log1p(latest_ret) * 100
    z_lr      = (latest_lr - mu) / sigma if sigma > 0 else 0

    # stats sidebar metrics
    skew_v  = float(stats.skew(log_rets))
    kurt_v  = float(stats.kurtosis(log_rets))         # excess kurtosis
    _, p_jb = stats.jarque_bera(log_rets)

    m1, m2, m3, m4, m5 = st.columns(5)
    def _stat(col, label, val, color=NAVY):
        col.markdown(
            f"<div style='background:#f0f2f8;border-radius:8px;padding:8px 14px'>"
            f"<div style='font-size:.58rem;color:#6e6e73;text-transform:uppercase;"
            f"letter-spacing:.1em'>{label}</div>"
            f"<div style='font-size:.95rem;font-weight:700;color:{color}'>{val}</div>"
            f"</div>", unsafe_allow_html=True)

    z_col = DRED if abs(z_lr) > 2 else AMBER if abs(z_lr) > 1 else GREEN
    _stat(m1, "Latest Return", f"{latest_lr:+.2f}%")
    _stat(m2, "Z-Score",       f"{z_lr:+.2f}σ", color=z_col)
    _stat(m3, "Skewness",      f"{skew_v:+.3f}")
    _stat(m4, "Excess Kurtosis", f"{kurt_v:+.3f}")
    _stat(m5, "Jarque-Bera p", f"{p_jb:.4f}", color=DRED if p_jb < 0.05 else GREEN)

    st.markdown("<div style='margin:10px 0'></div>", unsafe_allow_html=True)

    # histogram + normal fit
    x_range   = np.linspace(log_rets.min(), log_rets.max(), 400)
    bin_width = (log_rets.max() - log_rets.min()) / 80
    y_curve   = stats.norm.pdf(x_range, mu, sigma) * len(log_rets) * bin_width

    fig_hist = go.Figure()
    fig_hist.add_trace(go.Histogram(
        x=log_rets, nbinsx=80,
        marker_color=COMM_COLORS.get(sel_comm, NAVY),
        opacity=0.72, name="Log returns",
        hovertemplate="Return: %{x:.2f}%<br>Count: %{y}<extra></extra>"))
    fig_hist.add_trace(go.Scatter(
        x=x_range, y=y_curve, mode="lines",
        name="Normal fit",
        line=dict(color="#888", width=1.8, dash="dot")))

    # SD bands
    sd_colors = {1: "#f0a500", 2: "#e07000", 3: DRED}
    for n_sd, color in sd_colors.items():
        for sign in [1, -1]:
            val = mu + sign * n_sd * sigma
            fig_hist.add_vline(x=val, line_color=color, line_width=1.2, line_dash="dash",
                annotation_text=f"{sign*n_sd:+d}σ",
                annotation_position="top right" if sign > 0 else "top left",
                annotation_font=dict(size=8, color=color))

    # latest return line
    fig_hist.add_vline(x=latest_lr, line_color=z_col, line_width=2.5,
        annotation_text=f"Today  {latest_lr:+.2f}%  (z={z_lr:+.2f}σ)",
        annotation_position="top right",
        annotation_font=dict(size=9, color=z_col, family="monospace"))

    fig_hist.update_layout(height=420,
        showlegend=False,
        margin=dict(t=30, b=8, l=4, r=4),
        xaxis=dict(showgrid=False, tickfont=dict(size=9), title="Log Return %"),
        yaxis=dict(showgrid=True, gridcolor="#f0f0f0", tickfont=dict(size=9), title="Count"),
        **_D)
    st.plotly_chart(fig_hist, use_container_width=True)
