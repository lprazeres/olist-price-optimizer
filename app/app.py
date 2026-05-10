"""Olist Price Optimizer — Streamlit Application.

Portfolio project: price elasticity of demand and revenue optimisation
for Olist Brazilian e-commerce (2016–2018, ~100K transactions).
"""

from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

st.set_page_config(page_title="Olist Price Optimizer", layout="wide")

DATA_DIR = Path(__file__).parent.parent / "data"


# ── Data loaders ───────────────────────────────────────────────────────────────

@st.cache_data
def load_clusters() -> pd.DataFrame:
    return pd.read_parquet(DATA_DIR / "category_clusters.parquet")


@st.cache_data
def load_monthly() -> pd.DataFrame:
    return pd.read_parquet(DATA_DIR / "category_month_agg.parquet")


# ── Helpers ────────────────────────────────────────────────────────────────────

def fmt_cat(name: str) -> str:
    return name.replace("_", " ").title()


def build_revenue_curve(
    elasticity: float,
    base_price: float,
    base_sales: float,
    n_points: int = 200,
) -> tuple:
    prices = np.linspace(base_price * 0.5, base_price * 2.0, n_points)
    estimated_sales = base_sales * (prices / base_price) ** elasticity
    revenue = prices * estimated_sales
    return prices, revenue


def compute_uplift(row: pd.Series) -> pd.Series:
    base_price = row["avg_price"]
    base_sales = row["total_sales"] / row["n_obs"]
    prices, revenue = build_revenue_curve(row["price_elasticity"], base_price, base_sales)
    opt_idx = int(np.argmax(revenue))
    current_rev = base_price * base_sales
    optimal_rev = revenue[opt_idx]
    uplift_brl = optimal_rev - current_rev
    return pd.Series(
        {
            "base_price": base_price,
            "base_sales": base_sales,
            "optimal_price": prices[opt_idx],
            "current_rev": current_rev,
            "optimal_rev": optimal_rev,
            "uplift_brl": uplift_brl,
            "uplift_pct": uplift_brl / current_rev * 100,
        }
    )


def business_interpretation(elasticity: float, optimal_price: float, base_price: float) -> str:
    direction = "increase" if optimal_price > base_price else "decrease"
    pct_change = abs((optimal_price - base_price) / base_price * 100)
    if elasticity < -1:
        return (
            f"**Elastic demand** (β = {elasticity:.2f}). "
            f"A 1% price increase reduces sales by {abs(elasticity):.1f}%. "
            f"The model suggests a **{pct_change:.0f}% price {direction}** to "
            f"R$ {optimal_price:.2f} to maximise revenue."
        )
    elif elasticity < 0:
        return (
            f"**Inelastic demand** (β = {elasticity:.2f}). "
            f"Sales are relatively insensitive to price changes. "
            f"The model suggests a **{pct_change:.0f}% price {direction}** to "
            f"R$ {optimal_price:.2f} to maximise revenue."
        )
    else:
        return (
            f"**Positive elasticity** (β = {elasticity:.2f}). "
            f"Higher price correlates with more sales — possibly a Veblen or premium effect. "
            f"The model suggests a **{pct_change:.0f}% price {direction}** to "
            f"R$ {optimal_price:.2f} to maximise revenue."
        )


def price_sensitivity_label(row: pd.Series) -> str:
    if row["p_value"] >= 0.05:
        return "Low 🟢"
    elif row["price_elasticity"] < -1:
        return "High 🔴"
    else:
        return "Moderate 🟡"


# ── Load & prepare data ────────────────────────────────────────────────────────

df_clusters = load_clusters()
df_monthly = load_monthly()

df_sig = df_clusters[df_clusters["significant"]].copy()
uplifts = df_sig.apply(compute_uplift, axis=1)
df_sig = pd.concat([df_sig, uplifts], axis=1)

df_elastic = df_sig[df_sig["price_elasticity"] < -1].sort_values("price_elasticity")
df_resilient = df_sig[df_sig["price_elasticity"] >= -1].sort_values("price_elasticity")

total_categories = len(df_clusters)
sig_count = len(df_sig)
total_uplift = df_sig["uplift_brl"].sum()
elastic_uplift_avg_pct = df_elastic["uplift_pct"].mean()

CLUSTER_ICONS = {
    "Heavy-Freight Specialists": "🏋️",
    "Volume Leaders": "⭐",
    "Quality Niche": "💎",
    "Conversion Risk": "⚠️",
}

CLUSTER_ORDER = [
    "Heavy-Freight Specialists",
    "Volume Leaders",
    "Quality Niche",
    "Conversion Risk",
]

CLUSTER_DESCRIPTIONS = {
    "Heavy-Freight Specialists": (
        "These categories are dominated by large or heavy products where shipping cost is "
        "the #1 purchase barrier. Price adjustments alone won't move the needle — the bigger "
        "opportunity is in freight negotiation and logistics optimisation."
    ),
    "Volume Leaders": (
        "The backbone of the Olist platform. High sales volume with mixed price sensitivity — "
        "some categories respond strongly to pricing changes, others are driven by brand and "
        "product quality. Prioritise the price-sensitive ones for repricing initiatives."
    ),
    "Quality Niche": (
        "Lower volume but high customer satisfaction. These categories attract quality-conscious "
        "buyers who are less driven by price. The growth lever here is discoverability — better "
        "listings, more photos, and improved search positioning."
    ),
    "Conversion Risk": (
        "Categories with below-average review scores and lower sales volumes. Fixing operational "
        "issues (delivery speed, product accuracy, customer service) should come before any "
        "pricing strategy."
    ),
}

CLUSTER_COLORS = {
    "Heavy-Freight Specialists": "#F59E0B",
    "Volume Leaders": "#6366F1",
    "Quality Niche": "#22C55E",
    "Conversion Risk": "#EF4444",
}

TABLE_COLS = [
    "category", "price_elasticity", "avg_price", "optimal_price",
    "current_rev", "optimal_rev", "uplift_brl", "uplift_pct",
]
TABLE_RENAME = {
    "category": "Category",
    "price_elasticity": "Elasticity (β)",
    "avg_price": "Current Price (R$)",
    "optimal_price": "Optimal Price (R$)",
    "current_rev": "Current Revenue (R$)",
    "optimal_rev": "Optimal Revenue (R$)",
    "uplift_brl": "Uplift (R$)",
    "uplift_pct": "Uplift (%)",
}
TABLE_COL_CONFIG = {
    "Elasticity (β)": st.column_config.NumberColumn(format="%.2f"),
    "Current Price (R$)": st.column_config.NumberColumn(format="R$ %.2f"),
    "Optimal Price (R$)": st.column_config.NumberColumn(format="R$ %.2f"),
    "Current Revenue (R$)": st.column_config.NumberColumn(format="R$ %.0f"),
    "Optimal Revenue (R$)": st.column_config.NumberColumn(format="R$ %.0f"),
    "Uplift (R$)": st.column_config.NumberColumn(format="R$ %.0f"),
    "Uplift (%)": st.column_config.NumberColumn(format="%.1f%%"),
}


def make_table(df: pd.DataFrame) -> pd.DataFrame:
    out = df[TABLE_COLS].copy()
    out["category"] = out["category"].apply(fmt_cat)
    return out.rename(columns=TABLE_RENAME)


def make_table_with_total(df: pd.DataFrame) -> pd.DataFrame:
    out = make_table(df)
    total_current = df["current_rev"].sum()
    total_optimal = df["optimal_rev"].sum()
    total_uplift_brl = df["uplift_brl"].sum()
    total_uplift_pct = total_uplift_brl / total_current * 100
    total_row = pd.DataFrame([{
        "Category": "TOTAL",
        "Elasticity (β)": float("nan"),
        "Current Price (R$)": float("nan"),
        "Optimal Price (R$)": float("nan"),
        "Current Revenue (R$)": total_current,
        "Optimal Revenue (R$)": total_optimal,
        "Uplift (R$)": total_uplift_brl,
        "Uplift (%)": total_uplift_pct,
    }])
    return pd.concat([out, total_row], ignore_index=True)


# ── Sidebar — always static ────────────────────────────────────────────────────

with st.sidebar:
    st.title("Olist Price Optimizer")
    st.markdown("**Brazilian e-commerce · 2016–2018**")
    st.divider()

    st.metric("Categories Analyzed", total_categories)
    st.metric("Significant Results", f"{sig_count} / {total_categories}")
    st.metric("Combined Monthly Uplift", f"R$ {total_uplift:,.0f}")

    st.divider()
    st.markdown("**Tech Stack**")
    st.markdown("Python · Statsmodels · Scikit-learn  \nStreamlit · Plotly · Pandas")
    st.divider()
    st.caption("Portfolio project by Lucas Prazeres")


# ── Tabs ────────────────────────────────────────────────────────────────────────

tab_about, tab_impact, tab_optimizer, tab_explorer = st.tabs(
    ["About", "Revenue Impact", "Revenue Optimizer", "Category Explorer"]
)


# ── TAB 1 — ABOUT ─────────────────────────────────────────────────────────────

with tab_about:
    col_left, col_right = st.columns([0.60, 0.40])

    with col_left:
        st.header("Olist Price Optimizer")
        st.markdown(
            "A price elasticity analysis of **100K+ transactions** from the Brazilian "
            "Olist marketplace (2016–2018), combining econometric modelling, unsupervised "
            "clustering, and an interactive pricing simulator."
        )

        st.subheader("Business Context")
        st.markdown(
            "Olist connects small-to-medium sellers to major Brazilian e-commerce channels. "
            "Sellers often price by intuition rather than data, missing revenue opportunities. "
            "This project estimates **how sensitive each product category's demand is to price changes**, "
            "enabling data-driven pricing decisions."
        )

        st.subheader("Dataset")
        st.markdown(
            "- **Source:** Olist Brazilian E-Commerce (Kaggle)\n"
            "- ~100,000 orders · 71 product categories · 2016–2018\n"
            "- Features engineered: shipping distance (haversine), delivery delay, "
            "monthly aggregates\n"
            "- Final panel: **62 categories × up to 24 months** after quality filters"
        )

        st.subheader("What We Found")
        st.markdown(
            "Out of 62 product categories analysed, **13 showed a statistically reliable "
            "relationship between price and sales volume**. Among these, 6 categories "
            "demonstrated **high price sensitivity** — meaning that a price reduction leads "
            "to a proportionally larger gain in sales volume, creating a net revenue uplift.\n\n"
            f"For these 6 categories, moving to the model's recommended price point could "
            f"generate an average **+{elastic_uplift_avg_pct:.0f}% increase in monthly revenue** "
            "without any additional marketing spend or operational change.\n\n"
            "The remaining significant categories are either resilient to price changes or "
            "show premium dynamics — where higher prices correlate with higher demand. "
            "These require a different strategic lens: premium positioning, improved product "
            "listings, or operational fixes before pricing becomes a meaningful lever."
        )

        st.subheader("Strategic Clusters")
        st.markdown(
            "All 62 categories were grouped into 4 strategic segments based on their "
            "pricing dynamics, sales volume, review scores, and logistics profile:\n\n"
            "**🏋️ Heavy-Freight Specialists** — Large or heavy products where shipping cost "
            "is the primary purchase barrier. Freight negotiation yields more than price cuts.\n\n"
            "**⭐ Volume Leaders** — The platform backbone. High-volume categories with mixed "
            "sensitivity; the price-sensitive ones within this group are the highest-priority "
            "repricing targets.\n\n"
            "**💎 Quality Niche** — High satisfaction, lower volume. Buyers here are quality-"
            "driven. The growth opportunity is discoverability, not price.\n\n"
            "**⚠️ Conversion Risk** — Below-average reviews and lower volumes. Operational "
            "improvement must come before any pricing strategy."
        )

    with col_right:
        st.subheader("Key Findings")

        st.metric(
            label="Significant Categories",
            value=f"{sig_count} of {total_categories}",
            help="Categories where price has a statistically reliable impact on sales (p < 0.05)",
        )
        st.metric(
            label="Avg Revenue Uplift — Elastic Group",
            value=f"+{elastic_uplift_avg_pct:.0f}%",
            help="Average monthly revenue increase for the 6 elastic categories at their optimal price point",
        )
        st.metric(
            label="Combined Monthly Revenue Uplift",
            value=f"R$ {total_uplift:,.0f}",
            help="Aggregate monthly gain if all significant categories move to their optimal price",
        )

        st.divider()
        st.subheader("Tech Stack")
        st.markdown(
            "| Layer | Tools |\n"
            "|---|---|\n"
            "| Data wrangling | Pandas, PyArrow |\n"
            "| Econometrics | Statsmodels OLS |\n"
            "| Clustering | Scikit-learn KMeans |\n"
            "| Visualisation | Plotly |\n"
            "| App | Streamlit |"
        )

        st.divider()
        st.subheader("Author")
        st.markdown(
            "**Lucas Prazeres**  \n"
            "Data & Growth Analyst  \n"
            "[🔗 LinkedIn](https://linkedin.com/in/lucas-silveira-prazeres) · "
            "[💻 GitHub](https://github.com/lprazeres)"
        )


# ── TAB 2 — REVENUE IMPACT ────────────────────────────────────────────────────

with tab_impact:
    st.header("Pricing Opportunity Summary")
    st.markdown(
        "Revenue uplift estimated for the **13 statistically significant categories** "
        "(OLS log-log, p < 0.05). Each category's optimal price is the point on the "
        "simulated demand curve that maximises `price × estimated_sales`."
    )

    # ── Group A: Elastic ──────────────────────────────────────────────────────
    st.subheader("Group A — Elastic Demand (β < −1)")
    st.markdown(
        "Demand is highly price-sensitive. A **price reduction** is typically optimal: "
        "volume gains outweigh the margin drop."
    )

    st.dataframe(
        make_table_with_total(df_elastic),
        hide_index=True,
        use_container_width=True,
        column_config=TABLE_COL_CONFIG,
    )

    elastic_uplift = df_elastic["uplift_brl"].sum()
    st.success(
        f"**Group A opportunity:** R$ {elastic_uplift:,.0f} additional monthly revenue "
        f"across {len(df_elastic)} elastic categories."
    )

    # ── Group B: Resilient ────────────────────────────────────────────────────
    st.subheader("Group B — Resilient / Positive Elasticity (β ≥ −1)")
    st.markdown(
        "Demand is relatively insensitive to price — or even increases with price. "
        "**Price increases** or premium positioning may be viable."
    )

    st.dataframe(
        make_table_with_total(df_resilient),
        hide_index=True,
        use_container_width=True,
        column_config=TABLE_COL_CONFIG,
    )

    resilient_uplift = df_resilient["uplift_brl"].sum()
    st.success(
        f"**Group B opportunity:** R$ {resilient_uplift:,.0f} additional monthly revenue "
        f"across {len(df_resilient)} resilient categories."
    )

    st.info(
        "**Caution:** These estimates assume the log-log elasticity holds linearly "
        "across the simulated price range (50%–200% of current price). Real demand curves "
        "may flatten or shift at extreme prices. Treat these as directional signals, "
        "not precise forecasts."
    )

    # ── Horizontal bar chart ──────────────────────────────────────────────────
    st.subheader("Monthly Revenue Uplift by Category")

    chart_df = pd.concat(
        [df_elastic.assign(group="Elastic"), df_resilient.assign(group="Resilient")]
    ).sort_values("uplift_brl", ascending=True)
    chart_df["category_fmt"] = chart_df["category"].apply(fmt_cat)
    bar_colors = chart_df["group"].map({"Elastic": "#EF4444", "Resilient": "#22C55E"})

    fig_bar = go.Figure(
        go.Bar(
            x=chart_df["uplift_brl"],
            y=chart_df["category_fmt"],
            orientation="h",
            marker_color=bar_colors,
            text=chart_df["uplift_brl"].apply(lambda v: f"R$ {v:,.0f}"),
            textposition="outside",
            hovertemplate="<b>%{y}</b><br>Uplift: R$ %{x:,.0f}<extra></extra>",
        )
    )
    fig_bar.update_layout(
        xaxis_title="Monthly Revenue Uplift (R$)",
        yaxis_title="",
        height=520,
        margin=dict(l=20, r=100, t=20, b=40),
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
    )
    st.plotly_chart(fig_bar, use_container_width=True)
    st.caption("Red = elastic demand · Green = resilient / positive elasticity")


# ── TAB 3 — REVENUE OPTIMIZER ─────────────────────────────────────────────────

with tab_optimizer:
    st.header("Revenue Optimizer")
    st.markdown(
        "Simulate the effect of price changes on demand and revenue for a selected category."
    )

    sig_options = sorted(df_sig["category"].tolist())
    selected_cat = st.selectbox(
        "Select a category",
        options=sig_options,
        format_func=fmt_cat,
    )
    st.caption(
        f"Showing **{len(sig_options)} of {total_categories}** product categories — "
        "only those with statistically significant price elasticity (p < 0.05) are "
        "available. Categories without a reliable elasticity estimate are excluded."
    )

    row = df_sig[df_sig["category"] == selected_cat].iloc[0]
    elasticity = float(row["price_elasticity"])
    base_price = float(row["avg_price"])
    base_sales = float(row["total_sales"] / row["n_obs"])

    prices, revenue = build_revenue_curve(elasticity, base_price, base_sales)
    opt_idx = int(np.argmax(revenue))
    optimal_price = float(prices[opt_idx])
    current_rev = base_price * base_sales
    optimal_rev = float(revenue[opt_idx])
    uplift_brl = optimal_rev - current_rev
    uplift_pct = uplift_brl / current_rev * 100

    cluster_name = str(row["cluster_name"])
    icon = CLUSTER_ICONS.get(cluster_name, "📊")
    n_obs = int(row["n_obs"])
    st.caption(
        f"{icon} {cluster_name}  ·  OLS log-log model  ·  "
        f"{n_obs} months of data  ·  Statistically significant (p < 0.05)"
    )

    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Price Elasticity (β)", f"{elasticity:.2f}")
    m2.metric("Current Avg Price", f"R$ {base_price:.2f}")
    m3.metric("Optimal Price", f"R$ {optimal_price:.2f}")
    m4.metric(
        "Revenue Uplift",
        f"R$ {uplift_brl:,.0f} / mo",
        delta=f"{uplift_pct:+.1f}%",
        delta_color="normal",
    )

    fig_curve = go.Figure()
    fig_curve.add_trace(
        go.Scatter(
            x=prices,
            y=revenue,
            mode="lines",
            name="Revenue",
            line=dict(color="#6366F1", width=2),
            hovertemplate="Price: R$ %{x:.2f}<br>Revenue: R$ %{y:,.0f}<extra></extra>",
        )
    )
    fig_curve.add_vline(
        x=base_price,
        line_dash="dot",
        line_color="gray",
        annotation_text="Current",
        annotation_position="top",
    )
    fig_curve.add_vline(
        x=optimal_price,
        line_dash="dash",
        line_color="#22C55E",
        annotation_text="Optimal",
        annotation_position="top",
    )
    fig_curve.update_layout(
        xaxis_title="Price (R$)",
        yaxis_title="Estimated Monthly Revenue (R$)",
        height=380,
        margin=dict(l=20, r=20, t=40, b=40),
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
    )
    st.plotly_chart(fig_curve, use_container_width=True)

    st.markdown(business_interpretation(elasticity, optimal_price, base_price))

    with st.expander("📅 View Historical Price & Sales Trend", expanded=False):
        cat_monthly = df_monthly[df_monthly["product_category_name"] == selected_cat].copy()

        if len(cat_monthly) > 0:
            fig_trend = go.Figure()
            fig_trend.add_trace(
                go.Scatter(
                    x=cat_monthly["year_month"].astype(str),
                    y=cat_monthly["avg_price"],
                    mode="lines+markers",
                    name="Avg Price (R$)",
                    line=dict(color="#F59E0B"),
                    yaxis="y1",
                )
            )
            fig_trend.add_trace(
                go.Bar(
                    x=cat_monthly["year_month"].astype(str),
                    y=cat_monthly["sales"],
                    name="Monthly Sales",
                    marker_color="rgba(99,102,241,0.3)",
                    yaxis="y2",
                )
            )
            fig_trend.update_layout(
                yaxis=dict(title="Avg Price (R$)", side="left"),
                yaxis2=dict(title="Monthly Sales", side="right", overlaying="y"),
                xaxis_title="Month",
                height=350,
                margin=dict(l=20, r=20, t=20, b=40),
                plot_bgcolor="rgba(0,0,0,0)",
                paper_bgcolor="rgba(0,0,0,0)",
                legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
            )
            st.plotly_chart(fig_trend, use_container_width=True)
        else:
            st.info("No monthly data available for this category.")


# ── TAB 4 — CATEGORY EXPLORER ─────────────────────────────────────────────────

with tab_explorer:
    st.header("Category Explorer")
    st.markdown("All 62 categories organised by strategic cluster")

    # Prepare explorer data
    df_explorer = df_clusters.copy()
    df_explorer["monthly_sales"] = df_explorer["total_sales"] / df_explorer["n_obs"]
    df_explorer["category_fmt"] = df_explorer["category"].apply(fmt_cat)
    df_explorer["price_sensitivity"] = df_explorer.apply(price_sensitivity_label, axis=1)

    EXPLORER_COL_CONFIG = {
        "Avg Price (R$)": st.column_config.NumberColumn(format="R$ %.2f"),
        "Monthly Sales": st.column_config.NumberColumn(format="%.0f"),
    }

    for cluster in CLUSTER_ORDER:
        icon = CLUSTER_ICONS[cluster]
        description = CLUSTER_DESCRIPTIONS[cluster]
        cluster_df = (
            df_explorer[df_explorer["cluster_name"] == cluster]
            .sort_values("monthly_sales", ascending=False)
        )

        st.markdown(f"### {icon} {cluster}")
        st.markdown(description)

        display_df = (
            cluster_df[["category_fmt", "avg_price", "monthly_sales", "avg_review", "price_sensitivity"]]
            .copy()
            .rename(columns={
                "category_fmt": "Category",
                "avg_price": "Avg Price (R$)",
                "monthly_sales": "Monthly Sales",
                "avg_review": "Review Score",
                "price_sensitivity": "Price Sensitivity",
            })
        )
        display_df["Review Score"] = display_df["Review Score"].apply(lambda x: f"{x:.1f} ⭐")

        st.dataframe(
            display_df,
            hide_index=True,
            use_container_width=True,
            column_config=EXPLORER_COL_CONFIG,
        )
        st.markdown("")

    # ── Scatter plot ──────────────────────────────────────────────────────────
    st.subheader("Category Map — Price vs. Volume by Cluster")

    avg_monthly_sales = df_explorer["monthly_sales"].mean()

    fig_scatter = go.Figure()

    for cluster in CLUSTER_ORDER:
        cdf = df_explorer[df_explorer["cluster_name"] == cluster]
        cluster_icon = CLUSTER_ICONS[cluster]
        fig_scatter.add_trace(
            go.Scatter(
                x=cdf["avg_price"],
                y=cdf["monthly_sales"],
                mode="markers",
                name=f"{cluster_icon} {cluster}",
                marker=dict(
                    size=12,
                    color=CLUSTER_COLORS[cluster],
                    opacity=0.8,
                    line=dict(width=1, color="white"),
                ),
                customdata=np.column_stack([
                    cdf["category_fmt"],
                    cdf["cluster_name"],
                    cdf["avg_review"].round(1).astype(str),
                    cdf["price_sensitivity"],
                ]),
                hovertemplate=(
                    "<b>%{customdata[0]}</b><br>"
                    "Cluster: %{customdata[1]}<br>"
                    "Avg Price: R$ %{x:.2f}<br>"
                    "Monthly Sales: %{y:.0f}<br>"
                    "Review Score: %{customdata[2]} ⭐<br>"
                    "Price Sensitivity: %{customdata[3]}<extra></extra>"
                ),
            )
        )

    fig_scatter.add_hline(
        y=avg_monthly_sales,
        line_dash="dot",
        line_color="gray",
        annotation_text=f"Avg monthly sales ({avg_monthly_sales:.0f})",
        annotation_position="right",
    )
    fig_scatter.update_layout(
        xaxis_title="Avg Price (R$)",
        yaxis_title="Monthly Sales (units)",
        height=500,
        margin=dict(l=20, r=160, t=40, b=40),
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
        legend=dict(orientation="v", x=1.02, y=1),
    )
    st.plotly_chart(fig_scatter, use_container_width=True)
