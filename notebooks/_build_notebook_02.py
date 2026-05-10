"""Generates 02_elasticity_model.ipynb programmatically."""
import json
from pathlib import Path

NB_PATH = Path(__file__).parent / "02_elasticity_model.ipynb"
cells = []


def md(src: str) -> dict:
    return {
        "cell_type": "markdown",
        "id": f"md{len(cells):02d}",
        "metadata": {},
        "source": src,
    }


def code(src: str) -> dict:
    return {
        "cell_type": "code",
        "id": f"co{len(cells):02d}",
        "metadata": {},
        "source": src,
        "outputs": [],
        "execution_count": None,
    }


# ── 0. Title ──────────────────────────────────────────────────────────────────
cells.append(md(
    "# Olist Price Optimizer — Step 2: Price Elasticity Modelling\n"
    "\n"
    "**Goal:** Estimate how sensitive demand (sales volume) is to price changes —\n"
    "both globally (baseline) and per product category — then identify the price\n"
    "point that maximises monthly revenue for each category.\n"
    "\n"
    "**Input:** `data/category_month_agg.parquet` — 1,241 rows, 70 categories, 23 months  \n"
    "**Outputs:**\n"
    "* `data/elasticity_by_category.parquet / .csv` — per-category elasticity estimates\n"
    "* Revenue curve analysis for the top 5 highest-volume categories\n"
    "\n"
    "---\n"
    "### Analytical Framework\n"
    "\n"
    "We use a **log-log Ordinary Least Squares (OLS)** regression:\n"
    "\n"
    "```\n"
    "log(sales) = α + β·log(price) + γ·log(freight) + δ·avg_review\n"
    "           + ζ·late_delivery_rate + Σ month_dummies + ε\n"
    "```\n"
    "\n"
    "The coefficient **β** is the **price elasticity of demand**: a 1% increase in\n"
    "price leads to a β% change in monthly sales volume. Key thresholds:\n"
    "\n"
    "| β range | Demand type | Revenue implication |\n"
    "|---------|-------------|---------------------|\n"
    "| β < −1 | Elastic | Raising price **reduces** revenue |\n"
    "| β = −1 | Unit-elastic | Price changes leave revenue unchanged |\n"
    "| −1 < β < 0 | Inelastic | Raising price **increases** revenue |\n"
    "| β > 0 | Veblen/quality signal | Unusual; may indicate omitted variables |"
))

# ── 1. Imports ────────────────────────────────────────────────────────────────
cells.append(code(
    "import warnings\n"
    'warnings.filterwarnings("ignore")\n'
    "\n"
    "import pandas as pd\n"
    "import numpy as np\n"
    "import statsmodels.api as sm\n"
    "import statsmodels.formula.api as smf\n"
    "from pathlib import Path\n"
    "\n"
    'DATA_DIR = Path("../data")\n'
    'print("Libraries loaded.")'
))

# ── 2. Section 1 header ───────────────────────────────────────────────────────
cells.append(md(
    "---\n"
    "## Section 1 — Setup & Load\n"
    "\n"
    "### Why filter to 2017-01?\n"
    "\n"
    "The Olist marketplace launched in late 2016. Those early months have very few\n"
    "active categories (1–30 per month vs. 60–70 from 2017 onward), and prices and\n"
    "volumes reflect early-adopter behaviour rather than a stable market.\n"
    "Including them would add noise and bias the elasticity estimate.\n"
    "We keep **2017-01 onward** — the point where most categories reach consistent\n"
    "monthly activity (≥ 40 categories reporting every month)."
))

# ── 3. Load & filter ──────────────────────────────────────────────────────────
cells.append(code(
    'df = pd.read_parquet(DATA_DIR / "category_month_agg.parquet")\n'
    'df["year_month"] = pd.PeriodIndex(df["year_month"], freq="M")\n'
    "\n"
    'print(f"Full dataset:  {df.shape[0]:,} rows  |  {df[\'product_category_name\'].nunique()} categories")\n'
    'print(f"Date range:    {df[\'year_month\'].min()} to {df[\'year_month\'].max()}")\n'
    "print()\n"
    "\n"
    'df_model = df[df["year_month"] >= pd.Period("2017-01", "M")].copy()\n'
    'print(f"After filter:  {df_model.shape[0]:,} rows  |  {df_model[\'product_category_name\'].nunique()} categories")\n'
    'print(f"Date range:    {df_model[\'year_month\'].min()} to {df_model[\'year_month\'].max()}")\n'
    "print()\n"
    'print("Monthly sales distribution (items per category per month):")\n'
    'print(df_model["sales"].describe().round(1).to_string())'
))

# ── 4. Section 2 header ───────────────────────────────────────────────────────
cells.append(md(
    "---\n"
    "## Section 2 — Log Transformation\n"
    "\n"
    "### Why log-log?\n"
    "\n"
    "Price-demand relationships are multiplicative, not additive. A R\\$10 increase\n"
    "means something very different for a R\\$20 product vs. a R\\$500 product.\n"
    "Taking logarithms converts the relationship to **percentage changes**, making\n"
    "the OLS coefficient directly interpretable as elasticity:\n"
    "\n"
    "> **\"A 1% increase in average price leads to a β% change in monthly sales volume.\"**\n"
    "\n"
    "We use `np.log1p(x) = log(1 + x)` instead of `log(x)` to guard against any\n"
    "zero values that may appear in future data refreshes.\n"
    "\n"
    "| Original | Log version | Role in model |\n"
    "|----------|-------------|---------------|\n"
    "| `sales` | `log_sales` | Dependent variable (demand proxy) |\n"
    "| `avg_price` | `log_price` | Key predictor — gives us β (elasticity) |\n"
    "| `avg_freight` | `log_freight` | Confounder: high freight inflates perceived total cost |\n"
    "| `avg_distance` | `log_distance` | Diagnostic; correlated with freight |"
))

# ── 5. Log transform ──────────────────────────────────────────────────────────
cells.append(code(
    "for col, new_col in [\n"
    '    ("sales",        "log_sales"),\n'
    '    ("avg_price",    "log_price"),\n'
    '    ("avg_freight",  "log_freight"),\n'
    '    ("avg_distance", "log_distance"),\n'
    "]:\n"
    "    df_model[new_col] = np.log1p(df_model[col])\n"
    "\n"
    'log_cols = ["log_sales", "log_price", "log_freight", "log_distance"]\n'
    'print("Descriptive stats for log-transformed columns:")\n'
    "print(df_model[log_cols].describe().round(3).to_string())\n"
    "print()\n"
    'show = ["product_category_name", "year_month", "avg_price", "log_price", "sales", "log_sales"]\n'
    'print("Sample: original vs. log-transformed values:")\n'
    "print(df_model[show].head(5).to_string(index=False))"
))

# ── 6. Section 3 header ───────────────────────────────────────────────────────
cells.append(md(
    "---\n"
    "## Section 3 — Global Elasticity Model (Baseline)\n"
    "\n"
    "We start with a **pooled OLS** across all categories and months.\n"
    "This gives us the average price elasticity of the entire Olist platform —\n"
    "a useful benchmark before we disaggregate by category.\n"
    "\n"
    "**Model specification:**\n"
    "```\n"
    "log_sales ~ log_price + log_freight + avg_review + late_delivery_rate + C(month)\n"
    "```\n"
    "\n"
    "* `C(month)` adds 11 binary dummies (reference = January) to absorb calendar\n"
    "  seasonality. Without this, β would conflate price changes with seasonal swings.\n"
    "* `log_freight` controls for the fact that high-freight items often have higher\n"
    "  prices — omitting it would bias β toward zero or inflate it.\n"
    "* `avg_review` and `late_delivery_rate` capture service quality, which affects\n"
    "  repeat purchases and may correlate with price positioning.\n"
    "\n"
    "**Why C(month) works here but not per-category:**  \n"
    "The pooled dataset has 1,210 rows — far more than the 16 parameters in the model.\n"
    "For per-category models (Section 4), each category has at most 20 monthly\n"
    "observations, exactly equal to the number of months; adding month dummies would\n"
    "exhaust all degrees of freedom. Section 4 therefore uses the model *without*\n"
    "`C(month)`, isolating the pure price signal within each category."
))

# ── 7. Global OLS ────────────────────────────────────────────────────────────
cells.append(code(
    "GLOBAL_FORMULA = (\n"
    '    "log_sales ~ log_price + log_freight + avg_review "\n'
    '    "+ late_delivery_rate + C(month)"\n'
    ")\n"
    "CAT_FORMULA = (\n"
    '    "log_sales ~ log_price + log_freight + avg_review + late_delivery_rate"\n'
    ")\n"
    "\n"
    "global_model = smf.ols(GLOBAL_FORMULA, data=df_model).fit()\n"
    "print(global_model.summary())"
))

# ── 8. Extract global elasticity ──────────────────────────────────────────────
cells.append(code(
    'elas  = global_model.params["log_price"]\n'
    'pval  = global_model.pvalues["log_price"]\n'
    'ci    = global_model.conf_int()\n'
    'ci_lo = ci.loc["log_price"].iloc[0]\n'
    'ci_hi = ci.loc["log_price"].iloc[1]\n'
    "\n"
    "if pval < 0.01:\n"
    '    sig_label = "*** (p < 0.01)"\n'
    "elif pval < 0.05:\n"
    '    sig_label = "** (p < 0.05)"\n'
    "elif pval < 0.10:\n"
    '    sig_label = "* (p < 0.10)"\n'
    "else:\n"
    '    sig_label = "(not significant)"\n'
    "\n"
    'print("=" * 56)\n'
    'print("  GLOBAL PRICE ELASTICITY")\n'
    'print("=" * 56)\n'
    'print(f"  Elasticity (beta):  {elas:+.4f}  {sig_label}")\n'
    'print(f"  95% CI:             [{ci_lo:.4f},  {ci_hi:.4f}]")\n'
    'print(f"  R-squared:          {global_model.rsquared:.4f}")\n'
    'print(f"  N observations:     {int(global_model.nobs):,}")\n'
    'print("=" * 56)\n'
    "print()\n"
    'print(f"  Interpretation: a 1% increase in avg_price is associated")\n'
    'print(f"  with a {elas:+.2f}% change in monthly sales volume,")\n'
    'print(f"  holding freight, review score, and seasonality constant.")'
))

# ── 9. Business interpretation markdown ───────────────────────────────────────
cells.append(md(
    "### Business Interpretation of the Global Model\n"
    "\n"
    "The global coefficient is an average across all product categories and\n"
    "all Olist sellers. It tells us the baseline price sensitivity of the platform.\n"
    "\n"
    "| Scenario | What it means for pricing strategy |\n"
    "|----------|-------------------------------------|\n"
    "| **β < −1** (elastic) | Platform demand is price-sensitive. Promotions drive disproportionate volume; raising prices may shrink total GMV. |\n"
    "| **−1 < β < 0** (inelastic) | Demand is relatively stable to price. Modest increases grow revenue with limited volume loss. |\n"
    "| **β ≈ 0** | Price is not the main demand driver at a platform level. Quality, assortment, and logistics dominate. |\n"
    "| **β > 0** | May indicate positive brand/quality signalling, or an omitted variable (e.g., product newness, seasonality). |\n"
    "\n"
    "> **Caution:** This is an observational estimate, not a controlled experiment.\n"
    "> Sellers who raise prices may simultaneously improve their assortment or\n"
    "> invest in marketing, which would bias β upward. Treat as a directional signal."
))

# ── 10. Section 4 header ──────────────────────────────────────────────────────
cells.append(md(
    "---\n"
    "## Section 4 — Elasticity by Category\n"
    "\n"
    "The global estimate masks substantial heterogeneity: a luxury-electronics buyer\n"
    "behaves very differently from someone buying bed linen or garden tools.\n"
    "We now fit the category-level OLS for each category with ≥ 12 monthly obs:\n"
    "\n"
    "```\n"
    "log_sales ~ log_price + log_freight + avg_review + late_delivery_rate\n"
    "```\n"
    "\n"
    "Month dummies are excluded here because each category has at most 20 monthly\n"
    "rows — adding 19 dummies would leave zero degrees of freedom. The seasonality\n"
    "is already captured in the global model; here we isolate the price signal.\n"
    "\n"
    "Results are stored with:\n"
    "* `price_elasticity` — the `log_price` coefficient β for that category\n"
    "* `p_value` — statistical significance (H₀: β = 0)\n"
    "* `r_squared` — how well the 4-variable model explains demand variation\n"
    "* `significant` — True if p_value < 0.05"
))

# ── 11. Per-category loop ─────────────────────────────────────────────────────
cells.append(code(
    "records = []\n"
    "skipped = 0\n"
    "\n"
    'for cat, grp in df_model.groupby("product_category_name"):\n'
    "    if len(grp) < 12:\n"
    "        skipped += 1\n"
    "        continue\n"
    "    try:\n"
    "        m = smf.ols(CAT_FORMULA, data=grp).fit(disp=False)\n"
    '        elas_val = m.params.get("log_price", np.nan)\n'
    '        pval_val = m.pvalues.get("log_price", np.nan)\n'
    "        if pd.isna(elas_val):\n"
    "            skipped += 1\n"
    "            continue\n"
    "        records.append({\n"
    '            "category":         cat,\n'
    '            "price_elasticity": elas_val,\n'
    '            "p_value":          pval_val,\n'
    '            "r_squared":        m.rsquared,\n'
    '            "n_obs":            int(m.nobs),\n'
    '            "significant":      pval_val < 0.05,\n'
    "        })\n"
    "    except Exception:\n"
    "        skipped += 1\n"
    "\n"
    "elasticity_df = (\n"
    "    pd.DataFrame(records)\n"
    '    .sort_values("price_elasticity")\n'
    "    .reset_index(drop=True)\n"
    ")\n"
    "\n"
    'print(f"Categories modelled:            {len(elasticity_df)}")\n'
    'print(f"Skipped (< 12 obs or error):    {skipped}")\n'
    'print(f"Significant (p < 0.05):         {elasticity_df[\'significant\'].sum()}")\n'
    'print(f"Negative elasticity (expected): {(elasticity_df[\'price_elasticity\'] < 0).sum()}")\n'
    'print(f"Positive elasticity (unusual):  {(elasticity_df[\'price_elasticity\'] > 0).sum()}")\n'
    "print()\n"
    'print("Elasticity distribution:")\n'
    'print(elasticity_df["price_elasticity"].describe().round(3).to_string())'
))

# ── 12. Top 10 display ────────────────────────────────────────────────────────
cells.append(code(
    "HDR = (\n"
    "    f\"  {'Category':<42s} {'Elasticity':>11s}  {'p-val':>7s}  \"\n"
    "    f\"{'R2':>6s}  {'n':>4s}  {'Sig?':>5s}\"\n"
    ")\n"
    'SEP = "  " + chr(8212) * 80\n'
    "\n"
    "\n"
    "def print_rows(subset):\n"
    "    for _, r in subset.iterrows():\n"
    '        sig = "yes" if r["significant"] else "no "\n'
    "        print(\n"
    "            f\"  {r['category']:<42s} {r['price_elasticity']:>11.3f}  \"\n"
    "            f\"{r['p_value']:>7.4f}  {r['r_squared']:>6.3f}  \"\n"
    "            f\"{int(r['n_obs']):>4}  {sig:>5s}\"\n"
    "        )\n"
    "\n"
    "\n"
    "print(\"TOP 10 MOST PRICE-ELASTIC (highest demand sensitivity to price):\")\n"
    "print(HDR); print(SEP)\n"
    "print_rows(elasticity_df.head(10))\n"
    "\n"
    "print()\n"
    "print(\"TOP 10 LEAST PRICE-ELASTIC (demand least sensitive to price):\")\n"
    "print(HDR); print(SEP)\n"
    "print_rows(elasticity_df.tail(10).iloc[::-1])"
))

# ── 13. Commercial decisions markdown ─────────────────────────────────────────
cells.append(md(
    "### What Elasticity Means for Commercial Decisions\n"
    "\n"
    "| Category type | Elasticity range | Recommended action |\n"
    "|---------------|------------------|-----------|\n"
    "| **Highly elastic** (β < −1.5) | Price-sensitive buyers | Do not raise price; grow volume through promotions or bundles |\n"
    "| **Moderately elastic** (−1.5 ≤ β < −1) | Meaningful price sensitivity | Price near the revenue-maximising point (see Section 5) |\n"
    "| **Inelastic** (−1 ≤ β < 0) | Stable demand | Gradual increases grow revenue with minimal volume loss |\n"
    "| **Near-zero / positive** | Price not main driver | Invest in quality signals: photos, review score, faster delivery |\n"
    "\n"
    "> **Statistical caveat:** Only categories with `significant = True` (p < 0.05)\n"
    "> have reliable elasticity estimates. Treat insignificant results as directional\n"
    "> signals only — the confidence interval likely spans zero."
))

# ── 14. Section 5 header ──────────────────────────────────────────────────────
cells.append(md(
    "---\n"
    "## Section 5 — Revenue Curve per Category\n"
    "\n"
    "Given an elasticity estimate, we can simulate monthly revenue as price moves\n"
    "away from its current average. The log-log model implies a power-law demand:\n"
    "\n"
    "```\n"
    "Q(P) = Q_base * (P / P_base)^β\n"
    "Revenue(P) = P * Q(P)\n"
    "```\n"
    "\n"
    "The analytical revenue-maximising price is **P\\* = −P_base · β / (1 + β)**\n"
    "when β < −1 (elastic demand). For inelastic categories (β > −1), revenue\n"
    "increases monotonically with price — in the simulation the curve will peak at\n"
    "the 200% boundary, signalling that modest price increases are safe.\n"
    "\n"
    "We run this for the **top 5 categories by total sales volume** — those where\n"
    "pricing decisions have the highest absolute revenue impact on the platform."
))

# ── 15. Revenue curve function ────────────────────────────────────────────────
cells.append(code(
    "def build_revenue_curve(\n"
    "    category: str,\n"
    "    elasticity: float,\n"
    "    base_price: float,\n"
    "    base_sales: float,\n"
    "    n_points: int = 200,\n"
    ") -> tuple:\n"
    "    # Simulate prices from 50% to 200% of the current average price\n"
    "    prices = np.linspace(base_price * 0.50, base_price * 2.0, n_points)\n"
    "    # Power-law demand: Q(P) = Q_base * (P / P_base)^elasticity\n"
    "    estimated_sales = base_sales * (prices / base_price) ** elasticity\n"
    "    revenue = prices * estimated_sales\n"
    "    df_curve = pd.DataFrame({\n"
    '        "price":           prices,\n'
    '        "estimated_sales": estimated_sales,\n'
    '        "revenue":         revenue,\n'
    "    })\n"
    '    optimal_price = df_curve.loc[df_curve["revenue"].idxmax(), "price"]\n'
    "    return df_curve, optimal_price"
))

# ── 16. Run top 5 ─────────────────────────────────────────────────────────────
cells.append(code(
    "top5_cats = (\n"
    '    df_model.groupby("product_category_name")["sales"]\n'
    "    .sum()\n"
    '    .sort_values(ascending=False)\n'
    "    .head(5)\n"
    "    .index.tolist()\n"
    ")\n"
    "print(f\"Top 5 categories by total sales: {top5_cats}\")\n"
    "print()\n"
    "\n"
    "HDR5 = (\n"
    "    f\"{'Category':<28s}  {'Elasticity':>10s}  {'Base P':>8s}  {'Sales/mo':>8s}  \"\n"
    "    f\"{'Opt Price':>9s}  {'Curr Rev':>9s}  {'Max Rev':>9s}  {'Uplift':>7s}  {'Type':>9s}\"\n"
    ")\n"
    "print(HDR5)\n"
    'print(chr(8212) * 115)\n'
    "\n"
    "curve_results = {}\n"
    "for cat in top5_cats:\n"
    '    row = elasticity_df[elasticity_df["category"] == cat]\n'
    "    if row.empty:\n"
    '        print(f"{cat:<28s}  [no elasticity estimate]")\n'
    "        continue\n"
    "\n"
    '    elasticity   = row["price_elasticity"].values[0]\n'
    '    cat_data     = df_model[df_model["product_category_name"] == cat]\n'
    '    base_price   = cat_data["avg_price"].mean()\n'
    '    base_sales   = cat_data["sales"].mean()\n'
    "\n"
    "    df_curve, opt_price = build_revenue_curve(cat, elasticity, base_price, base_sales)\n"
    "    curve_results[cat] = df_curve\n"
    "\n"
    "    curr_rev   = base_price * base_sales\n"
    '    max_rev    = df_curve["revenue"].max()\n'
    "    uplift     = (max_rev - curr_rev) / curr_rev * 100\n"
    '    elas_type  = "elastic" if elasticity < -1 else "inelastic"\n'
    "\n"
    "    print(\n"
    "        f\"{cat:<28s}  {elasticity:>10.3f}  {base_price:>8.2f}  {base_sales:>8.1f}  \"\n"
    "        f\"{opt_price:>9.2f}  {curr_rev:>9.0f}  {max_rev:>9.0f}  \"\n"
    "        f\"{uplift:>6.1f}%  {elas_type:>9s}\"\n"
    "    )"
))

# ── 17. Curve interpretation markdown ─────────────────────────────────────────
cells.append(md(
    "### Reading the Revenue Curve Table\n"
    "\n"
    "* **Curr Rev** = `base_price × avg_sales_per_month` — the current revenue baseline.\n"
    "* **Max Rev** = peak of the simulated curve — achievable at the **Opt Price**.\n"
    "* **Uplift** = `(Max Rev − Curr Rev) / Curr Rev` — potential revenue gain from\n"
    "  repricing, assuming the demand curve is stable.\n"
    "\n"
    "| Demand type | What the curve looks like | Practical meaning |\n"
    "|-------------|--------------------------|-------------------|\n"
    "| **Elastic** (β < −1) | Inverted-U with clear interior peak | Move price toward Opt Price for maximum revenue |\n"
    "| **Inelastic** (β > −1) | Monotonically increasing | Price can be raised; Opt Price = simulation upper bound |\n"
    "\n"
    "> These curves are **model-based projections**, not guarantees. Real markets\n"
    "> are affected by competitor pricing, stock availability, and marketing spend.\n"
    "> Use as a directional tool, validated against A/B price tests."
))

# ── 18. Section 6 header ──────────────────────────────────────────────────────
cells.append(md(
    "---\n"
    "## Section 6 — Save Results\n"
    "\n"
    "We persist the per-category elasticity table in two formats:\n"
    "* **Parquet** — efficient binary format for the Streamlit app and notebook 03.\n"
    "* **CSV** — human-readable for quick inspection in Excel or a text editor.\n"
    "\n"
    "The table is sorted by `price_elasticity` (most elastic first) so that\n"
    "the highest-priority repricing opportunities appear at the top."
))

# ── 19. Save ──────────────────────────────────────────────────────────────────
cells.append(code(
    'parquet_path = DATA_DIR / "elasticity_by_category.parquet"\n'
    'csv_path     = DATA_DIR / "elasticity_by_category.csv"\n'
    "\n"
    "elasticity_df.to_parquet(parquet_path, index=False)\n"
    "elasticity_df.to_csv(csv_path, index=False)\n"
    "\n"
    'print(f"Saved: {parquet_path.resolve()}")\n'
    'print(f"Saved: {csv_path.resolve()}")\n'
    'print(f"Shape: {elasticity_df.shape}")\n'
    "print()\n"
    'print("First 10 rows (most elastic → least elastic):")\n'
    "print(elasticity_df.head(10).to_string(index=False))"
))

# ── Build & write ─────────────────────────────────────────────────────────────
nb = {
    "nbformat": 4,
    "nbformat_minor": 5,
    "metadata": {
        "kernelspec": {
            "display_name": "Python 3",
            "language": "python",
            "name": "python3",
        },
        "language_info": {"name": "python", "version": "3.9.0"},
    },
    "cells": cells,
}

with open(NB_PATH, "w", encoding="utf-8") as f:
    json.dump(nb, f, ensure_ascii=False, indent=1)

print(f"Notebook written: {NB_PATH}")
print(f"Total cells: {len(cells)}")
