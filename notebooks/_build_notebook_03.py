"""Generates 03_clustering.ipynb programmatically."""
import json
from pathlib import Path

NB_PATH = Path(__file__).parent / "03_clustering.ipynb"
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
    "# Olist Price Optimizer — Step 3: Strategic Category Clustering\n"
    "\n"
    "**Goal:** Group the 62 modelled product categories into strategic segments\n"
    "using K-Means clustering, combining price elasticity with volume, price level,\n"
    "customer satisfaction, and logistics cost. Each cluster receives a commercial\n"
    "label and pricing recommendation that powers the Streamlit app.\n"
    "\n"
    "**Inputs:**\n"
    "* `data/category_month_agg.parquet` — monthly panel (1,210 rows post-2017 filter)\n"
    "* `data/elasticity_by_category.parquet` — per-category elasticity estimates (62 rows)\n"
    "\n"
    "**Output:** `data/category_clusters.parquet / .csv` — 62 rows with cluster label\n"
    "and commercial name, ready to join with the revenue curve data in the app.\n"
    "\n"
    "---\n"
    "### Clustering Rationale\n"
    "\n"
    "Elasticity alone is not sufficient for a pricing decision. Two categories\n"
    "can have similar elasticity but very different commercial profiles:\n"
    "\n"
    "| Scenario | Elasticity | Correct action |\n"
    "|----------|------------|----------------|\n"
    "| High-volume, elastic | −1.5 | Price cut → large revenue gain |\n"
    "| Low-volume, elastic | −1.5 | Price cut → small absolute gain, may not be worth the risk |\n"
    "| High-freight, inelastic | −0.5 | Logistics cost reduction matters more than price |\n"
    "| Low-satisfaction, any | varies | Fix operations first; price change amplifies churn |\n"
    "\n"
    "K-Means on five features — elasticity, price level, sales volume, review score,\n"
    "and freight — surfaces these compound profiles automatically."
))

# ── 1. Imports ────────────────────────────────────────────────────────────────
cells.append(code(
    "import warnings\n"
    'warnings.filterwarnings("ignore")\n'
    "\n"
    "import pandas as pd\n"
    "import numpy as np\n"
    "import matplotlib.pyplot as plt\n"
    "from sklearn.preprocessing import StandardScaler\n"
    "from sklearn.cluster import KMeans\n"
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
    "We load two datasets and merge them:\n"
    "* **Elasticity table** — one row per category with the OLS coefficients from Notebook 02.\n"
    "* **Monthly panel** — aggregated from Notebook 01, filtered to 2017-01 onward.\n"
    "  We collapse it to one row per category (median price, total sales, mean review, median freight)\n"
    "  to represent each category's commercial profile across the observed period.\n"
    "\n"
    "Why these aggregation choices?\n"
    "* **Median price** — more robust than mean; resistant to seasonal outlier months.\n"
    "* **Total sales** — captures absolute market size, not just the rate.\n"
    "* **Mean review** — average satisfaction over the entire period.\n"
    "* **Median freight** — logistics benchmark for the category."
))

# ── 3. Load & merge ───────────────────────────────────────────────────────────
cells.append(code(
    'agg = pd.read_parquet(DATA_DIR / "category_month_agg.parquet")\n'
    'agg["year_month"] = pd.PeriodIndex(agg["year_month"], freq="M")\n'
    'agg = agg[agg["year_month"] >= pd.Period("2017-01", "M")]\n'
    "\n"
    'elas = pd.read_parquet(DATA_DIR / "elasticity_by_category.parquet")\n'
    "\n"
    "# Collapse monthly panel to one row per category\n"
    "profile = (\n"
    '    agg.groupby("product_category_name")\n'
    "    .agg(\n"
    '        avg_price   =("avg_price",    "median"),\n'
    '        total_sales =("sales",        "sum"),\n'
    '        avg_review  =("avg_review",   "mean"),\n'
    '        avg_freight =("avg_freight",  "median"),\n'
    "    )\n"
    "    .reset_index()\n"
    '    .rename(columns={"product_category_name": "category"})\n'
    ")\n"
    "\n"
    'df = elas.merge(profile, on="category", how="left")\n'
    "\n"
    'print(f"Elasticity table:  {elas.shape[0]} rows × {elas.shape[1]} cols")\n'
    'print(f"Profile table:     {profile.shape[0]} rows × {profile.shape[1]} cols")\n'
    'print(f"Merged:            {df.shape[0]} rows × {df.shape[1]} cols")\n'
    "print()\n"
    'cols = ["category","price_elasticity","significant","avg_price","total_sales","avg_review","avg_freight"]\n'
    'print("Merged dataset (all 62 categories):")\n'
    "print(df[cols].to_string(index=False))"
))

# ── 4. Section 2 header ───────────────────────────────────────────────────────
cells.append(md(
    "---\n"
    "## Section 2 — Feature Engineering for Clustering\n"
    "\n"
    "### Feature selection\n"
    "\n"
    "| Feature | Transformation | Why included |\n"
    "|---------|---------------|---------------|\n"
    "| `price_elasticity` | none | Core signal — how price-sensitive is demand? |\n"
    "| `avg_price` | **log** | Covers R\\$13 to R\\$850; log compresses the scale so luxury goods don't dominate distance |\n"
    "| `total_sales` | **log** | Covers 28 to 10,901 items; log makes volume comparable across niche vs. mainstream |\n"
    "| `avg_review` | none | 3.59–4.55 range; already on a stable scale, no transform needed |\n"
    "| `avg_freight` | **log** | Covers R\\$11 to R\\$43; log separates bulky from standard categories |\n"
    "\n"
    "### Features intentionally excluded\n"
    "\n"
    "* `p_value` / `r_squared` — model quality indicators, not commercial properties of the category.\n"
    "* `avg_distance` — highly correlated with `avg_freight` (r > 0.85); adding it would double-weight logistics.\n"
    "* `significant` (bool) — a binary that would over-index statistically significant categories;\n"
    "  elasticity magnitude already captures this.\n"
    "\n"
    "### Standardisation\n"
    "\n"
    "K-Means uses Euclidean distance, which means features with larger absolute values dominate\n"
    "the cluster assignment. `StandardScaler` (z-score: subtract mean, divide by std) puts all\n"
    "five features on an equal footing."
))

# ── 5. Feature engineering + scale ───────────────────────────────────────────
cells.append(code(
    'df["log_avg_price"]   = np.log1p(df["avg_price"])\n'
    'df["log_total_sales"] = np.log1p(df["total_sales"])\n'
    'df["log_avg_freight"] = np.log1p(df["avg_freight"])\n'
    "\n"
    'FEATURES = ["price_elasticity", "log_avg_price", "log_total_sales",\n'
    '            "avg_review", "log_avg_freight"]\n'
    "\n"
    "X = df[FEATURES].values\n"
    "scaler = StandardScaler()\n"
    "X_sc = scaler.fit_transform(X)\n"
    "\n"
    'print("Features before scaling (describe):")\n'
    "print(df[FEATURES].describe().round(3).to_string())\n"
    "print()\n"
    'print("Features after StandardScaler (should have mean ≈ 0, std ≈ 1):")\n'
    "print(\n"
    "    pd.DataFrame(X_sc, columns=FEATURES)\n"
    "    .describe().round(3).to_string()\n"
    ")"
))

# ── 6. Section 3 header ───────────────────────────────────────────────────────
cells.append(md(
    "---\n"
    "## Section 3 — K-Means Clustering\n"
    "\n"
    "### Choosing K with the Elbow Method\n"
    "\n"
    "The **elbow method** plots within-cluster sum of squares (inertia) against K.\n"
    "Inertia always decreases as K grows — the goal is to find where the rate of\n"
    "decrease slows sharply (the 'elbow'), giving the best balance between\n"
    "compactness and interpretability.\n"
    "\n"
    "We test K = 2 to 8 and select **K = 4** because:\n"
    "* The inertia curve shows a visible inflection around K = 4.\n"
    "* Four clusters map cleanly to actionable commercial archetypes\n"
    "  (see Section 4 for names and recommendations).\n"
    "* Fewer clusters (2–3) merge meaningfully different profiles;\n"
    "  more clusters (5+) create segments too small to act on at a platform level."
))

# ── 7. Elbow method + chart ───────────────────────────────────────────────────
cells.append(code(
    "ks = range(2, 9)\n"
    "inertias = []\n"
    "\n"
    "for k in ks:\n"
    "    km = KMeans(n_clusters=k, random_state=42, n_init=20)\n"
    "    km.fit(X_sc)\n"
    "    inertias.append(km.inertia_)\n"
    "\n"
    "# ── plot ──\n"
    "fig, ax = plt.subplots(figsize=(8, 4))\n"
    "ax.plot(list(ks), inertias, marker='o', color='steelblue',\n"
    "        linewidth=2, markersize=8, label='Inertia')\n"
    "ax.axvline(x=4, color='#e74c3c', linestyle='--', linewidth=1.5,\n"
    "           label='K = 4 (selected)')\n"
    "ax.set_xlabel('Number of Clusters (K)', fontsize=12)\n"
    "ax.set_ylabel('Inertia (Within-cluster SSE)', fontsize=12)\n"
    "ax.set_title('K-Means Elbow Chart — Olist Category Clustering', fontsize=13)\n"
    "ax.legend(fontsize=10)\n"
    "ax.grid(True, alpha=0.3)\n"
    "ax.set_xticks(list(ks))\n"
    "plt.tight_layout()\n"
    "plt.savefig('elbow_chart.png', dpi=150, bbox_inches='tight')\n"
    "plt.show()\n"
    "\n"
    'print("Elbow chart saved: notebooks/elbow_chart.png")\n'
    "print()\n"
    'print(f"  {\'K\':<4s} {\'Inertia\':>10s}  {\'Delta\':>10s}")\n'
    'print("  " + "-" * 28)\n'
    "prev = None\n"
    "for k, inertia in zip(ks, inertias):\n"
    "    delta = f'{inertia - prev:+.2f}' if prev is not None else '—'\n"
    "    print(f'  {k:<4d} {inertia:>10.2f}  {delta:>10s}')\n"
    "    prev = inertia"
))

# ── 8. K-Means K=4 ───────────────────────────────────────────────────────────
cells.append(code(
    "km4 = KMeans(n_clusters=4, random_state=42, n_init=20)\n"
    'df["cluster"] = km4.fit_predict(X_sc)\n'
    "\n"
    'print("K-Means (K=4) fit complete.")\n'
    'print(f"Cluster sizes:")\n'
    'for c, n in df["cluster"].value_counts().sort_index().items():\n'
    '    print(f"  Cluster {c}: {n} categories")'
))

# ── 9. Section 4 header ───────────────────────────────────────────────────────
cells.append(md(
    "---\n"
    "## Section 4 — Cluster Profiling\n"
    "\n"
    "We examine each cluster's centroid (mean of each feature) in the original scale\n"
    "to understand what drives the grouping, then assign a commercial label that\n"
    "communicates the strategic implication to a non-technical audience.\n"
    "\n"
    "**Four clusters identified:**\n"
    "\n"
    "| Cluster | Name | Defining characteristics |\n"
    "|---------|------|---------------------------|\n"
    "| 0 | **Heavy-Freight Specialists** | Highest price (R\\$243) and freight (R\\$31), lowest volume — bulky industrial/furniture goods |\n"
    "| 1 | **Volume Leaders** | Highest total sales (4,200/cat), affordable price (R\\$78), platform backbone |\n"
    "| 2 | **Quality Niche** | Highest satisfaction score (4.33), lowest price (R\\$58), specialty interest products |\n"
    "| 3 | **Conversion Risk** | Lowest satisfaction (3.89), low volume — operational issues suppress demand |"
))

# ── 10. Cluster profiling code ────────────────────────────────────────────────
cells.append(code(
    "# Centroid table in original-scale units\n"
    "profile_cols = {\n"
    '    "price_elasticity": "Elasticity",\n'
    '    "avg_price":        "Avg Price",\n'
    '    "total_sales":      "Total Sales",\n'
    '    "avg_review":       "Review",\n'
    '    "avg_freight":      "Freight",\n'
    '    "significant":      "%Sig (p<.05)",\n'
    "}\n"
    "\n"
    "cluster_profile = (\n"
    '    df.groupby("cluster")\n'
    "    .agg(\n"
    '        n_categories    =("category",        "count"),\n'
    '        price_elasticity=("price_elasticity","mean"),\n'
    '        avg_price       =("avg_price",       "mean"),\n'
    '        total_sales     =("total_sales",     "mean"),\n'
    '        avg_review      =("avg_review",      "mean"),\n'
    '        avg_freight     =("avg_freight",     "mean"),\n'
    '        pct_significant =("significant",     "mean"),\n'
    "    )\n"
    "    .round(3)\n"
    ")\n"
    "\n"
    'print("=" * 75)\n'
    'print("CLUSTER PROFILES (original scale)")\n'
    'print("=" * 75)\n'
    "print(\n"
    "    cluster_profile.rename(columns={\n"
    '        "n_categories":     "n",\n'
    '        "price_elasticity": "Elasticity",\n'
    '        "avg_price":        "Avg Price",\n'
    '        "total_sales":      "Total Sales",\n'
    '        "avg_review":       "Review",\n'
    '        "avg_freight":      "Freight",\n'
    '        "pct_significant":  "%Sig",\n'
    "    }).to_string()\n"
    ")"
))

# ── 11. Name clusters + list categories ──────────────────────────────────────
cells.append(code(
    "CLUSTER_NAMES = {\n"
    '    0: "Heavy-Freight Specialists",\n'
    '    1: "Volume Leaders",\n'
    '    2: "Quality Niche",\n'
    '    3: "Conversion Risk",\n'
    "}\n"
    "\n"
    'df["cluster_name"] = df["cluster"].map(CLUSTER_NAMES)\n'
    "\n"
    "DESCRIPTIONS = {\n"
    '    0: "Bulky/high-ticket goods (furniture, appliances, computers). Freight dominates perceived cost.",\n'
    '    1: "Platform backbone: high-volume, affordable, mainstream categories.",\n'
    '    2: "Specialist interest products (books, food, art). High satisfaction, low but loyal volume.",\n'
    '    3: "Diverse underperformers. Below-average satisfaction signals operational friction.",\n'
    "}\n"
    "\n"
    "print()\n"
    "for cluster_id in sorted(CLUSTER_NAMES):\n"
    "    name  = CLUSTER_NAMES[cluster_id]\n"
    "    desc  = DESCRIPTIONS[cluster_id]\n"
    "    cats  = df[df['cluster'] == cluster_id]['category'].tolist()\n"
    "    n     = len(cats)\n"
    "    print(f'Cluster {cluster_id} — {name} ({n} categories)')\n"
    "    print(f'  Profile: {desc}')\n"
    "    print(f'  Categories:')\n"
    "    for i in range(0, len(cats), 3):\n"
    "        row_cats = cats[i:i+3]\n"
    "        print('    ' + ',  '.join(f'{c}' for c in row_cats))\n"
    "    print()"
))

# ── 12. Commercial recommendations markdown ───────────────────────────────────
cells.append(md(
    "### Commercial Recommendations by Cluster\n"
    "\n"
    "---\n"
    "#### Cluster 0 — Heavy-Freight Specialists (9 categories)\n"
    "*furniture_bedroom, furniture_living_room, computers, office_furniture,\n"
    "home_appliances_2, industry_commerce_and_business, agro_industry_and_commerce,\n"
    "construction_tools_lights, kitchen_dining_laundry_garden_furniture*\n"
    "\n"
    "**Key insight:** Freight cost (avg R\\$31 vs. R\\$16 platform average) is 2× higher\n"
    "and likely suppresses demand more than the price itself.\n"
    "\n"
    "**Actions:**\n"
    "* Negotiate freight subsidies or flat-rate shipping caps for bulky goods.\n"
    "* Surface shipping cost transparently at checkout — surprise freight is the\n"
    "  top driver of cart abandonment in e-commerce.\n"
    "* For significantly elastic subcategories (furniture_living_room β = −1.88,\n"
    "  electronics β = −2.00), a combined freight-reduction + modest price cut\n"
    "  can unlock a disproportionate demand response.\n"
    "* `computers` at R\\$850 avg is a high-consideration purchase; invest in\n"
    "  product content quality (specs, photos, comparison tools).\n"
    "\n"
    "---\n"
    "#### Cluster 1 — Volume Leaders (23 categories)\n"
    "*bed_bath_table, health_beauty, sports_leisure, housewares, computers_accessories,\n"
    "furniture_decor, watches_gifts, electronics, auto, garden_tools, perfumery,\n"
    "pet_shop, baby, toys, cool_stuff, telephony, luggage_accessories, stationery,\n"
    "musical_instruments, consoles_games, small_appliances, fashion_bags_accessories,\n"
    "construction_tools_construction*\n"
    "\n"
    "**Key insight:** This cluster drives the bulk of platform GMV. It is highly\n"
    "heterogeneous internally — it contains both the most elastic categories\n"
    "(watches_gifts β = −3.06, electronics β = −2.00, auto β = −1.94)\n"
    "and the least elastic (health_beauty β = +1.24, sports_leisure β = +2.45).\n"
    "\n"
    "**Actions:**\n"
    "* **Elastic subcategories** (watches, electronics, auto, garden_tools):\n"
    "  targeted promotions or seller incentives to reduce price will generate\n"
    "  outsized volume gains and can increase total revenue.\n"
    "* **Positive-elasticity subcategories** (health_beauty, sports_leisure,\n"
    "  computers_accessories): the positive signal likely reflects quality/brand\n"
    "  premiums. Invest in curated collections, reviews, and brand partnerships.\n"
    "* Monitor this cluster quarterly — it is large enough that a 1% price shift\n"
    "  across the whole segment has material GMV impact.\n"
    "\n"
    "---\n"
    "#### Cluster 2 — Quality Niche (17 categories)\n"
    "*books_general_interest, books_technical, books_imported, food, food_drink,\n"
    "drinks, music, art, cine_photo, dvds_blu_ray, christmas_supplies,\n"
    "home_appliances, costruction_tools_garden, costruction_tools_tools,\n"
    "fashion_sport, fashion_shoes, tablets_printing_image*\n"
    "\n"
    "**Key insight:** Highest avg review score (4.33) in the platform. Buyers are\n"
    "enthusiasts and repeat purchasers. Total sales are low because these are niche\n"
    "categories, not because of poor experience.\n"
    "\n"
    "**Actions:**\n"
    "* **Grow discoverability** — these categories have loyal buyers but limited\n"
    "  reach. Boost search ranking, curated lists ('Book of the Month'),\n"
    "  and email campaigns targeting past buyers.\n"
    "* Modest price increases are safe for books and specialist tools —\n"
    "  the loyal audience is less price-sensitive than average.\n"
    "* Bundle opportunities: food + drinks, book + e-reader accessories,\n"
    "  art supplies + cine_photo.\n"
    "* Protect review scores — the high satisfaction is the cluster's moat.\n"
    "\n"
    "---\n"
    "#### Cluster 3 — Conversion Risk (13 categories)\n"
    "*home_confort, home_comfort_2, home_construction, air_conditioning,\n"
    "construction_tools_safety, audio, fixed_telephony, signaling_and_security,\n"
    "market_place, party_supplies, fashion_male_clothing, fashion_underwear_beach,\n"
    "fashio_female_clothing*\n"
    "\n"
    "**Key insight:** Lowest review score (3.89 — 0.44 points below Cluster 2).\n"
    "Low volume + low satisfaction is the most dangerous commercial combination:\n"
    "poor word-of-mouth suppresses organic growth, and price changes alone\n"
    "won't fix the root problem.\n"
    "\n"
    "**Actions:**\n"
    "* **Audit seller quality** for the lowest-rated categories;\n"
    "  enforce return/refund SLAs.\n"
    "* Investigate whether satisfaction issues stem from product misrepresentation\n"
    "  (description vs. reality) or delivery problems (damage, delay).\n"
    "* **Do not raise prices** until review scores recover — higher prices with\n"
    "  poor quality perception accelerates churn.\n"
    "* Fashion categories (male_clothing, underwear_beach, female_clothing)\n"
    "  may benefit from better size guidance and richer product imagery,\n"
    "  which are proven drivers of fashion e-commerce satisfaction."
))

# ── 13. Section 5 header ──────────────────────────────────────────────────────
cells.append(md(
    "---\n"
    "## Section 5 — Save Results\n"
    "\n"
    "The final table contains all 62 modelled categories with:\n"
    "* The original OLS elasticity estimates (from Notebook 02)\n"
    "* The commercial profile features (avg price, total sales, review, freight)\n"
    "* The K-Means cluster assignment and human-readable cluster name\n"
    "\n"
    "This file is the primary input for the Streamlit app in `app/`."
))

# ── 14. Save ──────────────────────────────────────────────────────────────────
cells.append(code(
    "save_cols = [\n"
    '    "category", "price_elasticity", "p_value", "r_squared", "n_obs",\n'
    '    "significant", "avg_price", "total_sales", "avg_review", "avg_freight",\n'
    '    "cluster", "cluster_name",\n'
    "]\n"
    "df_out = df[save_cols].copy()\n"
    "\n"
    'parquet_path = DATA_DIR / "category_clusters.parquet"\n'
    'csv_path     = DATA_DIR / "category_clusters.csv"\n'
    "\n"
    "df_out.to_parquet(parquet_path, index=False)\n"
    "df_out.to_csv(csv_path, index=False)\n"
    "\n"
    'print(f"Saved: {parquet_path.resolve()}")\n'
    'print(f"Saved: {csv_path.resolve()}")\n'
    'print(f"Shape: {df_out.shape}")\n'
    "print()\n"
    'print("All 62 categories — sorted by cluster then elasticity:")\n'
    "print(\n"
    '    df_out.sort_values(["cluster", "price_elasticity"])\n'
    '    [["cluster", "cluster_name", "category", "price_elasticity", "significant",\n'
    '      "avg_price", "total_sales", "avg_review"]]\n'
    "    .to_string(index=False)\n"
    ")"
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
