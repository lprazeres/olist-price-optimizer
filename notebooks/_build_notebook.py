"""Generates 01_aggregation.ipynb programmatically."""
import json
from pathlib import Path

NB_PATH = Path(__file__).parent / "01_aggregation.ipynb"

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


# ── Cell 0 ── Title ──────────────────────────────────────────────────────────
cells.append(md(
    "# Olist Price Optimizer — Step 1: Data Aggregation\n"
    "\n"
    "**Goal:** Transform the raw order-item dataset into a clean, monthly, category-level panel  \n"
    "that downstream notebooks will use to estimate price elasticity and optimise revenue.\n"
    "\n"
    "**Input:** `data/pricing_analysis_cleaned.parquet` — 109,653 order-item rows (2016-09-15 → 2018-08-29)  \n"
    "**Output:** `data/category_month_agg.parquet` — one row per (category × month), only categories  \n"
    "with ≥ 6 months of history (enough observations to fit a reliable price-response curve).\n"
    "\n"
    "---\n"
    "### Why aggregate at (category × month)?\n"
    "\n"
    "* **Category level** — price elasticity requires variation in price across comparable goods.  \n"
    "  Within a single category prices are set by different sellers, giving us natural experiments.  \n"
    "  Product-level aggregation would produce too-sparse cells; order-level would add noise.\n"
    "* **Month granularity** — weekly cells are too sparse for most categories; annual cells destroy  \n"
    "  the price variation we need. Monthly balances signal richness vs. sample size per cell."
))

# ── Cell 1 ── Imports ────────────────────────────────────────────────────────
cells.append(code(
    "import pandas as pd\n"
    "import numpy as np\n"
    "from pathlib import Path\n"
    "import warnings\n"
    'warnings.filterwarnings("ignore")\n'
    "\n"
    'DATA_DIR = Path("../data")\n'
    'print("Libraries loaded. DATA_DIR =", DATA_DIR.resolve())'
))

# ── Cell 2 ── Section header ─────────────────────────────────────────────────
cells.append(md(
    "---\n"
    "## 1 — Load Raw Data\n"
    "\n"
    "The parquet file is the result of joining five Olist public CSVs  \n"
    "(orders, order_items, products, customers, sellers) and computing  \n"
    "geolocation features at the order-item grain."
))

# ── Cell 3 ── Load data ──────────────────────────────────────────────────────
cells.append(code(
    'df = pd.read_parquet(DATA_DIR / "pricing_analysis_cleaned.parquet")\n'
    "\n"
    "print(f\"Shape:   {df.shape[0]:,} rows × {df.shape[1]} columns\")\n"
    "print(f\"\\nDate range:\")\n"
    "print(f\"  earliest order: {df['order_purchase_time'].min().date()}\")\n"
    "print(f\"  latest order:   {df['order_purchase_time'].max().date()}\")\n"
    "print(f\"\\nColumn inventory:\")\n"
    "for col in df.columns:\n"
    "    n_null = df[col].isna().sum()\n"
    "    null_str = f\"  ({n_null:,} nulls)\" if n_null else \"\"\n"
    "    print(f\"  {col:<40s} {str(df[col].dtype):<12s}{null_str}\")"
))

# ── Cell 4 ── Feature engineering header ─────────────────────────────────────
cells.append(md(
    "---\n"
    "## 2 — Engineer `shipping_distance_km` and `delivery_delay_days`\n"
    "\n"
    "These two features are not in the raw dataset but are needed both as  \n"
    "cleaning filters and as aggregation signals.\n"
    "\n"
    "* **`shipping_distance_km`** — great-circle (haversine) distance between  \n"
    "  customer and seller coordinates. Used to flag geocoding errors (> 4,300 km)  \n"
    "  and as a logistics-cost proxy in the aggregated panel.\n"
    "* **`delivery_delay_days`** — actual delivery date minus estimated delivery date.  \n"
    "  Positive = late. Used to compute a late-delivery rate per (category × month),  \n"
    "  an operational quality signal correlated with customer satisfaction."
))

# ── Cell 5 ── Compute distance + delay ───────────────────────────────────────
cells.append(code(
    "def haversine_km(\n"
    "    lat1: pd.Series, lon1: pd.Series,\n"
    "    lat2: pd.Series, lon2: pd.Series,\n"
    ") -> pd.Series:\n"
    '    """Return great-circle distance in km between two lat/lon point arrays."""\n'
    "    R = 6_371\n"
    "    dlat = np.radians(lat2 - lat1)\n"
    "    dlon = np.radians(lon2 - lon1)\n"
    "    a = (\n"
    "        np.sin(dlat / 2) ** 2\n"
    "        + np.cos(np.radians(lat1)) * np.cos(np.radians(lat2))\n"
    "        * np.sin(dlon / 2) ** 2\n"
    "    )\n"
    "    return 2 * R * np.arctan2(np.sqrt(a), np.sqrt(1 - a))\n"
    "\n"
    '\n'
    'df["shipping_distance_km"] = haversine_km(\n'
    '    df["customer_lat"], df["customer_lng"],\n'
    '    df["seller_lat"],   df["seller_lng"],\n'
    ")\n"
    "\n"
    'df["delivery_delay_days"] = (\n'
    '    df["order_delivered_customer_date"] - df["order_estimated_delivery_date"]\n'
    ").dt.days\n"
    "\n"
    'print("shipping_distance_km")\n'
    'print(df["shipping_distance_km"].describe().round(1).to_string())\n'
    'print()\n'
    'print("delivery_delay_days")\n'
    'print(df["delivery_delay_days"].describe().round(1).to_string())\n'
    "print(f\"\\nLate deliveries (delay > 0): {(df['delivery_delay_days'] > 0).sum():,}\")"
))

# ── Cell 6 ── Cleaning header ─────────────────────────────────────────────────
cells.append(md(
    "---\n"
    "## 3 — Data Cleaning\n"
    "\n"
    "We remove four types of problematic records before aggregating.  \n"
    "Each filter targets a specific data-quality issue; keeping them would  \n"
    "bias the price-response estimates.\n"
    "\n"
    "| Filter | Reason |\n"
    "|--------|--------|\n"
    "| `product_category_name.lower() == 'unknown'` | Catch-all bucket mixing unrelated goods; price signal is meaningless across mixed categories |\n"
    "| `freight_value == 0` | Data-entry error; inflates avg_price comparisons between categories with different logistics profiles |\n"
    "| `shipping_distance_km > 4,300` | Brazil's maximum terrestrial extent is ≈ 4,300 km — records beyond this are geocoding artifacts |\n"
    "| `product_weight_g == 0` | Physically impossible; missing weight recorded as zero, corrupting distance-adjusted pricing features |"
))

# ── Cell 7 ── Apply filters ───────────────────────────────────────────────────
cells.append(code(
    "n_before = len(df)\n"
    "\n"
    'mask_unknown  = df["product_category_name"].str.lower() == "unknown"\n'
    'mask_freight  = df["freight_value"] == 0\n'
    'mask_distance = df["shipping_distance_km"] > 4_300\n'
    'mask_weight   = df["product_weight_g"] == 0\n'
    "\n"
    "drop_mask = mask_unknown | mask_freight | mask_distance | mask_weight\n"
    "df_clean  = df[~drop_mask].copy()\n"
    "n_after   = len(df_clean)\n"
    "\n"
    'print("Records removed per filter (rows may overlap across filters):")\n'
    "print(f\"  unknown category :  {mask_unknown.sum():>6,}\")\n"
    "print(f\"  freight == 0     :  {mask_freight.sum():>6,}\")\n"
    "print(f\"  distance > 4,300 :  {mask_distance.sum():>6,}\")\n"
    "print(f\"  weight == 0      :  {mask_weight.sum():>6,}\")\n"
    "print(f\"  ── total unique ──:  {drop_mask.sum():>6,}\")\n"
    "print(f\"  rows before      :  {n_before:>6,}\")\n"
    "print(f\"  rows after       :  {n_after:>6,}  ({n_after / n_before:.1%} retained)\")\n"
    "print(f\"\\nCategories remaining: {df_clean['product_category_name'].nunique()}\")"
))

# ── Cell 8 ── Aggregation header ──────────────────────────────────────────────
cells.append(md(
    "---\n"
    "## 4 — Build `year_month` and Aggregate\n"
    "\n"
    "We collapse the order-item grain into a **(category × month)** panel.  \n"
    "Each row in the output dataset represents one category in one calendar month.\n"
    "\n"
    "**Metrics chosen:**\n"
    "\n"
    "| Column | Source | Aggregation | Rationale |\n"
    "|--------|--------|-------------|-----------|\n"
    "| `sales` | `order_id` | count | Demand proxy: number of items sold |\n"
    "| `revenue` | `price` | sum | Total revenue the category generated that month |\n"
    "| `avg_price` | `price` | **median** | Robust central tendency; median suppresses occasional flash-sale outliers |\n"
    "| `avg_freight` | `freight_value` | **median** | Logistics-cost proxy; affects perceived total price |\n"
    "| `avg_review` | `review_score` | mean | Customer-satisfaction signal; may moderate price sensitivity |\n"
    "| `avg_distance` | `shipping_distance_km` | **median** | Logistics complexity; high distance correlates with higher freight |\n"
    "| `late_delivery_rate` | `delivery_delay_days > 0` | mean | Operational quality; late deliveries erode willingness-to-pay |\n"
    "\n"
    "We use `pandas.Period` for `year_month` so that time arithmetic is unambiguous."
))

# ── Cell 9 ── Aggregate ───────────────────────────────────────────────────────
cells.append(code(
    'df_clean["year_month"] = df_clean["order_purchase_time"].dt.to_period("M")\n'
    "\n"
    "agg = (\n"
    '    df_clean\n'
    '    .groupby(["product_category_name", "year_month"])\n'
    "    .agg(\n"
    '        sales        =("order_id",             "count"),\n'
    '        revenue      =("price",                "sum"),\n'
    '        avg_price    =("price",                "median"),\n'
    '        avg_freight  =("freight_value",        "median"),\n'
    '        avg_review   =("review_score",         "mean"),\n'
    '        avg_distance =("shipping_distance_km", "median"),\n'
    "    )\n"
    "    .reset_index()\n"
    ")\n"
    "\n"
    "# Late-delivery rate: fraction of items arriving after estimated date\n"
    "late_rate = (\n"
    "    df_clean\n"
    '    .assign(is_late=df_clean["delivery_delay_days"] > 0)\n'
    '    .groupby(["product_category_name", "year_month"])["is_late"]\n'
    "    .mean()\n"
    '    .rename("late_delivery_rate")\n'
    "    .reset_index()\n"
    ")\n"
    'agg = agg.merge(late_rate, on=["product_category_name", "year_month"])\n'
    "\n"
    '# Month integer (1-12) for seasonality control\n'
    'agg["month"] = agg["year_month"].dt.month\n'
    "\n"
    "print(f\"Aggregated panel: {agg.shape[0]:,} rows × {agg.shape[1]} columns\")\n"
    "print(f\"Categories:       {agg['product_category_name'].nunique()}\")\n"
    "print(f\"Months covered:   {agg['year_month'].nunique()}\")\n"
    "print()\n"
    "print(agg.dtypes.to_string())\n"
    "print()\n"
    "print(agg.head(3).to_string(index=False))"
))

# ── Cell 10 ── Validation header ──────────────────────────────────────────────
cells.append(md(
    "---\n"
    "## 5 — Validate: Category Coverage\n"
    "\n"
    "A price-response curve estimated from fewer than **6 data points** (months)  \n"
    "is statistically unreliable — the polynomial can fit the noise rather than  \n"
    "the signal. We therefore keep only categories with ≥ 6 months of history.\n"
    "\n"
    "Below we show:\n"
    "1. How many categories pass the threshold.\n"
    "2. The top 15 categories by total sales volume (with their month counts)."
))

# ── Cell 11 ── Validate ───────────────────────────────────────────────────────
cells.append(code(
    "months_per_cat = (\n"
    '    agg.groupby("product_category_name")["year_month"]\n'
    "    .nunique()\n"
    '    .rename("n_months")\n'
    "    .sort_values(ascending=False)\n"
    ")\n"
    "\n"
    "n_pass = (months_per_cat >= 6).sum()\n"
    "n_fail = (months_per_cat <  6).sum()\n"
    "print(f\"Categories with >= 6 months: {n_pass} / {len(months_per_cat)}\")\n"
    "print(f\"Categories with  < 6 months: {n_fail}  (will be dropped)\")\n"
    "print()\n"
    "\n"
    "# Top 15 by total sales\n"
    "top15_sales = (\n"
    '    agg.groupby("product_category_name")["sales"]\n'
    "    .sum()\n"
    '    .rename("total_sales")\n'
    ")\n"
    'summary = pd.DataFrame({"total_sales": top15_sales, "n_months": months_per_cat})\n'
    'summary = summary.sort_values("total_sales", ascending=False).head(15)\n'
    'summary["qualifies"] = summary["n_months"] >= 6\n'
    "\n"
    "print(\"Top 15 categories by total sales:\")\n"
    "print(f\"  {'Category':<40s} {'Total Sales':>12s}  {'Months':>6s}  {'>=6?':>5s}\")\n"
    "print(\"  \" + \"-\" * 70)\n"
    "for cat, row in summary.iterrows():\n"
    "    tick = \"yes\" if row[\"qualifies\"] else \"NO \"\n"
    "    print(f\"  {cat:<40s} {int(row['total_sales']):>12,}  {int(row['n_months']):>6}  {tick:>5s}\")"
))

# ── Cell 12 ── Filter header ──────────────────────────────────────────────────
cells.append(md(
    "---\n"
    "## 6 — Filter to Qualified Categories\n"
    "\n"
    "We drop any category with fewer than 6 months of sales data.  \n"
    "These are typically niche products with very few transactions — not enough  \n"
    "to distinguish a price trend from random fluctuation."
))

# ── Cell 13 ── Filter ─────────────────────────────────────────────────────────
cells.append(code(
    "qualified_cats = months_per_cat[months_per_cat >= 6].index\n"
    'agg_filtered = agg[agg["product_category_name"].isin(qualified_cats)].copy()\n'
    "\n"
    "print(f\"Categories retained: {agg_filtered['product_category_name'].nunique()}\")\n"
    "print(f\"Rows retained:       {len(agg_filtered):,} (of {len(agg):,})\")\n"
    "print()\n"
    "print(\"Month distribution across the panel:\")\n"
    'print(agg_filtered["year_month"].value_counts().sort_index().to_string())'
))

# ── Cell 14 ── Save header ────────────────────────────────────────────────────
cells.append(md(
    "---\n"
    "## 7 — Save to Parquet\n"
    "\n"
    "We convert `year_month` from `Period` to `str` (e.g. `\"2017-03\"`) before  \n"
    "writing, because the Parquet format does not natively support pandas Period.  \n"
    "Downstream notebooks will parse it back with `pd.PeriodIndex(..., freq='M')`."
))

# ── Cell 15 ── Save ───────────────────────────────────────────────────────────
cells.append(code(
    'out_path = DATA_DIR / "category_month_agg.parquet"\n'
    "\n"
    "agg_out = agg_filtered.copy()\n"
    'agg_out["year_month"] = agg_out["year_month"].astype(str)\n'
    "\n"
    "agg_out.to_parquet(out_path, index=False)\n"
    "print(f\"Saved: {out_path.resolve()}\")\n"
    "print(f\"Shape: {agg_out.shape}\")\n"
    "print()\n"
    "print(\"Final column dtypes:\")\n"
    "print(agg_out.dtypes.to_string())\n"
    "print()\n"
    "print(\"Sample (first 5 rows):\")\n"
    "print(agg_out.head().to_string(index=False))"
))

# ── Build and write notebook ──────────────────────────────────────────────────
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
