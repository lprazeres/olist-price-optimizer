# 🛒 Olist Price Optimizer

![Python](https://img.shields.io/badge/Python-3.10%2B-blue?logo=python&logoColor=white)
![Streamlit](https://img.shields.io/badge/Streamlit-app-FF4B4B?logo=streamlit&logoColor=white)
![License](https://img.shields.io/badge/License-MIT-green)

**Price elasticity analysis of 100K+ e-commerce transactions — identifying where price changes drive revenue and recommending optimal price points per product category.**

---

## 🎯 Business Problem

> *How can Olist strategically adjust product prices without compromising sales volume and revenue?*

Olist is Brazil's largest B2B marketplace, connecting small retailers to major e-commerce platforms (Mercado Livre, Amazon BR, B2W, etc.). With over 100,000 transactions across 70+ product categories, pricing decisions are largely intuition-driven — leaving significant revenue on the table.

This project answers: **which categories are price-sensitive, by how much, and what is the revenue-optimal price for each?**

---

## 📊 Dataset

| Attribute | Detail |
|---|---|
| Source | [Olist Brazilian E-Commerce — Kaggle](https://www.kaggle.com/datasets/olistbr/brazilian-ecommerce) |
| Period | 2016–2018 |
| Scale | ~100,000 orders · 70+ product categories |
| Tables | `orders`, `order_items`, `products`, `customers`, `sellers`, `reviews`, `geolocation` |
| Join | 7 tables joined, engineered into a monthly panel per category |

---

## 🔬 Methodology

### 1. Data Aggregation (`01_aggregation.ipynb`)
Raw order-level data joined across 7 tables and aggregated into a **monthly panel per category** (1,241 observations). Features engineered include shipping distance (haversine formula), delivery delay rate, and monthly revenue/volume metrics.

### 2. OLS Log-Log Regression (`02_elasticity_model.ipynb`)
One model fitted per category using the log-log specification:

```
log(monthly_sales) ~ log(avg_price) + log(avg_freight)
                   + avg_review_score + late_delivery_rate
```

The coefficient on `log(avg_price)` is the **price elasticity of demand (β)**. A category is considered statistically significant at p < 0.05. Categories with fewer than 6 months of data are excluded.

### 3. K-Means Clustering (`03_clustering.ipynb`)
All 62 categories clustered into **4 strategic segments** using elasticity, price level, sales volume, review score, and freight cost — enabling segment-level pricing strategy.

---

## 📈 Key Findings

| Metric | Value |
|---|---|
| Categories modelled | 62 |
| Statistically significant elasticity | 13 |
| Price-elastic categories (β < −1) | 6 |
| Avg revenue uplift potential (elastic group) | ~+109% |
| Strategic clusters identified | 4 |

### 6 Price-Elastic Categories

These are the highest-priority repricing targets — demand drops more than proportionally when price rises, so a price reduction yields a net revenue gain.

| Category | Elasticity (β) | Cluster | Uplift Potential |
|---|---|---|---|
| Watches & Gifts | −3.06 | ⭐ Volume Leaders | +317% |
| Industry, Commerce & Business | −2.32 | 🏋️ Heavy-Freight Specialists | +149% |
| Electronics | −2.00 | ⭐ Volume Leaders | +100% |
| Auto | −1.94 | ⭐ Volume Leaders | +92% |
| Furniture — Living Room | −1.88 | 🏋️ Heavy-Freight Specialists | +84% |
| Garden Tools | −1.29 | ⭐ Volume Leaders | +23% |
| Books — Technical | −1.01 | 💎 Quality Niche | +1% |

> **Note:** Uplift estimates assume the log-log elasticity holds across the simulated price range (50%–200% of current price). Treat as directional signals, not precise forecasts.

### 4 Strategic Clusters

| Cluster | Description | Action |
|---|---|---|
| 🏋️ Heavy-Freight Specialists | Heavy/bulky products where freight dominates the purchase decision | Freight negotiation over price cuts |
| ⭐ Volume Leaders | Platform backbone — high volume, mixed price sensitivity | Reprice the elastic ones first |
| 💎 Quality Niche | High satisfaction, lower volume, quality-driven buyers | Invest in discoverability |
| ⚠️ Conversion Risk | Below-average reviews and lower sales | Fix operations before touching price |

---

## 🚀 Live Demo

🔗 *[Open in Streamlit](https://sales-price-optimizer.streamlit.app)*

The app features 4 tabs:
- **About** — project overview and key findings
- **Revenue Impact** — uplift table for all 13 significant categories
- **Revenue Optimizer** — interactive price simulator per category
- **Category Explorer** — all 62 categories by cluster with price sensitivity map

---

## 🗂️ Project Structure

```
olist-price-optimizer/
├── data/
│   ├── category_month_agg.parquet        # Monthly panel (1,241 rows)
│   ├── elasticity_by_category.parquet    # OLS results (62 categories)
│   └── category_clusters.parquet         # K-Means clusters (62 categories)
├── notebooks/
│   ├── 01_aggregation.ipynb              # Data join & feature engineering
│   ├── 02_elasticity_model.ipynb         # OLS log-log regression
│   ├── 03_clustering.ipynb               # K-Means segmentation
│   └── elbow_chart.png                   # K selection chart
├── app/
│   └── app.py                            # Streamlit app (4 tabs)
├── requirements.txt
├── .gitignore
└── README.md
```

> `pricing_analysis_cleaned.parquet` (raw, 109K rows) is excluded from this repo due to file size. Run `01_aggregation.ipynb` first to generate it, or download the source tables from [Kaggle](https://www.kaggle.com/datasets/olistbr/brazilian-ecommerce).

---

## ⚙️ How to Run Locally

```bash
git clone https://github.com/lprazeres/olist-price-optimizer.git
cd olist-price-optimizer
pip install -r requirements.txt
streamlit run app/app.py
```

The processed data files (`category_month_agg.parquet`, `elasticity_by_category.parquet`, `category_clusters.parquet`) are included in the repo and are sufficient to run the app directly.

---

## 🛠️ Tech Stack

| Layer | Tools |
|---|---|
| Data Processing | Python, pandas, numpy |
| Statistical Modelling | statsmodels (OLS) |
| Machine Learning | scikit-learn (K-Means) |
| Visualisation | Plotly, matplotlib |
| App | Streamlit |
| Data Format | Parquet |

---

## 👤 Author

**Lucas Silveira Prazeres**  
🎓 MSc Business Analytics — Kaplan Business School, Sydney  
💼 Business Analyst @ WebAlive  
🔗 [LinkedIn](https://linkedin.com/in/lucas-silveira-prazeres) · 💻 [GitHub](https://github.com/lprazeres)

---

*MIT License · Portfolio project · 2024*
