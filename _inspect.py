import pandas as pd, sys
sys.stdout.reconfigure(encoding="utf-8")

BASE = r"C:\Users\lpraz\OneDrive\Área de Trabalho\Masters Degree\02 - Portfolio\03 - Growth\02 - Olist price elasticity\data"

for name in ["category_month_agg", "elasticity_by_category", "category_clusters"]:
    df = pd.read_parquet(f"{BASE}\\{name}.parquet")
    print(f"=== {name} ===")
    print(f"  Shape: {df.shape}")
    print(f"  Cols:  {list(df.columns)}")
    if "cluster_name" in df.columns:
        print(f"  Clusters: {sorted(df['cluster_name'].unique())}")
    num = df.select_dtypes("number").columns.tolist()
    if num:
        print(df[num[:6]].describe().round(2).to_string())
    print()
