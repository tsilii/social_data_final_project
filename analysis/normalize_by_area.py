"""
Recompute danger scores normalized by zone area (crimes per km²),
rerun correlations, regenerate visualizations.
"""

import pandas as pd
import geopandas as gpd
import numpy as np
import json
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from scipy import stats
from pathlib import Path

ROOT = Path(__file__).parent.parent
VIZ  = ROOT / "visualizations"
DATA = Path(__file__).parent / "data"

# ── 1. Load shapefile and compute area in km² ─────────────────────────
# EPSG:2263 is NY State Plane (feet) — keep it projected for area calc
shp_path = DATA / "taxi_zones_shp" / "taxi_zones" / "taxi_zones.shp"
zones_gdf = gpd.read_file(shp_path)   # stays in EPSG:2263 (feet)
zones_gdf["LocationID"] = zones_gdf["LocationID"].astype(int)

# 1 sq ft = 9.2903e-8 km²
zones_gdf["area_km2"] = zones_gdf.geometry.area * 9.2903e-8
print("Area summary (km²):")
print(zones_gdf[["zone","area_km2"]].set_index("zone")
      .sort_values("area_km2", ascending=False).head(10).round(3))

# ── 2. Load raw danger scores and merge area ──────────────────────────
danger = pd.read_csv(DATA / "processed" / "zone_danger_scores.csv")
danger["LocationID"] = danger["LocationID"].astype(int)

merged = danger.merge(
    zones_gdf[["LocationID","area_km2"]], on="LocationID", how="left"
)

# Normalized metrics (per km²)
merged["crime_density"]   = merged["crime_count"]   / merged["area_km2"]
merged["felony_density"]  = merged["FELONY"]         / merged["area_km2"]
merged["danger_density"]  = merged["danger_score"]   / merged["area_km2"]  # weighted
merged["violent_density"] = merged["FELONY"]         / merged["area_km2"]  # felony proxy

print("\nTop 10 zones by crime density (crimes/km²):")
print(merged.nlargest(10, "crime_density")
      [["zone","borough","crime_count","area_km2","crime_density","danger_density"]]
      .round(1).to_string(index=False))

print("\nTop 10 zones by raw danger score (for comparison):")
print(merged.nlargest(10, "danger_score")
      [["zone","borough","danger_score","danger_density","area_km2"]]
      .round(1).to_string(index=False))

# Save updated danger file
merged.to_csv(DATA / "processed" / "zone_danger_scores_normalized.csv", index=False)
print(f"\nSaved zone_danger_scores_normalized.csv")

# ── 3. Merge with pickup tip summary ─────────────────────────────────
pickup = pd.read_csv(DATA / "processed" / "taxi_zone_tip_summary_pickup.csv")
pickup["LocationID"] = pickup["LocationID"].astype(int)

pu = pickup.merge(
    merged[["LocationID","area_km2","crime_density","felony_density","danger_density"]],
    on="LocationID", how="left"
)
pu_valid = pu[pu["trip_count"] >= 50].dropna(subset=["danger_density","mean_tip_pct"])

dropoff = pd.read_csv(DATA / "processed" / "taxi_zone_tip_summary_dropoff.csv")
dropoff["LocationID"] = dropoff["LocationID"].astype(int)

do = dropoff.merge(
    merged[["LocationID","area_km2","crime_density","felony_density","danger_density"]],
    on="LocationID", how="left"
)
do_valid = do[do["trip_count"] >= 50].dropna(subset=["danger_density","mean_tip_pct"])

# ── 4. Rerun correlations ─────────────────────────────────────────────
def correlate(df, danger_col, label):
    r_sp, p_sp = stats.spearmanr(df[danger_col], df["mean_tip_pct"])
    r_pe, p_pe = stats.pearsonr(df[danger_col], df["mean_tip_pct"])
    return {"label": label, "n": len(df), "spearman": r_sp, "pearson": r_pe, "p": p_sp}

results = []
results.append(correlate(pu_valid, "danger_density", "Pickup — normalized danger density"))
results.append(correlate(pu_valid, "felony_density", "Pickup — normalized felony density"))
results.append(correlate(do_valid, "danger_density", "Dropoff — normalized danger density"))

# Night / day
pu_night = pd.read_csv(DATA / "processed" / "taxi_zone_tip_summary_pickup_night.csv")
pu_day   = pd.read_csv(DATA / "processed" / "taxi_zone_tip_summary_pickup_day.csv")
for df_nd, label in [(pu_night, "Pickup NIGHT — normalized"), (pu_day, "Pickup DAY — normalized")]:
    df_nd = df_nd.merge(merged[["LocationID","danger_density"]], on="LocationID", how="left")
    df_nd = df_nd[df_nd["trip_count"] >= 50].dropna(subset=["danger_density","mean_tip_pct"])
    results.append(correlate(df_nd, "danger_density", label))

print("\n=== Correlations with NORMALIZED danger density ===")
for r in results:
    print(f"  {r['label']}: Spearman={r['spearman']:.3f}, Pearson={r['pearson']:.3f}, p={r['p']:.4f}, n={r['n']}")

# ── 5. Danger quartiles (normalized) ─────────────────────────────────
pu_valid = pu_valid.copy()
pu_valid["danger_group"] = pd.qcut(
    pu_valid["danger_density"], 4,
    labels=["Low", "Medium", "High", "Very High"]
)
bucket = (pu_valid.groupby("danger_group", observed=True)
          .agg(n_zones=("LocationID","count"),
               total_trips=("trip_count","sum"),
               mean_tip_pct=("mean_tip_pct","mean"))
          .reset_index())
print("\nBucket stats (normalized danger):")
print(bucket.round(2).to_string(index=False))

# ── 6. Regenerate crime hotspot map (normalized) ──────────────────────
zones_4326 = gpd.read_file(shp_path).to_crs(epsg=4326)
zones_4326["LocationID"] = zones_4326["LocationID"].astype(int)

map_df = zones_4326.merge(
    merged[["LocationID","zone","borough","crime_count","FELONY","MISDEMEANOR",
            "VIOLATION","danger_score","area_km2","crime_density","danger_density"]],
    on="LocationID", how="left"
)
map_df["danger_density_disp"] = map_df["danger_density"].round(1)
map_df["crime_density_disp"]  = map_df["crime_density"].round(1)
geojson = json.loads(map_df.to_json())

fig_map = px.choropleth_mapbox(
    map_df,
    geojson=geojson,
    locations=map_df.index,
    color="danger_density",
    color_continuous_scale=[
        [0.0, "#f1faee"],
        [0.3, "#f4a261"],
        [0.6, "#e76f51"],
        [0.85,"#c1121f"],
        [1.0, "#6b0000"],
    ],
    range_color=[0, map_df["danger_density"].quantile(0.95)],
    hover_data={
        "zone_y" if "zone_y" in map_df.columns else "zone": True,
        "borough_y" if "borough_y" in map_df.columns else "borough": True,
        "crime_density_disp": True,
        "danger_density_disp": True,
        "crime_count": True,
        "FELONY": True,
    },
    labels={
        "danger_density":      "Weighted danger/km²",
        "danger_density_disp": "Weighted danger/km²",
        "crime_density_disp":  "Crimes/km²",
        "crime_count":         "Total crimes",
        "FELONY":              "Felonies",
    },
    mapbox_style="carto-positron",
    zoom=9.8,
    center={"lat": 40.71, "lon": -73.94},
    opacity=0.78,
)
fig_map.update_layout(
    margin={"r":0,"t":40,"l":0,"b":0},
    height=560,
    title={"text": "Crime Density by NYC Taxi Zone — Weighted Danger Score per km² (2023)",
           "font": {"size": 14, "family": "Georgia"}, "x": 0.5},
    coloraxis_colorbar=dict(title="Danger/km²", thickness=14, len=0.55),
)

# Fix hover columns if suffixed after merge
for col in ["zone_y","borough_y"]:
    if col in map_df.columns:
        fig_map.update_traces(
            customdata=map_df[["zone_y","borough_y","crime_density_disp",
                               "danger_density_disp","crime_count","FELONY"]].values
        )
        break

fig_map.write_html(str(VIZ / "crime_hotspot_map.html"),
                   include_plotlyjs="cdn", full_html=True)
print("\nSaved crime_hotspot_map.html (normalized)")

# ── 7. Regenerate scatter plot (normalized) ───────────────────────────
borough_colors = {
    "Manhattan":    "#e63946",
    "Brooklyn":     "#457b9d",
    "Queens":       "#2a9d8f",
    "Bronx":        "#e9c46a",
    "Staten Island":"#8338ec",
    "EWR":          "#aaa",
}
scatter_df = pu_valid[pu_valid["danger_density"] > 0].copy()
scatter_df["color"] = scatter_df["borough"].map(borough_colors).fillna("#aaa")

fig_sc = go.Figure()
for bor, grp in scatter_df.groupby("borough"):
    fig_sc.add_trace(go.Scatter(
        x=grp["danger_density"],
        y=grp["mean_tip_pct"],
        mode="markers",
        name=bor,
        marker=dict(
            size=np.clip(np.sqrt(grp["trip_count"] / 300), 5, 30),
            color=borough_colors.get(bor, "#aaa"),
            opacity=0.75,
            line=dict(width=0.5, color="#fff"),
        ),
        text=grp["zone"],
        customdata=grp[["trip_count","danger_density","mean_tip_pct"]].values,
        hovertemplate=(
            "<b>%{text}</b><br>"
            "Danger density: %{customdata[1]:.1f}/km²<br>"
            "Avg tip: %{customdata[2]:.1f}%<br>"
            "Trips: %{customdata[0]:,}<extra></extra>"
        ),
    ))

# OLS trend line
x_vals = scatter_df["danger_density"].values
y_vals = scatter_df["mean_tip_pct"].values
m, b = np.polyfit(x_vals, y_vals, 1)
x_line = np.linspace(x_vals.min(), x_vals.max(), 100)
fig_sc.add_trace(go.Scatter(
    x=x_line, y=m * x_line + b,
    mode="lines",
    name="OLS trend",
    line=dict(color="#333", width=1.5, dash="dash"),
    showlegend=True,
))

fig_sc.update_layout(
    title={"text": "Avg Tip % vs Crime Density per Zone (2023)",
           "font": {"size": 14, "family": "Georgia"}, "x": 0.5},
    xaxis=dict(title="Weighted danger score per km²", gridcolor="#eee"),
    yaxis=dict(title="Average tip %", ticksuffix="%", gridcolor="#eee"),
    height=540,
    plot_bgcolor="#fafafa",
    paper_bgcolor="#fff",
    legend=dict(title="Borough", bgcolor="rgba(255,255,255,0.8)"),
    font=dict(family="Segoe UI, sans-serif"),
)
fig_sc.write_html(str(VIZ / "tips_vs_danger_scatter.html"),
                  include_plotlyjs="cdn", full_html=True)
print("Saved tips_vs_danger_scatter.html (normalized)")

# ── 8. Regenerate bucket bar chart (normalized) ───────────────────────
bucket_all = bucket.copy()
fig_bar = go.Figure(go.Bar(
    x=bucket_all["danger_group"].tolist(),
    y=bucket_all["mean_tip_pct"].round(2),
    marker_color=["#4a9f5c","#f4a261","#e76f51","#c1121f"],
    text=bucket_all["mean_tip_pct"].round(2).astype(str) + "%",
    textposition="outside",
    customdata=bucket_all[["n_zones","total_trips"]].values,
    hovertemplate=(
        "<b>%{x}</b><br>Avg tip: %{y:.2f}%<br>"
        "Zones: %{customdata[0]}<br>Trips: %{customdata[1]:,}<extra></extra>"
    ),
))
fig_bar.update_layout(
    title={"text": "Average Tip % by Crime-Density Quartile (normalized by zone area)",
           "font": {"size": 14, "family": "Georgia"}, "x": 0.5},
    xaxis=dict(title="Danger density quartile"),
    yaxis=dict(title="Average tip %", ticksuffix="%", range=[0, 18]),
    height=460,
    plot_bgcolor="#fafafa",
    paper_bgcolor="#fff",
    font=dict(family="Segoe UI, sans-serif"),
)
fig_bar.write_html(str(VIZ / "tip_by_danger_group.html"),
                   include_plotlyjs="cdn", full_html=True)
print("Saved tip_by_danger_group.html (normalized)")

# ── Print new bucket numbers for index.html update ────────────────────
print("\nBucket table values for index.html:")
for _, row in bucket.iterrows():
    print(f"  {row['danger_group']}: {row['n_zones']} zones, "
          f"{row['total_trips']:,.0f} trips, {row['mean_tip_pct']:.2f}%")

# Spearman r values for inline text
sp_pickup = results[0]["spearman"]
sp_night  = results[3]["spearman"]
print(f"\nSpearman r (pickup, normalized): {sp_pickup:.3f}")
print(f"Spearman r (night,  normalized): {sp_night:.3f}")
