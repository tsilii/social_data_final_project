"""Generate tip-percentage choropleth map (new hero visualization)."""

import pandas as pd
import geopandas as gpd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from pathlib import Path

ROOT = Path(__file__).parent.parent
VIZ  = ROOT / "visualizations"
DATA = Path(__file__).parent / "data"

# ── Load shapefile ────────────────────────────────────────────────────────
shp_path = DATA / "taxi_zones_shp" / "taxi_zones" / "taxi_zones.shp"
zones_gdf = gpd.read_file(shp_path).to_crs(epsg=4326)
zones_gdf["LocationID"] = zones_gdf["LocationID"].astype(int)

# ── Load pickup tip summary ───────────────────────────────────────────────
pickup = pd.read_csv(DATA / "processed" / "taxi_zone_tip_summary_pickup.csv")
pickup["LocationID"] = pickup["LocationID"].astype(int)

# ── Load danger scores ────────────────────────────────────────────────────
danger = pd.read_csv(DATA / "processed" / "zone_danger_scores.csv")
danger["LocationID"] = danger["LocationID"].astype(int)

# ── Merge ─────────────────────────────────────────────────────────────────
merged = zones_gdf.merge(pickup[["LocationID","trip_count","mean_tip_pct",
                                  "median_tip_pct","mean_tip_amt","danger_score","borough","zone"]],
                         on="LocationID", how="left")

# ── Assign colours / tip tier ────────────────────────────────────────────
def tip_tier(v):
    if pd.isna(v): return "No data"
    if v >= 20:    return "Top earners (≥20%)"
    if v >= 15:    return "High (15–20%)"
    if v >= 10:    return "Moderate (10–15%)"
    return "Low (<10%)"

merged["tip_tier"] = merged["mean_tip_pct"].apply(tip_tier)
merged["mean_tip_pct_disp"] = merged["mean_tip_pct"].round(1)

# ── Build GeoJSON for Plotly ──────────────────────────────────────────────
import json
geojson = json.loads(merged.to_json())

# ── Choropleth — tip percentage ───────────────────────────────────────────
fig = px.choropleth_mapbox(
    merged,
    geojson=geojson,
    locations=merged.index,
    color="mean_tip_pct",
    color_continuous_scale=[
        [0.0,  "#d62828"],   # deep red  → low tips
        [0.35, "#f77f00"],   # orange
        [0.6,  "#fcbf49"],   # yellow
        [0.8,  "#4a9f5c"],   # green
        [1.0,  "#1a5276"],   # dark blue → highest tips
    ],
    range_color=[0, 28],
    hover_data={
        "zone_y":            True,
        "borough_y":         True,
        "mean_tip_pct_disp": True,
        "trip_count":        True,
        "danger_score":      True,
    },
    labels={
        "mean_tip_pct":      "Avg tip %",
        "mean_tip_pct_disp": "Avg tip %",
        "zone_y":            "Zone",
        "borough_y":         "Borough",
        "trip_count":        "Trips",
        "danger_score":      "Danger score",
    },
    mapbox_style="carto-positron",
    zoom=9.8,
    center={"lat": 40.71, "lon": -73.94},
    opacity=0.78,
)

fig.update_layout(
    margin={"r":0,"t":40,"l":0,"b":0},
    height=560,
    title={
        "text": "Average Taxi Tip % by NYC Zone (2023 · Credit-card trips)",
        "font": {"size": 15, "family": "Georgia"},
        "x": 0.5,
    },
    coloraxis_colorbar=dict(
        title="Avg tip %",
        ticksuffix="%",
        thickness=14,
        len=0.55,
    ),
)

out = VIZ / "tip_pct_map.html"
fig.write_html(str(out), include_plotlyjs="cdn", full_html=True)
print(f"Saved → {out}")

# ── Top / Bottom zone tables ──────────────────────────────────────────────
top10 = (pickup[pickup["trip_count"] >= 200]
         .sort_values("mean_tip_pct", ascending=False)
         .head(10)[["zone","borough","mean_tip_pct","trip_count"]])
bot10 = (pickup[pickup["trip_count"] >= 200]
         .sort_values("mean_tip_pct")
         .head(10)[["zone","borough","mean_tip_pct","trip_count"]])

print("\nTop 10 highest-tip pickup zones:")
print(top10.to_string(index=False))
print("\nBottom 10 lowest-tip pickup zones:")
print(bot10.to_string(index=False))

# ── Bar chart: top 15 vs bottom 15 zones ────────────────────────────────
top15 = (pickup[pickup["trip_count"] >= 500]
         .sort_values("mean_tip_pct", ascending=False)
         .head(15))
bot15 = (pickup[pickup["trip_count"] >= 500]
         .sort_values("mean_tip_pct")
         .head(15))

combined = pd.concat([top15, bot15]).drop_duplicates("LocationID")
combined["label"] = combined["zone"] + " (" + combined["borough"] + ")"
combined = combined.sort_values("mean_tip_pct")

colors = ["#e63946" if v < 10 else "#4a9f5c" if v >= 20 else "#f77f00"
          for v in combined["mean_tip_pct"]]

fig2 = go.Figure(go.Bar(
    x=combined["mean_tip_pct"].round(1),
    y=combined["label"],
    orientation="h",
    marker_color=colors,
    text=combined["mean_tip_pct"].round(1).astype(str) + "%",
    textposition="outside",
    hovertemplate="<b>%{y}</b><br>Avg tip: %{x:.1f}%<extra></extra>",
))

fig2.update_layout(
    title={
        "text": "Highest & Lowest Tip Zones — NYC Yellow Taxis 2023",
        "font": {"size": 14, "family": "Georgia"},
        "x": 0.5,
    },
    xaxis=dict(title="Average tip %", ticksuffix="%", range=[0, 35]),
    yaxis=dict(title="", tickfont=dict(size=11)),
    height=700,
    margin=dict(l=220, r=60, t=55, b=40),
    plot_bgcolor="#fafafa",
    paper_bgcolor="#fff",
    font=dict(family="Segoe UI, sans-serif"),
)

out2 = VIZ / "zone_tip_ranking.html"
fig2.write_html(str(out2), include_plotlyjs="cdn", full_html=True)
print(f"Saved → {out2}")

# ── Borough-level average tip % ──────────────────────────────────────────
bor = (pickup.groupby("borough")
       .apply(lambda x: np.average(x["mean_tip_pct"], weights=x["trip_count"]))
       .reset_index(name="weighted_avg_tip_pct"))
print("\nBorough weighted avg tip %:")
print(bor.sort_values("weighted_avg_tip_pct", ascending=False).to_string(index=False))
