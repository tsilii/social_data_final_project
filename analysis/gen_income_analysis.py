"""
Fetch ACS 2023 median household income by census tract for NYC,
spatially join to taxi zones, correlate with tip percentages,
and generate new income-focused visualizations.
"""

import requests
import pandas as pd
import geopandas as gpd
import numpy as np
import json
import plotly.express as px
import plotly.graph_objects as go
from scipy import stats
from pathlib import Path
import io, zipfile, tempfile, os

ROOT = Path(__file__).parent.parent
VIZ  = ROOT / "visualizations"
DATA = Path(__file__).parent / "data"
PROC = DATA / "processed"

# ── 1. Fetch ACS 5-year 2023 median household income by tract ─────────────
print("Fetching ACS income data...")
counties = "061,047,081,005,085"   # Manhattan, Brooklyn, Queens, Bronx, Staten Island
url = (
    "https://api.census.gov/data/2023/acs/acs5"
    f"?get=B19013_001E,NAME&for=tract:*&in=state:36+county:{counties}"
)
resp = requests.get(url, timeout=60)
raw  = resp.json()

acs = pd.DataFrame(raw[1:], columns=raw[0])
acs["median_income"] = pd.to_numeric(acs["B19013_001E"], errors="coerce")
# -666666666 means suppressed / no data
acs.loc[acs["median_income"] < 0, "median_income"] = np.nan
# 11-digit GEOID: state(2) + county(3) + tract(6)
acs["GEOID"] = acs["state"] + acs["county"] + acs["tract"]
acs = acs[["GEOID", "median_income", "NAME"]].dropna(subset=["median_income"])
print(f"  Tracts with income data: {len(acs)}")
print(f"  Income range: ${acs['median_income'].min():,.0f} – ${acs['median_income'].max():,.0f}")

# ── 2. Download Census TIGER tract boundaries for New York State ──────────
print("Downloading census tract shapefile...")
tiger_url = "https://www2.census.gov/geo/tiger/TIGER2023/TRACT/tl_2023_36_tract.zip"
r2 = requests.get(tiger_url, timeout=120)
tmp = tempfile.mkdtemp()
zpath = os.path.join(tmp, "tracts.zip")
with open(zpath, "wb") as f:
    f.write(r2.content)
with zipfile.ZipFile(zpath, "r") as z:
    z.extractall(tmp)
shp = [os.path.join(tmp, x) for x in os.listdir(tmp) if x.endswith(".shp")][0]
tracts_gdf = gpd.read_file(shp).to_crs(epsg=4326)
# filter to NYC counties only
nyc_fips = {"061", "047", "081", "005", "085"}
tracts_gdf = tracts_gdf[tracts_gdf["COUNTYFP"].isin(nyc_fips)].copy()
tracts_gdf["GEOID"] = tracts_gdf["GEOID"].astype(str).str.strip()
print(f"  NYC tracts in shapefile: {len(tracts_gdf)}")

# merge income onto tract geometries
tracts_income = tracts_gdf.merge(acs[["GEOID","median_income"]], on="GEOID", how="left")
tracts_income = tracts_income.dropna(subset=["median_income"])
print(f"  Tracts with geometry+income: {len(tracts_income)}")

# ── 3. Load taxi zone shapefile ───────────────────────────────────────────
shp_path = DATA / "taxi_zones_shp" / "taxi_zones" / "taxi_zones.shp"
zones_gdf = gpd.read_file(shp_path).to_crs(epsg=4326)
zones_gdf["LocationID"] = zones_gdf["LocationID"].astype(int)

# ── 4. Spatial join: for each taxi zone compute area-weighted median income
print("Computing area-weighted income per taxi zone...")

# reproject to a metric CRS for area calculation
zones_m  = zones_gdf.to_crs(epsg=2263)
tracts_m = tracts_income.to_crs(epsg=2263)

# intersect taxi zones with income tracts
intersected = gpd.overlay(zones_m, tracts_m[["geometry","median_income"]], how="intersection")
intersected["area_int"] = intersected.geometry.area

# area-weighted average income per zone
def wavg(group):
    total = group["area_int"].sum()
    if total == 0:
        return np.nan
    return (group["median_income"] * group["area_int"]).sum() / total

zone_income = (
    intersected.groupby("LocationID")
    .apply(wavg)
    .reset_index(name="median_income")
)
print(f"  Zones with income estimate: {len(zone_income)}")
print(f"  Income range: ${zone_income['median_income'].min():,.0f} – ${zone_income['median_income'].max():,.0f}")

zone_income.to_csv(PROC / "zone_income.csv", index=False)
print("  Saved zone_income.csv")

# ── 5. Merge with pickup tip summary ─────────────────────────────────────
pickup = pd.read_csv(PROC / "taxi_zone_tip_summary_pickup.csv")
pickup["LocationID"] = pickup["LocationID"].astype(int)

df = pickup.merge(zone_income, on="LocationID", how="left")
df_valid = df[df["trip_count"] >= 50].dropna(subset=["median_income","mean_tip_pct"])
print(f"\nZones for correlation (n≥50 trips, income known): {len(df_valid)}")

# ── 6. Correlations ───────────────────────────────────────────────────────
r_sp, p_sp = stats.spearmanr(df_valid["median_income"], df_valid["mean_tip_pct"])
r_pe, p_pe = stats.pearsonr(df_valid["median_income"], df_valid["mean_tip_pct"])
print(f"\nIncome vs Avg Tip %:")
print(f"  Spearman r = {r_sp:.3f}, p = {p_sp:.4e}")
print(f"  Pearson  r = {r_pe:.3f}, p = {p_pe:.4e}")

# also check median tip
r_sp2, p_sp2 = stats.spearmanr(df_valid["median_income"], df_valid["median_tip_pct"])
print(f"\nIncome vs Median Tip %:")
print(f"  Spearman r = {r_sp2:.3f}, p = {p_sp2:.4e}")

# ── 7. Income quintile bucket analysis ───────────────────────────────────
df_valid2 = df_valid.copy()
df_valid2["income_group"] = pd.qcut(
    df_valid2["median_income"], 5,
    labels=["Q1 Lowest", "Q2", "Q3", "Q4", "Q5 Highest"]
)
bucket = (df_valid2.groupby("income_group", observed=True)
          .agg(
              n_zones   = ("LocationID","count"),
              total_trips = ("trip_count","sum"),
              mean_tip  = ("mean_tip_pct","mean"),
              income_mid= ("median_income","median"),
          )
          .reset_index())
print("\nTip % by income quintile:")
print(bucket.round(2).to_string(index=False))
bucket.to_csv(PROC / "income_bucket_stats.csv", index=False)

# ── 8. Visualization A — Income choropleth map ────────────────────────────
print("\nGenerating income choropleth map...")
map_df = zones_gdf.merge(
    df[["LocationID","zone","borough","median_income","mean_tip_pct","trip_count"]],
    on="LocationID", how="left", suffixes=("_shp", "")
)
# after merge: shapefile cols get _shp suffix, df cols keep their name
map_df["income_disp"] = (map_df["median_income"] / 1000).round(1)
map_df["tip_disp"]    = map_df["mean_tip_pct"].round(1)
geojson = json.loads(map_df.to_json())

fig_inc = px.choropleth_mapbox(
    map_df,
    geojson=geojson,
    locations=map_df.index,
    color="median_income",
    color_continuous_scale=[
        [0.0, "#d62828"],
        [0.25, "#f77f00"],
        [0.5,  "#fcbf49"],
        [0.75, "#4a9f5c"],
        [1.0,  "#1a5276"],
    ],
    range_color=[0, map_df["median_income"].quantile(0.97)],
    hover_data={
        "zone":        True,
        "borough":     True,
        "income_disp": True,
        "tip_disp":    True,
        "trip_count":  True,
    },
    labels={
        "median_income": "Median income ($)",
        "income_disp":   "Median income ($k)",
        "tip_disp":      "Avg tip %",
        "zone":          "Zone",
        "borough":       "Borough",
        "trip_count":    "Trips",
    },
    mapbox_style="carto-positron",
    zoom=9.8,
    center={"lat": 40.71, "lon": -73.94},
    opacity=0.78,
)
fig_inc.update_layout(
    margin={"r":0,"t":40,"l":0,"b":0},
    height=560,
    title={"text": "Median Household Income by NYC Taxi Zone (ACS 2023)",
           "font": {"size": 14, "family": "Georgia"}, "x": 0.5},
    coloraxis_colorbar=dict(title="Income ($)", tickprefix="$", thickness=14, len=0.55),
)
fig_inc.write_html(str(VIZ / "income_map.html"), include_plotlyjs="cdn", full_html=True)
print("  Saved income_map.html")

# ── 9. Visualization B — Income vs Tip scatter ────────────────────────────
print("Generating income vs tip scatter...")
borough_colors = {
    "Manhattan":    "#e63946",
    "Brooklyn":     "#457b9d",
    "Queens":       "#2a9d8f",
    "Bronx":        "#e9c46a",
    "Staten Island":"#8338ec",
    "EWR":          "#aaa",
}
sc = df_valid.copy()
sc["income_k"] = sc["median_income"] / 1000

fig_sc = go.Figure()
for bor, grp in sc.groupby("borough"):
    fig_sc.add_trace(go.Scatter(
        x=grp["income_k"],
        y=grp["mean_tip_pct"],
        mode="markers",
        name=bor,
        marker=dict(
            size=np.clip(np.sqrt(grp["trip_count"] / 500), 5, 30),
            color=borough_colors.get(bor, "#aaa"),
            opacity=0.75,
            line=dict(width=0.5, color="#fff"),
        ),
        text=grp["zone"],
        customdata=grp[["trip_count","median_income","mean_tip_pct"]].values,
        hovertemplate=(
            "<b>%{text}</b><br>"
            "Median income: $%{customdata[1]:,.0f}<br>"
            "Avg tip: %{customdata[2]:.1f}%<br>"
            "Trips: %{customdata[0]:,}<extra></extra>"
        ),
    ))

# OLS trend line
x_vals = sc["income_k"].values
y_vals = sc["mean_tip_pct"].values
m, b_ols = np.polyfit(x_vals, y_vals, 1)
x_line = np.linspace(x_vals.min(), x_vals.max(), 100)
fig_sc.add_trace(go.Scatter(
    x=x_line, y=m * x_line + b_ols,
    mode="lines", name="OLS trend",
    line=dict(color="#333", width=1.5, dash="dash"),
    showlegend=True,
))

# annotation for r value
fig_sc.add_annotation(
    x=0.97, y=0.05, xref="paper", yref="paper",
    text=f"Spearman r = {r_sp:.2f}  (p < 0.001)",
    showarrow=False, bgcolor="rgba(255,255,255,0.85)",
    bordercolor="#ccc", borderwidth=1,
    font=dict(size=12, family="Segoe UI"),
    align="right",
)

fig_sc.update_layout(
    title={"text": "Avg Tip % vs Neighbourhood Median Income (by Taxi Zone, 2023)",
           "font": {"size": 14, "family": "Georgia"}, "x": 0.5},
    xaxis=dict(title="Median household income ($k)", tickprefix="$", ticksuffix="k", gridcolor="#eee"),
    yaxis=dict(title="Average tip %", ticksuffix="%", gridcolor="#eee"),
    height=540,
    plot_bgcolor="#fafafa",
    paper_bgcolor="#fff",
    legend=dict(title="Borough", bgcolor="rgba(255,255,255,0.8)"),
    font=dict(family="Segoe UI, sans-serif"),
)
fig_sc.write_html(str(VIZ / "income_vs_tips_scatter.html"), include_plotlyjs="cdn", full_html=True)
print("  Saved income_vs_tips_scatter.html")

# ── 10. Visualization C — Income quintile bar chart ───────────────────────
print("Generating income quintile bar chart...")
fig_bar = go.Figure(go.Bar(
    x=bucket["income_group"].tolist(),
    y=bucket["mean_tip"].round(2),
    marker_color=["#d62828","#f77f00","#fcbf49","#4a9f5c","#1a5276"],
    text=bucket["mean_tip"].round(1).astype(str) + "%",
    textposition="outside",
    customdata=bucket[["n_zones","total_trips","income_mid"]].values,
    hovertemplate=(
        "<b>%{x}</b><br>Avg tip: %{y:.2f}%<br>"
        "Zones: %{customdata[0]}<br>Trips: %{customdata[1]:,}<br>"
        "Median income: $%{customdata[2]:,.0f}<extra></extra>"
    ),
))
fig_bar.update_layout(
    title={"text": "Average Tip % by Neighbourhood Income Quintile (NYC Taxi Zones, 2023)",
           "font": {"size": 14, "family": "Georgia"}, "x": 0.5},
    xaxis=dict(title="Income quintile (Q1 = lowest income neighbourhoods)"),
    yaxis=dict(title="Average tip %", ticksuffix="%", range=[0, 30]),
    height=460,
    plot_bgcolor="#fafafa",
    paper_bgcolor="#fff",
    font=dict(family="Segoe UI, sans-serif"),
)
fig_bar.write_html(str(VIZ / "tip_by_income_group.html"), include_plotlyjs="cdn", full_html=True)
print("  Saved tip_by_income_group.html")

# ── 11. Print summary for index.html update ───────────────────────────────
print("\n=== SUMMARY FOR index.html ===")
print(f"Spearman r (income vs mean_tip_pct) = {r_sp:.3f}, p = {p_sp:.2e}")
print(f"Pearson  r (income vs mean_tip_pct) = {r_pe:.3f}")
print("\nBucket table:")
for _, row in bucket.iterrows():
    print(f"  {row['income_group']}: {row['n_zones']} zones, "
          f"{row['total_trips']:,.0f} trips, {row['mean_tip']:.2f}%, "
          f"med income ${row['income_mid']:,.0f}")

top5 = df_valid.nlargest(5, "mean_tip_pct")[["zone","borough","mean_tip_pct","median_income"]]
bot5 = df_valid.nsmallest(5, "mean_tip_pct")[["zone","borough","mean_tip_pct","median_income"]]
print("\nTop 5 tip zones:")
print(top5.to_string(index=False))
print("\nBottom 5 tip zones:")
print(bot5.to_string(index=False))
