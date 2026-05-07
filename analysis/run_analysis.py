"""
NYC Dangerous Areas & Taxi Tips — Full Analysis
Steps:
  1. Load crime CSV, fix decimal separators
  2. Download NYC taxi zone GeoJSON
  3. Spatially join crimes → taxi zones
  4. Compute danger scores per zone
  5. Load all 12 taxi parquet files (credit-card trips only)
  6. Compute tip_pct, join danger scores
  7. Statistical analysis (correlations, bucket comparison)
  8. Create all visualizations (Plotly interactive HTML)
  9. Save processed data
"""

import os, sys, json, warnings
import numpy as np
import pandas as pd
import geopandas as gpd
import requests
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from scipy import stats
from pathlib import Path
import glob

warnings.filterwarnings("ignore")

# ── paths ───────────────────────────────────────────────────────────────────
BASE = Path(__file__).parent
DATA = BASE / "data"
PROCESSED = DATA / "processed"
VIZ = BASE.parent / "visualizations"
PROCESSED.mkdir(parents=True, exist_ok=True)
VIZ.mkdir(parents=True, exist_ok=True)

print("=" * 60)
print("NYC DANGER & TAXI TIPS — ANALYSIS")
print("=" * 60)

# ── 1. LOAD CRIME DATA ───────────────────────────────────────────────────────
print("\n[1] Loading crime data...")

def fix_coord(x):
    """Convert comma-decimal '40,608211' → float 40.608211"""
    try:
        return float(str(x).replace(",", "."))
    except:
        return np.nan

crimes = pd.read_csv(DATA / "NYPD_2023_Crimes_Only.csv")
print(f"  Raw rows: {len(crimes):,}")
print(f"  Columns: {crimes.columns.tolist()}")
print(f"  LAW_CAT_CD counts:\n{crimes['LAW_CAT_CD'].value_counts()}")

crimes["lat"] = crimes["Latitude"].apply(fix_coord)
crimes["lon"] = crimes["Longitude"].apply(fix_coord)
crimes = crimes.dropna(subset=["lat", "lon"])
# sanity filter: NYC bounding box
crimes = crimes[
    (crimes["lat"].between(40.4, 41.0)) &
    (crimes["lon"].between(-74.3, -73.6))
]
print(f"  After cleaning: {len(crimes):,} crimes with valid coords")

# ── 2. LOAD NYC TAXI ZONE SHAPEFILE ─────────────────────────────────────────
print("\n[2] Loading NYC taxi zone shapefile...")
SHP_PATH = DATA / "taxi_zones_shp" / "taxi_zones" / "taxi_zones.shp"

if not SHP_PATH.exists():
    import zipfile, io
    url = "https://d37ci6vzurychx.cloudfront.net/misc/taxi_zones.zip"
    r = requests.get(url, timeout=60)
    with zipfile.ZipFile(io.BytesIO(r.content)) as z:
        z.extractall(DATA / "taxi_zones_shp")
    print(f"  Downloaded and extracted shapefile")
else:
    print(f"  Already cached at {SHP_PATH}")

zones = gpd.read_file(SHP_PATH)
print(f"  Zones: {len(zones)} features")
print(f"  Zone CRS: {zones.crs}")
print(f"  Columns: {zones.columns.tolist()}")

# normalise LocationID column
if "location_id" in zones.columns:
    zones = zones.rename(columns={"location_id": "LocationID"})
elif "LocationID" not in zones.columns:
    # try to find it
    for c in zones.columns:
        if "location" in c.lower() or "id" in c.lower():
            print(f"  Found ID column: {c}")
            zones = zones.rename(columns={c: "LocationID"})
            break
zones["LocationID"] = zones["LocationID"].astype(int)

# normalise borough / zone name columns
if "borough" not in zones.columns:
    for c in zones.columns:
        if "boro" in c.lower():
            zones = zones.rename(columns={c: "borough"})
            break
if "zone" not in zones.columns:
    for c in zones.columns:
        if "zone" in c.lower() or "name" in c.lower():
            zones = zones.rename(columns={c: "zone"})
            break

print(f"  Zone columns (normalised): {zones.columns.tolist()}")
print(f"  Sample:\n{zones[['LocationID','borough','zone']].head()}")

# reproject to WGS84 if needed
zones = zones.to_crs("EPSG:4326")

# ── 3. SPATIAL JOIN: crimes → taxi zones ─────────────────────────────────────
print("\n[3] Spatial join: crimes → taxi zones...")

crimes_geo = gpd.GeoDataFrame(
    crimes,
    geometry=gpd.points_from_xy(crimes["lon"], crimes["lat"]),
    crs="EPSG:4326",
)
crimes_joined = gpd.sjoin(
    crimes_geo[["LAW_CAT_CD", "geometry"]],
    zones[["LocationID", "borough", "zone", "geometry"]],
    how="left",
    predicate="within",
)
crimes_joined = crimes_joined.dropna(subset=["LocationID"])
crimes_joined["LocationID"] = crimes_joined["LocationID"].astype(int)
print(f"  Joined crimes: {len(crimes_joined):,}")
print(f"  Unmatched: {crimes_joined['LocationID'].isna().sum():,}")

# ── 4. DANGER SCORES PER ZONE ───────────────────────────────────────────────
print("\n[4] Computing danger scores...")

# pivot law categories
cat_counts = (
    crimes_joined.groupby(["LocationID", "LAW_CAT_CD"])
    .size()
    .unstack(fill_value=0)
    .reset_index()
)
# ensure columns exist
for col in ["FELONY", "MISDEMEANOR", "VIOLATION"]:
    if col not in cat_counts.columns:
        cat_counts[col] = 0

zone_danger = cat_counts.copy()
zone_danger["crime_count"] = (
    zone_danger.get("FELONY", 0)
    + zone_danger.get("MISDEMEANOR", 0)
    + zone_danger.get("VIOLATION", 0)
)
zone_danger["danger_score"] = (
    3 * zone_danger["FELONY"]
    + 2 * zone_danger["MISDEMEANOR"]
    + 1 * zone_danger["VIOLATION"]
)
# felony-only score
zone_danger["felony_score"] = zone_danger["FELONY"]

# attach zone names / boroughs
zone_meta = zones[["LocationID", "borough", "zone"]].copy()
zone_danger = zone_danger.merge(zone_meta, on="LocationID", how="left")

zone_danger = zone_danger.sort_values("danger_score", ascending=False)
print(f"  Zones with crime data: {len(zone_danger)}")
print("\n  Top 10 most dangerous zones:")
top10 = zone_danger.head(10)[["LocationID", "zone", "borough", "crime_count", "FELONY", "danger_score"]]
print(top10.to_string(index=False))

zone_danger.to_csv(PROCESSED / "zone_danger_scores.csv", index=False)
print(f"  Saved → {PROCESSED / 'zone_danger_scores.csv'}")

# ── 5. LOAD TAXI DATA ────────────────────────────────────────────────────────
print("\n[5] Loading taxi data (all 12 months, credit-card only)...")

COLS = [
    "tpep_pickup_datetime",
    "PULocationID",
    "DOLocationID",
    "passenger_count",
    "trip_distance",
    "fare_amount",
    "tip_amount",
    "total_amount",
    "payment_type",
]

parquet_files = sorted(glob.glob(str(DATA / "taxi" / "yellow_tripdata_2023-*.parquet")))
print(f"  Found {len(parquet_files)} parquet files")

monthly_dfs = []
for f in parquet_files:
    month_label = Path(f).stem.split("-")[-1]
    df_m = pd.read_parquet(f, columns=COLS)
    # filter inside each chunk for memory efficiency
    df_m = df_m[
        (df_m["tpep_pickup_datetime"] >= "2023-01-01")
        & (df_m["tpep_pickup_datetime"] < "2024-01-01")
        & (df_m["payment_type"] == 1)           # credit card
        & (df_m["fare_amount"] >= 2.50)
        & (df_m["trip_distance"] > 0)
        & (df_m["tip_amount"] >= 0)
        & (df_m["total_amount"] > 0)
        & (df_m["PULocationID"].between(1, 263))
        & (df_m["DOLocationID"].between(1, 263))
    ]
    monthly_dfs.append(df_m)
    print(f"  Month {month_label}: {len(df_m):,} rows")

taxi = pd.concat(monthly_dfs, ignore_index=True)
print(f"\n  Total rows after filtering: {len(taxi):,}")

# tip_pct = tip / fare * 100
taxi["tip_pct"] = taxi["tip_amount"] / taxi["fare_amount"] * 100
# remove extreme outliers: tip_pct > 100 or fare > 500
taxi = taxi[(taxi["tip_pct"] >= 0) & (taxi["tip_pct"] <= 100)]
taxi = taxi[taxi["fare_amount"] <= 500]
taxi = taxi[taxi["trip_distance"] <= 100]

# time features
taxi["hour"] = taxi["tpep_pickup_datetime"].dt.hour
taxi["day_of_week"] = taxi["tpep_pickup_datetime"].dt.day_name()
taxi["month"] = taxi["tpep_pickup_datetime"].dt.month
taxi["is_night"] = (taxi["hour"] >= 20) | (taxi["hour"] <= 5)

print(f"  Rows after outlier removal: {len(taxi):,}")
print(f"\n  Tip % summary:\n{taxi['tip_pct'].describe().round(2)}")

# ── 6. JOIN DANGER SCORES ────────────────────────────────────────────────────
print("\n[6] Joining danger scores to taxi trips...")

danger_lookup = zone_danger[["LocationID", "danger_score", "felony_score", "crime_count"]].copy()

# pickup danger
taxi = taxi.merge(
    danger_lookup.rename(columns={
        "LocationID": "PULocationID",
        "danger_score": "pu_danger",
        "felony_score": "pu_felony",
        "crime_count": "pu_crime_count",
    }),
    on="PULocationID", how="left"
)
# dropoff danger
taxi = taxi.merge(
    danger_lookup.rename(columns={
        "LocationID": "DOLocationID",
        "danger_score": "do_danger",
        "felony_score": "do_felony",
        "crime_count": "do_crime_count",
    }),
    on="DOLocationID", how="left"
)

# fill zones with no crime record as 0
for col in ["pu_danger", "pu_felony", "pu_crime_count", "do_danger", "do_felony", "do_crime_count"]:
    taxi[col] = taxi[col].fillna(0)

print(f"  Trips with PU danger > 0: {(taxi['pu_danger'] > 0).sum():,}")
print(f"  Trips with DO danger > 0: {(taxi['do_danger'] > 0).sum():,}")

# ── 7. ZONE-LEVEL SUMMARY ────────────────────────────────────────────────────
print("\n[7] Building zone-level summaries...")

def zone_summary(df, loc_col, danger_col, felony_col, label):
    grp = df.groupby(loc_col).agg(
        trip_count=(loc_col, "count"),
        mean_tip_pct=("tip_pct", "mean"),
        median_tip_pct=("tip_pct", "median"),
        mean_tip_amt=("tip_amount", "mean"),
        danger_score=(danger_col, "first"),
        felony_score=(felony_col, "first"),
    ).reset_index()
    grp = grp.rename(columns={loc_col: "LocationID"})
    grp = grp.merge(zone_meta, on="LocationID", how="left")
    grp = grp[grp["trip_count"] >= 50]  # min 50 trips for reliability
    grp.to_csv(PROCESSED / f"taxi_zone_tip_summary_{label}.csv", index=False)
    print(f"  {label.upper()}: {len(grp)} zones (≥50 trips)")
    return grp

pu_summary = zone_summary(taxi, "PULocationID", "pu_danger", "pu_felony", "pickup")
do_summary = zone_summary(taxi, "DOLocationID", "do_danger", "do_felony", "dropoff")

# ── 8. STATISTICAL ANALYSIS ──────────────────────────────────────────────────
print("\n[8] Statistical analysis...")

def correlations(summary, danger_col, label):
    df = summary.dropna(subset=[danger_col, "mean_tip_pct", "median_tip_pct"])
    df = df[df[danger_col] > 0]
    r_pearson_mean, p_pearson_mean = stats.pearsonr(df[danger_col], df["mean_tip_pct"])
    r_spearman_mean, p_spearman_mean = stats.spearmanr(df[danger_col], df["mean_tip_pct"])
    r_pearson_med, p_pearson_med = stats.pearsonr(df[danger_col], df["median_tip_pct"])
    r_spearman_med, p_spearman_med = stats.spearmanr(df[danger_col], df["median_tip_pct"])
    print(f"\n  {label} ({len(df)} zones with crime data):")
    print(f"    Pearson  (mean tip%):   r={r_pearson_mean:.3f}, p={p_pearson_mean:.4f}")
    print(f"    Spearman (mean tip%):   r={r_spearman_mean:.3f}, p={p_spearman_mean:.4f}")
    print(f"    Pearson  (median tip%): r={r_pearson_med:.3f}, p={p_pearson_med:.4f}")
    print(f"    Spearman (median tip%): r={r_spearman_med:.3f}, p={p_spearman_med:.4f}")
    return {
        "label": label,
        "n_zones": len(df),
        "pearson_mean": round(r_pearson_mean, 3),
        "spearman_mean": round(r_spearman_mean, 3),
        "pearson_median": round(r_pearson_med, 3),
        "spearman_median": round(r_spearman_med, 3),
        "p_spearman_mean": round(p_spearman_mean, 4),
    }

corr_results = []
corr_results.append(correlations(pu_summary, "danger_score", "Pickup zones (all crimes)"))
corr_results.append(correlations(pu_summary, "felony_score", "Pickup zones (felony only)"))
corr_results.append(correlations(do_summary, "danger_score", "Dropoff zones (all crimes)"))
corr_results.append(correlations(do_summary, "felony_score", "Dropoff zones (felony only)"))

# nighttime vs daytime
night = taxi[taxi["is_night"]]
day   = taxi[~taxi["is_night"]]
night_pu = zone_summary(night, "PULocationID", "pu_danger", "pu_felony", "pickup_night")
day_pu   = zone_summary(day,   "PULocationID", "pu_danger", "pu_felony", "pickup_day")
corr_results.append(correlations(night_pu, "danger_score", "Pickup zones — NIGHT"))
corr_results.append(correlations(day_pu,   "danger_score", "Pickup zones — DAY"))

corr_df = pd.DataFrame(corr_results)
corr_df.to_csv(PROCESSED / "correlation_results.csv", index=False)
print(f"\n  Correlation table saved → {PROCESSED / 'correlation_results.csv'}")

# ── bucket analysis ──
print("\n  Danger bucket analysis (pickup zones):")
pu_valid = pu_summary[pu_summary["danger_score"] > 0].copy()
pu_valid["danger_group"] = pd.qcut(
    pu_valid["danger_score"], q=4,
    labels=["Low", "Medium", "High", "Very High"]
)
bucket_stats = pu_valid.groupby("danger_group", observed=True).agg(
    n_zones=("LocationID", "count"),
    total_trips=("trip_count", "sum"),
    mean_tip_pct=("mean_tip_pct", "mean"),
    median_tip_pct=("median_tip_pct", "median"),
).reset_index()
print(bucket_stats.to_string(index=False))
bucket_stats.to_csv(PROCESSED / "danger_bucket_stats.csv", index=False)

# ── 9. VISUALIZATIONS ───────────────────────────────────────────────────────
print("\n[9] Creating visualizations...")

# colour palette
BOROUGH_COLORS = {
    "Manhattan": "#e63946",
    "Brooklyn": "#457b9d",
    "Queens": "#2a9d8f",
    "Bronx": "#e9c46a",
    "Staten Island": "#f4a261",
    "EWR": "#adb5bd",
}

# ------ VIZ 1: Choropleth map of crime hotspots ------
print("  Viz 1: Crime hotspot choropleth...")

zones_plot = zones.merge(
    zone_danger[["LocationID", "danger_score", "crime_count", "FELONY", "MISDEMEANOR", "VIOLATION"]],
    on="LocationID", how="left"
)
zones_plot["danger_score"] = zones_plot["danger_score"].fillna(0)

# convert to lat/lon for plotly
zones_plot = zones_plot.to_crs("EPSG:4326")
zones_geojson = json.loads(zones_plot.to_json())

fig_map = px.choropleth_mapbox(
    zones_plot,
    geojson=zones_geojson,
    locations=zones_plot.index,
    color="danger_score",
    color_continuous_scale="Reds",
    mapbox_style="carto-positron",
    zoom=9.5,
    center={"lat": 40.71, "lon": -73.94},
    opacity=0.75,
    hover_name="zone",
    hover_data={
        "borough": True,
        "crime_count": True,
        "FELONY": True,
        "MISDEMEANOR": True,
        "VIOLATION": True,
        "danger_score": True,
    },
    labels={
        "danger_score": "Danger Score",
        "crime_count": "Total Crimes",
        "FELONY": "Felonies",
        "MISDEMEANOR": "Misdemeanors",
        "VIOLATION": "Violations",
    },
    title="NYC Crime Danger Score by Taxi Zone (2023)<br><sup>Score = 3×Felonies + 2×Misdemeanors + 1×Violations</sup>",
)
fig_map.update_layout(
    margin={"r": 0, "t": 60, "l": 0, "b": 0},
    coloraxis_colorbar=dict(title="Danger Score"),
    font=dict(family="Georgia, serif", size=13),
)
fig_map.write_html(VIZ / "crime_hotspot_map.html")
print(f"    Saved → {VIZ / 'crime_hotspot_map.html'}")

# ------ VIZ 2: Tips vs Danger scatter ------
print("  Viz 2: Tips vs danger scatter...")

pu_plot = pu_summary.dropna(subset=["borough"]).copy()
pu_plot["borough"] = pu_plot["borough"].fillna("Unknown")

fig_scatter = px.scatter(
    pu_plot,
    x="danger_score",
    y="mean_tip_pct",
    size="trip_count",
    color="borough",
    color_discrete_map=BOROUGH_COLORS,
    hover_name="zone",
    hover_data={"trip_count": True, "danger_score": True, "mean_tip_pct": ":.2f"},
    labels={
        "danger_score": "Danger Score (pickup zone)",
        "mean_tip_pct": "Average Tip %",
        "trip_count": "# Trips",
        "borough": "Borough",
    },
    title="Taxi Tip % vs Zone Danger Score — Pickup Zones (2023)<br><sup>Bubble size = number of trips; credit-card trips only</sup>",
    size_max=40,
    opacity=0.75,
)
# add trend line
from numpy.polynomial.polynomial import polyfit
pu_valid2 = pu_plot[pu_plot["danger_score"] > 0].dropna(subset=["mean_tip_pct"])
if len(pu_valid2) > 5:
    x_vals = pu_valid2["danger_score"].values
    y_vals = pu_valid2["mean_tip_pct"].values
    coefs = np.polyfit(x_vals, y_vals, 1)
    x_range = np.linspace(x_vals.min(), x_vals.max(), 200)
    y_trend = np.polyval(coefs, x_range)
    fig_scatter.add_trace(go.Scatter(
        x=x_range, y=y_trend,
        mode="lines",
        line=dict(color="black", width=2, dash="dash"),
        name="Trend",
        showlegend=True,
    ))
fig_scatter.update_layout(
    font=dict(family="Georgia, serif", size=13),
    legend=dict(title="Borough"),
)
fig_scatter.write_html(VIZ / "tips_vs_danger_scatter.html")
print(f"    Saved → {VIZ / 'tips_vs_danger_scatter.html'}")

# ------ VIZ 3: Danger bucket bar chart ------
print("  Viz 3: Tip % by danger group...")

# add zero-crime group
zero_stats = pd.DataFrame({
    "danger_group": ["No Crime Data"],
    "n_zones": [len(pu_summary[pu_summary["danger_score"] == 0])],
    "total_trips": [pu_summary[pu_summary["danger_score"] == 0]["trip_count"].sum()],
    "mean_tip_pct": [pu_summary[pu_summary["danger_score"] == 0]["mean_tip_pct"].mean()],
    "median_tip_pct": [pu_summary[pu_summary["danger_score"] == 0]["median_tip_pct"].median()],
})
bucket_all = pd.concat([zero_stats, bucket_stats], ignore_index=True)
bucket_all["danger_group"] = bucket_all["danger_group"].astype(str)

ORDER = ["No Crime Data", "Low", "Medium", "High", "Very High"]
bucket_all["danger_group"] = pd.Categorical(bucket_all["danger_group"], categories=ORDER, ordered=True)
bucket_all = bucket_all.sort_values("danger_group")

COLORS = ["#adb5bd", "#a8dadc", "#457b9d", "#e9c46a", "#e63946"]

fig_bucket = go.Figure()
fig_bucket.add_trace(go.Bar(
    x=bucket_all["danger_group"].astype(str),
    y=bucket_all["mean_tip_pct"],
    name="Mean tip %",
    marker_color=COLORS,
    text=bucket_all["mean_tip_pct"].round(2).astype(str) + "%",
    textposition="outside",
    customdata=bucket_all[["n_zones", "total_trips", "median_tip_pct"]].values,
    hovertemplate=(
        "<b>%{x}</b><br>"
        "Mean tip: %{y:.2f}%<br>"
        "Zones: %{customdata[0]}<br>"
        "Trips: %{customdata[1]:,}<br>"
        "Median tip: %{customdata[2]:.2f}%<extra></extra>"
    ),
))
fig_bucket.update_layout(
    title="Average Taxi Tip % by Crime Danger Level — Pickup Zones (2023)<br><sup>Zones divided into quartiles by weighted danger score; includes $0-tip credit-card trips</sup>",
    xaxis_title="Danger Level",
    yaxis_title="Average Tip % (zone-level mean)",
    font=dict(family="Georgia, serif", size=13),
    showlegend=False,
)
fig_bucket.write_html(VIZ / "tip_by_danger_group.html")
print(f"    Saved → {VIZ / 'tip_by_danger_group.html'}")

# ------ VIZ 4: Nighttime analysis ------
print("  Viz 4: Nighttime analysis...")

hourly = taxi.groupby(["hour", "is_night"]).agg(
    mean_tip_pct=("tip_pct", "mean"),
    trip_count=("tip_pct", "count"),
).reset_index()

# Join danger quintile per pickup zone then aggregate by hour × danger group
pu_valid["danger_quintile"] = pd.qcut(
    pu_valid["danger_score"], q=4, labels=["Low", "Medium", "High", "Very High"]
)
danger_map = pu_valid.set_index("LocationID")["danger_quintile"].to_dict()
taxi["pu_danger_group"] = taxi["PULocationID"].map(danger_map)

night_bucket = (
    taxi[taxi["is_night"] & taxi["pu_danger_group"].notna()]
    .groupby("pu_danger_group", observed=True)["tip_pct"].mean()
)
day_bucket = (
    taxi[~taxi["is_night"] & taxi["pu_danger_group"].notna()]
    .groupby("pu_danger_group", observed=True)["tip_pct"].mean()
)

fig_night = make_subplots(
    rows=1, cols=2,
    subplot_titles=["Average Tip % by Hour", "Day vs Night in Danger Zones"],
)

# Left: tip by hour
fig_night.add_trace(go.Scatter(
    x=hourly["hour"], y=hourly["mean_tip_pct"],
    mode="lines+markers",
    line=dict(color="#457b9d", width=2),
    name="All trips",
    hovertemplate="Hour %{x}: %{y:.2f}%",
), row=1, col=1)
# shade night
fig_night.add_vrect(x0=20, x1=23, fillcolor="navy", opacity=0.1,
                    annotation_text="Night", row=1, col=1)
fig_night.add_vrect(x0=0, x1=5, fillcolor="navy", opacity=0.1, row=1, col=1)

# Right: day vs night by danger group
groups = ["Low", "Medium", "High", "Very High"]
fig_night.add_trace(go.Bar(
    x=groups,
    y=[day_bucket.get(g, np.nan) for g in groups],
    name="Daytime",
    marker_color="#457b9d",
), row=1, col=2)
fig_night.add_trace(go.Bar(
    x=groups,
    y=[night_bucket.get(g, np.nan) for g in groups],
    name="Night (20h–5h)",
    marker_color="#e63946",
), row=1, col=2)

fig_night.update_layout(
    title_text="Nighttime vs Daytime Tipping Patterns (2023 NYC Yellow Taxis)",
    barmode="group",
    font=dict(family="Georgia, serif", size=13),
    yaxis_title="Average Tip %",
    yaxis2_title="Average Tip %",
)
fig_night.write_html(VIZ / "nighttime_tips_analysis.html")
print(f"    Saved → {VIZ / 'nighttime_tips_analysis.html'}")

# ------ VIZ 5: Crime type breakdown for top danger zones ------
print("  Viz 5: Crime breakdown for top zones...")

top10_ids = zone_danger.head(10)["LocationID"].tolist()
top10_crimes = zone_danger[zone_danger["LocationID"].isin(top10_ids)][
    ["zone", "FELONY", "MISDEMEANOR", "VIOLATION", "danger_score"]
].sort_values("danger_score", ascending=True)

fig_crime_breakdown = go.Figure()
for cat, color in [("VIOLATION", "#a8dadc"), ("MISDEMEANOR", "#457b9d"), ("FELONY", "#e63946")]:
    fig_crime_breakdown.add_trace(go.Bar(
        y=top10_crimes["zone"],
        x=top10_crimes[cat],
        name=cat.title(),
        orientation="h",
        marker_color=color,
    ))

fig_crime_breakdown.update_layout(
    barmode="stack",
    title="Crime Type Breakdown — Top 10 Most Dangerous NYC Taxi Zones (2023)",
    xaxis_title="Number of Crimes",
    font=dict(family="Georgia, serif", size=13),
    legend=dict(title="Crime Category"),
)
fig_crime_breakdown.write_html(VIZ / "crime_type_breakdown.html")
print(f"    Saved → {VIZ / 'crime_type_breakdown.html'}")

# ── 10. PRINT FINAL SUMMARY ─────────────────────────────────────────────────
print("\n" + "=" * 60)
print("ANALYSIS SUMMARY")
print("=" * 60)

print(f"\nCrime data: {len(crimes):,} records")
print(f"Taxi trips: {len(taxi):,} (credit-card, 2023)")
print(f"\nTop 10 most dangerous zones:")
print(zone_danger.head(10)[["zone", "borough", "crime_count", "FELONY", "danger_score"]].to_string(index=False))

print(f"\nDanger bucket summary (pickup zones):")
print(bucket_stats.to_string(index=False))

print(f"\nCorrelation table:")
print(corr_df.to_string(index=False))

# determine narrative angle
mean_corr = corr_df[corr_df["label"].str.contains("Pickup.*all", na=False)]["spearman_mean"].values
narrative = "C"  # default: no strong relationship
if len(mean_corr) > 0:
    r = mean_corr[0]
    if abs(r) >= 0.3:
        narrative = "A" if r > 0 else "B"
    else:
        narrative = "C"

print(f"\n📊 Recommended narrative angle: STORY {narrative}")
if narrative == "A":
    print("  → Tips are HIGHER in dangerous zones (risk premium)")
elif narrative == "B":
    print("  → Tips are LOWER in dangerous zones (economic divide)")
else:
    print("  → Weak relationship; story = tips driven by trip economics, not crime geography")

# Save summary JSON for the website to read
summary = {
    "total_crimes": int(len(crimes)),
    "total_trips": int(len(taxi)),
    "top_dangerous_zones": zone_danger.head(10)[["LocationID", "zone", "borough", "danger_score", "crime_count"]].to_dict(orient="records"),
    "bucket_stats": bucket_stats.to_dict(orient="records"),
    "correlations": corr_df.to_dict(orient="records"),
    "narrative": narrative,
    "spearman_pu_allcrimes": float(corr_df[corr_df["label"].str.contains("Pickup.*all", na=False)]["spearman_mean"].iloc[0]) if len(corr_df[corr_df["label"].str.contains("Pickup.*all", na=False)]) > 0 else None,
}
(PROCESSED / "analysis_summary.json").write_text(json.dumps(summary, indent=2, default=str))
print(f"\n  Summary saved → {PROCESSED / 'analysis_summary.json'}")
print("\n✅ Analysis complete!")
