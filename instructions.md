# Instructions for Claude — NYC Dangerous Areas & Taxi Tips Story

## Project Goal

We need to build a **GitHub Pages, magazine-style data story** about **dangerous areas in New York City in 2023**, using two datasets:

1. `NYPD_2023_Crimes_Only.csv`
   - Contains NYPD crime records for 2023.
   - Used to identify and explain dangerous areas in NYC.

2. Yellow taxi trip data, all `.parquet` files from:
   - `yellow_tripdata_2023-01.parquet`
   - ...
   - `yellow_tripdata_2023-12.parquet`
   - Used to study whether **taxi tips are related to the dangerousness of the pickup/dropoff area**.

The final deliverable should be a **magazine-style narrative website** hosted on **GitHub Pages**, combining text, visualizations, maps, and interpretation.

---

## Main Research Question

> Are taxi tips in NYC related to the dangerousness of the area where the trip starts or ends?

More specifically:

- Which areas of NYC were the most dangerous in 2023 according to NYPD crime data?
- Are taxi passengers tipping more or less in dangerous areas?
- Is the relationship stronger for:
  - pickup zones?
  - dropoff zones?
  - nighttime trips?
  - violent crimes only?
  - specific boroughs?
- Can we tell a coherent and data-backed story from these two datasets?

---

## Important Notes Before Analysis

The goal is **not only to create visualizations**, but first to check whether the data supports a meaningful story.

Before building the website, perform an exploratory analysis to answer:

1. Do the dangerous areas detected from crime data make sense?
2. Is there any visible relationship between danger and taxi tipping?
3. If there is no strong relationship, can we still tell an interesting story?
4. What is the most honest, data-supported narrative?

---

## Expected Dataset Structure

### Crime Dataset

Expected columns may include fields similar to:

- `CMPLNT_FR_DT`
- `CMPLNT_FR_TM`
- `BORO_NM`
- `OFNS_DESC`
- `LAW_CAT_CD`
- `Latitude`
- `Longitude`
- possibly precinct / jurisdiction fields

The exact column names should be checked in the file.

### Taxi Dataset

Yellow taxi parquet files usually include fields such as:

- `tpep_pickup_datetime`
- `tpep_dropoff_datetime`
- `PULocationID`
- `DOLocationID`
- `passenger_count`
- `trip_distance`
- `fare_amount`
- `tip_amount`
- `total_amount`
- `payment_type`

The exact schema should be checked from the parquet files.

Taxi zones require a taxi zone lookup file, usually named something like:

- `taxi_zone_lookup.csv`

If the lookup file is missing, mention that it is needed to map `PULocationID` and `DOLocationID` to boroughs/zones.

---

## Analysis Plan

## 1. Load and Inspect the Data

### Crime data

Load the NYPD crime CSV and inspect:

- shape
- column names
- missing values
- date/time range
- available location fields
- borough distribution
- offense categories
- felony/misdemeanor/violation counts

Check that the dataset really only contains crimes from 2023.

### Taxi data

Load all 12 parquet files, ideally efficiently because the taxi data is large.

For the taxi data:

- inspect schema
- check number of rows per month
- check missing values
- verify date range
- keep only trips from 2023
- remove invalid or suspicious records

Suggested filtering:

- `fare_amount > 0`
- `total_amount > 0`
- `tip_amount >= 0`
- `trip_distance > 0`
- `payment_type == 1` if we want tips to be meaningful, because credit card payments record tips more reliably than cash payments
- remove extreme outliers in `tip_amount`, `fare_amount`, `trip_distance`, and `tip_percentage`

Create:

```python
tip_percentage = tip_amount / fare_amount
```

or alternatively:

```python
tip_percentage = tip_amount / (total_amount - tip_amount)
```

Explain clearly which definition is used.

---

## 2. Define “Dangerous Areas”

We need to decide how to measure dangerousness.

Possible danger metrics:

### Basic metric

Crime count per area:

```text
danger_score = number of crimes in the area
```

### Better metric

Weighted crime severity score:

```text
danger_score = 
    3 * felony_count +
    2 * misdemeanor_count +
    1 * violation_count
```

If `LAW_CAT_CD` exists, use it.

### Violent crime metric

Create a subset of violent crimes using `OFNS_DESC`, for example categories containing:

- assault
- robbery
- murder
- rape
- felony assault
- dangerous weapons
- sex crimes

Then compute:

```text
violent_danger_score = violent crime count per area
```

### Important

Use at least two versions of dangerousness:

1. **All-crime danger score**
2. **Violent-crime danger score**

Then check whether the results are consistent.

---

## 3. Choose the Spatial Unit

The crime data has latitude/longitude, while taxi data has taxi zone IDs.

The best spatial unit is probably **NYC Taxi Zones**, because taxi trips already use `PULocationID` and `DOLocationID`.

Recommended approach:

1. Get the official NYC taxi zone shapefile or GeoJSON.
2. Convert the crime points into a GeoDataFrame.
3. Spatially join each crime to a taxi zone polygon.
4. Aggregate crimes by taxi zone.
5. Join danger scores to taxi pickup and dropoff zones.

Expected output:

One table per taxi zone with:

- `LocationID`
- `zone`
- `borough`
- `crime_count`
- `felony_count`
- `misdemeanor_count`
- `violation_count`
- `violent_crime_count`
- `danger_score`
- `violent_danger_score`

Then for taxi data, join:

- pickup danger score using `PULocationID`
- dropoff danger score using `DOLocationID`

---

## 4. Taxi Tip Analysis

After adding danger scores to taxi trips, compute summary statistics.

### Pickup-zone analysis

Group by pickup zone:

- number of trips
- average tip amount
- median tip amount
- average tip percentage
- median tip percentage
- average danger score
- violent danger score

### Dropoff-zone analysis

Same as above, but using dropoff zone.

### Time-based analysis

Create time variables:

```python
hour = pickup_datetime.dt.hour
day_of_week = pickup_datetime.dt.day_name()
month = pickup_datetime.dt.month
is_night = hour >= 20 or hour <= 5
```

Then compare:

- tips in dangerous vs safer zones
- daytime vs nighttime
- weekday vs weekend
- pickup danger vs dropoff danger

---

## 5. Statistical Checks

The story should not rely only on maps. We need simple quantitative checks.

### Correlation

Calculate correlations between danger and tips:

- Pearson correlation
- Spearman correlation

Use:

- `danger_score` vs `mean_tip_percentage`
- `violent_danger_score` vs `mean_tip_percentage`
- `danger_score` vs `median_tip_percentage`
- `violent_danger_score` vs `median_tip_percentage`

Do this separately for:

- pickup zones
- dropoff zones
- nighttime trips
- daytime trips

### Bucket analysis

Divide zones into danger groups:

```text
Low danger
Medium danger
High danger
Very high danger
```

For example using quartiles.

Then compare:

- average tip percentage by danger group
- median tip percentage by danger group
- number of trips by danger group

This is useful for storytelling because it is easier to explain than raw correlations.

### Regression, optional but useful

Run a simple regression:

```text
tip_percentage ~ pickup_danger_score + trip_distance + fare_amount + passenger_count + hour + borough
```

And another one:

```text
tip_percentage ~ dropoff_danger_score + trip_distance + fare_amount + passenger_count + hour + borough
```

The goal is not to create a complex econometric model, but to check whether danger still has any visible relationship with tipping after controlling for obvious trip factors.

---

## 6. Visualizations to Produce

The final GitHub Pages story should probably include 3–5 strong visualizations.

Recommended visualizations:

### Visualization 1 — Map of Crime Hotspots

A choropleth map of NYC taxi zones colored by:

- all-crime danger score
- or violent-crime danger score

Purpose:

> Show where the dangerous areas are.

### Visualization 2 — Crime Type Breakdown

Bar chart showing the most common crime categories in the most dangerous zones.

Purpose:

> Explain why these areas are classified as dangerous.

### Visualization 3 — Tips vs Danger

Scatter plot:

- x-axis: danger score per taxi zone
- y-axis: average or median tip percentage
- point size: number of taxi trips
- color: borough

Purpose:

> Show whether dangerous areas are associated with different tipping behavior.

### Visualization 4 — Tip Percentage by Danger Group

Bar chart or box plot:

- x-axis: danger group
- y-axis: tip percentage

Purpose:

> Make the relationship easier to interpret.

### Visualization 5 — Nighttime Analysis

Compare tip behavior in dangerous vs safer areas during night hours.

Possible options:

- line chart by hour
- grouped bar chart: danger group × day/night
- heatmap: hour × danger group

Purpose:

> Taxi tipping behavior may be more strongly connected to perceived risk at night.

---

## 7. Questions the Analysis Must Answer

After the exploratory analysis, write a clear summary answering:

### Dangerous areas

- Which NYC taxi zones are most dangerous?
- Are they concentrated in specific boroughs?
- Are the results different for all crimes vs violent crimes?

### Taxi tips

- Do taxis in dangerous pickup zones receive higher or lower tips?
- Do taxis going to dangerous dropoff zones receive higher or lower tips?
- Is the relationship stronger at night?
- Is the relationship actually meaningful, or very weak?

### Story validity

Answer honestly:

- Does the story make sense?
- Is the relationship strong enough to be the main narrative?
- If the relationship is weak, what alternative story should we tell?

---

## 8. Possible Story Angles

Depending on what the data shows, choose one of the following narratives.

### Story A — Risk Premium

If tips are higher in dangerous zones:

> Passengers may tip more when trips involve areas perceived as risky, especially at night.

Be careful: this is only an interpretation, not proof of causality.

### Story B — Economic Divide

If tips are lower in dangerous zones:

> Areas with higher crime may also be areas with lower income or different trip patterns, and tips may reflect broader socioeconomic differences rather than crime itself.

### Story C — No Strong Relationship

If correlation is weak:

> The city’s crime geography and taxi tipping geography do not strongly overlap. Taxi tips seem to be driven more by fare amount, trip distance, payment type, and tourist/business zones than by crime levels.

This is still a valid and honest story.

### Story D — Pickup vs Dropoff Difference

If pickup danger and dropoff danger behave differently:

> Where a trip starts and where it ends may shape tipping differently. Pickup zones may reflect passenger profile, while dropoff zones may reflect destination risk or convenience.

### Story E — Nighttime City

If the relationship appears mainly at night:

> Danger is not just spatial; it is temporal. The relationship between crime and tipping becomes more visible after dark.

---

## 9. Recommended Final Website Structure

The GitHub Pages site should read like a magazine article.

Suggested structure:

### Title

Example:

> Tips in the Danger Zone: What NYC Taxi Data Reveals About Crime and Generosity

### Subtitle

Example:

> Using NYPD crime reports and 2023 yellow taxi trips, we explore whether taxi tipping patterns change across New York’s most dangerous areas.

### Section 1 — Introduction

Explain:

- why NYC is interesting
- why crime and taxi data can be connected
- main research question

### Section 2 — Mapping Danger

Show the crime hotspot map.

Explain:

- how dangerousness was measured
- which zones are most dangerous
- what types of crime dominate

### Section 3 — Taxi Tips Enter the Story

Explain:

- how taxi tip percentage was calculated
- why credit card trips may be more reliable
- how taxi trips were connected to crime zones

### Section 4 — Do Tips Change in Dangerous Areas?

Show:

- scatter plot tips vs danger
- danger bucket chart

Explain the observed relationship.

### Section 5 — The Night Effect

Show a night/day or hourly comparison.

Explain whether the relationship changes after dark.

### Section 6 — What We Can and Cannot Conclude

Be explicit:

- This analysis shows association, not causation.
- Crime reports do not perfectly measure perceived danger.
- Taxi data mostly reflects yellow taxi activity, not all mobility.
- Recorded tips are more reliable for card payments than cash payments.
- High-crime areas may also differ economically, socially, and geographically.

### Section 7 — Conclusion

End with the main takeaway.

Possible ending depending on results:

```text
The data suggests that taxi tipping is only weakly connected to crime levels. New York’s danger map and tipping map overlap in some places, but tips appear to be shaped more by trip economics and city geography than by crime alone.
```

or:

```text
The data suggests a small but visible tipping difference in high-crime zones, especially at night. While this does not prove that passengers reward drivers for risk, it reveals how urban danger, mobility, and everyday economic behavior can intersect.
```

---

## 10. Technical Implementation Suggestions

Use Python for data analysis.

Recommended libraries:

```python
import pandas as pd
import geopandas as gpd
import numpy as np
import matplotlib.pyplot as plt
import plotly.express as px
import folium
import pyarrow.parquet as pq
```

For large taxi data:

- read only necessary columns
- process month by month
- aggregate before merging if memory is a problem
- save intermediate cleaned data as parquet

Suggested taxi columns to keep:

```python
cols = [
    "tpep_pickup_datetime",
    "tpep_dropoff_datetime",
    "PULocationID",
    "DOLocationID",
    "passenger_count",
    "trip_distance",
    "fare_amount",
    "tip_amount",
    "total_amount",
    "payment_type"
]
```

---

## 11. Suggested Intermediate Files

Create these intermediate files:

```text
data/processed/crimes_with_taxi_zones.parquet
data/processed/zone_danger_scores.csv
data/processed/taxi_zone_tip_summary_pickup.csv
data/processed/taxi_zone_tip_summary_dropoff.csv
data/processed/taxi_trips_cleaned_sample.parquet
```

Create these visualization files:

```text
visualizations/crime_hotspot_map.html
visualizations/tips_vs_danger_scatter.html
visualizations/tip_by_danger_group.html
visualizations/nighttime_tips_analysis.html
```

---

## 12. What Claude Should Return First

Before building the GitHub Pages website, first return an **analysis report** with:

1. Dataset inspection summary
2. Cleaning decisions
3. Definition of dangerousness
4. Top 10 dangerous taxi zones
5. Correlation table between danger and tips
6. Danger bucket comparison table
7. Initial visualizations
8. A recommendation about whether the story is strong, weak, or needs adjustment
9. The best final narrative angle

Do **not** jump directly to the final website before checking whether the data supports the story.

---

## 13. Final Output Requirements

Eventually, the final project should include:

```text
index.html
style.css
visualizations/
images/
data/processed/
README.md
```

The `index.html` should be suitable for GitHub Pages.

The tone should be:

- clear
- narrative
- data-driven
- magazine-style
- honest about limitations

The final story should not overclaim causality.

Use wording like:

```text
associated with
related to
suggests
appears to
is consistent with
```

Avoid wording like:

```text
crime causes tips
danger makes people tip more
taxi passengers definitely behave this way because of crime
```

---

## 14. Core Principle

The analysis should first discover what the data really says.

The final story must be built around the evidence, not forced into the original hypothesis.



For joining the two datasets:
The Problem

Taxi dataset uses PULocationID / DOLocationID — these are NYC Taxi Zone IDs (numbered polygons/zones)
Crime dataset uses raw Latitude / Longitude coordinates

The Bridge: NYC Taxi Zone Shapefile
The official solution is to use the NYC Taxi Zone shapefile (publicly available from NYC Open Data), which contains the polygon boundaries for each zone ID. Then:

Load the taxi zone shapefile (polygons mapped to zone IDs)
Spatially join each crime's lat/lon point → find which taxi zone polygon it falls inside
This gives every crime a LocationID (same as PULocationID/DOLocationID)
Now you can join the two datasets on that ID

Step 1 — Download the NYC Taxi Zone shapefile from NYC Open Data. It maps each LocationID (1–263) to a geographic polygon boundary.
Step 2 — Spatial join: crimes → zones. Use geopandas to do a point-in-polygon test. Each crime's lat/lon becomes a point geometry, and we check which zone polygon it falls inside. This assigns a LocationID to every crime.
Step 3 — Join the two datasets on LocationID — matching taxi pickups/dropoffs in the same zone as each crime.
One thing to note: the crime coordinates use commas as decimal separators (40,608211) instead of dots, so we'll need to clean those up first. 