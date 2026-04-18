# NYC Taxi Tipping Analysis — Notebook Summary

## The Question

Do NYC taxi passengers tip more when conditions are uncomfortable — bad weather or late at night? The hypothesis is behavioral: people may tip out of *sympathy or guilt* rather than for service quality alone.

---

## The Approach

We needed two datasets that share a timestamp, then join them so every taxi trip knows what the weather was like at that moment.

```
NYC Taxi trips  ──┐
                  ├──► Joined on date_hour ──► Analysis
NYC Weather     ──┘
```

---

## Dataset 1: NYC Yellow Taxi (Socrata API)

**Endpoint:** `https://data.cityofnewyork.us/resource/4b4i-vvec.json`

**Key filter — credit card only:**
```python
"$where": "payment_type = '1'"
```
Cash trips record `$0` tip because cash tips are not tracked digitally. Including them would pull all averages down and distort the analysis entirely.

**Columns used:** `tpep_pickup_datetime`, `tip_amount`, `fare_amount`, `trip_distance`, `passenger_count`, `payment_type`

---

## Dataset 2: NYC Weather (Open-Meteo API)

**Endpoint:** `https://archive-api.open-meteo.com/v1/archive`

**Location:** Central Park (lat `40.7831`, lon `-73.9712`) — the standard NYC weather reference point.

**Critical parameter:**
```python
"timezone": "America/New_York"
```
Without this, times come back in UTC (5 hours ahead of NYC), making the join completely misaligned.

**Columns used:** `temperature_2m` (°C), `precipitation` (mm), `rain` (mm), `snowfall` (cm), `windspeed_10m` (km/h), `weathercode`

---

## Cleaning & Feature Engineering

### Type conversion
Everything from the API arrives as strings. We convert explicitly:
```python
df["tpep_pickup_datetime"] = pd.to_datetime(...)
df["tip_amount"] = pd.to_numeric(...)
```

### Outlier removal
```python
df = df[df["fare_amount"] >= 2.50]   # NYC minimum fare
df = df[df["trip_distance"] > 0]     # removes ghost/cancelled trips
```

### The core metric — tip percentage
```python
df["tip_pct"] = (df["tip_amount"] / df["fare_amount"]) * 100
df = df[df["tip_pct"] <= 100]
```
We use **percentage, not raw dollars**, because a $3 tip on a $10 fare (30%) is very different from a $3 tip on a $50 fare (6%). Percentage normalizes for trip length.

### Time features
```python
df["hour"] = df["tpep_pickup_datetime"].dt.hour
df["is_late_night"] = df["hour"].between(22, 23) | df["hour"].between(0, 4)
df["date_hour"] = df["tpep_pickup_datetime"].dt.floor("h")
```
- `is_late_night` uses two ranges because late night wraps around midnight
- `date_hour` floors every trip to its nearest hour — this is the **join key**

---

## The Join

```python
merged = df.merge(weather_df, on="date_hour", how="left")
```

Every taxi trip gets matched to the weather row for the same hour. A `left` join ensures no taxi trips are lost if a weather match is missing.

After the join, we add two binary flags — the core independent variables:
```python
merged["bad_weather"] = (merged["precipitation"] > 0) | (merged["snowfall"] > 0)
merged["is_late_night"] = merged["hour"].between(22, 23) | merged["hour"].between(0, 4)
```

---

## Results (January 2023 sample)

| Condition | Average Tip % |
|-----------|--------------|
| Good weather | ~25–26% |
| Bad weather | ~25–26% |
| Normal hours | ~25–26% |
| Late night | ~25–26% |

**The result is surprisingly flat.** Tip percentages barely change regardless of weather or time of day. This null result is itself interesting — it suggests passengers may not tip reactively to external conditions, or that the NYC taxi tipping culture has converged on a norm (~25%) that overrides situational sympathy.

---

## Next Steps (Assignment B)

- Expand from January 2023 sample to the full year
- Test additional weather variables (temperature bins, weathercode categories)
- Build the course website and explainer Jupyter notebook
- Investigate whether the flat result holds across seasons