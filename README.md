# HANDSON-L10 — Spark Structured Streaming + MLlib

Two self-contained tasks for the ITCS6190/8190 hands-on:

- **Task 4:** Real-time fare prediction from `distance_km` using Spark MLlib (Linear Regression)
- **Task 5:** Time-based fare trend prediction using 5-minute windows (sliding every 1 minute)

Minimal code footprint: **`task4.py`**, **`task5.py`**, and an  **`data_generator.py`** that streams JSON over a TCP socket.

---

## Prerequisites

- Python 3.10+ (Codespaces or local)
- Java/JDK available (Codespaces has it preinstalled)
- Install packages:
  ```bash
  pip install pyspark Faker  # Faker is optional; only needed if your generator imports it

## Repository layout
HANDSON-L10/
├─ task4.py                   # Distance → fare (real-time inference + training-once)
├─ task5.py                   # 5-min fare trend (windowed streaming + training-once)
├─ data_generator.py          # JSON socket server emitting ride events (optional)
├─ training-dataset.csv       # Offline training data for both tasks
└─ models/
   ├─ fare_model/             # Saved by task4.py on first run
   └─ fare_trend_model_v2/    # Saved by task5.py on first run

# Data contracts

Streaming JSON events (from ` data_generator.py `):

```
{
  "trip_id": "uuid",
  "driver_id": 42,
  "distance_km": 12.34,
  "fare_amount": 27.50,
  "timestamp": "YYYY-MM-DD HH:MM:SS"
}
```


Training CSV (training-dataset.csv):

For Task 4: columns distance_km, fare_amount (extra columns are ignored)

For Task 5: must include fare_amount and a time column:

preferred name: timestamp (string or timestamp)

if yours is ride_timestamp, update `task5.py` (see Troubleshooting).

# Quick start (two terminals)
1) Start the generator (Terminal A)
   ```
   python data_generator.py
   ```
3) Run Task 4
   ```
   python task4.py
   ```
5) Run Task 5
   ```
   python task5.py
   ```

# What each task does
`Task 4` — Real-time fare prediction

- Model: Linear Regression (`fare_amount` ~ `distance_km`)

- Features: `VectorAssembler(["distance_km"])`

- Stream: Parses JSON from socket, applies trained model, prints predictions and `|actual - predicted|` as `deviation`

- Model path: `models/fare_model` (auto-trained if missing; reads `training-dataset.csv`)

Task 5 — Time-based fare trend (5-min windows)

-Windowing:` window(event_time, "5 minutes", "1 minute")` with watermark (default `1 minute`)

- Aggregation: average fare per window → `avg_fare`

- Features: `hour_of_day`, `minute_of_hour` extracted from `window.start`

- Model: Linear Regression on these time features
- Model path: `models/fare_trend_model_v2 `(auto-trained if missing; reads `training-dataset.csv`)

# Screen Shots
- Task 4 Image
![Task 4 — Console Output](/task4_image.png)


- Task 5 Image
![Task 5 — Console Output](/task5_image.png)


   
