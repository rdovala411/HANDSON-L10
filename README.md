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