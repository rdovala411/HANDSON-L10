
import os
from pyspark.sql import SparkSession
from pyspark.sql.functions import from_json, col, avg, window, hour, minute
from pyspark.sql.types import StructType, StructField, StringType, IntegerType, DoubleType, TimestampType

# TODO: Import VectorAssembler, LinearRegression, and LinearRegressionModel from pyspark.ml

# Initialize Spark Session
spark = SparkSession.builder.appName("Task7_FareTrendPrediction_Assignment").getOrCreate()
spark.sparkContext.setLogLevel("WARN")

# Paths
MODEL_PATH = "models/fare_trend_model_v2"
TRAINING_DATA_PATH = "training-dataset.csv"

# ------------------- MODEL TRAINING (Offline) ------------------- #
if not os.path.exists(MODEL_PATH):
    print(f"\n[Training Phase] Training new model with feature engineering using {TRAINING_DATA_PATH}...")

    # Load and process historical data
    hist_df_raw = spark.read.csv(TRAINING_DATA_PATH, header=True, inferSchema=False)
    hist_df_processed = hist_df_raw.withColumn("event_time", col("timestamp").cast(TimestampType())) \
                                   .withColumn("fare_amount", col("fare_amount").cast(DoubleType()))

    # TODO: Aggregate data into 5-minute time windows, calculating the average fare.
    # HINT: Use groupBy(window(...)) and agg(avg(...)).
    hist_windowed_df = None # Replace None with your implementation

    # TODO: Engineer time-based features from the window's start time.
    # Add two new columns: 'hour_of_day' and 'minute_of_hour'.
    # HINT: Use withColumn and the hour() and minute() functions on `col("window.start")`.
    hist_features = None # Replace None with your implementation

    # TODO: Create a VectorAssembler for the new time-based features.
    # Input columns should be "hour_of_day" and "minute_of_hour".
    assembler = None # Replace None with your implementation
    train_df = None # Replace None with your implementation

    # TODO: Create and train the LinearRegression model.
    # The label column is 'avg_fare'.
    lr = None # Replace None with your implementation
    model = None # Replace None with your implementation

    # TODO: Save the trained model.
    print(f"[Model Saved] -> {MODEL_PATH}")
else:
    print(f"[Model Found] Using existing model at {MODEL_PATH}")


# ------------------- STREAMING INFERENCE ------------------- #
print("\n[Inference Phase] Starting real-time trend prediction stream...")

# Define the schema for incoming data
schema = StructType([
    StructField("trip_id", StringType()),
    StructField("driver_id", IntegerType()),
    StructField("distance_km", DoubleType()),
    StructField("fare_amount", DoubleType()),
    StructField("timestamp", StringType())
])

# Read from socket and parse data
raw_stream = spark.readStream.format("socket").option("host", "localhost").option("port", 9999).load()
parsed_stream = raw_stream.select(from_json(col("value"), schema).alias("data")).select("data.*") \
    .withColumn("event_time", col("timestamp").cast(TimestampType()))

# Add a watermark to handle late-arriving data
parsed_stream = parsed_stream.withWatermark("event_time", "1 minute")

# TODO: Apply the same 5-minute windowed aggregation to the stream.
# The window should slide every 1 minute.
windowed_df = None # Replace None with your implementation

# TODO: Apply the same feature engineering to the streaming windowed data.
# Create 'hour_of_day' and 'minute_of_hour' columns.
windowed_features = None # Replace None with your implementation

# TODO: Create a VectorAssembler for the streaming features. It must match the one from training.
assembler_inference = None # Replace None with your implementation
feature_df = None # Replace None with your implementation

# TODO: Load the pre-trained regression model from MODEL_PATH.
trend_model = None # Replace None with your implementation

# TODO: Use the model to make predictions on the streaming features.
predictions = None # Replace None with your implementation

# Select final columns for output
output_df = predictions.select(
    col("window.start").alias("window_start"),
    col("window.end").alias("window_end"),
    "avg_fare",
    col("prediction").alias("predicted_next_avg_fare")
)

# Write predictions to the console
query = output_df.writeStream \
    .format("console") \
    .outputMode("append") \
    .option("truncate", False) \
    .start()

query.awaitTermination()
