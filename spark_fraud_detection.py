from pyspark.sql import SparkSession
from pyspark.sql.functions import from_json, col
from pyspark.sql.types import StructType, StructField, StringType, DoubleType, IntegerType
import json
import joblib
import pandas as pd
import numpy as np

import os

os.environ['HADOOP_HOME'] = 'D:\\Hadoop'

# Initialize Spark Session with Kafka Connector
spark = SparkSession.builder \
    .appName("RealTimeFraudDetection") \
    .config("spark.jars.packages", "org.apache.spark:spark-sql-kafka-0-10_2.12:3.5.0") \
    .getOrCreate()

# Set log level to WARN to reduce verbosity
spark.sparkContext.setLogLevel("WARN")

# Define the schema for incoming JSON data
schema = StructType([
    StructField("TransactionID", StringType(), True),
    StructField("UserID", StringType(), True),
    StructField("CardID", StringType(), True),
    StructField("Amount", DoubleType(), True),
    StructField("Time", StringType(), True),
    StructField("DayOfWeek", StringType(), True),
    StructField("Hour", IntegerType(), True),
    StructField("Location", StringType(), True),
    StructField("Merchant", StringType(), True),
    StructField("MerchantCategory", StringType(), True),
    StructField("TransactionType", StringType(), True),
    StructField("DeviceID", StringType(), True),
    StructField("IPAddress", StringType(), True),
    StructField("IsRecurring", StringType(), True),
    StructField("IsInternational", StringType(), True),
    StructField("Fraudulent", StringType(), True)
])

# Load the trained model, label encoder, and scaler
model = joblib.load('fraud_model.joblib')
le = joblib.load('label_encoder.joblib')
scaler = joblib.load('scaler.joblib')

# Function to preprocess and predict fraud
def predict_fraud(batch_df):
    if batch_df.count() == 0:
        return
    
    # Convert Spark DataFrame to Pandas DataFrame
    pandas_df = batch_df.toPandas()
    
    # Drop irrelevant columns
    pandas_df = pandas_df.drop(['TransactionID', 'UserID', 'CardID', 'DeviceID', 'IPAddress','Time','DayOfWeek','Hour','Location'], axis=1)
    
    # Encode categorical variables using the loaded LabelEncoder
    categorical_features = ['Location_City', 'Location State','Merchant', 'MerchantCategory', 'TransactionType', 'IsRecurring', 'IsInternational']
    for col_name in categorical_features:
        pandas_df[col_name] = le.transform(pandas_df[col_name])
    
    # Feature Scaling
    features = scaler.transform(pandas_df)
    
    # Predict using the loaded model
    predictions = model.predict(features)
    
    # Add predictions to the original batch DataFrame
    batch_df = batch_df.withColumn("Prediction", col("Fraudulent"))  # Initialize Prediction column
    batch_df = batch_df.drop("Fraudulent")  # Drop the original label
    # Convert predictions to string labels
    prediction_labels = ['Yes' if pred == 1 else 'No' for pred in predictions]
    batch_df = batch_df.withColumn("Prediction", spark.createDataFrame(pd.DataFrame(prediction_labels)).iloc[:,0])
    
    # Show the predictions (for demonstration purposes)
    batch_df.select("TransactionID", "Prediction").show(truncate=False)
    
    # Here, you can add code to save the predictions to a database, send alerts, etc.

# Read from Kafka
df = spark.readStream \
    .format("kafka") \
    .option("kafka.bootstrap.servers", "localhost:9092") \
    .option("subscribe", "transactions") \
    .option("startingOffsets", "latest") \
    .load()

# Convert the binary 'value' column to string
df = df.selectExpr("CAST(value AS STRING)")

# Parse the JSON data
df = df.select(from_json(col("value"), schema).alias("data")).select("data.*")

# Start the streaming query
query = df.writeStream \
    .foreachBatch(predict_fraud) \
    .start()

query.awaitTermination()
