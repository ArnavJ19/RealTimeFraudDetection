from kafka import KafkaProducer
import pandas as pd
import json
import time

def send_transactions(producer, df, topic, delay=0.01):
    """
    Sends transactions to the specified Kafka topic.

    :param producer: KafkaProducer instance
    :param df: Pandas DataFrame containing transactions
    :param topic: Kafka topic name
    :param delay: Delay in seconds between sending messages to simulate real-time streaming
    """
    for index, row in df.iterrows():
        transaction = row.to_dict()
        producer.send(topic, value=transaction)
        print(f"Sent Transaction ID: {transaction['TransactionID']}")
        time.sleep(delay)  # Simulate real-time streaming
    producer.flush()
    print("All transactions have been sent.")

if __name__ == "__main__":
    # Initialize Kafka producer
    producer = KafkaProducer(
        bootstrap_servers=['localhost:9092'],
        value_serializer=lambda v: json.dumps(v).encode('utf-8')
    )
    
    # Read the synthetic transactions CSV
    try:
        df = pd.read_csv('advanced_synthetic_transactions.csv')
        print(f"Loaded {len(df)} transactions.")
    except FileNotFoundError:
        print("CSV file not found. Please ensure 'advanced_synthetic_transactions.csv' exists.")
        exit(1)
    
    # Define Kafka topic
    topic = 'transactions'
    
    # Start sending transactions
    send_transactions(producer, df, topic, delay=0.0001)  # Adjust delay as needed
    
    # Close the producer
    producer.close()
