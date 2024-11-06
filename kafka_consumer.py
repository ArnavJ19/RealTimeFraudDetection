from kafka import KafkaConsumer
import json

def main():
    # Initialize Kafka consumer
    consumer = KafkaConsumer(
        'transactions',
        bootstrap_servers=['localhost:9092'],
        auto_offset_reset='earliest',
        enable_auto_commit=True,
        group_id='fraud_detection_group',
        value_deserializer=lambda x: json.loads(x.decode('utf-8'))
    )
    
    print("Starting Kafka Consumer...")
    for message in consumer:
        transaction = message.value
        print(f"Received Transaction ID: {transaction['TransactionID']} - Fraudulent: {transaction['Fraudulent']}")
        # Add processing logic here if needed

if __name__ == "__main__":
    main()
