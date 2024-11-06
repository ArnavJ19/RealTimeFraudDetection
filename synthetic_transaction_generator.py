import pandas as pd
import numpy as np
from faker import Faker
import random
from datetime import datetime, timedelta

# Initialize Faker
fake = Faker()

# Configuration
NUM_TRANSACTIONS = 500000      # Total number of transactions to generate
FRAUD_RATE = 0.1             # 10% of transactions will be fraudulent
OUTPUT_FILE = 'advanced_synthetic_transactions.csv'

# Predefined lists for merchants and categories
MERCHANTS = [
    'Amazon', 'Walmart', 'Starbucks', 'Apple Store', 'Target',
    'Best Buy', 'Uber', 'Netflix', 'eBay', 'Home Depot',
    'Costco', 'McDonald\'s', 'Kroger', 'Shell Gas', 'Delta Airlines',
    'Spotify', 'Lyft', 'Sephora', 'Nike', 'Adidas',
    'Visa Online', 'PayPal', 'Burger King', 'Subway', 'Shell Gas',
    'Chevron', 'PetSmart', 'GameStop', 'Office Depot', 'Panera Bread'
]

MERCHANT_CATEGORIES = {
    'Amazon': 'E-commerce',
    'Walmart': 'Retail',
    'Starbucks': 'Food & Beverage',
    'Apple Store': 'Electronics',
    'Target': 'Retail',
    'Best Buy': 'Electronics',
    'Uber': 'Transportation',
    'Netflix': 'Entertainment',
    'eBay': 'E-commerce',
    'Home Depot': 'Home Improvement',
    'Costco': 'Wholesale',
    'McDonald\'s': 'Food & Beverage',
    'Kroger': 'Grocery',
    'Shell Gas': 'Fuel',
    'Delta Airlines': 'Travel',
    'Spotify': 'Entertainment',
    'Lyft': 'Transportation',
    'Sephora': 'Beauty',
    'Nike': 'Apparel',
    'Adidas': 'Apparel',
    'Visa Online': 'Financial Services',
    'PayPal': 'Financial Services',
    'Burger King': 'Food & Beverage',
    'Subway': 'Food & Beverage',
    'Chevron': 'Fuel',
    'PetSmart': 'Pet Supplies',
    'GameStop': 'Entertainment',
    'Office Depot': 'Office Supplies',
    'Panera Bread': 'Food & Beverage'
}

TRANSACTION_TYPES = [
    'Online Purchase',
    'ATM Withdrawal',
    'Point of Sale',
    'Mobile Payment',
    'Wire Transfer',
    'Bill Payment',
    'Cash Deposit',
    'Peer-to-Peer Payment',
    'Subscription',
    'Recurring Payment'
]

DEVICE_TYPES = ['Mobile', 'Desktop', 'Tablet']

# Function to generate a list of unique UserIDs and CardIDs
def generate_users(num_users):
    users = []
    for user_id in range(1, num_users + 1):
        users.append({
            'UserID': f'U{user_id:05d}',
            'CardID': f'C{user_id:06d}'  # Assuming one card per user for simplicity
        })
    return users

# Function to generate a single transaction
def generate_transaction(transaction_id, users):
    user = random.choice(users)
    user_id = user['UserID']
    card_id = user['CardID']
    
    amount = round(random.uniform(1.0, 10000.0), 2)  # Transaction amount between $1 and $10,000
    time = fake.date_time_between(start_date='-90d', end_date='now')  # Transactions in the last 90 days
    day_of_week = time.strftime('%A')
    hour = time.hour
    location_city = fake.city()
    location_state=fake.state_abbr()
    merchant = random.choice(MERCHANTS)
    merchant_category = MERCHANT_CATEGORIES[merchant]
    transaction_type = random.choice(TRANSACTION_TYPES)
    device_id = fake.uuid4()
    ip_address = fake.ipv4()
    is_recurring = 'Yes' if transaction_type in ['Subscription', 'Recurring Payment'] else 'No'
    is_international = 'Yes' if random.random() < 0.05 else 'No'  # 5% international transactions
    
    # Fraudulent label will be assigned later
    return {
        'TransactionID': f'T{transaction_id:08d}',
        'UserID': user_id,
        'CardID': card_id,
        'Amount': amount,
        'Time': time.strftime("%Y-%m-%d %H:%M:%S"),
        'DayOfWeek': day_of_week,
        'Hour': hour,
        'Location_City': location_city,
        'Location State':location_state,
        'Merchant': merchant,
        'MerchantCategory': merchant_category,
        'TransactionType': transaction_type,
        'DeviceID': device_id,
        'IPAddress': ip_address,
        'IsRecurring': is_recurring,
        'IsInternational': is_international,
        'Fraudulent': 'No'  # Default, will be updated
    }

# Function to assign fraud labels based on certain patterns
def assign_fraud(transactions):
    num_frauds = int(len(transactions) * FRAUD_RATE)
    fraud_indices = random.sample(range(len(transactions)), num_frauds)
    
    for idx in fraud_indices:
        transaction = transactions[idx]
        # Introduce fraud by modifying certain features
        transaction['Fraudulent'] = 'Yes'
        # Optionally, adjust the amount or other features to simulate fraud
        transaction['Amount'] = round(transaction['Amount'] * random.uniform(1.5, 3.0), 2)  # Higher amount
        # Randomly change the location to a different state
        new_location = fake.city() + ", " + fake.state_abbr()
        transaction['Location'] = new_location
        # Change the transaction type
        transaction['TransactionType'] = random.choice(['Wire Transfer', 'ATM Withdrawal', 'Online Purchase'])
    
    return transactions

# Generate a list of users
def generate_user_data(num_users):
    print("Generating user data...")
    users = generate_users(num_users)
    print(f"Generated {num_users} users.")
    return users

# Generate all transactions
def generate_transactions(num_transactions, users):
    print("Starting transaction generation...")
    transactions = []
    for i in range(1, num_transactions + 1):
        transaction = generate_transaction(i, users)
        transactions.append(transaction)
        
        # Optional: Print progress every 5000 transactions
        if i % 5000 == 0:
            print(f"{i} transactions generated...")
                
    # Assign fraud labels
    print("Assigning fraud labels...")
    transactions = assign_fraud(transactions)
    print(f"Assigned fraud labels to {int(num_transactions * FRAUD_RATE)} transactions.")
    
    return transactions

# Save transactions to CSV
def save_to_csv(transactions, filename):
    print(f"Saving data to {filename}...")
    df = pd.DataFrame(transactions)
    df.to_csv(filename, index=False)
    print(f"Data saved to {filename}")

# Main execution
if __name__ == "__main__":
    NUM_USERS = 1000  # Number of unique users
    users = generate_user_data(NUM_USERS)
    transactions = generate_transactions(NUM_TRANSACTIONS, users)
    save_to_csv(transactions, OUTPUT_FILE)
    print("Data generation completed.")
