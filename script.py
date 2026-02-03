from pymongo import MongoClient
from dotenv import load_dotenv
import os
load_dotenv()

mongo_uri = os.getenv("MONGO_URL")

print("mongo_uri", mongo_uri)

if mongo_uri is None:
    raise SystemExit("MONGO_URI not found")

# Connect to MongoDB
client = MongoClient(mongo_uri)



# Select database and collection
db = client["test"]
collection = db["grocery"]

grocery_items = [
    {"name": "Rice", "unit": "1 kg", "price": 90},
    {"name": "Wheat Flour", "unit": "1 kg", "price": 75},
    {"name": "Milk", "unit": "1 L", "price": 120},
    {"name": "Eggs", "unit": "12 pcs", "price": 180},
    {"name": "Cooking Oil", "unit": "1 L", "price": 320},
    {"name": "Sugar", "unit": "1 kg", "price": 95},
    {"name": "Salt", "unit": "1 kg", "price": 30},
    {"name": "Tea", "unit": "500 g", "price": 450},
    {"name": "Coffee", "unit": "250 g", "price": 550},
    {"name": "Bread", "unit": "1 loaf", "price": 70},
]
result = collection.insert_many(grocery_items)

print(f"Inserted {len(result.inserted_ids)} grocery items successfully!")
