import requests
import json
import random
import time

URL = "http://localhost:8000/predict"

# Генерация случайных параметров цветка
def generate_fake_data():
    return [
        round(random.uniform(4.3, 7.9), 1), # sepal length
        round(random.uniform(2.0, 4.4), 1), # sepal width
        round(random.uniform(1.0, 6.9), 1), # petal length
        round(random.uniform(0.1, 2.5), 1)  # petal width
    ]

print("Starting API load test...")

for i in range(100):
    data = {"data": generate_fake_data()}
    try:
        response = requests.post(URL, json=data)
        if response.status_code == 200:
            print(f"Request {i+1}: Success")
        else:
            print(f"Request {i+1}: Error {response.status_code}")
    except Exception as e:
        print(f"Request {i+1}: Failed to connect - {e}")
    
    time.sleep(0.1)

print("Done. Check your data/ab_logs.csv")