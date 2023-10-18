import requests

url = "http://localhost:9696/predict"
client_test = {"job": "unknown", "duration": 270, "poutcome": "failure"}

print(requests.post(url, json=client_test).json())
