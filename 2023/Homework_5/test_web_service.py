import requests

url = "http://localhost:9696/predict"
client_test = {"job": "unknown", "duration": 270, "poutcome": "failure"}

print(requests.post(url, json=client_test).json())

client_test_2 = {"job": "retired", "duration": 445, "poutcome": "success"}

print(requests.post(url, json=client_test_2).json())
