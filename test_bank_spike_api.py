import requests

def print_response(name, response):
    if response.status_code == 200:
        print(f"{name} Success:", response.json())
    elif response.status_code == 400:
        print(f"{name} Bad Request (400):", response.text)
    elif response.status_code == 422:
        print(f"{name} Validation Error (422):", response.text)
    elif response.status_code == 500:
        print(f"{name} Internal Server Error (500):", response.text)
    else:
        print(f"{name} Error {response.status_code}:", response.text)

# Test root endpoint
print("Testing root endpoint...")
url = "http://127.0.0.1:8000/"
response = requests.get(url)
print_response("Root", response)
print()

# Test health endpoint
print("Testing health endpoint...")
url = "http://127.0.0.1:8000/health"
response = requests.get(url)
print_response("Health", response)
print()

# Test prediction with valid data
print("Testing prediction with valid customer data...")
url = "http://127.0.0.1:8000/predict"
data = {
    "age": 30,
    "job": "management",
    "marital": "married",
    "education": "tertiary",
    "default": "no",
    "balance": 1000,
    "housing": "yes",
    "loan": "no",
    "contact": "cellular",
    "day_of_week": 15,
    "month": "may",
    "duration": 200,
    "campaign": 1,
    "pdays": -1,
    "previous": 0,
    "poutcome": "unknown"
}
response = requests.post(url, json=data)
print_response("Prediction", response)
print()

# Test prediction with high-value customer
print("Testing prediction with high-value customer...")
data = {
    "age": 45,
    "job": "management",
    "marital": "married",
    "education": "tertiary",
    "default": "no",
    "balance": 50000,
    "housing": "yes",
    "loan": "no",
    "contact": "cellular",
    "day_of_week": 20,
    "month": "may",
    "duration": 500,
    "campaign": 1,
    "pdays": -1,
    "previous": 0,
    "poutcome": "unknown"
}
response = requests.post(url, json=data)
print_response("High-value Customer", response)
print()

# Test prediction with young customer
print("Testing prediction with young customer...")
data = {
    "age": 25,
    "job": "student",
    "marital": "single",
    "education": "secondary",
    "default": "no",
    "balance": 100,
    "housing": "no",
    "loan": "no",
    "contact": "cellular",
    "day_of_week": 10,
    "month": "jun",
    "duration": 100,
    "campaign": 2,
    "pdays": -1,
    "previous": 0,
    "poutcome": "unknown"
}
response = requests.post(url, json=data)
print_response("Young Customer", response)
print()

# Test prediction with previous customer
print("Testing prediction with previous customer...")
data = {
    "age": 35,
    "job": "technician",
    "marital": "married",
    "education": "tertiary",
    "default": "no",
    "balance": 5000,
    "housing": "yes",
    "loan": "yes",
    "contact": "telephone",
    "day_of_week": 25,
    "month": "aug",
    "duration": 300,
    "campaign": 3,
    "pdays": 100,
    "previous": 2,
    "poutcome": "success"
}
response = requests.post(url, json=data)
print_response("Previous Customer", response)
print()

# Test validation errors
print("Testing validation errors...")

# Test invalid age
print("Testing invalid age (too young)...")
data = {
    "age": 15,  # Invalid age
    "job": "management",
    "marital": "married",
    "education": "tertiary",
    "default": "no",
    "balance": 1000,
    "housing": "yes",
    "loan": "no",
    "contact": "cellular",
    "day_of_week": 15,
    "month": "may",
    "duration": 200,
    "campaign": 1,
    "pdays": -1,
    "previous": 0,
    "poutcome": "unknown"
}
response = requests.post(url, json=data)
print_response("Invalid Age", response)
print()

# Test invalid balance
print("Testing invalid balance (too high)...")
data = {
    "age": 30,
    "job": "management",
    "marital": "married",
    "education": "tertiary",
    "default": "no",
    "balance": 500000,  # Invalid balance
    "housing": "yes",
    "loan": "no",
    "contact": "cellular",
    "day_of_week": 15,
    "month": "may",
    "duration": 200,
    "campaign": 1,
    "pdays": -1,
    "previous": 0,
    "poutcome": "unknown"
}
response = requests.post(url, json=data)
print_response("Invalid Balance", response)
print()

# Test missing required field
print("Testing missing required field...")
data = {
    "age": 30,
    "job": "management",
    "marital": "married",
    "education": "tertiary",
    "default": "no",
    "balance": 1000,
    "housing": "yes",
    "loan": "no",
    "contact": "cellular",
    "day_of_week": 15,
    "month": "may",
    "duration": 200,
    "campaign": 1,
    "pdays": -1,
    "previous": 0
    # Missing "poutcome" field
}
response = requests.post(url, json=data)
print_response("Missing Field", response)
print()

print("All tests completed!")
