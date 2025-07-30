import requests
from hashlib import sha256
from pqc_sim import generate_key_pair, encrypt_session_key

# Simulate key pair
public_key, private_key = generate_key_pair()

# Simulate encryption of session key
session_key, cipher = encrypt_session_key(public_key)

# Legitimate node ID must be pre-registered in DB with its private key
payload = {
    "node_id": "NODE-001",
    "session_key": session_key,
    "cipher": cipher
}

gateway_url = "http://127.0.0.1:5000/connect"
response = requests.post(gateway_url, json=payload)

print("Status Code:", response.status_code)
try:
    print("Response JSON:", response.json())
except Exception as e:
    print("Error parsing response JSON:", e)
    print("Raw text response:", response.text)