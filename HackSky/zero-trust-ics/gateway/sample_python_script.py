import requests

payload = {
    "node_id": "NODE-001",
    "session_key": "dummy_session_key",
    "cipher": "dummy_cipher_text"
}

response = requests.post("http://127.0.0.1:5000/connect", json=payload)

print("Status Code:", response.status_code)
print("Raw Text Response:", response.text)
