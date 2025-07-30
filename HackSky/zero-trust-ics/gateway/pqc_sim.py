import hashlib
import secrets

def generate_key_pair():
    private_key = secrets.token_hex(32)
    public_key = hashlib.sha256(private_key.encode()).hexdigest()
    return public_key, private_key

def encrypt_session_key(public_key):
    # Client-side: simulate PQ encryption using shared key and nonce
    session_key = secrets.token_hex(16)
    cipher = hashlib.sha256((public_key + session_key).encode()).hexdigest()
    return session_key, cipher

def decrypt_and_verify(cipher, private_key, session_key):
    # Server-side: regenerate cipher and compare
    expected_cipher = hashlib.sha256((hashlib.sha256(private_key.encode()).hexdigest() + session_key).encode()).hexdigest()
    return cipher == expected_cipher