# 🔐 Zero-Trust ICS Gateway Simulator with Post-Quantum Authentication

> ⚙️ A next-gen cybersecurity architecture prototype designed to protect Industrial Control Systems (ICS) under extreme constraints, inspired by Kaspersky x MIT Hackathon 2025.

---

## 🚀 Project Overview

In a world of increasing ICS-targeted cyberattacks, traditional perimeter-based trust is no longer sufficient. Our solution simulates a **Zero-Trust ICS Gateway** that:
- Rejects any implicit trust between nodes
- Authenticates devices using **Post-Quantum Cryptography**
- Operates offline and with minimal compute
- Detects unauthorized or suspicious behaviors in real-time

---

## 🎯 Key Features

- ✅ Zero-Trust Access Control (no default trust, every access verified)
- 🔐 Post-Quantum Authentication using [Kyber](https://pq-crystals.org/kyber/)
- 📋 Real-time access logging and detection of rogue nodes
- ⚠️ Alerts and failsafe response on suspicious behavior
- 🖥️ Lightweight Flask-based ICS Gateway + multiple client nodes
- 💾 Fully offline-capable, designed for air-gapped environments

---

## 🛠️ Tech Stack

| Component | Tech Used |
|----------|------------|
| Backend  | Python, Flask, Socket |
| Crypto   | PQCrypto or OQS Lib (Kyber512 / FrodoKEM) |
| Database | SQLite (lightweight + embedded) |
| Dashboard (Optional) | HTML/CSS + Bootstrap |
| Others   | Threading, Logging, UUID, JSON |

---

## 🧱 System Architecture

```
+------------------+        +------------------------+        +-------------------+
|   ICS Client A   | <----> |   Zero-Trust Gateway   | <----> |  Access Log DB     |
| (Legit Node)     |        |   (Flask API)          |        |  (SQLite)          |
+------------------+        +------------------------+        +-------------------+

+------------------+        🔐 Post-Quantum Auth
|   ICS Client B   |  --->  ❌ Rejected (rogue node)
| (Unregistered)   | 
+------------------+
```

---

## 🧪 How It Works

1. **Node Registration**: Legitimate nodes are registered with their public keys.
2. **Connection Request**: Each client node sends a request to the gateway with its identity.
3. **Authentication**:
   - Post-Quantum Key Exchange (Kyber/FrodoKEM)
   - Secure session established if verified
4. **Access Granted/Denied**: Based on trust rules and node identity.
5. **Logging**: Every interaction is logged locally.

---

## 🧩 Folder Structure

```
zero-trust-ics/
├── gateway/              # Flask server (zero-trust gateway)
│   ├── app.py
│   └── auth/
│       └── pqc.py        # Post-quantum crypto handshake logic
├── clients/
│   ├── node1.py          # Legitimate node
│   └── rogue_node.py     # Unauthenticated node
├── logs/
│   └── access_log.db     # SQLite DB for access logs
├── dashboard/            # Optional frontend UI
├── README.md
└── requirements.txt
```

---

## 🚦 Demo

### ✅ Legitimate Node Access:
```
[2025-07-29 15:02:30] Node ID: NODE-1 — Access GRANTED ✅
```

### ❌ Rogue Node Blocked:
```
[2025-07-29 15:03:12] Node ID: UNKNOWN — Access DENIED ❌ — Reason: Auth failure
```

---

## 📌 Future Scope

- Integrate Role-Based Access Control (RBAC)
- Add automatic quarantine for misbehaving nodes
- Integrate with real ICS hardware/simulators
- Deploy inside Docker for isolation

---

## 👨‍💻 Team Members

- Member 1 – Gateway Logic, Auth System  
- Member 2 – PQC Integration, Secure Sessions  
- Member 3 – Simulated Clients, Log DB, Dashboard  

---

## 📝 References

- [Kyber Post-Quantum Crypto](https://pq-crystals.org/kyber/)
- [Open Quantum Safe Project](https://openquantumsafe.org/)
- [NIST PQC Standardization](https://csrc.nist.gov/projects/post-quantum-cryptography)