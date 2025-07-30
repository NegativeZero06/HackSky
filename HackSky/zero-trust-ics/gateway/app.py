from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from .models import AccessLogEntry as AccessLog, Anomaly, PredictionLog
from .database import SessionLocal
from datetime import datetime
import uvicorn

app = FastAPI()

@app.get("/ping")
def ping():
    return {"message": "pong"}

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Restrict in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ------------------- Input Models -------------------
class AccessInput(BaseModel):
    username: str
    node: str
    ip: str

class PredictionInput(BaseModel):
    packet_size: int
    command_type: str
    delta_time: float
    result: str

class AnomalyLogInput(BaseModel):
    node: str
    description: str

# ------------------- Zero Trust Policy -------------------
def zero_trust_policy(user: str, node: str, ip: str):
    if ip.startswith("192.168"):
        return "GRANTED", "Internal IP address"
    elif "intruder" in user.lower():
        return "DENIED", "User flagged as suspicious"
    elif node not in ["PLC-1", "PLC-2", "PLC-3"]:
        return "DENIED", "Unknown node"
    return "DENIED", "Default deny"

# ------------------- Routes -------------------

@app.post("/access")
async def handle_access(data: AccessInput):
    status, reason = zero_trust_policy(data.username, data.node, data.ip)

    db = SessionLocal()
    db.add(AccessLog(
        username=data.username,
        node=data.node,
        ip=data.ip,
        status=status,
        reason=reason,
        timestamp=datetime.utcnow()
    ))
    db.commit()
    db.close()

    return {"status": status, "reason": reason}

@app.post("/log_prediction")
async def log_prediction(data: PredictionInput):
    db = SessionLocal()
    db.add(PredictionLog(
        packet_size=data.packet_size,
        command_type=data.command_type,
        delta_time=data.delta_time,
        result=data.result,
        timestamp=datetime.utcnow()
    ))
    db.commit()
    db.close()
    return {"status": "success"}

@app.post("/log_anomaly")
async def log_anomaly(log: AnomalyLogInput):
    db = SessionLocal()
    db.add(Anomaly(
        node=log.node,
        description=log.description,
        timestamp=datetime.utcnow()
    ))
    db.commit()
    db.close()
    return {"status": "logged"}

@app.get("/metrics/access")
async def get_access_logs():
    db = SessionLocal()
    logs = db.query(AccessLog).order_by(AccessLog.timestamp.desc()).limit(50).all()
    db.close()
    return [
        {
            "username": log.username,
            "node": log.node,
            "ip": log.ip,
            "status": log.status,
            "reason": log.reason,
            "timestamp": log.timestamp.isoformat()
        } for log in logs
    ]

@app.get("/metrics/anomalies")
async def get_anomalies():
    db = SessionLocal()
    logs = db.query(Anomaly).order_by(Anomaly.timestamp.desc()).limit(50).all()
    db.close()
    return [
        {
            "node": log.node,
            "description": log.description,
            "timestamp": log.timestamp.isoformat()
        } for log in logs
    ]

@app.get("/metrics/predictions")
async def get_predictions():
    db = SessionLocal()
    logs = db.query(PredictionLog).order_by(PredictionLog.timestamp.desc()).limit(50).all()
    db.close()
    return [
        {
            "packet_size": log.packet_size,
            "command_type": log.command_type,
            "delta_time": log.delta_time,
            "result": log.result,
            "timestamp": log.timestamp.isoformat()
        } for log in logs
    ]

@app.get("/")
def root():
    return {"message": "Welcome to the Zero Trust ICS API Gateway"}

if __name__ == "__main__":
    uvicorn.run("gateway.app:app", host="0.0.0.0", port=8000, reload=True)
