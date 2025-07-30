# gateway/models.py

from sqlalchemy import Column, Integer, String, DateTime
from sqlalchemy.ext.declarative import declarative_base
from datetime import datetime

Base = declarative_base()

class AccessLogEntry(Base):
    __tablename__ = "access_logs"

    id = Column(Integer, primary_key=True, index=True)
    username = Column(String)
    node = Column(String)
    ip = Column(String)
    status = Column(String)  # GRANTED / DENIED
    reason = Column(String)  # Why the access was granted or denied
    timestamp = Column(DateTime, default=datetime.utcnow)


class Anomaly(Base):
    __tablename__ = "anomalies"

    id = Column(Integer, primary_key=True, index=True)
    node = Column(String)
    description = Column(String)
    timestamp = Column(DateTime, default=datetime.utcnow)

class PredictionLog(Base):
    __tablename__ = "prediction_logs"

    id = Column(Integer, primary_key=True, index=True)
    packet_size = Column(Integer)
    command_type = Column(String)
    delta_time = Column(String)
    result = Column(String)  # anomalous / normal
    timestamp = Column(DateTime, default=datetime.utcnow)
