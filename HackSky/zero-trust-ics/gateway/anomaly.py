from db import get_recent_logs

def detect_anomalies():
    try:
        logs = get_recent_logs(limit=100)

        if not logs:
            return {"anomalies": []}

        anomalies = []
        for log in logs:
            status = log.get("status", "").lower()
            if status in ["denied", "unauthorized", "suspicious", "blocked"]:
                anomalies.append(log)

        return {"anomalies": anomalies}
    
    except Exception as e:
        print("[ERROR] in detect_anomalies():", e)
        raise e
