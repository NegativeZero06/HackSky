# gateway/dashboard_api.py

from flask import Flask, request, jsonify, send_from_directory
from anomaly import AnomalyDetector
import joblib

app = Flask(__name__, static_folder="static")
detector = AnomalyDetector()

# Load the trained model and encoder
detector.model = joblib.load("anomaly_model.pkl")
detector.encoder = joblib.load("command_encoder.pkl")
detector.trained = True

# In-memory storage for dashboard
history = []

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()
    try:
        packet_size = float(data["packet_size"])
        command_type = data["command_type"]
        delta_time = float(data["delta_time"])

        encoded_command = detector.encoder.transform([[command_type]])[0][0]
        features = [packet_size, encoded_command, delta_time]
        result = detector.predict(features)

        history.append({
            "packet_size": packet_size,
            "command_type": command_type,
            "delta_time": delta_time,
            "result": "Anomaly" if result == -1 else "Normal"
        })

        return jsonify({"result": "Anomaly" if result == -1 else "Normal"}), 200

    except Exception as e:
        return jsonify({"error": str(e)}), 400

@app.route('/history', methods=['GET'])
def get_history():
    return jsonify(history[-100:])  # return last 100 records


@app.route('/')
def serve_dashboard():
    return send_from_directory('static', 'dashboard.html')

if __name__ == "__main__":
    app.run(debug=True, port=5001)
