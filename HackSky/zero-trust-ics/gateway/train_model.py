import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import LabelEncoder
import joblib

# Load synthetic data
df = pd.read_csv("synthetic_ics_data.csv")

# Encode 'command_type' (categorical -> numerical)
encoder = LabelEncoder()
df['command_type'] = encoder.fit_transform(df['command_type'])

# Save the encoder for later use
joblib.dump(encoder, "command_encoder.pkl")

# Features
X = df[["packet_size", "command_type"]]

# Train model
model = IsolationForest(n_estimators=100, contamination=0.05, random_state=42)
model.fit(X)

# Save model
joblib.dump(model, "anomaly_model.pkl")

print("✅ Model and encoder saved successfully!")
