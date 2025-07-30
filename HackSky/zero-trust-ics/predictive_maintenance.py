import numpy as np
import pandas as pd
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime, timedelta
import logging
from dataclasses import dataclass
from sklearn.ensemble import RandomForestRegressor, IsolationForest
from sklearn.preprocessing import StandardScaler
import joblib
import json

@dataclass
class MaintenancePrediction:
    device_id: str
    failure_probability: float
    time_to_failure: Optional[float]  # hours
    maintenance_recommendation: str
    confidence: float
    urgency_level: str  # low, medium, high, critical
    affected_components: List[str]
    timestamp: datetime

class PredictiveMaintenanceEngine:
    def __init__(self):
        self.models = {}
        self.scalers = {}
        self.device_health_scores = {}
        self.maintenance_history = {}
        self.failure_thresholds = {
            'critical': 0.9,
            'high': 0.7,
            'medium': 0.5,
            'low': 0.3
        }
        
        # Initialize ML models
        self._initialize_models()
        
    def _initialize_models(self):
        """Initialize machine learning models for different prediction tasks"""
        # Failure prediction model
        self.models['failure_prediction'] = RandomForestRegressor(
            n_estimators=100,
            max_depth=10,
            random_state=42
        )
        
        # Anomaly detection model
        self.models['anomaly_detection'] = IsolationForest(
            contamination=0.1,
            random_state=42
        )
        
        # Health scoring model
        self.models['health_scoring'] = RandomForestRegressor(
            n_estimators=50,
            max_depth=8,
            random_state=42
        )
        
        # Initialize scalers
        self.scalers['feature_scaler'] = StandardScaler()
        self.scalers['health_scaler'] = StandardScaler()
        
    def extract_maintenance_features(self, device_telemetry: Dict[str, Any], 
                                   network_context: Dict[str, Any]) -> Dict[str, float]:
        """Extract features for predictive maintenance"""
        features = {}
        
        # Device-specific features
        features['temperature'] = device_telemetry.get('temperature', 0)
        features['pressure'] = device_telemetry.get('pressure', 0)
        features['vibration'] = device_telemetry.get('vibration', 0)
        features['current_draw'] = device_telemetry.get('current_draw', 0)
        features['voltage'] = device_telemetry.get('voltage', 0)
        features['frequency'] = device_telemetry.get('frequency', 0)
        features['runtime_hours'] = device_telemetry.get('runtime_hours', 0)
        features['error_count'] = device_telemetry.get('error_count', 0)
        features['warning_count'] = device_telemetry.get('warning_count', 0)
        
        # Network-based features
        features['network_load'] = network_context.get('network_load', 0)
        features['connection_stability'] = network_context.get('connection_stability', 1.0)
        features['latency'] = network_context.get('latency', 0)
        features['packet_loss'] = network_context.get('packet_loss', 0)
        
        # Derived features
        features['temperature_variance'] = device_telemetry.get('temperature_variance', 0)
        features['pressure_variance'] = device_telemetry.get('pressure_variance', 0)
        features['vibration_variance'] = device_telemetry.get('vibration_variance', 0)
        
        # Time-based features
        current_time = datetime.utcnow()
        features['hour_of_day'] = current_time.hour
        features['day_of_week'] = current_time.weekday()
        features['month'] = current_time.month
        
        return features
    
    def predict_maintenance_needs(self, device_id: str, 
                                device_telemetry: Dict[str, Any],
                                network_context: Dict[str, Any]) -> MaintenancePrediction:
        """Predict maintenance needs for a device"""
        
        # Extract features
        features = self.extract_maintenance_features(device_telemetry, network_context)
        feature_vector = list(features.values())
        
        # Predict failure probability
        failure_prob = self._predict_failure_probability(feature_vector)
        
        # Predict time to failure
        time_to_failure = self._predict_time_to_failure(feature_vector)
        
        # Calculate health score
        health_score = self._calculate_health_score(feature_vector)
        
        # Determine urgency level
        urgency_level = self._determine_urgency_level(failure_prob, time_to_failure)
        
        # Generate maintenance recommendation
        recommendation = self._generate_maintenance_recommendation(
            failure_prob, time_to_failure, health_score, features
        )
        
        # Identify affected components
        affected_components = self._identify_affected_components(features)
        
        # Calculate confidence
        confidence = self._calculate_prediction_confidence(feature_vector, failure_prob)
        
        return MaintenancePrediction(
            device_id=device_id,
            failure_probability=failure_prob,
            time_to_failure=time_to_failure,
            maintenance_recommendation=recommendation,
            confidence=confidence,
            urgency_level=urgency_level,
            affected_components=affected_components,
            timestamp=datetime.utcnow()
        )
    
    def _predict_failure_probability(self, feature_vector: List[float]) -> float:
        """Predict the probability of device failure"""
        try:
            # Scale features
            scaled_features = self.scalers['feature_scaler'].transform([feature_vector])
            
            # Use isolation forest for anomaly detection
            anomaly_score = self.models['anomaly_detection'].score_samples(scaled_features)[0]
            
            # Convert anomaly score to failure probability
            # Lower anomaly score = higher failure probability
            failure_prob = 1.0 - (anomaly_score + 0.5)  # Normalize to [0,1]
            failure_prob = max(0.0, min(1.0, failure_prob))
            
            return failure_prob
            
        except Exception as e:
            logging.error(f"Error predicting failure probability: {e}")
            return 0.5  # Default to medium probability
    
    def _predict_time_to_failure(self, feature_vector: List[float]) -> Optional[float]:
        """Predict time to failure in hours"""
        try:
            # This would use a trained regression model
            # For now, use a simple heuristic based on health indicators
            
            # Extract relevant features for time prediction
            temp = feature_vector[0]  # temperature
            vib = feature_vector[2]   # vibration
            runtime = feature_vector[6]  # runtime_hours
            errors = feature_vector[7]   # error_count
            
            # Simple heuristic: higher values = shorter time to failure
            time_to_failure = 1000  # Base time in hours
            
            # Reduce time based on stress indicators
            if temp > 80:  # High temperature
                time_to_failure *= 0.7
            if vib > 0.5:  # High vibration
                time_to_failure *= 0.8
            if errors > 10:  # High error count
                time_to_failure *= 0.6
            if runtime > 5000:  # High runtime
                time_to_failure *= 0.9
            
            return max(24, time_to_failure)  # Minimum 24 hours
            
        except Exception as e:
            logging.error(f"Error predicting time to failure: {e}")
            return None
    
    def _calculate_health_score(self, feature_vector: List[float]) -> float:
        """Calculate overall device health score"""
        try:
            # Normalize health indicators
            temp = min(feature_vector[0] / 100.0, 1.0)  # Temperature (0-100°C)
            vib = min(feature_vector[2] / 1.0, 1.0)     # Vibration (0-1)
            errors = min(feature_vector[7] / 50.0, 1.0) # Error count (0-50)
            
            # Calculate health score (lower values = better health)
            health_score = (temp + vib + errors) / 3.0
            health_score = 1.0 - health_score  # Invert so higher = better health
            
            return max(0.0, min(1.0, health_score))
            
        except Exception as e:
            logging.error(f"Error calculating health score: {e}")
            return 0.5
    
    def _determine_urgency_level(self, failure_prob: float, time_to_failure: Optional[float]) -> str:
        """Determine urgency level based on failure probability and time"""
        if failure_prob >= self.failure_thresholds['critical']:
            return 'critical'
        elif failure_prob >= self.failure_thresholds['high']:
            return 'high'
        elif failure_prob >= self.failure_thresholds['medium']:
            return 'medium'
        elif failure_prob >= self.failure_thresholds['low']:
            return 'low'
        else:
            return 'normal'
    
    def _generate_maintenance_recommendation(self, failure_prob: float, 
                                           time_to_failure: Optional[float],
                                           health_score: float,
                                           features: Dict[str, float]) -> str:
        """Generate specific maintenance recommendations"""
        
        recommendations = []
        
        if failure_prob > 0.8:
            recommendations.append("IMMEDIATE SHUTDOWN RECOMMENDED")
        elif failure_prob > 0.6:
            recommendations.append("Schedule maintenance within 24 hours")
        elif failure_prob > 0.4:
            recommendations.append("Schedule maintenance within 1 week")
        elif failure_prob > 0.2:
            recommendations.append("Monitor closely, schedule maintenance within 1 month")
        
        # Component-specific recommendations
        if features['temperature'] > 80:
            recommendations.append("Check cooling system and thermal management")
        if features['vibration'] > 0.5:
            recommendations.append("Inspect mechanical components and bearings")
        if features['error_count'] > 10:
            recommendations.append("Review error logs and check system configuration")
        if features['pressure'] > 100:
            recommendations.append("Check pressure regulation system")
        
        if not recommendations:
            recommendations.append("Continue normal operation and monitoring")
        
        return "; ".join(recommendations)
    
    def _identify_affected_components(self, features: Dict[str, float]) -> List[str]:
        """Identify which components are likely affected"""
        affected = []
        
        if features['temperature'] > 80:
            affected.extend(['cooling_system', 'thermal_management', 'electronics'])
        if features['vibration'] > 0.5:
            affected.extend(['bearings', 'mechanical_components', 'mounting'])
        if features['pressure'] > 100:
            affected.extend(['pressure_regulator', 'valves', 'piping'])
        if features['error_count'] > 10:
            affected.extend(['control_system', 'sensors', 'communication'])
        if features['current_draw'] > 50:
            affected.extend(['electrical_system', 'power_supply', 'motors'])
        
        return list(set(affected))  # Remove duplicates
    
    def _calculate_prediction_confidence(self, feature_vector: List[float], 
                                       failure_prob: float) -> float:
        """Calculate confidence in the prediction"""
        # Higher confidence if we have more reliable indicators
        confidence = 0.5  # Base confidence
        
        # Increase confidence based on feature quality
        if len(feature_vector) >= 10:
            confidence += 0.2
        
        # Increase confidence if failure probability is very high or very low
        if failure_prob > 0.8 or failure_prob < 0.2:
            confidence += 0.2
        
        # Decrease confidence if we have missing or unreliable data
        if any(f == 0 for f in feature_vector[:5]):  # Critical features
            confidence -= 0.1
        
        return max(0.1, min(1.0, confidence))
    
    def train_models(self, training_data: List[Dict[str, Any]]):
        """Train the predictive maintenance models"""
        try:
            # Prepare training data
            X = []
            y_failure = []
            y_health = []
            
            for data_point in training_data:
                features = self.extract_maintenance_features(
                    data_point['telemetry'], 
                    data_point.get('network_context', {})
                )
                X.append(list(features.values()))
                
                # Target variables
                y_failure.append(data_point.get('failure_occurred', 0))
                y_health.append(data_point.get('health_score', 0.5))
            
            X = np.array(X)
            y_failure = np.array(y_failure)
            y_health = np.array(y_health)
            
            # Fit scalers
            self.scalers['feature_scaler'].fit(X)
            self.scalers['health_scaler'].fit(X)
            
            # Scale features
            X_scaled = self.scalers['feature_scaler'].transform(X)
            
            # Train models
            self.models['anomaly_detection'].fit(X_scaled)
            self.models['health_scoring'].fit(X_scaled, y_health)
            
            logging.info("Predictive maintenance models trained successfully")
            
        except Exception as e:
            logging.error(f"Error training predictive maintenance models: {e}")
    
    def save_models(self, filepath: str):
        """Save trained models to disk"""
        try:
            model_data = {
                'models': self.models,
                'scalers': self.scalers,
                'failure_thresholds': self.failure_thresholds
            }
            joblib.dump(model_data, filepath)
            logging.info(f"Models saved to {filepath}")
        except Exception as e:
            logging.error(f"Error saving models: {e}")
    
    def load_models(self, filepath: str):
        """Load trained models from disk"""
        try:
            model_data = joblib.load(filepath)
            self.models = model_data['models']
            self.scalers = model_data['scalers']
            self.failure_thresholds = model_data['failure_thresholds']
            logging.info(f"Models loaded from {filepath}")
        except Exception as e:
            logging.error(f"Error loading models: {e}")
    
    def get_maintenance_schedule(self, predictions: List[MaintenancePrediction]) -> Dict[str, List[MaintenancePrediction]]:
        """Generate maintenance schedule based on predictions"""
        schedule = {
            'immediate': [],
            'within_24h': [],
            'within_week': [],
            'within_month': [],
            'monitor': []
        }
        
        for prediction in predictions:
            if prediction.urgency_level == 'critical':
                schedule['immediate'].append(prediction)
            elif prediction.urgency_level == 'high':
                schedule['within_24h'].append(prediction)
            elif prediction.urgency_level == 'medium':
                schedule['within_week'].append(prediction)
            elif prediction.urgency_level == 'low':
                schedule['within_month'].append(prediction)
            else:
                schedule['monitor'].append(prediction)
        
        return schedule 