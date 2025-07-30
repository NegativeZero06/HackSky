import networkx as nx
import numpy as np
import pandas as pd
from typing import Dict, List, Any, Tuple, Optional
from datetime import datetime, timedelta
import json
import logging
from dataclasses import dataclass
from sqlalchemy import create_engine, Column, Integer, String, DateTime, Float, JSON, Boolean, Text
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker

Base = declarative_base()

@dataclass
class SuspiciousNode:
    node_id: str
    anomaly_score: float
    anomaly_type: str
    timestamp: datetime
    features: Dict[str, Any]
    network_context: Dict[str, Any]
    confidence: float
    requires_investigation: bool = True

class SuspiciousNodeDB(Base):
    __tablename__ = "suspicious_nodes"
    
    id = Column(Integer, primary_key=True, index=True)
    node_id = Column(String, index=True)
    anomaly_score = Column(Float)
    anomaly_type = Column(String)  # network, behavioral, temporal, predictive
    timestamp = Column(DateTime, default=datetime.utcnow)
    features = Column(JSON)  # Feature vector that caused the anomaly
    network_context = Column(JSON)  # Network topology context
    confidence = Column(Float)
    requires_investigation = Column(Boolean, default=True)
    investigation_status = Column(String, default="pending")  # pending, investigating, resolved, false_positive
    investigation_notes = Column(Text)
    model_feedback = Column(JSON)  # Feedback for model fine-tuning

class GraphAnomalyDetector:
    def __init__(self, db_url: str):
        self.engine = create_engine(db_url)
        Base.metadata.create_all(self.engine)
        Session = sessionmaker(bind=self.engine)
        self.session = Session()
        
        # Initialize graph
        self.network_graph = nx.Graph()
        self.device_profiles = {}
        self.anomaly_thresholds = {
            'network': 0.7,
            'behavioral': 0.8,
            'temporal': 0.75,
            'predictive': 0.85
        }
        
        # Graph-based features
        self.centrality_metrics = {}
        self.community_structure = {}
        self.path_analysis = {}
        
    def build_network_graph(self, device_connections: List[Dict[str, Any]]):
        """Build network graph from device connections"""
        self.network_graph.clear()
        
        for connection in device_connections:
            source = connection['source_device']
            target = connection['target_device']
            weight = connection.get('traffic_volume', 1)
            protocol = connection.get('protocol', 'unknown')
            
            self.network_graph.add_edge(source, target, 
                                      weight=weight, 
                                      protocol=protocol,
                                      last_seen=datetime.utcnow())
            
        # Calculate centrality metrics
        self.centrality_metrics = {
            'betweenness': nx.betweenness_centrality(self.network_graph),
            'closeness': nx.closeness_centrality(self.network_graph),
            'eigenvector': nx.eigenvector_centrality_numpy(self.network_graph, max_iter=1000),
            'degree': dict(self.network_graph.degree())
        }
        
        # Detect communities
        self.community_structure = self._detect_communities()
        
    def _detect_communities(self) -> Dict[str, List[str]]:
        """Detect network communities using Louvain method"""
        try:
            communities = nx.community.louvain_communities(self.network_graph)
            community_dict = {}
            for i, community in enumerate(communities):
                community_dict[f"community_{i}"] = list(community)
            return community_dict
        except Exception as e:
            logging.warning(f"Community detection failed: {e}")
            return {}
    
    def detect_network_anomalies(self, device_telemetry: Dict[str, Any]) -> List[SuspiciousNode]:
        """Detect anomalies based on network topology and device relationships"""
        suspicious_nodes = []
        
        for device_id, telemetry in device_telemetry.items():
            if device_id not in self.network_graph.nodes():
                continue
                
            # Calculate network-based features
            network_features = self._extract_network_features(device_id, telemetry)
            
            # Detect different types of anomalies
            anomalies = []
            
            # 1. Centrality anomaly
            centrality_anomaly = self._detect_centrality_anomaly(device_id, network_features)
            if centrality_anomaly:
                anomalies.append(centrality_anomaly)
            
            # 2. Community anomaly
            community_anomaly = self._detect_community_anomaly(device_id, network_features)
            if community_anomaly:
                anomalies.append(community_anomaly)
            
            # 3. Path anomaly
            path_anomaly = self._detect_path_anomaly(device_id, network_features)
            if path_anomaly:
                anomalies.append(path_anomaly)
            
            # 4. Traffic pattern anomaly
            traffic_anomaly = self._detect_traffic_anomaly(device_id, telemetry)
            if traffic_anomaly:
                anomalies.append(traffic_anomaly)
            
            # Combine anomalies
            if anomalies:
                combined_score = np.mean([a['score'] for a in anomalies])
                anomaly_type = self._determine_anomaly_type(anomalies)
                
                suspicious_node = SuspiciousNode(
                    node_id=device_id,
                    anomaly_score=combined_score,
                    anomaly_type=anomaly_type,
                    timestamp=datetime.utcnow(),
                    features=network_features,
                    network_context=self._get_network_context(device_id),
                    confidence=self._calculate_confidence(anomalies)
                )
                
                suspicious_nodes.append(suspicious_node)
                
                # Store in database for model fine-tuning
                self._store_suspicious_node(suspicious_node)
        
        return suspicious_nodes
    
    def _extract_network_features(self, device_id: str, telemetry: Dict[str, Any]) -> Dict[str, Any]:
        """Extract network-based features for anomaly detection"""
        features = {}
        
        # Centrality features
        features['betweenness_centrality'] = self.centrality_metrics['betweenness'].get(device_id, 0)
        features['closeness_centrality'] = self.centrality_metrics['closeness'].get(device_id, 0)
        features['eigenvector_centrality'] = self.centrality_metrics['eigenvector'].get(device_id, 0)
        features['degree_centrality'] = self.centrality_metrics['degree'].get(device_id, 0)
        
        # Neighbor features
        neighbors = list(self.network_graph.neighbors(device_id))
        features['neighbor_count'] = len(neighbors)
        features['avg_neighbor_degree'] = np.mean([self.network_graph.degree(n) for n in neighbors]) if neighbors else 0
        
        # Traffic features
        features['total_traffic'] = telemetry.get('traffic_volume', 0)
        features['traffic_variance'] = telemetry.get('traffic_variance', 0)
        features['connection_count'] = telemetry.get('active_connections', 0)
        
        # Community features
        device_community = self._get_device_community(device_id)
        features['community_size'] = len(device_community) if device_community else 0
        features['community_traffic_ratio'] = self._calculate_community_traffic_ratio(device_id, device_community)
        
        return features
    
    def _detect_centrality_anomaly(self, device_id: str, features: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Detect anomalies based on centrality metrics"""
        # Get historical centrality values for this device
        historical_centrality = self._get_historical_centrality(device_id)
        
        if not historical_centrality:
            return None
        
        # Calculate z-scores for current centrality values
        z_scores = {}
        for metric in ['betweenness_centrality', 'closeness_centrality', 'eigenvector_centrality']:
            if metric in historical_centrality:
                mean_val = np.mean(historical_centrality[metric])
                std_val = np.std(historical_centrality[metric])
                if std_val > 0:
                    z_scores[metric] = abs(features[metric] - mean_val) / std_val
        
        # Detect anomaly if any z-score exceeds threshold
        max_z_score = max(z_scores.values()) if z_scores else 0
        if max_z_score > 2.5:  # 2.5 standard deviations
            return {
                'type': 'centrality_anomaly',
                'score': min(max_z_score / 5.0, 1.0),  # Normalize to [0,1]
                'details': z_scores
            }
        
        return None
    
    def _detect_community_anomaly(self, device_id: str, features: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Detect anomalies based on community structure"""
        device_community = self._get_device_community(device_id)
        
        if not device_community:
            return None
        
        # Check if device is communicating outside its community more than usual
        community_traffic_ratio = features['community_traffic_ratio']
        
        if community_traffic_ratio < 0.3:  # Less than 30% traffic within community
            return {
                'type': 'community_anomaly',
                'score': 1.0 - community_traffic_ratio,
                'details': {'community_traffic_ratio': community_traffic_ratio}
            }
        
        return None
    
    def _detect_path_anomaly(self, device_id: str, features: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Detect anomalies based on network path analysis"""
        # Check if device is acting as an unexpected bridge
        if features['betweenness_centrality'] > 0.8 and features['degree_centrality'] < 0.3:
            return {
                'type': 'path_anomaly',
                'score': features['betweenness_centrality'],
                'details': {
                    'betweenness': features['betweenness_centrality'],
                    'degree': features['degree_centrality']
                }
            }
        
        return None
    
    def _detect_traffic_anomaly(self, device_id: str, telemetry: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Detect anomalies based on traffic patterns"""
        # Get historical traffic patterns
        historical_traffic = self._get_historical_traffic(device_id)
        
        if not historical_traffic:
            return None
        
        current_traffic = telemetry.get('traffic_volume', 0)
        mean_traffic = np.mean(historical_traffic)
        std_traffic = np.std(historical_traffic)
        
        if std_traffic > 0:
            z_score = abs(current_traffic - mean_traffic) / std_traffic
            if z_score > 3.0:  # 3 standard deviations
                return {
                    'type': 'traffic_anomaly',
                    'score': min(z_score / 5.0, 1.0),
                    'details': {
                        'current_traffic': current_traffic,
                        'mean_traffic': mean_traffic,
                        'z_score': z_score
                    }
                }
        
        return None
    
    def _get_device_community(self, device_id: str) -> List[str]:
        """Get the community that a device belongs to"""
        for community_name, community_devices in self.community_structure.items():
            if device_id in community_devices:
                return community_devices
        return []
    
    def _calculate_community_traffic_ratio(self, device_id: str, community: List[str]) -> float:
        """Calculate the ratio of traffic within the device's community"""
        if not community:
            return 0.0
        
        total_traffic = 0
        community_traffic = 0
        
        for neighbor in self.network_graph.neighbors(device_id):
            edge_data = self.network_graph.get_edge_data(device_id, neighbor)
            traffic = edge_data.get('weight', 0)
            total_traffic += traffic
            
            if neighbor in community:
                community_traffic += traffic
        
        return community_traffic / total_traffic if total_traffic > 0 else 0.0
    
    def _get_network_context(self, device_id: str) -> Dict[str, Any]:
        """Get network context for a device"""
        return {
            'neighbors': list(self.network_graph.neighbors(device_id)),
            'community': self._get_device_community(device_id),
            'centrality_metrics': {
                metric: values.get(device_id, 0) 
                for metric, values in self.centrality_metrics.items()
            },
            'network_density': nx.density(self.network_graph),
            'total_nodes': self.network_graph.number_of_nodes(),
            'total_edges': self.network_graph.number_of_edges()
        }
    
    def _determine_anomaly_type(self, anomalies: List[Dict[str, Any]]) -> str:
        """Determine the primary anomaly type from multiple detected anomalies"""
        anomaly_types = [a['type'] for a in anomalies]
        
        # Priority order for anomaly types
        priority_order = ['traffic_anomaly', 'centrality_anomaly', 'community_anomaly', 'path_anomaly']
        
        for anomaly_type in priority_order:
            if anomaly_type in anomaly_types:
                return anomaly_type
        
        return 'unknown'
    
    def _calculate_confidence(self, anomalies: List[Dict[str, Any]]) -> float:
        """Calculate confidence score based on anomaly consistency"""
        if not anomalies:
            return 0.0
        
        # Higher confidence if multiple anomaly types agree
        scores = [a['score'] for a in anomalies]
        return np.mean(scores) * (1 + 0.1 * len(anomalies))
    
    def _store_suspicious_node(self, suspicious_node: SuspiciousNode):
        """Store suspicious node in database for model fine-tuning"""
        try:
            db_node = SuspiciousNodeDB(
                node_id=suspicious_node.node_id,
                anomaly_score=suspicious_node.anomaly_score,
                anomaly_type=suspicious_node.anomaly_type,
                timestamp=suspicious_node.timestamp,
                features=suspicious_node.features,
                network_context=suspicious_node.network_context,
                confidence=suspicious_node.confidence,
                requires_investigation=suspicious_node.requires_investigation
            )
            
            self.session.add(db_node)
            self.session.commit()
            
            logging.info(f"Stored suspicious node {suspicious_node.node_id} with score {suspicious_node.anomaly_score}")
            
        except Exception as e:
            logging.error(f"Failed to store suspicious node: {e}")
            self.session.rollback()
    
    def get_suspicious_nodes_for_training(self, limit: int = 1000) -> List[Dict[str, Any]]:
        """Retrieve suspicious nodes for model fine-tuning"""
        try:
            nodes = self.session.query(SuspiciousNodeDB).filter(
                SuspiciousNodeDB.investigation_status.in_(['resolved', 'false_positive'])
            ).limit(limit).all()
            
            return [
                {
                    'node_id': node.node_id,
                    'anomaly_score': node.anomaly_score,
                    'anomaly_type': node.anomaly_type,
                    'features': node.features,
                    'network_context': node.network_context,
                    'investigation_status': node.investigation_status,
                    'model_feedback': node.model_feedback
                }
                for node in nodes
            ]
            
        except Exception as e:
            logging.error(f"Failed to retrieve suspicious nodes: {e}")
            return []
    
    def update_investigation_status(self, node_id: str, status: str, notes: str = "", feedback: Dict[str, Any] = None):
        """Update investigation status and provide feedback for model fine-tuning"""
        try:
            node = self.session.query(SuspiciousNodeDB).filter(
                SuspiciousNodeDB.node_id == node_id
            ).first()
            
            if node:
                node.investigation_status = status
                node.investigation_notes = notes
                if feedback:
                    node.model_feedback = feedback
                
                self.session.commit()
                logging.info(f"Updated investigation status for node {node_id} to {status}")
            
        except Exception as e:
            logging.error(f"Failed to update investigation status: {e}")
            self.session.rollback()
    
    # Placeholder methods for historical data (would be implemented with actual data storage)
    def _get_historical_centrality(self, device_id: str) -> Dict[str, List[float]]:
        """Get historical centrality values for a device"""
        # This would query a time-series database
        return {}
    
    def _get_historical_traffic(self, device_id: str) -> List[float]:
        """Get historical traffic values for a device"""
        # This would query a time-series database
        return [] 