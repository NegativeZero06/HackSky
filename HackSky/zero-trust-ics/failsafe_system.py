import asyncio
import logging
import time
import json
import hashlib
from typing import Dict, List, Any, Optional, Callable
from datetime import datetime, timedelta
from dataclasses import dataclass, field
from enum import Enum
import threading
import signal
import sys

class FailsafeLevel(Enum):
    NORMAL = "normal"
    WARNING = "warning"
    CRITICAL = "critical"
    EMERGENCY = "emergency"

class FailsafeAction(Enum):
    MONITOR = "monitor"
    ISOLATE = "isolate"
    SHUTDOWN = "shutdown"
    ROLLBACK = "rollback"
    EMERGENCY_STOP = "emergency_stop"

@dataclass
class FailsafeEvent:
    event_id: str
    level: FailsafeLevel
    action: FailsafeAction
    device_id: str
    description: str
    timestamp: datetime
    metadata: Dict[str, Any] = field(default_factory=dict)
    resolved: bool = False
    resolution_time: Optional[datetime] = None

@dataclass
class SystemState:
    state_id: str
    device_id: str
    state_hash: str
    state_data: Dict[str, Any]
    timestamp: datetime
    is_verified: bool = False
    backup_location: Optional[str] = None

class FailsafeSystem:
    def __init__(self):
        self.failsafe_events: List[FailsafeEvent] = []
        self.system_states: Dict[str, List[SystemState]] = {}
        self.recovery_procedures: Dict[str, Callable] = {}
        self.monitoring_threads: Dict[str, threading.Thread] = {}
        self.emergency_contacts: List[str] = []
        self.failsafe_config = self._load_failsafe_config()
        
        # Initialize recovery procedures
        self._initialize_recovery_procedures()
        
        # Set up signal handlers for graceful shutdown
        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)
        
        # Start monitoring threads
        self._start_monitoring_threads()
        
    def _load_failsafe_config(self) -> Dict[str, Any]:
        """Load failsafe configuration"""
        return {
            'max_response_time': 10.0,  # seconds
            'max_rollback_time': 300.0,  # seconds
            'emergency_threshold': 3,  # number of critical events
            'auto_recovery_enabled': True,
            'manual_override_required': False,
            'notification_channels': ['email', 'sms', 'dashboard'],
            'backup_retention_days': 30,
            'health_check_interval': 5.0,  # seconds
            'state_verification_interval': 60.0,  # seconds
            'emergency_contacts': [
                'admin@company.com',
                '+1234567890'
            ]
        }
    
    def _initialize_recovery_procedures(self):
        """Initialize recovery procedures for different failure scenarios"""
        self.recovery_procedures = {
            'network_isolation': self._recover_network_isolation,
            'device_compromise': self._recover_device_compromise,
            'data_corruption': self._recover_data_corruption,
            'system_overload': self._recover_system_overload,
            'authentication_failure': self._recover_authentication_failure,
            'communication_failure': self._recover_communication_failure,
            'sensor_failure': self._recover_sensor_failure,
            'control_system_failure': self._recover_control_system_failure
        }
    
    def _start_monitoring_threads(self):
        """Start background monitoring threads"""
        # Health monitoring thread
        health_thread = threading.Thread(target=self._health_monitoring_loop, daemon=True)
        health_thread.start()
        self.monitoring_threads['health'] = health_thread
        
        # State verification thread
        state_thread = threading.Thread(target=self._state_verification_loop, daemon=True)
        state_thread.start()
        self.monitoring_threads['state'] = state_thread
        
        # Event cleanup thread
        cleanup_thread = threading.Thread(target=self._event_cleanup_loop, daemon=True)
        cleanup_thread.start()
        self.monitoring_threads['cleanup'] = cleanup_thread
    
    async def create_system_snapshot(self, device_id: str, state_data: Dict[str, Any]) -> SystemState:
        """Create a system snapshot for rollback capability"""
        state_hash = self._calculate_state_hash(state_data)
        
        state = SystemState(
            state_id=f"state_{int(time.time())}_{device_id}",
            device_id=device_id,
            state_hash=state_hash,
            state_data=state_data,
            timestamp=datetime.utcnow()
        )
        
        # Store state
        if device_id not in self.system_states:
            self.system_states[device_id] = []
        
        self.system_states[device_id].append(state)
        
        # Keep only last 10 snapshots per device
        if len(self.system_states[device_id]) > 10:
            self.system_states[device_id] = self.system_states[device_id][-10:]
        
        # Backup to persistent storage
        await self._backup_state(state)
        
        logging.info(f"Created system snapshot for device {device_id}")
        return state
    
    def _calculate_state_hash(self, state_data: Dict[str, Any]) -> str:
        """Calculate SHA256 hash of state data"""
        state_str = json.dumps(state_data, sort_keys=True)
        return hashlib.sha256(state_str.encode()).hexdigest()
    
    async def _backup_state(self, state: SystemState):
        """Backup state to persistent storage"""
        try:
            backup_data = {
                'state_id': state.state_id,
                'device_id': state.device_id,
                'state_hash': state.state_hash,
                'state_data': state.state_data,
                'timestamp': state.timestamp.isoformat(),
                'is_verified': state.is_verified
            }
            
            # Save to file (in production, this would be a database)
            backup_path = f"backups/{state.device_id}_{state.state_id}.json"
            with open(backup_path, 'w') as f:
                json.dump(backup_data, f, indent=2)
            
            state.backup_location = backup_path
            logging.info(f"State backed up to {backup_path}")
            
        except Exception as e:
            logging.error(f"Failed to backup state: {e}")
    
    async def trigger_failsafe(self, level: FailsafeLevel, action: FailsafeAction, 
                             device_id: str, description: str, metadata: Dict[str, Any] = None) -> FailsafeEvent:
        """Trigger a failsafe event"""
        event = FailsafeEvent(
            event_id=f"failsafe_{int(time.time())}_{device_id}",
            level=level,
            action=action,
            device_id=device_id,
            description=description,
            timestamp=datetime.utcnow(),
            metadata=metadata or {}
        )
        
        self.failsafe_events.append(event)
        
        # Execute failsafe action
        await self._execute_failsafe_action(event)
        
        # Send notifications
        await self._send_notifications(event)
        
        # Log the event
        logging.warning(f"Failsafe triggered: {event.level.value} - {event.action.value} for {device_id}")
        
        return event
    
    async def _execute_failsafe_action(self, event: FailsafeEvent):
        """Execute the appropriate failsafe action"""
        try:
            if event.action == FailsafeAction.MONITOR:
                await self._action_monitor(event)
            elif event.action == FailsafeAction.ISOLATE:
                await self._action_isolate(event)
            elif event.action == FailsafeAction.SHUTDOWN:
                await self._action_shutdown(event)
            elif event.action == FailsafeAction.ROLLBACK:
                await self._action_rollback(event)
            elif event.action == FailsafeAction.EMERGENCY_STOP:
                await self._action_emergency_stop(event)
            
            # Check if emergency threshold is reached
            if self._check_emergency_threshold():
                await self._trigger_emergency_protocol()
                
        except Exception as e:
            logging.error(f"Failed to execute failsafe action: {e}")
            # If action fails, escalate to emergency stop
            await self._action_emergency_stop(event)
    
    async def _action_monitor(self, event: FailsafeEvent):
        """Monitor the device closely"""
        logging.info(f"Enhanced monitoring enabled for device {event.device_id}")
        # Implement enhanced monitoring logic
        
    async def _action_isolate(self, event: FailsafeEvent):
        """Isolate the device from the network"""
        logging.warning(f"Isolating device {event.device_id} from network")
        # Implement network isolation logic
        
    async def _action_shutdown(self, event: FailsafeEvent):
        """Gracefully shutdown the device"""
        logging.critical(f"Shutting down device {event.device_id}")
        # Implement graceful shutdown logic
        
    async def _action_rollback(self, event: FailsafeEvent):
        """Rollback to last known good state"""
        logging.critical(f"Rolling back device {event.device_id} to last known good state")
        
        if event.device_id in self.system_states and self.system_states[event.device_id]:
            last_good_state = self.system_states[event.device_id][-1]
            await self._perform_rollback(event.device_id, last_good_state)
        else:
            logging.error(f"No rollback state available for device {event.device_id}")
    
    async def _action_emergency_stop(self, event: FailsafeEvent):
        """Emergency stop - immediate shutdown"""
        logging.critical(f"EMERGENCY STOP for device {event.device_id}")
        # Implement emergency stop logic
        
    async def _perform_rollback(self, device_id: str, target_state: SystemState):
        """Perform rollback to target state"""
        try:
            # Verify state integrity
            if not self._verify_state_integrity(target_state):
                raise Exception("Target state integrity verification failed")
            
            # Stop current operations
            await self._stop_device_operations(device_id)
            
            # Restore state
            await self._restore_device_state(device_id, target_state.state_data)
            
            # Verify restoration
            if await self._verify_restoration(device_id, target_state):
                logging.info(f"Rollback successful for device {device_id}")
                await self._resume_device_operations(device_id)
            else:
                raise Exception("Restoration verification failed")
                
        except Exception as e:
            logging.error(f"Rollback failed for device {device_id}: {e}")
            # If rollback fails, trigger emergency stop
            await self.trigger_failsafe(
                FailsafeLevel.EMERGENCY,
                FailsafeAction.EMERGENCY_STOP,
                device_id,
                f"Rollback failed: {str(e)}"
            )
    
    def _verify_state_integrity(self, state: SystemState) -> bool:
        """Verify the integrity of a system state"""
        current_hash = self._calculate_state_hash(state.state_data)
        return current_hash == state.state_hash
    
    async def _stop_device_operations(self, device_id: str):
        """Stop device operations safely"""
        logging.info(f"Stopping operations for device {device_id}")
        # Implement device operation stopping logic
        
    async def _restore_device_state(self, device_id: str, state_data: Dict[str, Any]):
        """Restore device to specified state"""
        logging.info(f"Restoring state for device {device_id}")
        # Implement state restoration logic
        
    async def _verify_restoration(self, device_id: str, target_state: SystemState) -> bool:
        """Verify that restoration was successful"""
        # Implement restoration verification logic
        return True
    
    async def _resume_device_operations(self, device_id: str):
        """Resume device operations"""
        logging.info(f"Resuming operations for device {device_id}")
        # Implement device operation resumption logic
    
    def _check_emergency_threshold(self) -> bool:
        """Check if emergency threshold has been reached"""
        recent_critical_events = [
            event for event in self.failsafe_events
            if event.level in [FailsafeLevel.CRITICAL, FailsafeLevel.EMERGENCY]
            and event.timestamp > datetime.utcnow() - timedelta(minutes=5)
        ]
        
        return len(recent_critical_events) >= self.failsafe_config['emergency_threshold']
    
    async def _trigger_emergency_protocol(self):
        """Trigger emergency protocol when threshold is reached"""
        logging.critical("EMERGENCY PROTOCOL TRIGGERED - Multiple critical events detected")
        
        # Notify emergency contacts
        await self._notify_emergency_contacts()
        
        # Implement emergency protocol logic
        # This could include shutting down non-critical systems,
        # activating backup systems, etc.
    
    async def _send_notifications(self, event: FailsafeEvent):
        """Send notifications for failsafe events"""
        notification_message = f"Failsafe Event: {event.level.value} - {event.action.value} for {event.device_id}"
        
        for channel in self.failsafe_config['notification_channels']:
            try:
                if channel == 'email':
                    await self._send_email_notification(notification_message, event)
                elif channel == 'sms':
                    await self._send_sms_notification(notification_message, event)
                elif channel == 'dashboard':
                    await self._send_dashboard_notification(notification_message, event)
            except Exception as e:
                logging.error(f"Failed to send {channel} notification: {e}")
    
    async def _notify_emergency_contacts(self):
        """Notify emergency contacts"""
        emergency_message = "EMERGENCY: Multiple critical failsafe events detected. Immediate attention required."
        
        for contact in self.failsafe_config['emergency_contacts']:
            try:
                if '@' in contact:
                    await self._send_email_notification(emergency_message, None, contact)
                else:
                    await self._send_sms_notification(emergency_message, None, contact)
            except Exception as e:
                logging.error(f"Failed to notify emergency contact {contact}: {e}")
    
    async def _send_email_notification(self, message: str, event: Optional[FailsafeEvent] = None, recipient: str = None):
        """Send email notification"""
        # Implement email notification logic
        logging.info(f"Email notification sent: {message}")
    
    async def _send_sms_notification(self, message: str, event: Optional[FailsafeEvent] = None, recipient: str = None):
        """Send SMS notification"""
        # Implement SMS notification logic
        logging.info(f"SMS notification sent: {message}")
    
    async def _send_dashboard_notification(self, message: str, event: Optional[FailsafeEvent] = None):
        """Send dashboard notification"""
        # Implement dashboard notification logic
        logging.info(f"Dashboard notification sent: {message}")
    
    def _health_monitoring_loop(self):
        """Background health monitoring loop"""
        while True:
            try:
                # Check system health
                health_status = self._check_system_health()
                
                if health_status['overall_health'] < 0.5:
                    # Trigger failsafe if health is poor
                    asyncio.run(self.trigger_failsafe(
                        FailsafeLevel.WARNING,
                        FailsafeAction.MONITOR,
                        'system',
                        f"System health degraded: {health_status['overall_health']}"
                    ))
                
                time.sleep(self.failsafe_config['health_check_interval'])
                
            except Exception as e:
                logging.error(f"Health monitoring error: {e}")
                time.sleep(5)  # Wait before retrying
    
    def _state_verification_loop(self):
        """Background state verification loop"""
        while True:
            try:
                # Verify system states
                for device_id, states in self.system_states.items():
                    for state in states:
                        if not state.is_verified:
                            state.is_verified = self._verify_state_integrity(state)
                
                time.sleep(self.failsafe_config['state_verification_interval'])
                
            except Exception as e:
                logging.error(f"State verification error: {e}")
                time.sleep(10)  # Wait before retrying
    
    def _event_cleanup_loop(self):
        """Background event cleanup loop"""
        while True:
            try:
                # Remove old events
                cutoff_time = datetime.utcnow() - timedelta(days=self.failsafe_config['backup_retention_days'])
                self.failsafe_events = [
                    event for event in self.failsafe_events
                    if event.timestamp > cutoff_time
                ]
                
                time.sleep(3600)  # Run every hour
                
            except Exception as e:
                logging.error(f"Event cleanup error: {e}")
                time.sleep(3600)  # Wait before retrying
    
    def _check_system_health(self) -> Dict[str, Any]:
        """Check overall system health"""
        # Implement system health checking logic
        return {
            'overall_health': 0.8,  # Placeholder
            'cpu_usage': 0.3,
            'memory_usage': 0.4,
            'network_health': 0.9,
            'storage_health': 0.7
        }
    
    def _signal_handler(self, signum, frame):
        """Handle shutdown signals gracefully"""
        logging.info(f"Received signal {signum}, initiating graceful shutdown")
        
        # Stop monitoring threads
        for thread_name, thread in self.monitoring_threads.items():
            if thread.is_alive():
                logging.info(f"Stopping {thread_name} thread")
                # In a real implementation, you would set a stop flag
        
        # Perform cleanup
        self._cleanup()
        
        logging.info("Failsafe system shutdown complete")
        sys.exit(0)
    
    def _cleanup(self):
        """Perform cleanup operations"""
        # Save current state
        # Close database connections
        # Clean up temporary files
        logging.info("Performing cleanup operations")
    
    async def resolve_failsafe_event(self, event_id: str, resolution_notes: str = ""):
        """Mark a failsafe event as resolved"""
        for event in self.failsafe_events:
            if event.event_id == event_id:
                event.resolved = True
                event.resolution_time = datetime.utcnow()
                logging.info(f"Failsafe event {event_id} resolved: {resolution_notes}")
                break
    
    def get_active_failsafe_events(self) -> List[FailsafeEvent]:
        """Get all active (unresolved) failsafe events"""
        return [event for event in self.failsafe_events if not event.resolved]
    
    def get_failsafe_statistics(self) -> Dict[str, Any]:
        """Get failsafe system statistics"""
        total_events = len(self.failsafe_events)
        active_events = len(self.get_active_failsafe_events())
        resolved_events = total_events - active_events
        
        events_by_level = {}
        for level in FailsafeLevel:
            events_by_level[level.value] = len([
                event for event in self.failsafe_events
                if event.level == level
            ])
        
        return {
            'total_events': total_events,
            'active_events': active_events,
            'resolved_events': resolved_events,
            'events_by_level': events_by_level,
            'system_health': self._check_system_health()
        } 