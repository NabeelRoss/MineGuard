"""
Alert Management System for MineGuard AI.

Maintains an in-memory event buffer for real-time dashboard notifications and browser audio alerts,
enforces duplicate notification prevention, and dispatches alerts when a checkpoint inspection
produces a confirmed FAIL decision.
"""

import threading
from collections import deque
from typing import List, Dict, Any, Optional
from inspection.inspection_result import InspectionResult


class AlertManager:
    """
    Thread-safe alert dispatcher managing confirmed safety violations.
    """

    def __init__(self, max_buffer_size: int = 50):
        self._lock = threading.Lock()
        self._recent_alerts: deque = deque(maxlen=max_buffer_size)
        self._unacknowledged_alerts: List[Dict[str, Any]] = []

    def dispatch_inspection_alert(self, result: InspectionResult):
        """
        Processes a terminal inspection result.
        If decision is FAIL, dispatches notification to queue for browser toast and audio chime.
        """
        if result.decision != "FAIL":
            return

        alert_dict = {
            "inspection_reference": result.inspection_reference,
            "timestamp": result.completion_timestamp,
            "missing_ppe": result.missing_ppe,
            "decision": result.decision,
            "confidence": round(result.average_confidence, 2),
            "reason": result.reinspection_reason,
            "annotated_image_path": result.annotated_image_path,
        }

        with self._lock:
            # Check duplicate by inspection reference
            if not any(a.get("inspection_reference") == result.inspection_reference for a in self._recent_alerts):
                self._recent_alerts.appendleft(alert_dict)
                self._unacknowledged_alerts.append(alert_dict)

    def get_recent_alerts(self, limit: int = 10) -> List[Dict[str, Any]]:
        """Returns the most recent violation alerts."""
        with self._lock:
            return list(self._recent_alerts)[:limit]

    def poll_new_alerts(self) -> List[Dict[str, Any]]:
        """
        Returns all unacknowledged alerts since last poll and clears the unacknowledged queue.
        Used by the frontend to trigger audio chimes and modal toasts.
        """
        with self._lock:
            alerts = list(self._unacknowledged_alerts)
            self._unacknowledged_alerts.clear()
            return alerts

    def clear(self):
        """Clears in-memory alert buffers."""
        with self._lock:
            self._recent_alerts.clear()
            self._unacknowledged_alerts.clear()


# Global Singleton Alert Manager
alert_manager = AlertManager()
