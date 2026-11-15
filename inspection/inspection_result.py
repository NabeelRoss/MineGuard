"""
Inspection Data Structures and Result Containers for MineGuard AI.

Defines checkpoint inspection states, decisions, per-frame observations,
and final immutable inspection records.
"""

from enum import Enum
from dataclasses import dataclass, field
from typing import List, Optional


class InspectionState(str, Enum):
    """Lifecycle states of the checkpoint inspection station."""
    IDLE = "IDLE"              # No worker in the inspection zone
    DETECTING = "DETECTING"    # Worker entered inspection zone, stabilizing
    VERIFYING = "VERIFYING"    # Multi-frame observation and PPE analysis in progress
    PASS = "PASS"              # Verified fully compliant (Helmet + Vest)
    FAIL = "FAIL"              # Confirmed missing PPE violation
    REINSPECT = "REINSPECT"    # Ambiguous, occluded, multiple workers, or timed out


class PPEStatus(str, Enum):
    """Status of an individual PPE component."""
    PRESENT = "PRESENT"        # Confidently detected and associated
    MISSING = "MISSING"        # Verified absent on visible anatomical zone
    UNCERTAIN = "UNCERTAIN"    # Insufficient visibility / ambiguous overlap


class InspectionDecision(str, Enum):
    """Terminal decision for a completed inspection."""
    PENDING = "PENDING"
    PASS = "PASS"
    FAIL = "FAIL"
    REINSPECT = "REINSPECT"


@dataclass
class InspectionFrameObservation:
    """Single-frame observation record collected during inspection."""
    timestamp: float
    frame_idx: int
    person_box: List[float]
    person_confidence: float
    helmet_status: PPEStatus
    vest_status: PPEStatus
    helmet_confidence: float = 0.0
    vest_confidence: float = 0.0
    is_ambiguous: bool = False
    ambiguity_reason: str = ""

    @property
    def is_safe(self) -> bool:
        return (
            self.helmet_status == PPEStatus.PRESENT
            and self.vest_status == PPEStatus.PRESENT
            and not self.is_ambiguous
        )

    @property
    def is_violation(self) -> bool:
        return (
            (self.helmet_status == PPEStatus.MISSING or self.vest_status == PPEStatus.MISSING)
            and not self.is_ambiguous
        )


@dataclass
class InspectionResult:
    """Immutable record of an individual worker inspection event."""
    inspection_reference: str
    entry_timestamp: float
    completion_timestamp: float
    duration: float
    helmet_status: str                     # 'PRESENT', 'MISSING', 'UNCERTAIN'
    vest_status: str                       # 'PRESENT', 'MISSING', 'UNCERTAIN'
    decision: str                          # 'PASS', 'FAIL', 'REINSPECT'
    person_confidence: float
    helmet_confidence: float
    vest_confidence: float
    average_confidence: float
    reinspection_reason: str = ""
    missing_ppe: str = ""
    original_image_path: Optional[str] = None
    annotated_image_path: Optional[str] = None
    session_id: Optional[int] = None
    total_frames_analyzed: int = 0
