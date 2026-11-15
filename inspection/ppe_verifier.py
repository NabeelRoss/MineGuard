"""
PPE Verifier Module for MineGuard AI.

Implements multi-frame consensus rules for checkpoint compliance decisions.
Guarantees:
1. SAFE confirmation requires at least 3 reliable compliant frames.
2. VIOLATION confirmation requires at least 5 reliable non-compliant frames.
3. UNKNOWN / Ambiguous is NEVER converted to PASS.
4. Insufficient visibility or timeout resolves strictly to REINSPECT.
"""

from typing import Tuple
import utils.config as config
from inspection.inspection_result import InspectionDecision, PPEStatus
from inspection.frame_collector import FrameCollector


class PPEVerifier:
    """
    Evaluates accumulated multi-frame observations to reach an inspection decision.
    """

    def __init__(
        self,
        safe_confirm_frames: int = config.REQUIRED_SAFE_OBSERVATIONS,
        fail_confirm_frames: int = config.REQUIRED_VIOLATION_OBSERVATIONS,
        timeout_seconds: float = config.INSPECTION_TIMEOUT_SECONDS,
    ):
        self.safe_confirm_frames = safe_confirm_frames
        self.fail_confirm_frames = fail_confirm_frames
        self.timeout_seconds = timeout_seconds

    def evaluate_cycle(
        self,
        collector: FrameCollector,
        elapsed_seconds: float,
        is_multiple_workers: bool = False,
    ) -> Tuple[InspectionDecision, str, str, str]:
        """
        Evaluates current accumulated observations in collector.
        Returns:
            (decision, helmet_status, vest_status, reason)
        """
        # If multiple workers were in the zone, immediately trigger REINSPECT
        if is_multiple_workers:
            return (
                InspectionDecision.REINSPECT,
                PPEStatus.UNCERTAIN.value,
                PPEStatus.UNCERTAIN.value,
                "Multiple workers entered inspection zone simultaneously",
            )

        if collector.total_frames == 0:
            return InspectionDecision.PENDING, PPEStatus.UNCERTAIN.value, PPEStatus.UNCERTAIN.value, "Awaiting frames"

        safe_count = collector.safe_observations_count
        violation_count = collector.violation_observations_count
        ambiguous_count = collector.ambiguous_observations_count

        # 1. Check for PASS confirmation: Requires at least safe_confirm_frames and ZERO violations
        if safe_count >= self.safe_confirm_frames and violation_count == 0:
            return (
                InspectionDecision.PASS,
                PPEStatus.PRESENT.value,
                PPEStatus.PRESENT.value,
                "Fully compliant (Hardhat and Safety Vest verified)",
            )

        # 2. Check for FAIL confirmation: Requires at least fail_confirm_frames
        if violation_count >= self.fail_confirm_frames:
            # Determine specific missing PPE
            recent_violations = [o for o in collector.observations if o.is_violation]
            helmet_missing_count = sum(1 for o in recent_violations if o.helmet_status == PPEStatus.MISSING)
            vest_missing_count = sum(1 for o in recent_violations if o.vest_status == PPEStatus.MISSING)

            if helmet_missing_count >= 3 and vest_missing_count >= 3:
                h_stat = PPEStatus.MISSING.value
                v_stat = PPEStatus.MISSING.value
                reason = "Missing Hardhat & Safety Vest"
            elif helmet_missing_count >= 3:
                h_stat = PPEStatus.MISSING.value
                v_stat = PPEStatus.PRESENT.value if safe_count > 0 else PPEStatus.UNCERTAIN.value
                reason = "Missing Hardhat"
            elif vest_missing_count >= 3:
                h_stat = PPEStatus.PRESENT.value if safe_count > 0 else PPEStatus.UNCERTAIN.value
                v_stat = PPEStatus.MISSING.value
                reason = "Missing Safety Vest"
            else:
                h_stat = PPEStatus.MISSING.value
                v_stat = PPEStatus.MISSING.value
                reason = "Non-compliant PPE verified"

            return InspectionDecision.FAIL, h_stat, v_stat, reason

        # 3. Check for Timeout Exceeded -> Enforce REINSPECT
        if elapsed_seconds >= self.timeout_seconds:
            # If excessive ambiguity or could not reach consensus
            reason = (
                "Inspection timed out before reliable consensus"
                if ambiguous_count == 0
                else "Inspection timed out due to visual ambiguity / occlusion"
            )
            return (
                InspectionDecision.REINSPECT,
                PPEStatus.UNCERTAIN.value,
                PPEStatus.UNCERTAIN.value,
                reason,
            )

        # Still gathering evidence
        return (
            InspectionDecision.PENDING,
            PPEStatus.UNCERTAIN.value,
            PPEStatus.UNCERTAIN.value,
            f"Verifying ({safe_count}/{self.safe_confirm_frames} safe, {violation_count}/{self.fail_confirm_frames} viol)",
        )
