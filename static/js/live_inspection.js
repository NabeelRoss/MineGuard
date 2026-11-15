/**
 * MineGuard AI - Live Checkpoint Inspection Controller
 * Polls active station state, controls camera streaming, and renders live consensus.
 */

document.addEventListener("DOMContentLoaded", () => {
    // DOM Elements
    const liveRefId = document.getElementById("liveRefId");
    const liveStatePill = document.getElementById("liveStatePill");
    const checkpointStateBadge = document.getElementById("checkpointStateBadge");
    const liveTimer = document.getElementById("liveTimer");
    const liveDecisionBanner = document.getElementById("liveDecisionBanner");
    const liveDecisionTitle = document.getElementById("liveDecisionTitle");
    const liveDecisionSub = document.getElementById("liveDecisionSub");

    const liveHardhatStatus = document.getElementById("liveHardhatStatus");
    const liveHardhatConf = document.getElementById("liveHardhatConf");
    const liveHardhatFrames = document.getElementById("liveHardhatFrames");
    const liveHardhatBar = document.getElementById("liveHardhatBar");

    const liveVestStatus = document.getElementById("liveVestStatus");
    const liveVestConf = document.getElementById("liveVestConf");
    const liveVestFrames = document.getElementById("liveVestFrames");
    const liveVestBar = document.getElementById("liveVestBar");

    const videoOverlayStatus = document.getElementById("videoOverlayStatus");
    const videoStatusDot = document.getElementById("videoStatusDot");
    const videoStatusText = document.getElementById("videoStatusText");
    const videoOverlayFps = document.getElementById("videoOverlayFps");
    const recentInspectionsList = document.getElementById("recentInspectionsList");

    const btnStart = document.getElementById("btnStartMonitoring");
    const btnStop = document.getElementById("btnStopMonitoring");
    const btnReconnect = document.getElementById("btnReconnectCam");

    let lastKnownDecision = "IDLE";
    let lastRefId = null;

    // Helper: Map state to CSS badge
    function updateStateBadge(state) {
        liveStatePill.className = "";
        let pillClass = "badge-idle";
        if (state === "DETECTING") pillClass = "badge-detecting";
        else if (state === "VERIFYING") pillClass = "badge-verifying";
        else if (state === "PASS") pillClass = "badge-pass";
        else if (state === "FAIL") pillClass = "badge-fail";
        else if (state === "REINSPECT") pillClass = "badge-reinspect";

        liveStatePill.className = pillClass;
        liveStatePill.textContent = state;

        if (checkpointStateBadge) {
            checkpointStateBadge.innerHTML = `<span class="mg-status-dot me-1"></span> State: ${state}`;
        }
    }

    // Helper: Render PPE Pill
    function renderPPEPill(status, el) {
        if (status === "PASS" || status === "PRESENT") {
            el.className = "badge bg-success text-white small";
            el.textContent = "COMPLIANT";
        } else if (status === "FAIL" || status === "MISSING") {
            el.className = "badge bg-danger text-white small";
            el.textContent = "NON-COMPLIANT";
        } else {
            el.className = "badge bg-secondary text-white small";
            el.textContent = "UNKNOWN";
        }
    }

    // Helper: Render Decision Banner
    function updateDecisionBanner(state, decision, reason, lastResult) {
        liveDecisionBanner.className = "mg-decision-banner";

        if (state === "IDLE") {
            liveDecisionTitle.textContent = "READY FOR WORKER";
            liveDecisionTitle.className = "mg-decision-title text-secondary";
            liveDecisionSub.textContent = "Step into designated checkpoint box and face camera";
        } else if (state === "DETECTING") {
            liveDecisionTitle.textContent = "WORKER DETECTED";
            liveDecisionTitle.className = "mg-decision-title text-warning";
            liveDecisionSub.textContent = "Stabilizing position inside inspection zone...";
        } else if (state === "VERIFYING") {
            liveDecisionBanner.classList.add("verifying");
            liveDecisionTitle.textContent = "INSPECTING PPE...";
            liveDecisionTitle.className = "mg-decision-title text-cyan";
            liveDecisionSub.textContent = reason || "Analyzing hardhat & safety vest presence across frames";
        } else if (state === "PASS" || decision === "PASS") {
            liveDecisionBanner.classList.add("pass");
            liveDecisionTitle.textContent = "PASS - COMPLIANT";
            liveDecisionTitle.className = "mg-decision-title text-success";
            liveDecisionSub.textContent = reason || "All required PPE verified. Worker cleared for underground entry.";
        } else if (state === "FAIL" || decision === "FAIL") {
            liveDecisionBanner.classList.add("fail");
            liveDecisionTitle.textContent = "FAIL - ACCESS DENIED";
            liveDecisionTitle.className = "mg-decision-title text-danger";
            liveDecisionSub.textContent = reason || "Required safety equipment missing. Entry prohibited.";
        } else if (state === "REINSPECT" || decision === "REINSPECT") {
            liveDecisionBanner.classList.add("reinspect");
            liveDecisionTitle.textContent = "RE-INSPECTION NEEDED";
            liveDecisionTitle.className = "mg-decision-title text-warning";
            liveDecisionSub.textContent = reason || "Worker position occluded or multiple workers in zone. Reposition.";
        }
    }

    // Main Telemetry Polling (every 400ms)
    async function pollCurrentInspection() {
        try {
            const res = await fetch("/api/inspection/current");
            if (!res.ok) return;
            const data = await res.json();

            const state = data.state || "IDLE";
            const ref = data.reference_id;
            const elapsed = (data.elapsed_seconds || 0).toFixed(1);

            // Update Header Ref ID
            if (ref) {
                liveRefId.textContent = ref;
                liveRefId.className = "fw-bold font-monospace text-warning";
            } else {
                liveRefId.textContent = "STANDBY - NO WORKER";
                liveRefId.className = "fw-bold font-monospace text-secondary";
            }

            liveTimer.textContent = `${elapsed}s`;
            updateStateBadge(state);

            // Update Decision Banner
            const lastRes = data.last_result || {};
            const decision = lastRes.decision || null;
            const reason = lastRes.notes || data.reason || "";
            updateDecisionBanner(state, decision, reason, lastRes);

            // Check if a terminal FAIL just occurred to play alarm
            if (state === "FAIL" && lastKnownDecision !== "FAIL") {
                if (typeof playSafetyAlarm === "function") {
                    playSafetyAlarm();
                }
            }
            lastKnownDecision = state;

            // Update PPE consensus bars
            const hh = data.hardhat || {};
            const vest = data.vest || {};

            renderPPEPill(hh.status || "UNKNOWN", liveHardhatStatus);
            const hhConf = Math.round((hh.confidence || 0) * 100);
            liveHardhatConf.textContent = `${hhConf}%`;
            liveHardhatFrames.textContent = hh.frames_observed || 0;
            liveHardhatBar.style.width = `${hhConf}%`;
            if (hh.status === "PASS" || hh.status === "PRESENT") {
                liveHardhatBar.className = "mg-progress-bar safe";
            } else if (hh.status === "FAIL" || hh.status === "MISSING") {
                liveHardhatBar.className = "mg-progress-bar violation";
            } else {
                liveHardhatBar.className = "mg-progress-bar brand";
            }

            renderPPEPill(vest.status || "UNKNOWN", liveVestStatus);
            const vestConf = Math.round((vest.confidence || 0) * 100);
            liveVestConf.textContent = `${vestConf}%`;
            liveVestFrames.textContent = vest.frames_observed || 0;
            liveVestBar.style.width = `${vestConf}%`;
            if (vest.status === "PASS" || vest.status === "PRESENT") {
                liveVestBar.className = "mg-progress-bar safe";
            } else if (vest.status === "FAIL" || vest.status === "MISSING") {
                liveVestBar.className = "mg-progress-bar violation";
            } else {
                liveVestBar.className = "mg-progress-bar cyan";
            }

        } catch (e) {
            console.warn("Error polling inspection status:", e);
        }
    }

    // Camera Status Polling (every 1.5s)
    async function pollCameraStatus() {
        try {
            const res = await fetch("/api/status");
            if (!res.ok) return;
            const data = await res.json();

            videoOverlayFps.textContent = `${data.current_fps.toFixed(1)} FPS`;

            if (data.monitoring_status === "ACTIVE") {
                videoStatusDot.className = "mg-status-dot pulse text-success";
                videoStatusText.textContent = "INSPECTION ACTIVE";
            } else {
                videoStatusDot.className = "mg-status-dot text-danger";
                videoStatusText.textContent = "STATION PAUSED";
            }
        } catch (e) {}
    }

    // Recent Inspections Polling (every 3s)
    async function loadRecentInspections() {
        try {
            const res = await fetch("/api/inspections?per_page=5");
            if (!res.ok) return;
            const data = await res.json();
            const list = data.inspections || [];

            if (list.length === 0) {
                recentInspectionsList.innerHTML = `<div class="text-muted small text-center py-3">No inspections recorded yet</div>`;
                return;
            }

            let html = "";
            list.forEach(item => {
                let badgeClass = "badge-idle";
                if (item.decision === "PASS") badgeClass = "badge-pass";
                else if (item.decision === "FAIL") badgeClass = "badge-fail";
                else if (item.decision === "REINSPECT") badgeClass = "badge-reinspect";

                html += `
                    <div class="p-2 rounded d-flex justify-content-between align-items-center" style="background-color: var(--bg-card); border: 1px solid var(--border-subtle);">
                        <div>
                            <div class="font-monospace text-white small fw-bold">${item.reference_id}</div>
                            <div class="text-muted small" style="font-size:0.7rem;">${item.timestamp} | ${(item.duration_seconds || 0).toFixed(1)}s</div>
                        </div>
                        <span class="${badgeClass}">${item.decision}</span>
                    </div>
                `;
            });
            recentInspectionsList.innerHTML = html;
        } catch (e) {}
    }

    // Controls listeners
    btnStart.addEventListener("click", async () => {
        try {
            const res = await fetch("/api/camera/start", { method: "POST" });
            const d = await res.json();
            if (res.ok) pollCameraStatus();
        } catch (e) { alert("Failed to start station inspection"); }
    });

    btnStop.addEventListener("click", async () => {
        try {
            const res = await fetch("/api/camera/stop", { method: "POST" });
            const d = await res.json();
            if (res.ok) pollCameraStatus();
        } catch (e) { alert("Failed to stop station inspection"); }
    });

    btnReconnect.addEventListener("click", async () => {
        try {
            btnReconnect.disabled = true;
            btnReconnect.innerHTML = `<span class="spinner-border spinner-border-sm me-1"></span> Reconnecting...`;
            const res = await fetch("/api/camera/reconnect", { method: "POST" });
            await res.json();
            setTimeout(() => {
                btnReconnect.disabled = false;
                btnReconnect.innerHTML = `<i class="bi bi-arrow-repeat"></i> Reconnect`;
                pollCameraStatus();
            }, 1000);
        } catch (e) {
            btnReconnect.disabled = false;
            btnReconnect.innerHTML = `<i class="bi bi-arrow-repeat"></i> Reconnect`;
        }
    });

    // Start intervals
    setInterval(pollCurrentInspection, 400);
    setInterval(pollCameraStatus, 1500);
    setInterval(loadRecentInspections, 3000);

    pollCurrentInspection();
    pollCameraStatus();
    loadRecentInspections();
});
