/**
 * MineGuard AI - Settings & System Health Controller
 * Manages configuration updates, hardware telemetry polling, and database maintenance.
 */

document.addEventListener("DOMContentLoaded", () => {
    // Health Telemetry Elements
    const healthDevice = document.getElementById("healthDevice");
    const healthCudaVer = document.getElementById("healthCudaVer");
    const healthLatency = document.getElementById("healthLatency");
    const healthFps = document.getElementById("healthFps");
    const healthGpuMem = document.getElementById("healthGpuMem");
    const healthGpuReserved = document.getElementById("healthGpuReserved");
    const healthDbSize = document.getElementById("healthDbSize");
    const healthCamDriver = document.getElementById("healthCamDriver");

    // Form inputs
    const cfgCameraSource = document.getElementById("cfgCameraSource");
    const cfgConfPerson = document.getElementById("cfgConfPerson");
    const cfgConfHardhat = document.getElementById("cfgConfHardhat");
    const cfgConfVest = document.getElementById("cfgConfVest");
    const cfgIoU = document.getElementById("cfgIoU");

    const cfgSafeFrames = document.getElementById("cfgSafeFrames");
    const cfgFailFrames = document.getElementById("cfgFailFrames");
    const cfgTimeout = document.getElementById("cfgTimeout");
    const cfgCooldown = document.getElementById("cfgCooldown");

    const btnSaveDetection = document.getElementById("btnSaveDetectionSettings");
    const btnSaveConsensus = document.getElementById("btnSaveConsensusSettings");

    const btnClearInspections = document.getElementById("btnClearInspectionsBtn");
    const btnClearViolations = document.getElementById("btnClearViolationsBtn");
    const btnClearAll = document.getElementById("btnClearAllHistoryBtn");

    // Fetch live hardware telemetry
    async function pollHardwareHealth() {
        try {
            const res = await fetch("/api/system/health");
            if (!res.ok) return;
            const h = await res.json();

            // Device
            if (h.cuda_available) {
                healthDevice.textContent = h.device_name || "CUDA GPU";
                healthCudaVer.textContent = `PyTorch CUDA ${h.cuda_version || "12.x"}`;
                healthGpuMem.textContent = `${(h.gpu_memory_allocated_mb || 0).toFixed(1)} MB Allocated`;
                healthGpuReserved.textContent = `Reserved: ${(h.gpu_memory_reserved_mb || 0).toFixed(1)} MB`;
            } else {
                healthDevice.textContent = "CPU (x86_64)";
                healthCudaVer.textContent = "No CUDA GPU Acceleration";
                healthGpuMem.textContent = "CPU RAM Only";
                healthGpuReserved.textContent = "N/A";
            }

            // Performance
            healthLatency.textContent = `${(h.inference_latency_ms || 0).toFixed(1)} ms`;
            healthFps.textContent = `Station FPS: ${(h.current_fps || 0).toFixed(1)}`;

            // Storage & Camera
            const dbKb = Math.round((h.database_size_bytes || 0) / 1024);
            healthDbSize.textContent = `DB: ${dbKb} KB (WAL)`;
            healthCamDriver.textContent = `Camera: ${h.camera_status} | Model: ${h.model_status}`;
        } catch (e) {
            console.warn("Failed to poll health:", e);
        }
    }

    // Load current settings
    async function loadSettings() {
        try {
            const res = await fetch("/api/settings");
            if (!res.ok) return;
            const s = await res.json();

            cfgCameraSource.value = s.camera_source || "0";
            cfgConfPerson.value = s.conf_person !== undefined ? s.conf_person : "0.35";
            cfgConfHardhat.value = s.conf_hardhat !== undefined ? s.conf_hardhat : "0.28";
            cfgConfVest.value = s.conf_vest !== undefined ? s.conf_vest : "0.35";
            cfgIoU.value = s.iou_threshold !== undefined ? s.iou_threshold : "0.45";

            cfgSafeFrames.value = s.required_safe_frames !== undefined ? s.required_safe_frames : "3";
            cfgFailFrames.value = s.required_fail_frames !== undefined ? s.required_fail_frames : "5";
            cfgTimeout.value = s.inspection_timeout_seconds !== undefined ? s.inspection_timeout_seconds : "15.0";
            cfgCooldown.value = s.departure_cooldown_seconds !== undefined ? s.departure_cooldown_seconds : "3.0";
        } catch (e) {
            console.error("Failed to load settings:", e);
        }
    }

    // Save Detection Settings
    btnSaveDetection.addEventListener("click", async () => {
        const payload = {
            camera_source: cfgCameraSource.value,
            conf_person: parseFloat(cfgConfPerson.value),
            conf_hardhat: parseFloat(cfgConfHardhat.value),
            conf_vest: parseFloat(cfgConfVest.value),
            iou_threshold: parseFloat(cfgIoU.value),
        };

        try {
            const res = await fetch("/api/settings", {
                method: "POST",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify(payload),
            });
            if (res.ok) {
                alert("Detection parameters successfully saved.");
            } else {
                alert("Failed to save detection settings.");
            }
        } catch (e) {
            alert("Connection error while saving settings.");
        }
    });

    // Save Consensus Settings
    btnSaveConsensus.addEventListener("click", async () => {
        const payload = {
            required_safe_frames: parseInt(cfgSafeFrames.value),
            required_fail_frames: parseInt(cfgFailFrames.value),
            inspection_timeout_seconds: parseFloat(cfgTimeout.value),
            departure_cooldown_seconds: parseFloat(cfgCooldown.value),
        };

        try {
            const res = await fetch("/api/settings", {
                method: "POST",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify(payload),
            });
            if (res.ok) {
                alert("Consensus rules successfully saved.");
            } else {
                alert("Failed to save consensus rules.");
            }
        } catch (e) {
            alert("Connection error while saving consensus rules.");
        }
    });

    // Danger Zone Actions
    btnClearInspections.addEventListener("click", async () => {
        if (!confirm("Are you sure you want to clear all inspection records?")) return;
        try {
            const res = await fetch("/api/history/clear?target=inspections", { method: "POST" });
            if (res.ok) alert("Inspection history cleared.");
        } catch (e) { alert("Failed to clear inspections."); }
    });

    btnClearViolations.addEventListener("click", async () => {
        if (!confirm("Are you sure you want to clear all violation records?")) return;
        try {
            const res = await fetch("/api/history/clear?target=violations", { method: "POST" });
            if (res.ok) alert("Violation records cleared.");
        } catch (e) { alert("Failed to clear violations."); }
    });

    btnClearAll.addEventListener("click", async () => {
        if (!confirm("WARNING: Clear ALL historical records (inspections and violations)? Machine Learning models and datasets will NOT be affected.")) return;
        try {
            const res = await fetch("/api/history/clear?target=all", { method: "POST" });
            if (res.ok) alert("All historical database records cleared.");
        } catch (e) { alert("Failed to clear all history."); }
    });

    // Poll intervals
    loadSettings();
    pollHardwareHealth();
    setInterval(pollHardwareHealth, 2500);
});
