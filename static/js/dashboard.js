/**
 * MineGuard AI - Overview Dashboard Controller
 * Updates station KPIs, active inspection state, and recent history table.
 */

document.addEventListener("DOMContentLoaded", () => {
    // KPI elements
    const kpiTotalInspections = document.getElementById("kpiTotalInspections");
    const kpiTotalPassed = document.getElementById("kpiTotalPassed");
    const kpiTotalViolations = document.getElementById("kpiTotalViolations");
    const kpiTotalReinspections = document.getElementById("kpiTotalReinspections");
    const kpiComplianceRate = document.getElementById("kpiComplianceRate");
    const kpiAdherenceSub = document.getElementById("kpiAdherenceSub");
    const kpiAvgDuration = document.getElementById("kpiAvgDuration");

    // Station preview elements
    const dashStationFps = document.getElementById("dashStationFps");
    const dashCurrentRef = document.getElementById("dashCurrentRef");
    const dashStatePill = document.getElementById("dashStatePill");
    const dashLatency = document.getElementById("dashLatency");

    // Table
    const dashRecentTableBody = document.getElementById("dashRecentTableBody");

    // 1. Poll Status & Overview Metrics (every 2s)
    async function pollOverviewData() {
        try {
            const res = await fetch("/api/status");
            if (!res.ok) return;
            const data = await res.json();

            // Status strip
            if (dashStationFps) dashStationFps.textContent = `${(data.current_fps || 0).toFixed(1)} FPS`;
            if (dashLatency) dashLatency.textContent = `${(data.inference_latency_ms || 0).toFixed(1)} ms`;

            // Statistics
            const s = data.statistics || {};
            if (kpiTotalInspections) kpiTotalInspections.textContent = s.total_inspections || 0;
            if (kpiTotalPassed) kpiTotalPassed.textContent = s.total_passed || 0;
            if (kpiTotalViolations) kpiTotalViolations.textContent = s.total_violations || 0;
            if (kpiTotalReinspections) kpiTotalReinspections.textContent = s.total_reinspections || 0;
            if (kpiComplianceRate) kpiComplianceRate.textContent = `${s.compliance_rate_pct || 100}%`;
            if (kpiAvgDuration) kpiAvgDuration.textContent = `${(s.average_duration_seconds || 0).toFixed(1)}s`;

            if (kpiAdherenceSub) {
                kpiAdherenceSub.textContent = `H: ${s.hardhat_adherence_pct || 100}% | V: ${s.vest_adherence_pct || 100}%`;
            }
        } catch (e) {
            console.warn("Error polling overview status:", e);
        }
    }

    // 2. Poll Active Inspection (every 500ms)
    async function pollActiveInspection() {
        try {
            const res = await fetch("/api/inspection/current");
            if (!res.ok) return;
            const data = await res.json();

            if (data.reference_id) {
                dashCurrentRef.textContent = data.reference_id;
                dashCurrentRef.className = "fw-bold font-monospace text-warning fs-5 mt-1";
            } else {
                dashCurrentRef.textContent = "STANDBY - NO WORKER";
                dashCurrentRef.className = "fw-bold font-monospace text-secondary fs-5 mt-1";
            }

            const state = data.state || "IDLE";
            dashStatePill.textContent = state;
            dashStatePill.className = "";
            let pCls = "badge-idle";
            if (state === "DETECTING") pCls = "badge-detecting";
            else if (state === "VERIFYING") pCls = "badge-verifying";
            else if (state === "PASS") pCls = "badge-pass";
            else if (state === "FAIL") pCls = "badge-fail";
            else if (state === "REINSPECT") pCls = "badge-reinspect";
            dashStatePill.className = pCls;
        } catch (e) {}
    }

    // 3. Poll Recent Inspections (every 3s)
    async function pollRecentInspections() {
        try {
            const res = await fetch("/api/inspections?per_page=10");
            if (!res.ok) return;
            const data = await res.json();
            const list = data.inspections || [];

            if (list.length === 0) {
                dashRecentTableBody.innerHTML = `
                    <tr>
                        <td colspan="7" class="text-center text-muted py-4">No inspections recorded in database yet</td>
                    </tr>
                `;
                return;
            }

            let html = "";
            list.forEach(item => {
                let badgeClass = "badge-idle";
                if (item.decision === "PASS") badgeClass = "badge-pass";
                else if (item.decision === "FAIL") badgeClass = "badge-fail";
                else if (item.decision === "REINSPECT") badgeClass = "badge-reinspect";

                const hhStatus = item.hardhat_status || "UNKNOWN";
                const hhClass = (hhStatus === "PASS" || hhStatus === "PRESENT") ? "text-success fw-bold" : (hhStatus === "FAIL" ? "text-danger fw-bold" : "text-muted");

                const vestStatus = item.vest_status || "UNKNOWN";
                const vestClass = (vestStatus === "PASS" || vestStatus === "PRESENT") ? "text-success fw-bold" : (vestStatus === "FAIL" ? "text-danger fw-bold" : "text-muted");

                html += `
                    <tr>
                        <td><a href="/history" class="font-monospace text-warning fw-bold text-decoration-none">${item.reference_id}</a></td>
                        <td class="text-secondary small">${item.timestamp}</td>
                        <td><span class="${badgeClass}">${item.decision}</span></td>
                        <td><span class="${hhClass}">${hhStatus}</span></td>
                        <td><span class="${vestClass}">${vestStatus}</span></td>
                        <td class="small text-secondary">${(item.duration_seconds || 0).toFixed(1)}s</td>
                        <td class="small text-muted" style="max-width: 200px; overflow: hidden; text-overflow: ellipsis; white-space: nowrap;">
                            ${item.notes || "--"}
                        </td>
                    </tr>
                `;
            });

            dashRecentTableBody.innerHTML = html;
        } catch (e) {}
    }

    pollOverviewData();
    pollActiveInspection();
    pollRecentInspections();

    setInterval(pollActiveInspection, 500);
    setInterval(pollOverviewData, 2000);
    setInterval(pollRecentInspections, 3500);
});
