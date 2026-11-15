/**
 * MineGuard AI - Reports Controller
 * Aggregates periodic safety audits, daily breakdowns, and exports.
 */

document.addEventListener("DOMContentLoaded", () => {
    const reportDateFrom = document.getElementById("reportDateFrom");
    const reportDateTo = document.getElementById("reportDateTo");
    const btnGenerate = document.getElementById("btnGenerateReport");
    const btnExportCsv = document.getElementById("btnExportReportCsv");
    const btnPrint = document.getElementById("btnPrintReport");

    const presetToday = document.getElementById("presetToday");
    const presetWeek = document.getElementById("presetWeek");
    const presetMonth = document.getElementById("presetMonth");

    const reportPeriodBadge = document.getElementById("reportPeriodBadge");
    const repTotalInspected = document.getElementById("repTotalInspected");
    const repComplianceRatio = document.getElementById("repComplianceRatio");
    const repTotalViolations = document.getElementById("repTotalViolations");
    const repTotalReinspections = document.getElementById("repTotalReinspections");
    const repAvgDuration = document.getElementById("repAvgDuration");
    const repPrimaryDefect = document.getElementById("repPrimaryDefect");
    const reportDailyTableBody = document.getElementById("reportDailyTableBody");

    function formatDate(d) {
        return d.toISOString().split("T")[0];
    }

    // Presets
    presetToday.addEventListener("click", () => {
        const today = formatDate(new Date());
        reportDateFrom.value = today;
        reportDateTo.value = today;
        loadReport();
    });

    presetWeek.addEventListener("click", () => {
        const d = new Date();
        const end = formatDate(d);
        d.setDate(d.getDate() - 7);
        const start = formatDate(d);
        reportDateFrom.value = start;
        reportDateTo.value = end;
        loadReport();
    });

    presetMonth.addEventListener("click", () => {
        const d = new Date();
        const end = formatDate(d);
        d.setDate(d.getDate() - 30);
        const start = formatDate(d);
        reportDateFrom.value = start;
        reportDateTo.value = end;
        loadReport();
    });

    async function loadReport() {
        let url = `/api/reports/summary`;
        const params = [];
        if (reportDateFrom.value) params.push(`date_from=${reportDateFrom.value}`);
        if (reportDateTo.value) params.push(`date_to=${reportDateTo.value}`);
        if (params.length > 0) url += `?${params.join("&")}`;

        if (reportDateFrom.value && reportDateTo.value) {
            reportPeriodBadge.textContent = `${reportDateFrom.value} to ${reportDateTo.value}`;
        } else if (reportDateFrom.value) {
            reportPeriodBadge.textContent = `From ${reportDateFrom.value}`;
        } else {
            reportPeriodBadge.textContent = `All Time`;
        }

        try {
            const res = await fetch(url);
            if (!res.ok) return;
            const data = await res.json();
            const s = data.summary || {};
            const breakdown = data.daily_breakdown || [];

            // Executive Summary
            repTotalInspected.textContent = s.total_inspections || 0;
            repComplianceRatio.textContent = `${s.compliance_rate_pct || 100}%`;
            repTotalViolations.textContent = s.total_violations || 0;
            repTotalReinspections.textContent = s.total_reinspections || 0;
            repAvgDuration.textContent = `${(s.average_duration_seconds || 0).toFixed(1)}s`;

            const ppe = s.missing_ppe_breakdown || {};
            let maxPpe = "None";
            let maxCnt = 0;
            for (const [k, v] of Object.entries(ppe)) {
                if (v > maxCnt) {
                    maxCnt = v;
                    maxPpe = `${k} (${v})`;
                }
            }
            repPrimaryDefect.textContent = maxPpe;

            // Daily Table
            if (breakdown.length === 0) {
                reportDailyTableBody.innerHTML = `
                    <tr>
                        <td colspan="8" class="text-center text-muted py-4">No inspection logs in the selected date range</td>
                    </tr>
                `;
                return;
            }

            let html = "";
            breakdown.forEach(row => {
                const total = row.total || 0;
                const pass = row.pass || 0;
                const fail = row.fail || 0;
                const reinspect = row.reinspect || 0;
                const rate = total > 0 ? ((pass / total) * 100).toFixed(1) : "100.0";
                const hhDef = row.missing_hardhat || 0;
                const vestDef = row.missing_vest || 0;

                html += `
                    <tr>
                        <td class="fw-bold text-white">${row.date}</td>
                        <td>${total}</td>
                        <td><span class="badge-safe">${pass}</span></td>
                        <td><span class="${fail > 0 ? 'badge-violation' : 'text-muted'}">${fail}</span></td>
                        <td><span class="${reinspect > 0 ? 'badge-unknown' : 'text-muted'}">${reinspect}</span></td>
                        <td><strong class="${parseFloat(rate) >= 90 ? 'text-success' : 'text-danger'}">${rate}%</strong></td>
                        <td class="${hhDef > 0 ? 'text-danger fw-bold' : 'text-muted'}">${hhDef}</td>
                        <td class="${vestDef > 0 ? 'text-danger fw-bold' : 'text-muted'}">${vestDef}</td>
                    </tr>
                `;
            });

            reportDailyTableBody.innerHTML = html;
        } catch (e) {
            console.error("Failed to load safety report:", e);
        }
    }

    // Export CSV
    btnExportCsv.addEventListener("click", () => {
        let url = `/api/export/inspections`;
        const params = [];
        if (reportDateFrom.value) params.push(`date_from=${reportDateFrom.value}`);
        if (reportDateTo.value) params.push(`date_to=${reportDateTo.value}`);
        if (params.length > 0) url += `?${params.join("&")}`;
        window.location.href = url;
    });

    // Print
    btnPrint.addEventListener("click", () => {
        window.print();
    });

    btnGenerate.addEventListener("click", loadReport);

    // Initial load
    loadReport();
});
