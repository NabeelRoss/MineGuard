/**
 * MineGuard AI - Analytics & Chart Controller
 * Fetches real database metrics from /api/analytics/summary and renders interactive Chart.js visualizations.
 */

document.addEventListener("DOMContentLoaded", () => {
    // Chart instances
    let dailyTrendChart = null;
    let decisionDistChart = null;
    let ppeDefectsChart = null;
    let hourlyTrafficChart = null;

    // Chart.js Global Dark Theme Styling
    Chart.defaults.color = "#94a3b8";
    Chart.defaults.borderColor = "rgba(39, 62, 107, 0.4)";
    Chart.defaults.font.family = "-apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif";

    // KPI Elements
    const kpiTotalInspections = document.getElementById("kpiTotalInspections");
    const kpiComplianceRate = document.getElementById("kpiComplianceRate");
    const kpiHardhatRate = document.getElementById("kpiHardhatRate");
    const kpiVestRate = document.getElementById("kpiVestRate");
    const btnRefresh = document.getElementById("btnRefreshCharts");

    async function loadAnalyticsData() {
        try {
            const res = await fetch("/api/analytics/summary");
            if (!res.ok) return;
            const data = await res.json();

            // Update KPI cards
            if (kpiTotalInspections) kpiTotalInspections.textContent = data.total_inspections || 0;
            if (kpiComplianceRate) kpiComplianceRate.textContent = `${data.compliance_rate_pct || 100}%`;
            if (kpiHardhatRate) kpiHardhatRate.textContent = `${data.hardhat_compliance_rate_pct || 100}%`;
            if (kpiVestRate) kpiVestRate.textContent = `${data.vest_compliance_rate_pct || 100}%`;

            renderDailyTrend(data.daily_trends || []);
            renderDecisionDist(data.decision_distribution || {});
            renderPPEDefects(data.missing_ppe_breakdown || {});
            renderHourlyTraffic(data.hourly_distribution || []);
        } catch (e) {
            console.error("Failed to load analytics summary:", e);
        }
    }

    // 1. Daily Inspection Trend (Line Chart)
    function renderDailyTrend(dailyTrends) {
        const ctx = document.getElementById("chartDailyTrend");
        if (!ctx) return;

        const labels = dailyTrends.map(d => d.date);
        const passData = dailyTrends.map(d => d.pass);
        const failData = dailyTrends.map(d => d.fail);
        const reinspectData = dailyTrends.map(d => d.reinspect);

        if (dailyTrendChart) dailyTrendChart.destroy();

        dailyTrendChart = new Chart(ctx, {
            type: "line",
            data: {
                labels: labels.length > 0 ? labels : ["No Data"],
                datasets: [
                    {
                        label: "Passed (Compliant)",
                        data: passData.length > 0 ? passData : [0],
                        borderColor: "#10b981",
                        backgroundColor: "rgba(16, 185, 129, 0.1)",
                        fill: true,
                        tension: 0.35,
                        pointRadius: 4,
                    },
                    {
                        label: "Failed (Violations)",
                        data: failData.length > 0 ? failData : [0],
                        borderColor: "#ef4444",
                        backgroundColor: "rgba(239, 68, 68, 0.1)",
                        fill: true,
                        tension: 0.35,
                        pointRadius: 4,
                    },
                    {
                        label: "Re-Inspected",
                        data: reinspectData.length > 0 ? reinspectData : [0],
                        borderColor: "#f59e0b",
                        backgroundColor: "rgba(245, 158, 11, 0.05)",
                        fill: true,
                        tension: 0.35,
                        pointRadius: 3,
                    }
                ]
            },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                plugins: {
                    legend: { position: "top", labels: { boxWidth: 12 } },
                    tooltip: { mode: "index", intersect: false }
                },
                scales: {
                    x: { grid: { display: false } },
                    y: { beginAtZero: true, ticks: { precision: 0 } }
                }
            }
        });
    }

    // 2. Decision Breakdown (Doughnut Chart)
    function renderDecisionDist(decisionDist) {
        const ctx = document.getElementById("chartDecisionDist");
        if (!ctx) return;

        const passes = decisionDist.PASS || 0;
        const fails = decisionDist.FAIL || 0;
        const reinspects = decisionDist.REINSPECT || 0;
        const total = passes + fails + reinspects;

        if (decisionDistChart) decisionDistChart.destroy();

        decisionDistChart = new Chart(ctx, {
            type: "doughnut",
            data: {
                labels: ["Pass", "Fail", "Reinspect"],
                datasets: [{
                    data: total > 0 ? [passes, fails, reinspects] : [1, 0, 0],
                    backgroundColor: ["#10b981", "#ef4444", "#f59e0b"],
                    borderWidth: 2,
                    borderColor: "#111e38",
                }]
            },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                plugins: {
                    legend: { position: "bottom", labels: { boxWidth: 12 } }
                },
                cutout: "68%"
            }
        });
    }

    // 3. PPE Defect Category Breakdown (Bar Chart)
    function renderPPEDefects(ppeBreakdown) {
        const ctx = document.getElementById("chartPPEDefects");
        if (!ctx) return;

        const labels = Object.keys(ppeBreakdown);
        const counts = Object.values(ppeBreakdown);

        if (ppeDefectsChart) ppeDefectsChart.destroy();

        ppeDefectsChart = new Chart(ctx, {
            type: "bar",
            data: {
                labels: labels.length > 0 ? labels : ["Hardhat", "Safety Vest"],
                datasets: [{
                    label: "Violations Count",
                    data: counts.length > 0 ? counts : [0, 0],
                    backgroundColor: ["#ef4444", "#f97316", "#eab308"],
                    borderRadius: 4,
                }]
            },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                plugins: {
                    legend: { display: false },
                },
                scales: {
                    x: { grid: { display: false } },
                    y: { beginAtZero: true, ticks: { precision: 0 } }
                }
            }
        });
    }

    // 4. Hourly Shift Load Distribution (Bar Chart)
    function renderHourlyTraffic(hourly) {
        const ctx = document.getElementById("chartHourlyTraffic");
        if (!ctx) return;

        const labels = hourly.map(h => `${String(h.hour).padStart(2, '0')}:00`);
        const totalCounts = hourly.map(h => h.count);
        const failCounts = hourly.map(h => h.failures);

        if (hourlyTrafficChart) hourlyTrafficChart.destroy();

        hourlyTrafficChart = new Chart(ctx, {
            type: "bar",
            data: {
                labels: labels,
                datasets: [
                    {
                        label: "Total Inspections",
                        data: totalCounts,
                        backgroundColor: "rgba(6, 182, 212, 0.6)",
                        borderRadius: 3,
                    },
                    {
                        label: "Violations",
                        data: failCounts,
                        backgroundColor: "rgba(239, 68, 68, 0.8)",
                        borderRadius: 3,
                    }
                ]
            },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                plugins: {
                    legend: { position: "top", labels: { boxWidth: 12 } }
                },
                scales: {
                    x: { grid: { display: false } },
                    y: { beginAtZero: true, ticks: { precision: 0 } }
                }
            }
        });
    }

    if (btnRefresh) {
        btnRefresh.addEventListener("click", loadAnalyticsData);
    }

    loadAnalyticsData();
});
