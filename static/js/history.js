/**
 * MineGuard AI - Inspection History Controller
 * Handles paginated inspection history table, filtering, details modal, and CSV export.
 */

document.addEventListener("DOMContentLoaded", () => {
    let currentPage = 1;
    let totalPages = 1;

    const tbody = document.getElementById("historyTableBody");
    const totalBadge = document.getElementById("totalRecordsBadge");
    const paginationInfo = document.getElementById("paginationInfo");
    const pageNumberDisplay = document.getElementById("pageNumberDisplay");
    const btnPrev = document.getElementById("btnPrevPage");
    const btnNext = document.getElementById("btnNextPage");
    const perPageSelect = document.getElementById("perPageSelect");

    const searchRef = document.getElementById("searchRef");
    const filterDecision = document.getElementById("filterDecision");
    const filterDateFrom = document.getElementById("filterDateFrom");
    const filterDateTo = document.getElementById("filterDateTo");
    const btnApply = document.getElementById("btnApplyFilter");
    const btnReset = document.getElementById("btnResetFilter");
    const btnRefresh = document.getElementById("btnRefreshHistory");
    const btnExportCsv = document.getElementById("btnExportCsv");

    const detailsModalEl = document.getElementById("detailsModal");
    const detailsModal = new bootstrap.Modal(detailsModalEl);

    // Modal elements
    const modalRefId = document.getElementById("modalRefId");
    const modalTimestamp = document.getElementById("modalTimestamp");
    const modalDecisionBadge = document.getElementById("modalDecisionBadge");
    const modalDurationFrames = document.getElementById("modalDurationFrames");
    const modalHardhatPill = document.getElementById("modalHardhatPill");
    const modalHardhatConf = document.getElementById("modalHardhatConf");
    const modalVestPill = document.getElementById("modalVestPill");
    const modalVestConf = document.getElementById("modalVestConf");
    const modalNotes = document.getElementById("modalNotes");
    const modalEvidenceSection = document.getElementById("modalEvidenceSection");
    const modalAnnotatedImg = document.getElementById("modalAnnotatedImg");
    const modalOriginalImg = document.getElementById("modalOriginalImg");

    // Fetch inspections with filter params
    async function fetchInspections(page = 1) {
        const perPage = perPageSelect ? perPageSelect.value : 25;
        let url = `/api/inspections?page=${page}&per_page=${perPage}`;

        if (filterDecision && filterDecision.value !== "ALL") {
            url += `&decision=${encodeURIComponent(filterDecision.value)}`;
        }
        if (filterDateFrom && filterDateFrom.value) {
            url += `&date_from=${encodeURIComponent(filterDateFrom.value)}`;
        }
        if (filterDateTo && filterDateTo.value) {
            url += `&date_to=${encodeURIComponent(filterDateTo.value)}`;
        }
        if (searchRef && searchRef.value.trim()) {
            url += `&search=${encodeURIComponent(searchRef.value.trim())}`;
        }

        try {
            const res = await fetch(url);
            if (res.ok) {
                const data = await res.json();
                currentPage = data.page;
                totalPages = data.pages || 1;
                renderTable(data.inspections || [], data.total || 0);
            }
        } catch (e) {
            console.error("Failed to load inspections:", e);
        }
    }

    // Render table
    function renderTable(inspections, total) {
        totalBadge.textContent = `${total} Records`;
        paginationInfo.textContent = `Page ${currentPage} of ${totalPages}`;
        pageNumberDisplay.textContent = `Page ${currentPage}`;

        btnPrev.disabled = (currentPage <= 1);
        btnNext.disabled = (currentPage >= totalPages);

        if (inspections.length === 0) {
            tbody.innerHTML = `
                <tr>
                    <td colspan="8" class="text-center text-muted py-5">
                        <i class="bi bi-shield-slash fs-2 d-block mb-2 text-secondary"></i>
                        <div>No inspection records found matching your filters</div>
                    </td>
                </tr>
            `;
            return;
        }

        let html = "";
        inspections.forEach(item => {
            let badgeClass = "badge-idle";
            if (item.decision === "PASS") badgeClass = "badge-pass";
            else if (item.decision === "FAIL") badgeClass = "badge-fail";
            else if (item.decision === "REINSPECT") badgeClass = "badge-reinspect";

            const hhStatus = item.hardhat_status || "UNKNOWN";
            const hhClass = (hhStatus === "PASS" || hhStatus === "PRESENT") ? "text-success fw-bold" : (hhStatus === "FAIL" ? "text-danger fw-bold" : "text-muted");
            const hhConf = Math.round((item.hardhat_confidence || 0) * 100);

            const vestStatus = item.vest_status || "UNKNOWN";
            const vestClass = (vestStatus === "PASS" || vestStatus === "PRESENT") ? "text-success fw-bold" : (vestStatus === "FAIL" ? "text-danger fw-bold" : "text-muted");
            const vestConf = Math.round((item.vest_confidence || 0) * 100);

            const dur = (item.duration_seconds || 0).toFixed(1);
            const frames = item.frames_analyzed || 0;

            html += `
                <tr>
                    <td>
                        <span class="font-monospace text-warning fw-bold cursor-pointer btn-view-details" data-id="${item.id}" style="cursor:pointer;" title="Click to view details">
                            ${item.reference_id}
                        </span>
                    </td>
                    <td class="text-secondary small">${item.timestamp}</td>
                    <td><span class="${badgeClass}">${item.decision}</span></td>
                    <td>
                        <span class="${hhClass}">${hhStatus}</span>
                        <span class="text-muted small">(${hhConf}%)</span>
                    </td>
                    <td>
                        <span class="${vestClass}">${vestStatus}</span>
                        <span class="text-muted small">(${vestConf}%)</span>
                    </td>
                    <td class="small text-secondary">${dur}s <span class="text-muted">(${frames} f)</span></td>
                    <td class="small text-muted" style="max-width: 220px; overflow: hidden; text-overflow: ellipsis; white-space: nowrap;">
                        ${item.notes || "--"}
                    </td>
                    <td class="text-end">
                        <div class="d-flex justify-content-end gap-1">
                            <button class="btn btn-sm btn-mg-outline py-0 px-2 btn-view-details" data-id="${item.id}" title="View Details">
                                <i class="bi bi-eye-fill text-warning"></i>
                            </button>
                            <button class="btn btn-sm btn-mg-outline py-0 px-2 text-danger btn-delete-insp" data-id="${item.id}" title="Delete Record">
                                <i class="bi bi-trash3"></i>
                            </button>
                        </div>
                    </td>
                </tr>
            `;
        });

        tbody.innerHTML = html;

        // Attach action listeners
        document.querySelectorAll(".btn-view-details").forEach(btn => {
            btn.addEventListener("click", () => openDetailsModal(btn.getAttribute("data-id")));
        });

        document.querySelectorAll(".btn-delete-insp").forEach(btn => {
            btn.addEventListener("click", () => deleteInspectionRecord(btn.getAttribute("data-id")));
        });
    }

    // Open Details Modal
    async function openDetailsModal(id) {
        try {
            const res = await fetch(`/api/inspections/${id}`);
            if (!res.ok) return;
            const data = await res.json();
            const insp = data.inspection || {};

            modalRefId.textContent = insp.reference_id;
            modalTimestamp.textContent = `Timestamp: ${insp.timestamp}`;

            let badgeClass = "badge-idle";
            if (insp.decision === "PASS") badgeClass = "badge-pass";
            else if (insp.decision === "FAIL") badgeClass = "badge-fail";
            else if (insp.decision === "REINSPECT") badgeClass = "badge-reinspect";

            modalDecisionBadge.innerHTML = `<span class="${badgeClass} fs-6">${insp.decision}</span>`;
            modalDurationFrames.textContent = `Inspection Dwell: ${(insp.duration_seconds || 0).toFixed(1)}s | Processed Frames: ${insp.frames_analyzed || 0}`;

            const hhStatus = insp.hardhat_status || "UNKNOWN";
            modalHardhatPill.className = (hhStatus === "PASS" || hhStatus === "PRESENT") ? "badge bg-success" : (hhStatus === "FAIL" ? "badge bg-danger" : "badge bg-secondary");
            modalHardhatPill.textContent = hhStatus;
            modalHardhatConf.textContent = `${Math.round((insp.hardhat_confidence || 0) * 100)}%`;

            const vestStatus = insp.vest_status || "UNKNOWN";
            modalVestPill.className = (vestStatus === "PASS" || vestStatus === "PRESENT") ? "badge bg-success" : (vestStatus === "FAIL" ? "badge bg-danger" : "badge bg-secondary");
            modalVestPill.textContent = vestStatus;
            modalVestConf.textContent = `${Math.round((insp.vest_confidence || 0) * 100)}%`;

            modalNotes.textContent = insp.notes || "No additional audit notes recorded.";

            // Handle Evidence Images
            const annPath = insp.annotated_image_path;
            const origPath = insp.original_image_path;

            if (annPath || origPath) {
                modalEvidenceSection.style.display = "block";

                // Format path to URL
                function formatImgUrl(path) {
                    if (!path) return "";
                    const parts = path.replace(/\\/g, "/").split("/");
                    const dateFolder = parts[parts.length - 2];
                    const filename = parts[parts.length - 1];
                    return `/evidence/${dateFolder}/${filename}`;
                }

                modalAnnotatedImg.src = formatImgUrl(annPath);
                modalOriginalImg.src = formatImgUrl(origPath);
            } else {
                modalEvidenceSection.style.display = "none";
            }

            detailsModal.show();
        } catch (e) {
            console.error("Error opening details modal:", e);
        }
    }

    // Delete record
    async function deleteInspectionRecord(id) {
        if (!confirm(`Delete inspection record #${id}? This action cannot be undone.`)) {
            return;
        }

        try {
            const res = await fetch(`/api/inspections/${id}`, { method: "DELETE" });
            if (res.ok) {
                fetchInspections(currentPage);
            } else {
                alert("Failed to delete inspection record");
            }
        } catch (e) {
            alert("Connection error while deleting record");
        }
    }

    // Export CSV
    btnExportCsv.addEventListener("click", () => {
        let url = `/api/export/inspections`;
        const params = [];
        if (filterDecision.value !== "ALL") params.push(`decision=${encodeURIComponent(filterDecision.value)}`);
        if (filterDateFrom.value) params.push(`date_from=${encodeURIComponent(filterDateFrom.value)}`);
        if (filterDateTo.value) params.push(`date_to=${encodeURIComponent(filterDateTo.value)}`);
        if (params.length > 0) url += `?${params.join("&")}`;
        window.location.href = url;
    });

    // Event listeners
    btnApply.addEventListener("click", () => fetchInspections(1));
    btnReset.addEventListener("click", () => {
        searchRef.value = "";
        filterDecision.value = "ALL";
        filterDateFrom.value = "";
        filterDateTo.value = "";
        fetchInspections(1);
    });
    btnRefresh.addEventListener("click", () => fetchInspections(currentPage));
    btnPrev.addEventListener("click", () => { if (currentPage > 1) fetchInspections(currentPage - 1); });
    btnNext.addEventListener("click", () => { if (currentPage < totalPages) fetchInspections(currentPage + 1); });
    perPageSelect.addEventListener("change", () => fetchInspections(1));

    // Initial load
    fetchInspections(1);
});
