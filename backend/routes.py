"""
REST API and Web View Routes for MineGuard AI Checkpoint Inspection.

Implements all frontend view endpoints, MJPEG video streaming, evidence serving,
and comprehensive REST APIs for inspection telemetry, violation management,
CSV reporting, and system health.
"""

from flask import Blueprint, render_template, Response, request, jsonify, send_from_directory
import datetime
from pathlib import Path
from config import AppConfig
from backend.camera_service import camera_service
from backend.alert_manager import alert_manager
from backend.database import (
    get_overview_statistics,
    get_inspections,
    get_inspection_by_id,
    delete_inspection,
    get_violations,
    get_violation_by_id,
    delete_violation,
    clear_all_history,
    get_analytics_summary,
    export_inspections_csv,
    export_violations_csv,
    get_system_settings,
    update_system_setting,
)

api_bp = Blueprint("api", __name__)


# ==============================================================================
# HTML PAGE ROUTES
# ==============================================================================

@api_bp.route("/")
def index_view():
    """Overview dashboard view."""
    return render_template("index.html")


@api_bp.route("/live")
def live_inspection_view():
    """Primary operational live checkpoint inspection view."""
    return render_template("live_inspection.html")


@api_bp.route("/history")
def inspection_history_view():
    """Comprehensive inspection history log view."""
    return render_template("inspection_history.html")


@api_bp.route("/violations")
def violations_view():
    """Confirmed violation evidence and audit view."""
    return render_template("violations.html")


@api_bp.route("/analytics")
def analytics_view():
    """Safety compliance analytics and trend charts view."""
    return render_template("analytics.html")


@api_bp.route("/reports")
def reports_view():
    """Periodic safety reporting and summary view."""
    return render_template("reports.html")


@api_bp.route("/settings")
def settings_view():
    """System configuration and hardware health view."""
    return render_template("settings.html")


# ==============================================================================
# MJPEG STREAMING & EVIDENCE SERVING
# ==============================================================================

@api_bp.route("/video_feed")
def video_feed():
    """Exposes real-time MJPEG annotated camera stream from internal buffer."""
    return Response(
        camera_service.generate_mjpeg(),
        mimetype="multipart/x-mixed-replace; boundary=frame",
    )


@api_bp.route("/evidence/<path:filename>")
def serve_evidence_image(filename):
    """Serves recorded violation evidence images from outputs/violations/."""
    return send_from_directory(AppConfig.VIOLATIONS_DIR, filename)


# ==============================================================================
# SYSTEM & HEALTH APIS
# ==============================================================================

@api_bp.route("/api/status", methods=["GET"])
def get_system_status():
    """Returns application, model, camera, and inspection operational status."""
    status_data = camera_service.get_status()
    status_data["statistics"] = get_overview_statistics()
    return jsonify(status_data)


@api_bp.route("/api/system/health", methods=["GET"])
def get_system_health():
    """Returns measured hardware, GPU VRAM, FPS, and latency metrics."""
    return jsonify(camera_service.get_system_health())


# ==============================================================================
# CAMERA CONTROL APIS
# ==============================================================================

@api_bp.route("/api/camera/status", methods=["GET"])
def get_camera_status():
    """Returns camera connection and FPS metrics."""
    return jsonify({
        "camera_connected": camera_service.camera_connected,
        "is_monitoring": camera_service.is_monitoring,
        "camera_fps": round(camera_service.current_camera_fps, 1),
        "inference_fps": camera_service.current_inference_fps,
        "inference_latency_ms": camera_service.current_inference_latency_ms,
        "camera_source": str(camera_service.camera_source),
        "camera_width": camera_service.camera_width,
        "camera_height": camera_service.camera_height,
        "session_id": camera_service.session_id,
        "total_frames": camera_service.total_frames_processed,
    })


@api_bp.route("/api/camera/start", methods=["POST"])
def start_camera():
    """Starts checkpoint inspection monitoring."""
    req_data = request.get_json(silent=True) or {}
    source = req_data.get("source", None)
    success, message = camera_service.start_monitoring(source=source)
    status_code = 200 if success else 400
    return jsonify({"success": success, "message": message, "session_id": camera_service.session_id}), status_code


@api_bp.route("/api/camera/stop", methods=["POST"])
def stop_camera():
    """Stops checkpoint inspection monitoring."""
    success, message = camera_service.stop_monitoring()
    return jsonify({"success": success, "message": message})


@api_bp.route("/api/camera/restart", methods=["POST"])
def restart_camera():
    """Restarts camera stream."""
    success, message = camera_service.restart_camera()
    return jsonify({"success": success, "message": message})


# ==============================================================================
# INSPECTION APIS
# ==============================================================================

@api_bp.route("/api/inspection/current", methods=["GET"])
def get_current_inspection():
    """Returns real-time telemetry for the live checkpoint inspection cycle."""
    telemetry = camera_service.get_current_inspection()
    telemetry["camera_connected"] = camera_service.camera_connected
    telemetry["is_monitoring"] = camera_service.is_monitoring
    telemetry["camera_fps"] = round(camera_service.current_camera_fps, 1)
    telemetry["inference_fps"] = camera_service.current_inference_fps
    telemetry["inference_latency_ms"] = camera_service.current_inference_latency_ms
    telemetry["reference_id"] = telemetry.get("reference")
    telemetry["elapsed_seconds"] = telemetry.get("duration", 0.0)
    telemetry["hardhat"] = {
        "status": telemetry.get("helmet_status"),
        "confidence": telemetry.get("helmet_confidence", 0.0),
        "frames_observed": telemetry.get("frames_analyzed", 0),
    }
    telemetry["vest"] = {
        "status": telemetry.get("vest_status"),
        "confidence": telemetry.get("vest_confidence", 0.0),
        "frames_observed": telemetry.get("frames_analyzed", 0),
    }
    return jsonify(telemetry)


@api_bp.route("/api/inspections", methods=["GET"])
def list_inspections():
    """Retrieves paginated, filtered inspection audit records."""
    page = request.args.get("page", 1, type=int)
    per_page = request.args.get("per_page", 10, type=int)
    search = request.args.get("search", None)
    date_from = request.args.get("date_from", None)
    date_to = request.args.get("date_to", None)
    result_filter = request.args.get("result", None)

    records, total, pages = get_inspections(
        page=page,
        per_page=per_page,
        search=search,
        date_from=date_from,
        date_to=date_to,
        result_filter=result_filter,
    )

    return jsonify({
        "inspections": records,
        "total": total,
        "page": page,
        "pages": pages,
        "per_page": per_page,
    })


@api_bp.route("/api/inspections/<int:inspection_id>", methods=["GET", "DELETE"])
def handle_single_inspection(inspection_id: int):
    """Retrieves or deletes a single inspection record."""
    if request.method == "GET":
        rec = get_inspection_by_id(inspection_id)
        if rec:
            return jsonify(rec)
        return jsonify({"error": "Inspection record not found"}), 404

    # DELETE
    deleted = delete_inspection(inspection_id)
    if deleted:
        return jsonify({"success": True, "message": f"Inspection {inspection_id} deleted successfully"})
    return jsonify({"error": "Failed to delete inspection record"}), 400


# ==============================================================================
# VIOLATIONS APIS
# ==============================================================================

@api_bp.route("/api/violations", methods=["GET"])
def list_violations():
    """Retrieves paginated, filtered violation audit records."""
    page = request.args.get("page", 1, type=int)
    per_page = request.args.get("per_page", 10, type=int)
    search = request.args.get("search", None)
    date_from = request.args.get("date_from", None)
    date_to = request.args.get("date_to", None)
    missing_ppe = request.args.get("missing_ppe", None)

    records, total, pages = get_violations(
        page=page,
        per_page=per_page,
        search=search,
        date_from=date_from,
        date_to=date_to,
        missing_ppe=missing_ppe,
    )

    return jsonify({
        "violations": records,
        "total": total,
        "page": page,
        "pages": pages,
        "per_page": per_page,
    })


@api_bp.route("/api/violations/<int:violation_id>", methods=["GET", "DELETE"])
def handle_single_violation(violation_id: int):
    """Retrieves or deletes an individual violation record."""
    if request.method == "GET":
        rec = get_violation_by_id(violation_id)
        if rec:
            return jsonify(rec)
        return jsonify({"error": "Violation record not found"}), 404

    deleted = delete_violation(violation_id)
    if deleted:
        return jsonify({"success": True, "message": f"Violation {violation_id} deleted successfully"})
    return jsonify({"error": "Failed to delete violation record"}), 400


# ==============================================================================
# HISTORY MANAGEMENT & DESTRUCTIVE OPERATIONS
# ==============================================================================

@api_bp.route("/api/history/clear", methods=["POST", "DELETE"])
def clear_history():
    """
    Clears inspections, violations, and/or sessions.
    Guaranteed: NEVER touches models/ or dataset/.
    """
    req_data = request.get_json(silent=True) or {}
    target = request.args.get("target") or req_data.get("target", "all")
    result = clear_all_history(target=target)
    alert_manager.clear()
    return jsonify({
        "success": True,
        "message": f"History ({target}) cleared successfully (models and datasets preserved)",
        "details": result,
    })


# ==============================================================================
# ANALYTICS & REPORTING APIS
# ==============================================================================

@api_bp.route("/api/analytics/summary", methods=["GET"])
def get_analytics_summary_api():
    """Returns overview statistics and analytical trends for dashboard."""
    date_from = request.args.get("date_from", None)
    date_to = request.args.get("date_to", None)
    trends = get_analytics_summary(date_from=date_from, date_to=date_to)
    return jsonify(trends)


@api_bp.route("/api/analytics/trends", methods=["GET"])
def get_analytics_trends_api():
    """Returns time-series and categorical aggregates for Chart.js analytics."""
    date_from = request.args.get("date_from", None)
    date_to = request.args.get("date_to", None)
    trends = get_analytics_summary(date_from=date_from, date_to=date_to)
    return jsonify(trends)


@api_bp.route("/api/reports/summary", methods=["GET"])
def get_reports_summary_api():
    """Returns detailed statistical summary and daily breakdown for safety reports view."""
    date_from = request.args.get("date_from", None)
    date_to = request.args.get("date_to", None)
    trends = get_analytics_summary(date_from=date_from, date_to=date_to)
    return jsonify({
        "summary": trends,
        "daily_breakdown": trends.get("daily_trends", []),
    })


# ==============================================================================
# CSV EXPORT APIS
# ==============================================================================

@api_bp.route("/api/export/inspections", methods=["GET"])
def export_inspections():
    """Exports inspection history as CSV file."""
    date_from = request.args.get("date_from", None)
    date_to = request.args.get("date_to", None)
    result_filter = request.args.get("result", None)

    csv_data = export_inspections_csv(date_from=date_from, date_to=date_to, result_filter=result_filter)
    filename = f"mineguard_inspections_{datetime.date.today().strftime('%Y%m%d')}.csv"

    return Response(
        csv_data,
        mimetype="text/csv",
        headers={"Content-Disposition": f"attachment; filename={filename}"},
    )


@api_bp.route("/api/export/violations", methods=["GET"])
def export_violations():
    """Exports violation audit history as CSV file."""
    date_from = request.args.get("date_from", None)
    date_to = request.args.get("date_to", None)

    csv_data = export_violations_csv(date_from=date_from, date_to=date_to)
    filename = f"mineguard_violations_{datetime.date.today().strftime('%Y%m%d')}.csv"

    return Response(
        csv_data,
        mimetype="text/csv",
        headers={"Content-Disposition": f"attachment; filename={filename}"},
    )


# ==============================================================================
# ALERTS POLLING
# ==============================================================================

@api_bp.route("/api/alerts/poll", methods=["GET"])
def poll_alerts():
    """Polls unacknowledged alerts for audio chimes and modal toasts."""
    new_alerts = alert_manager.poll_new_alerts()
    return jsonify({"alerts": new_alerts, "count": len(new_alerts)})


# ==============================================================================
# SETTINGS CONFIGURATION
# ==============================================================================

@api_bp.route("/api/settings", methods=["GET", "POST"])
def manage_settings():
    """Retrieves or dynamically updates system settings."""
    if request.method == "GET":
        return jsonify(get_system_settings())

    # POST: Validate and update
    req_data = request.get_json(silent=True) or {}
    try:
        if "camera_source" in req_data:
            update_system_setting("camera_source", str(req_data["camera_source"]))

        if "conf_person" in req_data:
            val = float(req_data["conf_person"])
            if not (0.05 <= val <= 0.95):
                return jsonify({"error": "Person confidence must be between 0.05 and 0.95"}), 400
            update_system_setting("conf_person", str(val))

        if "conf_hardhat" in req_data:
            val = float(req_data["conf_hardhat"])
            if not (0.05 <= val <= 0.95):
                return jsonify({"error": "Hardhat confidence must be between 0.05 and 0.95"}), 400
            update_system_setting("conf_hardhat", str(val))

        if "conf_vest" in req_data:
            val = float(req_data["conf_vest"])
            if not (0.05 <= val <= 0.95):
                return jsonify({"error": "Vest confidence must be between 0.05 and 0.95"}), 400
            update_system_setting("conf_vest", str(val))

        if "safe_confirm_frames" in req_data:
            val = int(req_data["safe_confirm_frames"])
            if not (1 <= val <= 30):
                return jsonify({"error": "Safe confirmation frames must be between 1 and 30"}), 400
            update_system_setting("safe_confirm_frames", str(val))

        if "fail_confirm_frames" in req_data:
            val = int(req_data["fail_confirm_frames"])
            if not (1 <= val <= 30):
                return jsonify({"error": "Fail confirmation frames must be between 1 and 30"}), 400
            update_system_setting("fail_confirm_frames", str(val))

        if "inspection_timeout" in req_data:
            val = float(req_data["inspection_timeout"])
            if not (2.0 <= val <= 60.0):
                return jsonify({"error": "Timeout must be between 2.0 and 60.0 seconds"}), 400
            update_system_setting("inspection_timeout", str(val))

        # Propagate settings live to camera service
        camera_service.update_settings(
            camera_source=req_data.get("camera_source"),
            conf_person=req_data.get("conf_person"),
            conf_hardhat=req_data.get("conf_hardhat"),
            conf_vest=req_data.get("conf_vest"),
            safe_confirm_frames=req_data.get("safe_confirm_frames"),
            fail_confirm_frames=req_data.get("fail_confirm_frames"),
            inspection_timeout=req_data.get("inspection_timeout"),
        )

        return jsonify({"success": True, "message": "Settings updated and applied dynamically"})
    except (ValueError, TypeError) as e:
        return jsonify({"error": f"Invalid settings format: {e}"}), 400
