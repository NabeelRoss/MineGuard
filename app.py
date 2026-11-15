"""
MineGuard AI - Flask Web Application Entry Point.

Initializes Flask server, registers REST and streaming routes, ensures SQLite database
schema is up to date, and coordinates graceful resource cleanup on process shutdown.
"""

import sys
import atexit
import signal
from pathlib import Path
from flask import Flask

# Ensure project root is on sys.path
PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from config import AppConfig
from backend.database import init_db
from backend.routes import api_bp
from backend.camera_service import camera_service


def create_app() -> Flask:
    """Factory function initializing the MineGuard AI Flask application."""
    AppConfig.ensure_directories()
    
    app = Flask(
        __name__,
        template_folder="templates",
        static_folder="static",
    )
    app.config.from_object(AppConfig)

    # Initialize SQLite Database tables
    print("[MineGuard AI] Initializing SQLite database...")
    init_db()

    # Register blueprint
    app.register_blueprint(api_bp)

    return app


app = create_app()


def cleanup_on_exit():
    """Ensures camera and background worker threads are cleanly released on termination."""
    print("\n[MineGuard AI] Releasing hardware resources and terminating background threads...")
    if camera_service and camera_service.is_monitoring:
        camera_service.stop_monitoring()
    print("[MineGuard AI] Graceful shutdown complete.")


# Register clean shutdown hooks
atexit.register(cleanup_on_exit)


def signal_handler(sig, frame):
    cleanup_on_exit()
    sys.exit(0)


signal.signal(signal.SIGINT, signal_handler)
signal.signal(signal.SIGTERM, signal_handler)


if __name__ == "__main__":
    print("=" * 70)
    print("           MINEGUARD AI - INTELLIGENT PPE MONITORING SYSTEM")
    print("=" * 70)
    print(f"Target URL              : http://{AppConfig.HOST}:{AppConfig.PORT}")
    print(f"Hardware Compute Device : {AppConfig.DEVICE}")
    print(f"Trained Model Weights   : {AppConfig.TRAINED_MODEL_PATH}")
    print(f"Database Storage        : {AppConfig.DATABASE_PATH}")
    print("=" * 70)

    # Note: threaded=True ensures concurrent handling of streaming and REST API polls.
    # use_reloader=False prevents double-instantiation of camera threads on startup.
    app.run(
        host=AppConfig.HOST,
        port=AppConfig.PORT,
        debug=AppConfig.DEBUG,
        threaded=True,
        use_reloader=False,
    )
