"""Flask web frontend for the multi-sport prediction system.

Run with:
    python -m web.app
    # or
    python web/app.py

Opens at http://localhost:5000
"""

from __future__ import annotations

import sys
import os
import subprocess
import threading
import logging
from pathlib import Path

from flask import Flask, render_template, request, jsonify
from flask_socketio import SocketIO, emit

# Ensure project root is on the path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
os.chdir(PROJECT_ROOT)

app = Flask(__name__,
            template_folder="templates",
            static_folder="static")
app.config["SECRET_KEY"] = "sports-predict-dev"
socketio = SocketIO(app, cors_allowed_origins="*", async_mode="threading")

logger = logging.getLogger(__name__)

# Track running processes so we can cancel them
_running_processes: dict[str, subprocess.Popen] = {}


# ------------------------------------------------------------------
# Routes
# ------------------------------------------------------------------

@app.route("/")
def index():
    """Main dashboard page."""
    return render_template("index.html")


@app.route("/api/models", methods=["GET"])
def list_models():
    """List available trained models."""
    models_dir = PROJECT_ROOT / "models"
    models = []
    if models_dir.exists():
        for f in models_dir.glob("*.pkl"):
            size_mb = f.stat().st_size / (1024 * 1024)
            models.append({
                "name": f.name,
                "path": str(f),
                "size_mb": round(size_mb, 2),
            })
    return jsonify(models)


# ------------------------------------------------------------------
# SocketIO: run commands with live streaming output
# ------------------------------------------------------------------

@socketio.on("run_command")
def handle_run_command(data):
    """Execute a CLI command and stream output back to the client."""
    sport = data.get("sport", "ufc")
    action = data.get("action", "")
    args = data.get("args", {})

    # Build the command
    cmd = _build_command(sport, action, args)
    if cmd is None:
        emit("output", {"text": f"[ERROR] Unknown action: {action}\n", "done": True})
        return

    emit("output", {"text": f"$ {' '.join(cmd)}\n\n", "done": False})

    # Run in a background thread so we don't block the socket
    thread = threading.Thread(target=_run_process, args=(cmd,), daemon=True)
    thread.start()


@socketio.on("cancel_command")
def handle_cancel():
    """Cancel any running command."""
    for key, proc in list(_running_processes.items()):
        proc.terminate()
        del _running_processes[key]
    emit("output", {"text": "\n[CANCELLED]\n", "done": True})


def _run_process(cmd: list[str]):
    """Run a subprocess and stream stdout/stderr to the client."""
    try:
        proc = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            cwd=str(PROJECT_ROOT),
            bufsize=1,
            encoding="utf-8",
            errors="replace",
        )
        _running_processes["current"] = proc

        for line in iter(proc.stdout.readline, ""):
            socketio.emit("output", {"text": line, "done": False})

        proc.wait()
        exit_code = proc.returncode
        status = "completed" if exit_code == 0 else f"exited with code {exit_code}"
        socketio.emit("output", {
            "text": f"\n[{status.upper()}]\n",
            "done": True,
            "exit_code": exit_code,
        })

    except Exception as e:
        socketio.emit("output", {"text": f"\n[ERROR] {e}\n", "done": True})
    finally:
        _running_processes.pop("current", None)


def _build_command(sport: str, action: str, args: dict) -> list[str] | None:
    """Build a CLI command list from the web request."""
    python = sys.executable
    base = [python, "cli.py", "--sport", sport]

    if action == "scrape":
        return base + ["scrape"]

    elif action == "run_full":
        cmd = base + ["run-full"]
        if args.get("data"):
            cmd += ["--data", args["data"]]
        if args.get("save_model"):
            cmd += ["--save-model", args["save_model"]]
        return cmd

    elif action == "predict":
        entity_a = args.get("entity_a", "").strip()
        entity_b = args.get("entity_b", "").strip()
        if not entity_a or not entity_b:
            return None
        cmd = base + ["predict", "--entity-a", entity_a, "--entity-b", entity_b]
        if args.get("date"):
            cmd += ["--date", args["date"]]
        if args.get("model"):
            cmd += ["--model", args["model"]]
        return cmd

    elif action == "ingest":
        cmd = base + ["ingest"]
        if args.get("source_dir"):
            cmd += ["--source-dir", args["source_dir"]]
        if args.get("rss_url"):
            cmd += ["--rss-url", args["rss_url"]]
        return cmd

    elif action == "ufc_predict":
        # Direct UFC predict via the existing CLI
        entity_a = args.get("entity_a", "").strip()
        entity_b = args.get("entity_b", "").strip()
        if not entity_a or not entity_b:
            return None
        cmd = [python, "-m", "ufc_predict.cli", "predict",
               "--fighter-a", entity_a, "--fighter-b", entity_b]
        if args.get("date"):
            cmd += ["--date", args["date"]]
        if args.get("model"):
            cmd += ["--model", args["model"]]
        if args.get("rounds"):
            cmd += ["--rounds", str(args["rounds"])]
        return cmd

    elif action == "ufc_run_full":
        cmd = [python, "-m", "ufc_predict.cli", "run-full",
               "--data", args.get("data", "data/raw/ufcstats_fights.csv")]
        if args.get("save_model"):
            cmd += ["--save-model", args["save_model"]]
        return cmd

    elif action == "ufc_scrape":
        return [python, "-m", "ufc_predict.cli", "scrape"]

    return None


# ------------------------------------------------------------------
# Main
# ------------------------------------------------------------------

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    print("Starting Sports Predict dashboard at http://localhost:5000")
    socketio.run(app, host="0.0.0.0", port=5000, debug=True, allow_unsafe_werkzeug=True)
