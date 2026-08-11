"""Shared STATE.json management — import in every script."""
import json, datetime, os

STATE_FILE = os.path.join(os.path.dirname(__file__), "..", "STATE.json")

def load():
    with open(STATE_FILE) as f:
        return json.load(f)

def save(stage, substep, metrics=None, next_action=""):
    s = load()
    s["stage"] = stage
    s["substep"] = substep
    s["last_completed"] = datetime.datetime.utcnow().isoformat()
    if metrics:
        s["metrics"].update(metrics)
    s["next_action"] = next_action
    with open(STATE_FILE, "w") as f:
        json.dump(s, f, indent=2)

def log_error(stage, script, error):
    s = load()
    s["errors"].append({
        "stage": stage, "script": script,
        "error": str(error)[:400],
        "time": datetime.datetime.utcnow().isoformat()
    })
    with open(STATE_FILE, "w") as f:
        json.dump(s, f, indent=2)

def flag(name):
    open(f"{name}.flag", "w").close()

def done(name):
    return os.path.exists(f"{name}.flag")
