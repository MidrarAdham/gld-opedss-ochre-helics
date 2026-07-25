import os
import tomllib
from pathlib import Path
from datetime import datetime


def load_config (path=None):

    if path is None:
        path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "config", "config.toml")

    with open(path, "rb") as f:
        data = tomllib.load(f)
        up = data["data"]["upgrades"]
        if data["project"]["run_id"] == "auto":
            timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
            data["project"]["run_id"] = f"run_{timestamp}_{up[0]}"
    return data