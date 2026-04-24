import os
import json


DEFAULT_EMBED_DIM = 1536
DEFAULT_METRIC = "cosine"


def load_config() -> dict:
    cfg_path = os.getenv("CONFIG_PATH") or os.path.join(os.path.dirname(__file__), "..", "..", "config.json")
    cfg_path = os.path.abspath(cfg_path)
    with open(cfg_path, "r", encoding="utf-8") as f:
        return json.load(f)