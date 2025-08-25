# src/config.py
from __future__ import annotations
import os, yaml
from dataclasses import dataclass
from typing import Any, Dict

def load_config(path: str) -> Dict[str, Any]:
    with open(path, 'r', encoding='utf-8') as f:
        cfg = yaml.safe_load(f)
    return cfg

def ensure_dirs(cfg: Dict[str, Any]) -> None:
    p = cfg['paths']
    for k in ['artifacts_dir', 'checkpoints_dir', 'metrics_dir', 'logs_dir']:
        os.makedirs(p[k], exist_ok=True)
