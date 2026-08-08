"""Small helpers used across modules."""

from __future__ import annotations

import datetime as dt
import platform
import subprocess
import sys
from pathlib import Path
from typing import Any

import psutil


def utc_now_iso() -> str:
    return dt.datetime.now(dt.UTC).isoformat(timespec="seconds")


def detect_hardware() -> dict[str, Any]:
    """Return hardware info used to auto-size batch / worker count.

    Uses ``sysctl`` on macOS to pull the CPU brand string. Falls back to platform info.
    """
    info: dict[str, Any] = {
        "platform": platform.platform(),
        "python": sys.version.split()[0],
        "cpu_count": psutil.cpu_count(logical=False) or 1,
        "logical_cpus": psutil.cpu_count(logical=True) or 1,
        "ram_gb": round(psutil.virtual_memory().total / (1024**3), 2),
        "arch": platform.machine(),
        "is_apple_silicon": False,
        "apple_chip": None,
    }
    if platform.system() == "Darwin" and info["arch"] == "arm64":
        info["is_apple_silicon"] = True
        try:
            out = subprocess.check_output(
                ["/usr/sbin/sysctl", "-n", "machdep.cpu.brand_string"],
                stderr=subprocess.DEVNULL,
            ).decode().strip()
            info["apple_chip"] = out
        except (subprocess.SubprocessError, FileNotFoundError, OSError):
            info["apple_chip"] = "Apple Silicon"
    return info


def auto_workers(cfg_workers: int | str, ram_gb: float) -> int:
    """Pick a safe worker count for unified-memory VLM inference."""
    if isinstance(cfg_workers, int):
        return max(1, cfg_workers)
    # 1 worker per ~8GB RAM, capped at physical cores - 1, min 1.
    cores = max(1, (psutil.cpu_count(logical=False) or 2) - 1)
    return max(1, min(cores, int(ram_gb // 8) or 1))


def auto_batch(cfg_batch: int | str, ram_gb: float) -> int:
    """Rows-per-DB-transaction. Small enough to survive crashes without losing much."""
    if isinstance(cfg_batch, int):
        return max(1, cfg_batch)
    if ram_gb < 12:
        return 16
    if ram_gb < 24:
        return 32
    return 64


def format_duration(seconds: float) -> str:
    seconds = max(0, int(seconds))
    h, rem = divmod(seconds, 3600)
    m, s = divmod(rem, 60)
    if h:
        return f"{h}h {m:02d}m {s:02d}s"
    if m:
        return f"{m}m {s:02d}s"
    return f"{s}s"


def free_disk_gb(path: Path) -> float:
    usage = psutil.disk_usage(str(path))
    return round(usage.free / (1024**3), 2)


def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path
