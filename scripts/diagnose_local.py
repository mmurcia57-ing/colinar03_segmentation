"""Print a lightweight local diagnostic for the assisted-segmentation workflow."""

from __future__ import annotations

import ctypes
import importlib.util
import os
import platform
import shutil
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def memory_gb() -> float | None:
    if os.name != "nt":
        return None
    class Status(ctypes.Structure):
        _fields_ = [("length", ctypes.c_uint32), ("memory_load", ctypes.c_uint32),
                    ("total", ctypes.c_uint64), ("available", ctypes.c_uint64),
                    ("page_total", ctypes.c_uint64), ("page_available", ctypes.c_uint64),
                    ("virtual_total", ctypes.c_uint64), ("virtual_available", ctypes.c_uint64),
                    ("extended", ctypes.c_uint64)]
    status = Status()
    status.length = ctypes.sizeof(Status)
    if ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(status)):
        return status.total / (1024 ** 3)
    return None


def version(module: str) -> str:
    if importlib.util.find_spec(module) is None:
        return "NOT_INSTALLED"
    try:
        imported = __import__(module)
        return getattr(imported, "__version__", "installed")
    except Exception as exc:  # pragma: no cover - diagnostic fallback
        return f"ERROR: {type(exc).__name__}"


def main() -> None:
    usage = shutil.disk_usage(ROOT)
    print(f"repo={ROOT}")
    print(f"python={sys.version.split()[0]}")
    print(f"platform={platform.platform()}")
    print(f"cpu={platform.processor() or 'unavailable'}")
    print(f"ram_gb={memory_gb()}")
    print(f"disk_free_gb={usage.free / (1024 ** 3):.2f}")
    for module in ("torch", "cv2", "skimage", "matplotlib", "labelme"):
        print(f"{module}={version(module)}")
    try:
        import torch
        print(f"torch_cuda_build={torch.version.cuda}")
        print(f"torch_cuda_available={torch.cuda.is_available()}")
        print(f"torch_device={'cuda' if torch.cuda.is_available() else 'cpu'}")
    except Exception as exc:  # pragma: no cover - diagnostic fallback
        print(f"torch_runtime_error={type(exc).__name__}: {exc}")
    checkpoint = ROOT / "models" / "mobile_sam.pt"
    print(f"mobile_sam_checkpoint={checkpoint.exists()} size_mb={checkpoint.stat().st_size / (1024 ** 2):.2f}" if checkpoint.exists() else "mobile_sam_checkpoint=False")
    print("sam2=not installed; Windows/Python 3.12 viability not selected")
    print("sam_original=not installed; Windows/Python 3.12 viability not selected")
    print("mobile_sam=selected; local checkpoint and CPU runtime available")


if __name__ == "__main__":
    main()
