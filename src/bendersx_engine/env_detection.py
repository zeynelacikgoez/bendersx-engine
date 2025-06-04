"""Environment and capability detection utilities."""

from __future__ import annotations

import os
import shutil

try:
    import highspy  # type: ignore
except ImportError:  # pragma: no cover - highspy is optional
    highspy = None


try:
    import numba  # type: ignore

    NUMBA_AVAILABLE = True
except ImportError:  # pragma: no cover - numba is optional
    NUMBA_AVAILABLE = False


def check_highs_version() -> bool:
    """Check HiGHS version for multi-threading support."""
    if highspy is None:
        print("HiGHS not installed")
        return False

    try:
        version = highspy.Highs().versionNumber()
        major, minor = version.split(".")[:2]
        if int(minor) < 8:
            print(f"HiGHS {version} detected. Upgrade recommended")
            return False
        print(f"HiGHS {version} detected")
        return True
    except Exception as exc:  # pragma: no cover - version detection may fail
        print(f"Could not detect HiGHS version: {exc}")
        return False


def detect_gpu_support() -> bool:
    """Return True if a CUDA GPU seems available."""
    cuda_env = os.getenv("CUDA_VISIBLE_DEVICES", "") != ""
    nvidia_smi = shutil.which("nvidia-smi") is not None
    if cuda_env and nvidia_smi:
        print("CUDA environment detected")
        return True
    if cuda_env and not nvidia_smi:
        print("CUDA_VISIBLE_DEVICES set but no driver")
    return False


def setup_numba_cache() -> None:
    """Ensure Numba cache directory exists."""
    if NUMBA_AVAILABLE and "NUMBA_CACHE_DIR" not in os.environ:
        cache_dir = os.path.expanduser("~/.numba_cache")
        os.environ["NUMBA_CACHE_DIR"] = cache_dir
        os.makedirs(cache_dir, exist_ok=True)
        print(f"Numba cache directory: {cache_dir}")
