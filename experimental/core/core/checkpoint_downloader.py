"""
Downloads pretrained model weights on first use.
Ships nothing. Downloads once. Caches forever.
"""

import threading
import urllib.request
from pathlib import Path

CHECKPOINT_URL = (
    "https://github.com/RayAKaan/AAlgoI/releases/download/"
    "v1.2.0/pretrained_final.pt"
)

CHECKPOINT_DIR  = Path.home() / ".aalgoi" / "checkpoints"
CHECKPOINT_PATH = CHECKPOINT_DIR / "pretrained_final.pt"
CHECKPOINT_SIZE = 2_200_000


def get_checkpoint_path() -> str:
    """
    Returns the local path to the pretrained model.
    Downloads it first if not present. Blocks until download completes.
    """
    if CHECKPOINT_PATH.exists():
        return str(CHECKPOINT_PATH)

    _download_checkpoint(show_progress=True)
    return str(CHECKPOINT_PATH)


from typing import Any


def ensure_checkpoint_async() -> None:
    """
    Start downloading in background without blocking.
    Called on UniversalSolver.__init__() so the model
    is ready by the time solve() is first called.
    """
    if CHECKPOINT_PATH.exists():
        return

    thread = threading.Thread(
        target=_download_checkpoint,
        kwargs={"show_progress": False},
        daemon=True,
    )
    thread.start()


def checkpoint_exists() -> bool:
    return CHECKPOINT_PATH.exists()


_download_lock = threading.Lock()

def _download_checkpoint(show_progress: bool = True) -> None:
    with _download_lock:
        if CHECKPOINT_PATH.exists():
            return

        CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)

        if show_progress:
            print("AAlgoI: Downloading pretrained model (~2.2 MB)...", flush=True)

        tmp_path = CHECKPOINT_PATH.with_suffix(".tmp")

        try:
            _download_with_progress(
                url=CHECKPOINT_URL,
                dest=tmp_path,
                show_progress=show_progress,
            )

            if tmp_path.stat().st_size < CHECKPOINT_SIZE * 0.5:
                raise RuntimeError(f"Downloaded file too small: {tmp_path.stat().st_size} bytes")

            tmp_path.rename(CHECKPOINT_PATH)

            if show_progress:
                print(f"AAlgoI: Model saved to {CHECKPOINT_PATH}", flush=True)

        except Exception as e:
            if tmp_path.exists():
                tmp_path.unlink()
            if show_progress:
                print(
                    f"AAlgoI: Download failed ({e}). "
                    f"Continuing with random weights.",
                    flush=True,
                )


def _download_with_progress(url: str, dest: Path, show_progress: bool) -> None:
    def _reporthook(block_num: int, block_size: int, total_size: int) -> None:
        if not show_progress or total_size <= 0:
            return
        downloaded = block_num * block_size
        pct = min(100, downloaded * 100 // total_size)
        bar = "#" * (pct // 5) + "." * (20 - pct // 5)
        print(
            f"\r  [{bar}] {pct}%  {downloaded//1024}KB / {total_size//1024}KB",
            end="",
            flush=True,
        )
        if downloaded >= total_size:
            print()

    urllib.request.urlretrieve(url, dest, reporthook=_reporthook)
