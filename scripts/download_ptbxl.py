"""Download and extract PTB-XL (PhysioNet) into ``<data_dir>/raw/ptbxl/``.

The dataset is published as a single ~1.7 GB ZIP containing both the 100 Hz and
500 Hz signals. Download is resumable (HTTP Range) and retried, so an
interrupted connection continues from the last byte instead of restarting.
"""

from __future__ import annotations

import argparse
import sys
import time
import urllib.request
import zipfile
from pathlib import Path

PTBXL_URL = (
    "https://physionet.org/static/published-projects/ptb-xl/"
    "ptb-xl-a-large-publicly-available-electrocardiography-dataset-1.0.3.zip"
)
EXTRACTED_DIRNAME = "ptb-xl-a-large-publicly-available-electrocardiography-dataset-1.0.3"
REPO_ROOT = Path(__file__).resolve().parents[1]


def _remote_size(url: str) -> int:
    """Total file size in bytes, via HEAD with a ranged-GET fallback."""
    try:
        with urllib.request.urlopen(urllib.request.Request(url, method="HEAD"), timeout=60) as resp:
            length = resp.headers.get("Content-Length")
            if length:
                return int(length)
    except Exception:  # noqa: BLE001 — fall back to a ranged GET
        pass
    req = urllib.request.Request(url, headers={"Range": "bytes=0-0"})
    with urllib.request.urlopen(req, timeout=60) as resp:
        # Content-Range: "bytes 0-0/<total>"
        return int(resp.headers["Content-Range"].split("/")[-1])


def download_with_resume(url: str, dest: Path, retries: int = 200, chunk: int = 1 << 20) -> None:
    """Stream ``url`` to ``dest``, resuming until the full size is on disk.

    A clean EOF before the expected size (a dropped connection) is treated as a
    partial download and retried with an HTTP Range request, not as completion.
    The file is only renamed into place once every byte has been received.
    """
    tmp = dest.with_suffix(dest.suffix + ".part")
    total = _remote_size(url)
    print(f"  expected size: {total / 1e6:.0f} MB")

    for attempt in range(1, retries + 1):
        pos = tmp.stat().st_size if tmp.exists() else 0
        if pos >= total:
            break
        try:
            req = urllib.request.Request(url, headers={"Range": f"bytes={pos}-"})
            with urllib.request.urlopen(req, timeout=60) as resp:
                with open(tmp, "ab" if pos else "wb") as f:
                    while True:
                        block = resp.read(chunk)
                        if not block:
                            break
                        f.write(block)
                        pos += len(block)
                        print(
                            f"\r  {pos / 1e6:7.0f} MB ({100 * pos / total:.1f}%)",
                            end="",
                            flush=True,
                        )
            print()
        except Exception as exc:  # noqa: BLE001 — any network error is retriable
            got = tmp.stat().st_size / 1e6 if tmp.exists() else 0
            print(
                f"\n  [attempt {attempt}] {type(exc).__name__}: {exc} — resuming from {got:.0f} MB"
            )
            time.sleep(3)

    size = tmp.stat().st_size if tmp.exists() else 0
    if size != total:
        raise RuntimeError(f"incomplete download: {size}/{total} bytes")
    tmp.rename(dest)


def extract(zip_path: Path, raw_dir: Path) -> None:
    """Extract the ZIP and normalise the folder to ``raw_dir/ptbxl``."""
    print(f"Extracting {zip_path.name} ...")
    with zipfile.ZipFile(zip_path) as z:
        z.extractall(raw_dir)
    src, dst = raw_dir / EXTRACTED_DIRNAME, raw_dir / "ptbxl"
    if dst.exists():
        import shutil

        shutil.rmtree(dst)
    src.rename(dst)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=REPO_ROOT / "data",
        help="root data directory (default: <repo>/data)",
    )
    parser.add_argument(
        "--keep-zip", action="store_true", help="keep the downloaded ZIP after extraction"
    )
    args = parser.parse_args()

    raw_dir = args.data_dir / "raw"
    raw_dir.mkdir(parents=True, exist_ok=True)
    ptbxl_dir = raw_dir / "ptbxl"

    if (ptbxl_dir / "ptbxl_database.csv").exists():
        print(f"PTB-XL already present at {ptbxl_dir}")
        return 0

    zip_path = raw_dir / "ptbxl.zip"
    print(f"Downloading PTB-XL to {zip_path}")
    download_with_resume(PTBXL_URL, zip_path)
    extract(zip_path, raw_dir)
    if not args.keep_zip:
        zip_path.unlink(missing_ok=True)

    ok = (ptbxl_dir / "ptbxl_database.csv").exists()
    print(f"Done. PTB-XL at {ptbxl_dir}" if ok else "ERROR: database CSV missing")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
