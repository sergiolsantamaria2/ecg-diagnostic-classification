"""Download PhysioNet/CinC 2021 source databases into ``<data_dir>/raw/cinc2021/``.

The Challenge 2021 training set is split into per-source databases, each a tree of
``gN/`` subfolders holding paired WFDB records (``<rec>.hea`` header + ``<rec>.mat``
signal). Every subfolder lists its records in a ``RECORDS`` file, which drives the
download: each record's two files are fetched in parallel into a flat
``raw/cinc2021/<db>/`` directory. Downloads are resumable — records already present
on disk are skipped — and individual file fetches are retried on network errors.

The five databases used for cross-source pretraining (PTB-XL is held out, since it
is consumed natively with its diagnostic labels) are downloaded by default.
"""

from __future__ import annotations

import argparse
import re
import sys
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

BASE_URL = "https://physionet.org/files/challenge-2021/1.0.3/training"
REPO_ROOT = Path(__file__).resolve().parents[2]

# Five non-PTB-XL databases. PTB-XL is held out (used natively, with labels); PTB
# (small) and St Petersburg INCART (257 Hz, 30-min records) are atypical and dropped.
DEFAULT_DATABASES = (
    "cpsc_2018",
    "cpsc_2018_extra",
    "georgia",
    "chapman_shaoxing",
    "ningbo",
)


def _get(url: str, timeout: int = 60) -> bytes:
    with urllib.request.urlopen(url, timeout=timeout) as resp:
        return resp.read()


def list_subfolders(db: str) -> list[str]:
    """Names of the ``gN/`` subfolders under a database, from its index page."""
    html = _get(f"{BASE_URL}/{db}/").decode("utf-8", "replace")
    folders = sorted(set(re.findall(r'href="(g\d+)/"', html)))
    if not folders:
        raise RuntimeError(f"no gN subfolders found for {db!r}")
    return folders


def list_records(db: str, subfolder: str) -> list[str]:
    """Record stems listed in a subfolder's ``RECORDS`` file."""
    text = _get(f"{BASE_URL}/{db}/{subfolder}/RECORDS").decode("utf-8", "replace")
    return [line.strip() for line in text.splitlines() if line.strip()]


def fetch_file(url: str, dest: Path, retries: int = 10) -> bool:
    """Download ``url`` to ``dest`` atomically; return ``False`` if skipped.

    Transient errors (timeouts, 5xx) are retried; a 4xx (e.g. a ``RECORDS`` entry
    whose file is genuinely absent) is permanent, so it is skipped rather than
    retried or raised — one missing record must not abort a 65k-file download.
    """
    if dest.exists() and dest.stat().st_size > 0:
        return True
    tmp = dest.with_suffix(dest.suffix + ".part")
    for attempt in range(1, retries + 1):
        try:
            tmp.write_bytes(_get(url))
            tmp.rename(dest)
            return True
        except urllib.error.HTTPError as exc:
            if 400 <= exc.code < 500 and exc.code != 429:
                return False
            if attempt == retries:
                raise
            time.sleep(min(2 * attempt, 10))
        except Exception:  # noqa: BLE001 — any other network error is retriable
            if attempt == retries:
                raise
            time.sleep(min(2 * attempt, 10))
    return False


def download_database(db: str, raw_dir: Path, workers: int) -> int:
    """Download every record of one database into ``raw_dir/<db>/``."""
    out = raw_dir / db
    out.mkdir(parents=True, exist_ok=True)

    subfolders = list_subfolders(db)
    records: list[tuple[str, str]] = []  # (subfolder, stem)
    for sub in subfolders:
        records.extend((sub, stem) for stem in list_records(db, sub))
    print(f"{db}: {len(subfolders)} subfolders, {len(records)} records", flush=True)

    jobs: list[tuple[str, Path]] = []
    for sub, stem in records:
        for ext in (".hea", ".mat"):
            jobs.append((f"{BASE_URL}/{db}/{sub}/{stem}{ext}", out / f"{stem}{ext}"))

    done = skipped = 0
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {pool.submit(fetch_file, url, dest): url for url, dest in jobs}
        for fut in as_completed(futures):
            if not fut.result():
                skipped += 1
            done += 1
            if done % 2000 == 0 or done == len(jobs):
                print(f"  {db}: {done}/{len(jobs)} files", flush=True)
    if skipped:
        print(f"  {db}: skipped {skipped} missing files (absent on server)", flush=True)
    return len(records)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data-dir", type=Path, default=REPO_ROOT / "data", help="root data directory"
    )
    parser.add_argument(
        "--db",
        action="append",
        choices=(*DEFAULT_DATABASES, "all"),
        help="database(s) to download; repeatable. Default: all five.",
    )
    parser.add_argument("--workers", type=int, default=16, help="parallel download workers")
    args = parser.parse_args()

    dbs = DEFAULT_DATABASES if not args.db or "all" in args.db else tuple(dict.fromkeys(args.db))
    raw_dir = args.data_dir / "raw" / "cinc2021"
    raw_dir.mkdir(parents=True, exist_ok=True)

    for db in dbs:
        n = download_database(db, raw_dir, args.workers)
        print(f"Done {db}: {n} records at {raw_dir / db}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
