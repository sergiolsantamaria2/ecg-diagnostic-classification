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
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import requests
from requests.adapters import HTTPAdapter

BASE_URL = "https://physionet.org/files/challenge-2021/1.0.3/training"
REPO_ROOT = Path(__file__).resolve().parents[2]

# One pooled session shared across workers: HTTP keep-alive reuses TCP+TLS
# connections instead of paying a handshake per file (~2.6x faster), and the
# bounded pool keeps the concurrent-connection count low enough to avoid throttling.
_SESSION: requests.Session | None = None


def _session(workers: int) -> requests.Session:
    global _SESSION
    if _SESSION is None:
        s = requests.Session()
        s.headers["User-Agent"] = "deep-ecg-downloader"
        adapter = HTTPAdapter(pool_connections=workers, pool_maxsize=workers, max_retries=0)
        s.mount("https://", adapter)
        _SESSION = s
    return _SESSION

# Five non-PTB-XL databases. PTB-XL is held out (used natively, with labels); PTB
# (small) and St Petersburg INCART (257 Hz, 30-min records) are atypical and dropped.
DEFAULT_DATABASES = (
    "cpsc_2018",
    "cpsc_2018_extra",
    "georgia",
    "chapman_shaoxing",
    "ningbo",
)


def _get(url: str, workers: int = 1, timeout: tuple[int, int] = (15, 60)) -> bytes:
    resp = _session(workers).get(url, timeout=timeout)
    resp.raise_for_status()
    return resp.content


def list_subfolders(db: str) -> list[str]:
    """Names of the ``gN/`` subfolders under a database, from its index page."""
    import re

    html = _get(f"{BASE_URL}/{db}/").decode("utf-8", "replace")
    folders = sorted(set(re.findall(r'href="(g\d+)/"', html)))
    if not folders:
        raise RuntimeError(f"no gN subfolders found for {db!r}")
    return folders


def list_records(db: str, subfolder: str) -> list[str]:
    """Record stems listed in a subfolder's ``RECORDS`` file."""
    text = _get(f"{BASE_URL}/{db}/{subfolder}/RECORDS").decode("utf-8", "replace")
    return [line.strip() for line in text.splitlines() if line.strip()]


def fetch_file(url: str, dest: Path, workers: int, retries: int = 10) -> bool:
    """Download ``url`` to ``dest`` atomically; return ``False`` if not fetched.

    A 4xx (e.g. a ``RECORDS`` entry whose file is genuinely absent) is permanent
    and skipped at once. Transient errors (timeouts, 5xx) are retried, but once
    retries are exhausted the file is left unfetched rather than raised: one
    record — or even a brief PhysioNet outage — must never abort a 40k-file
    download. Missing files stay off disk, so re-running the script retries them.
    """
    if dest.exists() and dest.stat().st_size > 0:
        return True
    tmp = dest.with_suffix(dest.suffix + ".part")
    for attempt in range(1, retries + 1):
        try:
            tmp.write_bytes(_get(url, workers))
            tmp.rename(dest)
            return True
        except requests.HTTPError as exc:
            if 400 <= exc.response.status_code < 500 and exc.response.status_code != 429:
                return False
            time.sleep(min(2 * attempt, 10))
        except Exception:  # noqa: BLE001 — any other network error is retriable
            time.sleep(min(2 * attempt, 10))
    return False


def download_database(db: str, raw_dir: Path, workers: int, max_records: int | None = None) -> int:
    """Download records of one database into ``raw_dir/<db>/``.

    With ``max_records`` set, only the first that many records (in subfolder /
    ``RECORDS`` order) are fetched — enough to build a source-balanced corpus
    without downloading a large database in full.
    """
    out = raw_dir / db
    out.mkdir(parents=True, exist_ok=True)

    subfolders = list_subfolders(db)
    records: list[tuple[str, str]] = []  # (subfolder, stem)
    for sub in subfolders:
        records.extend((sub, stem) for stem in list_records(db, sub))
    if max_records is not None:
        records = records[:max_records]
    print(f"{db}: {len(subfolders)} subfolders, {len(records)} records", flush=True)

    jobs: list[tuple[str, Path]] = []
    for sub, stem in records:
        for ext in (".hea", ".mat"):
            jobs.append((f"{BASE_URL}/{db}/{sub}/{stem}{ext}", out / f"{stem}{ext}"))

    done = missing = 0
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {pool.submit(fetch_file, url, dest, workers): url for url, dest in jobs}
        for fut in as_completed(futures):
            if not fut.result():
                missing += 1
            done += 1
            if done % 2000 == 0 or done == len(jobs):
                print(f"  {db}: {done}/{len(jobs)} files", flush=True)
    return len(records), missing


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
    parser.add_argument("--workers", type=int, default=12, help="parallel download workers")
    parser.add_argument(
        "--max-records",
        type=int,
        default=None,
        help="cap records downloaded per database (for a source-balanced corpus)",
    )
    args = parser.parse_args()

    dbs = DEFAULT_DATABASES if not args.db or "all" in args.db else tuple(dict.fromkeys(args.db))
    raw_dir = args.data_dir / "raw" / "cinc2021"
    raw_dir.mkdir(parents=True, exist_ok=True)

    for db in dbs:
        prev_missing = -1
        for pass_i in range(1, 6):  # repeat until a pass fetches nothing new
            n, missing = download_database(db, raw_dir, args.workers, args.max_records)
            if missing == 0 or missing == prev_missing:
                break
            print(f"  {db}: {missing} files still missing, retry pass {pass_i + 1}", flush=True)
            prev_missing = missing
        note = f" ({missing} absent on server)" if missing else ""
        print(f"Done {db}: {n} records at {raw_dir / db}{note}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
