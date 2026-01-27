#!/usr/bin/env python3
"""
Fetch PDB deposition/release dates from RCSB for a list of PDB IDs in a CSV.

Input CSV must contain a column named 'pdb_name' (e.g., 3dyt, 4dyx).
Other columns will be preserved in the output.

Usage:
  python fetch_pdb_dates.py --in your.csv --out with_dates.csv
  # If your csv has a different column for PDB ids:
  python fetch_pdb_dates.py --in your.csv --out with_dates.csv --col pdb_name

Fields retrieved (when available):
- deposit_date:          Date the entry was deposited to PDB (YYYY-MM-DD)
- initial_release_date:  First public release date (YYYY-MM-DD)
- revision_date:         Most recent revision date (YYYY-MM-DD)
- entry_id:              Canonical PDB id (uppercase)
- status:                Release status (if present)

The script uses the RCSB REST endpoint:
  https://data.rcsb.org/rest/v1/core/entry/{pdb_id}
"""

import argparse
import sys
import time
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Dict, Any, Optional

import pandas as pd
import requests
from tqdm import tqdm

RCSB_ENTRY_URL = "https://data.rcsb.org/rest/v1/core/entry/{pdb_id}"
_THREAD_LOCAL = threading.local()
_SESSION_HEADERS = {"User-Agent": "fetch_pdb_dates/1.0 (https://rcsb.org)"}


def _get_thread_session() -> requests.Session:
    session = getattr(_THREAD_LOCAL, "session", None)
    if session is None:
        session = requests.Session()
        session.headers.update(_SESSION_HEADERS)
        _THREAD_LOCAL.session = session
    return session


def fetch_entry_dates(pdb_id: str, session: Optional[requests.Session] = None, retries: int = 3, backoff: float = 0.8) -> Dict[str, Optional[str]]:
    """
    Query RCSB REST API for a single PDB id and extract key dates.
    Returns a dict with dates or None on failure.
    """
    pdb_id_clean = str(pdb_id).strip().lower()
    url = RCSB_ENTRY_URL.format(pdb_id=pdb_id_clean)
    last_err: Optional[Exception] = None

    for attempt in range(retries):
        try:
            sess = session or _get_thread_session()
            resp = sess.get(url, timeout=10)
            if resp.status_code == 404:
                return {
                    "entry_id": pdb_id_clean.upper(),
                    "deposit_date": None,
                    "initial_release_date": None,
                    "revision_date": None,
                    "status": "NOT_FOUND",
                }
            resp.raise_for_status()
            data = resp.json()

            # Known locations for the fields:
            acc = data.get("rcsb_accession_info", {}) if isinstance(data, dict) else {}
            entry = data if isinstance(data, dict) else {}

            deposit_date = acc.get("deposit_date")
            initial_release_date = acc.get("initial_release_date")
            revision_date = acc.get("revision_date")

            # Sometimes status or other metadata may be present elsewhere; keep optional.
            status = acc.get("status") or entry.get("status")

            return {
                "entry_id": (entry.get("rcsb_id") or pdb_id_clean.upper()),
                "deposit_date": deposit_date,
                "initial_release_date": initial_release_date,
                "revision_date": revision_date,
                "status": status,
            }
        except Exception as e:
            last_err = e
            # Simple exponential backoff
            time.sleep(backoff * (2 ** attempt))
    # If we exhausted retries, return error markers
    return {
        "entry_id": pdb_id_clean.upper(),
        "deposit_date": None,
        "initial_release_date": None,
        "revision_date": None,
        "status": f"ERROR: {type(last_err).__name__ if last_err else 'Unknown'}",
    }

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="inp", required=True, help="Input CSV path (must contain a pdb_name column by default)")
    ap.add_argument("--out", dest="out", required=True, help="Output CSV path")
    ap.add_argument("--col", dest="col", default="pdb_name", help="Column name containing PDB IDs (default: pdb_name)")
    ap.add_argument("--batch_sleep", type=float, default=0.1, help="Sleep seconds between requests (default: 0.1)")
    ap.add_argument("--workers", type=int, default=32, help="Number of concurrent threads (default: 32)")
    args = ap.parse_args()

    df = pd.read_csv(args.inp)
    if args.col not in df.columns:
        print(f"ERROR: Column '{args.col}' not found in CSV. Available columns: {list(df.columns)}", file=sys.stderr)
        sys.exit(2)

    ids = df[args.col].astype(str).tolist()
    session = requests.Session()
    # Identify ourselves politely
    session.headers.update({"User-Agent": "fetch_pdb_dates/1.0 (https://rcsb.org)"})
    results = []
    num_workers = min(args.workers, max(1, len(ids)))
    with ThreadPoolExecutor(max_workers=num_workers) as executor:
        futures = {executor.submit(fetch_entry_dates, pid, session if num_workers == 1 else None): pid for pid in ids}
        for future in tqdm(as_completed(futures), total=len(futures), desc="Fetching PDB dates", unit="entry"):
            rec = future.result()
            results.append(rec)

    dates_df = pd.DataFrame(results)
    # Merge back to original by matching on uppercase ID vs original column
    df["_pdb_upper"] = df[args.col].astype(str).str.upper()
    dates_df["_pdb_upper"] = dates_df["entry_id"].astype(str).str.upper()
    merged = df.merge(dates_df.drop(columns=["entry_id"]), on="_pdb_upper", how="left")
    merged = merged.drop(columns=["_pdb_upper"])

    merged.to_csv(args.out, index=False)
    print(f"Done. Wrote: {args.out}")
    # Show a small preview
    with pd.option_context("display.max_columns", None):
        print(merged[[args.col, "deposit_date", "initial_release_date", "revision_date", "status"]].head())

if __name__ == "__main__":
    main()
