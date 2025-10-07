#!/usr/bin/env python3
import argparse
import os
import re
import sys
from pathlib import Path

try:
    import yaml
except Exception:
    yaml = None

import csv


def load_cfg_csv_path(cfg_path: Path) -> Path | None:
    if yaml is None or not cfg_path.exists():
        return None
    try:
        with open(cfg_path, 'r') as f:
            cfg = yaml.safe_load(f)
    except Exception:
        return None

    # First try the fallback: scan for any key named csv_path and pick the first valid one
    def find_csv_path(d):
        if isinstance(d, dict):
            for k, v in d.items():
                if k == 'csv_path' and isinstance(v, str):
                    p = Path(v)
                    if not p.is_absolute():
                        p = cfg_path.parent / p
                    if p.exists():  # Only return if the file actually exists
                        return p
                res = find_csv_path(v)
                if res is not None:
                    return res
        return None

    res = find_csv_path(cfg)
    if res is not None:
        return res

    # If no existing file found, try common locations
    for key_path in [
        ['data', 'csv_path'],
        ['val_dataset', 'csv_path'], 
        ['scope_dataset', 'csv_path'],
        ['pdb_dataset', 'csv_path'],
        ['datasets', 'csv_path'],
    ]:
        cur = cfg
        ok = True
        for k in key_path:
            if not isinstance(cur, dict) or k not in cur:
                ok = False
                break
            cur = cur[k]
        if ok and isinstance(cur, str) and cur:
            p = Path(cur)
            if not p.is_absolute():
                p = cfg_path.parent / p
            return p

    return None


def load_name_map(csv_path: Path) -> list[str] | None:
    if not csv_path or not csv_path.exists():
        return None
    try:
        import pandas as pd
        df = pd.read_csv(csv_path)
        col = None
        for c in ['pdb_name', 'name']:
            if c in df.columns:
                col = c
                break
        if col is None:
            return None
        return df[col].astype(str).tolist()
    except Exception:
        # very small csv fallback
        try:
            with open(csv_path, 'r') as f:
                hdr = f.readline().strip().split(',')
                col_idx = None
                for c in ['pdb_name', 'name']:
                    if c in hdr:
                        col_idx = hdr.index(c)
                        break
                if col_idx is None:
                    return None
                names = []
                for line in f:
                    parts = line.strip().split(',')
                    if len(parts) > col_idx:
                        names.append(parts[col_idx])
                return names
        except Exception:
            return None


def sanitize(s: str) -> str:
    return ''.join(ch if ch.isalnum() or ch in ('_', '-') else '_' for ch in s)


def retag_run(run_dir: Path, name_map: list[str] | None, dry_run: bool = False) -> list[tuple[str, str]]:
    mapping = []
    sample_dirs = sorted([p for p in run_dir.glob('sample_*') if p.is_dir()])
    id_re = re.compile(r'^sample_(\d+)$')
    for sd in sample_dirs:
        m = id_re.match(sd.name)
        if not m:
            # already retagged or custom
            continue
        sid = int(m.group(1))
        base_name = None
        if name_map is not None and 0 <= sid < len(name_map):
            base_name = sanitize(str(name_map[sid]))
        if base_name is None:
            # keep numeric tag but zero-pad to 6
            new_name = f'sample_{sid:06d}'
        else:
            new_name = f'sample_{base_name}_{sid:06d}'

        if new_name == sd.name:
            continue
        dst = sd.parent / new_name
        i = 1
        while dst.exists():
            dst = sd.parent / f'{new_name}_{i}'
            i += 1
        mapping.append((sd.name, dst.name))
        if not dry_run:
            os.rename(sd, dst)

            # also retag fasta headers
            fasta = dst / 'sequence.fasta'
            if fasta.exists():
                try:
                    with open(fasta, 'r') as f:
                        lines = f.readlines()
                    lines = [
                        re.sub(r'^>original_\d+', f'>original_{sid:06d}', ln)
                        if ln.startswith('>original_') else ln
                        for ln in lines
                    ]
                    lines = [
                        re.sub(r'^>predicted_\d+', f'>predicted_{sid:06d}', ln)
                        if ln.startswith('>predicted_') else ln
                        for ln in lines
                    ]
                    with open(fasta, 'w') as f:
                        f.writelines(lines)
                except Exception:
                    pass
    # write mapping log
    if mapping and not dry_run:
        with open(run_dir / 'retag_mapping.csv', 'w', newline='') as f:
            w = csv.writer(f)
            w.writerow(['old', 'new'])
            w.writerows(mapping)
    return mapping


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--root', type=str, default='inference_outputs', help='Root dir of inference outputs or a single run dir')
    ap.add_argument('--csv', type=str, default='', help='Optional CSV path with names (overrides config.yaml)')
    ap.add_argument('--dry', action='store_true', help='Dry run: only print planned changes')
    ap.add_argument('--verbose', '-v', action='store_true', help='Verbose output')
    args = ap.parse_args()

    root = Path(args.root).expanduser().resolve()
    if not root.exists():
        print(f'Root not found: {root}')
        sys.exit(1)

    runs = [root]
    if not (root / 'config.yaml').exists():
        runs = [p for p in root.iterdir() if p.is_dir() and (p / 'config.yaml').exists()]
    if not runs:
        print('No runs with config.yaml found under root')
        sys.exit(0)

    for run in runs:
        cfg_csv = load_cfg_csv_path(run / 'config.yaml')
        csv_path = Path(args.csv) if args.csv else cfg_csv
        name_map = load_name_map(csv_path) if csv_path else None
        
        if args.verbose:
            print(f'Processing {run.name}:')
            print(f'  CSV path: {csv_path}')
            print(f'  Names loaded: {len(name_map) if name_map else 0}')
        
        mapping = retag_run(run, name_map, dry_run=args.dry)
        tag = '(dry-run) ' if args.dry else ''
        print(f'{tag}{run.name}: {len(mapping)} retagged')
        
        if args.verbose and mapping:
            for old, new in mapping[:5]:  # Show first 5 mappings
                print(f'  {old} -> {new}')
            if len(mapping) > 5:
                print(f'  ... and {len(mapping) - 5} more')


if __name__ == '__main__':
    main()


