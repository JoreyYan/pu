"""Run TM-score on 100-step ODE ESMFold predictions vs CASP15 GT."""

from __future__ import annotations

import json
import subprocess
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
from Bio import pairwise2

TM_EXEC = Path("/home/junyu/project/esm/genie/packages/TMscore/TMscore")
PRED_DIR = Path("outputs/evaluation_129_complete/esmfold_predictions/ODE_129_100step")
GT_DIR = Path("/home/junyu/project/casp15/targets/casp15.targets.TS-domains.public_12.20.2022")

THREE_TO_ONE = {
    "ALA": "A", "CYS": "C", "ASP": "D", "GLU": "E", "PHE": "F",
    "GLY": "G", "HIS": "H", "ILE": "I", "LYS": "K", "LEU": "L",
    "MET": "M", "ASN": "N", "PRO": "P", "GLN": "Q", "ARG": "R",
    "SER": "S", "THR": "T", "VAL": "V", "TRP": "W", "TYR": "Y",
}


def parse_tm_output(text: str) -> Tuple[float | None, float | None]:
    tm = None
    rmsd = None
    for line in text.splitlines():
        if line.startswith("TM-score") and tm is None:
            try:
                tm = float(line.split("=")[1].split("(")[0].strip())
            except Exception:
                pass
        elif line.startswith("RMSD of"):
            try:
                rmsd = float(line.split("=")[1].strip())
            except Exception:
                pass
        if tm is not None and rmsd is not None:
            break
    return tm, rmsd


def read_sequence(pdb_path: Path) -> str:
    seq = []
    seen = set()
    with pdb_path.open() as fh:
        for line in fh:
            if not line.startswith("ATOM"):
                continue
            res_id = (line[21], line[17:20], line[22:27])
            if res_id in seen:
                continue
            seen.add(res_id)
            seq.append(THREE_TO_ONE.get(line[17:20].strip(), "X"))
    return "".join(seq)


def mean_plddt(pdb_path: Path) -> float | None:
    values = []
    with pdb_path.open() as fh:
        for line in fh:
            if line.startswith("ATOM") and line[13:15].strip() == "CA":
                try:
                    values.append(float(line[60:66]))
                except ValueError:
                    continue
    return float(np.mean(values)) if values else None


def seq_identity(seq_a: str, seq_b: str) -> tuple[float, float]:
    if not seq_a or not seq_b:
        return float("nan"), float("nan")
    align = pairwise2.align.globalxx(seq_a, seq_b, one_alignment_only=True)[0]
    matches = gap_free = 0
    for a, b in zip(align.seqA, align.seqB):
        if a != "-" and b != "-":
            gap_free += 1
            if a == b:
                matches += 1
    identity = (matches / gap_free) if gap_free else float("nan")
    full = matches / len(seq_b) if seq_b else float("nan")
    return full, identity


def main() -> None:
    samples: Dict[str, Dict[str, float]] = {}

    for pred_pdb in sorted(PRED_DIR.glob("*_esmfold.pdb")):
        sample = pred_pdb.stem.replace("_esmfold", "")
        target = sample[len("sample_"):].split("_")[0]
        gt_pdb = GT_DIR / f"{target}.pdb"
        if not gt_pdb.exists():
            continue

        proc = subprocess.run(
            [str(TM_EXEC), str(pred_pdb), str(gt_pdb)],
            capture_output=True,
            text=True,
        )
        tm, rmsd = parse_tm_output(proc.stdout)
        seq_pred = read_sequence(pred_pdb)
        seq_gt = read_sequence(gt_pdb)
        full_rec, gapfree_id = seq_identity(seq_pred, seq_gt)
        plddt = mean_plddt(pred_pdb)

        samples[sample] = {
            "tm_score": tm if tm is not None else float("nan"),
            "ca_rmsd": rmsd if rmsd is not None else float("nan"),
            "recovery": full_rec,
            "identity_gapfree": gapfree_id,
            "plddt": plddt if plddt is not None else float("nan"),
        }

    tm_vals = [v["tm_score"] for v in samples.values() if np.isfinite(v["tm_score"])]
    rmsd_vals = [v["ca_rmsd"] for v in samples.values() if np.isfinite(v["ca_rmsd"])]
    rec_vals = [v["recovery"] for v in samples.values() if np.isfinite(v["recovery"])]
    plddt_vals = [v["plddt"] for v in samples.values() if np.isfinite(v["plddt"])]

    summary = {
        "n": len(samples),
        "tm_mean": float(np.mean(tm_vals)) if tm_vals else float("nan"),
        "tm_std": float(np.std(tm_vals)) if tm_vals else float("nan"),
        "ca_rmsd_mean": float(np.mean(rmsd_vals)) if rmsd_vals else float("nan"),
        "ca_rmsd_std": float(np.std(rmsd_vals)) if rmsd_vals else float("nan"),
        "recovery_mean": float(np.mean(rec_vals)) if rec_vals else float("nan"),
        "plddt_mean": float(np.mean(plddt_vals)) if plddt_vals else float("nan"),
    }

    out_path = Path("outputs/evaluation_129_complete/tmscore_recovery_plddt.json")
    out_path.write_text(json.dumps({"summary": summary, "samples": samples}, indent=2))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
