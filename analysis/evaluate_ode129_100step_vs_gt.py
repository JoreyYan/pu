"""Re-evaluate ODE_129_100step ESMFold predictions against CASP15 GT."""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import List

import numpy as np
from Bio import pairwise2
from Bio.PDB import PDBParser

sys.path.append('/home/junyu/project/esm')
from genie.evaluations.compute_tm_rmsd import analyze_method as tm_analyze  # type: ignore
from genie.evaluations.analyze_plddt import analyze_method as plddt_analyze  # type: ignore


THREE_TO_ONE = {
    "ALA": "A",
    "CYS": "C",
    "ASP": "D",
    "GLU": "E",
    "PHE": "F",
    "GLY": "G",
    "HIS": "H",
    "ILE": "I",
    "LYS": "K",
    "LEU": "L",
    "MET": "M",
    "ASN": "N",
    "PRO": "P",
    "GLN": "Q",
    "ARG": "R",
    "SER": "S",
    "THR": "T",
    "VAL": "V",
    "TRP": "W",
    "TYR": "Y",
}


def extract_sequence(pdb_file: Path) -> str:
    parser = PDBParser(QUIET=True)
    structure = parser.get_structure('prot', str(pdb_file))
    seq: List[str] = []
    for residue in structure.get_residues():
        if residue.id[0] != ' ':
            continue
        seq.append(THREE_TO_ONE.get(residue.resname, 'X'))
    return ''.join(seq)


def seq_identity(pred_seq: str, gt_seq: str) -> float:
    align = pairwise2.align.globalxx(pred_seq, gt_seq, one_alignment_only=True)[0]
    matches = sum(1 for a, b in zip(align.seqA, align.seqB) if a != '-' and b != '-' and a == b)
    length = sum(1 for a, b in zip(align.seqA, align.seqB) if a != '-' and b != '-')
    return matches / length if length else 0.0


def main():
    pred_dir = Path('outputs/evaluation_129_complete/esmfold_predictions/ODE_129_100step')
    gt_dir = Path('/home/junyu/project/casp15/targets/casp15.targets.TS-domains.public_12.20.2022')

    tm_results = tm_analyze(str(pred_dir), 'ODE_129_100step', str(gt_dir))
    ca_vals = [r['ca_rmsd'] for r in tm_results if r.get('ca_rmsd') is not None]
    tm_vals = [r['tm_score'] for r in tm_results if r.get('tm_score') is not None]

    tm_summary = {
        'n': len(tm_results),
        'ca_rmsd_mean': float(np.mean(ca_vals)) if ca_vals else None,
        'ca_rmsd_std': float(np.std(ca_vals)) if ca_vals else None,
        'tm_mean': float(np.mean(tm_vals)) if tm_vals else None,
        'tm_std': float(np.std(tm_vals)) if tm_vals else None,
    }

    plddt_stats = plddt_analyze(str(pred_dir), 'ODE_129_100step')

    recoveries = []
    for entry in tm_results:
        sample = entry['sample_id']
        target = sample[len('sample_'):].split('_')[0]
        pred_seq = extract_sequence(pred_dir / f'{sample}_esmfold.pdb')
        gt_seq = extract_sequence(gt_dir / f'{target}.pdb')
        recoveries.append(seq_identity(pred_seq, gt_seq))

    summary = {
        'ca_rmsd_mean': tm_summary['ca_rmsd_mean'],
        'ca_rmsd_std': tm_summary['ca_rmsd_std'],
        'tm_mean': tm_summary['tm_mean'],
        'tm_std': tm_summary['tm_std'],
        'recovery_mean': float(np.mean(recoveries)) if recoveries else None,
        'plddt_mean': plddt_stats['per_sample_mean'] if plddt_stats else None,
        'plddt_std': plddt_stats['per_sample_std'] if plddt_stats else None,
        'plddt_distribution': {
            'very_high': plddt_stats['very_high'] if plddt_stats else None,
            'high': plddt_stats['high'] if plddt_stats else None,
            'medium': plddt_stats['medium'] if plddt_stats else None,
            'low': plddt_stats['low'] if plddt_stats else None,
        },
    }

    output = {
        'summary': summary,
        'tm_results': tm_results,
        'recoveries': recoveries,
    }
    out_path = Path('outputs/evaluation_129_complete/ode129_100step_vs_gt_metrics.json')
    out_path.write_text(json.dumps(output, indent=2))
    print(json.dumps(summary, indent=2))


if __name__ == '__main__':
    main()
