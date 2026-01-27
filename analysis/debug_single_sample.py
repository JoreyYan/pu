"""Debug TMscore alignment for a single sample."""

from pathlib import Path
import subprocess
from Bio import pairwise2

THREE_TO_ONE = {
    "ALA": "A", "CYS": "C", "ASP": "D", "GLU": "E",
    "PHE": "F", "GLY": "G", "HIS": "H", "ILE": "I",
    "LYS": "K", "LEU": "L", "MET": "M", "ASN": "N",
    "PRO": "P", "GLN": "Q", "ARG": "R", "SER": "S",
    "THR": "T", "VAL": "V", "TRP": "W", "TYR": "Y",
}


def extract_seq(path: Path) -> str:
    seq = []
    seen = set()
    with path.open() as fh:
        for line in fh:
            if not line.startswith("ATOM"):
                continue
            res_id = (line[21], line[17:20], line[22:27])
            if res_id in seen:
                continue
            seen.add(res_id)
            seq.append(THREE_TO_ONE.get(line[17:20].strip(), "X"))
    return "".join(seq)


def main():
    tm_exec = Path("/home/junyu/project/esm/genie/packages/TMscore/TMscore")
    pred = Path("outputs/evaluation_129_complete/esmfold_predictions/ODE_129_100step/sample_T1104-D1_000034_esmfold.pdb")
    gt = Path("/home/junyu/project/casp15/targets/casp15.targets.TS-domains.public_12.20.2022/T1104-D1.pdb")

    proc = subprocess.run([str(tm_exec), str(pred), str(gt)], capture_output=True, text=True)
    print(proc.stdout)

    seq_pred = extract_seq(pred)
    seq_gt = extract_seq(gt)
    align = pairwise2.align.globalxx(seq_pred, seq_gt, one_alignment_only=True)[0]
    print("Seq align:")
    print(align.seqA)
    print(align.seqB)


if __name__ == "__main__":
    main()
