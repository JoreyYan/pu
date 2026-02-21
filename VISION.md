# Project Vision: Gaussian Ellipsoid as Universal Biomolecular Representation

> **Priority**: HIGHEST. All agents (Claude, Codex, Cursor, human) MUST read this document
> before making architectural decisions. Every code change should be evaluated against
> these goals.
>
> **Last updated**: 2026-02-21

---

## The Thesis

**Gaussian ellipsoids are to biomolecular structure what tokens are to language.**

A single representation — an oriented 3D Gaussian (position, rotation, scale, local offset)
— can describe any biomolecular entity at any granularity: a protein residue, a nucleotide,
a ligand functional group, or even a coarse-grained domain. IGA (Invariant Gaussian
Attention), built on Gaussian overlap, is the natural attention mechanism for this
representation — analogous to how scaled dot-product attention is natural for token
embeddings.

This is NOT a protein-specific tool. It is a **universal structural language**.

---

## Two End Goals

### Goal 1: Universal Biomolecular Structure Prediction

An "AlphaFold" built on Gaussian ellipsoids that handles:

- Proteins (backbone + sidechain)
- RNA / DNA
- Small molecules (ligands, drugs)
- Complexes (protein-protein, protein-ligand, protein-nucleic acid)

All in ONE unified model with ONE representation.

### Goal 2: High-Affinity Binder and Functional Binder Design

Given a target (protein, RNA, etc.), generate binders with:

- High binding affinity (experimentally validated)
- Designed functional properties
- Coverage of protein-protein, protein-small molecule, protein-nucleic acid interfaces

This requires Goal 1 as a foundation — the model must understand structure before it can design structure.

---

## Strategy: Foundation Model Approach

```
Phase    What                                      Validates
─────    ────                                      ─────────
  1      Protein backbone SE(3) flow matching       ✅ Done. Flow matching works.
  2      SH sidechain decoder                       ✅ Done. (info leakage found)
  3      IGA replaces IPA (6D backbone-only)        ✅ Done. IGA is competitive.
  4      12D: backbone + ellipsoid denoising        ◀ HERE. Can ellipsoids be learned?
  5      Full protein (backbone + sidechain)         Ellipsoid encodes full residue.
  6      Multi-molecule (RNA, DNA, ligands)          Ellipsoid is universal.
  7      Complex modeling (multi-chain, binding)     IGA captures inter-molecular interactions.
  8      Conditional generation (binder design)      Structure prediction → design.
  9      Affinity optimization + functional design   Final product.
```

**Current position: Phase 4.** The central question right now is: **can Gaussian ellipsoid
parameters be effectively denoised via flow matching?** If yes, the representation
generalizes to all molecule types. If no, we need to rethink the representation before
scaling up.

---

## Why This Ordering Matters

1. **Phase 4 is the critical gate.** If ellipsoid denoising doesn't work on proteins
   (where we have the most data and understanding), it won't work on RNA/ligands.
   Solve it here first.

2. **Structure prediction before design.** Just like GPT learned language before it
   could follow instructions, the model must learn to predict/reconstruct structure
   before it can design novel structures with desired properties.

3. **Single molecule before complex.** Modeling a protein-ligand complex requires
   understanding both the protein and the ligand individually. Gaussian overlap in
   IGA naturally extends to inter-molecular attention once intra-molecular works.

---

## Key Architectural Decisions (Open)

### Ellipsoid Granularity

| Molecule type  | Current / proposed granularity        | Open question                         |
|----------------|---------------------------------------|---------------------------------------|
| Protein        | 1 residue = 1 ellipsoid (12D)         | Settled for now                       |
| RNA / DNA      | 1 nucleotide = 1 ellipsoid? or 3?     | base + sugar + phosphate?             |
| Small molecule | 1 heavy atom = 1 ellipsoid? or group? | functional-group vs atom-level?       |
| Multi-scale    | Hierarchical ellipsoids?              | Needed for large complexes?           |

**This decision affects everything downstream.** A unified granularity scheme must be
chosen before Phase 6.

### IGA as Universal Attention

IGA computes attention weights from Gaussian overlap:

```
w_ij ~ exp(-0.5 * d^2_Mahalanobis(G_i, G_j) - 0.5 * log|Sigma_i + Sigma_j|)
```

This is physically motivated: nearby, shape-compatible residues attend more strongly.
For cross-molecule attention (e.g., protein-ligand), the same formula applies — a ligand
ellipsoid overlapping with a binding-pocket ellipsoid naturally gets high attention.

---

## Current Phase 4 Status (2026-02-21)

### What works
- 12D update chain (GaussianUpdateBlock) is correct — no bugs
- Backbone losses (trans=0.985, rot=1.346 at step 4709) are healthy
- IGA attention is not saturating (wmax_sat_frac=0.035)
- Geometry branch is active (geo_scaled_absmax=140.89)

### What's broken
- `ellipsoid_scaling_loss` has extreme spikes (mean=55.67, max~9000)
- This dominates `se3_vf_loss` (61.24), masking backbone progress
- Root cause: likely time normalization `1/t` at low t amplifying scaling errors

### Active changes (not yet in a training run)
- GT ellipsoid geometry for IGA attention (decouple attention from corrupted params)
- Pending: reduce `ellipsoid_scaling_loss_weight` from 1.0 to stabilize

### Current training run
- Run ID: `run-20260221_205904-4jfw6c1f` (old code, no GT geometry fix)
- Let it continue to ~10k steps as 12D baseline

---

## For AI Agents: How to Help

When working on this codebase, keep in mind:

1. **Every change to IGA / ellipsoid code is on the critical path.** Phase 4 success
   unlocks everything else. Treat it with care.

2. **Don't over-optimize for proteins.** Architectural decisions should consider future
   extension to RNA/DNA/ligands. Avoid protein-specific hacks.

3. **The ellipsoid representation must remain general.** `OffsetGaussianRigid` = (rotation,
   translation, scaling_log, local_mean). This parameterization should work for any
   molecular entity, not just amino acid residues.

4. **Log all experiments.** Use `analysis/openai_experiment_log/` for change records.
   Include: what changed, why, expected effect, W&B run name, results.

5. **Key files:**
   - `models/flow_model.py` — `FlowModelIGA` (main model)
   - `models/IGA.py` — `InvariantGaussianAttention`, `GaussianUpdateBlock`
   - `data/GaussianRigid.py` — `OffsetGaussianRigid` (core representation)
   - `data/interpolant.py` — noise corruption + flow matching
   - `models/loss.py` — loss computation
   - `configs/Train_fm.yaml` — training config

---

## Success Criteria

### Phase 4 (current): "Ellipsoids can be learned"
- [ ] `ellipsoid_scaling_loss` converges (no spikes, steady decrease)
- [ ] `ellipsoid_local_mean_loss` converges
- [ ] Backbone losses remain competitive with 6D IGA baseline
- [ ] Reconstructed ellipsoids visually match GT sidechain shapes

### Phase 5: "Full protein generation"
- [ ] Generated proteins have correct sidechain geometry (bond lengths, angles, planarity)
- [ ] Recovery rate competitive with ProteinMPNN
- [ ] TM-score and pLDDT competitive with FrameFlow IPA baseline

### Phase 6+: "Universality"
- [ ] Same model architecture handles protein + RNA with minimal modification
- [ ] Ellipsoid representation captures nucleotide geometry
- [ ] Cross-molecule attention (IGA) correctly identifies binding interfaces

### Ultimate: "Binder design"
- [ ] Generate binders with predicted high affinity (computational validation)
- [ ] Experimental validation of designed binders
