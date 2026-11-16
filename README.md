# CHEMSMART-ML

Independent CHEMSMART plugin for low-to-high level molecular property prediction.
Version 0.1 implements an auditable linear energy delta baseline, not a learned
interatomic potential. Tree and 3D graph backends are future work.

## Installation and execution

```sh
python -m pip install -e '.[test]'
chemsmart-ml -f xtb.db -r dft.db train --pairing revqm9-index -o runs/baseline
chemsmart-ml -f new_xtb.db predict --model runs/baseline/model.json -o predictions.csv
```

Add `--features electronic` to train with xTB HOMO/LUMO (eV) and dipole magnitude
(Debye), in addition to element counts and xTB total energy (Hartree). Missing
features fail explicitly. Only source-side features enter the model.

With a compatible CHEMSMART host supporting `chemsmart.job_commands` entry points:

```sh
chemsmart run ml -f xtb.db -r dft.db train --pairing revqm9-index -o runs/baseline
chemsmart sub ml -f xtb.db -r dft.db train --pairing revqm9-index -o runs/cluster
```

The current main CHEMSMART checkout lacks plugin discovery. A plugin-capable host
is required for these two commands; standalone execution does not require the
host. Scheduler submission and actual cluster execution require separate host
integration validation. Configure an ML environment in host server settings and
use persistent paths, with scratch disabled.

## Dataset contract

This version supports CHEMSMART databases using the explicit `revqm9-index`
convention: source paths contain `idxN` (one-based), reference
`record_structures.structure_index_in_file` is N (one-based). The two
files MUST derive from the same NPZ ordering. This rule is not a general-purpose
identity matcher. No row-order or InChIKey matching fallback is used.

One final stored geometry is selected per record by maximum index_in_record.
Before production use, confirm your importer stores this geometry and
records.total_energy as the same final state. Element composition mismatches
are reported and block training; molecular identity changes are counted, retained,
and grouped together across both endpoints for splitting. The adapters reject
ambiguous duplicate indices. Energy units must be Hartree on both sides;
this version relies on the CHEMSMART database convention, not automatic unit
inference. Source/reference methods must be homogeneous and verified by the user.

## Learning and artifacts

Fit a standardized least-squares model of DFT minus xTB energy using only the
training set. Element counts provide composition-dependent offsets. Stable
seeded hashes allocate molecular connected components to approximately 80/10/10
train/validation/test splits. Neither source nor reference molecule identities
can cross splits. This groups stored identities, not independently verified
chemical equivalence. Small datasets can produce empty splits and are rejected.

Outputs: `pairing.json` (unmatched and rejected indices), `samples.csv`
(explicit pairing, split, predictions), `model.json` (portable coefficients and
training configuration), `metrics.json` (MAE in Hartree). Output directories must
be new; artifacts are not silently overwritten. Failed runs may leave partial
reports; use a new directory after correcting the cause.

The target is the reference database energy at its stored geometry. If xTB and
DFT geometries differ, this learns both method and geometry effects. Training
only on minima does not provide trajectory force labels or a dynamics model.
No accuracy claim is made before held-out evaluation. Large-kernel models,
resuming, GPU training, automatic hyperparameter search, structural descriptors,
and arbitrary property targets are not implemented in this baseline release.

## Validation

```sh
python -m pytest
```

Tests cover zero/one-based pairing, duplicate rejection, element mismatches,
transitive leakage grouping, exact recovery of a synthetic delta relationship,
and model prediction round-trip. Real database training is a separate check.

## Previous pairing bug

The first implementation incorrectly used NPZ inds as array positions. These
fields diverge starting at position 20001 in the supplied data. Pairing now uses
the stored one-based structure_index_in_file, independently of inds.
The plugin stops training when composition mismatches are present. Matching
composition alone is necessary but does not establish molecular identity.

## Explicit model selection and development host

Training accepts `--model linear` (ordinary least squares, default) or
`--model ridge --alpha 1.0` (L2 regularization; intercept unpenalized).
Both learn the energy delta; neither is a graph neural network.

The local plugin-capable development host is `/Users/tt/chemsmart-mlip-host`.
From outside a conflicting CHEMSMART source checkout, select that host using
`PYTHONPATH=/Users/tt/chemsmart-mlip-host` before invoking the CHEMSMART
executable in the chemsmart environment. This does not update the main checkout.
Do not assume plain `chemsmart` selects that development host.

## YAML configuration and CLI overrides

```sh
chemsmart run ml -p examples/energy.yaml train
chemsmart run ml -p examples/energy.yaml train --model linear --seed 123 -o /tmp/new-run
chemsmart sub ml -p examples/energy.yaml train --model ridge --alpha 0.1
```

Priority: explicit CLI arguments > YAML > command defaults. Input overrides
`-f` and `-r` belong before `train`, while training overrides belong after it.
YAML paths resolve relative to the YAML file; CLI paths resolve from the working
directory. Unknown YAML keys fail instead of being silently ignored. The `data`
section accepts `filename` and `reference`; `train` accepts the training CLI
option names with underscores; `predict` accepts `model_path` and `output`.
The development plugin-host requirement above still applies.
