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

The local plugin-capable development host is `/Users/tt/chemsmart-mlip/chemsmart-plugin-work`.
From outside a conflicting CHEMSMART source checkout, select that host using
`PYTHONPATH=/Users/tt/chemsmart-mlip/chemsmart-plugin-work` before invoking the CHEMSMART
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

The intended CHEMSMART integration base is the `plugin` branch. Its local
checkout above successfully discovers `run ml` with the YAML-enabled CLI.
Previous full-data training used the older integration host; scheduler execution
and full training on this exact plugin branch remain separate validation steps.

## Plot saved training results

Matplotlib is installed with the plugin; the `[plot]` extra remains supported.
On the plugin-capable host:

```sh
chemsmart run ml plot --workdir ./runs/energy-linear
chemsmart run ml plot --workdir ./runs/energy-linear --split valid -o ./validation-plots
```

This local command requires only samples.csv, not the original databases. It
exports a three-panel PNG and PDF (prediction/reference, error histogram,
error/reference energy), plus JSON statistics (MAE, RMSE, bias, 95th percentile
absolute error). Default split is test; train and valid are selectable. Default
output is WORKDIR/plots. Existing output files are protected against overwrite.
`sub ml plot` is rejected because plotting runs locally. The standalone command
`chemsmart-ml plot` exposes the same behavior. No learning curve is available for
the closed-form linear models. Total-energy agreement should be assessed with
the residual panels, since a broad absolute-energy range can obscure errors.

## CHEMSMART named projects

Store configurations in `~/.chemsmart/ml/NAME.yaml`, then use
`chemsmart run ml -p NAME train` (no extension required). The configuration root
uses CHEMSMARTUserSettings.resolve_config_dir(), including CHEMSMART_CONFIG_DIR.
Explicit YAML paths remain supported. Project-name shell completion is provided
through Click; the shell must already have CHEMSMART completion enabled.
CLI overrides retain precedence. YAML-relative paths still resolve against the
configuration directory; use absolute dataset/output paths for shared presets.

`--features orbitals` adds HOMO and LUMO without requiring dipole magnitude.
`electronic` still requires all three electronic properties. Each paired training
run writes feature-availability.json, including counts of finite energy/HOMO/LUMO/
dipole inputs. Missing feature errors identify the field and record explicitly.
The supplied xTB database has complete HOMO/LUMO fields but no stored structure
dipoles, so use orbitals for its next experiment. Failed runs leave a directory;
choose a fresh output directory rather than overwriting that report.

## Morgan structural features

Install `python -m pip install -e '.[morgan]'`. The optional RDKit backend uses
Morgan bit fingerprints, matching CHEMSMART's Tanimoto Morgan algorithm without
running pairwise clustering or calling a private grouper method. Explicit
hydrogens are removed consistently. The public RDKit generator API avoids the
legacy method's deprecation warnings; no new fingerprint algorithm is invented.
The plugin's database adapter reads the molecule SMILES associated with each
record's last stored structure; the reference's SMILES is never used as input.

See examples/energy-morgan.yaml; named configurations belong in ~/.chemsmart/ml.

```sh
chemsmart run ml -p energy-morgan prepare
chemsmart run ml -p energy-morgan train
chemsmart run ml -p energy-morgan train --morgan-size 512 --alpha 10 -o /tmp/morgan-512
chemsmart run ml plot --workdir /Users/tt/Desktop/qm9/runs/energy-morgan
```

Train options: `--fingerprint none|morgan`, `--morgan-radius`, `--morgan-size`,
`--morgan-chirality/--no-morgan-chirality`, `--cache`. YAML uses corresponding
underscored names inside train. prepare accepts radius/size/chirality/cache/output
in its own YAML section; keep train and prepare parameters identical to reuse
that cache. Training automatically generates missing fingerprints, so prepare is
optional. Raw database files remain read-only. Prediction reads fingerprint
settings from model.json, requiring the same RDKit version, and reuses the cache.
Existing models without fingerprint settings still work.

The default shared cache is CONFIG_ROOT/ml/cache/morgan.sqlite. Cache identity
includes canonical isomeric SMILES, radius, size, chirality, hydrogen policy,
representation and RDKit version. Duplicate raw SMILES reuse a vector in memory;
canonical equivalents and subsequent runs reuse packed bits on disk. Periodic
commits permit reuse of completed batches after interruption. fingerprints.json
records settings and cache hit/generation counts. No pickle loading is involved.

The example starts with 256 bits to reduce memory/solver cost; the CLI default is
2048 bits. Short fingerprints have more hash collisions and are not guaranteed
to be optimal. Dense regression memory scales with samples times feature count;
full-dataset 2048-bit fitting can require several GB. Ridge with positive alpha
uses block Gram accumulation instead of a tall SVD. This is an approximation-free
ridge formulation but still costs O(samples * features^2) arithmetic; sample
selection and sparse/iterative solvers are not yet implemented. Do not assume
fingerprint preparation and regression have the same cost.

Invalid/missing SMILES stop preparation or training with the record ID, and write
fingerprint-error.json. No silent sample dropping or automatic bond repair is
performed. A real-data spot check found an invalid-valence SMILES in the supplied
xTB database, so full Morgan training on that database is not yet validated and
will require resolving those source structures. Failed output directories are
retained for inspection; choose a new output directory when retrying.

SOAP is not yet connected to this plugin. Morgan produces 2D connectivity
features and does not encode the final geometry's distances or angles.

## Screen an entire paired dataset before Morgan training

```sh
chemsmart run ml -p energy-morgan validate --pairing revqm9-index --features orbitals --fingerprint morgan -o /path/validation
chemsmart run ml -p energy-morgan prepare --manifest /path/validation/valid-samples.json -o /path/prepared-valid
chemsmart run ml -p energy-morgan train --manifest /path/validation/valid-samples.json -o /path/trained-valid
```

Validation continues past invalid SMILES and collects all failures. It writes
validation.json, excluded.csv (one-based idx and zero-based npz_index, record IDs,
SMILES and reasons), and valid-samples.json. Required numeric inputs and target
energies must be finite; paired compositions must match. Unmatched source or
reference indices are listed separately in the pairing report. Parsing checks
are task-specific: use --fingerprint none for numerical-feature-only validation.
This is not a full coordinate, convergence or chemical-correctness audit, and
statistical energy outliers are not automatically removed.

Exclusion is explicit: prepare/train only filter when --manifest is supplied.
Both accept a YAML manifest path in their respective sections. validate accepts
pairing/features/fingerprint/output in its own YAML section. Databases remain
unchanged. The manifest is tied to absolute source/reference paths, sizes and
modification timestamps; rerun validation if inputs change. This lightweight
identity check is not a cryptographic content hash. Training copies the manifest
to its output, checks sample record IDs and preserves grouping based on the full
paired set before filtering. For a fair feature comparison, train numerical and
Morgan baselines with the same manifest rather than compare different cohorts.
Previously failed output directories remain intact; always choose new outputs.

## Automatic plots after training

Training now generates test-set PNG, PDF and error statistics in OUTPUT/plots by
default. Set `train.plot: false` in project YAML or pass `train --no-plot` to opt
out. `train --plot` overrides YAML false. The independent plot command remains
available for older runs or other splits. Automatic plotting uses a headless
backend and does not open a window. Metrics and model artifacts are written
before plotting; if plotting fails, those artifacts remain available.
