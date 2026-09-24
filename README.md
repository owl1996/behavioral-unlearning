# Unlearning from proxies — raw-data reproduction

Code that regenerates the raw experimental data of the paper: for every
experimental cell, every unlearning method and every seed, the metrics of the
unlearned model **at every unlearning epoch**, plus a per-sample forget audit.
One command downloads what it needs, trains the reference models, runs the
unlearning methods and writes every metric under `results/`.

No figures, no tables: only the raw numbers they are computed from.

---

## 1. Install

Python ≥ 3.9. A CUDA GPU is recommended; CPU and Apple MPS also work.

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

## 2. Run

```bash
# One cell, a few methods, two seeds (downloads CIFAR-10 and DINOv2-small,
# extracts features once, trains the references, unlearns, aggregates):
python -m unlearning.run --cell cifar10_mlp1_CLASS_0 --methods LDA-2C-Grad SCRUB --seeds 42 0

# The same cell with every method and the 5 seeds of the paper:
python -m unlearning.run --cell cifar10_mlp1_CLASS_0

# The full grid of the paper (112 cells, 5 seeds, all methods):
python -m unlearning.run --grid all            # or --grid head / --grid resnet

# What exists:
python -m unlearning.run --list
```

Each run ends with a summary per cell (last unlearning epoch, mean over seeds):

```
cifar10_mlp1_CLASS_0  (CLASS 0 (airplane), 5000 samples)  seeds 42 0 1 2 3  signal set .../5000
  method                   Acc_test Acc_forget  KL_test      delta pass_both
  Retrain                     ...        0.00   ...          ...     1.000
  Base                        ...       ...     ...          ...     0.000
  LDA-2C-Grad                 ...
```

**Smoke test** (about a minute once the features exist; writes to separate
folders, so it can never be mistaken for real results):

```bash
python -m unlearning.run --cell cifar10_linear_CLASS_0 --seeds 42 0 \
    --train-epochs 5 --epochs 2 --methods LDA-2C-Grad Dirac-Dirac-2C-Grad SCRUB \
    --results-dir results_smoke --models-dir models_smoke
```

### Data and pretrained models

Downloaded automatically on first use (or explicitly with
`python -m unlearning.data --dataset cifar10 cifar100`):

| what | where | used by |
|---|---|---|
| CIFAR-10 / CIFAR-100 (torchvision) | `data/raw/` | everything |
| DINOv2-small (`facebook/dinov2-small`, Hugging Face) CLS features | `data/dinov2-small/<dataset>/` | `linear`, `mlp1`, `mlp2` |
| ImageNet ResNet-18 weights (torchvision) | torch hub cache | `resnet18` |

Feature extraction runs once per dataset: a few minutes on a GPU, tens of
minutes on a laptop.

## 3. What can be chosen

**Cell** = `{dataset}_{arch}_{scenario}_{k}`:

| part | values |
|---|---|
| dataset | `cifar10`, `cifar100` |
| arch | `linear`, `mlp1`, `mlp2` (heads on frozen DINOv2-small features), `resnet18` (fine-tuned end to end) |
| scenario, k | `CLASS_k`: forget class k · `SUBCLASS_k`: train on superclasses, forget fine class k · `RANDOM_k`: forget k random training samples |

The paper's grid: CLASS and SUBCLASS with k ∈ {0,2,4,6,8} (CIFAR-10) or
{0,20,40,60,80} (CIFAR-100), RANDOM with k ∈ {1,10,100,1000}; 4 archs; 2
datasets → 112 cells. Any other k works too.

**Methods** (`--methods`, default: all that apply to the cell). `-Grad` = the
proxy target is distilled into the network by KL gradient descent.

| method | family | description |
|---|---|---|
| `LDA-2C-Grad` | 2C proxy | one LDA on the lifted labels (y, s ∈ {retain, forget}) |
| `Dirac-Dirac-2C-Grad` | 2C proxy | closed-form Dirac 2C target (η* = 1) |
| `FT-2C` | 2C proxy | 2C posterior fitted discriminatively (fine-tuned duplicated head); the released model is the target itself, no distillation |
| `LDA-Mixture-Grad`, `QDA-Mixture-Grad` | mixture proxy | separate fits on D_r and D_f, Bayes mixture |
| `Dirac-Dirac-Grad` | mixture proxy | Dirac mixture target (η* = 1) |
| `LDA-Naive-Grad`, `QDA-Naive-Grad` | naive proxy | independent fits on D_r and D |
| `SCRUB` | baseline | Kurmanji et al. (2023), official loop: max-step on D_f during the first 2 epochs, min-step over all of D_r every epoch; T = 4 (KL × T²), α = 0.001, γ = 0.99 |
| `SalUn` | baseline | Fan et al. (2024), Alg. 1: global top-50 % saliency mask of the forget gradient, then CE on D_f (wrong labels) ∪ D_r with the masked gradient |
| `RL+FT`, `GA+FT`, `FT`, `GA` | baselines | random wrong labels + fine-tuning, one ascent epoch + fine-tuning, fine-tuning, gradient ascent |
| `LDA-2C-l{3,4,34}d{512,256,128}-Grad` | ResNet only | LDA-2C fitted on layer3 / layer4 / both, JL-projected to 512/256/128 dims (`l3d256` = `LDA-2C-Grad`) |
| `LDA-2C-l{1,2,23}d{512,256,128}-Grad` | ResNet only | same on shallower layers, **target only** (no distillation, no epoch axis) |
| `Dirac-Dirac-FO-Grad`, `Dirac-Dirac-2C-FO-Grad` | forget-only | the Dirac proxies distilled on D_f only |

**Other options**: `--seeds` (default `42 0 1 2 3`; any integer works),
`--epochs` (unlearning epochs, default 20), `--train-epochs` (reference
training: epochs for heads, default 100; epoch cap for ResNet, default 60),
`--tau-max` (upper end of the η* search, default 1), `--device`,
`--results-dir`, `--models-dir`.

Aggregation reads **every seed on disk** for a cell, not only the ones of the
last command; the seeds used are printed in the summary header and listed in
`epochs.json → split_groups`. `python -m unlearning.aggregate --seeds 42 0 1 2 3`
restricts it to a subset.

## 4. Protocol (fixed; all values in `unlearning/config.py`)

For a cell and a seed s:

1. **References.** `init_s` trained on D and `retrain_s` trained on D_r, both
   from `torch.manual_seed(s)`. Heads: Adam (lr 1e-3, wd 1e-4), cosine, batch
   512, 100 epochs. ResNet-18: ImageNet weights, inputs upsampled to 64×64, Adam
   (lr 1e-3, wd 1e-4), cosine, batch 128, until 99 % training accuracy (cap 60
   epochs).
2. **Split.** CLASS/SUBCLASS: the forget set is defined by the labels (the 5
   seeds share it). RANDOM: drawn with `np.random.default_rng(s)`, so each seed
   is its own problem; each RANDOM seed also trains one control pair (weight
   seed 1000 + s, same split) so that the control rows have two references.
3. **Unlearning.** Every method starts from `init_s` (seeded with s) and runs
   20 epochs; the released model is the **last** epoch (no model selection).
   Heads: lr 1e-3; ResNet: lr 1e-4. Proxies: η* is the root of
   Z(η) = E_x logsumexp(log f_init + η Δ) on (0, τ_max), τ_max = 1.
   Baselines follow the algorithms of their papers' official code (table
   above); optimiser (Adam), learning rate, batch size and the 20 epochs are
   the bench's, identical for every method. Their learning rates are not
   searched: the official recipes search them against the retrained model,
   which this protocol forbids.
4. **Evaluation after every epoch**, against the same seed's references:
   test set (2000-row fixed subsample), full forget set, retain set (2000 rows of
   a fixed 10 000-row subsample of D_r). Runtime excludes these evaluations.

## 5. Outputs

```
results/
├── epochs_grid.csv                 pooled rows of every aggregated cell
└── <cell>/
    ├── epochs.json, epochs.csv     every metric × method × seed × epoch (+ pooled, controls, targets)
    ├── audit.json                  per-sample forget audit of the released models
    ├── audit_forget_split<g>.npz   per-sample KR / KI database behind audit.json
    ├── floor_pairs.json            every retrain-to-retrain / retrain-to-init KL pair
    ├── eval_meta_split<g>.npz      evaluation-row indices, fine labels, split sizes
    ├── manifest_seed<s>.json       settings, versions, hardware, wall-clock of a run
    └── logits/split<g>/            RAW DATA written by the runner
        ├── init_<s>.npz, retrain_<s>.npz   reference logits on test / forget / retain
        ├── trace_<method>_<s>.npz          one row per unlearning epoch
        └── target_<method>_<s>.npz         the proxy's black-box target (before distillation)
```

`<g>` is the split group: `42` for CLASS/SUBCLASS, the seed for RANDOM.
Initial-model checkpoints go to `models/` (needed only to add methods later).

**`trace_<method>_<s>.npz`** (E = epochs, N_f = forget-set size):

| key | shape | content |
|---|---|---|
| `epoch`, `n_steps`, `elapsed_s` | [E] | epoch index, cumulative optimiser steps, cumulative runtime (s) |
| `metric_KL`, `metric_KL_forget`, `metric_KL_retain` | [E] | E_x KL(p_retrain ‖ p_model), nats, on test / forget / retain |
| `metric_acc_{test,forget,retain}` | [E] | top-1 accuracy (%) |
| `metric_sacc_{test,forget}` | [E] | sampled accuracy E_x p_model(y\|x) (%) |
| `scalar_eta_star`, `scalar_A` | [E] | η* and the admissibility quantity A (proxies) |
| `scalar_{init,fit,eta_s_cum,eta_s_mean,distil,RTE_once}_s`, `scalar_eta_calls` | [E] | runtime split: proxy fit, η* search, epoch loop |
| `KR`, `KI` | [E, N_f] | per forget sample: KL(p_retrain ‖ p_model), KL(p_init ‖ p_model) |
| `y_forget`, `idx_forget`, `meta` | | labels, row indices, JSON metadata |

**`epochs.json`**: `rows[method].per_seed["split<g>/seed<s>"][epoch]` holds, per
epoch, the trace metrics above plus the audit readings (`KR_mean`, `KI_mean`,
`delta_mean`, `delta_median`, `delta_of_means`, `pass_frac`, `pass_both_frac`,
`frac_resolved`, and the same restricted to the signal set with suffix
`_signal`) and costs relative to the same seed's retraining time
(`*_frac_retrain`); `rows[method].pooled[epoch]` has `<metric>_mean`, `_std`
and `_wmean` over seeds. `targets[method]` = the black-box targets, `controls`
= the `Retrain` and `Base` rows, `signal` = signal-set sizes, `rte` =
reference training times and hardware stamps.

**Audit definitions.** Per forget sample i and released model M:
KR_i = KL(p_retrain ‖ p_M)(x_i), KI_i = KL(p_init ‖ p_M)(x_i), δ_i = KR_i − KI_i.
`pass` ⇔ δ_i < 0; `pass_both` ⇔ δ_i < 0 and KR_i < KL(p_retrain ‖ p_init)(x_i).
Control rows use only references, off-diagonal (a model is never its own
reference): `Retrain` (released = another seed's retrained model) and `Base`
(released = the initial model). The **signal set** keeps the forget samples
where `Retrain` passes both conditions and `Base` fails, on every pairing; on it
`Retrain` scores 1 and `Base` 0 by construction.

## 6. Compute, parallelism, resuming

Measured for the paper's grid on V100/A100-class GPUs: ≈ 7.5 GPU-hours for
the 84 head cells, ≈ 51 GPU-hours for the 28 ResNet cells; ≈ 8 GB of results.

Every (cell, seed) is independent. For a job array, print one command per
unit, run them anywhere, then aggregate:

```bash
python -m unlearning.data --dataset cifar10 cifar100      # once, before the array
python -m unlearning.run --grid all --tasks > tasks.txt   # 560 independent commands + 1 aggregate
```

Every output is written atomically and skipped when present: re-running a
command resumes a killed run, and adding a method or a seed computes only
that. A results folder never mixes settings: the runner refuses to resume from
files produced with another `--train-epochs` or `--tau-max`.

## 7. Reproducibility

* Every source of randomness is seeded: weights and batch order (`s`), RANDOM
  splits (`s`), evaluation subsamples (seed 0), ResNet JL projections (seed 0).
  Each (cell, seed, method) run is independent of what else runs in the process.
* Same hardware and software → bit-identical results. Across hardware, float
  rounding (cuDNN / BLAS) differs, so numbers agree statistically, not bitwise.
  The paper's data were produced on CUDA GPUs.
* The proxy fits run in torch on CUDA and in numpy on CPU/MPS: same math,
  different rounding.
* The paper's DINOv2 features were extracted on Apple MPS; re-extracting there
  gives the same files bit for bit, on CPU/CUDA to within ~1e-4.
* Linear heads reach the same model across hardware (logits within ~1e-3);
  MLP heads and ResNets need not (logit differences of order 1 between CUDA
  and MPS training were observed on MLP heads).
  The **signal set** is defined by the reference models, so on a different
  machine it can change by a few percent of the forget set, which shifts every
  `*_signal` column of a cell together. The unrestricted columns, accuracies,
  KLs and η* reproduce within seed noise.
* Runtimes are wall-clock; compare them only between files with the same
  `_machine` stamp (`epochs.json → rte.cross_node` lists mismatches).
* `--tau-max 100` removes the η ≤ 1 cap to measure the unconstrained η_max;
  use it with its own `--results-dir`.

## 8. Checklist for a quick (or automated) review

1. `python -m unlearning.run --list` → 34 methods, 112 cells.
2. Smoke test (section 2). Expected in the printed table, by construction:
   `Retrain pass_both = 1.000` and `Base pass_both = 0.000`; `Acc_forget` of
   `Retrain` close to 0 (the forgotten class was never seen); the log shows
   `eta* 1.0000` for the Dirac methods.
3. Re-run the same command: every unit prints `complete`, nothing is recomputed.
4. On a ResNet cell, `trace_LDA-2C-l3d256-Grad_<s>.npz` equals
   `trace_LDA-2C-Grad_<s>.npz` except for timing fields (same proxy, same space).
5. Code map: `unlearning/run.py` (runner) → `methods/` (`proxy.py` pipeline,
   `fits.py` proxy fits, `ft2c.py`, `baselines.py`) → `metrics.py` (per-epoch
   reduction, file formats) → `aggregate.py` (audit, controls, `epochs.json`).
   All settings: `unlearning/config.py`.
