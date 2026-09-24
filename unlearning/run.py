"""
Run unlearning experiments and write every metric to results/.
────────────────────────────────────────────────────────────────────────────
For each cell and seed s:
  1. references  train init_s on D and retrain_s on D_r (same seed), cache
                 their logits on the test / forget / retain sets
  2. methods     unlearn every method from init_s for --epochs epochs and
                 reduce the model after EVERY epoch against retrain_s / init_s
                 (paper metrics + per-forget-sample KR, KI) -> trace_*.npz;
                 proxies also get their black-box target -> target_*.npz
  3. aggregate   controls, signal set, audit -> epochs.json / audit.json

Every output file is written atomically and skipped if present, so a killed
run resumes where it stopped and adding a method or a seed runs only that.

    python -m unlearning.run --cell cifar10_mlp1_CLASS_0 --methods LDA-2C-Grad SCRUB
    python -m unlearning.run --grid head                 # 84 DINOv2-head cells
    python -m unlearning.run --list                      # cells and methods
"""
import argparse
import os
import sys
import time

import numpy as np
import torch

from unlearning import config as CFG
from unlearning import data as D
from unlearning import metrics as M
from unlearning.methods import REGISTRY, unlearn_lr
from unlearning.methods import proxy
from unlearning.models import default_train_epochs, make_model, train_reference

STATE_SCALARS = ("eta_star", "A", "h_eta_star")
USES_TAU = {m for m in CFG.METHODS if m not in CFG.BASELINES and "Dirac" not in m}


def _ts():
    return time.strftime("%H:%M:%S")


def _log(msg):
    print(f"[{_ts()}] {msg}", flush=True)


# ── Resume predicates ────────────────────────────────────────────────────────

def _incompatible(path, **want):
    """Refuse to resume from files produced with other settings."""
    meta = M.read_meta(path)
    bad = {k: (meta.get(k), v) for k, v in want.items()
           if k in meta and meta[k] != v}
    if bad:
        sys.exit(f"{path} was produced with {bad} (found, requested). "
                 f"Use another --results-dir or delete it.")


def zoo_done(cell, split, role, seed, train_epochs):
    path = (CFG.retrain_path if role == "retrain" else CFG.init_path)(cell, split, seed)
    if not os.path.exists(path) or (role == "init" and not os.path.exists(
            CFG.init_ckpt(cell, split, seed))):
        return False
    _incompatible(path, train_epochs_requested=train_epochs)
    return True


def method_done(cell, split, method, seed, epochs):
    target_only = method in CFG.TARGET_ONLY
    path = (CFG.target_path if target_only else CFG.trace_path)(cell, split, method, seed)
    if not os.path.exists(path) or (not target_only and M.n_epochs(path) < epochs):
        return False
    if method in CFG.RECIPES and M.read_meta(path).get("recipe") != CFG.RECIPES[method]:
        _log(f"{path}: written by another {method} recipe, re-running")
        return False
    if method in USES_TAU:
        _incompatible(path, tau_max=CFG.TAU_MAX)
    return True


# ── 1. References ────────────────────────────────────────────────────────────

def train_one(cell, role, seed, split, device, train_epochs):
    X = cell.X_retain if role == "retrain" else cell.X_train
    y = cell.y_retain if role == "retrain" else cell.y_train
    torch.manual_seed(int(seed))
    np.random.seed(int(seed) % 2 ** 32)
    model = make_model(cell.arch, cell.num_classes).to(device)
    t0 = time.time()
    model, stats = train_reference(cell.arch, model, X, y, train_epochs, device, seed)
    elapsed = time.time() - t0
    if role == "init":
        ck = CFG.init_ckpt(cell.name, split, seed)
        os.makedirs(os.path.dirname(ck), exist_ok=True)
        torch.save({"state_dict": model.state_dict(), "train_time": elapsed}, ck + ".tmp")
        os.replace(ck + ".tmp", ck)
    meta = {"role": role, "train_seed": int(seed), "split_seed": split,
            "cell": cell.name, "train_epochs_requested": train_epochs,
            "train_epochs": stats["epochs_run"], "n_train": int(len(y)),
            "train_time_s": round(elapsed, 2), **stats}
    path = (CFG.retrain_path if role == "retrain" else CFG.init_path)(cell.name, split, seed)
    M.save_logits(path, model, cell.eval_sets(), device, meta)
    z = M.load_npz(path)
    _log(f"{cell.name} split{split} {role:7s} seed {seed:<4d} "
         f"acc_test {M.argmax_acc(z['logits_test'], z['y_test']):6.2f}  "
         f"acc_forget {M.argmax_acc(z['logits_forget'], z['y_forget']):6.2f}  "
         f"({elapsed:.0f}s)")


# ── 2. Methods ───────────────────────────────────────────────────────────────

def _timing(elapsed, init_s, eta_s, eta_n, eta_s_init):
    """Runtime split: proxy fit, eta search (counted once) and epoch loop."""
    mean = eta_s / eta_n if eta_n else 0.0
    return {"init_s": float(init_s), "fit_s": float(max(0.0, init_s - eta_s_init)),
            "eta_s_cum": float(eta_s), "eta_calls": int(eta_n),
            "eta_s_mean": float(mean) if eta_n else None,
            "distil_s": float(max(0.0, elapsed - init_s - max(0.0, eta_s - eta_s_init))),
            "RTE_once_s": float(elapsed - eta_s + mean)}


def _count_steps(state):
    """Real optimiser steps (one 'epoch' can be a single step when D_f is small)."""
    counter = {"n": 0}

    def hook(*_a, **_k):
        counter["n"] += 1

    for v in state.values():
        if isinstance(v, torch.optim.Optimizer):
            v.register_step_post_hook(hook)
    return counter


def run_method(cell, name, base_weights, seed, device, epochs, views, reduce_fn,
               init_lp, refs):
    """Unlearn `name` from `base_weights`; write its trace and/or target."""
    target_only = name in CFG.TARGET_ONLY
    forget_ldr, retain_ldr = cell.loaders()
    init_fn, step_fn = REGISTRY[name]
    torch.manual_seed(int(seed))
    np.random.seed(int(seed) % 2 ** 32)
    proxy.ETA_CLOCK.update(s=0.0, n=0)
    clock = proxy.ETA_CLOCK
    t0 = time.time()
    state = init_fn(base_weights, forget_ldr, retain_ldr, cell.X_forget, cell.y_forget,
                    device, cell.arch, cell.num_classes, unlearn_lr(cell.arch, name))
    init_s, eta_s_init = time.time() - t0, clock["s"]
    meta = {"role": "unlearn", "method": name, "init_seed": int(seed),
            "cell": cell.name, "split_seed": cell.split_seed, "tau_max": CFG.TAU_MAX,
            "trace_eval_n": CFG.TRACE_EVAL_N, "trace_full_sets": [CFG.AUDIT_SET],
            "reduced_in_loop": True, "pairing": "diagonal", **refs}
    if name in CFG.RECIPES:
        meta["recipe"] = CFG.RECIPES[name]
    if target_only:
        rte = init_s
        meta.update(target_only=True, n_steps=0, epochs=0)
    else:
        writer = M.TraceWriter(views, device, reduce_fn)
        counter = _count_steps(state)
        for ep in range(1, epochs + 1):
            step_fn(state)
            rte = time.time() - t0 - writer.eval_s
            writer.capture(state["m"], ep, rte, n_steps=counter["n"],
                           scalars={**{k: state[k] for k in STATE_SCALARS if k in state},
                                    **_timing(rte, init_s, clock["s"], clock["n"], eta_s_init)})
        rte = time.time() - t0 - writer.eval_s
        meta.update(trace_eval_s=round(writer.eval_s, 2), n_steps=counter["n"], epochs=epochs)
    parts = _timing(rte, init_s, clock["s"], clock["n"],
                    clock["s"] if target_only else eta_s_init)
    meta.update(RTE_s=round(rte, 2), eta_star=state.get("eta_star"), A=state.get("A"),
                **{k: (round(v, 4) if isinstance(v, float) else v) for k, v in parts.items()})
    tgt = None
    if name not in CFG.NO_TARGET:
        tgt = M.target_readings(state, views, init_lp, reduce_fn)
    if not target_only:
        writer.save(CFG.trace_path(cell.name, cell.split_seed, name, seed), meta)
        last = {k: v[-1] for k, v in writer.metrics.items()}
    else:
        last = {k[7:]: float(v[0]) for k, v in tgt.items() if k.startswith("metric_")}
    if tgt is not None:
        M.save_target(CFG.target_path(cell.name, cell.split_seed, name, seed), tgt, meta)
    eta = state.get("eta_star")
    _log(f"{cell.name} split{cell.split_seed} {name:22s} seed {seed:<4d} "
         f"acc_test {last['acc_test']:6.2f}  acc_forget {last['acc_forget']:6.2f}  "
         f"KL {last['KL']:.4f}" + (f"  eta* {eta:.4f}" if isinstance(eta, float) else "")
         + ("  [target only]" if target_only else "") + f"  ({rte:.0f}s)")


def write_eval_meta(cell, views):
    path = CFG.eval_meta_path(cell.name, cell.split_seed)
    if os.path.exists(path):
        return
    payload = {"y_test_fine": cell.y_test_fine.cpu().numpy().astype(np.int16),
               "y_forget_fine": cell.y_train_fine[cell.forget_mask].cpu().numpy().astype(np.int16),
               "retain_eval_idx": cell.retain_eval_idx().astype(np.int32),
               "num_classes": np.int32(cell.num_classes),
               "n_forget": np.int32(int(cell.forget_mask.sum())),
               "n_retain": np.int32(int(cell.retain_mask.sum())),
               "scenario": np.array(cell.scenario), "label": np.array(cell.label),
               **{f"idx_{k}": np.asarray(v[2], np.int32) for k, v in views.items()}}
    M._savez(path, payload)


# ── One (cell, seed) unit ────────────────────────────────────────────────────

def run_unit(cell_name, seed, methods, epochs, train_epochs, device):
    methods = CFG.methods_for_cell(cell_name, methods)
    arch = CFG.parse_cell(cell_name)[1]
    train_epochs = train_epochs or default_train_epochs(arch)
    groups = CFG.split_groups(cell_name, [seed])
    zoo = [(sp, role, s) for sp, g in groups.items() for s in CFG.group_seeds(g)
           for role in ("retrain", "init")
           if not zoo_done(cell_name, sp, role, s, train_epochs)]
    stale = {(sp, s) for sp, _role, s in zoo}      # new references -> re-run methods
    todo = [(sp, s, m) for sp, g in groups.items() for s in g["paper"] for m in methods
            if (sp, s) in stale or not method_done(cell_name, sp, m, s, epochs)]
    if not zoo and not todo:
        _log(f"{cell_name} seed {seed}: complete")
        return
    cells = {sp: D.build_cell(cell_name, split_seed=sp) for sp in groups}
    t0 = time.time()
    for sp, role, s in zoo:
        train_one(cells[sp], role, s, sp, device, train_epochs)
    views = {}
    for sp, c in cells.items():
        views[sp] = M.trace_views(c.eval_sets())
        write_eval_meta(c, views[sp])
    for sp, s in dict.fromkeys((sp, s) for sp, s, _ in todo):
        c = cells[sp]
        w = torch.load(CFG.init_ckpt(cell_name, sp, s), map_location=device,
                       weights_only=True)["state_dict"]
        r_path, i_path = CFG.retrain_path(cell_name, sp, s), CFG.init_path(cell_name, sp, s)
        reduce_fn, init_lp = M.reducer(r_path, i_path, views[sp])
        refs = {"ref_retrain": os.path.basename(r_path), "ref_init": os.path.basename(i_path)}
        for m in [m for sp2, s2, m in todo if (sp2, s2) == (sp, s)]:
            run_method(c, m, w, s, device, epochs, views[sp], reduce_fn, init_lp, refs)
    M.write_manifest(os.path.join(CFG.cell_dir(cell_name), f"manifest_seed{seed}.json"),
                     cell=cell_name, seed=seed, device=str(device),
                     split_groups={str(k): v for k, v in groups.items()},
                     methods=methods, unlearn_epochs=epochs, train_epochs=train_epochs,
                     n_references_trained=len(zoo), n_methods_run=len(todo),
                     wall_clock_s=round(time.time() - t0, 1))


# ── CLI ──────────────────────────────────────────────────────────────────────

def _cells(args):
    if args.cell:
        return args.cell
    return {"head": CFG.CELLS_HEAD, "resnet": CFG.CELLS_RESNET, "all": CFG.CELLS}[args.grid]


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--cell", nargs="+", default=None,
                   help="cell(s) {cifar10|cifar100}_{linear|mlp1|mlp2|resnet18}_"
                        "{CLASS|SUBCLASS|RANDOM}_{k}")
    p.add_argument("--grid", choices=["head", "resnet", "all"], default=None,
                   help="every cell of the paper's grid (or one tier of it)")
    p.add_argument("--methods", nargs="+", default=None, metavar="M",
                   help="default: all (see --list)")
    p.add_argument("--seeds", type=int, nargs="+", default=CFG.SEEDS)
    p.add_argument("--epochs", type=int, default=CFG.UNLEARN_EPOCHS,
                   help="unlearning epochs (default %(default)s)")
    p.add_argument("--train-epochs", type=int, default=None,
                   help=f"reference training epochs (heads, default {CFG.EPOCHS}) or "
                        f"epoch cap (ResNet, default {CFG.RESNET_MAX_EPOCHS})")
    p.add_argument("--tau-max", type=float, default=CFG.TAU_MAX,
                   help="upper end of the eta* search (default %(default)s)")
    p.add_argument("--device", default=None, help="cuda | mps | cpu (default: auto)")
    p.add_argument("--results-dir", default=None)
    p.add_argument("--models-dir", default=None)
    p.add_argument("--no-aggregate", action="store_true")
    p.add_argument("--list", action="store_true", help="print cells and methods")
    p.add_argument("--tasks", action="store_true",
                   help="print one independent command per (cell, seed), for job arrays")
    args = p.parse_args()

    if args.list:
        print("methods:\n  " + "\n  ".join(CFG.METHODS))
        print(f"\ncells ({len(CFG.CELLS)}):\n  " + "\n  ".join(CFG.CELLS))
        return
    if not args.cell and not args.grid:
        p.error("give --cell or --grid (or --list)")
    for c in args.cell or []:
        try:
            CFG.parse_cell(c)
        except ValueError as e:
            p.error(str(e))
    unknown = [m for m in (args.methods or []) if m not in REGISTRY]
    if unknown:
        p.error(f"unknown method(s) {unknown}; see --list")
    cells = _cells(args)
    if args.tasks:
        def flag(name):
            v = getattr(args, name)
            if v is None or v == p.get_default(name):
                return ""
            v = " ".join(map(str, v)) if isinstance(v, list) else v
            return f" --{name.replace('_', '-')} {v}"
        extra = "".join(flag(f) for f in ("methods", "epochs", "train_epochs",
                                          "tau_max", "device", "results_dir",
                                          "models_dir"))
        for c in cells:
            for s in args.seeds:
                print(f"python -m unlearning.run --cell {c} --seeds {s} --no-aggregate{extra}")
        print(f"python -m unlearning.aggregate{flag('results_dir')}")
        return

    CFG.set_dirs(args.results_dir, args.models_dir)
    CFG.set_tau_max(args.tau_max)
    device = torch.device(args.device) if args.device else D.default_device()
    for ds in dict.fromkeys(CFG.parse_cell(c)[0] for c in cells):
        archs = {CFG.parse_cell(c)[1] for c in cells if c.startswith(ds + "_")}
        D.prepare(ds, features=bool(archs - {"resnet18"}),
                  resnet="resnet18" in archs, device=device)
    _log(f"device={device}  cells={len(cells)}  seeds={args.seeds}  "
         f"epochs={args.epochs}  tau_max={CFG.TAU_MAX}  -> {CFG.RESULTS_DIR}")
    for c in cells:
        for s in args.seeds:
            run_unit(c, s, args.methods, args.epochs, args.train_epochs, device)
        if not args.no_aggregate:
            from unlearning.aggregate import aggregate_cell
            aggregate_cell(c, verbose=True)
    if not args.no_aggregate:
        from unlearning.aggregate import write_grid
        write_grid()


if __name__ == "__main__":
    main()
