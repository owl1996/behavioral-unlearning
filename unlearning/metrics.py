"""
Metrics and on-disk formats.
────────────────────────────────────────────────────────────────────────────
KL convention (reference first, nats):  KL(p_ref || p_model), averaged over x.

References (initial / retrained models) are cached as full-set logits:
    {init,retrain}_{s}.npz    logits_{test,forget,retain}, y_*, meta

A released model is reduced ON THE FLY at every unlearning epoch, against the
same seed's references, and only numbers are stored:
    trace_{m}_{s}.npz  epoch, elapsed_s, n_steps          [E]
                       metric_{KL,KL_forget,KL_retain,acc_*,sacc_*}  [E]
                       scalar_{eta_star,A,...,fit_s,distil_s,...}     [E]
                       KR, KI   per forget sample                   [E, N_f]
                       y_forget, idx_forget, meta
    KR_i = KL(p_retrain || p_model)(x_i),   KI_i = KL(p_init || p_model)(x_i)
The test and retain sets are read on a fixed TRACE_EVAL_N-row subsample
(indices in eval_meta_split*.npz); the forget set is kept whole.
"""
import hashlib
import json
import os
import platform
import sys
import time

import numpy as np
import torch

from unlearning import config as CFG


# ── Elementary metrics (numpy, float64) ──────────────────────────────────────

def log_softmax(z):
    z = np.asarray(z, dtype=np.float64)
    z = z - z.max(axis=-1, keepdims=True)
    return z - np.log(np.exp(z).sum(axis=-1, keepdims=True))


def kl_rows(p_ref, lp_ref, lp_model):
    """KL(ref || model) per row, from log-probabilities."""
    return (p_ref * (lp_ref - lp_model)).sum(axis=-1)


def argmax_acc(logits, y):
    return float((np.asarray(logits).argmax(-1) == np.asarray(y)).mean() * 100.0)


def sampled_acc(logits, y):
    """E_x p_model(y(x)|x) in %: accuracy of the classifier sampling its output."""
    p = np.exp(log_softmax(logits))
    return float(p[np.arange(len(y)), np.asarray(y).astype(int)].mean() * 100.0)


KL_KEY = {"test": "KL", "forget": "KL_forget", "retain": "KL_retain"}
ACC_KEY = {"test": "acc_test", "forget": "acc_forget", "retain": "acc_retain"}
SACC_KEY = {"test": "sacc_test", "forget": "sacc_forget"}


def paper_metrics(lp_model, y, ref_pairs):
    """Accuracies, sampled accuracies and KL to the reference, per eval set.
    `ref_pairs` = {set: (p_ref, log p_ref)} on the same rows."""
    out = {}
    for name, lp in lp_model.items():
        if name in ACC_KEY:
            out[ACC_KEY[name]] = argmax_acc(lp, y[name])
        if name in SACC_KEY:
            out[SACC_KEY[name]] = sampled_acc(lp, y[name])
        pair = (ref_pairs or {}).get(name)
        if pair is not None and name in KL_KEY:
            out[KL_KEY[name]] = float(kl_rows(pair[0], pair[1], lp).mean())
    return out


def ref_pairs_from(logits_by_set, idx=None):
    """{set: (p, log p)} from full-set reference logits, sliced to `idx`."""
    out = {}
    for name, a in logits_by_set.items():
        i = (idx or {}).get(name)
        lp = log_softmax(a if i is None or len(i) == a.shape[0] else a[i])
        out[name] = (np.exp(lp), lp)
    return out


# ── Logit caches ─────────────────────────────────────────────────────────────

@torch.no_grad()
def model_logits(model, X, device, batch=4096):
    model.eval()
    out = [model(X[i:i + batch].to(device)).float().cpu().numpy()
           for i in range(0, X.shape[0], batch)]
    return np.concatenate(out, 0) if out else np.zeros((0, 0), np.float32)


_MACHINE = None


def machine():
    """Node identity stamped into every file (hostname hashed): runtimes are
    wall-clock, so an RTE ratio is only valid between files of one node."""
    global _MACHINE
    if _MACHINE is None:
        info = {"host": hashlib.sha1(platform.node().encode()).hexdigest()[:10],
                "torch": torch.__version__}
        if torch.cuda.is_available():
            info["gpu"] = torch.cuda.get_device_name(0)
            info["cuda"] = torch.version.cuda
        elif torch.backends.mps.is_available():
            info["gpu"] = "apple-mps"
        _MACHINE = info
    return _MACHINE


def _savez(path, payload):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    tmp = path + ".tmp.npz"
    np.savez_compressed(tmp, **payload)
    os.replace(tmp, path)          # atomic: a killed run never leaves half a file
    return path


def save_logits(path, model, eval_sets, device, meta):
    payload = {}
    for name, (X, y) in eval_sets.items():
        payload[f"logits_{name}"] = model_logits(model, X, device)
        payload[f"y_{name}"] = y.cpu().numpy().astype(np.int16)
    payload["meta"] = np.array(json.dumps({**meta, "_machine": machine()}))
    return _savez(path, payload)


def load_npz(path):
    """{array keys..., "meta": dict}."""
    with np.load(path, allow_pickle=False) as z:
        out = {k: z[k] for k in z.files if k != "meta"}
        out["meta"] = json.loads(str(z["meta"])) if "meta" in z.files else {}
    return out


def read_meta(path):
    with np.load(path, allow_pickle=False) as z:
        return json.loads(str(z["meta"])) if "meta" in z.files else {}


def n_epochs(path):
    """Epoch count of a trace file, 0 if absent or unreadable."""
    try:
        with np.load(path, allow_pickle=False) as z:
            return int(z["epoch"].shape[0])
    except (OSError, ValueError, KeyError):
        return 0


# ── Per-epoch trace ──────────────────────────────────────────────────────────

def trace_indices(n, k, seed=0):
    """Sorted deterministic k-row subsample of n rows (all rows if k <= 0 or >= n)."""
    if k is None or k <= 0 or k >= n:
        return np.arange(n, dtype=np.int32)
    rng = np.random.default_rng(seed)
    return np.sort(rng.choice(n, k, replace=False)).astype(np.int32)


def trace_views(eval_sets):
    """{set: (X, y, idx)}: the audited set whole, the others subsampled."""
    out = {}
    for name, (X, y) in eval_sets.items():
        k = 0 if name == CFG.AUDIT_SET else CFG.TRACE_EVAL_N
        idx = trace_indices(int(X.shape[0]), k, seed=CFG.TRACE_SEED)
        take = torch.from_numpy(idx.astype(np.int64))
        out[name] = (X[take], y[take], idx)
    return out


def _labels(y):
    return y.cpu().numpy() if hasattr(y, "cpu") else np.asarray(y)


class TraceWriter:
    """Reduces one released model after every epoch. The capture runs in
    eval mode under no_grad (no RNG draw) and restores the training mode, so a
    traced run is the untraced run; its time is excluded from the RTE."""

    def __init__(self, views, device, reduce_fn, batch=4096):
        self.views, self.device, self.reduce_fn, self.batch = views, device, reduce_fn, batch
        self.epochs, self.elapsed, self.steps = [], [], []
        self.scalars, self.metrics = {}, {}
        self.kr, self.ki = [], []
        self.eval_s = 0.0

    @staticmethod
    def _push(store, i, name, v):
        col = store.setdefault(name, [])
        col.extend([float("nan")] * (i - len(col)))
        try:
            col.append(float("nan") if v is None else float(v))
        except (TypeError, ValueError):
            col.append(float("nan"))

    def capture(self, model, epoch, elapsed_s, n_steps=None, scalars=None):
        t0 = time.time()
        was_training = model.training
        lp = {k: log_softmax(model_logits(model, X, self.device, self.batch))
              for k, (X, _y, _i) in self.views.items()}
        y = {k: _labels(yy) for k, (_X, yy, _i) in self.views.items()}
        model.train(was_training)
        self.epochs.append(int(epoch))
        self.elapsed.append(float(elapsed_s))
        self.steps.append(-1 if n_steps is None else int(n_steps))
        i = len(self.epochs) - 1
        for k, v in (scalars or {}).items():
            self._push(self.scalars, i, k, v)
        met, kr, ki = self.reduce_fn(lp, y)
        for k, v in met.items():
            self._push(self.metrics, i, k, v)
        self.kr.append(np.asarray(kr, np.float32))
        self.ki.append(np.asarray(ki, np.float32))
        self.eval_s += time.time() - t0

    def save(self, path, meta):
        n = len(self.epochs)
        out = {"epoch": np.asarray(self.epochs, np.int32),
               "elapsed_s": np.asarray(self.elapsed, np.float32),
               "n_steps": np.asarray(self.steps, np.int32)}
        for store, prefix in ((self.scalars, "scalar_"), (self.metrics, "metric_")):
            for k, col in store.items():
                col = list(col) + [float("nan")] * (n - len(col))
                out[f"{prefix}{k}"] = np.asarray(col[:n], np.float32)
        out["KR"] = np.stack(self.kr, 0)
        out["KI"] = np.stack(self.ki, 0)
        _X, y, idx = self.views[CFG.AUDIT_SET]
        out[f"y_{CFG.AUDIT_SET}"] = _labels(y).astype(np.int16)
        out[f"idx_{CFG.AUDIT_SET}"] = np.asarray(idx, np.int32)
        out["meta"] = np.array(json.dumps({**meta, "_machine": machine()}))
        return _savez(path, out)


def reducer(retrain_npz, init_npz, views):
    """(reduce_fn, init_log_probs) against one seed's two references.

    reduce_fn(lp, y) -> (paper metrics vs the retrained model, KR, KI)."""
    R, I = load_npz(retrain_npz), load_npz(init_npz)
    idx = {k: np.asarray(v[2]) for k, v in views.items()}
    ref = ref_pairs_from({k: R[f"logits_{k}"] for k in views if f"logits_{k}" in R}, idx)
    A = CFG.AUDIT_SET
    p_R, lp_R = ref[A]

    def _cut(a, i):
        return a if len(i) == a.shape[0] else a[i]

    lp_I = log_softmax(_cut(I[f"logits_{A}"], idx[A]))
    p_I = np.exp(lp_I)

    def reduce_fn(lp, y):
        return (paper_metrics(lp, y, ref), kl_rows(p_R, lp_R, lp[A]),
                kl_rows(p_I, lp_I, lp[A]))

    init_lp = {k: log_softmax(_cut(I[f"logits_{k}"], idx[k]))
               for k in views if f"logits_{k}" in I}
    return reduce_fn, init_lp


def target_readings(state, views, init_lp, reduce_fn):
    """Evaluate the BLACK-BOX target log_softmax(log p_init + eta* . Delta)
    before any distillation. None when Delta has no closed form off the
    training rows (Dirac proxies)."""
    delta_fn = state.get("delta_fn")
    if delta_fn is None:
        return None
    eta = float(state.get("eta_star") or 0.0)
    lp, y = {}, {}
    for name, (X, yy, _idx) in views.items():
        d = np.asarray(delta_fn(X), np.float64)
        lp[name] = log_softmax(init_lp[name] + eta * d)
        y[name] = _labels(yy)
    met, kr, ki = reduce_fn(lp, y)
    out = {"epoch": np.zeros(1, np.int32)}
    for k, v in met.items():
        out[f"metric_{k}"] = np.asarray([np.nan if v is None else float(v)], np.float32)
    out["KR"] = np.asarray(kr, np.float32)[None, :]
    out["KI"] = np.asarray(ki, np.float32)[None, :]
    A = CFG.AUDIT_SET
    out[f"y_{A}"] = _labels(views[A][1]).astype(np.int16)
    out[f"idx_{A}"] = np.asarray(views[A][2], np.int32)
    out["_eta_star"] = float(eta)
    return out


def save_target(path, payload, meta):
    return _savez(path, {**payload, "meta": np.array(json.dumps(
        {**meta, "role": "target", "distilled": False, "_machine": machine()}))})


# ── Manifests ────────────────────────────────────────────────────────────────

def write_manifest(path, **fields):
    doc = {"argv": sys.argv[1:], "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
           "python": sys.version.split()[0], "platform": platform.platform(),
           "numpy": np.__version__, "machine": machine(),
           "config": CFG.snapshot(), **fields}
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path + ".tmp", "w") as f:
        json.dump(doc, f, indent=1, default=str)
    os.replace(path + ".tmp", path)
    return path
