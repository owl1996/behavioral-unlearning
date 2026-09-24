"""
Aggregate the raw traces of a cell into its metric files (numpy, no GPU).
────────────────────────────────────────────────────────────────────────────
Per cell (results/<cell>/):
  epochs.json / epochs.csv   every metric, method, seed and unlearning epoch;
                             pooled over seeds; black-box targets; controls
  audit.json                 per-sample forget audit of the released model
                             (last epoch): population and worst-sample stats
  audit_forget_split*.npz    the per-sample KR / KI database behind it
  floor_pairs.json           every retrain-to-retrain and init-to-retrain KL
and results/epochs_grid.csv (pooled rows of every aggregated cell).

Audit, per forget sample i, released model M:
  KR_i = KL(p_retrain || p_M)(x_i),  KI_i = KL(p_init || p_M)(x_i),
  delta_i = KR_i - KI_i;  pass iff delta_i < 0;  pass_both iff also
  KR_i < KR_i(Base) (M must also be nearer the retrained model than f_init is).
Control rows, built from the references only (off-diagonal pairings, a model
is never its own reference): Retrain (released = another seed's retrain) and
Base (released = f_init). The *signal set* keeps the samples where Retrain
passes both conditions and Base fails, on every pairing; `*_signal` columns
are restricted to it (Retrain = 1 and Base = 0 there by construction).
Method rows use the diagonal pairing (same seed's references).

Every seed found on disk is aggregated (not only config.SEEDS); --seeds
restricts to a subset.

    python -m unlearning.aggregate                       # every cell on disk
    python -m unlearning.aggregate --cell cifar10_mlp1_CLASS_0
    python -m unlearning.aggregate --seeds 42 0 1 2 3    # the paper's seeds only
"""
import argparse
import csv
import json
import os
import warnings

import numpy as np

from unlearning import config as CFG
from unlearning.metrics import load_npz, log_softmax, paper_metrics, ref_pairs_from

PAPER_KEYS = ["KL", "KL_forget", "KL_retain", "acc_test", "acc_forget",
              "acc_retain", "sacc_test", "sacc_forget", "RTE_s", "n_steps"]
PROXY_KEYS = ["eta_star", "A", "h_eta_star"]
COST_KEYS = ["init_s", "fit_s", "distil_s", "eta_s_cum", "eta_s_mean",
             "eta_calls", "RTE_once_s", "eta_frac_retrain", "fit_frac_retrain",
             "RTE_once_frac_retrain"]
AUDIT_KEYS = ["KR_mean", "KI_mean", "delta_mean", "delta_of_means",
              "delta_median", "pass_frac", "pass_both_frac", "KR_mean_signal",
              "KI_mean_signal", "delta_mean_signal", "delta_of_means_signal",
              "pass_frac_signal", "pass_both_frac_signal", "frac_resolved"]
METRIC_KEYS = PAPER_KEYS + PROXY_KEYS + COST_KEYS + AUDIT_KEYS


def _r(x, sig=6):
    if x is None:
        return None
    x = float(x)
    return float(f"%.{sig}g" % x) if np.isfinite(x) else None


def _node(z):
    m = (z.get("meta") or {}).get("_machine")
    return f"{m.get('host')}|{m.get('gpu')}" if m else None


def _eval_meta(cell, split):
    p = CFG.eval_meta_path(cell, split)
    if not os.path.exists(p):
        return None
    with np.load(p, allow_pickle=False) as z:
        return {k: z[k] for k in z.files}


# ── Loading ──────────────────────────────────────────────────────────────────

def stale_recipe(method, meta):
    """A SCRUB / SalUn trace written under another algorithm (config.RECIPES):
    another method's numbers, never pooled with the current recipe's."""
    return method in CFG.RECIPES and (meta or {}).get("recipe") != CFG.RECIPES[method]


def _trace_last(cell, split, method, seed):
    p = CFG.trace_path(cell, split, method, seed)
    if not os.path.exists(p):
        return None
    with np.load(p, allow_pickle=False) as z:
        if "KR" not in z.files:
            return None
        if method in CFG.RECIPES and stale_recipe(
                method, json.loads(str(z["meta"])) if "meta" in z.files else {}):
            return None
        return {"KR": np.asarray(z["KR"][-1], np.float32),
                "KI": np.asarray(z["KI"][-1], np.float32)}


def load_group(cell, split, grp, methods):
    seeds = CFG.group_seeds(grp)
    inits = {s: load_npz(CFG.init_path(cell, split, s)) for s in seeds
             if os.path.exists(CFG.init_path(cell, split, s))}
    refs = {t: load_npz(CFG.retrain_path(cell, split, t)) for t in seeds
            if os.path.exists(CFG.retrain_path(cell, split, t))}
    unl = {}
    for m in methods:
        got = {s: d for s in grp["paper"] if s in inits
               and (d := _trace_last(cell, split, m, s)) is not None}
        if got:
            unl[m] = got
    return inits, refs, unl


# ── Per-sample database and control rows ─────────────────────────────────────

def _kl_rows(p_ref, lp_ref, lp):
    return (p_ref * (lp_ref - lp)).sum(axis=-1)


def build_arrays(inits, refs, unl, eval_set, paper_seeds):
    """Axes (reference, released model, sample); methods on the diagonal only."""
    seeds = CFG.sort_seeds(inits, paper_seeds)
    ref_seeds = CFG.sort_seeds(refs, paper_seeds)
    methods = [m for m in CFG.METHODS if m in unl]
    S, K, M = len(seeds), len(ref_seeds), len(methods)
    lp_I = {s: log_softmax(inits[s][f"logits_{eval_set}"]) for s in seeds}
    lp_R = {t: log_softmax(refs[t][f"logits_{eval_set}"]) for t in ref_seeds}
    p_I = {s: np.exp(v) for s, v in lp_I.items()}
    p_R = {t: np.exp(v) for t, v in lp_R.items()}
    N = next(iter(lp_I.values())).shape[0]
    out = {"KR": np.zeros((M, K, S, N), np.float32), "KI": np.zeros((M, S, N), np.float32),
           "KR_base": np.zeros((K, S, N), np.float32), "KI_base": np.zeros((S, S, N), np.float32),
           "KR_retrain": np.zeros((K, K, N), np.float32), "KI_retrain": np.zeros((S, K, N), np.float32)}
    for mi, m in enumerate(methods):
        out["KR"][mi] = np.nan
        out["KI"][mi] = np.nan
        for si, s in enumerate(seeds):
            d = unl[m].get(s)
            if d is None or s not in ref_seeds:
                continue
            out["KR"][mi, ref_seeds.index(s), si] = d["KR"]
            out["KI"][mi, si] = d["KI"]
    for si, s in enumerate(seeds):
        for ti, t in enumerate(ref_seeds):
            out["KR_base"][ti, si] = _kl_rows(p_R[t], lp_R[t], lp_I[s])
        for sj, s2 in enumerate(seeds):
            out["KI_base"][sj, si] = _kl_rows(p_I[s2], lp_I[s2], lp_I[s])
    for ti, t in enumerate(ref_seeds):
        for tj, t2 in enumerate(ref_seeds):
            out["KR_retrain"][tj, ti] = _kl_rows(p_R[t2], lp_R[t2], lp_R[t])
        for si, s in enumerate(seeds):
            out["KI_retrain"][si, ti] = _kl_rows(p_I[s], lp_I[s], lp_R[t])
    out["methods"] = np.array(methods)
    out["seeds"] = np.array(seeds, np.int32)
    out["ref_seeds"] = np.array(ref_seeds, np.int32)
    out["paper_seeds"] = np.array([s for s in seeds if s in paper_seeds], np.int32)
    return out


def row_pairs(db, key):
    """(KR, KI, labels), each [P, N], over the pairings the row admits."""
    seeds, refs = list(db["seeds"]), list(db["ref_seeds"])
    kr, ki, lab = [], [], []
    if key == "Retrain":
        for ti, t in enumerate(refs):
            si = seeds.index(t) if t in seeds else None
            for tj, t2 in enumerate(refs):
                if tj != ti and si is not None:
                    kr.append(db["KR_retrain"][tj, ti])
                    ki.append(db["KI_retrain"][si, ti])
                    lab.append(f"ref={t2}->R{t}|init={t}")
    elif key == "Base":
        for si, s in enumerate(seeds):
            ti = refs.index(s) if s in refs else None
            for sj, s2 in enumerate(seeds):
                if sj != si and ti is not None:
                    kr.append(db["KR_base"][ti, si])
                    ki.append(db["KI_base"][sj, si])
                    lab.append(f"ref={s}->I{s}|init={s2}")
    else:
        methods = list(db["methods"])
        if key not in methods:
            return None, None, None
        mi, released = methods.index(key), set(int(v) for v in db["paper_seeds"])
        for si, s in enumerate(seeds):
            for ti, t in enumerate(refs):
                if int(s) in released and t == s:
                    kr.append(db["KR"][mi, ti, si])
                    ki.append(db["KI"][mi, si])
                    lab.append(f"ref={t}|seed={s}")
    if not kr:
        return None, None, None
    return np.asarray(kr, np.float64), np.asarray(ki, np.float64), lab


def per_sample_floor(db):
    """Median over retrain pairs of KL(R_t' || R_t)(x_i): the resolution floor."""
    A = np.asarray(db["KR_retrain"], np.float64)
    K = A.shape[0]
    return np.median(np.asarray([A[i, j] for i in range(K) for j in range(K) if i != j]), axis=0)


def effective_floor(db):
    fl = per_sample_floor(db)
    return np.maximum(fl, float(fl.mean()))


def per_sample_base(db):
    """Mean over pairs of KL(R || f_init)(x_i): how much there is to unlearn."""
    return np.asarray(db["KR_base"], np.float64).mean(axis=(0, 1))


def signal_mask(db):
    KR, KI, _ = row_pairs(db, "Retrain")
    if KR is None:
        return None
    keep = ((KR < KI) & (KR < per_sample_base(db)[None, :])).all(axis=0)
    bKR, bKI, _ = row_pairs(db, "Base")
    if bKR is not None:
        keep &= (bKR >= bKI).all(axis=0)
    return keep


# ── Audit summary (audit.json) ───────────────────────────────────────────────

def _finite_stats(name, a):
    a = np.asarray(a, np.float64)
    ok = np.isfinite(a)
    n_inf = int((~ok).sum())
    if not ok.any():
        return {f"{name}_mean": None, f"{name}_std": None, f"{name}_min": None,
                f"{name}_max": None, f"{name}_n_nonfinite": n_inf, f"{name}_n": 0}
    v = a[ok]
    return {f"{name}_mean": float(v.mean()), f"{name}_std": float(v.std()),
            f"{name}_min": float(v.min()), f"{name}_max": float(v.max()),
            f"{name}_n_nonfinite": n_inf, f"{name}_n": int(v.size)}


def audit_row(KR, KI, labels, y=None, top=5, floor=None, kr_base=None):
    """Population audit and worst-sample (hard) audit for one row, [P, N]."""
    with np.errstate(divide="ignore", invalid="ignore"):
        rho = KR / KI
    finite = np.isfinite(rho)
    delta = KR - KI
    ok = KR < KI
    P = KR.shape[0]
    improves = (KR < kr_base) if kr_base is not None else np.zeros_like(ok)
    both = ok & improves
    resolved = (KR > floor) if floor is not None else np.ones_like(ok, dtype=bool)

    def q(a, p):
        a = a[np.isfinite(a)]
        return float(np.percentile(a, p)) if a.size else float("nan")

    def rowmax(a, mask=None):
        out = np.empty(P)
        for p in range(P):
            v = a[p] if mask is None else a[p][mask[p]]
            v = v[np.isfinite(v)] if mask is None else v
            m = v.max() if v.size else np.nan
            out[p] = np.nan if m == -np.inf else m
        return out

    pp_max = np.array([np.nanmax(rho[p]) if np.any(np.isfinite(rho[p])) else np.inf
                       for p in range(P)])
    pp_max_g = rowmax(np.where(np.isfinite(rho), rho, -np.inf), resolved)
    order = np.argsort(np.where(np.isfinite(rho[0]), rho[0], np.inf))[::-1]
    worst = [{"pairing": labels[0] if labels else None, "index": int(i),
              "rho": float(rho[0, i]), "KR": float(KR[0, i]), "KI": float(KI[0, i]),
              **({"y": int(y[i])} if y is not None else {})} for i in order[:top]]
    log_rho = np.log(rho[finite & (rho > 0)]) if finite.any() else np.array([])
    pp_pass = ok.mean(axis=1)
    return {
        "n_pairings": int(P), "n_samples": int(KR.shape[1]),
        "pairings": labels[:8] + (["..."] if len(labels) > 8 else []),
        "KR_mean": float(KR.mean()), "KR_std": float(KR.std()),
        "KI_mean": float(KI.mean()), "KI_std": float(KI.std()),
        "ratio_of_means": float(KR.mean() / KI.mean()) if KI.mean() else float("inf"),
        "aggregate_pass": bool(KR.mean() < KI.mean()),
        "rho_mean": float(np.nanmean(np.where(finite, rho, np.nan))) if finite.any() else float("inf"),
        "rho_std": float(np.nanstd(np.where(finite, rho, np.nan))) if finite.any() else float("nan"),
        "rho_median": q(rho, 50), "rho_q1": q(rho, 25), "rho_q3": q(rho, 75),
        "rho_p95": q(rho, 95), "rho_p99": q(rho, 99),
        "delta_mean": float(delta.mean()), "delta_std": float(delta.std()),
        "delta_median": q(delta, 50), "delta_q1": q(delta, 25),
        "delta_q3": q(delta, 75), "delta_p95": q(delta, 95),
        "delta_of_means": float(KR.mean() - KI.mean()),
        "geo_mean_rho": float(np.exp(log_rho.mean())) if log_rho.size else float("nan"),
        "log_rho_std": float(log_rho.std()) if log_rho.size else float("nan"),
        "pass_frac": float(ok.mean()),
        "pass_frac_per_pairing_mean": float(pp_pass.mean()),
        "pass_frac_per_pairing_std": float(pp_pass.std()),
        "improve_frac": float(improves.mean()),
        "pass_both_frac": float(both.mean()),
        "aggregate_improves": bool(KR.mean() < kr_base.mean()) if kr_base is not None else False,
        "delta_max_mean": float(np.mean(rowmax(delta))),
        "delta_max_std": float(np.std(rowmax(delta))),
        "delta_max_max": float(np.max(rowmax(delta))),
        **_finite_stats("rho_max", pp_max),
        "rho_max_guarded_mean": float(np.nanmean(pp_max_g)),
        "rho_max_guarded_std": float(np.nanstd(pp_max_g)),
        "frac_resolved": float(resolved.mean()),
        "KR_max_mean": float(KR.max(axis=1).mean()), "KR_max_std": float(KR.max(axis=1).std()),
        "KR_p95": q(KR, 95), "KR_p99": q(KR, 99),
        "hard_pass_frac": float(ok.all(axis=1).mean()),
        "hard_pass_both_frac": float(both.all(axis=1).mean()),
        "worst_samples": worst,
        "n_KI_zero": int((KI == 0).sum()),
        "n_both_zero": int(((KI == 0) & (KR == 0)).sum()),
        "n_nonfinite_rho": int((~finite).sum()),
    }


def group_rows(db, mask=None):
    floor, kr_base = effective_floor(db), per_sample_base(db)
    if mask is not None:
        floor, kr_base = floor[mask], kr_base[mask]
    out = {}
    for key in CFG.CONTROLS + list(db["methods"]):
        KR, KI, lab = row_pairs(db, key)
        if KR is None:
            continue
        if mask is not None:
            KR, KI = KR[:, mask], KI[:, mask]
        P = KR.shape[0]
        out[key] = (KR, KI, np.tile(floor, (P, 1)), np.tile(kr_base, (P, 1)), lab)
    return out


def _pool_axis(arrs):
    if len({a.shape[1] for a in arrs}) == 1:
        return 0
    if len({a.shape[0] for a in arrs}) == 1:
        return 1
    return -1


def _stack(arrs):
    """Pool the groups along the pairing axis, or along the sample axis once
    the signal mask left each group a different number of samples."""
    ax = _pool_axis(arrs)
    if ax >= 0:
        return np.concatenate(arrs, axis=ax)
    return np.concatenate([a.reshape(1, -1) for a in arrs], axis=1)


def summarise(groups, restrict=False):
    """One audit per row, pooled over the split groups `[(split, db, y)]`."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)   # empty guarded sets -> NaN
        return _summarise(groups, restrict)


def _summarise(groups, restrict):
    acc, y0, floors, n_tot, n_sig = {}, {}, [], 0, 0
    for split, db, y in groups:
        mask = signal_mask(db) if restrict else None
        n_tot += len(y)
        n_sig += int(mask.sum()) if mask is not None else len(y)
        if mask is not None and not mask.any():
            continue
        floors.append(effective_floor(db)[mask] if mask is not None else effective_floor(db))
        for key, (KR, KI, fl, kb, lab) in group_rows(db, mask).items():
            a = acc.setdefault(key, ([], [], [], [], []))
            for lst, v in zip(a, (KR, KI, fl, kb, [f"split{split}|{l}" for l in lab])):
                lst.append(v)
            y0.setdefault(key, []).append(y[mask] if mask is not None else y)
    fl = np.concatenate(floors) if floors else np.zeros(0)
    rows = {"_floor": {"mean": float(fl.mean()) if fl.size else None,
                       "median": float(np.median(fl)) if fl.size else None,
                       "p95": float(np.percentile(fl, 95)) if fl.size else None,
                       "n_groups": len(groups)},
            "_signal": {"restricted": bool(restrict), "n_total": n_tot, "n_signal": n_sig,
                        "frac_signal": (n_sig / n_tot) if n_tot else None}}
    for key, (KRs, KIs, fls, kbs, labs) in acc.items():
        if _pool_axis(KRs) == 1:
            lab, yy = list(labs[0]), np.concatenate(y0[key])
        else:
            lab, yy = [l for g in labs for l in g], y0[key][0]
        rows[key] = audit_row(_stack(KRs), _stack(KIs), lab, y=yy,
                              floor=_stack(fls), kr_base=_stack(kbs))
    rows["_headroom"] = headroom(rows)
    return rows


def headroom(rows):
    """Controls ordered (Retrain beats Base)? `degenerate` if not; `narrow` if
    ordered but close to the resolution floor."""
    ref, base = rows.get("Retrain"), rows.get("Base")
    fl = rows.get("_floor", {}).get("mean")
    if ref is None or base is None:
        return {"available": False}
    ordered = ref["pass_both_frac"] > base["pass_both_frac"]
    over = (base["KR_mean"] / fl) if fl else None
    return {"available": True, "base_KR_over_floor": over,
            "base_frac_resolved": base["frac_resolved"],
            "retrain_pass_both": ref["pass_both_frac"],
            "base_pass_both": base["pass_both_frac"],
            "controls_ordered": bool(ordered), "degenerate": bool(not ordered),
            "narrow": bool(ordered and ((base["frac_resolved"] < 0.5)
                                        or (over is not None and over < 10)))}


# ── Per-epoch rows (epochs.json) ─────────────────────────────────────────────

def _audit_stats(r, KR, KI, floor, kr_base, suffix=""):
    d = KR - KI
    base = np.broadcast_to(kr_base[None, :], KR.shape)
    fl = np.broadcast_to(floor[None, :], KR.shape)
    r[f"KR_mean{suffix}"] = float(np.nanmean(KR))
    r[f"KI_mean{suffix}"] = float(np.nanmean(KI))
    r[f"delta_mean{suffix}"] = float(np.nanmean(d))
    r[f"delta_median{suffix}"] = float(np.nanmedian(d))
    r[f"delta_of_means{suffix}"] = float(np.nanmean(KR) - np.nanmean(KI))
    r[f"pass_frac{suffix}"] = float(np.nanmean(d < 0))
    r[f"pass_both_frac{suffix}"] = float(np.nanmean((d < 0) & (KR < base)))
    r[f"frac_resolved{suffix}"] = float(np.nanmean(KI > fl))


def _audit_both(r, KR, KI, mask, floor, kr_base):
    _audit_stats(r, KR, KI, floor, kr_base)
    if mask is not None and mask.any():
        _audit_stats(r, KR[:, mask], KI[:, mask], floor[mask], kr_base[mask], "_signal")


def epoch_rows(tr, mask, floor, kr_base):
    E = int(tr["epoch"].shape[0])
    met = {k[7:]: v for k, v in tr.items() if k.startswith("metric_")}
    sc = {k[7:]: v for k, v in tr.items() if k.startswith("scalar_")}
    rows = []
    for e in range(E):
        r = {"epoch": int(tr["epoch"][e]), "RTE_s": float(tr["elapsed_s"][e]),
             "n_steps": None if int(tr["n_steps"][e]) < 0 else int(tr["n_steps"][e])}
        for k in PAPER_KEYS[:-2]:
            v = met.get(k)
            r[k] = None if v is None or e >= len(v) or not np.isfinite(v[e]) else float(v[e])
        for k in PROXY_KEYS + COST_KEYS:
            v = sc.get(k)
            r[k] = None if v is None or e >= len(v) or not np.isfinite(v[e]) else float(v[e])
        if r.get("eta_calls") is not None:
            r["eta_calls"] = int(r["eta_calls"])
        if r.get("RTE_once_s") is None:
            r["RTE_once_s"] = r["RTE_s"]
        _audit_both(r, np.asarray(tr["KR"][e], np.float64)[None, :],
                    np.asarray(tr["KI"][e], np.float64)[None, :], mask, floor, kr_base)
        rows.append(r)
    return rows


def target_row(cell, split, method, seed, mask, floor, kr_base):
    """The black-box target (before any distillation), epoch 0."""
    p = CFG.target_path(cell, split, method, seed)
    if not os.path.exists(p):
        return None
    with np.load(p, allow_pickle=False) as z:
        d = {k: z[k] for k in z.files if k != "meta"}
    r = {"epoch": 0, "eta_star": float(d["_eta_star"])}
    for k in d:
        if k.startswith("metric_"):
            v = float(d[k][0])
            r[k[7:]] = v if np.isfinite(v) else None
    _audit_both(r, np.asarray(d["KR"], np.float64), np.asarray(d["KI"], np.float64),
                mask, floor, kr_base)
    return r


def pool(per_seed, weights):
    """Per epoch: `_mean` / `_std` over models, `_wmean` weighted by the
    group's signal-set size (differs only on RANDOM restricted columns)."""
    if not per_seed:
        return []
    E = min(len(v) for v in per_seed.values())
    tags = list(per_seed)
    w = np.asarray([float(weights.get(t, 0.0)) for t in tags], float)
    out = []
    for e in range(E):
        row = {"epoch": int(per_seed[tags[0]][e]["epoch"])}
        for k in METRIC_KEYS:
            raw = [per_seed[t][e].get(k) for t in tags]
            keep = [i for i, x in enumerate(raw) if x is not None and np.isfinite(float(x))]
            vals = np.asarray([float(raw[i]) for i in keep], float)
            row[f"{k}_mean"] = _r(vals.mean()) if vals.size else None
            row[f"{k}_std"] = _r(vals.std()) if vals.size else None
            ww = w[keep]
            row[f"{k}_wmean"] = (_r(float((vals * ww).sum() / ww.sum()))
                                 if vals.size and ww.sum() > 0 else row[f"{k}_mean"])
        row["n"] = len(tags)
        out.append(row)
    return out


def control_row(db, key, mask):
    KR, KI, _ = row_pairs(db, key)
    if KR is None:
        return None
    r = {}
    _audit_both(r, KR, KI, mask, effective_floor(db), per_sample_base(db))
    return {k: _r(v) for k, v in r.items()}


def _slice(z, name, idx):
    a = z[f"logits_{name}"]
    return a if idx is None or len(idx) == a.shape[0] else a[idx]


def control_paper_row(inits, refs, key, idx):
    """Paper metrics of a control row on the trace rows. Base reads the
    diagonal KL(R_s || I_s); Retrain the off-diagonal KL(R_t || R_s), t != s
    (so Retrain carries the seed-to-seed floor, and may exceed Base)."""
    seeds, ref_seeds = CFG.sort_seeds(list(inits)), CFG.sort_seeds(list(refs))
    released, order = (refs, ref_seeds) if key == "Retrain" else (inits, seeds)
    acc = {}
    for s in order:
        z = released[s]
        sets = [k for k in ("test", "forget", "retain") if f"logits_{k}" in z]
        lp = {k: log_softmax(_slice(z, k, (idx or {}).get(k))) for k in sets}
        y = {k: z[f"y_{k}"] if (idx or {}).get(k) is None else z[f"y_{k}"][idx[k]]
             for k in sets}
        others = [t for t in ref_seeds if not (key == "Retrain" and t == s)]
        if key == "Base" and s in ref_seeds:
            others = [s]
        per_ref = [paper_metrics(lp, y, ref_pairs_from({k: refs[t][f"logits_{k}"]
                                                        for k in sets}, idx))
                   for t in others]
        r = paper_metrics(lp, y, {})
        for k in ("KL", "KL_forget", "KL_retain"):
            vals = [m[k] for m in per_ref if m.get(k) is not None]
            r[k] = float(np.mean(vals)) if vals else None
        for k, v in r.items():
            acc.setdefault(k, []).append(v)
    out = {}
    for k, v in acc.items():
        v = [x for x in v if x is not None and np.isfinite(x)]
        out[k] = _r(np.mean(v)) if v else None
    return out


def _pool_controls(rows):
    out = {}
    for k in sorted({k for r in rows for k in r}):
        vals = [r[k] for r in rows if r.get(k) is not None]
        out[k] = _r(np.mean(vals)) if vals else None
    out["n_groups"] = len(rows)
    return out


def _floor_pairs(inits, refs, idx, split):
    """Every KL(R_t || R_s), t != s, and KL(R_s || I_s) on test and forget."""
    sets = ("test", "forget")

    def kl(ref, model):
        lp = {k: log_softmax(_slice(model, k, (idx or {}).get(k))) for k in sets}
        y = {k: model[f"y_{k}"] if (idx or {}).get(k) is None else model[f"y_{k}"][idx[k]]
             for k in sets}
        m = paper_metrics(lp, y, ref_pairs_from({k: ref[f"logits_{k}"] for k in sets}, idx))
        return {"test": float(m["KL"]), "forget": float(m["KL_forget"])}

    rs = CFG.sort_seeds(list(refs))
    pairs = [{"split": int(split), "ref": int(t), "model": int(s), **kl(refs[t], refs[s])}
             for s in rs for t in rs if t != s]
    diag = [{"split": int(split), "seed": int(s), **kl(refs[s], inits[s])}
            for s in CFG.sort_seeds(list(inits)) if s in refs]
    return pairs, diag


# ── One cell ─────────────────────────────────────────────────────────────────

def aggregate_cell(cell, verbose=False, seeds=None):
    """Write epochs.json/csv, audit.json, audit npz and floor_pairs.json.
    `seeds`: the paper seeds to read (default: every seed on disk).
    Returns the epochs report, or None if the references are incomplete."""
    methods = CFG.methods_for_cell(cell)
    groups = CFG.split_groups(cell, CFG.seeds_on_disk(cell) if seeds is None else seeds)
    per_method, per_target, controls, signal, floors = {m: {} for m in methods}, {}, {}, {}, []
    rte = {"retrain_s": {}, "init_s": {}, "node": {}, "cross_node": []}
    built, fl_pairs, fl_diag, found = [], [], [], {}
    label = n_classes = first = None
    n_ep = n_traces = n_missing = 0
    stale = []
    for split, grp in groups.items():
        inits, refs, unl = load_group(cell, split, grp, methods)
        found[split] = (CFG.sort_seeds(list(inits)), CFG.sort_seeds(list(refs)))
        if len(inits) < 2 or len(refs) < 2:
            continue
        meta = _eval_meta(cell, split)
        label, n_classes = str(meta["label"]), int(meta["num_classes"])
        first = first or (label, n_classes)
        idx = {k: np.asarray(meta[f"idx_{k}"]) for k in ("test", "forget", "retain")
               if f"idx_{k}" in meta} or None
        for tag_role, src in (("retrain", refs), ("init", inits)):
            for s, z in src.items():
                tag = f"split{split}/seed{s}"
                v = (z.get("meta") or {}).get("train_time_s")
                if v is not None:
                    rte[f"{tag_role}_s"][tag] = _r(v)
                rte["node"][f"{tag_role}:{tag}"] = _node(z)

        # per-sample database (audit.json) and the controls / signal set
        y = next(iter(refs.values()))[f"y_{CFG.AUDIT_SET}"]
        for z in list(inits.values()) + list(refs.values()):
            if not np.array_equal(z[f"y_{CFG.AUDIT_SET}"], y):
                raise SystemExit(f"{cell} split{split}: references disagree on the forget rows")
        db = build_arrays(inits, refs, unl, CFG.AUDIT_SET, grp["paper"])
        db["y"] = y
        np.savez_compressed(CFG.audit_npz(cell, split) + ".tmp.npz", **db)
        os.replace(CFG.audit_npz(cell, split) + ".tmp.npz", CFG.audit_npz(cell, split))
        built.append((split, db, y))
        mask = signal_mask(db)
        floor, kr_base = effective_floor(db), per_sample_base(db)
        floors.append(floor)
        signal[str(split)] = {"n_total": int(mask.size), "n_signal": int(mask.sum()),
                              "frac_signal": _r(float(mask.mean()))}
        p, d = _floor_pairs(inits, refs, idx, split)
        fl_pairs += p
        fl_diag += d
        for key in CFG.CONTROLS:
            cr = control_row(db, key, mask)
            if cr is None:
                continue
            cr.update(control_paper_row(inits, refs, key, idx))
            vals = [v for k, v in rte["retrain_s" if key == "Retrain" else "init_s"].items()
                    if k.startswith(f"split{split}/") and v is not None]
            cr["RTE_s"] = _r(np.mean(vals)) if vals else None
            controls.setdefault(key, []).append(cr)

        for m in methods:
            for s in CFG.sort_seeds(list(grp["paper"])):
                tag = f"split{split}/seed{s}"
                path = CFG.trace_path(cell, split, m, s)
                tr = (load_npz(path) if m not in CFG.TARGET_ONLY and os.path.exists(path)
                      else None)
                if tr is not None and stale_recipe(m, tr.get("meta")):
                    stale.append(f"{m}:{tag}")
                    tr = None
                if m not in CFG.TARGET_ONLY and (tr is None or "KR" not in tr):
                    n_missing += 1
                    continue
                tgt = target_row(cell, split, m, s, mask, floor, kr_base)
                if tgt is not None:
                    per_target.setdefault(m, {})[tag] = tgt
                if m in CFG.TARGET_ONLY:
                    n_missing += int(tgt is None)
                    continue
                rows = epoch_rows(tr, mask, floor, kr_base)
                ref_s = rte["retrain_s"].get(tag)
                for r in rows:
                    for k, num in (("eta_frac_retrain", r.get("eta_s_mean")),
                                   ("fit_frac_retrain", r.get("fit_s")),
                                   ("RTE_once_frac_retrain", r.get("RTE_once_s"))):
                        r[k] = None if not ref_s or num is None else float(num) / float(ref_s)
                node = _node(tr)
                rte["node"][f"{m}:{tag}"] = node
                if node != rte["node"].get(f"retrain:{tag}"):
                    rte["cross_node"].append({"method": m, "tag": tag, "method_node": node,
                                              "retrain_node": rte["node"].get(f"retrain:{tag}")})
                per_method[m][tag] = rows
                n_ep, n_traces = max(n_ep, len(rows)), n_traces + 1
    if stale:
        print(f"[aggregate] {cell}: {len(stale)} trace(s) written by another recipe "
              f"of the same method left out (re-run them): {', '.join(stale)}")
    if not built:
        if verbose:
            got = "; ".join(f"split{sp}: init {i or '-'}, retrain {r or '-'}"
                            for sp, (i, r) in found.items()) or "no reference on disk"
            print(f"[aggregate] {cell}: no split group has >= 2 initial and >= 2 "
                  f"retrained models ({got}), skipped")
        return None

    mean = lambda d: _r(np.mean(list(d.values()))) if d else None
    all_fl = np.concatenate(floors)
    report = {
        "cell": cell, "label": label, "num_classes": n_classes, "mode": "paired",
        "epochs": n_ep, "selection": "every epoch is reported; the RELEASED model is the last",
        "trace": {"eval_n": CFG.TRACE_EVAL_N, "full_sets": [CFG.AUDIT_SET]},
        "split_groups": {str(k): v for k, v in groups.items()},
        "signal": signal,
        "rte": {**rte, "retrain_mean_s": mean(rte["retrain_s"]),
                "init_mean_s": mean(rte["init_s"])},
        "floor": {"median": _r(np.median(all_fl)), "mean": _r(np.mean(all_fl))},
        "controls": {k: _pool_controls(v) for k, v in controls.items()},
        "n_traces": n_traces, "n_missing": n_missing, "rows": {},
    }
    weights = {tag: (signal.get(tag.split("/")[0][5:]) or {}).get("n_signal", 0)
               for per_seed in per_method.values() for tag in per_seed}
    for m, per_seed in per_target.items():
        report.setdefault("targets", {})[m] = _pool_controls(list(per_seed.values()))
    for m, per_seed in per_method.items():
        if per_seed:
            report["rows"][m] = {
                "distils": m in CFG.DISTILLING,
                "per_seed": {k: [{kk: vv if kk in ("epoch", "n_steps") else _r(vv)
                                  for kk, vv in row.items()} for row in v]
                             for k, v in per_seed.items()},
                "pooled": pool(per_seed, weights)}
    _write_json(CFG.epochs_json(cell), report, indent=1)
    write_csv(os.path.join(CFG.cell_dir(cell), "epochs.csv"), report)

    audit = {"cell": cell, "label": first[0], "mode": "paired", "num_classes": first[1],
             "split_groups": {str(sp): groups[sp] for sp, *_ in built},
             "eval_sets": {CFG.AUDIT_SET: summarise(built)},
             "eval_sets_signal": {CFG.AUDIT_SET: summarise(built, restrict=True)}}
    _write_json(CFG.audit_json(cell), audit, indent=2)

    ctl = report["controls"]
    for key, rows, name in (("Retrain", fl_pairs, "model"), ("Base", fl_diag, "seed")):
        for k, col in (("test", "KL"), ("forget", "KL_forget")):
            got = float(np.mean([np.mean([r[k] for r in rows if r[name] == s and r["split"] == sp])
                                 for sp, s in dict.fromkeys((r["split"], r[name]) for r in rows)]))
            assert abs(got - ctl[key][col]) <= 1e-5 * abs(ctl[key][col]) + 1e-12, (cell, key, k)
    _write_json(CFG.floor_pairs_json(cell), {"cell": cell, "retrain_pairs": fl_pairs,
                                             "base_diag": fl_diag})
    if verbose:
        print_summary(report)
    return report


def _write_json(path, obj, indent=1):
    with open(path + ".tmp", "w") as f:
        json.dump(obj, f, indent=indent, default=str)
    os.replace(path + ".tmp", path)


def write_csv(path, report):
    """One row per (method, split, seed, epoch); controls at epoch 0."""
    cols = ["cell", "method", "split", "seed", "epoch"] + \
           [k for k in METRIC_KEYS if k != "n_steps"] + ["n_steps"]
    with open(path + ".tmp", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols, extrasaction="ignore")
        w.writeheader()
        for m, blk in report["rows"].items():
            for tag, rows in blk["per_seed"].items():
                split, seed = tag.split("/")
                for r in rows:
                    w.writerow({"cell": report["cell"], "method": m,
                                "split": split[5:], "seed": seed[4:], **r})
        for key, blk in report["controls"].items():
            w.writerow({"cell": report["cell"], "method": key, "split": "*",
                        "seed": "*", "epoch": 0, **blk})
    os.replace(path + ".tmp", path)


def write_grid_csv(path, reports):
    """One row per (cell, method, epoch), pooled over seeds."""
    cols = ["cell", "label", "method", "epoch", "n"] + \
           [f"{k}_{s}" for k in METRIC_KEYS for s in ("mean", "std")]
    with open(path + ".tmp", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols, extrasaction="ignore")
        w.writeheader()
        for rep in reports:
            for m, blk in rep["rows"].items():
                for r in blk["pooled"]:
                    w.writerow({"cell": rep["cell"], "label": rep.get("label"), "method": m, **r})
            for key, blk in rep["controls"].items():
                w.writerow({"cell": rep["cell"], "label": rep.get("label"), "method": key,
                            "epoch": 0, "n": blk.get("n_groups"),
                            **{f"{k}_mean": v for k, v in blk.items()}})
    os.replace(path + ".tmp", path)


def _fmt(v, spec):
    return "--" if v is None else format(v, spec)


def print_summary(report):
    """Last epoch of each method, pooled over seeds, on the signal set."""
    ctl = report["controls"]
    n_sig = sum(v["n_signal"] for v in report["signal"].values())
    n_tot = sum(v["n_total"] for v in report["signal"].values())
    seeds = CFG.sort_seeds({s for g in report["split_groups"].values() for s in g["paper"]})
    print(f"\n{report['cell']}  ({report['label']})  seeds {' '.join(map(str, seeds))}  "
          f"signal set {n_sig}/{n_tot}")
    print(f"  {'method':24s} {'Acc_test':>8s} {'Acc_forget':>10s} {'KL_test':>8s} "
          f"{'delta':>10s} {'pass_both':>9s}")
    for key in CFG.CONTROLS:
        c = ctl.get(key, {})
        print(f"  {key:24s} {_fmt(c.get('acc_test'), '8.2f')} "
              f"{_fmt(c.get('acc_forget'), '10.2f')} {_fmt(c.get('KL'), '8.4f')} "
              f"{_fmt(c.get('delta_of_means_signal'), '10.4g')} "
              f"{_fmt(c.get('pass_both_frac_signal'), '9.3f')}")
    for m, blk in report["rows"].items():
        r = blk["pooled"][-1]
        print(f"  {m:24s} {_fmt(r.get('acc_test_mean'), '8.2f')} "
              f"{_fmt(r.get('acc_forget_mean'), '10.2f')} {_fmt(r.get('KL_mean'), '8.4f')} "
              f"{_fmt(r.get('delta_of_means_signal_mean'), '10.4g')} "
              f"{_fmt(r.get('pass_both_frac_signal_mean'), '9.3f')}")


def cells_on_disk():
    """Every cell with raw data, grid order first, then any other k."""
    if not os.path.isdir(CFG.RESULTS_DIR):
        return []
    got = {c for c in os.listdir(CFG.RESULTS_DIR) if CFG.CELL_RE.match(c)
           and os.path.isdir(os.path.join(CFG.cell_dir(c), "logits"))}
    return [c for c in CFG.CELLS if c in got] + sorted(got - set(CFG.CELLS))


def write_grid():
    """results/epochs_grid.csv from every epochs.json on disk."""
    reports = []
    for c in cells_on_disk():
        if os.path.exists(CFG.epochs_json(c)):
            with open(CFG.epochs_json(c)) as f:
                reports.append(json.load(f))
    if reports:
        write_grid_csv(os.path.join(CFG.RESULTS_DIR, "epochs_grid.csv"), reports)
        print(f"\n[aggregate] {len(reports)} cell(s) -> "
              f"{os.path.join(CFG.RESULTS_DIR, 'epochs_grid.csv')}")


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--cell", nargs="+", default=None, help="default: every cell on disk")
    p.add_argument("--seeds", type=int, nargs="+", default=None,
                   help="paper seeds to aggregate (default: every seed on disk)")
    p.add_argument("--results-dir", default=None)
    p.add_argument("--quiet", action="store_true")
    args = p.parse_args()
    CFG.set_dirs(args.results_dir)
    for c in args.cell or cells_on_disk():
        aggregate_cell(c, verbose=not args.quiet, seeds=args.seeds)
    write_grid()


if __name__ == "__main__":
    main()
