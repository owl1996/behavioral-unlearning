"""
Proxy unlearning: fit -> Delta -> eta* -> KL distillation of the target.
────────────────────────────────────────────────────────────────────────────
For a proxy (p_ideal, p_init) fitted on D = [D_f ; D_r]:

    Delta(x)  = log p_ideal(.|x) - log p_init(.|x)
    target(x) = log_softmax( log f_init(.|x) + eta* . Delta(x) )
    eta*      = root of Z(eta) = E_x logsumexp(log f_init + eta . Delta) on
                (0, TAU_MAX]; TAU_MAX if Z(TAU_MAX) < 0; 0 if Z(eps) > 0.

The released model is f_init distilled towards the target by KL on
D_f and D_r (weights |D_f|/|D| and |D_r|/|D|, one retain batch per forget
batch), or on D_f alone for the forget-only variants. eta* = 0 returns
f_init untouched. Dirac proxies: pointwise Delta, eta* = 1.

On ResNet the proxy is fitted in a feature space of the INITIAL network:
the flattened output of one or more layers, JL-projected to k dims by a fixed
semi-orthogonal Omega (seeded QR) and standardised with the statistics of the
fit rows D; the black-box target reads the evaluation sets with those same
statistics, never with their own.
"""
import time

import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim
from scipy.optimize import brentq
from torch.utils.data import DataLoader, TensorDataset

from unlearning import config as CFG
from unlearning.methods import fits
from unlearning.models import CifarResNet18, fresh

# ResNet taps: backbone[:j] ends right after layer (1, 2, 3, 4).
LAYER_TAP = {1: 5, 2: 6, 3: 7, 4: 8}


def compute_device(device):
    """Where the proxy fit runs: CUDA on device, otherwise CPU (numpy)."""
    return device if device.type == "cuda" else torch.device("cpu")


def torch_path(cdev):
    return cdev.type == "cuda"


# ── eta* search (timed: its cost is reported separately in the trace) ────────
ETA_CLOCK = {"s": 0.0, "n": 0}


def find_eta(h):
    t0 = time.time()
    try:
        eps, tau = CFG.ETA_SEARCH_EPS, CFG.TAU_MAX
        if h(tau) < 0.0:
            return tau
        if h(eps) > 0.0:
            return 0.0
        return float(brentq(h, eps, tau, xtol=1e-4, maxiter=100))
    finally:
        ETA_CLOCK["s"] += time.time() - t0
        ETA_CLOCK["n"] += 1


def lse_mean_t(z):
    return float(torch.logsumexp(z, dim=1).mean().item())


# ── ResNet feature space ─────────────────────────────────────────────────────

def build_jl_omega(D, k, seed):
    """(k, D) with orthonormal rows: QR of a seeded (D, k) Gaussian."""
    G = np.random.default_rng(seed).standard_normal((D, k)).astype(np.float32)
    Q, _ = np.linalg.qr(G)
    return torch.from_numpy(Q.T.copy()).to(torch.float32)


@torch.no_grad()
def resnet_features(base_weights, X_cpu, device, num_classes, layers=(3,),
                    k=CFG.RESNET_PROXY_K, out="numpy", batch_size=1024, stats=None):
    """((N, k) standardised JL projection of the concatenated layer taps,
    (mean, std) it was standardised with). stats=None computes them on X_cpu
    (the fit rows); pass the fit's stats to read any other set in its space."""
    taps = {LAYER_TAP[l] for l in layers}
    end = max(taps)
    model = CifarResNet18(num_classes=num_classes).to(device).eval()
    model.load_state_dict(base_weights)
    bb = model.backbone
    h, D = model._maybe_resize(X_cpu[:1].to(device)), 0
    for j in range(end):
        h = bb[j](h)
        if j + 1 in taps:
            D += int(h.flatten(1).shape[1])
    if k > D:
        raise ValueError(f"projection dim k={k} exceeds the tap dim D={D}")
    Omega = build_jl_omega(D, k, CFG.RESNET_PROXY_SEED).to(device)
    chunks = []
    for i in range(0, X_cpu.shape[0], batch_size):
        h = model._maybe_resize(X_cpu[i:i + batch_size].to(device, non_blocking=True))
        parts = []
        for j in range(end):
            h = bb[j](h)
            if j + 1 in taps:
                parts.append(h.flatten(1))
        z = (parts[0] if len(parts) == 1 else torch.cat(parts, dim=1)) @ Omega.T
        chunks.append(z if out == "torch_device" else z.cpu())
    if out == "torch_device":
        Z = torch.cat(chunks, dim=0).float()
        mu, sd = (stats if stats is not None else
                  (Z.mean(dim=0, keepdim=True), Z.std(dim=0, keepdim=True) + 1e-6))
        return ((Z - mu) / sd).float(), (mu, sd)
    Z = torch.cat(chunks, dim=0).numpy().astype(np.float32)
    mu, sd = (stats if stats is not None else
              (Z.mean(axis=0, keepdims=True), Z.std(axis=0, keepdims=True) + 1e-6))
    return ((Z - mu) / sd).astype(np.float32), (mu, sd)


# ── Shared pipeline ──────────────────────────────────────────────────────────

def materialise(forget_loader, retain_loader, X_forget, y_forget, dev):
    """X_all = [D_f ; D_r] (D_r in the retain loader's shuffled order)."""
    Xr, yr = [], []
    for x, y in retain_loader:
        Xr.append(x.to(dev))
        yr.append(y.to(dev))
    X_r, y_r = torch.cat(Xr), torch.cat(yr)
    X_f, y_f = X_forget.to(dev), y_forget.to(dev)
    return dict(X_f=X_f, y_f=y_f, X_r=X_r, y_r=y_r,
                X_all=torch.cat([X_f, X_r]), y_all=torch.cat([y_f, y_r]),
                Nf=X_f.shape[0], Nr=X_r.shape[0], N=X_f.shape[0] + X_r.shape[0])


@torch.no_grad()
def forward_all(model, X, device, out_device, bs=1024):
    model.eval()
    return torch.cat([model(X[i:i + bs].float().to(device)).to(out_device)
                      for i in range(0, X.shape[0], bs)]).float()


def make_delta_fn(post, featurise):
    """Delta on arbitrary inputs (for the black-box target on the eval sets).
    `featurise` must map them into the fit's space, standardisation included."""
    @torch.no_grad()
    def delta_fn(X):
        a, b = post(featurise(X))
        d = a - b
        if isinstance(d, torch.Tensor):
            return d.float().cpu().numpy()
        return np.asarray(d, dtype=np.float32)
    return delta_fn


def proxy_init(base_weights, forget_loader, retain_loader, X_forget, y_forget,
               device, arch, num_classes, kind, fit=None, layers=(3,),
               k=CFG.RESNET_PROXY_K):
    """kind: "fit" (closed-form proxy, `fit` = (numpy fit, torch fit)),
    "dirac" (mixture Dirac) or "dirac2c". Returns the distillation inputs."""
    model = fresh(base_weights, device, arch, num_classes)
    C = model[-1].out_features
    cdev = compute_device(device)
    data = materialise(forget_loader, retain_loader, X_forget, y_forget, cdev)
    Nf, N = data["Nf"], data["N"]
    y_r = data["y_r"].to(cdev, dtype=torch.long)
    y_f = data["y_f"].to(cdev, dtype=torch.long)
    y_all = data["y_all"].to(cdev, dtype=torch.long)
    lsm = fits.lsm_t(forward_all(model, data["X_all"], device, cdev))

    delta_fn = None
    if kind == "dirac":
        delta = fits.dirac_mixture_delta(Nf, y_f, N, C, cdev)
    elif kind == "dirac2c":
        cnt_r, cnt_f = fits.bincount_t(y_r, C), fits.bincount_t(y_f, C)
        cnt = cnt_r + cnt_f
        log_pi_r = torch.log(cnt_r / (cnt + 1e-30) + 1e-30).to(cdev)
        log_pi_f = torch.log(cnt_f / (cnt + 1e-30) + 1e-30).to(cdev)
        delta = fits.dirac_2c_delta(Nf, y_f, y_all, C, lsm, log_pi_r, log_pi_f, cdev)
    else:
        fit_np, fit_t = fit
        resnet, on_device = arch == "resnet18", torch_path(cdev)
        if resnet:
            out = "torch_device" if on_device else "numpy"
            X_all, stats = resnet_features(base_weights, data["X_all"], device,
                                           num_classes, layers, k, out=out)

            def feat(X):                     # the fit rows' standardisation
                return resnet_features(base_weights, X, device, num_classes,
                                       layers, k, out=out, stats=stats)[0]
        elif on_device:
            def feat(X):
                return X.reshape(X.shape[0], -1).to(cdev, dtype=torch.float32)
            X_all = feat(data["X_all"])
        else:
            def feat(X):
                return X.reshape(X.shape[0], -1).cpu().numpy().astype(np.float32)
            X_all = feat(data["X_all"])
        if on_device:
            with torch.no_grad():
                post = fit_t(X_all[Nf:], y_r, X_all[:Nf], y_f, X_all, y_all, C,
                             int(X_all.shape[1]), cdev)
                lp_ideal, lp_init = post(X_all)
                delta = (lp_ideal - lp_init).float()
        else:
            if resnet:
                X_f, X_r = X_all[:Nf], X_all[Nf:]
            else:
                X_f, X_r = feat(data["X_f"]), feat(data["X_r"])
            post = fit_np(X_r, y_r.cpu().numpy(), X_f, y_f.cpu().numpy(), X_all,
                          y_all.cpu().numpy(), C, X_all.shape[1])
            lp_ideal, lp_init = post(X_all)
            delta = (torch.from_numpy(lp_ideal).to(cdev).float()
                     - torch.from_numpy(lp_init).to(cdev).float()).float()
        delta_fn = make_delta_fn(post, feat)

    A = float(delta[Nf:].gather(1, y_r.unsqueeze(1)).mean().item())
    if kind in ("dirac", "dirac2c"):
        eta = 1.0
    else:
        eta = find_eta(lambda e: lse_mean_t(lsm + float(e) * delta))
    return dict(model=model, data=data, delta_all=delta, lsm=lsm,
                delta_fn=delta_fn, eta_star=eta, A=A)


def grad_init(common, device, lr, forget_only=False):
    model, data = common["model"], common["data"]
    Nf, Nr, N = data["Nf"], data["Nr"], data["N"]
    target = F.log_softmax(common["lsm"] + common["eta_star"] * common["delta_all"],
                           dim=-1).float().cpu()
    forget_ldr = DataLoader(TensorDataset(data["X_f"].cpu(), target[:Nf]),
                            batch_size=CFG.BATCH_SIZE, shuffle=True)
    retain_ldr = DataLoader(TensorDataset(data["X_r"].cpu(), target[Nf:]),
                            batch_size=CFG.BATCH_SIZE, shuffle=True)
    opt = optim.Adam(model.parameters(), lr=lr)
    sched = optim.lr_scheduler.MultiplicativeLR(opt, lr_lambda=lambda e: CFG.LR_GAMMA)
    return {"m": model, "opt": opt, "scheduler": sched, "_epoch": 0,
            "forget_ldr": forget_ldr, "retain_ldr": retain_ldr,
            "pi_f": Nf / N, "pi_r": Nr / N, "_rit": None, "device": device,
            "eta_star": common["eta_star"], "A": common["A"],
            "delta_fn": common["delta_fn"], "forget_only": bool(forget_only),
            # eta* = 0 (admissibility failed): f_init is the answer; skip the
            # loop, since Adam would still move the weights on a zero signal.
            "no_op": float(common["eta_star"] or 0.0) == 0.0}


def _kl(m, x, t):
    return F.kl_div(F.log_softmax(m(x), -1), t, reduction="batchmean", log_target=True)


def grad_step(state):
    """One distillation epoch: one pass over the forget loader."""
    if not state["no_op"]:
        m, opt, dev = state["m"], state["opt"], state["device"]
        m.train()
        if state["forget_only"]:
            for x, t in state["forget_ldr"]:
                m.zero_grad()
                _kl(m, x.to(dev), t.to(dev)).backward()
                opt.step()
        else:
            rit = state["_rit"] or iter(state["retain_ldr"])
            for x, t in state["forget_ldr"]:
                x, t = x.to(dev), t.to(dev)
                try:
                    xr, tr = next(rit)
                except StopIteration:
                    rit = iter(state["retain_ldr"])
                    xr, tr = next(rit)
                m.zero_grad()
                (state["pi_f"] * _kl(m, x, t)
                 + state["pi_r"] * _kl(m, xr.to(dev), tr.to(dev))).backward()
                opt.step()
            state["_rit"] = rit
    state["_epoch"] += 1
    if state["_epoch"] > 1 and not state["no_op"]:
        state["scheduler"].step()


def proxy_method(kind, fit=None, layers=(3,), k=CFG.RESNET_PROXY_K,
                 forget_only=False, resnet_only=False):
    """(init_fn, step_fn) for one proxy configuration."""
    def init_fn(base_weights, forget_loader, retain_loader, X_forget, y_forget,
                device, arch, num_classes, lr):
        if resnet_only and arch != "resnet18":
            raise ValueError("ResNet feature-space variant on a non-ResNet cell")
        common = proxy_init(base_weights, forget_loader, retain_loader, X_forget,
                            y_forget, device, arch, num_classes, kind, fit=fit,
                            layers=layers, k=k)
        return grad_init(common, device, lr, forget_only=forget_only)
    return init_fn, grad_step
