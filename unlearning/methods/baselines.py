"""
Baselines: FT, GA, GA+FT, RL+FT, SCRUB, SalUn — as (init_fn, step_fn) pairs.
────────────────────────────────────────────────────────────────────────────
    FT      fine-tune on D_r (constant LR)
    GA      gradient ascent on D_f
    GA+FT   one GA epoch at init, then FT on D_r
    RL+FT   (1-a) CE(random wrong labels on D_f) + a CE(D_r), one retain batch
            per forget batch
    SCRUB   official loop (Kurmanji et al., 2023): during the first
            SCRUB_MSTEPS epochs a max-step pass over D_f (ascend
            KL(teacher||student)), then every epoch a min-step pass over ALL of
            D_r (gamma CE + alpha KL); KL at temperature T scaled by T^2, one
            optimiser, no clipping
    SalUn   Fan et al. (2024), Alg. 1: saliency mask = the top SALUN_SPARSITY
            fraction of |grad CE(D_f)| at theta_o, ranked over ALL parameters
            (eval mode); then CE on D_f' U D_r (D_f' = D_f with wrong labels),
            gradient masked to the salient weights
The optimiser, learning rates, batch size and epoch count are the bench's
(Adam, 20 epochs) for every method; only the algorithms follow the papers.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import ConcatDataset, DataLoader, TensorDataset

from unlearning import config as CFG
from unlearning.models import fresh


def _decay(opt, gamma):
    return optim.lr_scheduler.MultiplicativeLR(opt, lr_lambda=lambda e: gamma)


def _train_ep(m, loader, opt, device):
    m.train()
    crit = nn.CrossEntropyLoss()
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        opt.zero_grad()
        crit(m(x), y).backward()
        opt.step()


def _ascent_ep(m, loader, opt, device):
    m.train()
    crit = nn.CrossEntropyLoss()
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        opt.zero_grad()
        (-crit(m(x), y)).backward()
        opt.step()


def _next(state, key="_rit"):
    """Next retain batch, cycling the retain loader."""
    try:
        return next(state[key])
    except StopIteration:
        state[key] = iter(state["retain_loader"])
        return next(state[key])


def _tick(state, sched="scheduler"):
    state["_epoch"] += 1
    if state["_epoch"] > 1:
        state[sched].step()


# ── FT / GA / GA+FT ──────────────────────────────────────────────────────────

def init_FT(bw, forget_loader, retain_loader, X_forget, y_forget, device, arch,
            num_classes, lr):
    m = fresh(bw, device, arch, num_classes)
    opt = optim.Adam(m.parameters(), lr=lr)
    return {"m": m, "opt": opt, "scheduler": _decay(opt, CFG.FT_LR_GAMMA),
            "_epoch": 0, "retain_loader": retain_loader, "device": device}


def step_FT(state):
    _train_ep(state["m"], state["retain_loader"], state["opt"], state["device"])
    _tick(state)


def init_GA(bw, forget_loader, retain_loader, X_forget, y_forget, device, arch,
            num_classes, lr):
    m = fresh(bw, device, arch, num_classes)
    opt = optim.Adam(m.parameters(), lr=lr)
    return {"m": m, "opt": opt, "scheduler": _decay(opt, CFG.LR_GAMMA),
            "_epoch": 0, "forget_loader": forget_loader, "device": device}


def step_GA(state):
    _ascent_ep(state["m"], state["forget_loader"], state["opt"], state["device"])
    _tick(state)


def init_GA_FT(bw, forget_loader, retain_loader, X_forget, y_forget, device,
               arch, num_classes, lr):
    m = fresh(bw, device, arch, num_classes)
    opt = optim.Adam(m.parameters(), lr=lr)
    _ascent_ep(m, forget_loader, opt, device)
    return {"m": m, "opt": opt, "scheduler": _decay(opt, CFG.LR_GAMMA),
            "_epoch": 0, "retain_loader": retain_loader, "device": device}


# ── RL+FT ────────────────────────────────────────────────────────────────────

def _wrong_labels(y, num_classes):
    """One uniformly drawn label != y per row."""
    return (y + torch.randint(1, num_classes, y.shape)) % num_classes


def _random_wrong_labels(X_forget, y_forget, num_classes):
    return DataLoader(TensorDataset(X_forget, _wrong_labels(y_forget, num_classes)),
                      batch_size=CFG.BATCH_SIZE, shuffle=True)


def init_RL_FT(bw, forget_loader, retain_loader, X_forget, y_forget, device,
               arch, num_classes, lr):
    m = fresh(bw, device, arch, num_classes)
    opt = optim.Adam(m.parameters(), lr=lr)
    rl = _random_wrong_labels(X_forget, y_forget, num_classes)
    return {"m": m, "opt": opt, "scheduler": _decay(opt, CFG.LR_GAMMA),
            "_epoch": 0, "rl_loader": rl, "retain_loader": retain_loader,
            "device": device}


def step_RL_FT(state):
    m, opt, dev, a = state["m"], state["opt"], state["device"], CFG.RL_FT_ALPHA
    crit = nn.CrossEntropyLoss()
    m.train()
    state["_it"] = iter(state["retain_loader"])      # fresh retain pass each epoch
    for xf, yf in state["rl_loader"]:
        xf, yf = xf.to(dev), yf.to(dev)
        xr, yr = _next(state, "_it")
        xr, yr = xr.to(dev), yr.to(dev)
        loss = (1 - a) * crit(m(xf), yf) + a * crit(m(xr), yr)
        opt.zero_grad()
        loss.backward()
        opt.step()
    _tick(state)


# ── SCRUB ────────────────────────────────────────────────────────────────────

def _distill_kl(logit_s, logit_t, T):
    """KL(teacher || student) at temperature T, times T^2, per example: the
    official SCRUB's DistillKL (Hinton et al.)."""
    return (F.kl_div(F.log_softmax(logit_s / T, dim=-1), F.softmax(logit_t / T, dim=-1),
                     reduction="sum") * (T * T) / logit_s.shape[0])


def init_SCRUB(bw, forget_loader, retain_loader, X_forget, y_forget, device,
               arch, num_classes, lr):
    teacher = fresh(bw, device, arch, num_classes)
    teacher.eval()
    student = fresh(bw, device, arch, num_classes)
    opt = optim.Adam(student.parameters(), lr=lr)
    return {"m": student, "teacher": teacher, "opt": opt,
            "scheduler": _decay(opt, CFG.LR_GAMMA), "_epoch": 0,
            "forget_loader": forget_loader, "retain_loader": retain_loader,
            "device": device}


def _scrub_pass(state, loader, maximize):
    s, t, opt, dev, T = (state["m"], state["teacher"], state["opt"],
                         state["device"], CFG.SCRUB_T)
    crit = nn.CrossEntropyLoss()
    s.train()
    for x, y in loader:
        x, y = x.to(dev), y.to(dev)
        logit_s = s(x)
        with torch.no_grad():
            logit_t = t(x)
        div = _distill_kl(logit_s, logit_t, T)
        loss = -div if maximize else CFG.SCRUB_GAMMA * crit(logit_s, y) + CFG.SCRUB_ALPHA * div
        opt.zero_grad()
        loss.backward()
        opt.step()


def step_SCRUB(state):
    """One epoch: max-step on D_f (first SCRUB_MSTEPS epochs only), then a
    min-step pass over the whole of D_r."""
    if state["_epoch"] < CFG.SCRUB_MSTEPS:
        _scrub_pass(state, state["forget_loader"], maximize=True)
    _scrub_pass(state, state["retain_loader"], maximize=False)
    _tick(state)


# ── SalUn ────────────────────────────────────────────────────────────────────

def _salun_mask(m, forget_loader, device, sparsity):
    """|grad of the forget loss| at theta_o, summed over the batches of D_f in
    eval mode, ranked over EVERY parameter at once; the top `sparsity`
    fraction is salient (1), the rest is frozen (0). Official generate_mask.py."""
    crit = nn.CrossEntropyLoss()
    m.eval()
    m.zero_grad()
    for xf, yf in forget_loader:
        (-crit(m(xf.to(device)), yf.to(device))).backward()
    params = list(m.parameters())
    g = torch.cat([(p.grad if p.grad is not None else torch.zeros_like(p)).abs().flatten()
                   for p in params])
    flat = torch.zeros_like(g)
    flat[torch.topk(g, int(g.numel() * sparsity)).indices] = 1.0
    m.zero_grad()
    return [c.view_as(p) for c, p in zip(torch.split(flat, [p.numel() for p in params]),
                                          params)]


def init_SalUn(bw, forget_loader, retain_loader, X_forget, y_forget, device,
               arch, num_classes, lr):
    m = fresh(bw, device, arch, num_classes)
    mask = _salun_mask(m, forget_loader, device, CFG.SALUN_SPARSITY)
    relabelled = TensorDataset(X_forget, _wrong_labels(y_forget, num_classes))
    loader = DataLoader(ConcatDataset([relabelled, retain_loader.dataset]),
                        batch_size=retain_loader.batch_size, shuffle=True)
    opt = optim.Adam(m.parameters(), lr=lr)
    return {"m": m, "opt": opt, "scheduler": _decay(opt, CFG.SALUN_LR_GAMMA),
            "_epoch": 0, "mask": mask, "loader": loader, "device": device}


def step_SalUn(state):
    """One pass over D_f' U D_r. A masked coordinate gets a zero gradient, so
    Adam (no weight decay) never moves it: it stays at theta_o."""
    m, opt, dev = state["m"], state["opt"], state["device"]
    crit = nn.CrossEntropyLoss()
    m.train()
    for x, y in state["loader"]:
        x, y = x.to(dev), y.to(dev)
        opt.zero_grad()
        crit(m(x), y).backward()
        for p, mk in zip(m.parameters(), state["mask"]):
            if p.grad is not None:
                p.grad.mul_(mk)
        opt.step()
    _tick(state)


STEPS = {"FT": (init_FT, step_FT), "GA": (init_GA, step_GA),
         "GA+FT": (init_GA_FT, step_FT), "RL+FT": (init_RL_FT, step_RL_FT),
         "SCRUB": (init_SCRUB, step_SCRUB), "SalUn": (init_SalUn, step_SalUn)}
