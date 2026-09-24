"""
FT-2C: the 2C lift fitted discriminatively, no distillation.
────────────────────────────────────────────────────────────────────────────
The C-way head of f_init is duplicated into a 2C-way head (rows copied, biases
shifted by log pi_s(y), so the marginal readout starts exactly at f_init) and
the whole network is fine-tuned by cross-entropy on k = y + C * 1[x in D_f].
Readout from the 2C logits u:
    log P(y|x)   = lsm(logaddexp(u_r, u_f)),   log P_r(y|x) = lsm(u_r)
    Delta(x)     = log P_r(y|x) - log P(y|x)
The released model IS the target log f_init + eta* . Delta, with Delta and eta*
recomputed after every fine-tuning epoch (eta* = 0 returns f_init exactly).
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

from unlearning import config as CFG
from unlearning.methods import fits
from unlearning.methods import proxy as P
from unlearning.models import fresh

LOG_PI_FLOOR = -30.0      # log pi_s(y) floor on structurally empty cells
_IDENTITY_TOL = 1e-3


def _readout(u, C):
    u_r, u_f = u[:, :C], u[:, C:]
    u_m = fits.logsumexp_pair_t(u_r, u_f)
    return (u_r - torch.logsumexp(u_r, dim=1, keepdim=True),
            u_m - torch.logsumexp(u_m, dim=1, keepdim=True))


def _class_priors(y_all, Nf, C):
    n_y, n_f = fits.bincount_t(y_all, C), fits.bincount_t(y_all[:Nf], C)
    denom = n_y.clamp_min(1.0)
    pi_r, pi_f, empty = (n_y - n_f) / denom, n_f / denom, (n_y == 0)
    return (torch.where(empty, torch.full_like(pi_r, 0.5), pi_r),
            torch.where(empty, torch.full_like(pi_f, 0.5), pi_f))


def _duplicate_head(model, log_pi_r, log_pi_f):
    head = model[-1]
    C, dev = head.out_features, head.weight.device
    new = nn.Linear(head.in_features, 2 * C, bias=(head.bias is not None)).to(dev)
    with torch.no_grad():
        new.weight.copy_(torch.cat([head.weight, head.weight], dim=0))
        if head.bias is not None:
            b = head.bias.detach()
            new.bias.copy_(torch.cat([b + log_pi_r.to(dev), b + log_pi_f.to(dev)], dim=0))
    if hasattr(model, "fc"):
        model.fc = new
    else:
        model[len(model) - 1] = new
    return model


class FT2CModel(nn.Module):
    """forward(x) = log f_init(.|x) + eta . Delta(x) (unnormalised log-probs)."""

    def __init__(self, init_model, model_2c, C):
        super().__init__()
        self.init_model, self.model_2c, self.C, self.eta = init_model, model_2c, int(C), 0.0

    def train(self, mode=True):
        self.model_2c.train(mode)
        self.init_model.eval()
        self.training = mode
        return self

    def forward(self, x):
        lsm = F.log_softmax(self.init_model(x), dim=-1)
        if self.eta == 0.0:
            return lsm
        log_p_r, log_p = _readout(self.model_2c(x), self.C)
        return lsm + self.eta * (log_p_r - log_p)


@torch.no_grad()
def _refresh(state):
    """Delta from the current 2C network over D, then eta* again."""
    data = state["data"]
    log_p_r, log_p = _readout(P.forward_all(state["m2c"], data["X_all"],
                                            state["device"], state["cdev"]), state["C"])
    delta, lsm = (log_p_r - log_p).float(), state["lsm"]

    def h(e):
        return P.lse_mean_t(lsm + float(e) * delta)

    eta = P.find_eta(h)
    state.update(log_p=log_p, eta_star=float(eta), h_eta_star=float(h(eta)),
                 A=float(delta[data["Nf"]:].gather(1, state["y_r"].unsqueeze(1)).mean().item()))
    state["m"].eta = float(eta)


def init(base_weights, forget_loader, retain_loader, X_forget, y_forget,
         device, arch, num_classes, lr):
    init_model = fresh(base_weights, device, arch, num_classes)
    init_model.eval()
    for p in init_model.parameters():
        p.requires_grad_(False)
    C = init_model[-1].out_features
    cdev = P.compute_device(device)
    data = P.materialise(forget_loader, retain_loader, X_forget, y_forget, cdev)
    Nf, N = data["Nf"], data["N"]
    y_all = data["y_all"].to(cdev, dtype=torch.long)
    y_r = data["y_r"].to(cdev, dtype=torch.long)
    lsm = fits.lsm_t(P.forward_all(init_model, data["X_all"], device, cdev))
    pi_r, pi_f = _class_priors(y_all, Nf, C)
    model_2c = _duplicate_head(fresh(base_weights, device, arch, num_classes),
                               torch.log(pi_r).clamp_min(LOG_PI_FLOOR),
                               torch.log(pi_f).clamp_min(LOG_PI_FLOOR))
    s = torch.zeros(N, dtype=torch.long, device=cdev)
    s[:Nf] = 1
    train_ldr = DataLoader(TensorDataset(data["X_all"].cpu(), (y_all + C * s).cpu()),
                           batch_size=CFG.BATCH_SIZE, shuffle=True)
    opt = optim.Adam(model_2c.parameters(), lr=lr)
    sched = optim.lr_scheduler.MultiplicativeLR(opt, lr_lambda=lambda e: CFG.FT_LR_GAMMA)
    state = {"m": FT2CModel(init_model, model_2c, C), "m2c": model_2c, "opt": opt,
             "scheduler": sched, "_epoch": 0, "train_ldr": train_ldr,
             "device": device, "cdev": cdev, "data": data, "lsm": lsm, "C": C,
             "y_r": y_r}
    _refresh(state)
    gap = float((state["log_p"] - lsm).abs().max().item())
    assert gap < _IDENTITY_TOL, f"FT-2C head duplication broke the identity ({gap:.2e})"
    return state


def step(state):
    m2c, opt, dev = state["m2c"], state["opt"], state["device"]
    m2c.train()
    for x, k in state["train_ldr"]:
        x, k = x.to(dev), k.to(dev)
        opt.zero_grad()
        F.cross_entropy(m2c(x), k).backward()
        opt.step()
    state["_epoch"] += 1
    if state["_epoch"] > 1:
        state["scheduler"].step()
    _refresh(state)
