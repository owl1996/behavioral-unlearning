"""
Closed-form proxy fits.
────────────────────────────────────────────────────────────────────────────
Each fit returns `compute_log_post(X) -> (log p_ideal(.|x), log p_init(.|x))`,
two (n, C) log-posteriors; the unlearning direction is their difference
Delta(x) = log p_ideal(.|x) - log p_init(.|x). Rows of X_all are [D_f ; D_r].

    Naive    one model on D_r (p_ideal) and one on D (p_init)
    Mixture  one model on D_r, one on D_f, p_init(x|y) = pi_r(y) p_r + pi_f(y) p_f
    2C       ONE model on the lifted labels y' = y + C * 1[x in D_f]:
             log p_ideal = lsm(u[:, :C]),  log p_init = lsm(logaddexp(u_r, u_f))
    Dirac    pointwise Delta on the training rows only (no closure)

`*_np` run on CPU (numpy, used on CPU/MPS), `*_t` on the compute device (CUDA).
"""
import numpy as np
import torch
import torch.nn.functional as F

LOG_FLOOR = -100.0


# ── Helpers ──────────────────────────────────────────────────────────────────

def lsm_np(z):
    mx = z.max(axis=1, keepdims=True)
    return z - np.log(np.exp(z - mx).sum(axis=1, keepdims=True)) - mx


def logsumexp_pair_np(a, b):
    mx = np.maximum(a, b)
    return mx + np.log(np.exp(a - mx) + np.exp(b - mx))


def lsm_t(z):
    return F.log_softmax(z, dim=-1)


def logsumexp_pair_t(a, b):
    """Elementwise logaddexp; the log1p form is MPS-safe for very negative inputs."""
    m = torch.maximum(a, b)
    return m + torch.log1p(torch.exp(-(a - b).abs()))


def bincount_t(y, C):
    return torch.bincount(y, minlength=C).to(torch.float32)


def _gauss_quad_np(X, S_inv, M, Q):
    XS = X @ S_inv
    return (XS * X).sum(1, keepdims=True) - 2 * (XS @ M.T) + Q[None, :]


def _gauss_quad_t(X, S_inv, M, Q):
    XS = X @ S_inv
    return (XS * X).sum(1, keepdim=True) - 2 * (XS @ M.t()) + Q.unsqueeze(0)


# ── LDA / QDA on a single population ─────────────────────────────────────────

def _lda_np(X, y, C, d):
    """(M, S_inv, logdet, Q, log_P): class means, pooled covariance (1e-2 ridge)."""
    N = X.shape[0]
    cnt = np.bincount(y, minlength=C).astype(np.float32)
    log_P = np.log(cnt / (cnt.sum() + 1e-30) + 1e-30).astype(np.float32)
    M = np.zeros((C, d), dtype=np.float32)
    for c in range(C):
        idx = y == c
        if idx.any():
            M[c] = X[idx].mean(0)
    D = X - M[y]
    S = (D.T @ D) / max(N - int((cnt > 0).sum()), 1)
    S_reg = S + np.eye(d, dtype=np.float32) * (1e-2 * np.diag(S).mean())
    S_inv = np.linalg.solve(S_reg, np.eye(d, dtype=np.float32))
    logdet = float(np.linalg.slogdet(S_reg)[1])
    return M, S_inv, logdet, (M @ S_inv * M).sum(1), log_P


def _lda_t(X, y, C, d, device):
    N = X.shape[0]
    eye = torch.eye(d, dtype=torch.float32, device=device)
    cnt = bincount_t(y, C)
    log_P = torch.log(cnt / (cnt.sum() + 1e-30) + 1e-30)
    M = torch.zeros((C, d), dtype=torch.float32, device=device)
    M.index_add_(0, y, X)
    M = M / cnt.clamp(min=1.0).unsqueeze(1)
    D = X - M[y]
    S = (D.t() @ D) / float(max(N - int((cnt > 0).sum().item()), 1))
    S_reg = S + eye * (1e-2 * torch.diagonal(S).mean())
    S_inv = torch.linalg.solve(S_reg, eye)
    logdet = float(torch.linalg.slogdet(S_reg).logabsdet.item())
    return M, S_inv, logdet, (M @ S_inv * M).sum(1), log_P


def _qda_np(X, y, C, d):
    """(M, V_reg, log_norm, log_P): per-class diagonal Gaussians."""
    cnt = np.bincount(y, minlength=C).astype(np.float32)
    log_P = np.log(cnt / (cnt.sum() + 1e-30) + 1e-30).astype(np.float32)
    M = np.zeros((C, d), dtype=np.float32)
    V = np.zeros((C, d), dtype=np.float32)
    for c in range(C):
        idx = y == c
        if idx.any():
            Xc = X[idx]
            M[c] = Xc.mean(0)
            V[c] = ((Xc - M[c]) ** 2).mean(0)
    has_var = V.sum(1) > 0
    eps = 0.1 * V[has_var].mean() if has_var.any() else 1e-6
    V_reg = V + max(float(eps), 1e-6)
    return M, V_reg, 0.5 * np.log(V_reg).sum(1), log_P


def _qda_t(X, y, C, d, device):
    cnt = bincount_t(y, C)
    log_P = torch.log(cnt / (cnt.sum() + 1e-30) + 1e-30)
    cnt_safe = cnt.clamp(min=1.0)
    M = torch.zeros((C, d), dtype=torch.float32, device=device)
    M.index_add_(0, y, X)
    M = M / cnt_safe.unsqueeze(1)
    D = X - M[y]
    V = torch.zeros((C, d), dtype=torch.float32, device=device)
    V.index_add_(0, y, D * D)
    V = V / cnt_safe.unsqueeze(1)
    has_var = V.sum(1) > 0
    eps = (0.1 * V[has_var].mean() if bool(has_var.any().item())
           else torch.tensor(1e-6, device=device))
    V_reg = V + torch.clamp(eps, min=1e-6)
    return M, V_reg, 0.5 * torch.log(V_reg).sum(1), log_P


def _diag_mah_np(X, M, V, chunk=4096):
    """sum_d (x - mu_c)^2 / v_c for every (row, class), chunked over rows
    (identical per-row reduction; bounds the (n, C, d) temporary)."""
    return np.concatenate([((X[i:i + chunk, None, :] - M[None, :, :]) ** 2
                            / V[None, :, :]).sum(-1)
                           for i in range(0, X.shape[0], chunk)], axis=0)


def _diag_mah_t(X, invV, M_invV, bias):
    return (X * X) @ invV.t() - 2.0 * (X @ M_invV.t()) + bias.unsqueeze(0)


# ── Naive: two independent models ────────────────────────────────────────────

def lda_naive_np(X_r, y_r, X_f, y_f, X_all, y_all, C, d):
    M_r, Si_r, ld_r, Q_r, lP_r = _lda_np(X_r, y_r, C, d)
    M_a, Si_a, ld_a, Q_a, lP_a = _lda_np(X_all, y_all, C, d)

    def post(X):
        li = -0.5 * ld_r - 0.5 * _gauss_quad_np(X, Si_r, M_r, Q_r) + lP_r[None, :]
        la = -0.5 * ld_a - 0.5 * _gauss_quad_np(X, Si_a, M_a, Q_a) + lP_a[None, :]
        return lsm_np(li).astype(np.float32), lsm_np(la).astype(np.float32)
    return post


def lda_naive_t(X_r, y_r, X_f, y_f, X_all, y_all, C, d, device):
    M_r, Si_r, ld_r, Q_r, lP_r = _lda_t(X_r, y_r, C, d, device)
    M_a, Si_a, ld_a, Q_a, lP_a = _lda_t(X_all, y_all, C, d, device)

    def post(X):
        li = -0.5 * ld_r - 0.5 * _gauss_quad_t(X, Si_r, M_r, Q_r) + lP_r.unsqueeze(0)
        la = -0.5 * ld_a - 0.5 * _gauss_quad_t(X, Si_a, M_a, Q_a) + lP_a.unsqueeze(0)
        return lsm_t(li).float(), lsm_t(la).float()
    return post


def qda_naive_np(X_r, y_r, X_f, y_f, X_all, y_all, C, d):
    M_r, V_r, ln_r, lP_r = _qda_np(X_r, y_r, C, d)
    M_a, V_a, ln_a, lP_a = _qda_np(X_all, y_all, C, d)

    def post(X):
        li = -ln_r[None, :] - 0.5 * _diag_mah_np(X, M_r, V_r) + lP_r[None, :]
        la = -ln_a[None, :] - 0.5 * _diag_mah_np(X, M_a, V_a) + lP_a[None, :]
        return lsm_np(li).astype(np.float32), lsm_np(la).astype(np.float32)
    return post


def qda_naive_t(X_r, y_r, X_f, y_f, X_all, y_all, C, d, device):
    M_r, V_r, ln_r, lP_r = _qda_t(X_r, y_r, C, d, device)
    M_a, V_a, ln_a, lP_a = _qda_t(X_all, y_all, C, d, device)
    iV_r, iV_a = 1.0 / V_r, 1.0 / V_a
    MiV_r, MiV_a = M_r * iV_r, M_a * iV_a
    b_r, b_a = (M_r * MiV_r).sum(1), (M_a * MiV_a).sum(1)

    def post(X):
        li = -ln_r.unsqueeze(0) - 0.5 * _diag_mah_t(X, iV_r, MiV_r, b_r) + lP_r.unsqueeze(0)
        la = -ln_a.unsqueeze(0) - 0.5 * _diag_mah_t(X, iV_a, MiV_a, b_a) + lP_a.unsqueeze(0)
        return lsm_t(li).float(), lsm_t(la).float()
    return post


# ── Mixture: p_init(x|y) = pi_r(y) p_r(x|y) + pi_f(y) p_f(x|y) ───────────────

def _mixture_priors_np(y_r, y_f, C):
    cnt_r = np.bincount(y_r, minlength=C).astype(np.float32)
    cnt_f = np.bincount(y_f, minlength=C).astype(np.float32)
    cnt = cnt_r + cnt_f
    return (cnt_r, cnt_f, cnt_r / (cnt + 1e-30), cnt_f / (cnt + 1e-30),
            np.log(cnt_r / (cnt_r.sum() + 1e-30) + 1e-30),
            np.log(cnt / (cnt.sum() + 1e-30) + 1e-30))


def _mixture_priors_t(y_r, y_f, C):
    cnt_r, cnt_f = bincount_t(y_r, C), bincount_t(y_f, C)
    cnt = cnt_r + cnt_f
    return (cnt_r, cnt_f, cnt_r / (cnt + 1e-30), cnt_f / (cnt + 1e-30),
            torch.log(cnt_r / (cnt_r.sum() + 1e-30) + 1e-30),
            torch.log(cnt / (cnt.sum() + 1e-30) + 1e-30))


def _mix_init_np(a, b, cnt_f):
    """log p_init(x|y) = logaddexp(a, b), without the forget component b where
    pi_f(y) = 0. That component keeps its mean at the origin, and
    log(0 + 1e-30) = -69 does not cancel the logdet gap between S_f and S_r
    (+83 nats on cifar10 CLASS): it has to drop out exactly. The retain
    component `a` stays even at pi_r(y) = 0 — it is p_ideal(x|y), floored as
    p_ideal floors that class, and the two floors cancel in Delta."""
    return np.where((cnt_f > 0)[None, :], logsumexp_pair_np(a, b), a)


def _mix_init_t(a, b, cnt_f):
    return torch.where((cnt_f > 0).unsqueeze(0), logsumexp_pair_t(a, b), a)


def _pooled_np(X, y, C, d, n):
    M = np.zeros((C, d), dtype=np.float32)
    for c in range(C):
        idx = y == c
        if idx.any():
            M[c] = X[idx].mean(0)
    D = X - M[y]
    # N minus the classes PRESENT, as in `_lda_np` (D_f holds one on CLASS)
    return M, (D.T @ D) / max(n - int(np.unique(y).size), 1)


def _inv_np(S, d):
    S_reg = S + np.eye(d, dtype=np.float32) * (1e-2 * np.diag(S).mean())
    return np.linalg.solve(S_reg, np.eye(d, dtype=np.float32)), float(np.linalg.slogdet(S_reg)[1])


def lda_mixture_np(X_r, y_r, X_f, y_f, X_all, y_all, C, d):
    Nr, Nf = X_r.shape[0], X_f.shape[0]
    _, cnt_f, pi_r, pi_f, lP_r, lP_a = _mixture_priors_np(y_r, y_f, C)
    M_r, S_r = _pooled_np(X_r, y_r, C, d, Nr)
    Si_r, ld_r = _inv_np(S_r, d)
    Q_r = (M_r @ Si_r * M_r).sum(1)
    M_f, S_f = _pooled_np(X_f, y_f, C, d, Nf)
    if np.diag(S_f).mean() < 1e-8:          # singular D_f (e.g. |D_f| = 1): blend with S_r
        a = Nf / (Nf + Nr)
        S_f = a * S_f + (1.0 - a) * S_r
    Si_f, ld_f = _inv_np(S_f, d)
    Q_f = (M_f @ Si_f * M_f).sum(1)

    def post(X):
        lp_r = -0.5 * ld_r - 0.5 * _gauss_quad_np(X, Si_r, M_r, Q_r)
        lp_f = -0.5 * ld_f - 0.5 * _gauss_quad_np(X, Si_f, M_f, Q_f)
        lp_i = _mix_init_np(np.log(pi_r + 1e-30)[None, :] + lp_r,
                            np.log(pi_f + 1e-30)[None, :] + lp_f, cnt_f)
        return (lsm_np(lp_r + lP_r[None, :]).astype(np.float32),
                lsm_np(lp_i + lP_a[None, :]).astype(np.float32))
    return post


def lda_mixture_t(X_r, y_r, X_f, y_f, X_all, y_all, C, d, device):
    Nr, Nf = X_r.shape[0], X_f.shape[0]
    eye = torch.eye(d, dtype=torch.float32, device=device)
    cnt_r, cnt_f, pi_r, pi_f, lP_r, lP_a = _mixture_priors_t(y_r, y_f, C)

    def pooled(X, y, cnt, n):
        M = torch.zeros((C, d), dtype=torch.float32, device=device)
        M.index_add_(0, y, X)
        M = M / cnt.clamp(min=1.0).unsqueeze(1)
        D = X - M[y]
        return M, (D.t() @ D) / float(max(n - int((cnt > 0).sum().item()), 1))

    def inv(S):
        S_reg = S + eye * (1e-2 * torch.diagonal(S).mean())
        return torch.linalg.solve(S_reg, eye), float(torch.linalg.slogdet(S_reg).logabsdet.item())

    M_r, S_r = pooled(X_r, y_r, cnt_r, Nr)
    Si_r, ld_r = inv(S_r)
    Q_r = (M_r @ Si_r * M_r).sum(1)
    M_f, S_f = pooled(X_f, y_f, cnt_f, Nf)
    if float(torch.diagonal(S_f).mean().item()) < 1e-8:
        a = Nf / (Nf + Nr)
        S_f = a * S_f + (1.0 - a) * S_r
    Si_f, ld_f = inv(S_f)
    Q_f = (M_f @ Si_f * M_f).sum(1)
    lpi_r, lpi_f = torch.log(pi_r + 1e-30), torch.log(pi_f + 1e-30)

    def post(X):
        lp_r = -0.5 * ld_r - 0.5 * _gauss_quad_t(X, Si_r, M_r, Q_r)
        lp_f = -0.5 * ld_f - 0.5 * _gauss_quad_t(X, Si_f, M_f, Q_f)
        lp_i = _mix_init_t(lpi_r.unsqueeze(0) + lp_r, lpi_f.unsqueeze(0) + lp_f,
                           cnt_f)
        return (lsm_t(lp_r + lP_r.unsqueeze(0)).float(),
                lsm_t(lp_i + lP_a.unsqueeze(0)).float())
    return post


def _diag_stats_np(X, y, C, d):
    M = np.zeros((C, d), dtype=np.float32)
    V = np.zeros((C, d), dtype=np.float32)
    for c in range(C):
        idx = y == c
        if idx.any():
            Xc = X[idx]
            M[c] = Xc.mean(0)
            V[c] = ((Xc - M[c]) ** 2).mean(0)
    return M, V


def qda_mixture_np(X_r, y_r, X_f, y_f, X_all, y_all, C, d):
    cnt_r, cnt_f, pi_r, pi_f, lP_r, lP_a = _mixture_priors_np(y_r, y_f, C)
    M_r, V_r = _diag_stats_np(X_r, y_r, C, d)
    V_r = V_r + max(0.1 * V_r[V_r.sum(1) > 0].mean(), 1e-6)
    ln_r = 0.5 * np.log(V_r).sum(1)
    M_f, V_f = _diag_stats_np(X_f, y_f, C, d)
    if not (V_f.sum(1) > 0).any():           # all-zero V_f: per-class blend with V_r
        a = (cnt_f / (cnt_f + cnt_r + 1e-30))[:, None]
        V_f = a * V_f + (1.0 - a) * V_r
    has = V_f.sum(1) > 0                     # after the blend, as in the torch path
    eps_f = 0.1 * V_f[has].mean() if has.any() else 0.0
    V_f = V_f + max(eps_f, 1e-6)
    ln_f = 0.5 * np.log(V_f).sum(1)

    def post(X):
        lp_r = -ln_r[None, :] - 0.5 * _diag_mah_np(X, M_r, V_r)
        lp_f = -ln_f[None, :] - 0.5 * _diag_mah_np(X, M_f, V_f)
        lp_i = _mix_init_np(np.log(pi_r + 1e-30)[None, :] + lp_r,
                            np.log(pi_f + 1e-30)[None, :] + lp_f, cnt_f)
        return (lsm_np(lp_r + lP_r[None, :]).astype(np.float32),
                lsm_np(lp_i + lP_a[None, :]).astype(np.float32))
    return post


def qda_mixture_t(X_r, y_r, X_f, y_f, X_all, y_all, C, d, device):
    cnt_r, cnt_f, pi_r, pi_f, lP_r, lP_a = _mixture_priors_t(y_r, y_f, C)

    def stats(X, y, cnt):
        cs = cnt.clamp(min=1.0)
        M = torch.zeros((C, d), dtype=torch.float32, device=device)
        M.index_add_(0, y, X)
        M = M / cs.unsqueeze(1)
        D = X - M[y]
        V = torch.zeros((C, d), dtype=torch.float32, device=device)
        V.index_add_(0, y, D * D)
        return M, V / cs.unsqueeze(1)

    M_r, V_r = stats(X_r, y_r, cnt_r)
    has_r = V_r.sum(1) > 0
    eps_r = 0.1 * V_r[has_r].mean() if bool(has_r.any().item()) else torch.tensor(0.0, device=device)
    V_r = V_r + torch.clamp(eps_r, min=1e-6)
    ln_r = 0.5 * torch.log(V_r).sum(1)
    M_f, V_f = stats(X_f, y_f, cnt_f)
    if not bool((V_f.sum(1) > 0).any().item()):
        a = (cnt_f / (cnt_f + cnt_r + 1e-30)).unsqueeze(1)
        V_f = a * V_f + (1.0 - a) * V_r
    has_f = V_f.sum(1) > 0
    eps_f = 0.1 * V_f[has_f].mean() if bool(has_f.any().item()) else torch.tensor(0.0, device=device)
    V_f = V_f + torch.clamp(eps_f, min=1e-6)
    ln_f = 0.5 * torch.log(V_f).sum(1)
    lpi_r, lpi_f = torch.log(pi_r + 1e-30), torch.log(pi_f + 1e-30)
    iV_r, iV_f = 1.0 / V_r, 1.0 / V_f
    MiV_r, MiV_f = M_r * iV_r, M_f * iV_f
    b_r, b_f = (M_r * MiV_r).sum(1), (M_f * MiV_f).sum(1)

    def post(X):
        lp_r = -ln_r.unsqueeze(0) - 0.5 * _diag_mah_t(X, iV_r, MiV_r, b_r)
        lp_f = -ln_f.unsqueeze(0) - 0.5 * _diag_mah_t(X, iV_f, MiV_f, b_f)
        lp_i = _mix_init_t(lpi_r.unsqueeze(0) + lp_r, lpi_f.unsqueeze(0) + lp_f,
                           cnt_f)
        return (lsm_t(lp_r + lP_r.unsqueeze(0)).float(),
                lsm_t(lp_i + lP_a.unsqueeze(0)).float())
    return post


# ── 2C: one model on the lifted labels ───────────────────────────────────────

def _labels_2c_np(y_r, y_f, C):
    Nf = y_f.shape[0]
    y2 = np.empty(Nf + y_r.shape[0], dtype=np.int64)
    y2[:Nf] = y_f + C
    y2[Nf:] = y_r
    cnt = np.bincount(y2, minlength=2 * C).astype(np.float32)
    lP = np.where(cnt > 0, np.log(cnt / (cnt.sum() + 1e-30) + 1e-30),
                  LOG_FLOOR).astype(np.float32)
    return y2, cnt, lP


def _labels_2c_t(y_r, y_f, C, device):
    Nf = y_f.shape[0]
    y2 = torch.empty(Nf + y_r.shape[0], dtype=torch.long, device=device)
    y2[:Nf] = y_f + C
    y2[Nf:] = y_r
    cnt = bincount_t(y2, 2 * C)
    lP = torch.where(cnt > 0, torch.log(cnt / (cnt.sum() + 1e-30) + 1e-30),
                     torch.full_like(cnt, LOG_FLOOR))
    return y2, cnt, lP


def _readout_2c_np(u, C):
    u_r, u_f = u[:, :C], u[:, C:]
    return (lsm_np(u_r).astype(np.float32),
            lsm_np(logsumexp_pair_np(u_r, u_f)).astype(np.float32))


def _readout_2c_t(u, C):
    u_r, u_f = u[:, :C], u[:, C:]
    return lsm_t(u_r).float(), lsm_t(logsumexp_pair_t(u_r, u_f)).float()


def lda_2c_np(X_r, y_r, X_f, y_f, X_all, y_all, C, d):
    N, K = X_all.shape[0], 2 * C
    y2, cnt, lP = _labels_2c_np(y_r, y_f, C)
    M = np.zeros((K, d), dtype=np.float32)
    for k in range(K):
        idx = y2 == k
        if idx.any():
            M[k] = X_all[idx].mean(0)
    D = X_all - M[y2]
    S = (D.T @ D) / max(N - int((cnt > 0).sum()), 1)
    S_reg = S + np.eye(d, dtype=np.float32) * (1e-2 * np.diag(S).mean())
    S_inv = np.linalg.solve(S_reg, np.eye(d, dtype=np.float32))
    logdet = float(np.linalg.slogdet(S_reg)[1])
    Q = (M @ S_inv * M).sum(1)

    def post(X):
        return _readout_2c_np(-0.5 * logdet - 0.5 * _gauss_quad_np(X, S_inv, M, Q)
                              + lP[None, :], C)
    return post


def lda_2c_t(X_r, y_r, X_f, y_f, X_all, y_all, C, d, device):
    N, K = X_all.shape[0], 2 * C
    eye = torch.eye(d, dtype=torch.float32, device=device)
    y2, cnt, lP = _labels_2c_t(y_r, y_f, C, device)
    M = torch.zeros((K, d), dtype=torch.float32, device=device)
    M.index_add_(0, y2, X_all)
    M = M / cnt.clamp(min=1.0).unsqueeze(1)
    D = X_all - M[y2]
    S = (D.t() @ D) / float(max(N - int((cnt > 0).sum().item()), 1))
    S_reg = S + eye * (1e-2 * torch.diagonal(S).mean())
    S_inv = torch.linalg.solve(S_reg, eye)
    logdet = float(torch.linalg.slogdet(S_reg).logabsdet.item())
    Q = (M @ S_inv * M).sum(1)

    def post(X):
        return _readout_2c_t(-0.5 * logdet - 0.5 * _gauss_quad_t(X, S_inv, M, Q)
                             + lP.unsqueeze(0), C)
    return post


# ── Dirac proxies (pointwise Delta on the training rows, eta* = 1) ───────────

def dirac_mixture_delta(Nf, y_f, N, C, device):
    """Delta = 0 on D_r; on D_f, 0 at the forget label and +100 elsewhere."""
    delta = torch.zeros((N, C), dtype=torch.float32, device=device)
    fi = torch.arange(Nf, device=device, dtype=torch.long)
    delta[fi] = -LOG_FLOOR
    delta[fi, y_f] = 0.0
    return delta


def dirac_2c_delta(Nf, y_f, y_all, C, lsm, log_pi_r, log_pi_f, device):
    """P_2C((y, s)|x) = pi_s(y) p_init(y|x); kill (y0(x), f) for x in D_f;
    renormalise each block to its mass pi_s(y0(x)) and sum the two blocks."""
    z_r = log_pi_r.unsqueeze(0) + lsm
    z_f = log_pi_f.unsqueeze(0) + lsm
    fi = torch.arange(Nf, device=device, dtype=torch.long)
    z_f[fi, y_f] = z_f[fi, y_f] + LOG_FLOOR
    log_target = torch.logaddexp(
        log_pi_r[y_all].unsqueeze(-1) + F.log_softmax(z_r, dim=-1),
        log_pi_f[y_all].unsqueeze(-1) + F.log_softmax(z_f, dim=-1))
    return (log_target - lsm).float()
