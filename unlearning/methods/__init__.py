"""
Method registry: name -> (init_fn, step_fn).

    state = init_fn(base_weights, forget_loader, retain_loader, X_forget,
                    y_forget, device, arch, num_classes, lr)
    for _ in range(epochs): step_fn(state)      # state["m"] is the model
"""
from unlearning import config as CFG
from unlearning.methods import baselines, fits, ft2c
from unlearning.methods.proxy import proxy_method

LDA_2C = (fits.lda_2c_np, fits.lda_2c_t)

REGISTRY = {
    "LDA-2C-Grad": proxy_method("fit", LDA_2C),
    "Dirac-Dirac-2C-Grad": proxy_method("dirac2c"),
    "FT-2C": (ft2c.init, ft2c.step),
    "LDA-Mixture-Grad": proxy_method("fit", (fits.lda_mixture_np, fits.lda_mixture_t)),
    "QDA-Mixture-Grad": proxy_method("fit", (fits.qda_mixture_np, fits.qda_mixture_t)),
    "Dirac-Dirac-Grad": proxy_method("dirac"),
    "LDA-Naive-Grad": proxy_method("fit", (fits.lda_naive_np, fits.lda_naive_t)),
    "QDA-Naive-Grad": proxy_method("fit", (fits.qda_naive_np, fits.qda_naive_t)),
    **baselines.STEPS,
    **{CFG.variant_name(layers, k): proxy_method("fit", LDA_2C, layers=layers, k=k,
                                                 resnet_only=True)
       for layers, k in CFG.VARIANT_SPECS + CFG.VARIANT_SPECS_SHALLOW},
    "Dirac-Dirac-FO-Grad": proxy_method("dirac", forget_only=True),
    "Dirac-Dirac-2C-FO-Grad": proxy_method("dirac2c", forget_only=True),
}
assert set(REGISTRY) == set(CFG.METHODS)


def unlearn_lr(arch, method):
    if arch == "resnet18":
        return CFG.RESNET_UNLEARN_LR
    return CFG.SCRUB_LR if method == "SCRUB" else CFG.UNLEARN_LR
