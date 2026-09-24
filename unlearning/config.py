"""
Configuration: paths, hyper-parameters, experiment grid and method roster.
────────────────────────────────────────────────────────────────────────────
Every value that changes a number lives here. `snapshot()` is written into
every manifest so a result tree records the settings that produced it.
"""
import os
import re

# ── Paths ────────────────────────────────────────────────────────────────────
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(ROOT, "data")
RAW_DIR = os.path.join(DATA_DIR, "raw")                 # torchvision CIFAR
FEATURE_DIR = os.path.join(DATA_DIR, "dinov2-small")    # cached CLS features
DINOV2_MODEL = "facebook/dinov2-small"
RESULTS_DIR = os.path.join(ROOT, "results")
MODELS_DIR = os.path.join(ROOT, "models")               # initial checkpoints


def set_dirs(results=None, models=None):
    global RESULTS_DIR, MODELS_DIR
    if results:
        RESULTS_DIR = os.path.abspath(results)
    if models:
        MODELS_DIR = os.path.abspath(models)


# ── Datasets ─────────────────────────────────────────────────────────────────
DATASETS = ["cifar10", "cifar100"]
NUM_CLASSES = {"cifar10": 10, "cifar100": 100}
CIFAR10_CLASSES = ["airplane", "automobile", "bird", "cat", "deer", "dog",
                   "frog", "horse", "ship", "truck"]
CIFAR100_CLASSES = [
    "apple", "aquarium_fish", "baby", "bear", "beaver", "bed", "bee", "beetle",
    "bicycle", "bottle", "bowl", "boy", "bridge", "bus", "butterfly", "camel",
    "can", "castle", "caterpillar", "cattle", "chair", "chimpanzee", "clock",
    "cloud", "cockroach", "couch", "crab", "crocodile", "cup", "dinosaur",
    "dolphin", "elephant", "flatfish", "forest", "fox", "girl", "hamster",
    "house", "kangaroo", "keyboard", "lamp", "lawn_mower", "leopard", "lion",
    "lizard", "lobster", "man", "maple_tree", "motorcycle", "mountain",
    "mouse", "mushroom", "oak_tree", "orange", "orchid", "otter", "palm_tree",
    "pear", "pickup_truck", "pine_tree", "plain", "plate", "poppy",
    "porcupine", "possum", "rabbit", "raccoon", "ray", "road", "rocket",
    "rose", "sea", "seal", "shark", "shrew", "skunk", "skyscraper", "snail",
    "snake", "spider", "squirrel", "streetcar", "sunflower", "sweet_pepper",
    "table", "tank", "telephone", "television", "tiger", "tractor", "train",
    "trout", "tulip", "turtle", "wardrobe", "whale", "willow_tree", "wolf",
    "woman", "worm",
]
# SUBCLASS scenario: train on superclasses, forget one fine class.
# CIFAR-10: vehicle (0) / animal (1). CIFAR-100: the 20 canonical superclasses.
CIFAR10_SUPERCLASS_MAP = {**{c: 0 for c in [0, 1, 8, 9]},
                          **{c: 1 for c in [2, 3, 4, 5, 6, 7]}}
CIFAR10_SUPERCLASS_NAMES = {0: "vehicle", 1: "animal"}
CIFAR100_SUPERCLASS_NAMES = {
    0: "aquatic_mammals", 1: "fish", 2: "flowers", 3: "food_containers",
    4: "fruit_and_vegetables", 5: "household_electrical_devices",
    6: "household_furniture", 7: "insects", 8: "large_carnivores",
    9: "large_man-made_outdoor_things", 10: "large_natural_outdoor_scenes",
    11: "large_omnivores_and_herbivores", 12: "medium_mammals",
    13: "non-insect_invertebrates", 14: "people", 15: "reptiles",
    16: "small_mammals", 17: "trees", 18: "vehicles_1", 19: "vehicles_2",
}
_CIFAR100_FINE_TO_COARSE = [
    4, 1, 14, 8, 0, 6, 7, 7, 18, 3, 3, 14, 9, 18, 7, 11, 3, 9, 7, 11,
    6, 11, 5, 10, 7, 6, 13, 15, 3, 15, 0, 11, 1, 10, 12, 14, 16, 9, 11, 5,
    5, 19, 8, 8, 15, 13, 14, 17, 18, 10, 16, 4, 17, 4, 2, 0, 17, 4, 18, 17,
    10, 3, 2, 12, 12, 16, 12, 1, 9, 19, 2, 10, 0, 1, 16, 12, 9, 13, 15, 13,
    16, 19, 2, 4, 6, 19, 5, 5, 8, 19, 18, 1, 2, 15, 6, 0, 17, 8, 14, 13,
]
CIFAR100_SUPERCLASS_MAP = dict(enumerate(_CIFAR100_FINE_TO_COARSE))


def class_name(dataset, idx):
    names = CIFAR10_CLASSES if dataset == "cifar10" else CIFAR100_CLASSES
    return names[idx] if 0 <= idx < len(names) else str(idx)


def superclass_map(dataset):
    return CIFAR10_SUPERCLASS_MAP if dataset == "cifar10" else CIFAR100_SUPERCLASS_MAP


def superclass_names(dataset):
    return CIFAR10_SUPERCLASS_NAMES if dataset == "cifar10" else CIFAR100_SUPERCLASS_NAMES


# ── Training of the references (initial + retrained models) ─────────────────
# Heads: Adam + cosine, fixed epoch count, on cached DINOv2 features.
FEATURE_DIM = 384
HIDDEN_DIM = 256
EPOCHS = 100
LR = 1e-3
WEIGHT_DECAY = 1e-4
BATCH_SIZE = 512
# ResNet-18: ImageNet weights, fine-tuned end to end on 64x64 upsampled CIFAR
# with Adam + cosine, until 99 % TRAIN accuracy (a model that has not
# memorised its training set has nothing to unlearn), capped at 60 epochs.
RESNET_INPUT_SIZE = 64
RESNET_BATCH_SIZE = 128
RESNET_LR = 1e-3
RESNET_WD = 1e-4
RESNET_TRAIN_ACC_TARGET = 99.0
RESNET_MAX_EPOCHS = 60

# ── Unlearning ───────────────────────────────────────────────────────────────
UNLEARN_EPOCHS = 20           # every method; the LAST epoch is the released model
UNLEARN_LR = 1e-3             # heads (Adam)
RESNET_UNLEARN_LR = 1e-4      # ResNet (every method)
LR_GAMMA = 0.95               # per-epoch LR decay (from epoch 2 on)
FT_LR_GAMMA = 1.0             # FT and FT-2C: constant LR
# SCRUB (Kurmanji et al., 2023): the official loop with the official
# large-scale values (ResNet-18 / CIFAR config of the authors' code): alpha
# (KL) and gamma (CE) of the min-step, KD temperature, max-steps on D_f only
# during the first SCRUB_MSTEPS epochs. The lr is the bench's (Adam).
SCRUB_ALPHA, SCRUB_GAMMA, SCRUB_T, SCRUB_MSTEPS, SCRUB_LR = 0.001, 0.99, 4.0, 2, 3e-4
# SalUn (Fan et al., 2024, Alg. 1): fraction of weights kept salient (global
# ranking), constant LR
SALUN_SPARSITY, SALUN_LR_GAMMA = 0.5, 1.0
# Algorithm settings stamped into each trace: a trace written under another
# recipe is re-run rather than resumed.
RECIPES = {
    "SCRUB": {"loop": "Kurmanji2023-official", "alpha": SCRUB_ALPHA,
              "gamma": SCRUB_GAMMA, "T": SCRUB_T, "msteps": SCRUB_MSTEPS},
    "SalUn": {"loop": "Fan2024-Alg1", "mask": "global-topk",
              "sparsity": SALUN_SPARSITY},
}
RL_FT_ALPHA = 0.90            # RL+FT: (1-a) CE(random labels on D_f) + a CE(D_r)

# ── Proxies ──────────────────────────────────────────────────────────────────
# eta* is the root of Z(eta) = E_x logsumexp(log p_init + eta * Delta) searched
# on (0, TAU_MAX]; the paper's operating point is eta* <= 1.
TAU_MAX = 1.0
ETA_SEARCH_EPS = 1e-4
RESNET_PROXY_K = 256          # ResNet proxy space: layer3 + JL projection to k
RESNET_PROXY_SEED = 0


def set_tau_max(v):
    global TAU_MAX
    TAU_MAX = float(v)


# ── Seeds and splits ─────────────────────────────────────────────────────────
# One seed drives one (initial, retrained) pair of references and every method
# unlearned from that initial model. CLASS/SUBCLASS: the forget set is a
# function of the labels, so the five seeds share one split (group "42").
# RANDOM: the split is drawn with the seed itself, so each seed is its own
# group, completed by one extra control replica (same split, weight seed
# 1000 + s) so that both control rows have two references.
SEEDS = [42, 0, 1, 2, 3]
SPLIT_SEED = 42
CTRL_SEED_OFFSET = 1000
RETAIN_EVAL_N, RETAIN_EVAL_SEED = 10_000, 0     # retain evaluation subsample
TRACE_EVAL_N, TRACE_SEED = 2000, 0              # per-epoch test/retain subsample
AUDIT_SET = "forget"                            # kept whole, sample by sample

# ── Grid: 2 datasets x 4 architectures x (5 CLASS + 5 SUBCLASS + 4 RANDOM) ───
ARCHS = ["linear", "mlp1", "mlp2", "resnet18"]
CLASS_KEYS = {"cifar10": [0, 2, 4, 6, 8], "cifar100": [0, 20, 40, 60, 80]}
RANDOM_KEYS = [1, 10, 100, 1000]
CELL_RE = re.compile(r"^(cifar10|cifar100)_(linear|mlp1|mlp2|resnet18)_"
                     r"(CLASS|SUBCLASS|RANDOM)_(\d+)$")


def _grid():
    out = []
    for ds in DATASETS:
        for a in ARCHS:
            out += [f"{ds}_{a}_CLASS_{k}" for k in CLASS_KEYS[ds]]
            out += [f"{ds}_{a}_SUBCLASS_{k}" for k in CLASS_KEYS[ds]]
            out += [f"{ds}_{a}_RANDOM_{n}" for n in RANDOM_KEYS]
    return out


CELLS = _grid()
CELLS_HEAD = [c for c in CELLS if "_resnet18_" not in c]
CELLS_RESNET = [c for c in CELLS if "_resnet18_" in c]


def parse_cell(cell):
    m = CELL_RE.match(cell)
    if not m:
        raise ValueError(f"bad cell {cell!r}: expected {{cifar10|cifar100}}_"
                         f"{{linear|mlp1|mlp2|resnet18}}_{{CLASS|SUBCLASS|RANDOM}}_"
                         f"{{int}}, e.g. cifar10_mlp1_SUBCLASS_0")
    ds, arch, scen, k = m.groups()
    return ds, arch, scen, int(k)


def is_resnet(cell):
    return "_resnet18_" in cell


def split_groups(cell, seeds=None):
    """{split: {"paper": [...], "ctrl": [...]}} — see the note above."""
    seeds = list(SEEDS if seeds is None else seeds)
    if parse_cell(cell)[2] == "RANDOM":
        return {s: {"paper": [s], "ctrl": [CTRL_SEED_OFFSET + s]} for s in seeds}
    return {SPLIT_SEED: {"paper": seeds, "ctrl": []}}


def seeds_on_disk(cell):
    """Paper seeds with a reference on disk, whatever SEEDS says: one split
    directory per seed for RANDOM, the files of the one group otherwise."""
    root = os.path.join(cell_dir(cell), "logits")
    if parse_cell(cell)[2] == "RANDOM":
        names, pat = (os.listdir(root) if os.path.isdir(root) else []), r"split(\d+)"
    else:
        d = logits_dir(cell, SPLIT_SEED)
        names, pat = (os.listdir(d) if os.path.isdir(d) else []), r"(?:init|retrain)_(\d+)\.npz"
    return sort_seeds({int(m.group(1)) for n in names if (m := re.fullmatch(pat, n))})


def group_seeds(grp):
    return list(grp["paper"]) + list(grp["ctrl"])


def sort_seeds(got, order=None):
    order = list(SEEDS if order is None else order)
    return sorted(got, key=lambda s: (order.index(s) if s in order
                                      else len(order) + int(s)))


# ── Methods ──────────────────────────────────────────────────────────────────
BASELINES = ["SCRUB", "SalUn", "RL+FT", "GA+FT", "FT", "GA"]
PROXIES = ["LDA-2C-Grad", "Dirac-Dirac-2C-Grad", "FT-2C",
           "LDA-Mixture-Grad", "QDA-Mixture-Grad", "Dirac-Dirac-Grad",
           "LDA-Naive-Grad", "QDA-Naive-Grad"]
# LDA-2C fitted on other ResNet feature spaces: tap x JL projection dim.
# LDA-2C-l3d256-Grad is LDA-2C-Grad itself (the control row of that ablation).
VARIANT_SPECS = [((3,), 512), ((3,), 256), ((3,), 128),
                 ((4,), 512), ((4,), 256), ((4,), 128),
                 ((3, 4), 512), ((3, 4), 256), ((3, 4), 128)]
VARIANT_SPECS_SHALLOW = [((1,), 512), ((1,), 256), ((1,), 128),
                         ((2,), 512), ((2,), 256), ((2,), 128),
                         ((2, 3), 512), ((2, 3), 256), ((2, 3), 128)]


def variant_name(layers, k):
    return f"LDA-2C-l{''.join(str(l) for l in layers)}d{int(k)}-Grad"


RESNET_VARIANTS = [variant_name(*s) for s in VARIANT_SPECS]
RESNET_VARIANTS_SHALLOW = [variant_name(*s) for s in VARIANT_SPECS_SHALLOW]
# The two Dirac proxies distilled on the forget set only.
DIRAC_FO = ["Dirac-Dirac-FO-Grad", "Dirac-Dirac-2C-FO-Grad"]

METHODS = PROXIES + BASELINES + RESNET_VARIANTS + RESNET_VARIANTS_SHALLOW + DIRAC_FO
RESNET_ONLY = set(RESNET_VARIANTS) | set(RESNET_VARIANTS_SHALLOW)
# Target-only: proxy fit + eta search + black-box target, no distillation.
TARGET_ONLY = set(RESNET_VARIANTS_SHALLOW)
# Methods whose epoch loop is a KL distillation of the proxy target
# (FT-2C and the baselines run their own update).
DISTILLING = (set(PROXIES) - {"FT-2C"}) | RESNET_ONLY | set(DIRAC_FO)
# Proxies whose target Delta is pointwise (no target outside the training rows)
# or whose released model is the target itself (FT-2C): no `target_*.npz`.
NO_TARGET = {"Dirac-Dirac-Grad", "Dirac-Dirac-2C-Grad", "FT-2C"} | set(DIRAC_FO)
CONTROLS = ["Retrain", "Base"]


def methods_for_cell(cell, methods=None):
    roster = list(METHODS if methods is None else methods)
    return roster if is_resnet(cell) else [m for m in roster if m not in RESNET_ONLY]


# ── Output layout ────────────────────────────────────────────────────────────

def safe_method(m):
    return m.replace("+", "-plus-").replace("/", "-")


def cell_dir(cell):
    return os.path.join(RESULTS_DIR, cell)


def logits_dir(cell, split):
    return os.path.join(cell_dir(cell), "logits", f"split{split}")


def retrain_path(cell, split, seed):
    return os.path.join(logits_dir(cell, split), f"retrain_{seed}.npz")


def init_path(cell, split, seed):
    return os.path.join(logits_dir(cell, split), f"init_{seed}.npz")


def init_ckpt(cell, split, seed):
    return os.path.join(MODELS_DIR, cell, f"split{split}", f"init_{seed}.pt")


def trace_path(cell, split, method, seed):
    return os.path.join(logits_dir(cell, split), f"trace_{safe_method(method)}_{seed}.npz")


def target_path(cell, split, method, seed):
    return os.path.join(logits_dir(cell, split), f"target_{safe_method(method)}_{seed}.npz")


def eval_meta_path(cell, split):
    return os.path.join(cell_dir(cell), f"eval_meta_split{split}.npz")


def epochs_json(cell):
    return os.path.join(cell_dir(cell), "epochs.json")


def audit_json(cell):
    return os.path.join(cell_dir(cell), "audit.json")


def audit_npz(cell, split):
    return os.path.join(cell_dir(cell), f"audit_{AUDIT_SET}_split{split}.npz")


def floor_pairs_json(cell):
    return os.path.join(cell_dir(cell), "floor_pairs.json")


def snapshot():
    """Every setting that can change a number, for the manifests."""
    keys = ["EPOCHS", "LR", "WEIGHT_DECAY", "BATCH_SIZE", "RESNET_INPUT_SIZE",
            "RESNET_BATCH_SIZE", "RESNET_LR", "RESNET_WD",
            "RESNET_TRAIN_ACC_TARGET", "RESNET_MAX_EPOCHS", "UNLEARN_EPOCHS",
            "UNLEARN_LR", "RESNET_UNLEARN_LR", "LR_GAMMA", "FT_LR_GAMMA",
            "SCRUB_ALPHA", "SCRUB_GAMMA", "SCRUB_LR", "SCRUB_T",
            "SCRUB_MSTEPS", "SALUN_SPARSITY", "SALUN_LR_GAMMA", "RL_FT_ALPHA", "TAU_MAX", "ETA_SEARCH_EPS",
            "RESNET_PROXY_K", "RESNET_PROXY_SEED", "RETAIN_EVAL_N",
            "TRACE_EVAL_N", "DINOV2_MODEL"]
    g = globals()
    return {k: g[k] for k in keys}
