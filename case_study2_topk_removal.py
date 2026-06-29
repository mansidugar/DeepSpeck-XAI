import numpy as np
import os
import matplotlib.pyplot as plt

from speck import make_train_data
from train_nets import make_resnet

# ============================================
# CONFIG
# ============================================

NUM_SAMPLES = 5000
NUM_ROUNDS = 7

TOP_BITS = [
    22,
    54,
    36,
    4,
    50
]

MODEL_CANDIDATES = [
    "./saved_models/net7_small.h5",
    "./net7_small.h5",
    "net7_small.h5"
]

print("========================================")
print("CASE STUDY 2 : TOP-K HOTSPOT REMOVAL")
print("========================================")

# ============================================
# FIND MODEL
# ============================================

MODEL_PATH = None

for path in MODEL_CANDIDATES:

    if os.path.exists(path):

        MODEL_PATH = path
        break

if MODEL_PATH is None:

    raise FileNotFoundError(
        "Could not locate net7_small.h5"
    )

# ============================================
# LOAD MODEL
# ============================================

model = make_resnet(
    depth=1,
    reg_param=1e-5
)

model.load_weights(
    MODEL_PATH
)

print("[+] Model loaded.")

# ============================================
# DATA
# ============================================

X, Y = make_train_data(
    NUM_SAMPLES,
    NUM_ROUNDS
)

# ============================================
# BASELINE
# ============================================

pred_base = model.predict(
    X,
    verbose=0
).flatten()

acc_base = np.mean(
    (pred_base > 0.5) == Y
)

conf_base = np.mean(
    np.abs(pred_base - 0.5)
)

print(
    f"Baseline Accuracy = {acc_base:.6f}"
)

# ============================================
# REMOVE TOP BITS
# ============================================

X_mod = X.copy()

for bit in TOP_BITS:

    X_mod[:, bit] = 0

pred_mod = model.predict(
    X_mod,
    verbose=0
).flatten()

acc_mod = np.mean(
    (pred_mod > 0.5) == Y
)

conf_mod = np.mean(
    np.abs(pred_mod - 0.5)
)

print(
    f"Modified Accuracy = {acc_mod:.6f}"
)

print(
    f"Accuracy Drop = {acc_base - acc_mod:.6f}"
)

# ============================================
# FIGURE 1
# ============================================

plt.figure(figsize=(6,4))

plt.bar(
    ["Baseline",
     "Top-K Removed"],
    [
        acc_base * 100,
        acc_mod * 100
    ]
)

plt.ylabel("Accuracy (%)")

plt.title(
    "Impact of Top-K Bit Removal"
)

plt.tight_layout()

plt.savefig(
    "case2_accuracy.png",
    dpi=300
)

plt.close()

# ============================================
# FIGURE 2
# ============================================

plt.figure(figsize=(6,4))

plt.bar(
    ["Baseline",
     "Top-K Removed"],
    [
        conf_base,
        conf_mod
    ]
)

plt.ylabel(
    "Mean Confidence"
)

plt.title(
    "Prediction Confidence Comparison"
)

plt.tight_layout()

plt.savefig(
    "case2_confidence.png",
    dpi=300
)

plt.close()

print("\n[+] Saved:")
print("case2_accuracy.png")
print("case2_confidence.png")

print("\n[+] Case Study Complete.")