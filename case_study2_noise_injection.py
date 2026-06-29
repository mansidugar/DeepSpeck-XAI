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

NOISE_RATE = 0.05

MODEL_CANDIDATES = [
    "./saved_models/net7_small.h5",
    "./net7_small.h5",
    "net7_small.h5"
]

print("========================================")
print("CASE STUDY 2 : NOISE INJECTION")
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
# ORIGINAL DATA
# ============================================

print("[+] Generating dataset.")

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
# ADD NOISE
# ============================================

print(
    f"[+] Injecting {NOISE_RATE*100:.0f}% noise."
)

X_noisy = X.copy()

noise_mask = (
    np.random.rand(*X.shape)
    < NOISE_RATE
)

X_noisy[noise_mask] ^= 1

# ============================================
# NOISY RESULTS
# ============================================

pred_noisy = model.predict(
    X_noisy,
    verbose=0
).flatten()

acc_noisy = np.mean(
    (pred_noisy > 0.5) == Y
)

conf_noisy = np.mean(
    np.abs(pred_noisy - 0.5)
)

print(
    f"Noisy Accuracy = {acc_noisy:.6f}"
)

print(
    f"Accuracy Drop = {acc_base - acc_noisy:.6f}"
)

print(
    f"Baseline Confidence = {conf_base:.6f}"
)

print(
    f"Noisy Confidence = {conf_noisy:.6f}"
)

# ============================================
# GRAPH 1
# ============================================

plt.figure(figsize=(6,4))

plt.bar(
    ["Baseline",
     "5% Noise"],
    [
        acc_base*100,
        acc_noisy*100
    ]
)

plt.ylabel("Accuracy (%)")

plt.title(
    "Impact of Noise Injection"
)

plt.tight_layout()

plt.savefig(
    "case2_noise_accuracy.png",
    dpi=300
)

plt.close()

# ============================================
# GRAPH 2
# ============================================

plt.figure(figsize=(6,4))

plt.bar(
    ["Baseline",
     "5% Noise"],
    [
        conf_base,
        conf_noisy
    ]
)

plt.ylabel(
    "Mean Confidence"
)

plt.title(
    "Prediction Confidence Under Noise"
)

plt.tight_layout()

plt.savefig(
    "case2_noise_confidence.png",
    dpi=300
)

plt.close()

print("\n[+] Saved:")
print("case2_noise_accuracy.png")
print("case2_noise_confidence.png")

print("\n[+] Case Study Complete.")