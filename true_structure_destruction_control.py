import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import pearsonr
import os
import sys

sys.path.append("./")

from speck import make_train_data
from train_nets import make_resnet

# ============================================
# CONFIG
# ============================================

MODEL_CANDIDATES = [
    "./saved_models/net7_small.h5",
    "./net7_small.h5",
    "net7_small.h5"
]

NUM_SAMPLES = 12000
NUM_ROUNDS = 7

# ============================================
# FIND MODEL FILE
# ============================================

print("========================================")
print("TRUE STRUCTURE DESTRUCTION CONTROL")
print("========================================")

print("[+] Loading model.")

MODEL_PATH = None

for path in MODEL_CANDIDATES:

    if os.path.exists(path):

        MODEL_PATH = path
        break

if MODEL_PATH is None:

    raise FileNotFoundError(
        "Could not locate net7_small.h5"
    )

print(f"[+] Using model file: {MODEL_PATH}")

# ============================================
# LOAD MODEL
# ============================================

model = make_resnet(
    depth=1,
    reg_param=1e-5
)

model.load_weights(MODEL_PATH)

print("[+] Model loaded successfully.")

# ============================================
# GENERATE REAL DATA
# ============================================

print("[+] Generating REAL dataset.")

X_real, Y_real = make_train_data(
    NUM_SAMPLES,
    NUM_ROUNDS
)

# ============================================
# FILTER CORRECT SAMPLES
# ============================================

preds = model.predict(
    X_real,
    verbose=0
).flatten()

pred_labels = (
    preds > 0.5
).astype(np.uint8)

correct_mask = (
    pred_labels == Y_real
)

X_real = X_real[correct_mask]
Y_real = Y_real[correct_mask]

print(
    f"[+] Correctly classified samples: "
    f"{len(X_real)}"
)

# ============================================
# CREATE TRUE RANDOM DATA
# ============================================

print("[+] Creating TRUE RANDOM dataset.")

X_random = np.random.randint(
    0,
    2,
    size=X_real.shape
).astype(np.uint8)

Y_random = np.random.randint(
    0,
    2,
    size=len(X_random)
).astype(np.uint8)

# ============================================
# BASELINE
# ============================================

baseline_preds = model.predict(
    X_real,
    verbose=0
).flatten()

baseline_labels = (
    baseline_preds > 0.5
).astype(np.uint8)

baseline_acc = np.mean(
    baseline_labels == Y_real
)

print(
    f"[+] Baseline Accuracy: "
    f"{baseline_acc:.6f}"
)

# ============================================
# CAUSAL IMPORTANCE
# ============================================

def compute_importance(X, Y):

    baseline_preds = model.predict(
        X,
        verbose=0
    ).flatten()

    baseline_labels = (
        baseline_preds > 0.5
    ).astype(np.uint8)

    baseline_acc = np.mean(
        baseline_labels == Y
    )

    importance = []

    for bit in range(16):

        X_mod = X.copy()

        X_mod[:, bit] = 0

        preds = model.predict(
            X_mod,
            verbose=0
        ).flatten()

        labels = (
            preds > 0.5
        ).astype(np.uint8)

        acc = np.mean(
            labels == Y
        )

        drop = baseline_acc - acc

        importance.append(drop)

    importance = np.array(importance)

    importance -= np.min(importance)

    max_val = np.max(importance)

    if max_val > 0:

        importance /= max_val

    return importance

# ============================================
# COMPUTE STRUCTURES
# ============================================

print("[+] Computing REAL structure.")

real_importance = compute_importance(
    X_real,
    Y_real
)

print("[+] Computing TRUE RANDOM structure.")

random_importance = compute_importance(
    X_random,
    Y_random
)

# ============================================
# ENTROPY
# ============================================

def entropy(x):

    x = np.abs(x)

    x += 1e-10

    x /= np.sum(x)

    return -np.sum(
        x * np.log(x)
    )

real_entropy = entropy(
    real_importance
)

random_entropy = entropy(
    random_importance
)

# ============================================
# ROTATIONAL CORRELATION
# ============================================

def rotr(x, r=7):

    return np.roll(x, r)

real_rotr_corr, _ = pearsonr(
    real_importance,
    rotr(real_importance)
)

random_rotr_corr, _ = pearsonr(
    random_importance,
    rotr(random_importance)
)

# ============================================
# CROSS CORRELATION
# ============================================

cross_corr, p_val = pearsonr(
    real_importance,
    random_importance
)

# ============================================
# RESULTS
# ============================================

print("\n========================================")
print("TRUE STRUCTURE DESTRUCTION RESULTS")
print("========================================")

print(
    f"REAL entropy           : "
    f"{real_entropy:.6f}"
)

print(
    f"RANDOM entropy         : "
    f"{random_entropy:.6f}"
)

print()

print(
    f"REAL ROTR correlation  : "
    f"{real_rotr_corr:.6f}"
)

print(
    f"RANDOM ROTR correlation: "
    f"{random_rotr_corr:.6f}"
)

print()

print(
    f"REAL vs RANDOM correlation : "
    f"{cross_corr:.6f}"
)

print(
    f"P-value                    : "
    f"{p_val:.6f}"
)

# ============================================
# TOP REAL HOTSPOTS
# ============================================

print("\n========================================")
print("TOP REAL HOTSPOTS")
print("========================================")

real_sorted = np.argsort(
    real_importance
)[::-1]

for i in range(10):

    bit = real_sorted[i]

    print(
        f"Rank {i+1}: "
        f"Bit={bit}, "
        f"Importance={real_importance[bit]:.6f}"
    )

# ============================================
# TOP RANDOM HOTSPOTS
# ============================================

print("\n========================================")
print("TOP RANDOM HOTSPOTS")
print("========================================")

random_sorted = np.argsort(
    random_importance
)[::-1]

for i in range(10):

    bit = random_sorted[i]

    print(
        f"Rank {i+1}: "
        f"Bit={bit}, "
        f"Importance={random_importance[bit]:.6f}"
    )

# ============================================
# LINE PLOT
# ============================================

plt.figure(figsize=(12,6))

plt.plot(
    real_importance,
    marker='o',
    linewidth=2,
    label='REAL'
)

plt.plot(
    random_importance,
    marker='s',
    linewidth=2,
    label='TRUE_RANDOM'
)

plt.xlabel("Bit Position")

plt.ylabel("Normalized Importance")

plt.title(
    "Real vs True Random Structure"
)

plt.legend()

plt.grid(True)

plt.tight_layout()

plt.savefig(
    "true_random_structure.png"
)

# ============================================
# HEATMAP
# ============================================

heatmap = np.vstack([
    real_importance,
    random_importance
])

plt.figure(figsize=(10,4))

plt.imshow(
    heatmap,
    cmap='hot',
    aspect='auto'
)

plt.yticks(
    [0,1],
    ["REAL", "TRUE_RANDOM"]
)

plt.colorbar(
    label="Normalized Causal Importance"
)

plt.xlabel("Bit Position")

plt.title(
    "True Structure Destruction"
)

plt.tight_layout()

plt.savefig(
    "true_structure_heatmap.png"
)

# ============================================
# ENTROPY BAR
# ============================================

plt.figure(figsize=(6,5))

plt.bar(
    ["REAL", "TRUE_RANDOM"],
    [real_entropy, random_entropy]
)

plt.ylabel("Entropy")

plt.title(
    "Entropy Collapse"
)

plt.tight_layout()

plt.savefig(
    "true_entropy_collapse.png"
)

# ============================================
# ROTATIONAL BAR
# ============================================

plt.figure(figsize=(6,5))

plt.bar(
    ["REAL", "TRUE_RANDOM"],
    [real_rotr_corr, random_rotr_corr]
)

plt.ylabel("ROTR(7) Correlation")

plt.title(
    "Rotational Structure Collapse"
)

plt.tight_layout()

plt.savefig(
    "true_rotational_collapse.png"
)

print("\n[+] True structure destruction complete.")