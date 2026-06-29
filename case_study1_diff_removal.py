import numpy as np
import os

from speck import make_train_data
from train_nets import make_resnet

# ============================================
# CONFIG
# ============================================

NUM_SAMPLES = 5000
NUM_ROUNDS = 7

MODEL_CANDIDATES = [
    "./saved_models/net7_small.h5",
    "./net7_small.h5",
    "net7_small.h5"
]

print("========================================")
print("CASE STUDY 1 : DIFFERENTIAL REMOVAL")
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

print(f"[+] Using model file: {MODEL_PATH}")

# ============================================
# LOAD MODEL
# ============================================

print("[+] Loading model.")

model = make_resnet(
    depth=1,
    reg_param=1e-5
)

model.load_weights(MODEL_PATH)

print("[+] Model loaded successfully.")

# ============================================
# REAL DATA
# ============================================

print("\n[+] Generating REAL dataset.")

X_real, Y_real = make_train_data(
    NUM_SAMPLES,
    NUM_ROUNDS
)

pred_real = (
    model.predict(
        X_real,
        verbose=0
    ).flatten() > 0.5
).astype(np.uint8)

acc_real = np.mean(
    pred_real == Y_real
)

print(
    f"Original Accuracy = {acc_real:.6f}"
)

# ============================================
# RANDOMIZED DATA
# ============================================

print("\n[+] Creating RANDOMIZED dataset.")

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

pred_random = (
    model.predict(
        X_random,
        verbose=0
    ).flatten() > 0.5
).astype(np.uint8)

acc_random = np.mean(
    pred_random == Y_random
)

print(
    f"Randomized Accuracy = {acc_random:.6f}"
)

# ============================================
# FEATURE STATISTICS
# ============================================

real_var = np.var(
    X_real,
    axis=0
)

random_var = np.var(
    X_random,
    axis=0
)

print("\n========================================")
print("RESULTS")
print("========================================")

print(
    f"Real Mean Variance   : {np.mean(real_var):.6f}"
)

print(
    f"Random Mean Variance : {np.mean(random_var):.6f}"
)

print("\nTop Real Bits")

top_real = np.argsort(
    real_var
)[::-1][:10]

for rank, bit in enumerate(
    top_real,
    start=1
):
    print(
        f"Rank {rank}: Bit={bit}"
    )

print("\nTop Random Bits")

top_random = np.argsort(
    random_var
)[::-1][:10]

for rank, bit in enumerate(
    top_random,
    start=1
):
    print(
        f"Rank {rank}: Bit={bit}"
    )

print("\n[+] Case Study Complete.")
# ============================================
# GENERATE FIGURES
# ============================================

import matplotlib.pyplot as plt

print("\n[+] Generating figures.")

# ============================================
# FIGURE 1
# ACCURACY COMPARISON
# ============================================

plt.figure(figsize=(6,4))

plt.bar(
    ["Original", "Randomized"],
    [acc_real * 100, acc_random * 100]
)

plt.ylabel("Accuracy (%)")
plt.title("Effect of Differential Structure Removal")

plt.tight_layout()

plt.savefig(
    "case1_accuracy_comparison.png",
    dpi=300
)

plt.close()

# ============================================
# FIGURE 2
# VARIANCE DISTRIBUTION
# ============================================

plt.figure(figsize=(10,4))

plt.plot(
    real_var,
    linewidth=2,
    label="Original Dataset"
)

plt.plot(
    random_var,
    linewidth=2,
    label="Randomized Dataset"
)

plt.xlabel("Bit Position")
plt.ylabel("Variance")
plt.title("Variance Distribution Across 64 Input Bits")

plt.legend()

plt.tight_layout()

plt.savefig(
    "case1_variance_distribution.png",
    dpi=300
)

plt.close()

# ============================================
# FIGURE 3
# TOP BIT COMPARISON
# ============================================

real_top_values = real_var[top_real]
random_top_values = random_var[top_random]

x = np.arange(10)

plt.figure(figsize=(10,5))

plt.bar(
    x - 0.2,
    real_top_values,
    width=0.4,
    label="Original"
)

plt.bar(
    x + 0.2,
    random_top_values,
    width=0.4,
    label="Randomized"
)

plt.xticks(
    x,
    [str(bit) for bit in top_real]
)

plt.xlabel("Top Important Bit Positions")
plt.ylabel("Variance")
plt.title("Top Bit Comparison")

plt.legend()

plt.tight_layout()

plt.savefig(
    "case1_top_bits.png",
    dpi=300
)

plt.close()

print("[+] Saved: case1_accuracy_comparison.png")
print("[+] Saved: case1_variance_distribution.png")
print("[+] Saved: case1_top_bits.png")