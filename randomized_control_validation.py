# randomized_control_validation.py

import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

import warnings
warnings.filterwarnings("ignore")

import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

from scipy.stats import pearsonr
from tensorflow.keras.models import load_model

import speck as sp
from train_nets import make_resnet


# ============================================================
# CONFIGURATION
# ============================================================

ROUNDS = 7

NUM_SAMPLES = 12000
BATCH_SIZE = 256

ROTATION = 7


# ============================================================
# LOAD MODEL
# ============================================================

print("=" * 40)
print("RANDOMIZED CONTROL VALIDATION")
print("=" * 40)

print("[+] Loading model.")

MODEL_CANDIDATES = [
    f'net{ROUNDS}_small.h5',
    f'./net{ROUNDS}_small.h5',
    f'saved_models/net{ROUNDS}_small.h5',
    f'./saved_models/net{ROUNDS}_small.h5'
]

MODEL_PATH = None

for path in MODEL_CANDIDATES:

    if os.path.exists(path):

        MODEL_PATH = path
        break

if MODEL_PATH is None:

    raise FileNotFoundError(
        f"Could not find model for {ROUNDS} rounds."
    )

print(f"[+] Using model file: {MODEL_PATH}")

try:

    model = load_model(MODEL_PATH)

except:

    model = make_resnet(
        depth=1,
        reg_param=1e-5
    )

    model.load_weights(MODEL_PATH)

print("[+] Model loaded successfully.")


# ============================================================
# DATASET GENERATION
# ============================================================

print("[+] Generating dataset.")

X, Y = sp.make_train_data(
    NUM_SAMPLES,
    ROUNDS,
    diff=(0x0040, 0)
)

preds = model.predict(
    X,
    batch_size=BATCH_SIZE
).flatten()

correct_mask = ((preds > 0.5) == Y)

X = X[correct_mask]
Y = Y[correct_mask]

print(f"[+] Correctly classified samples: {len(X)}")


# ============================================================
# INTEGRATED GRADIENTS
# ============================================================

def integrated_gradients(
    model,
    sample,
    baseline=None,
    steps=32
):

    sample = sample.astype(np.float32)

    if baseline is None:

        baseline = np.zeros_like(sample).astype(np.float32)

    interpolated_inputs = []

    for alpha in np.linspace(0, 1, steps):

        interpolated = baseline + alpha * (sample - baseline)

        interpolated_inputs.append(interpolated)

    interpolated_inputs = np.array(interpolated_inputs)

    interpolated_tensor = tf.convert_to_tensor(
        interpolated_inputs,
        dtype=tf.float32
    )

    with tf.GradientTape() as tape:

        tape.watch(interpolated_tensor)

        preds = model(interpolated_tensor)

    grads = tape.gradient(preds, interpolated_tensor)

    grads = grads.numpy()

    avg_grads = np.mean(grads, axis=0)

    ig = (sample - baseline) * avg_grads

    return ig


# ============================================================
# COMPUTE REAL CAUSAL STRUCTURE
# ============================================================

print("\n[+] Computing REAL causal structure.")

num_eval = min(256, len(X))

real_attr = np.zeros(64)

for i in range(num_eval):

    ig = integrated_gradients(
        model,
        X[i]
    )

    real_attr += np.abs(ig)

real_attr /= num_eval

real_matrix = real_attr.reshape(4, 16)

real_importance = np.mean(
    real_matrix,
    axis=0
)

# normalize
real_importance = (
    real_importance - np.min(real_importance)
)

real_importance = (
    real_importance /
    (np.max(real_importance) + 1e-10)
)


# ============================================================
# ROTATION-BREAKING RANDOMIZATION
# ============================================================

print("[+] Creating ROTATION-BROKEN dataset.")

X_random = X.copy()

rng = np.random.default_rng(1234)

for i in range(len(X_random)):

    sample = X_random[i].copy()

    reshaped = sample.reshape(4, 16)

    # independently rotate each ciphertext word
    for word in range(4):

        shift = rng.integers(1, 16)

        reshaped[word] = np.roll(
            reshaped[word],
            shift
        )

    # randomly permute ciphertext words
    rng.shuffle(reshaped)

    X_random[i] = reshaped.flatten()


# ============================================================
# COMPUTE RANDOMIZED STRUCTURE
# ============================================================

print("[+] Computing RANDOMIZED causal structure.")

random_attr = np.zeros(64)

for i in range(num_eval):

    ig = integrated_gradients(
        model,
        X_random[i]
    )

    random_attr += np.abs(ig)

random_attr /= num_eval

random_matrix = random_attr.reshape(4, 16)

random_importance = np.mean(
    random_matrix,
    axis=0
)

# normalize
random_importance = (
    random_importance - np.min(random_importance)
)

random_importance = (
    random_importance /
    (np.max(random_importance) + 1e-10)
)


# ============================================================
# SAFE ENTROPY
# ============================================================

real_safe = np.abs(real_importance)
real_safe = real_safe / (np.sum(real_safe) + 1e-10)

real_entropy = -np.sum(
    real_safe * np.log(real_safe + 1e-10)
)

random_safe = np.abs(random_importance)
random_safe = random_safe / (np.sum(random_safe) + 1e-10)

random_entropy = -np.sum(
    random_safe * np.log(random_safe + 1e-10)
)


# ============================================================
# ROTATIONAL STRUCTURE
# ============================================================

rot_real = np.roll(
    real_importance,
    ROTATION
)

real_corr, _ = pearsonr(
    real_importance,
    rot_real
)

rot_random = np.roll(
    random_importance,
    ROTATION
)

random_corr, _ = pearsonr(
    random_importance,
    rot_random
)

cross_corr, pval = pearsonr(
    real_importance,
    random_importance
)


# ============================================================
# VISUALIZATION
# ============================================================

# ------------------------------------------------------------
# HEATMAP
# ------------------------------------------------------------

plt.figure(figsize=(12, 4))

heatmap = np.vstack([
    real_importance,
    random_importance
])

plt.imshow(
    heatmap,
    cmap='hot',
    aspect='auto'
)

plt.yticks(
    [0, 1],
    ['REAL_STRUCTURE', 'RANDOMIZED_STRUCTURE']
)

plt.xlabel("Bit Position")

plt.colorbar(
    label="Normalized Causal Importance"
)

plt.title("Real vs Randomized Causal Structure")

plt.tight_layout()

plt.savefig(
    "real_vs_randomized_heatmap.png",
    dpi=300
)


# ------------------------------------------------------------
# LINE PLOT
# ------------------------------------------------------------

plt.figure(figsize=(12, 6))

plt.plot(
    real_importance,
    marker='o',
    label='REAL'
)

plt.plot(
    random_importance,
    marker='s',
    label='RANDOMIZED'
)

plt.xlabel("Bit Position")

plt.ylabel("Normalized Importance")

plt.title("Real vs Randomized Motif Structure")

plt.grid(True)

plt.legend()

plt.tight_layout()

plt.savefig(
    "real_vs_randomized_structure.png",
    dpi=300
)


# ------------------------------------------------------------
# ROTATIONAL COLLAPSE
# ------------------------------------------------------------

plt.figure(figsize=(6, 5))

plt.bar(
    ['REAL', 'RANDOMIZED'],
    [real_corr, random_corr]
)

plt.ylabel("ROTR(7) Correlation")

plt.title("Rotational Structure Collapse")

plt.tight_layout()

plt.savefig(
    "rotational_collapse.png",
    dpi=300
)


# ------------------------------------------------------------
# ENTROPY COLLAPSE
# ------------------------------------------------------------

plt.figure(figsize=(6, 5))

plt.bar(
    ['REAL', 'RANDOMIZED'],
    [real_entropy, random_entropy]
)

plt.ylabel("Entropy")

plt.title("Structural Entropy Collapse")

plt.tight_layout()

plt.savefig(
    "entropy_collapse.png",
    dpi=300
)


# ============================================================
# REPORTING
# ============================================================

print("\n" + "=" * 40)
print("CONTROL VALIDATION RESULTS")
print("=" * 40)

print(f"REAL entropy           : {real_entropy:.6f}")
print(f"RANDOMIZED entropy     : {random_entropy:.6f}")

print()

print(f"REAL ROTR correlation  : {real_corr:.6f}")
print(f"RANDOM ROTR correlation: {random_corr:.6f}")

print()

print(f"REAL vs RANDOM correlation : {cross_corr:.6f}")
print(f"P-value                    : {pval:.6f}")

print("\n" + "=" * 40)
print("TOP REAL HOTSPOTS")
print("=" * 40)

top_real = np.argsort(
    real_importance
)[::-1][:10]

for rank, bit in enumerate(top_real):

    print(
        f"Rank {rank+1}: "
        f"Bit={bit}, "
        f"Importance={real_importance[bit]:.6f}"
    )

print("\n" + "=" * 40)
print("TOP RANDOMIZED HOTSPOTS")
print("=" * 40)

top_random = np.argsort(
    random_importance
)[::-1][:10]

for rank, bit in enumerate(top_random):

    print(
        f"Rank {rank+1}: "
        f"Bit={bit}, "
        f"Importance={random_importance[bit]:.6f}"
    )

print("\n[+] Randomized control validation complete.")