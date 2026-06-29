# ============================================================
# wrong_key_structure_validation.py
# ============================================================
#
# PURPOSE
# -------
# Compare attribution structure between:
#
#   1. CORRECT-KEY ciphertext pairs
#   2. WRONG-KEY / STRUCTURE-COLLAPSED pairs
#
# using:
#   - Integrated Gradients
#   - Entropy analysis
#   - Structural concentration
#   - Correlation analysis
#
# ============================================================

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

from scipy.stats import entropy
from scipy.stats import pearsonr

from tensorflow.keras.models import model_from_json

import speck as sp

# ============================================================
# OUTPUT DIRECTORY
# ============================================================

os.makedirs("outputs", exist_ok=True)

# ============================================================
# CONFIGURATION
# ============================================================

ROUNDS = 7
MODEL_PATH = f'net{ROUNDS}_small.h5'

NUM_SAMPLES = 4000
BATCH_SIZE = 256

# ============================================================
# LOAD MODEL
# ============================================================

print(f"[+] Loading {ROUNDS}-round distinguisher.")

with open('single_block_resnet.json', 'r') as f:
    json_model = f.read()

model = model_from_json(json_model)

model.load_weights(MODEL_PATH)

print("[+] Model loaded successfully.")

# ============================================================
# GENERATE CORRECT-KEY DATASET
# ============================================================

print("[+] Generating REAL differential dataset.")

X_real, Y_real = sp.make_train_data(
    NUM_SAMPLES,
    ROUNDS
)

preds = model.predict(
    X_real,
    batch_size=BATCH_SIZE
).flatten()

pred_labels = (
    preds > 0.5
).astype(np.uint8)

correct_mask = (
    pred_labels == Y_real
)

X_correct = X_real[correct_mask]

print(
    f"[+] Correctly classified samples: "
    f"{len(X_correct)}"
)

# ============================================================
# CREATE WRONG-KEY / COLLAPSED STRUCTURE DATA
# ============================================================

print("[+] Creating structure-collapsed samples.")

X_wrong = np.copy(X_correct)

# ------------------------------------------------------------
# destroy pairwise differential alignment
# ------------------------------------------------------------

np.random.shuffle(X_wrong[:, :32])
np.random.shuffle(X_wrong[:, 32:])

# ============================================================
# INTEGRATED GRADIENTS
# ============================================================

def integrated_gradients(
    model,
    inputs,
    steps=32
):

    inputs = tf.cast(
        inputs,
        tf.float32
    )

    baseline = tf.zeros_like(inputs)

    integrated_grads = tf.zeros_like(inputs)

    # --------------------------------------------------------
    # interpolate baseline -> input
    # --------------------------------------------------------

    for alpha in np.linspace(
        0.0,
        1.0,
        steps
    ):

        interpolated = (
            baseline
            + alpha * (inputs - baseline)
        )

        with tf.GradientTape() as tape:

            tape.watch(interpolated)

            predictions = model(
                interpolated,
                training=False
            )

        grads = tape.gradient(
            predictions,
            interpolated
        )

        # ----------------------------------------------------
        # safety check
        # ----------------------------------------------------

        if grads is None:

            raise ValueError(
                "Gradient computation failed."
            )

        integrated_grads += grads

    avg_grads = integrated_grads / steps

    final_grads = (
        (inputs - baseline)
        * avg_grads
    )

    return final_grads.numpy()

# ============================================================
# COMPUTE ATTRIBUTIONS
# ============================================================

print("[+] Computing CORRECT-KEY attributions.")

attr_correct = integrated_gradients(
    model,
    X_correct[:512]
)

print("[+] Computing WRONG-KEY attributions.")

attr_wrong = integrated_gradients(
    model,
    X_wrong[:512]
)

# ============================================================
# MEAN ATTRIBUTION MAPS
# ============================================================

mean_correct = np.mean(
    np.abs(attr_correct),
    axis=0
)

mean_wrong = np.mean(
    np.abs(attr_wrong),
    axis=0
)

mean_correct = mean_correct.reshape(4, 16)
mean_wrong = mean_wrong.reshape(4, 16)

# ============================================================
# ENTROPY ANALYSIS
# ============================================================

flat_correct = mean_correct.flatten()
flat_wrong = mean_wrong.flatten()

flat_correct /= np.sum(flat_correct)
flat_wrong /= np.sum(flat_wrong)

entropy_correct = entropy(flat_correct)
entropy_wrong = entropy(flat_wrong)

# ============================================================
# STRUCTURAL CORRELATION
# ============================================================

corr, pval = pearsonr(
    mean_correct.flatten(),
    mean_wrong.flatten()
)

# ============================================================
# CONCENTRATION
# ============================================================

max_correct = np.max(mean_correct)
max_wrong = np.max(mean_wrong)

# ============================================================
# PRINT RESULTS
# ============================================================

print("\n======================================")
print("WRONG-KEY STRUCTURAL VALIDATION")
print("======================================")

print(
    f"Correct-key entropy : "
    f"{entropy_correct:.6f}"
)

print(
    f"Wrong-key entropy   : "
    f"{entropy_wrong:.6f}"
)

print()

print(
    f"Correct-key max attribution : "
    f"{max_correct:.6f}"
)

print(
    f"Wrong-key max attribution   : "
    f"{max_wrong:.6f}"
)

print()

print(
    f"Structure correlation : "
    f"{corr:.6f}"
)

print(
    f"P-value               : "
    f"{pval:.6f}"
)

# ============================================================
# TOP HOTSPOTS
# ============================================================

print("\n======================================")
print("TOP CORRECT-KEY HOTSPOTS")
print("======================================")

indices = np.argsort(
    mean_correct.flatten()
)[::-1]

for rank, idx in enumerate(indices[:10]):

    word = idx // 16
    bit = idx % 16
    val = mean_correct[word, bit]

    print(
        f"Rank {rank+1}: "
        f"Word={word}, "
        f"Bit={bit}, "
        f"Value={val:.6f}"
    )

print("\n======================================")
print("TOP WRONG-KEY HOTSPOTS")
print("======================================")

indices = np.argsort(
    mean_wrong.flatten()
)[::-1]

for rank, idx in enumerate(indices[:10]):

    word = idx // 16
    bit = idx % 16
    val = mean_wrong[word, bit]

    print(
        f"Rank {rank+1}: "
        f"Word={word}, "
        f"Bit={bit}, "
        f"Value={val:.6f}"
    )

# ============================================================
# HEATMAPS
# ============================================================

fig, axs = plt.subplots(
    1,
    2,
    figsize=(14, 5)
)

# ------------------------------------------------------------
# CORRECT KEY
# ------------------------------------------------------------

im0 = axs[0].imshow(
    mean_correct,
    cmap='hot',
    aspect='auto'
)

axs[0].set_title(
    "CORRECT KEY Attribution"
)

axs[0].set_xlabel(
    "Bit Position"
)

axs[0].set_ylabel(
    "Ciphertext Word"
)

axs[0].set_yticks([0,1,2,3])

axs[0].set_yticklabels([
    'ct0a',
    'ct1a',
    'ct0b',
    'ct1b'
])

plt.colorbar(
    im0,
    ax=axs[0]
)

# ------------------------------------------------------------
# WRONG KEY
# ------------------------------------------------------------

im1 = axs[1].imshow(
    mean_wrong,
    cmap='hot',
    aspect='auto'
)

axs[1].set_title(
    "WRONG KEY Attribution"
)

axs[1].set_xlabel(
    "Bit Position"
)

axs[1].set_ylabel(
    "Ciphertext Word"
)

axs[1].set_yticks([0,1,2,3])

axs[1].set_yticklabels([
    'ct0a',
    'ct1a',
    'ct0b',
    'ct1b'
])

plt.colorbar(
    im1,
    ax=axs[1]
)

plt.suptitle(
    "Correct vs Wrong Key Structural Analysis"
)

plt.tight_layout()

plt.savefig(
    "outputs/wrong_key_vs_correct_key.png"
)

# ============================================================
# ENTROPY COMPARISON
# ============================================================

plt.figure(figsize=(6,5))

plt.bar(
    ['CORRECT_KEY', 'WRONG_KEY'],
    [entropy_correct, entropy_wrong]
)

plt.ylabel(
    "Attribution Entropy"
)

plt.title(
    "Structural Entropy Comparison"
)

plt.savefig(
    "outputs/wrong_key_entropy.png"
)

# ============================================================
# CONCENTRATION COMPARISON
# ============================================================

plt.figure(figsize=(6,5))

plt.bar(
    ['CORRECT_KEY', 'WRONG_KEY'],
    [max_correct, max_wrong]
)

plt.ylabel(
    "Maximum Attribution"
)

plt.title(
    "Attribution Concentration"
)

plt.savefig(
    "outputs/wrong_key_concentration.png"
)

# ============================================================
# SAVE ATTRIBUTIONS
# ============================================================

np.save(
    "outputs/correct_key_attr.npy",
    mean_correct
)

np.save(
    "outputs/wrong_key_attr.npy",
    mean_wrong
)

# ============================================================
# FINAL
# ============================================================

print("\n[+] Wrong-key structure validation complete.")