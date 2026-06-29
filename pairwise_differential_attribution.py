import matplotlib
matplotlib.use('Agg')

import speck as sp
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

from tensorflow.keras.models import model_from_json
from scipy.stats import entropy

# ============================================================
# CONFIGURATION
# ============================================================

NUM_SAMPLES = 4000
NUM_EXPLANATIONS = 500
ROUNDS = 7
IG_STEPS = 25

# ============================================================
# LOAD FIXED DATASET OR CREATE ONE
# ============================================================

try:

    X = np.load("X_fixed_7r.npy")
    Y = np.load("Y_fixed_7r.npy")

    print("[+] Loaded fixed evaluation corpus.")

except:

    print("[+] Creating fixed evaluation corpus.")

    X, Y = sp.make_train_data(
        NUM_SAMPLES,
        ROUNDS
    )

    np.save("X_fixed_7r.npy", X)
    np.save("Y_fixed_7r.npy", Y)

# ============================================================
# LOAD MODEL
# ============================================================

json_file = open(
    'single_block_resnet.json',
    'r'
)

json_model = json_file.read()

model = model_from_json(
    json_model
)

model.load_weights(
    'net7_small.h5'
)

print("[+] Loaded 7-round distinguisher.")

# ============================================================
# FILTER CORRECT CLASSIFICATIONS
# ============================================================

pred = model.predict(
    X,
    batch_size=5000
).flatten()

pred_bin = (
    pred > 0.5
).astype(np.uint8)

correct_idx = np.where(
    pred_bin == Y
)[0]

X_correct = X[correct_idx]

print(
    "[+] Correctly classified:",
    len(X_correct)
)

# ============================================================
# INTEGRATED GRADIENTS
# ============================================================

baseline = np.zeros(
    (64,),
    dtype=np.float32
)

def integrated_gradients(
    model,
    sample,
    baseline,
    steps=25
):

    sample = tf.cast(
        sample,
        tf.float32
    )

    baseline = tf.cast(
        baseline,
        tf.float32
    )

    interpolated = []

    for alpha in np.linspace(
        0,
        1,
        steps
    ):

        x = baseline + alpha * (
            sample - baseline
        )

        interpolated.append(x)

    interpolated = tf.convert_to_tensor(
        interpolated
    )

    with tf.GradientTape() as tape:

        tape.watch(interpolated)

        pred = model(interpolated)

    grads = tape.gradient(
        pred,
        interpolated
    )

    avg_grads = tf.reduce_mean(
        grads,
        axis=0
    )

    ig = (
        sample - baseline
    ) * avg_grads

    return ig.numpy()

# ============================================================
# ATTRIBUTION ANALYSIS
# ============================================================

print("[+] Computing Integrated Gradients...")

raw_attr = []

diff_attr = []

limit = min(
    NUM_EXPLANATIONS,
    len(X_correct)
)

for i in range(limit):

    sample = X_correct[i]

    # --------------------------------------------------------
    # Compute IG on ORIGINAL representation
    # --------------------------------------------------------

    ig = integrated_gradients(
        model,
        sample,
        baseline,
        steps=IG_STEPS
    )

    ig = np.abs(ig)

    raw_attr.append(ig)

    # --------------------------------------------------------
    # Recover ciphertext pair
    # --------------------------------------------------------

    ct0a = sample[0:16]
    ct1a = sample[16:32]

    ct0b = sample[32:48]
    ct1b = sample[48:64]

    # --------------------------------------------------------
    # Compute XOR differential structure
    # --------------------------------------------------------

    delta0 = np.bitwise_xor(
        ct0a.astype(np.uint8),
        ct0b.astype(np.uint8)
    )

    delta1 = np.bitwise_xor(
        ct1a.astype(np.uint8),
        ct1b.astype(np.uint8)
    )

    # --------------------------------------------------------
    # Project attribution into differential space
    # --------------------------------------------------------

    diff_projection = np.zeros(64)

    for bit in range(16):

        # pairwise attribution aggregation

        diff_projection[bit] = (
            ig[bit] +
            ig[32 + bit]
        ) * delta0[bit]

        diff_projection[16 + bit] = (
            ig[16 + bit] +
            ig[48 + bit]
        ) * delta1[bit]

        # mirror for visualization symmetry

        diff_projection[32 + bit] = (
            diff_projection[bit]
        )

        diff_projection[48 + bit] = (
            diff_projection[16 + bit]
        )

    diff_attr.append(
        diff_projection
    )

raw_attr = np.array(raw_attr)

diff_attr = np.array(diff_attr)

# ============================================================
# AGGREGATE
# ============================================================

raw_mean = np.mean(
    raw_attr,
    axis=0
)

diff_mean = np.mean(
    diff_attr,
    axis=0
)

raw_heatmap = raw_mean.reshape(4,16)

diff_heatmap = diff_mean.reshape(4,16)

# ============================================================
# ENTROPY ANALYSIS
# ============================================================

raw_norm = raw_mean / np.sum(raw_mean)

diff_norm = diff_mean / np.sum(diff_mean)

raw_entropy = entropy(raw_norm)

diff_entropy = entropy(diff_norm)

# ============================================================
# CONCENTRATION ANALYSIS
# ============================================================

raw_max = np.max(raw_mean)

diff_max = np.max(diff_mean)

# ============================================================
# DIFFERENTIAL STABILITY
# ============================================================

diff_variance = np.var(
    diff_attr,
    axis=0
)

stability = diff_mean / (
    np.sqrt(diff_variance) + 1e-8
)

stability_heatmap = stability.reshape(4,16)

# ============================================================
# VISUALIZATION
# ============================================================

fig, axs = plt.subplots(
    1,
    2,
    figsize=(18,5)
)

# ------------------------------------------------------------
# RAW ATTRIBUTION
# ------------------------------------------------------------

im1 = axs[0].imshow(
    raw_heatmap,
    cmap='hot',
    aspect='auto'
)

axs[0].set_title(
    'RAW Ciphertext Attribution'
)

axs[0].set_xlabel(
    'Bit Position'
)

axs[0].set_ylabel(
    'Ciphertext Word'
)

axs[0].set_yticks(
    [0,1,2,3]
)

axs[0].set_yticklabels(
    ['ct0a','ct1a','ct0b','ct1b']
)

fig.colorbar(
    im1,
    ax=axs[0]
)

# ------------------------------------------------------------
# DIFFERENTIAL ATTRIBUTION
# ------------------------------------------------------------

im2 = axs[1].imshow(
    diff_heatmap,
    cmap='hot',
    aspect='auto'
)

axs[1].set_title(
    'PROJECTED XOR Differential Attribution'
)

axs[1].set_xlabel(
    'Bit Position'
)

axs[1].set_ylabel(
    'Differential Word'
)

axs[1].set_yticks(
    [0,1,2,3]
)

axs[1].set_yticklabels(
    ['Δ0','Δ1','Δ0','Δ1']
)

fig.colorbar(
    im2,
    ax=axs[1]
)

plt.tight_layout()

plt.savefig(
    'raw_vs_projected_diff.png',
    dpi=300
)

# ============================================================
# STABILITY MAP
# ============================================================

plt.figure(figsize=(12,4))

plt.imshow(
    stability_heatmap,
    cmap='viridis',
    aspect='auto'
)

plt.colorbar(
    label='Differential Stability'
)

plt.xlabel('Bit Position')

plt.ylabel('Differential Word')

plt.yticks(
    [0,1,2,3],
    ['Δ0','Δ1','Δ0','Δ1']
)

plt.title(
    'Differential Attribution Stability'
)

plt.tight_layout()

plt.savefig(
    'diff_stability.png',
    dpi=300
)

# ============================================================
# ENTROPY COMPARISON
# ============================================================

plt.figure(figsize=(6,5))

plt.bar(
    ['RAW','PROJECTED_DIFF'],
    [raw_entropy, diff_entropy]
)

plt.ylabel(
    'Attribution Entropy'
)

plt.title(
    'Entropy Comparison'
)

plt.tight_layout()

plt.savefig(
    'entropy_compare.png',
    dpi=300
)

# ============================================================
# CONCENTRATION COMPARISON
# ============================================================

plt.figure(figsize=(6,5))

plt.bar(
    ['RAW','PROJECTED_DIFF'],
    [raw_max, diff_max]
)

plt.ylabel(
    'Maximum Attribution'
)

plt.title(
    'Attribution Concentration'
)

plt.tight_layout()

plt.savefig(
    'concentration_compare.png',
    dpi=300
)

# ============================================================
# TOP DIFFERENTIAL REGIONS
# ============================================================

print("\n===================================")
print("TOP DIFFERENTIAL REGIONS")
print("===================================")

indices = np.argsort(
    diff_mean
)[::-1]

for rank in range(15):

    idx = indices[rank]

    word = idx // 16
    bit = idx % 16

    print(
        f"Rank {rank+1}: "
        f"DiffWord={word}, "
        f"Bit={bit}, "
        f"Value={diff_mean[idx]:.6f}"
    )

# ============================================================
# FINAL ANALYSIS
# ============================================================

print("\n===================================")
print("ENTROPY ANALYSIS")
print("===================================")

print(
    f"RAW entropy               : {raw_entropy:.6f}"
)

print(
    f"PROJECTED DIFF entropy    : {diff_entropy:.6f}"
)

print("\n===================================")
print("CONCENTRATION ANALYSIS")
print("===================================")

print(
    f"RAW max attribution       : {raw_max:.6f}"
)

print(
    f"PROJECTED DIFF max attr   : {diff_max:.6f}"
)

# ============================================================
# SAVE RESULTS
# ============================================================

np.save(
    'projected_diff_attr.npy',
    diff_mean
)

np.save(
    'raw_attr.npy',
    raw_mean
)

np.save(
    'diff_stability.npy',
    stability
)

print("\n[+] Pairwise differential attribution complete.")