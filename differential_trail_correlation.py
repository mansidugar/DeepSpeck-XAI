import matplotlib
matplotlib.use('Agg')

import speck as sp
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

from tensorflow.keras.models import model_from_json
from scipy.stats import pearsonr

# ============================================================
# CONFIG
# ============================================================

NUM_SAMPLES = 3000
NUM_EXPLANATIONS = 400
ROUNDS = 7
IG_STEPS = 20

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
# GENERATE DATA
# ============================================================

X, Y = sp.make_train_data(
    NUM_SAMPLES,
    ROUNDS
)

print("[+] Dataset generated.")

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
    steps=20
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
# COMPUTE ATTRIBUTIONS
# ============================================================

print("[+] Computing Integrated Gradients...")

all_attr = []

limit = min(
    NUM_EXPLANATIONS,
    len(X_correct)
)

for i in range(limit):

    sample = X_correct[i]

    ig = integrated_gradients(
        model,
        sample,
        baseline,
        steps=IG_STEPS
    )

    all_attr.append(
        np.abs(ig)
    )

all_attr = np.array(all_attr)

mean_attr = np.mean(
    all_attr,
    axis=0
)

attr_heatmap = mean_attr.reshape(4,16)

print("[+] Attribution computation complete.")

# ============================================================
# DIFFERENTIAL ACTIVITY ANALYSIS
# ============================================================

print("[+] Computing differential activity...")

diff_activity = np.zeros(16)

for i in range(limit):

    sample = X_correct[i]

    # --------------------------------------------------------
    # Recover ciphertext words
    # --------------------------------------------------------

    ct0a = sample[0:16]
    ct1a = sample[16:32]
    ct0b = sample[32:48]
    ct1b = sample[48:64]

    # --------------------------------------------------------
    # Compute XOR differential
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
    # Aggregate bit activity
    # --------------------------------------------------------

    diff_activity += delta0
    diff_activity += delta1

# normalize

diff_activity = (
    diff_activity /
    (2 * limit)
)

print("[+] Differential activity computed.")

# ============================================================
# MAP ATTRIBUTION TO BIT POSITIONS
# ============================================================

bit_attr = np.zeros(16)

for word in range(4):

    for bit in range(16):

        idx = word * 16 + bit

        bit_attr[bit] += mean_attr[idx]

# normalize

bit_attr = (
    bit_attr /
    np.max(bit_attr)
)

diff_activity_norm = (
    diff_activity /
    np.max(diff_activity)
)

# ============================================================
# CORRELATION
# ============================================================

corr, pval = pearsonr(
    diff_activity_norm,
    bit_attr
)

print("\n===================================")
print("DIFFERENTIAL TRAIL CORRELATION")
print("===================================")

print(
    f"Pearson correlation : {corr:.6f}"
)

print(
    f"P-value             : {pval:.6f}"
)

# ============================================================
# VISUALIZATION
# ============================================================

# ------------------------------------------------------------
# ATTRIBUTION HEATMAP
# ------------------------------------------------------------

plt.figure(figsize=(12,4))

plt.imshow(
    attr_heatmap,
    cmap='hot',
    aspect='auto'
)

plt.colorbar(
    label='Mean Absolute Attribution'
)

plt.xlabel('Bit Position')
plt.ylabel('Ciphertext Word')

plt.yticks(
    [0,1,2,3],
    ['ct0a','ct1a','ct0b','ct1b']
)

plt.title(
    'Integrated Gradients Attribution'
)

plt.tight_layout()

plt.savefig(
    'diff_attr_heatmap.png',
    dpi=300
)

# ------------------------------------------------------------
# DIFFERENTIAL ACTIVITY VS ATTRIBUTION
# ------------------------------------------------------------

plt.figure(figsize=(10,5))

plt.plot(
    range(16),
    diff_activity_norm,
    marker='o',
    label='Differential Activity'
)

plt.plot(
    range(16),
    bit_attr,
    marker='s',
    label='Attribution Magnitude'
)

plt.xlabel('Bit Position')

plt.ylabel('Normalized Magnitude')

plt.title(
    'Differential Activity vs Attribution'
)

plt.legend()

plt.grid(True)

plt.tight_layout()

plt.savefig(
    'diff_vs_attr.png',
    dpi=300
)

# ------------------------------------------------------------
# SCATTER PLOT
# ------------------------------------------------------------

plt.figure(figsize=(6,6))

plt.scatter(
    diff_activity_norm,
    bit_attr
)

for i in range(16):

    plt.text(
        diff_activity_norm[i],
        bit_attr[i],
        str(i)
    )

plt.xlabel(
    'Differential Activity'
)

plt.ylabel(
    'Attribution Magnitude'
)

plt.title(
    f'Differential/Attribution Correlation\n'
    f'r = {corr:.4f}'
)

plt.grid(True)

plt.tight_layout()

plt.savefig(
    'diff_scatter.png',
    dpi=300
)

# ============================================================
# BIT-LEVEL ANALYSIS
# ============================================================

print("\n===================================")
print("BIT-LEVEL DIFFERENTIAL ANALYSIS")
print("===================================")

for bit in range(16):

    print(
        f"Bit {bit:02d} | "
        f"DiffActivity={diff_activity_norm[bit]:.4f} | "
        f"Attr={bit_attr[bit]:.4f}"
    )

# ============================================================
# SAVE NUMERICAL RESULTS
# ============================================================

np.save(
    'diff_activity.npy',
    diff_activity
)

np.save(
    'diff_attr.npy',
    bit_attr
)

print("\n[+] Differential correlation experiment complete.")