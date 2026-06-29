import matplotlib
matplotlib.use('Agg')

import speck as sp
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

from tensorflow.keras.models import model_from_json
from scipy.stats import pearsonr

# ============================================================
# CARRY CORRELATION ANALYSIS
# ============================================================

NUM_SAMPLES = 2000
NUM_EXPLANATIONS = 300
STEPS = 20
ROUNDS = 7

# ============================================================
# LOAD MODEL
# ============================================================

json_file = open('single_block_resnet.json', 'r')
json_model = json_file.read()

model = model_from_json(json_model)

model.load_weights('net7_small.h5')

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

pred_bin = (pred > 0.5).astype(np.uint8)

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
        0.0,
        1.0,
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

all_attr = []

print("[+] Computing attributions...")

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
        steps=STEPS
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

print("[+] Attribution analysis complete.")

# ============================================================
# CARRY ANALYSIS
# ============================================================

# ------------------------------------------------------------
# Estimate carry activity for modular addition
# ------------------------------------------------------------

def compute_carry_bits(a, b):

    carries = np.zeros(
        16,
        dtype=np.uint8
    )

    carry = 0

    for bit in range(16):

        abit = (a >> bit) & 1
        bbit = (b >> bit) & 1

        carry_out = (
            (abit & bbit) |
            (abit & carry) |
            (bbit & carry)
        )

        carries[bit] = carry_out

        carry = carry_out

    return carries

# ============================================================
# GENERATE CARRY STATISTICS
# ============================================================

print("[+] Computing carry statistics...")

carry_counts = np.zeros(16)

for _ in range(NUM_SAMPLES):

    a = np.random.randint(0, 2**16)
    b = np.random.randint(0, 2**16)

    carries = compute_carry_bits(a,b)

    carry_counts += carries

carry_freq = carry_counts / NUM_SAMPLES

print("[+] Carry frequencies computed.")

# ============================================================
# MAP ATTRIBUTIONS TO BIT POSITIONS
# ============================================================

# aggregate attribution over words

bit_attr = np.zeros(16)

for word in range(4):

    for bit in range(16):

        idx = word * 16 + bit

        bit_attr[bit] += mean_attr[idx]

# normalize

bit_attr = bit_attr / np.max(bit_attr)

carry_freq_norm = (
    carry_freq /
    np.max(carry_freq)
)

# ============================================================
# CORRELATION
# ============================================================

corr, pval = pearsonr(
    carry_freq_norm,
    bit_attr
)

print("\n===================================")
print("CARRY CORRELATION ANALYSIS")
print("===================================")

print(f"Pearson correlation : {corr:.6f}")
print(f"P-value             : {pval:.6f}")

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
    'carry_attr_heatmap.png',
    dpi=300
)

# ------------------------------------------------------------
# CARRY FREQUENCY
# ------------------------------------------------------------

plt.figure(figsize=(10,5))

plt.plot(
    range(16),
    carry_freq_norm,
    marker='o',
    label='Carry Frequency'
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
    'Carry Frequency vs Attribution'
)

plt.legend()

plt.grid(True)

plt.tight_layout()

plt.savefig(
    'carry_vs_attr.png',
    dpi=300
)

# ------------------------------------------------------------
# SCATTER PLOT
# ------------------------------------------------------------

plt.figure(figsize=(6,6))

plt.scatter(
    carry_freq_norm,
    bit_attr
)

for i in range(16):

    plt.text(
        carry_freq_norm[i],
        bit_attr[i],
        str(i)
    )

plt.xlabel(
    'Carry Frequency'
)

plt.ylabel(
    'Attribution Magnitude'
)

plt.title(
    f'Carry/Attribution Correlation\n'
    f'r = {corr:.4f}'
)

plt.grid(True)

plt.tight_layout()

plt.savefig(
    'carry_scatter.png',
    dpi=300
)

# ============================================================
# TOP CARRY-ATTRIBUTED BITS
# ============================================================

print("\n===================================")
print("BIT-LEVEL ANALYSIS")
print("===================================")

for bit in range(16):

    print(
        f"Bit {bit:02d} | "
        f"CarryFreq={carry_freq_norm[bit]:.4f} | "
        f"Attr={bit_attr[bit]:.4f}"
    )

# ============================================================
# SAVE NUMERICAL RESULTS
# ============================================================

np.save(
    'carry_frequency.npy',
    carry_freq
)

np.save(
    'bit_attribution.npy',
    bit_attr
)

print("\n[+] Carry correlation experiment complete.")