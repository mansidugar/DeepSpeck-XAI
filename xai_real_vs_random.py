import matplotlib
matplotlib.use('Agg')

import speck as sp
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

from tensorflow.keras.models import model_from_json

# ============================================================
# REAL VS RANDOM ATTRIBUTION ANALYSIS
# ============================================================

NUM_SAMPLES = 1000
NUM_EXPLANATIONS = 150
STEPS = 20
ROUNDS = 7

# ------------------------------------------------------------
# LOAD MODEL
# ------------------------------------------------------------

json_file = open('single_block_resnet.json', 'r')
json_model = json_file.read()

model = model_from_json(json_model)

model.load_weights('net7_small.h5')

print("[+] Loaded 7-round distinguisher.")

# ------------------------------------------------------------
# GENERATE DATA
# ------------------------------------------------------------

X, Y = sp.make_train_data(NUM_SAMPLES, ROUNDS)

print("[+] Dataset generated.")

# ------------------------------------------------------------
# PREDICTIONS
# ------------------------------------------------------------

pred = model.predict(X, batch_size=5000).flatten()

pred_bin = (pred > 0.5).astype(np.uint8)

correct_idx = np.where(pred_bin == Y)[0]

X_correct = X[correct_idx]
Y_correct = Y[correct_idx]

print("[+] Correctly classified:",
      len(X_correct))

# ------------------------------------------------------------
# SPLIT REAL vs RANDOM
# ------------------------------------------------------------

X_real = X_correct[Y_correct == 1]
X_random = X_correct[Y_correct == 0]

print("[+] REAL samples:",
      len(X_real))

print("[+] RANDOM samples:",
      len(X_random))

# ------------------------------------------------------------
# BASELINE
# ------------------------------------------------------------

baseline = np.zeros((64,), dtype=np.float32)

# ------------------------------------------------------------
# INTEGRATED GRADIENTS
# ------------------------------------------------------------

def integrated_gradients(model,
                         sample,
                         baseline,
                         steps=20):

    sample = tf.cast(sample, tf.float32)

    baseline = tf.cast(baseline,
                       tf.float32)

    interpolated = []

    for alpha in np.linspace(0.0,
                             1.0,
                             steps):

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

# ------------------------------------------------------------
# ATTRIBUTION COMPUTATION
# ------------------------------------------------------------

def compute_group_attributions(data,
                               label_name):

    attrs = []

    limit = min(
        NUM_EXPLANATIONS,
        len(data)
    )

    print(f"\n[+] Processing {label_name}")

    for i in range(limit):

        sample = data[i]

        ig = integrated_gradients(
            model,
            sample,
            baseline,
            steps=STEPS
        )

        attrs.append(ig)

    attrs = np.array(attrs)

    mean_attr = np.mean(
        np.abs(attrs),
        axis=0
    )

    variance_attr = np.var(
        attrs,
        axis=0
    )

    entropy = attribution_entropy(
        mean_attr
    )

    return (
        attrs,
        mean_attr,
        variance_attr,
        entropy
    )

# ------------------------------------------------------------
# ENTROPY
# ------------------------------------------------------------

def attribution_entropy(attr):

    p = np.abs(attr)

    p = p / np.sum(p)

    eps = 1e-10

    entropy = -np.sum(
        p * np.log2(p + eps)
    )

    return entropy

# ------------------------------------------------------------
# RUN ANALYSIS
# ------------------------------------------------------------

(
    real_attrs,
    real_mean,
    real_var,
    real_entropy

) = compute_group_attributions(
    X_real,
    "REAL PAIRS"
)

(
    random_attrs,
    random_mean,
    random_var,
    random_entropy

) = compute_group_attributions(
    X_random,
    "RANDOM PAIRS"
)

# ------------------------------------------------------------
# RESHAPE
# ------------------------------------------------------------

real_heatmap = real_mean.reshape(4,16)

random_heatmap = random_mean.reshape(4,16)

real_var_map = real_var.reshape(4,16)

random_var_map = random_var.reshape(4,16)

# ============================================================
# COMPARISON PLOT
# ============================================================

fig, axes = plt.subplots(
    1,
    2,
    figsize=(16,5)
)

# ------------------------------------------------------------
# REAL
# ------------------------------------------------------------

im1 = axes[0].imshow(
    real_heatmap,
    cmap='hot',
    aspect='auto'
)

axes[0].set_title(
    'REAL Ciphertext Pairs'
)

axes[0].set_xlabel(
    'Bit Position'
)

axes[0].set_ylabel(
    'Ciphertext Word'
)

axes[0].set_yticks(
    [0,1,2,3]
)

axes[0].set_yticklabels(
    ['ct0a','ct1a',
     'ct0b','ct1b']
)

fig.colorbar(
    im1,
    ax=axes[0]
)

# ------------------------------------------------------------
# RANDOM
# ------------------------------------------------------------

im2 = axes[1].imshow(
    random_heatmap,
    cmap='hot',
    aspect='auto'
)

axes[1].set_title(
    'RANDOM Ciphertext Pairs'
)

axes[1].set_xlabel(
    'Bit Position'
)

axes[1].set_ylabel(
    'Ciphertext Word'
)

axes[1].set_yticks(
    [0,1,2,3]
)

axes[1].set_yticklabels(
    ['ct0a','ct1a',
     'ct0b','ct1b']
)

fig.colorbar(
    im2,
    ax=axes[1]
)

plt.suptitle(
    'REAL vs RANDOM Attribution Structure'
)

plt.tight_layout()

plt.savefig(
    'real_vs_random_comparison.png',
    dpi=300
)

# ============================================================
# VARIANCE COMPARISON
# ============================================================

fig, axes = plt.subplots(
    1,
    2,
    figsize=(16,5)
)

# REAL VARIANCE

im1 = axes[0].imshow(
    real_var_map,
    cmap='viridis',
    aspect='auto'
)

axes[0].set_title(
    'REAL Pair Variance'
)

axes[0].set_xlabel(
    'Bit Position'
)

axes[0].set_ylabel(
    'Ciphertext Word'
)

axes[0].set_yticks(
    [0,1,2,3]
)

axes[0].set_yticklabels(
    ['ct0a','ct1a',
     'ct0b','ct1b']
)

fig.colorbar(
    im1,
    ax=axes[0]
)

# RANDOM VARIANCE

im2 = axes[1].imshow(
    random_var_map,
    cmap='viridis',
    aspect='auto'
)

axes[1].set_title(
    'RANDOM Pair Variance'
)

axes[1].set_xlabel(
    'Bit Position'
)

axes[1].set_ylabel(
    'Ciphertext Word'
)

axes[1].set_yticks(
    [0,1,2,3]
)

axes[1].set_yticklabels(
    ['ct0a','ct1a',
     'ct0b','ct1b']
)

fig.colorbar(
    im2,
    ax=axes[1]
)

plt.suptitle(
    'REAL vs RANDOM Attribution Variance'
)

plt.tight_layout()

plt.savefig(
    'real_vs_random_variance.png',
    dpi=300
)

# ============================================================
# NUMERICAL ANALYSIS
# ============================================================

print("\n===================================")
print("ENTROPY ANALYSIS")
print("===================================")

print(f"REAL entropy   : {real_entropy:.6f}")
print(f"RANDOM entropy : {random_entropy:.6f}")

# ------------------------------------------------------------
# CONCENTRATION SCORE
# ------------------------------------------------------------

real_max = np.max(real_mean)
random_max = np.max(random_mean)

print("\n===================================")
print("CONCENTRATION ANALYSIS")
print("===================================")

print(f"REAL max attribution   : {real_max:.6f}")
print(f"RANDOM max attribution : {random_max:.6f}")

# ------------------------------------------------------------
# TOP BITS
# ------------------------------------------------------------

print("\n===================================")
print("TOP REAL ATTRIBUTED BITS")
print("===================================")

real_idx = np.argsort(real_mean)[::-1]

for i in range(10):

    idx = real_idx[i]

    word = idx // 16
    bit = idx % 16

    print(
        f"Rank {i+1}: "
        f"Word={word}, "
        f"Bit={bit}, "
        f"Value={real_mean[idx]:.6f}"
    )

print("\n===================================")
print("TOP RANDOM ATTRIBUTED BITS")
print("===================================")

random_idx = np.argsort(random_mean)[::-1]

for i in range(10):

    idx = random_idx[i]

    word = idx // 16
    bit = idx % 16

    print(
        f"Rank {i+1}: "
        f"Word={word}, "
        f"Bit={bit}, "
        f"Value={random_mean[idx]:.6f}"
    )

# ============================================================
# SAVE RAW DATA
# ============================================================

np.save(
    'real_mean_attr.npy',
    real_mean
)

np.save(
    'random_mean_attr.npy',
    random_mean
)

np.save(
    'real_variance.npy',
    real_var
)

np.save(
    'random_variance.npy',
    random_var
)

print("\n[+] Experiment complete.")