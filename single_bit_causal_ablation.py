import numpy as np
import tensorflow as tf
from tensorflow.keras.models import model_from_json
from sklearn.metrics import accuracy_score
from scipy.stats import pearsonr
import matplotlib.pyplot as plt

from speck import make_train_data


# ============================================================
# CONFIG
# ============================================================

NUM_SAMPLES = 12000
ROUNDS = 7

MODEL_WEIGHTS = "net7_small.h5"
MODEL_JSON = "single_block_resnet.json"

# ============================================================
# LOAD MODEL
# ============================================================

print("[+] Loading model.")

with open(MODEL_JSON, "r") as f:
    json_model = f.read()

model = model_from_json(json_model)
model.load_weights(MODEL_WEIGHTS)

print("[+] Model loaded successfully.")

# ============================================================
# GENERATE DATA
# ============================================================

print("[+] Generating evaluation dataset.")

X, Y = make_train_data(NUM_SAMPLES, ROUNDS)

preds = model.predict(X, batch_size=1024).flatten()
pred_labels = (preds > 0.5).astype(np.uint8)

correct_mask = (pred_labels == Y)

X = X[correct_mask]
Y = Y[correct_mask]

print(f"[+] Correctly classified samples: {len(X)}")

# ============================================================
# BASELINE ACCURACY
# ============================================================

baseline_preds = model.predict(
    X,
    batch_size=1024
).flatten()

baseline_labels = (baseline_preds > 0.5).astype(np.uint8)

baseline_acc = accuracy_score(Y, baseline_labels)

print(f"[+] Baseline Accuracy: {baseline_acc:.6f}")

# ============================================================
# INTEGRATED GRADIENTS
# ============================================================

print("[+] Computing Integrated Gradients.")

X_tensor = tf.convert_to_tensor(X[:2000], dtype=tf.float32)

baseline = tf.zeros_like(X_tensor)

steps = 50

scaled_inputs = [
    baseline + (float(i) / steps) * (X_tensor - baseline)
    for i in range(steps + 1)
]

grads = []

for scaled in scaled_inputs:

    with tf.GradientTape() as tape:

        tape.watch(scaled)

        preds = model(scaled, training=False)

    grad = tape.gradient(preds, scaled)

    grads.append(grad)

avg_grads = tf.reduce_mean(tf.stack(grads), axis=0)

integrated_grads = (
    (X_tensor - baseline) * avg_grads
).numpy()

ig_attr = np.mean(np.abs(integrated_grads), axis=0)

# ============================================================
# SINGLE-BIT ABLATION
# ============================================================

print("[+] Performing single-bit ablations.")

bit_importance = []

for bit in range(16):

    X_modified = X.copy()

    # Mask same bit across all ciphertext words

    X_modified[:, bit] = 0
    X_modified[:, 16 + bit] = 0
    X_modified[:, 32 + bit] = 0
    X_modified[:, 48 + bit] = 0

    preds_mod = model.predict(
        X_modified,
        batch_size=1024,
        verbose=0
    ).flatten()

    labels_mod = (preds_mod > 0.5).astype(np.uint8)

    acc_mod = accuracy_score(Y, labels_mod)

    drop = baseline_acc - acc_mod

    bit_importance.append(drop)

    print(
        f"Bit {bit:02d} | "
        f"Accuracy={acc_mod:.6f} | "
        f"Drop={drop:.6f}"
    )

bit_importance = np.array(bit_importance)

# ============================================================
# IG BIT AGGREGATION
# ============================================================

print("[+] Aggregating IG attribution by bit.")

ig_bit_scores = np.zeros(16)

for bit in range(16):

    ig_bit_scores[bit] = (
        ig_attr[bit]
        + ig_attr[16 + bit]
        + ig_attr[32 + bit]
        + ig_attr[48 + bit]
    )

# Normalize

ig_bit_scores = ig_bit_scores / np.max(ig_bit_scores)

bit_importance_norm = (
    bit_importance / np.max(bit_importance)
)

# ============================================================
# CORRELATION ANALYSIS
# ============================================================

corr, pval = pearsonr(
    ig_bit_scores,
    bit_importance_norm
)

print("\n======================================")
print("CAUSAL FAITHFULNESS ANALYSIS")
print("======================================")

print(f"IG vs Causal Correlation : {corr:.6f}")
print(f"P-value                  : {pval:.6f}")

# ============================================================
# TOP CAUSAL BITS
# ============================================================

print("\n======================================")
print("TOP CAUSALLY IMPORTANT BITS")
print("======================================")

top_bits = np.argsort(bit_importance)[::-1]

for i in range(10):

    bit = top_bits[i]

    print(
        f"Rank {i+1}: "
        f"Bit={bit}, "
        f"CausalDrop={bit_importance[bit]:.6f}, "
        f"IG={ig_bit_scores[bit]:.6f}"
    )

# ============================================================
# PLOT: IG vs CAUSAL
# ============================================================

plt.figure(figsize=(10, 6))

plt.plot(
    range(16),
    ig_bit_scores,
    marker='o',
    label="Integrated Gradients"
)

plt.plot(
    range(16),
    bit_importance_norm,
    marker='s',
    label="Causal Importance"
)

plt.xlabel("Bit Position")
plt.ylabel("Normalized Importance")
plt.title("IG Attribution vs True Causal Importance")

plt.legend()

plt.tight_layout()

plt.savefig("ig_vs_causal_importance.png")

plt.close()

# ============================================================
# SCATTER PLOT
# ============================================================

plt.figure(figsize=(7, 7))

plt.scatter(
    ig_bit_scores,
    bit_importance_norm
)

for i in range(16):

    plt.text(
        ig_bit_scores[i],
        bit_importance_norm[i],
        str(i)
    )

plt.xlabel("Integrated Gradients Importance")
plt.ylabel("Causal Importance")

plt.title(
    f"IG Faithfulness Correlation\n"
    f"r = {corr:.4f}"
)

plt.tight_layout()

plt.savefig("ig_causal_scatter.png")

plt.close()

# ============================================================
# BAR PLOT
# ============================================================

plt.figure(figsize=(12, 6))

x = np.arange(16)

width = 0.35

plt.bar(
    x - width/2,
    ig_bit_scores,
    width,
    label="Integrated Gradients"
)

plt.bar(
    x + width/2,
    bit_importance_norm,
    width,
    label="Causal Importance"
)

plt.xlabel("Bit Position")
plt.ylabel("Normalized Score")

plt.title("Per-Bit IG vs Causal Importance")

plt.legend()

plt.tight_layout()

plt.savefig("bitwise_causal_importance.png")

plt.close()

print("\n[+] Single-bit causal ablation complete.")