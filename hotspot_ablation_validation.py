import numpy as np
import tensorflow as tf
from tensorflow.keras.models import model_from_json
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt

from speck import make_train_data


# ============================================================
# CONFIG
# ============================================================

NUM_SAMPLES = 12000
ROUNDS = 7

MODEL_WEIGHTS = "net7_small.h5"
MODEL_JSON = "single_block_resnet.json"

# IMPORTANT HOTSPOTS DISCOVERED EARLIER
HOTSPOT_BITS = [3, 4, 10, 11, 12, 13]

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
# HELPER FUNCTIONS
# ============================================================

def zero_bits(data, bit_positions):
    """
    Zero out selected bit positions
    across all 4 ciphertext words.
    """

    modified = data.copy()

    for bit in bit_positions:

        modified[:, bit] = 0
        modified[:, 16 + bit] = 0
        modified[:, 32 + bit] = 0
        modified[:, 48 + bit] = 0

    return modified


def random_mask(data, num_bits):
    """
    Randomly mask same number of bits
    as hotspot experiment.
    """

    rand_bits = np.random.choice(
        np.arange(16),
        size=num_bits,
        replace=False
    )

    return zero_bits(data, rand_bits), rand_bits


# ============================================================
# BASELINE PERFORMANCE
# ============================================================

print("[+] Evaluating baseline accuracy.")

baseline_preds = model.predict(
    X,
    batch_size=1024
).flatten()

baseline_labels = (baseline_preds > 0.5).astype(np.uint8)

baseline_acc = accuracy_score(Y, baseline_labels)

print(f"[+] Baseline Accuracy: {baseline_acc:.6f}")


# ============================================================
# HOTSPOT ABLATION
# ============================================================

print("[+] Performing hotspot ablation.")

X_hotspot = zero_bits(X, HOTSPOT_BITS)

hotspot_preds = model.predict(
    X_hotspot,
    batch_size=1024
).flatten()

hotspot_labels = (hotspot_preds > 0.5).astype(np.uint8)

hotspot_acc = accuracy_score(Y, hotspot_labels)

print(f"[+] Hotspot Ablation Accuracy: {hotspot_acc:.6f}")


# ============================================================
# RANDOM ABLATION
# ============================================================

print("[+] Performing random ablation.")

X_random, rand_bits = random_mask(
    X,
    len(HOTSPOT_BITS)
)

random_preds = model.predict(
    X_random,
    batch_size=1024
).flatten()

random_labels = (random_preds > 0.5).astype(np.uint8)

random_acc = accuracy_score(Y, random_labels)

print(f"[+] Random Ablation Accuracy: {random_acc:.6f}")

print(f"[+] Random bits used: {rand_bits}")


# ============================================================
# PERFORMANCE DROP
# ============================================================

baseline_drop_hotspot = baseline_acc - hotspot_acc
baseline_drop_random = baseline_acc - random_acc

print("\n======================================")
print("HOTSPOT ABLATION ANALYSIS")
print("======================================")

print(f"Baseline Accuracy        : {baseline_acc:.6f}")
print(f"Hotspot Mask Accuracy    : {hotspot_acc:.6f}")
print(f"Random Mask Accuracy     : {random_acc:.6f}")

print()
print(f"Hotspot Performance Drop : {baseline_drop_hotspot:.6f}")
print(f"Random Performance Drop  : {baseline_drop_random:.6f}")


# ============================================================
# PLOTS
# ============================================================

plt.figure(figsize=(8, 5))

labels = [
    "BASELINE",
    "HOTSPOT_MASK",
    "RANDOM_MASK"
]

values = [
    baseline_acc,
    hotspot_acc,
    random_acc
]

plt.bar(labels, values)

plt.ylabel("Accuracy")
plt.title("Hotspot Ablation Validation")

plt.tight_layout()
plt.savefig("hotspot_ablation_accuracy.png")
plt.close()


# ============================================================
# CONFIDENCE ANALYSIS
# ============================================================

baseline_conf = np.mean(np.abs(baseline_preds - 0.5))
hotspot_conf = np.mean(np.abs(hotspot_preds - 0.5))
random_conf = np.mean(np.abs(random_preds - 0.5))

plt.figure(figsize=(8, 5))

conf_labels = [
    "BASELINE",
    "HOTSPOT_MASK",
    "RANDOM_MASK"
]

conf_values = [
    baseline_conf,
    hotspot_conf,
    random_conf
]

plt.bar(conf_labels, conf_values)

plt.ylabel("Mean Confidence")
plt.title("Prediction Confidence After Ablation")

plt.tight_layout()
plt.savefig("hotspot_ablation_confidence.png")
plt.close()


print("\n[+] Hotspot ablation experiment complete.")