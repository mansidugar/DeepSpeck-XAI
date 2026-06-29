# round_transport_analysis.py

import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import pearsonr
from tensorflow.keras.models import model_from_json
from speck import make_train_data

# ============================================================
# CONFIG
# ============================================================

ROUNDS = [5, 6, 7, 8]

MODEL_FILES = {
    5: "net5_small.h5",
    6: "net6_small.h5",
    7: "net7_small.h5",
    8: "net8_small.h5"
}

JSON_PATH = "single_block_resnet.json"

NUM_SAMPLES = 12000

# ROTR(7) transport hypothesis
ROTATION = 7

# ============================================================
# LOAD MODEL
# ============================================================

def load_speck_model(weight_path):

    with open(JSON_PATH, "r") as f:
        json_model = f.read()

    model = model_from_json(json_model)
    model.load_weights(weight_path)

    return model

# ============================================================
# GET CORRECTLY CLASSIFIED SAMPLES
# ============================================================

def get_correct_samples(model, X, Y):

    preds = model.predict(X, verbose=0).flatten()
    labels = (preds > 0.5).astype(np.uint8)

    idx = np.where(labels == Y)[0]

    return X[idx]

# ============================================================
# BASELINE ACCURACY
# ============================================================

def evaluate_accuracy(model, X):

    preds = model.predict(X, verbose=0).flatten()
    labels = (preds > 0.5).astype(np.uint8)

    return np.mean(labels == 1)

# ============================================================
# SINGLE BIT CAUSAL IMPORTANCE
# ============================================================

def compute_causal_importance(model, X):

    baseline = evaluate_accuracy(model, X)

    importance = np.zeros(16)

    for bit in range(16):

        X_mod = X.copy()

        for word in range(4):

            idx = word * 16 + bit

            X_mod[:, idx] = 0

        acc = evaluate_accuracy(model, X_mod)

        drop = baseline - acc

        importance[bit] = drop

    # normalize
    importance = importance / np.max(importance)

    return importance

# ============================================================
# ROTATE VECTOR
# ============================================================

def rotr(vec, r):

    return np.roll(vec, r)

# ============================================================
# MAIN
# ============================================================

all_round_importance = {}

print("\n========================================")
print("ROUND-TO-ROUND MOTIF TRANSPORT ANALYSIS")
print("========================================")

# ------------------------------------------------------------
# compute causal maps
# ------------------------------------------------------------

for r in ROUNDS:

    print(f"\n[+] ANALYZING {r} ROUNDS")

    model = load_speck_model(MODEL_FILES[r])

    X, Y = make_train_data(NUM_SAMPLES, r)

    X = X.astype(np.float32)

    X_real = X[Y == 1]

    X_real = get_correct_samples(model, X_real, np.ones(len(X_real)))

    print(f"[+] Correctly classified: {len(X_real)}")

    importance = compute_causal_importance(model, X_real)

    all_round_importance[r] = importance

# ============================================================
# ROUND-TO-ROUND TRANSPORT
# ============================================================

transport_corrs = []

print("\n========================================")
print("ROUND TRANSPORT CORRELATIONS")
print("========================================")

for i in range(len(ROUNDS)-1):

    r1 = ROUNDS[i]
    r2 = ROUNDS[i+1]

    v1 = all_round_importance[r1]
    v2 = all_round_importance[r2]

    # transport via ROTR(7)
    transported = rotr(v1, ROTATION)

    corr, _ = pearsonr(transported, v2)

    transport_corrs.append(corr)

    print(f"{r1} -> {r2} rounds : {corr:.6f}")

# ============================================================
# HEATMAP
# ============================================================

heatmap = np.array([all_round_importance[r] for r in ROUNDS])

plt.figure(figsize=(12, 6))

plt.imshow(
    heatmap,
    cmap="hot",
    aspect="auto"
)

plt.colorbar(label="Normalized Causal Importance")

plt.xticks(range(16))
plt.yticks(range(len(ROUNDS)), [f"{r} rounds" for r in ROUNDS])

plt.xlabel("Bit Position")
plt.ylabel("Rounds")

plt.title("Round-to-Round Causal Motif Evolution")

plt.tight_layout()

plt.savefig("motif_transport_heatmap.png")

# ============================================================
# TRANSPORT CORRELATION PLOT
# ============================================================

plt.figure(figsize=(8,5))

x_labels = [
    "5→6",
    "6→7",
    "7→8"
]

plt.plot(
    x_labels,
    transport_corrs,
    marker='o'
)

plt.ylabel("ROTR(7) Transport Correlation")
plt.xlabel("Round Transition")

plt.title("Motif Transport Across Rounds")

plt.grid(True)

plt.tight_layout()

plt.savefig("motif_transport_correlation.png")

# ============================================================
# TRANSPORT VISUALIZATION
# ============================================================

plt.figure(figsize=(14,8))

for r in ROUNDS:

    plt.plot(
        range(16),
        all_round_importance[r],
        marker='o',
        label=f"{r} rounds"
    )

plt.xlabel("Bit Position")
plt.ylabel("Normalized Causal Importance")

plt.title("Causal Motif Propagation Across Rounds")

plt.legend()

plt.grid(True)

plt.tight_layout()

plt.savefig("motif_transport_structure.png")

# ============================================================
# FINAL SUMMARY
# ============================================================

print("\n========================================")
print("TRANSPORT SUMMARY")
print("========================================")

for i in range(len(transport_corrs)):

    print(
        f"{ROUNDS[i]} -> {ROUNDS[i+1]} : "
        f"{transport_corrs[i]:.6f}"
    )

mean_transport = np.mean(transport_corrs)

print(f"\nMean transport correlation : {mean_transport:.6f}")

print("\n[+] Round-to-round motif transport complete.")