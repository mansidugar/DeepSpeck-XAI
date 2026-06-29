import numpy as np
import tensorflow as tf
from tensorflow.keras.models import model_from_json
from sklearn.metrics import accuracy_score
from scipy.stats import pearsonr
from scipy.stats import entropy
import matplotlib.pyplot as plt

from speck import make_train_data


# ============================================================
# CONFIG
# ============================================================

ROUNDS_LIST = [5, 6, 7, 8]

NUM_SAMPLES = 10000

MODEL_JSON = "single_block_resnet.json"

MODEL_FILES = {
    5: "net5_small.h5",
    6: "net6_small.h5",
    7: "net7_small.h5",
    8: "net8_small.h5"
}

# ============================================================
# STORAGE
# ============================================================

all_causal_maps = {}
entropy_values = []
rotation_corrs = []

# ============================================================
# HELPER
# ============================================================

def rotate_right(arr, shift):

    return np.roll(arr, shift)

# ============================================================
# MAIN LOOP
# ============================================================

for rounds in ROUNDS_LIST:

    print("\n====================================")
    print(f"ANALYZING {rounds} ROUNDS")
    print("====================================")

    # ========================================================
    # LOAD MODEL
    # ========================================================

    print(f"[+] Loading {rounds}-round model.")

    with open(MODEL_JSON, "r") as f:
        json_model = f.read()

    model = model_from_json(json_model)

    model.load_weights(MODEL_FILES[rounds])

    # ========================================================
    # DATA
    # ========================================================

    X, Y = make_train_data(NUM_SAMPLES, rounds)

    preds = model.predict(
        X,
        batch_size=1024,
        verbose=0
    ).flatten()

    pred_labels = (preds > 0.5).astype(np.uint8)

    correct_mask = (pred_labels == Y)

    X = X[correct_mask]
    Y = Y[correct_mask]

    print(f"[+] Correctly classified: {len(X)}")

    # ========================================================
    # BASELINE
    # ========================================================

    baseline_acc = accuracy_score(
        Y,
        pred_labels[correct_mask]
    )

    # ========================================================
    # SINGLE-BIT CAUSAL ANALYSIS
    # ========================================================

    causal_scores = []

    for bit in range(16):

        X_mod = X.copy()

        # remove same bit from all words

        X_mod[:, bit] = 0
        X_mod[:, 16 + bit] = 0
        X_mod[:, 32 + bit] = 0
        X_mod[:, 48 + bit] = 0

        preds_mod = model.predict(
            X_mod,
            batch_size=1024,
            verbose=0
        ).flatten()

        labels_mod = (
            preds_mod > 0.5
        ).astype(np.uint8)

        acc_mod = accuracy_score(
            Y,
            labels_mod
        )

        drop = baseline_acc - acc_mod

        causal_scores.append(drop)

    causal_scores = np.array(causal_scores)

    # normalize

    causal_scores = (
        causal_scores /
        np.max(causal_scores)
    )

    all_causal_maps[rounds] = causal_scores

    # ========================================================
    # ENTROPY
    # ========================================================

    ent = entropy(
        causal_scores + 1e-10
    )

    entropy_values.append(ent)

    # ========================================================
    # ROTATION ALIGNMENT
    # ========================================================

    rotated = rotate_right(
        causal_scores,
        7
    )

    corr, _ = pearsonr(
        causal_scores,
        rotated
    )

    rotation_corrs.append(corr)

    print(f"[+] ROTR(7) correlation : {corr:.6f}")
    print(f"[+] Entropy             : {ent:.6f}")

    # ========================================================
    # TOP BITS
    # ========================================================

    print("\nTOP CAUSAL BITS")

    top_bits = np.argsort(causal_scores)[::-1]

    for i in range(10):

        b = top_bits[i]

        print(
            f"Rank {i+1}: "
            f"Bit={b}, "
            f"Importance={causal_scores[b]:.6f}"
        )

# ============================================================
# HEATMAP
# ============================================================

heatmap = np.array([
    all_causal_maps[r]
    for r in ROUNDS_LIST
])

plt.figure(figsize=(12, 5))

plt.imshow(
    heatmap,
    cmap='hot',
    aspect='auto'
)

plt.colorbar(
    label="Normalized Causal Importance"
)

plt.xticks(
    np.arange(16)
)

plt.yticks(
    np.arange(len(ROUNDS_LIST)),
    [f"{r} rounds" for r in ROUNDS_LIST]
)

plt.xlabel("Bit Position")
plt.ylabel("Rounds")

plt.title("Round-to-Round Causal Flow")

plt.tight_layout()

plt.savefig("causal_flow_heatmap.png")

plt.close()

# ============================================================
# FLOW EVOLUTION
# ============================================================

plt.figure(figsize=(10, 6))

for bit in range(16):

    vals = [
        all_causal_maps[r][bit]
        for r in ROUNDS_LIST
    ]

    plt.plot(
        ROUNDS_LIST,
        vals,
        marker='o',
        label=f"Bit {bit}"
    )

plt.xlabel("Rounds")
plt.ylabel("Causal Importance")

plt.title("Bitwise Causal Evolution")

plt.tight_layout()

plt.savefig("causal_bit_evolution.png")

plt.close()

# ============================================================
# ENTROPY PLOT
# ============================================================

plt.figure(figsize=(8, 5))

plt.plot(
    ROUNDS_LIST,
    entropy_values,
    marker='o'
)

plt.xlabel("Rounds")
plt.ylabel("Causal Entropy")

plt.title("Causal Diffusion Across Rounds")

plt.tight_layout()

plt.savefig("causal_entropy_evolution.png")

plt.close()

# ============================================================
# ROTATION ALIGNMENT PLOT
# ============================================================

plt.figure(figsize=(8, 5))

plt.plot(
    ROUNDS_LIST,
    rotation_corrs,
    marker='o'
)

plt.xlabel("Rounds")
plt.ylabel("ROTR(7) Correlation")

plt.title("Rotation Alignment of Causal Structure")

plt.tight_layout()

plt.savefig("causal_rotation_alignment.png")

plt.close()

# ============================================================
# CROSS-ROUND SIMILARITY
# ============================================================

print("\n====================================")
print("CROSS-ROUND CAUSAL SIMILARITY")
print("====================================")

for i in range(len(ROUNDS_LIST)-1):

    r1 = ROUNDS_LIST[i]
    r2 = ROUNDS_LIST[i+1]

    corr, _ = pearsonr(
        all_causal_maps[r1],
        all_causal_maps[r2]
    )

    print(
        f"{r1} -> {r2} rounds : "
        f"{corr:.6f}"
    )

print("\n[+] Causal flow analysis complete.")