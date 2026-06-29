import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import pearsonr

from tensorflow.keras.models import model_from_json

import speck as sp
from train_nets import make_resnet


# ============================================================
# SETTINGS
# ============================================================

ROUNDS_LIST = [5, 6, 7, 8]

MODEL_FILES = {
    5: "net5_small.h5",
    6: "net6_small.h5",
    7: "net7_small.h5",
    8: "net8_small.h5"
}

NUM_SAMPLES = 3000
TOP_BITS = 16


# ============================================================
# LOAD MODEL
# ============================================================

def load_speck_model(rounds):

    model_path = MODEL_FILES[rounds]

    try:
        model = make_resnet(
            depth=1,
            word_size=16,
            num_filters=32,
            num_outputs=1,
            d1=64,
            d2=64,
            ks=3,
            reg_param=1e-5,
            final_activation='sigmoid'
        )

        model.load_weights(model_path)

        model.compile(
            optimizer='adam',
            loss='mse',
            metrics=['acc']
        )

        return model

    except:
        json_file = open("single_block_resnet.json", "r")
        loaded_model_json = json_file.read()
        json_file.close()

        model = model_from_json(loaded_model_json)
        model.load_weights(model_path)

        model.compile(
            optimizer='adam',
            loss='mse',
            metrics=['acc']
        )

        return model


# ============================================================
# DATASET
# ============================================================

def create_dataset(rounds, n):

    X, Y = sp.make_train_data(n, rounds)

    return X.astype(np.float32), Y


# ============================================================
# EVALUATE CAUSAL IMPORTANCE
# ============================================================

def evaluate_bit_importance(model, X, Y):

    baseline_preds = model.predict(X, batch_size=512).flatten()
    baseline_binary = (baseline_preds > 0.5).astype(np.uint8)

    baseline_acc = np.mean(baseline_binary == Y)

    importance = []

    for bit in range(16):

        X_mod = X.copy()

        positions = [bit, bit+16, bit+32, bit+48]

        for pos in positions:
            X_mod[:, pos] = 0

        preds = model.predict(X_mod, batch_size=512).flatten()
        binary = (preds > 0.5).astype(np.uint8)

        acc = np.mean(binary == Y)

        drop = baseline_acc - acc

        importance.append(drop)

    importance = np.array(importance)

    importance /= np.max(importance)

    return importance


# ============================================================
# ROTATIONAL CORRELATION
# ============================================================

def rotational_alignment(v, shift=7):

    rotated = np.roll(v, shift)

    corr, p = pearsonr(v, rotated)

    return corr, p, rotated


# ============================================================
# MAIN
# ============================================================

all_causal = []
rotation_scores = []

plt.figure(figsize=(12, 8))

for rounds in ROUNDS_LIST:

    print("\n====================================")
    print(f"ANALYZING {rounds} ROUNDS")
    print("====================================")

    print(f"[+] Loading {rounds}-round model.")
    model = load_speck_model(rounds)

    print("[+] Creating dataset.")
    X, Y = create_dataset(rounds, NUM_SAMPLES)

    preds = model.predict(X, batch_size=512).flatten()
    binary = (preds > 0.5).astype(np.uint8)

    mask = (binary == Y)

    X = X[mask]
    Y = Y[mask]

    print(f"[+] Correctly classified: {len(X)}")

    print("[+] Computing causal importance.")
    importance = evaluate_bit_importance(model, X, Y)

    all_causal.append(importance)

    corr, p, rotated = rotational_alignment(importance)

    rotation_scores.append(corr)

    print(f"[+] ROTR(7) correlation : {corr:.6f}")
    print(f"[+] P-value             : {p:.6f}")

    print("\nTOP BITS")

    top_idx = np.argsort(importance)[::-1][:10]

    for rank, idx in enumerate(top_idx):

        print(
            f"Rank {rank+1}: "
            f"Bit={idx}, "
            f"Importance={importance[idx]:.6f}, "
            f"RotPartner={(idx+7)%16}"
        )

    plt.plot(
        range(16),
        importance,
        marker='o',
        label=f"{rounds} rounds"
    )


# ============================================================
# PLOT 1
# ============================================================

plt.title("Rotational Causal Motif Structure")
plt.xlabel("Bit Position")
plt.ylabel("Normalized Causal Importance")
plt.legend()

plt.savefig(
    "outputs/rotational_motif_structure.png",
    dpi=300,
    bbox_inches='tight'
)

plt.close()


# ============================================================
# PLOT 2
# ============================================================

plt.figure(figsize=(8, 5))

plt.plot(
    ROUNDS_LIST,
    rotation_scores,
    marker='o'
)

plt.title("Rotational Equivariance Across Rounds")
plt.xlabel("Rounds")
plt.ylabel("ROTR(7) Correlation")

plt.savefig(
    "outputs/rotational_equivariance.png",
    dpi=300,
    bbox_inches='tight'
)

plt.close()


# ============================================================
# HEATMAP
# ============================================================

heatmap = np.array(all_causal)

plt.figure(figsize=(12, 6))

plt.imshow(
    heatmap,
    cmap='hot',
    aspect='auto'
)

plt.yticks(
    range(len(ROUNDS_LIST)),
    [f"{r} rounds" for r in ROUNDS_LIST]
)

plt.xticks(range(16))

plt.xlabel("Bit Position")
plt.ylabel("Rounds")

plt.title("Rotational Causal Motif Heatmap")

plt.colorbar(label="Normalized Importance")

plt.savefig(
    "outputs/rotational_motif_heatmap.png",
    dpi=300,
    bbox_inches='tight'
)

plt.close()


# ============================================================
# SUMMARY
# ============================================================

print("\n====================================")
print("ROTATIONAL MOTIF SUMMARY")
print("====================================")

for r, corr in zip(ROUNDS_LIST, rotation_scores):

    print(
        f"{r} rounds | "
        f"ROTR(7) correlation = {corr:.6f}"
    )

print("\n[+] Rotational motif discovery complete.")