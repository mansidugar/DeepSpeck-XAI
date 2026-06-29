# ============================================================
# rotation_aware_attribution_tracking.py
# ============================================================
#
# PURPOSE:
# --------
# Investigate whether neural attribution hotspots
# migrate according to Speck rotational structure.
#
# CORE RESEARCH QUESTION:
# -----------------------
# Does attribution movement follow:
#
#     ROTR(alpha=7)
#     ROTL(beta=2)
#
# used internally by Speck32/64?
#
# If YES:
# -------
# This becomes very strong evidence that the neural
# distinguisher has internalized cipher mechanics.
#
# ============================================================

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.models import model_from_json
from scipy.stats import pearsonr

import speck as sp


# ============================================================
# SETTINGS
# ============================================================

NUM_SAMPLES = 4000
NUM_STEPS = 50
ROUNDS = [5, 6, 7, 8]

WORD_SIZE = 16

ALPHA = 7
BETA = 2

OUTPUT_DIR = "./outputs"

os.makedirs(OUTPUT_DIR, exist_ok=True)


# ============================================================
# LOAD MODEL
# ============================================================

def load_model(rounds):

    print(f"[+] Loading {rounds}-round model.")

    json_file = open("single_block_resnet.json", "r")
    json_model = json_file.read()

    net = model_from_json(json_model)

    weight_file = f"net{rounds}_small.h5"

    net.load_weights(weight_file)

    return net


# ============================================================
# INTEGRATED GRADIENTS
# ============================================================

def integrated_gradients(model, x,
                         baseline=None,
                         steps=50):

    if baseline is None:
        baseline = np.zeros_like(x).astype(np.float32)

    x = x.astype(np.float32)

    scaled_inputs = [
        baseline + (float(i) / steps) * (x - baseline)
        for i in range(steps + 1)
    ]

    grads = []

    for s_in in scaled_inputs:

        s_tensor = tf.convert_to_tensor(s_in)

        with tf.GradientTape() as tape:

            tape.watch(s_tensor)

            pred = model(s_tensor, training=False)

        grad = tape.gradient(pred, s_tensor)

        grads.append(grad.numpy())

    grads = np.array(grads)

    avg_grads = np.mean(grads[:-1], axis=0)

    integrated_grad = (x - baseline) * avg_grads

    return integrated_grad


# ============================================================
# ROTATION UTILITIES
# ============================================================

def rotate_right_positions(positions, amount):

    return [(p + amount) % WORD_SIZE for p in positions]


def rotate_left_positions(positions, amount):

    return [(p - amount) % WORD_SIZE for p in positions]


# ============================================================
# EXTRACT HOTSPOTS
# ============================================================

def extract_hotspots(attr_map, topk=5):

    flat = attr_map.flatten()

    idx = np.argsort(flat)[::-1][:topk]

    hotspots = []

    for i in idx:

        row = i // WORD_SIZE
        col = i % WORD_SIZE

        hotspots.append((row, col, flat[i]))

    return hotspots


# ============================================================
# ROTATION ALIGNMENT SCORE
# ============================================================

def compute_rotation_alignment(attr_map):

    """
    Compare whether attribution peaks align
    with expected rotational propagation.
    """

    # focus on ciphertext words

    ct0a = attr_map[0]
    ct1a = attr_map[1]
    ct0b = attr_map[2]
    ct1b = attr_map[3]

    # normalize

    ct0a = ct0a / (np.max(ct0a) + 1e-8)
    ct1a = ct1a / (np.max(ct1a) + 1e-8)
    ct0b = ct0b / (np.max(ct0b) + 1e-8)
    ct1b = ct1b / (np.max(ct1b) + 1e-8)

    # Speck uses:
    #
    # x = ROTR(x,7)
    # y = ROTL(y,2)

    rotated_r = np.roll(ct0a, ALPHA)
    rotated_l = np.roll(ct1a, -BETA)

    corr_r, _ = pearsonr(rotated_r, ct0b)
    corr_l, _ = pearsonr(rotated_l, ct1b)

    return corr_r, corr_l


# ============================================================
# MAIN ANALYSIS
# ============================================================

rotation_scores_r = []
rotation_scores_l = []

entropy_scores = []

for nr in ROUNDS:

    print("\n====================================")
    print(f"ANALYZING {nr} ROUNDS")
    print("====================================")

    # --------------------------------------------------------
    # LOAD MODEL
    # --------------------------------------------------------

    model = load_model(nr)

    # --------------------------------------------------------
    # GENERATE DATA
    # --------------------------------------------------------

    X, Y = sp.make_train_data(NUM_SAMPLES, nr)

    # --------------------------------------------------------
    # KEEP CORRECTLY CLASSIFIED
    # --------------------------------------------------------

    preds = model.predict(X, batch_size=5000).flatten()

    pred_bin = (preds > 0.5)

    correct_idx = np.where(pred_bin == Y)[0]

    X = X[correct_idx]

    print(f"[+] Correctly classified: {len(X)}")

    # --------------------------------------------------------
    # COMPUTE INTEGRATED GRADIENTS
    # --------------------------------------------------------

    attr = integrated_gradients(
        model,
        X,
        steps=NUM_STEPS
    )

    attr = np.abs(attr)

    mean_attr = np.mean(attr, axis=0)

    attr_map = mean_attr.reshape(4, 16)

    # --------------------------------------------------------
    # ENTROPY
    # --------------------------------------------------------

    p = mean_attr / np.sum(mean_attr)

    entropy = -np.sum(p * np.log2(p + 1e-12))

    entropy_scores.append(entropy)

    # --------------------------------------------------------
    # ROTATION ALIGNMENT
    # --------------------------------------------------------

    corr_r, corr_l = compute_rotation_alignment(attr_map)

    rotation_scores_r.append(corr_r)
    rotation_scores_l.append(corr_l)

    print(f"[+] ROTR correlation : {corr_r:.6f}")
    print(f"[+] ROTL correlation : {corr_l:.6f}")

    # --------------------------------------------------------
    # HOTSPOTS
    # --------------------------------------------------------

    hotspots = extract_hotspots(attr_map, topk=10)

    print("\nTOP HOTSPOTS")

    for i, (r, c, v) in enumerate(hotspots):

        print(
            f"Rank {i+1}: "
            f"Word={r}, "
            f"Bit={c}, "
            f"Value={v:.6f}"
        )

    # --------------------------------------------------------
    # HEATMAP
    # --------------------------------------------------------

    plt.figure(figsize=(12, 5))

    plt.imshow(attr_map,
               cmap='hot',
               aspect='auto')

    plt.colorbar(
        label='Mean Absolute Attribution'
    )

    plt.title(
        f'Rotation-Aware Attribution\n'
        f'{nr}-Round Distinguisher'
    )

    plt.xlabel('Bit Position')
    plt.ylabel('Ciphertext Word')

    plt.yticks(
        [0,1,2,3],
        ['ct0a','ct1a','ct0b','ct1b']
    )

    plt.tight_layout()

    plt.savefig(
        f"{OUTPUT_DIR}/rotation_attr_{nr}r.png"
    )

    plt.close()


# ============================================================
# ROTATION EVOLUTION PLOT
# ============================================================

plt.figure(figsize=(10, 6))

plt.plot(
    ROUNDS,
    rotation_scores_r,
    marker='o',
    linewidth=2,
    label='ROTR(7) Alignment'
)

plt.plot(
    ROUNDS,
    rotation_scores_l,
    marker='s',
    linewidth=2,
    label='ROTL(2) Alignment'
)

plt.xlabel('Rounds')
plt.ylabel('Correlation')

plt.title(
    'Rotation-Aware Attribution Evolution'
)

plt.legend()

plt.grid(True)

plt.tight_layout()

plt.savefig(
    f"{OUTPUT_DIR}/rotation_alignment_evolution.png"
)

plt.close()


# ============================================================
# ENTROPY EVOLUTION
# ============================================================

plt.figure(figsize=(10, 6))

plt.plot(
    ROUNDS,
    entropy_scores,
    marker='o',
    linewidth=2
)

plt.xlabel('Rounds')
plt.ylabel('Attribution Entropy')

plt.title(
    'Attribution Diffusion Across Rounds'
)

plt.grid(True)

plt.tight_layout()

plt.savefig(
    f"{OUTPUT_DIR}/rotation_entropy_evolution.png"
)

plt.close()


# ============================================================
# FINAL SUMMARY
# ============================================================

print("\n====================================")
print("ROTATION-AWARE ANALYSIS SUMMARY")
print("====================================")

for i, nr in enumerate(ROUNDS):

    print(
        f"{nr} rounds | "
        f"ROTR corr = {rotation_scores_r[i]:.6f} | "
        f"ROTL corr = {rotation_scores_l[i]:.6f} | "
        f"Entropy = {entropy_scores[i]:.6f}"
    )

print("\n[+] Rotation-aware attribution tracking complete.")