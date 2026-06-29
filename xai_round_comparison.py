import matplotlib
matplotlib.use('Agg')

import speck as sp
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

from tensorflow.keras.models import model_from_json

# ============================================================
# ROUND-WISE XAI ANALYSIS
# ============================================================

ROUNDS = [5, 6, 7, 8]

WEIGHT_FILES = {
    5: 'net5_small.h5',
    6: 'net6_small.h5',
    7: 'net7_small.h5',
    8: 'net8_small.h5'
}

# ------------------------------------------------------------
# CONFIG
# ------------------------------------------------------------

NUM_SAMPLES = 500
NUM_EXPLANATIONS = 100
STEPS = 20

# ------------------------------------------------------------
# LOAD MODEL ARCHITECTURE
# ------------------------------------------------------------

json_file = open('single_block_resnet.json', 'r')
json_model = json_file.read()

# ------------------------------------------------------------
# INTEGRATED GRADIENTS
# ------------------------------------------------------------

def integrated_gradients(model, sample, baseline, steps=20):

    sample = tf.cast(sample, tf.float32)
    baseline = tf.cast(baseline, tf.float32)

    interpolated = []

    for alpha in np.linspace(0.0, 1.0, steps):
        x = baseline + alpha * (sample - baseline)
        interpolated.append(x)

    interpolated = tf.convert_to_tensor(interpolated)

    with tf.GradientTape() as tape:

        tape.watch(interpolated)

        pred = model(interpolated)

    grads = tape.gradient(pred, interpolated)

    avg_grads = tf.reduce_mean(grads, axis=0)

    ig = (sample - baseline) * avg_grads

    return ig.numpy()

# ------------------------------------------------------------
# ENTROPY METRIC
# ------------------------------------------------------------

def attribution_entropy(attr):

    p = np.abs(attr)

    p = p / np.sum(p)

    eps = 1e-10

    entropy = -np.sum(p * np.log2(p + eps))

    return entropy

# ------------------------------------------------------------
# MAIN ANALYSIS
# ------------------------------------------------------------

all_round_heatmaps = []

baseline = np.zeros((64,), dtype=np.float32)

entropy_scores = []

for rounds in ROUNDS:

    print("\n===================================")
    print(f"ANALYZING {rounds} ROUNDS")
    print("===================================")

    # --------------------------------------------------------
    # LOAD MODEL
    # --------------------------------------------------------

    model = model_from_json(json_model)

    model.load_weights(WEIGHT_FILES[rounds])

    print("[+] Model loaded.")

    # --------------------------------------------------------
    # GENERATE DATA
    # --------------------------------------------------------

    X, Y = sp.make_train_data(NUM_SAMPLES, rounds)

    pred = model.predict(X, batch_size=5000).flatten()

    pred_bin = (pred > 0.5).astype(np.uint8)

    correct_idx = np.where(pred_bin == Y)[0]

    X_correct = X[correct_idx]

    print("[+] Correctly classified:",
          len(X_correct))

    # --------------------------------------------------------
    # COMPUTE ATTRIBUTIONS
    # --------------------------------------------------------

    attributions = []

    for i in range(NUM_EXPLANATIONS):

        sample = X_correct[i]

        ig = integrated_gradients(
            model,
            sample,
            baseline,
            steps=STEPS
        )

        attributions.append(ig)

    attributions = np.array(attributions)

    # --------------------------------------------------------
    # AGGREGATE
    # --------------------------------------------------------

    mean_attr = np.mean(
        np.abs(attributions),
        axis=0
    )

    variance_attr = np.var(
        attributions,
        axis=0
    )

    entropy = attribution_entropy(mean_attr)

    entropy_scores.append(entropy)

    heatmap = mean_attr.reshape(4, 16)

    all_round_heatmaps.append(heatmap)

    print("[+] Attribution entropy:",
          entropy)

    # --------------------------------------------------------
    # SAVE INDIVIDUAL HEATMAP
    # --------------------------------------------------------

    plt.figure(figsize=(12, 4))

    plt.imshow(
        heatmap,
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
        f'Integrated Gradients Heatmap\n'
        f'{rounds}-Round Distinguisher'
    )

    plt.tight_layout()

    plt.savefig(
        f'heatmap_{rounds}r.png',
        dpi=300
    )

    plt.close()

    # --------------------------------------------------------
    # SAVE VARIANCE MAP
    # --------------------------------------------------------

    variance_heatmap = variance_attr.reshape(4,16)

    plt.figure(figsize=(12,4))

    plt.imshow(
        variance_heatmap,
        cmap='viridis',
        aspect='auto'
    )

    plt.colorbar(
        label='Attribution Variance'
    )

    plt.xlabel('Bit Position')
    plt.ylabel('Ciphertext Word')

    plt.yticks(
        [0,1,2,3],
        ['ct0a','ct1a','ct0b','ct1b']
    )

    plt.title(
        f'Attribution Variance\n'
        f'{rounds}-Round Distinguisher'
    )

    plt.tight_layout()

    plt.savefig(
        f'variance_{rounds}r.png',
        dpi=300
    )

    plt.close()

# ============================================================
# COMPARATIVE VISUALIZATION
# ============================================================

fig, axes = plt.subplots(
    1,
    4,
    figsize=(20,4)
)

for idx, rounds in enumerate(ROUNDS):

    ax = axes[idx]

    im = ax.imshow(
        all_round_heatmaps[idx],
        cmap='hot',
        aspect='auto'
    )

    ax.set_title(f'{rounds} Rounds')

    ax.set_xlabel('Bit Position')

    if idx == 0:
        ax.set_ylabel('Ciphertext Word')

    ax.set_yticks(
        [0,1,2,3]
    )

    ax.set_yticklabels(
        ['ct0a','ct1a','ct0b','ct1b']
    )

fig.colorbar(
    im,
    ax=axes.ravel().tolist()
)

plt.suptitle(
    'Round-wise Attribution Evolution'
)

plt.tight_layout()

plt.savefig(
    'round_comparison.png',
    dpi=300
)

# ============================================================
# ENTROPY ANALYSIS
# ============================================================

plt.figure(figsize=(8,5))

plt.plot(
    ROUNDS,
    entropy_scores,
    marker='o'
)

plt.xlabel('Rounds')

plt.ylabel('Attribution Entropy')

plt.title(
    'Attribution Diffusion Across Rounds'
)

plt.grid(True)

plt.savefig(
    'entropy_progression.png',
    dpi=300
)

# ============================================================
# INTERPRETATION
# ============================================================

print("\n===================================")
print("ROUND-WISE ENTROPY")
print("===================================")

for r,e in zip(ROUNDS, entropy_scores):

    print(f"{r} rounds -> entropy = {e:.4f}")

print("\n[+] Analysis complete.")