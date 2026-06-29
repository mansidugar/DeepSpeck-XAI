import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

from tensorflow.keras.models import model_from_json
from train_nets import cyclic_lr

import speck as sp


# ============================================================
# SETTINGS
# ============================================================

NUM_SAMPLES = 12000
ROUNDS = 7
WORD_SIZE = 16

TOP_K_IG_BITS = 8
PAIR_LIMIT = 28

MODEL_WEIGHTS = f'net{ROUNDS}_small.h5'
MODEL_JSON = 'single_block_resnet.json'

OUTPUT_DIR = "outputs"

os.makedirs(OUTPUT_DIR, exist_ok=True)


# ============================================================
# LOAD MODEL
# ============================================================

print("[+] Loading model.")

with open(MODEL_JSON, "r") as f:
    json_model = f.read()

model = model_from_json(json_model)

optimizer = tf.keras.optimizers.legacy.Adam(
    learning_rate=cyclic_lr(10, 0.002, 0.0001)(0)
)

model.compile(
    optimizer=optimizer,
    loss='mse',
    metrics=['acc']
)

model.load_weights(MODEL_WEIGHTS)

print("[+] Model loaded successfully.")


# ============================================================
# DATASET
# ============================================================

print("[+] Generating dataset.")

X, Y = sp.make_train_data(NUM_SAMPLES, ROUNDS)

preds = model.predict(
    X,
    batch_size=1024
).flatten()

correct_mask = (
    ((preds > 0.5) & (Y == 1)) |
    ((preds <= 0.5) & (Y == 0))
)

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

baseline_acc = np.mean(
    ((baseline_preds > 0.5) & (Y == 1)) |
    ((baseline_preds <= 0.5) & (Y == 0))
)

print(f"[+] Baseline Accuracy: {baseline_acc:.6f}")


# ============================================================
# INTEGRATED GRADIENTS
# ============================================================

print("[+] Computing Integrated Gradients.")


def integrated_gradients(
    model,
    inputs,
    baseline=None,
    steps=32
):

    inputs = tf.cast(inputs, tf.float32)

    if baseline is None:
        baseline = tf.zeros_like(inputs)

    interpolated = [
        baseline + (float(i)/steps)*(inputs-baseline)
        for i in range(steps+1)
    ]

    grads = []

    for x in interpolated:

        with tf.GradientTape() as tape:

            tape.watch(x)

            preds = model(x)

        grad = tape.gradient(preds, x)

        grads.append(grad)

    grads = tf.reduce_mean(
        tf.stack(grads),
        axis=0
    )

    ig = (inputs-baseline) * grads

    return ig.numpy()


subset = X[:1024]

ig_attr = integrated_gradients(
    model,
    subset
)

# ============================================================
# COLLAPSE 64 FEATURES -> 16 BIT POSITIONS
# ============================================================

ig_mean_full = np.mean(
    np.abs(ig_attr),
    axis=0
)

ig_mean = np.zeros(WORD_SIZE)

for bit in range(WORD_SIZE):

    positions = [
        bit,
        bit + 16,
        bit + 32,
        bit + 48
    ]

    ig_mean[bit] = np.mean(
        ig_mean_full[positions]
    )

ig_norm = ig_mean / np.max(ig_mean)

top_bits = np.argsort(
    ig_norm
)[::-1][:TOP_K_IG_BITS]

print(f"[+] Top IG bits: {top_bits}")


# ============================================================
# SINGLE-BIT CAUSAL IMPORTANCE
# ============================================================

print("[+] Computing single-bit importance.")

single_importance = np.zeros(WORD_SIZE)

for bit in range(WORD_SIZE):

    X_masked = np.copy(X)

    positions = [
        bit,
        bit + 16,
        bit + 32,
        bit + 48
    ]

    for p in positions:
        X_masked[:, p] = 0

    preds_masked = model.predict(
        X_masked,
        batch_size=1024,
        verbose=0
    ).flatten()

    acc = np.mean(
        ((preds_masked > 0.5) & (Y == 1)) |
        ((preds_masked <= 0.5) & (Y == 0))
    )

    single_importance[bit] = (
        baseline_acc - acc
    )

single_importance_norm = (
    single_importance /
    np.max(single_importance)
)


# ============================================================
# PAIRWISE INTERACTION ANALYSIS
# ============================================================

print("[+] Computing pairwise causal interactions.")

interaction_matrix = np.zeros(
    (WORD_SIZE, WORD_SIZE)
)

pairs_tested = []

count = 0

for i in top_bits:

    for j in top_bits:

        if j <= i:
            continue

        if count >= PAIR_LIMIT:
            break

        X_pair = np.copy(X)

        positions_i = [
            i,
            i + 16,
            i + 32,
            i + 48
        ]

        positions_j = [
            j,
            j + 16,
            j + 32,
            j + 48
        ]

        for p in positions_i:
            X_pair[:, p] = 0

        for p in positions_j:
            X_pair[:, p] = 0

        preds_pair = model.predict(
            X_pair,
            batch_size=1024,
            verbose=0
        ).flatten()

        pair_acc = np.mean(
            ((preds_pair > 0.5) & (Y == 1)) |
            ((preds_pair <= 0.5) & (Y == 0))
        )

        pair_drop = baseline_acc - pair_acc

        additive_expectation = (
            single_importance[i] +
            single_importance[j]
        )

        synergy = (
            pair_drop -
            additive_expectation
        )

        interaction_matrix[i, j] = synergy
        interaction_matrix[j, i] = synergy

        pairs_tested.append(
            (i, j, synergy, pair_drop)
        )

        count += 1


# ============================================================
# NORMALIZATION
# ============================================================

max_abs = np.max(
    np.abs(interaction_matrix)
)

if max_abs > 0:

    interaction_vis = (
        interaction_matrix / max_abs
    )

else:

    interaction_vis = interaction_matrix


# ============================================================
# TOP INTERACTIONS
# ============================================================

pairs_tested = sorted(
    pairs_tested,
    key=lambda x: abs(x[2]),
    reverse=True
)

print("\n======================================")
print("TOP CAUSAL INTERACTIONS")
print("======================================")

for rank, (i, j, synergy, pair_drop) in enumerate(
    pairs_tested[:15],
    start=1
):

    print(
        f"Rank {rank}: "
        f"Bits=({i},{j}) | "
        f"Synergy={synergy:.6f} | "
        f"PairDrop={pair_drop:.6f}"
    )


# ============================================================
# ROTATIONAL INTERACTION ANALYSIS
# ============================================================

print("\n======================================")
print("ROTATIONAL INTERACTION ANALYSIS")
print("======================================")

rot_synergies = []
nonrot_synergies = []

for i, j, synergy, _ in pairs_tested:

    rot_partner = (i + 7) % WORD_SIZE

    if j == rot_partner:

        rot_synergies.append(
            abs(synergy)
        )

    else:

        nonrot_synergies.append(
            abs(synergy)
        )

rot_mean = (
    np.mean(rot_synergies)
    if len(rot_synergies) > 0 else 0
)

nonrot_mean = (
    np.mean(nonrot_synergies)
    if len(nonrot_synergies) > 0 else 0
)

print(
    f"Rotation-aligned synergy : "
    f"{rot_mean:.6f}"
)

print(
    f"Non-rotation synergy    : "
    f"{nonrot_mean:.6f}"
)


# ============================================================
# INTERACTION DENSITY
# ============================================================

positive_count = np.sum(
    interaction_matrix > 0
)

negative_count = np.sum(
    interaction_matrix < 0
)

print("\n======================================")
print("INTERACTION DENSITY")
print("======================================")

print(
    f"Positive synergy pairs : "
    f"{positive_count}"
)

print(
    f"Negative synergy pairs : "
    f"{negative_count}"
)


# ============================================================
# INTERACTION HEATMAP
# ============================================================

plt.figure(figsize=(10, 8))

plt.imshow(
    interaction_vis,
    cmap='seismic',
    interpolation='nearest'
)

plt.colorbar(
    label="Normalized Interaction Synergy"
)

plt.title(
    "Pairwise Causal Interaction Matrix"
)

plt.xlabel("Bit Position")
plt.ylabel("Bit Position")

plt.tight_layout()

plt.savefig(
    os.path.join(
        OUTPUT_DIR,
        "pairwise_interaction_heatmap.png"
    )
)

plt.close()


# ============================================================
# TOP INTERACTIONS BARPLOT
# ============================================================

top_pairs = pairs_tested[:10]

plt.figure(figsize=(10, 5))

labels = [
    f"({i},{j})"
    for i, j, _, _ in top_pairs
]

values = [
    s
    for _, _, s, _ in top_pairs
]

plt.bar(
    range(len(values)),
    values
)

plt.xticks(
    range(len(values)),
    labels,
    rotation=45
)

plt.ylabel("Synergy")

plt.title(
    "Strongest Bitwise Causal Interactions"
)

plt.tight_layout()

plt.savefig(
    os.path.join(
        OUTPUT_DIR,
        "top_interactions.png"
    )
)

plt.close()


print("\n[+] Multi-bit interaction analysis complete.")