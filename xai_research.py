import speck as sp
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

from tensorflow.keras.models import model_from_json
from tensorflow.keras import Model

# ============================================================
# XAI RESEARCH MODULE
# Integrated Gradients Analysis for Gohr Neural Distinguishers
# ============================================================

# ------------------------------------------------------------
# LOAD TRAINED NETWORK
# ------------------------------------------------------------

json_file = open('single_block_resnet.json', 'r')
json_model = json_file.read()

net = model_from_json(json_model)
net.load_weights('net5_small.h5')

print("[+] Loaded pretrained 5-round distinguisher.")

# ------------------------------------------------------------
# GENERATE DATASET
# ------------------------------------------------------------

# smaller sample first for experimentation
NUM_SAMPLES = 2000

X, Y = sp.make_train_data(NUM_SAMPLES, 5)

print("[+] Generated dataset.")
print("X shape:", X.shape)
print("Y shape:", Y.shape)

# ------------------------------------------------------------
# FILTER CORRECTLY CLASSIFIED SAMPLES
# ------------------------------------------------------------

pred = net.predict(X, batch_size=5000).flatten()

pred_bin = (pred > 0.5).astype(np.uint8)

correct_idx = np.where(pred_bin == Y)[0]

X_correct = X[correct_idx]
Y_correct = Y[correct_idx]

print("[+] Correctly classified samples:", len(X_correct))

# ------------------------------------------------------------
# INTEGRATED GRADIENTS IMPLEMENTATION
# ------------------------------------------------------------

# baseline = all-zero ciphertext pair
baseline = np.zeros((64,), dtype=np.float32)

# number of integration steps
STEPS = 50

# tensor conversion
baseline_tf = tf.convert_to_tensor(baseline.reshape(1, -1))

# ------------------------------------------------------------
# COMPUTE INTEGRATED GRADIENTS
# ------------------------------------------------------------

def compute_integrated_gradients(model, sample, baseline, steps=50):

    sample = tf.cast(sample, tf.float32)
    baseline = tf.cast(baseline, tf.float32)

    # interpolate inputs
    interpolated_inputs = []

    for alpha in np.linspace(0.0, 1.0, steps):
        interpolated = baseline + alpha * (sample - baseline)
        interpolated_inputs.append(interpolated)

    interpolated_inputs = tf.convert_to_tensor(interpolated_inputs)

    # compute gradients
    with tf.GradientTape() as tape:
        tape.watch(interpolated_inputs)

        predictions = model(interpolated_inputs)

    grads = tape.gradient(predictions, interpolated_inputs)

    # average gradients
    avg_grads = tf.reduce_mean(grads, axis=0)

    # integrated gradients formula
    integrated_grads = (sample - baseline) * avg_grads

    return integrated_grads.numpy()

# ------------------------------------------------------------
# COMPUTE ATTRIBUTIONS
# ------------------------------------------------------------

all_attributions = []

NUM_EXPLANATIONS = 500

print("[+] Computing Integrated Gradients...")

for i in range(NUM_EXPLANATIONS):

    sample = X_correct[i]

    attribution = compute_integrated_gradients(
        net,
        sample,
        baseline,
        steps=STEPS
    )

    all_attributions.append(attribution)

    if i % 50 == 0:
        print(f"[+] Processed {i} samples")

all_attributions = np.array(all_attributions)

print("[+] Attribution tensor shape:", all_attributions.shape)

# ------------------------------------------------------------
# AGGREGATE ATTRIBUTIONS
# ------------------------------------------------------------

# mean absolute attribution
mean_attr = np.mean(np.abs(all_attributions), axis=0)

# reshape into 4 x 16 bit structure
heatmap = mean_attr.reshape(4, 16)

print("[+] Heatmap shape:", heatmap.shape)

# ------------------------------------------------------------
# VISUALIZATION
# ------------------------------------------------------------

plt.figure(figsize=(14, 4))

plt.imshow(heatmap, cmap='hot', aspect='auto')

plt.colorbar(label='Mean Absolute Attribution')

plt.xlabel('Bit Position')
plt.ylabel('Ciphertext Word')

plt.yticks(
    [0, 1, 2, 3],
    ['ct0a', 'ct1a', 'ct0b', 'ct1b']
)

plt.title(
    'Integrated Gradients Attribution Map\n'
    'Gohr 5-Round Neural Distinguisher'
)

plt.tight_layout()

plt.savefig('integrated_gradients_heatmap.png', dpi=300)

plt.show()

print("[+] Saved heatmap to integrated_gradients_heatmap.png")

# ------------------------------------------------------------
# STATISTICAL ANALYSIS
# ------------------------------------------------------------

print("\n==============================")
print("TOP ATTRIBUTED BITS")
print("==============================")

flat_indices = np.argsort(mean_attr)[::-1]

for rank in range(10):

    idx = flat_indices[rank]

    word = idx // 16
    bit = idx % 16

    print(
        f"Rank {rank+1}: "
        f"Word={word}, Bit={bit}, "
        f"Attribution={mean_attr[idx]:.6f}"
    )

# ------------------------------------------------------------
# WORD-LEVEL IMPORTANCE
# ------------------------------------------------------------

print("\n==============================")
print("WORD-LEVEL IMPORTANCE")
print("==============================")

for w in range(4):

    score = np.sum(heatmap[w])

    print(f"Word {w}: {score:.6f}")

# ------------------------------------------------------------
# BIT POSITION ANALYSIS
# ------------------------------------------------------------

print("\n==============================")
print("BIT POSITION IMPORTANCE")
print("==============================")

bit_scores = np.sum(heatmap, axis=0)

for bit in range(16):

    print(
        f"Bit {bit:02d}: "
        f"{bit_scores[bit]:.6f}"
    )

# ------------------------------------------------------------
# ATTRIBUTION VARIANCE
# ------------------------------------------------------------

variance_attr = np.var(all_attributions, axis=0)

variance_heatmap = variance_attr.reshape(4, 16)

plt.figure(figsize=(14, 4))

plt.imshow(variance_heatmap, cmap='viridis', aspect='auto')

plt.colorbar(label='Attribution Variance')

plt.xlabel('Bit Position')
plt.ylabel('Ciphertext Word')

plt.yticks(
    [0, 1, 2, 3],
    ['ct0a', 'ct1a', 'ct0b', 'ct1b']
)

plt.title(
    'Integrated Gradients Attribution Variance'
)

plt.tight_layout()

plt.savefig('integrated_gradients_variance.png', dpi=300)

plt.show()

print("[+] Saved variance map.")

# ------------------------------------------------------------
# SAVE NUMERICAL RESULTS
# ------------------------------------------------------------

np.save('integrated_gradients_raw.npy', all_attributions)
np.save('integrated_gradients_mean.npy', mean_attr)
np.save('integrated_gradients_variance.npy', variance_attr)

print("[+] Saved all attribution arrays.")

print("\n[+] Phase 2 Integrated Gradients Analysis Complete.")

