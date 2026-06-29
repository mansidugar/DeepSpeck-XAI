# Explainability Framework for Neural Cryptanalysis of Speck32/64

This repository contains the complete implementation accompanying the research paper:

> **Explainability Framework for Neural Cryptanalysis of Speck32/64**

The project proposes a comprehensive Explainable Artificial Intelligence (XAI) framework for interpreting the decisions of the DeepSpeck neural distinguisher. While previous work has primarily focused on improving the distinguishing capability of neural cryptanalysis, this repository investigates **why** the neural distinguisher makes its predictions and whether the learned features correspond to genuine cryptographic structures.

The framework combines attribution analysis, causal validation, interaction analysis, structural stability analysis, and multiple control experiments to provide a systematic interpretation of neural distinguishers operating on the Speck32/64 lightweight block cipher.

---

# Repository Contents

The repository contains:

* Implementation of the proposed eight-layer explainability framework.
* Source code for all attribution and explainability experiments.
* Case studies demonstrating the effect of controlled input manipulations.
* Pre-trained DeepSpeck neural distinguishers for 5, 6, 7 and 8 rounds.
* Experimental datasets and intermediate NumPy files used during analysis.
* Generated figures used in the research paper.
* Complete manuscript (PDF and LaTeX source).

The implementation covers the following experimental modules:

* Integrated Gradients Attribution
* Single-Bit Causal Ablation
* Faithfulness Evaluation
* Pairwise Interaction Analysis
* Structural Stability Across Encryption Rounds
* Randomized Control Validation
* True Structure Destruction Validation
* Noise Injection Analysis
* Differential Structure Analysis

---

# Repository Structure

```text
.
├── outputs/                    # Generated figures and experimental outputs
├── supplementary_data/         # Supporting datasets
├── simple_net/                 # Baseline DeepSpeck implementation
├── cpp/                        # Original C++ utilities
├── *.py                        # Explainability experiments
├── *.npy                       # Saved attribution and analysis data
├── net5_small.h5               # Pre-trained 5-round model
├── net6_small.h5               # Pre-trained 6-round model
├── net7_small.h5               # Pre-trained 7-round model
├── net8_small.h5               # Pre-trained 8-round model
└── Explainability Frameworks for Neural Cryptanalysis.pdf
```

---

# Main Experiments

The primary experiments implemented in this repository include:

| Script                                  | Description                                                        |
| --------------------------------------- | ------------------------------------------------------------------ |
| `single_bit_causal_ablation.py`         | Evaluates causal importance of individual ciphertext bits.         |
| `multi_bit_causal_interaction.py`       | Studies cooperative interactions between important features.       |
| `pairwise_differential_attribution.py`  | Generates attribution maps using Integrated Gradients.             |
| `round_transport_analysis.py`           | Tracks attribution stability across multiple encryption rounds.    |
| `randomized_control_validation.py`      | Validates explanations using randomized ciphertext structures.     |
| `true_structure_destruction_control.py` | Removes differential structure to verify explanation faithfulness. |
| `case_study1_diff_removal.py`           | Differential structure removal case study.                         |
| `case_study2_noise_injection.py`        | Noise injection robustness analysis.                               |

---

# Requirements

The project was developed using Python 3.

Required packages include:

* TensorFlow
* Keras
* NumPy
* SciPy
* Matplotlib
* scikit-learn
* h5py

Install the required packages using:

```bash
pip install -r requirements.txt
```

---

# Running the Experiments

Each experiment can be executed independently.

Example:

```bash
python3 single_bit_causal_ablation.py
```

or

```bash
python3 case_study2_noise_injection.py
```

The generated figures are automatically saved inside the `outputs/` directory.

---

# Experimental Pipeline

The explainability framework follows the pipeline below:

```text
Speck32/64 Ciphertext Generation
            ↓
Binary Feature Representation
            ↓
DeepSpeck Neural Distinguisher
            ↓
Integrated Gradients Attribution
            ↓
Causal Validation
            ↓
Faithfulness Evaluation
            ↓
Interaction Analysis
            ↓
Structural Stability Analysis
            ↓
Control Experiments
```

---

# Citation

If you use this repository in your research, please cite:

> **Explainability Framework for Neural Cryptanalysis of Speck32/64**

(BibTeX will be added after publication.)

---

# Acknowledgements

This work builds upon the DeepSpeck neural cryptanalysis framework introduced by:

**A. Gohr**, *Improving Attacks on Round-Reduced Speck32/64 Using Deep Learning*, CRYPTO 2019.

The original DeepSpeck implementation served as the baseline neural distinguisher used in this research. All explainability modules, validation experiments, case studies, and analysis presented in this repository are original contributions developed as part of this work.
