# Contributing

Thank you for your interest in this project.

This repository accompanies the research paper **"Explainability Framework for Neural Cryptanalysis of Speck32/64"** and is intended to support the reproducibility of the experimental results presented in the paper.

## Reporting Issues

If you encounter bugs, unexpected behaviour, or reproducibility issues, please open a GitHub Issue describing:

* The operating system used.
* Python version.
* Installed package versions.
* The script being executed.
* Complete error messages and stack traces.
* Steps required to reproduce the issue.

Providing this information will help identify and resolve problems efficiently.

---

## Reproducing the Experiments

Before running any experiment:

1. Clone the repository.
2. Install the required Python packages.

```bash
pip install -r requirements.txt
```

3. Execute the desired experiment.

Example:

```bash
python3 single_bit_causal_ablation.py
```

Most experiments automatically save figures and outputs inside the `outputs/` directory.

---

## Contributing Code

Contributions that improve the framework are welcome.

Examples include:

* Additional explainability techniques.
* New neural cryptanalysis experiments.
* Improvements to visualization.
* Performance optimizations.
* Documentation enhancements.
* Support for additional lightweight block ciphers.

Please ensure that new code is clearly documented and follows the existing project structure.

---

## Citation

If you use this repository in your research, please cite the accompanying paper.

```text
Mansi Dugar,
Explainability Framework for Neural Cryptanalysis of Speck32/64,
2026.
```

The citation will be updated after publication.

---

## Acknowledgements

This work builds upon the DeepSpeck neural distinguisher introduced by:

> A. Gohr,
> *Improving Attacks on Round-Reduced Speck32/64 Using Deep Learning*,
> CRYPTO 2019.

The original DeepSpeck implementation serves as the baseline neural distinguisher used in this research. The explainability framework, attribution analysis, causal validation methodology, interaction analysis, stability analysis, control experiments, and case studies presented in this repository are original contributions of this work.
