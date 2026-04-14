# PhonoSemantics

**Official code repository for:**

<<<<<<< HEAD
> Kumar, A. (2026). *Phonosemantic Grounding: Sanskrit as a Formalized Case of Motivated Sign Structure for Interpretable AI* (Version 2.0). Zenodo.
> DOI: [10.5281/zenodo.19564026](https://doi.org/10.5281/zenodo.19564026)
=======
> Kumar, A. (2026). *Phonosemantic Grounding: Sanskrit as a Formalized Case of Motivated Sign Structure for Interpretable AI*. Zenodo.
> DOI: [10.5281/zenodo.19508957](https://doi.org/10.5281/zenodo.19508957)
>>>>>>> 82dd80f82021287f5b49546d6d25568743521a2b

---

## What this is

This repository contains all experimental code, the phonosemantic embedding layer, and the figure generation script associated with the paper above.

The paper proposes that the articulatory anatomy of speech production — the five loci of the vocal tract, manner of constriction, phonation type, and somatic resonance — constitutes a physically real coordinate system for AI semantic representations. Sanskrit, with its rigorously formalized phonological system, serves as the proof-of-concept case.

---

## Repository contents

The main repository contains the core scripts for Phase 1-3 validation. The newly added `/Experiments` folder contains all exploratory tracking from Phase 4 (Formant Grounding) and Phase 5 (Neuromorphic SNN constraints).

| File/Folder | What it does |
|---|---|
| `phonosemantic_embedding.py` | The core embedding layer. Replaces `nn.Embedding` with a 10-dimensional physically grounded coordinate system derived from articulatory anatomy. Importable module. |
| `experiment_v3.py` | Canonical root clustering experiment. 150 Sanskrit verbal roots, 5 phonological groups, 3 statistical tests (Wilcoxon, Mann-Whitney, multinomial classification). All semantic scores derived independently from Monier-Williams dictionary definitions. Reproduces the main results in Section 9 of the paper. |
| `linear_probe.py` | Linear probe comparison. Tests whether articulatory geometry carries semantic signal beyond phoneme identity alone. Phonosemantic coordinates vs one-hot PCA baseline vs random at equal dimensionality. Reproduces the +14pp result in Section 9.4. |
| `blind_clustering.py` | Blind clustering experiment. Automatic TF-IDF semantic clustering with permutation test — no manual axis labels. Reproduces the null result reported honestly in Section 9.5. |
| `make_figure.py` | Generates Figure 1: the phonosemantic manifold map (phonemes in locus × manner space with word trajectory overlays). Requires matplotlib. |
| `phonosemantic_figure.png` | Figure 1 from the paper — pre-generated for convenience. |
| `/Experiments` | Contains Phase 4 (Continuous Formant Grounding) and Phase 5 (SNN simulations including Epileptiform Synchrony limit). |

---

## Installation

```bash
git clone https://github.com/HmbleCreator/PhonoSemantics.git
cd PhonoSemantics
pip install -r requirements.txt
```

---

## Running the experiments

Run in this order — each script builds on the root data defined in `experiment_v3.py`:

```bash
# 1. Main clustering experiment (reproduces Section 9.1–9.3)
python experiment_v3.py

# 2. Linear probe (reproduces Section 9.4)
python linear_probe.py

# 3. Blind clustering with permutation test (reproduces Section 9.5)
python blind_clustering.py

# 4. Regenerate Figure 1
python make_figure.py
```

All scripts are self-contained and print results to stdout. No GPU required. Runs on CPU in under 2 minutes total.

---

## Using the embedding layer

```python
from phonosemantic_embedding import PhonosemantikEmbedding

# Initialize
embed = PhonosemantikEmbedding()

# Get 10D coordinate for a phonological group
vec = embed.get_group_vector('LABIAL')   # returns np.array shape (10,)

# Get centroid trajectory for a root
centroid = embed.get_root_centroid('LABIAL')  # mean over phoneme vectors

# Full coordinate breakdown
print(embed.describe('THROAT'))
# → locus: [1,0,0,0,0,0], manner: 0.30, phonation: [0.5, 0.40], resonance: 1.0
```

---

## The coordinate system

Every phoneme maps to a 10-dimensional vector:

```
φ(p) = ( ℓ(p), α(p), β(p), ρ(p) )
```

| Dimension | Description | Size |
|---|---|---|
| D1 — Articulation locus | Where in the vocal tract (throat, palate, cerebral, dental, labial, nasal) | 6D |
| D2 — Articulation manner | Degree of closure (0 = full stop, 1 = fully open) | 1D |
| D3 — Phonation type | (voicing, breath force) | 2D |
| D4 — Somatic resonance | Primary body region of proprioceptive feedback (R1–R5, spinal axis) | 1D |

**Total: 10 dimensions**, fixed by articulatory anatomy. No statistical learning required for the coordinate system itself.

---

## Key results

| Experiment | Result |
|---|---|
| Axis clustering (Test 2, between-group) | 5/5 groups significant, all p < 0.001 |
| Multinomial classification | 41.3% vs 20% chance, p ≈ 10⁻¹⁴ |
| Linear probe — phonosemantic geometry | 63.3% ± 10.5% |
| Linear probe — phoneme identity baseline | 49.3% ± 6.8% |
| Linear probe — geometry advantage | +14.0 percentage points, p < 0.001 |
| Blind TF-IDF clustering | ARI = 0.007, p = 0.143 (not significant — reported in full) |
| Phase 4: Acoustic Formant Grounding (Sec 10) | ARI = 0.0366 via unsupervised F1/F2 continuous clustering. |
| Phase 5: SNN Validation Limit (Sec 11) | Epileptiform Synchrony Limit identified at ~500 Hz when modeling temporal context extension. |

---

## Citation

```bibtex
@misc{kumar2026phonosemantic,
  author    = {Kumar, Amit},
  title     = {Phonosemantic Grounding: {Sanskrit} as a Formalized Case
               of Motivated Sign Structure for Interpretable {AI}},
  year      = {2026},
  publisher = {Zenodo},
  version   = {2.0},
  doi       = {10.5281/zenodo.19564026},
  url       = {https://doi.org/10.5281/zenodo.19564026}
}
```

---

## License

Code: [MIT License](LICENSE)
Paper: [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/)

---

## Author

**Amit Kumar** — Independent Researcher, Bihar, India
GitHub: [@HmbleCreator](https://github.com/HmbleCreator)
