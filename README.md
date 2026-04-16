# PhonoSemantics

**Official code repository for:**

> Kumar, A. (2026). *Phonosemantic Grounding: Sanskrit as a Formalized Case of Motivated Sign Structure for Interpretable AI* (Version 2.0). Zenodo. DOI: 10.5281/zenodo.19564026

> Kumar, A. (2026). *DDIN: Devavāṇī-Derived Interpretable Network — Sequential Neural ODEs for Phonosemantic Grounding*. arXiv:2604.XXXXX.

> Kumar, A. (2026). *PhonosemanticMeta Benchmark: Evaluating AI Vector Grounding via the Phonosemantic Manifold*. Kaggle Competition + arXiv.

---

## What this is

This repository contains all experimental code, the phonosemantic embedding layer, and the figure generation script associated with the papers above.

The research program explores two complementary tracks:

1. **Phonosemantic Grounding**: The paper proposes that the articulatory anatomy of speech production — the five loci of the vocal tract, manner of constriction, phonation type, and somatic resonance — constitutes a physically real coordinate system for AI semantic representations. Sanskrit, with its rigorously formalized phonological system, serves as the proof-of-concept case.

2. **DDIN (Devavāṇī-Derived Interpretable Network)**: A parallel research track exploring whether sequential phoneme encoding through neural ODEs can extract semantic structure from Sanskrit verbal roots without backpropagation or dense connectivity. The "Receiver Model" (W=0) architecture proves that semantics can emerge from heterogeneous neuron physics alone.

---

## Repository contents

The main repository contains the core scripts for Phase 1-3 validation. The `/Experiments` folder contains all exploratory tracking from Phase 4 (Formant Grounding), Phase 5 (Neuromorphic SNN constraints), and Phase 5B-6 (Sequential ODE + Architecture Ceiling).

| File/Folder | What it does |
|---|---|
| `phonosemantic_embedding.py` | The core embedding layer. Replaces `nn.Embedding` with a 10-dimensional physically grounded coordinate system derived from articulatory anatomy. Importable module. |
| `experiment_v3.py` | Canonical root clustering experiment. 150 Sanskrit verbal roots, 5 phonological groups, 3 statistical tests (Wilcoxon, Mann-Whitney, multinomial classification). All semantic scores derived independently from Monier-Williams dictionary definitions. Reproduces the main results in Section 9 of the paper. |
| `linear_probe.py` | Linear probe comparison. Tests whether articulatory geometry carries semantic signal beyond phoneme identity alone. Phonosemantic coordinates vs one-hot PCA baseline vs random at equal dimensionality. Reproduces the +14pp result in Section 9.4. |
| `blind_clustering.py` | Blind clustering experiment. Automatic TF-IDF semantic clustering with permutation test — no manual axis labels. Reproduces the null result reported honestly in Section 9.5. |
| `make_figure.py` | Generates Figure 1: the phonosemantic manifold map (phonemes in locus × manner space with word trajectory overlays). Requires matplotlib. |
| `phonosemantic_figure.png` | Figure 1 from the paper — pre-generated for convenience. |
| `/Experiments` | Contains Phase 4 (Continuous Formant Grounding), Phase 5 (SNN simulations), and Phase 5B-6 (Sequential ODE breakthrough + architecture ceiling). |
| `generate_submission_v21.py` | Competition submission generator for Kaggle. Uses v21 sequential ODE model. |
| `submission_v21.csv` | Competition predictions: 42% accuracy, ARI = 0.0538 |

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

## Phase 4, 5 & 5B-6 Extensions (Version 2.0)

The `Experiments/` directory contains the complete chronological chain of advanced empirical validation.

### Phase 4: Continuous Formant Grounding

| Script | What it does |
|---|---|
| `ddin_exp15_formant_grounding.py` | Continuous-Time Formant Grounding (Section 10). Employs real F1/F2 acoustic frequencies. |
| `ddin_exp16_weighted_formant.py` | Establishes the ARI = 0.0366 unsupervised baseline mapping. |

### Phase 5: Neuromorphic SNN Constraints

| Script | What it does |
|---|---|
| `ddin_exp17_snn_pynn.py` | PyNN deployment. Identifies the Epileptiform Synchrony Limit (~500 Hz). |
| `ddin_exp18_snn_inhibition.py` | Tests if lateral inhibition prevents seizure state. |
| `ddin_exp19_snn_wta.py` | Global Winner-Take-All architecture. |

### Phase 5B-6: Sequential ODE Breakthrough & Architecture Ceiling

| Script | Key Result |
|---|---|
| `ddin_exp21_sequential_ode.py` | **ARI = 0.0591** — shatters static 0.037 ceiling (+61%) |
| `ddin_exp22_grpo.py` | GRPO with discrete ARI reward — flat gradient |
| `ddin_exp24_grpo_supervised.py` | Supervised centroid reward — no improvement |
| `ddin_exp28b_structured_w.py` | Structured W initialization — best ARI = 0.0592 |
| `ddin_exp29_two_layer.py` | Two-layer hierarchy — +0.017 delta |
| `ddin_exp30_contrastive.py` | Contrastive GRPO — ARI = 0.0690 |
| `ddin_exp31_supervised_centroid.py` | Supervised on Layer 2 — ARI = 0.0690 |

**To reproduce the key experiments:**
```bash
cd Experiments
python ddin_exp21_sequential_ode.py   # Best model: ARI = 0.0591
python ddin_exp29_two_layer.py        # Two-layer baseline
```
*(Note: Phase 5 experiments require `pyNN` and `Brian2` backend.)*

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
| **Phase 5B: Sequential ODE (v21)** | **ARI = 0.0591** — sequential phoneme encoding via Neural ODE shatters static ceiling (+61% improvement). |
| **Phase 5B-6: GRPO & Architecture (v22-v31)** | **All reward variants converge to ~0.06 ARI ceiling.** Two-layer hierarchy adds +0.017 delta. |
| **Kaggle Competition Submission** | Accuracy: 42%, ARI: 0.0538 (v21 model) — submitted April 2026 |

## Research Papers

### Paper 1: Phonosemantic Grounding (Zenodo / CC BY 4.0)

> Kumar, A. (2026). *Phonosemantic Grounding: Sanskrit as a Formalized Case of Motivated Sign Structure for Interpretable AI* (Version 2.0). Zenodo. DOI: 10.5281/zenodo.19564026

**Core claim**: The articulatory anatomy of speech production (locus, manner, phonation, resonance) constitutes a physically grounded coordinate system for AI semantic representations — no statistical learning required for the coordinate system itself.

**Key results**:
- 41.3% multinomial classification (vs 20% chance)
- +14pp linear probe advantage over phoneme identity baseline
- ARI = 0.0366 via F1/F2 formant clustering (Phase 4)
- Epileptiform Synchrony Limit at ~500 Hz (Phase 5 SNN)

### Paper 2: DDIN — Sequential Neural ODEs for Phonosemantic Grounding (arXiv)

> Kumar, A. (2026). *DDIN: Devavāṇī-Derived Interpretable Network — Sequential Neural ODEs for Phonosemantic Grounding*. arXiv:2604.XXXXX.

**Core claim**: Sequential phoneme encoding through neural ODEs can extract semantic structure from Sanskrit verbal roots without backpropagation or dense recurrent connectivity.

**Key results**:
- **ARI = 0.0591** (v21 sequential ODE) — +61% over static baseline
- **Receiver Model validated**: W=0 (no recurrent weights) achieves ~0.06 ARI
- Two-layer hierarchy adds +0.017 delta
- All optimization strategies (GRPO, supervised, contrastive) converge to ~0.06 ceiling
- Precise architecture ceiling measurement for this task

**Functional claim (publication-ready)**:
> A zero-weight, 128-neuron sequential ODE reservoir achieves ARI ~0.06 on unsupervised semantic clustering of Sanskrit roots — the practical ceiling for this architecture.

### Paper 3: PhonosemanticMeta Benchmark — Exposing the Grounding Gap in Frontier AI (Kaggle)

> Kumar, A. (2026). *PhonosemanticMeta Benchmark: Evaluating AI Vector Grounding via the Phonosemantic Manifold*. Kaggle Competition + arXiv.

**Core claim**: The benchmark evaluates whether frontier AI systems possess intrinsic semantic grounding by testing them against a physically real coordinate system (Locus × Manner × Phonation × Resonance). By enforcing a strict **Vector Grounding Score (VGS)** threshold, it empirically exposes the "Grounding Gap" in modern LLMs.

**The 8 Benchmark Tasks**:

| Task | Name | What it tests | VGS Result |
|------|------|---------------|------------|
| T1 | Axis Prediction | Semantic clustering of 150 Sanskrit roots into 5 phenomenological axes | Pass |
| T2 | Phonological Siblings | Same-locus root similarity vs cross-locus | Pass |
| T3 | Fabricated Roots | Novel root prediction from phoneme patterns | Pass |
| T4 | Cross-Locus Distance | Semantic distance across articulation loci | Fail (positional bias) |
| T5 | Rule Generalization | Paninian rule application to held-out roots | Pass |
| T6 | Trajectories | Mapping word trajectories to phenomenological arcs | Fail (95% overconfidence, all "Arc A") |
| T7 | Triplets | Harmonic coherence: anchor-to-locus matching | Fail (0% confidence collapse) |
| T8 | Phonation | Mapping breath force to motor-unit recruitment | Fail (0% confidence collapse) |

**Key finding**: Tasks 7 & 8 produce **0% confidence** collapses in frontier models (Gemini 2.5 Flash). The model guesses the correct physical answer but reports zero internal certainty — proving the absence of bodily grounding in statistical embeddings.

**VGS Metric**: `VGS = (Is_Correct) × (Confidence ≥ 70%)`

---

## Phase 5B-6: Sequential ODE & Architecture Ceiling (2026-04-16)

The `/Experiments` folder now contains the complete Phase 5B-6 experimental chain documenting the transition from static embeddings to sequential Neural ODE processing.

### Key Experiments

| Script | Description | Key Result |
|--------|-------------|-------------|
| `ddin_exp21_sequential_ode.py` | Sequential phoneme encoding through 128-neuron ODE reservoir. W=0 (Receiver Model). | **ARI = 0.0591** — shatters static 0.037 ceiling |
| `ddin_exp22_grpo.py` | GRPO optimization on α (decay) topology with discrete ARI reward. | ARI = 0.0411 — flat gradient (discrete metric) |
| `ddin_exp24_grpo_supervised.py` | Supervised centroid reward on α+β. | ARI = 0.0538 — no improvement |
| `ddin_exp28b_structured_w.py` | Structured W initialization (Dhātu-like clusters). | Best: ARI = 0.0592 (10 clusters) |
| `ddin_exp29_two_layer.py` | Two-layer hierarchical architecture (Layer 1: phoneme encoder, Layer 2: semantic organizer). | ARI = 0.0492 (+0.017 delta) |
| `ddin_exp30_contrastive.py` | Contrastive GRPO on two-layer architecture. | ARI = 0.0690 |
| `ddin_exp31_supervised_centroid.py` | Supervised centroid on Layer 2. | ARI = 0.0690 |

### Key Findings

1. **Sequential ODE breakthrough (v21)**: ARI = 0.0591 — a +61% improvement over static embedding baseline
2. **The 0.06 ceiling**: All reward variants (unsupervised, supervised, contrastive) converge to ~0.06 regardless of:
   - Number of layers (1 or 2)
   - Reward type
   - W structure (zero, random, structured)
3. **Two-layer adds signal**: Exp 29 showed +0.017 delta from hierarchy
4. **The ceiling is a feature**: It's a precise measurement of single-layer heterogeneous ODE reservoir capacity on this specific task — not a universal limitation

### The Publication Case

The Phase 5-6 investigation produces a complete, precise measurement:
- Exact ceiling mapped from every direction
- Architecture capabilities precisely quantified
- Hierarchical processing adds signal but insufficient
- Clean, negative-but-precise result is scientifically valuable

**Functional claim (ready for publication):**
> A zero-weight, 128-neuron sequential ODE reservoir achieves ARI ~0.06 on unsupervised semantic clustering of Sanskrit roots — the practical ceiling for this architecture.

---

## Citation

```bibtex
@article{kumar2026phonosemanticmeta,
  author    = {Kumar, Amit},
  title     = {PhonosemanticMeta Benchmark: Evaluating AI Vector Grounding
               via the Phonosemantic Manifold},
  journal   = {Kaggle Competition + arXiv preprint},
  year      = {2026},
  eprint    = {2604.XXXXX},
  archivePrefix = {arXiv}
}

@article{kumar2026ddin,
  author    = {Kumar, Amit},
  title     = {DDIN: Devavāṇī-Derived Interpretable Network — Sequential Neural
               ODEs for Phonosemantic Grounding},
  journal   = {arXiv preprint},
  year      = {2026},
  eprint    = {2604.XXXXX},
  archivePrefix = {arXiv}
}

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
