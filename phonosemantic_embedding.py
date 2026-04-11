"""
Phonosemantic Embedding Layer
==============================
Replaces the standard nn.Embedding (token_id → arbitrary learned vector)
with a physically grounded representation built on Sanskrit phonology.

Architecture:
    Input: Sanskrit word as phoneme sequence
    Output: Phonosemantic trajectory vector in manifold M

The four dimensions (per phoneme):
    Dim 1 — Articulation locus    l(p)  ∈ ℝ⁶   [throat, palate, cerebral, dental, labial, nasal]
    Dim 2 — Articulation manner   α(p)  ∈ [0,1] [0=full stop → 1=fully open vowel]
    Dim 3 — Phonation type        β(p)  ∈ ℝ²    [voiced ∈{0,1}, breath_force ∈[0,1]]
    Dim 4 — Somatic resonance     ρ(p)  ∈ {1..5} [spinal axis: pelvic floor → throat]

Per-phoneme descriptor: φ(p) ∈ M, dim(M) = 10
Word trajectory: Φ(w) = (φ(p₁), φ(p₂), ..., φ(pₙ)) ∈ Mⁿ

The harmonic coherence metric H(w₁,w₂) replaces cosine similarity.
Weights λ_L, λ_A, λ_R are FREE PARAMETERS — not hardcoded.
"""

import numpy as np
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass

# ─────────────────────────────────────────────────────────────────────────────
# SECTION 1: PHONEME DATABASE
# Every Sanskrit phoneme mapped to its four-dimensional descriptor.
# Source: Shiksha classification (Paninian phonology) + somatic resonance mapping.
#
# Locus vector: [throat, palate, cerebral, dental, labial, nasal]
# Manner:  0.0 = full stop (plosive)
#          0.5 = partial contact (approximant, fricative)
#          1.0 = fully open (vowel)
# Voice:   0 = unvoiced, 1 = voiced
# Force:   0.0 = alpaprana (low breath), 1.0 = mahaprana (high breath/aspirated)
# Resonance: R1=pelvic floor, R2=pelvis/sacral, R3=navel/lumbar,
#            R4=heart/thoracic, R5=throat/cervical
# ─────────────────────────────────────────────────────────────────────────────

PHONEME_DB: Dict[str, Dict] = {

    # ── VOWELS (all articulation locus: throat; all resonance: R5) ──────────
    # The vowel is pure throat energy — no secondary shaping.
    # Compound vowels (e, ai, o, au) are superpositions of two loci.
    'a':    {'locus': [1.0, 0.0, 0.0, 0.0, 0.0, 0.0], 'manner': 1.0, 'voice': 1, 'force': 0.0, 'resonance': 5},
    'aa':   {'locus': [1.0, 0.0, 0.0, 0.0, 0.0, 0.0], 'manner': 1.0, 'voice': 1, 'force': 0.0, 'resonance': 5},
    'i':    {'locus': [1.0, 0.0, 0.0, 0.0, 0.0, 0.0], 'manner': 1.0, 'voice': 1, 'force': 0.0, 'resonance': 5},
    'ii':   {'locus': [1.0, 0.0, 0.0, 0.0, 0.0, 0.0], 'manner': 1.0, 'voice': 1, 'force': 0.0, 'resonance': 5},
    'u':    {'locus': [1.0, 0.0, 0.0, 0.0, 0.0, 0.0], 'manner': 1.0, 'voice': 1, 'force': 0.0, 'resonance': 5},
    'uu':   {'locus': [1.0, 0.0, 0.0, 0.0, 0.0, 0.0], 'manner': 1.0, 'voice': 1, 'force': 0.0, 'resonance': 5},
    'ri':   {'locus': [1.0, 0.0, 0.0, 0.0, 0.0, 0.0], 'manner': 1.0, 'voice': 1, 'force': 0.0, 'resonance': 5},
    'rii':  {'locus': [1.0, 0.0, 0.0, 0.0, 0.0, 0.0], 'manner': 1.0, 'voice': 1, 'force': 0.0, 'resonance': 5},
    'lri':  {'locus': [1.0, 0.0, 0.0, 0.0, 0.0, 0.0], 'manner': 1.0, 'voice': 1, 'force': 0.0, 'resonance': 5},
    # Compound vowels: superposition of two loci (as Shiksha explicitly states)
    'e':    {'locus': [0.5, 0.5, 0.0, 0.0, 0.0, 0.0], 'manner': 1.0, 'voice': 1, 'force': 0.0, 'resonance': 5},
    'ai':   {'locus': [0.5, 0.5, 0.0, 0.0, 0.0, 0.0], 'manner': 1.0, 'voice': 1, 'force': 0.5, 'resonance': 5},
    'o':    {'locus': [0.5, 0.0, 0.0, 0.0, 0.5, 0.0], 'manner': 1.0, 'voice': 1, 'force': 0.0, 'resonance': 5},
    'au':   {'locus': [0.5, 0.0, 0.0, 0.0, 0.5, 0.0], 'manner': 1.0, 'voice': 1, 'force': 0.5, 'resonance': 5},

    # ── KA-VARGA: throat plosives ─────────────────────────────────────────
    # Resonance R4: heart/thoracic region
    'k':    {'locus': [1.0, 0.0, 0.0, 0.0, 0.0, 0.0], 'manner': 0.0, 'voice': 0, 'force': 0.0, 'resonance': 4},
    'kh':   {'locus': [1.0, 0.0, 0.0, 0.0, 0.0, 0.0], 'manner': 0.0, 'voice': 0, 'force': 1.0, 'resonance': 4},
    'g':    {'locus': [1.0, 0.0, 0.0, 0.0, 0.0, 0.0], 'manner': 0.0, 'voice': 1, 'force': 0.0, 'resonance': 4},
    'gh':   {'locus': [1.0, 0.0, 0.0, 0.0, 0.0, 0.0], 'manner': 0.0, 'voice': 1, 'force': 1.0, 'resonance': 4},
    'nga':  {'locus': [1.0, 0.0, 0.0, 0.0, 0.0, 1.0], 'manner': 0.0, 'voice': 1, 'force': 0.0, 'resonance': 4},
    'h':    {'locus': [1.0, 0.0, 0.0, 0.0, 0.0, 0.0], 'manner': 0.5, 'voice': 1, 'force': 1.0, 'resonance': 4},
    'visarga': {'locus': [1.0, 0.0, 0.0, 0.0, 0.0, 0.0], 'manner': 0.5, 'voice': 0, 'force': 1.0, 'resonance': 4},

    # ── CA-VARGA: palate plosives ──────────────────────────────────────────
    # Resonance R4: heart/thoracic region
    'c':    {'locus': [0.0, 1.0, 0.0, 0.0, 0.0, 0.0], 'manner': 0.0, 'voice': 0, 'force': 0.0, 'resonance': 4},
    'ch':   {'locus': [0.0, 1.0, 0.0, 0.0, 0.0, 0.0], 'manner': 0.0, 'voice': 0, 'force': 1.0, 'resonance': 4},
    'j':    {'locus': [0.0, 1.0, 0.0, 0.0, 0.0, 0.0], 'manner': 0.0, 'voice': 1, 'force': 0.0, 'resonance': 4},
    'jh':   {'locus': [0.0, 1.0, 0.0, 0.0, 0.0, 0.0], 'manner': 0.0, 'voice': 1, 'force': 1.0, 'resonance': 4},
    'nya':  {'locus': [0.0, 1.0, 0.0, 0.0, 0.0, 1.0], 'manner': 0.0, 'voice': 1, 'force': 0.0, 'resonance': 4},

    # ── TA-VARGA CEREBRAL (retroflex) ─────────────────────────────────────
    # Resonance R3/R4 transition: navel-to-heart region
    'T':    {'locus': [0.0, 0.0, 1.0, 0.0, 0.0, 0.0], 'manner': 0.0, 'voice': 0, 'force': 0.0, 'resonance': 4},
    'Th':   {'locus': [0.0, 0.0, 1.0, 0.0, 0.0, 0.0], 'manner': 0.0, 'voice': 0, 'force': 1.0, 'resonance': 4},
    'D':    {'locus': [0.0, 0.0, 1.0, 0.0, 0.0, 0.0], 'manner': 0.0, 'voice': 1, 'force': 0.0, 'resonance': 3},
    'Dh':   {'locus': [0.0, 0.0, 1.0, 0.0, 0.0, 0.0], 'manner': 0.0, 'voice': 1, 'force': 1.0, 'resonance': 3},
    'N':    {'locus': [0.0, 0.0, 1.0, 0.0, 0.0, 1.0], 'manner': 0.0, 'voice': 1, 'force': 0.0, 'resonance': 3},

    # ── TA-VARGA DENTAL ────────────────────────────────────────────────────
    # Resonance R3: navel/lumbar region
    't':    {'locus': [0.0, 0.0, 0.0, 1.0, 0.0, 0.0], 'manner': 0.0, 'voice': 0, 'force': 0.0, 'resonance': 3},
    'th':   {'locus': [0.0, 0.0, 0.0, 1.0, 0.0, 0.0], 'manner': 0.0, 'voice': 0, 'force': 1.0, 'resonance': 3},
    'd':    {'locus': [0.0, 0.0, 0.0, 1.0, 0.0, 0.0], 'manner': 0.0, 'voice': 1, 'force': 0.0, 'resonance': 3},
    'dh':   {'locus': [0.0, 0.0, 0.0, 1.0, 0.0, 0.0], 'manner': 0.0, 'voice': 1, 'force': 1.0, 'resonance': 3},
    'n':    {'locus': [0.0, 0.0, 0.0, 1.0, 0.0, 1.0], 'manner': 0.0, 'voice': 1, 'force': 0.0, 'resonance': 3},

    # ── PA-VARGA: labial plosives ──────────────────────────────────────────
    # Resonance R2/R3: pelvis/sacral-to-navel region
    'p':    {'locus': [0.0, 0.0, 0.0, 0.0, 1.0, 0.0], 'manner': 0.0, 'voice': 0, 'force': 0.0, 'resonance': 3},
    'ph':   {'locus': [0.0, 0.0, 0.0, 0.0, 1.0, 0.0], 'manner': 0.0, 'voice': 0, 'force': 1.0, 'resonance': 3},
    'b':    {'locus': [0.0, 0.0, 0.0, 0.0, 1.0, 0.0], 'manner': 0.0, 'voice': 1, 'force': 0.0, 'resonance': 2},
    'bh':   {'locus': [0.0, 0.0, 0.0, 0.0, 1.0, 0.0], 'manner': 0.0, 'voice': 1, 'force': 1.0, 'resonance': 2},
    'm':    {'locus': [0.0, 0.0, 0.0, 0.0, 1.0, 1.0], 'manner': 0.0, 'voice': 1, 'force': 0.0, 'resonance': 2},

    # ── SEMI-VOWELS / APPROXIMANTS ────────────────────────────────────────
    'y':    {'locus': [0.0, 1.0, 0.0, 0.0, 0.0, 0.0], 'manner': 0.5, 'voice': 1, 'force': 0.0, 'resonance': 2},
    'r':    {'locus': [0.0, 0.0, 1.0, 0.0, 0.0, 0.0], 'manner': 0.5, 'voice': 1, 'force': 0.5, 'resonance': 2},
    'l':    {'locus': [0.0, 0.0, 0.0, 1.0, 0.0, 0.0], 'manner': 0.5, 'voice': 1, 'force': 0.0, 'resonance': 2},
    'v':    {'locus': [0.0, 0.0, 0.0, 0.0, 1.0, 0.0], 'manner': 0.5, 'voice': 1, 'force': 0.0, 'resonance': 1},

    # ── SIBILANTS ─────────────────────────────────────────────────────────
    # Resonance R1: pelvic floor region
    'sh_palatal':  {'locus': [0.0, 1.0, 0.0, 0.0, 0.0, 0.0], 'manner': 0.5, 'voice': 0, 'force': 0.5, 'resonance': 1},
    'sh_cerebral': {'locus': [0.0, 0.0, 1.0, 0.0, 0.0, 0.0], 'manner': 0.5, 'voice': 0, 'force': 0.5, 'resonance': 1},
    's':           {'locus': [0.0, 0.0, 0.0, 1.0, 0.0, 0.0], 'manner': 0.5, 'voice': 0, 'force': 0.5, 'resonance': 1},

    # ── ANUSVARA / CHANDRABINDU ───────────────────────────────────────────
    'anusvara': {'locus': [0.0, 0.0, 0.0, 0.0, 0.0, 1.0], 'manner': 0.0, 'voice': 1, 'force': 0.0, 'resonance': 3},
}

# ─────────────────────────────────────────────────────────────────────────────
# SECTION 2: PHONEME DESCRIPTOR — φ(p)
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class PhonemeDescriptor:
    """
    The four-dimensional physical descriptor of a single Sanskrit phoneme.
    This is φ(p) ∈ M.
    
    dim(M) = 6 (locus) + 1 (manner) + 2 (phonation) + 1 (resonance) = 10
    """
    phoneme: str
    locus: np.ndarray      # ℝ⁶  — articulation locus vector (may be superposition)
    manner: float          # ℝ    — degree of openness [0=stop, 1=vowel]
    voice: int             # {0,1} — unvoiced vs voiced
    force: float           # ℝ    — breath force [0=alpaprana, 1=mahaprana]
    resonance: int         # {1..5} — spinal axis resonance locus

    def to_vector(self) -> np.ndarray:
        """
        Flatten φ(p) to a 10-dimensional vector:
        [l₁, l₂, l₃, l₄, l₅, l₆, α, β_voice, β_force, ρ_norm]
        
        Resonance is normalized to [0,1]: ρ_norm = (ρ - 1) / 4
        """
        return np.array([
            *self.locus,           # 6 dims: articulation locus
            self.manner,           # 1 dim:  articulation manner
            float(self.voice),     # 1 dim:  voicing
            self.force,            # 1 dim:  breath force
            (self.resonance - 1) / 4.0  # 1 dim:  resonance (normalized)
        ], dtype=np.float32)

    @property
    def is_vowel(self) -> bool:
        return self.manner == 1.0

    @property
    def primary_locus_name(self) -> str:
        names = ['throat', 'palate', 'cerebral', 'dental', 'labial', 'nasal']
        idx = int(np.argmax(self.locus))
        return names[idx]

    @property
    def resonance_name(self) -> str:
        names = {1: 'pelvic_floor', 2: 'pelvis', 3: 'navel', 4: 'heart', 5: 'throat'}
        return names[self.resonance]

    def __repr__(self) -> str:
        return (f"φ({self.phoneme}): locus={self.primary_locus_name}, "
                f"manner={self.manner:.1f}, voice={self.voice}, force={self.force:.1f}, "
                f"resonance=R{self.resonance}({self.resonance_name})")


def get_phoneme_descriptor(phoneme: str) -> PhonemeDescriptor:
    """
    Look up a phoneme and return its full four-dimensional descriptor φ(p).
    Falls back to neutral vowel 'a' if phoneme not found.
    """
    if phoneme not in PHONEME_DB:
        # Soft fallback — warn but don't crash
        print(f"  [Warning] Unknown phoneme '{phoneme}', defaulting to 'a'")
        phoneme = 'a'
    d = PHONEME_DB[phoneme]
    return PhonemeDescriptor(
        phoneme=phoneme,
        locus=np.array(d['locus'], dtype=np.float32),
        manner=float(d['manner']),
        voice=int(d['voice']),
        force=float(d['force']),
        resonance=int(d['resonance'])
    )


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 3: WORD TRAJECTORY — Φ(w)
# A word is a sequence of phoneme descriptors — a trajectory through M.
# ─────────────────────────────────────────────────────────────────────────────

class PhonosematicTrajectory:
    """
    The full phonosemantic representation of a word:
    Φ(w) = (φ(p₁), φ(p₂), ..., φ(pₙ)) ∈ Mⁿ
    
    This is NOT a single point — it is a trajectory.
    The meaning is encoded in the shape of the path, not a position.
    
    For practical comparison we also provide aggregate representations,
    but the trajectory is the primary object.
    """

    def __init__(self, word: str, phonemes: List[str]):
        self.word = word
        self.phonemes = phonemes
        self.descriptors: List[PhonemeDescriptor] = [
            get_phoneme_descriptor(p) for p in phonemes
        ]
        self._vectors: np.ndarray = np.stack(
            [d.to_vector() for d in self.descriptors]
        )  # shape: (n_phonemes, 10)

    @property
    def length(self) -> int:
        return len(self.phonemes)

    @property
    def vectors(self) -> np.ndarray:
        """Raw trajectory matrix — shape (n, 10)."""
        return self._vectors

    def mean_vector(self) -> np.ndarray:
        """
        Centroid of the trajectory in M.
        Useful for quick comparison but loses sequential information.
        Shape: (10,)
        """
        return self._vectors.mean(axis=0)

    def root_vector(self) -> np.ndarray:
        """
        Phonosemantic descriptor of the initial consonant — the root carrier.
        Per the framework: the root consonant is the primary phonosemantic unit.
        The vowels carry energy; the consonants carry form.
        """
        # Find first consonant (manner < 1.0)
        for d in self.descriptors:
            if not d.is_vowel:
                return d.to_vector()
        # All vowels — return first phoneme
        return self.descriptors[0].to_vector()

    def locus_sequence(self) -> List[str]:
        """The trajectory through articulation loci — the 'path' of the word."""
        return [d.primary_locus_name for d in self.descriptors]

    def resonance_sequence(self) -> List[int]:
        """The trajectory through somatic resonance loci."""
        return [d.resonance for d in self.descriptors]

    def resonance_center(self) -> float:
        """
        Mean resonance position — where on the spinal axis the word
        'lives' on average.
        """
        return np.mean([d.resonance for d in self.descriptors])

    def describe(self) -> str:
        """Human-readable analysis of the word's phonosemantic trajectory."""
        lines = [f"\n{'─'*50}",
                 f"Word: {self.word}",
                 f"Phonemes: {' -> '.join(self.phonemes)}",
                 f"{'─'*50}"]
        for d in self.descriptors:
            vowel_mark = " [vowel: energy carrier]" if d.is_vowel else " [consonant: form carrier]"
            lines.append(f"  {repr(d)}{vowel_mark}")
        lines.append(f"{'─'*50}")
        lines.append(f"Locus path:     {' -> '.join(self.locus_sequence())}")
        lines.append(f"Resonance path: R{' -> R'.join(str(r) for r in self.resonance_sequence())}")
        lines.append(f"Resonance center: R{self.resonance_center():.1f}")
        return '\n'.join(lines)


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 4: HARMONIC COHERENCE METRIC — H(w₁, w₂)
# Replaces cosine similarity between statistical vectors.
# Measures structural resonance between two word trajectories.
#
# H(w₁,w₂) = λ_L·H_L + λ_A·H_A + λ_R·H_R
#
# λ_L, λ_A, λ_R are FREE PARAMETERS — defaults are equal weights (1/3 each)
# until empirically determined via corpus study.
# ─────────────────────────────────────────────────────────────────────────────

class HarmonicCoherence:
    """
    The phonosemantic similarity metric.
    
    Three components:
        H_L: locus coherence — do the words share articulation origin?
        H_A: manner coherence — do they share articulatory gesture type?
        H_R: resonance coherence — do they resonate in the same body region?
    
    The weights λ_L, λ_A, λ_R are not fixed — they are parameters
    to be determined empirically. Default = equal weights.
    """

    def __init__(self, lambda_L: float = 1/3, lambda_A: float = 1/3, lambda_R: float = 1/3):
        assert abs(lambda_L + lambda_A + lambda_R - 1.0) < 1e-6, \
            "Weights must sum to 1.0"
        self.lambda_L = lambda_L
        self.lambda_A = lambda_A
        self.lambda_R = lambda_R

    def _locus_similarity(self, phi1: PhonemeDescriptor, phi2: PhonemeDescriptor) -> float:
        """
        Cosine similarity between locus vectors.
        Handles superposition correctly: e (throat+palate) will have
        partial similarity with both pure throat and pure palate sounds.
        """
        l1, l2 = phi1.locus, phi2.locus
        norm1, norm2 = np.linalg.norm(l1), np.linalg.norm(l2)
        if norm1 == 0 or norm2 == 0:
            return 1.0 if (norm1 == 0 and norm2 == 0) else 0.0
        return float(np.dot(l1, l2) / (norm1 * norm2))

    def _manner_similarity(self, phi1: PhonemeDescriptor, phi2: PhonemeDescriptor) -> float:
        """
        Similarity in articulation manner — degree of openness/closure.
        1.0 = identical manner, 0.0 = maximum difference (stop vs vowel).
        """
        return 1.0 - abs(phi1.manner - phi2.manner)

    def _resonance_similarity(self, phi1: PhonemeDescriptor, phi2: PhonemeDescriptor) -> float:
        """
        Similarity in somatic resonance locus.
        Adjacent spinal regions are more similar than distant ones.
        Normalized so maximum distance (R1 to R5) = 0.0, same region = 1.0.
        """
        return 1.0 - abs(phi1.resonance - phi2.resonance) / 4.0

    def phoneme_coherence(self, phi1: PhonemeDescriptor, phi2: PhonemeDescriptor) -> Tuple[float, float, float, float]:
        """
        Compute H(p₁, p₂) between two individual phonemes.
        Returns (H_total, H_L, H_A, H_R) for interpretability.
        """
        H_L = self._locus_similarity(phi1, phi2)
        H_A = self._manner_similarity(phi1, phi2)
        H_R = self._resonance_similarity(phi1, phi2)
        H = self.lambda_L * H_L + self.lambda_A * H_A + self.lambda_R * H_R
        return H, H_L, H_A, H_R

    def word_coherence(
        self,
        traj1: PhonosematicTrajectory,
        traj2: PhonosematicTrajectory,
        mode: str = 'root'
    ) -> Tuple[float, Dict]:
        """
        Compute H(w₁, w₂) between two word trajectories.
        
        mode options:
            'root'    — compare initial consonants (the phonosemantic carriers)
            'mean'    — compare mean trajectory vectors
            'full'    — compare full trajectories via dynamic time warping
        
        Returns (H_total, breakdown_dict) for interpretability.
        """
        if mode == 'root':
            # Use root consonants — primary phonosemantic unit
            phi1 = None
            phi2 = None
            for d in traj1.descriptors:
                if not d.is_vowel:
                    phi1 = d
                    break
            for d in traj2.descriptors:
                if not d.is_vowel:
                    phi2 = d
                    break
            if phi1 is None:
                phi1 = traj1.descriptors[0]
            if phi2 is None:
                phi2 = traj2.descriptors[0]
            H, H_L, H_A, H_R = self.phoneme_coherence(phi1, phi2)

        elif mode == 'mean':
            # Compare centroids
            v1 = traj1.mean_vector()
            v2 = traj2.mean_vector()
            # Decompose back into components for interpretability
            H_L = float(np.dot(v1[:6], v2[:6]) / (
                np.linalg.norm(v1[:6]) * np.linalg.norm(v2[:6]) + 1e-9))
            H_A = 1.0 - abs(v1[6] - v2[6])
            H_R = 1.0 - abs(v1[9] - v2[9])
            H = self.lambda_L * H_L + self.lambda_A * H_A + self.lambda_R * H_R

        elif mode == 'full':
            # Full trajectory comparison — average over aligned phoneme pairs
            # Simple version: compare all pairs, take max-average (soft DTW)
            scores = []
            for d1 in traj1.descriptors:
                for d2 in traj2.descriptors:
                    h, _, _, _ = self.phoneme_coherence(d1, d2)
                    scores.append(h)
            H = float(np.mean(scores))
            H_L = H_A = H_R = H  # Not decomposed in full mode
        else:
            raise ValueError(f"Unknown mode: {mode}. Choose 'root', 'mean', or 'full'.")

        breakdown = {
            'H_total': H,
            'H_L (locus)': H_L,
            'H_A (manner)': H_A,
            'H_R (resonance)': H_R,
            'lambda_L': self.lambda_L,
            'lambda_A': self.lambda_A,
            'lambda_R': self.lambda_R,
            'mode': mode
        }
        return H, breakdown


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 5: THE EMBEDDING LAYER
# Drop-in replacement for nn.Embedding — no parameters, no training needed
# for the base representation. Can be extended with a learned projection head
# for task-specific fine-tuning.
# ─────────────────────────────────────────────────────────────────────────────

class PhonosematicEmbedding:
    """
    The phonosemantic embedding layer.
    
    This replaces the standard nn.Embedding(vocab_size, d_model) which:
      - Requires training
      - Produces arbitrary learned vectors
      - Has no intrinsic meaning
    
    With a physics-derived embedding that:
      - Requires NO training (zero parameters for base representation)
      - Produces vectors with physically interpretable coordinates
      - Has intrinsic meaning determined by vocal anatomy
    
    Input:  Sanskrit word + phoneme sequence
    Output: Fixed 10-dimensional phonosemantic vector (or projected to d_model)
    
    Optional projection: A learned linear layer can project from dim 10
    to any target dimension d_model for compatibility with downstream
    transformer blocks. This projection can be fine-tuned, but the base
    10-dim representation is fixed by physics, not learned.
    """

    DIM = 10  # Fixed by physics: 6 (locus) + 1 (manner) + 2 (phonation) + 1 (resonance)

    def __init__(
        self,
        lambda_L: float = 1/3,
        lambda_A: float = 1/3,
        lambda_R: float = 1/3,
        projection_dim: Optional[int] = None
    ):
        """
        Args:
            lambda_L: Weight for locus coherence in H metric
            lambda_A: Weight for manner coherence in H metric
            lambda_R: Weight for resonance coherence in H metric
            projection_dim: If set, adds a fixed random projection to this dim
                           (for compatibility with downstream layers)
        """
        self.metric = HarmonicCoherence(lambda_L, lambda_A, lambda_R)
        self.projection_dim = projection_dim
        self._projection_matrix: Optional[np.ndarray] = None

        if projection_dim is not None:
            # Fixed random projection (not learned — preserves interpretability)
            rng = np.random.RandomState(42)  # fixed seed for reproducibility
            self._projection_matrix = rng.randn(self.DIM, projection_dim).astype(np.float32)
            # Normalize columns
            self._projection_matrix /= np.linalg.norm(
                self._projection_matrix, axis=0, keepdims=True)

    def encode(self, word: str, phonemes: List[str]) -> np.ndarray:
        """
        Encode a word as its phonosemantic trajectory mean vector.
        
        Returns: shape (10,) or (projection_dim,) if projection is set
        """
        traj = PhonosematicTrajectory(word, phonemes)
        vec = traj.mean_vector()  # (10,)
        if self._projection_matrix is not None:
            vec = vec @ self._projection_matrix  # (projection_dim,)
        return vec

    def encode_root(self, word: str, phonemes: List[str]) -> np.ndarray:
        """
        Encode using only the root consonant — the primary phonosemantic carrier.
        This is the theoretically preferred mode for root-level comparisons.
        """
        traj = PhonosematicTrajectory(word, phonemes)
        vec = traj.root_vector()  # (10,)
        if self._projection_matrix is not None:
            vec = vec @ self._projection_matrix
        return vec

    def similarity(
        self,
        word1: str, phonemes1: List[str],
        word2: str, phonemes2: List[str],
        mode: str = 'root'
    ) -> Tuple[float, Dict]:
        """
        Compute harmonic coherence H(w₁, w₂) between two words.
        Returns (H_score, full_breakdown) — fully interpretable.
        """
        traj1 = PhonosematicTrajectory(word1, phonemes1)
        traj2 = PhonosematicTrajectory(word2, phonemes2)
        return self.metric.word_coherence(traj1, traj2, mode=mode)

    def get_trajectory(self, word: str, phonemes: List[str]) -> PhonosematicTrajectory:
        """Return the full trajectory object for inspection."""
        return PhonosematicTrajectory(word, phonemes)


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 6: DEMONSTRATION
# ─────────────────────────────────────────────────────────────────────────────

def demo():
    print("=" * 60)
    print("PHONOSEMANTIC EMBEDDING LAYER — DEMONSTRATION")
    print("=" * 60)

    embed = PhonosematicEmbedding()

    # ── Example 1: Aham (अहम्) — the self ───────────────────────────────
    # a (throat) → ha (throat breath) → m (lip seal + nasal resonance)
    # The word traces: emergence from interior → breath outward → return to self
    aham = embed.get_trajectory('aham', ['a', 'h', 'm'])
    print(aham.describe())

    # ── Example 2: Prana (प्राण) — life breath ───────────────────────────
    # p (lip threshold) → r (cerebral fire) → aa (throat open) → N (cerebral nasal hum)
    prana = embed.get_trajectory('prana', ['p', 'r', 'aa', 'N'])
    print(prana.describe())

    # ── Example 3: Namaskara (नमस्कार) — the greeting ────────────────────
    # n (dental dissolution) → m (lip hum) → s (dental) → k (throat) → aa → r (cerebral)
    namaskara = embed.get_trajectory('namaskara', ['n', 'm', 's', 'k', 'aa', 'r', 'a'])
    print(namaskara.describe())

    # ── Harmonic Coherence comparisons ───────────────────────────────────
    print("\n" + "=" * 60)
    print("HARMONIC COHERENCE H(w₁, w₂) — ROOT MODE")
    print("(comparing primary root consonants)")
    print("=" * 60)

    # Same locus group — should have HIGH coherence
    # pavana and prana — both lip-origin, same locus
    H1, b1 = embed.similarity('pavana', ['p', 'a', 'v', 'a', 'n', 'a'],
                               'prana',  ['p', 'r', 'aa', 'N'], mode='root')
    print(f"\npavana vs prana (both labial root):")
    print(f"  H = {H1:.4f}  |  H_L={b1['H_L (locus)']:.3f}, H_A={b1['H_A (manner)']:.3f}, H_R={b1['H_R (resonance)']:.3f}")
    print(f"  -> Same labial root, similar resonance: expect HIGH")

    # Cross-locus — should have LOW coherence
    H2, b2 = embed.similarity('prana', ['p', 'r', 'aa', 'N'],
                               'karma', ['k', 'a', 'r', 'm', 'a'], mode='root')
    print(f"\nprana vs karma (labial vs throat root):")
    print(f"  H = {H2:.4f}  |  H_L={b2['H_L (locus)']:.3f}, H_A={b2['H_A (manner)']:.3f}, H_R={b2['H_R (resonance)']:.3f}")
    print(f"  -> Different locus + resonance: expect LOWER")

    # Same root family — pavana, pavaka, prana (all pa-varga)
    print(f"\n── PA-VARGA ROOT FAMILY ──")
    pa_words = [
        ('prana',  ['p', 'r', 'aa', 'N']),
        ('pavana', ['p', 'a', 'v', 'a', 'n', 'a']),
        ('pavaka', ['p', 'aa', 'v', 'a', 'k', 'a']),
    ]
    for i, (w1, ph1) in enumerate(pa_words):
        for j, (w2, ph2) in enumerate(pa_words):
            if i < j:
                H, b = embed.similarity(w1, ph1, w2, ph2, mode='root')
                print(f"  {w1:10s} ↔ {w2:10s}: H = {H:.4f}")

    # ── Show vector for prana ─────────────────────────────────────────────
    print(f"\n── PRANA vector (mean trajectory in M) ──")
    vec = embed.encode('prana', ['p', 'r', 'aa', 'N'])
    print(f"  φ(prana) = {vec}")
    print(f"  Dimensions: [throat, palate, cerebral, dental, labial, nasal, manner, voice, force, resonance_norm]")

    # ── Interpretability check — what does each dimension mean? ──────────
    print(f"\n── INTERPRETABILITY: What does φ(aham) tell us? ──")
    vec_aham = embed.encode('aham', ['a', 'h', 'm'])
    dims = ['throat', 'palate', 'cerebral', 'dental', 'labial', 'nasal',
            'manner', 'voice', 'force', 'resonance_norm']
    for name, val in zip(dims, vec_aham):
        bar = '█' * int(val * 20)
        print(f"  {name:15s}: {val:.3f}  {bar}")

    print("\n" + "=" * 60)
    print("EMBEDDING LAYER PROPERTIES:")
    print(f"  Base dimension:     {PhonosematicEmbedding.DIM} (fixed by physics)")
    print(f"  Free parameters:    0 (base representation)")
    print(f"  Metric weights:     λ_L=1/3, λ_A=1/3, λ_R=1/3 (empirically TBD)")
    print(f"  Vocabulary size:    {len(PHONEME_DB)} phonemes")
    print(f"  Every dimension:    physically interpretable")
    print("=" * 60)


if __name__ == '__main__':
    demo()
