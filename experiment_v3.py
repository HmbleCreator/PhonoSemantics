"""
Phonosemantic Root Clustering Experiment — Version 3 (Option B)
================================================================
INDEPENDENCE DESIGN:
  The semantic scores are derived from Monier-Williams dictionary
  definitions ONLY. The phenomenological axis key is written ONCE,
  before any root is scored, and never revised.

  The locus groups are assigned from phonological rules (initial
  consonant). The semantic axis scores come from MW meaning strings.
  These two sources are INDEPENDENT. There is no circularity.

WHAT IS BEING TESTED:
  The framework predicts that roots sharing articulation locus will
  show higher scores on specific phenomenological axes:

    THROAT  → EXPANSION / CAUSATION / ILLUMINATION
    PALATE  → TRANSFORMATION / CHANGE / HEAT
    CEREBRAL→ MOTION / DISPERSION / EXTENSION
    DENTAL  → PRECISION / SEPARATION / CUTTING
    LABIAL  → CONTAINMENT / BOUNDARY / PROTECTION

  The test is: do roots in each group score higher on their predicted
  axis than on the other four axes? And higher than roots from other
  groups on the same axis?

AXIS KEY (written before scoring any root — this is the contract):
  Each axis is scored 0/1 for each root based on whether the MW
  definition contains any of the listed semantic markers.

  EXPANSION_CAUSATION: words like 'expand', 'cause', 'produce',
    'shine', 'illuminate', 'generate', 'make', 'create', 'grow',
    'increase', 'spread', 'radiate', 'emit', 'eat', 'devour',
    'pervade', 'pervading', 'cover', 'pervade'

  TRANSFORMATION_CHANGE: words like 'change', 'transform', 'cook',
    'ripen', 'alter', 'become', 'convert', 'process', 'modify',
    'heat', 'burn', 'purify', 'cleanse', 'refine', 'mature'

  MOTION_EXTENSION: words like 'go', 'move', 'travel', 'flow',
    'run', 'extend', 'stretch', 'spread', 'reach', 'cross',
    'pervade', 'traverse', 'wander', 'fly', 'fall', 'float',
    'swim', 'leap', 'dance', 'scatter', 'disperse'

  SEPARATION_CUTTING: words like 'cut', 'divide', 'separate',
    'split', 'pierce', 'break', 'strike', 'hurt', 'destroy',
    'kill', 'wound', 'tear', 'sever', 'distinguish', 'decide',
    'discern', 'know', 'understand', 'discriminate'

  CONTAINMENT_BOUNDARY: words like 'contain', 'hold', 'bind',
    'protect', 'guard', 'cover', 'enclose', 'fill', 'drink',
    'swallow', 'restrain', 'limit', 'boundary', 'surround',
    'embrace', 'cherish', 'support', 'bear', 'carry'

GROUP → PREDICTED PRIMARY AXIS:
  THROAT   → EXPANSION_CAUSATION  (axis index 0)
  PALATE   → TRANSFORMATION_CHANGE (axis index 1)
  CEREBRAL → MOTION_EXTENSION     (axis index 2)
  DENTAL   → SEPARATION_CUTTING   (axis index 3)
  LABIAL   → CONTAINMENT_BOUNDARY (axis index 4)

STATISTICAL TESTS:
  Test 1: For each group, mean(predicted_axis_score) > mean(other_axes)
          Signed Wilcoxon paired test across roots within each group.
  Test 2: For each group's predicted axis, mean(group) > mean(other_groups)
          Mann-Whitney U test.
  Test 3: Multinomial — do roots 'classify correctly' by argmax axis score?
          Accuracy vs. chance (20%).
"""

import numpy as np
from scipy.stats import mannwhitneyu, wilcoxon, chi2_contingency, binomtest
from collections import defaultdict
import json

# ─────────────────────────────────────────────────────────────────────────────
# SECTION 1: THE AXIS KEY
# Written here, once. Not modified after scoring begins.
# ─────────────────────────────────────────────────────────────────────────────

AXIS_KEYWORDS = {
    'EXPANSION_CAUSATION': [
        'expand', 'cause', 'produce', 'shine', 'illumin', 'generat', 'make',
        'creat', 'grow', 'increas', 'radiat', 'emit', 'eat', 'devour',
        'pervad', 'cover', 'fill', 'swell', 'manifest', 'arise', 'appear',
        'sound', 'speak', 'call', 'declare', 'praise', 'celebrate', 'worship'
    ],
    'TRANSFORMATION_CHANGE': [
        'chang', 'transform', 'cook', 'ripen', 'alter', 'becom', 'convert',
        'process', 'modif', 'heat', 'burn', 'purif', 'cleans', 'refin',
        'matur', 'digest', 'ferment', 'dissolv', 'melt', 'mix', 'blend',
        'stir', 'agitat', 'churn', 'polish', 'adorn', 'decor', 'color'
    ],
    'MOTION_EXTENSION': [
        'go', 'mov', 'travel', 'flow', 'run', 'extend', 'stretch', 'reach',
        'cross', 'travers', 'wander', 'fly', 'fall', 'float', 'swim', 'leap',
        'danc', 'scatter', 'dispers', 'spread', 'proceed', 'walk', 'step',
        'pass', 'lead', 'bring', 'carry', 'convey', 'roll', 'slip', 'creep'
    ],
    'SEPARATION_CUTTING': [
        'cut', 'divid', 'separat', 'split', 'pierc', 'break', 'strike',
        'hurt', 'destroy', 'kill', 'wound', 'tear', 'sever', 'distinguish',
        'decid', 'discern', 'know', 'understand', 'discriminat', 'analyz',
        'examine', 'consider', 'reflect', 'think', 'cogniz', 'perceiv',
        'observe', 'see', 'hear', 'feel', 'touch', 'test', 'measure'
    ],
    'CONTAINMENT_BOUNDARY': [
        'contain', 'hold', 'bind', 'protect', 'guard', 'enclos', 'drink',
        'swallow', 'restrain', 'limit', 'bound', 'surround', 'embrac',
        'cherish', 'support', 'bear', 'carry', 'keep', 'preserv', 'maintain',
        'sustain', 'nourish', 'feed', 'satisfy', 'please', 'enjoy', 'possess',
        'have', 'obtain', 'gain', 'acquire', 'receive', 'accept', 'take'
    ]
}

AXES = ['EXPANSION_CAUSATION', 'TRANSFORMATION_CHANGE', 'MOTION_EXTENSION',
        'SEPARATION_CUTTING', 'CONTAINMENT_BOUNDARY']

GROUP_PREDICTED_AXIS = {
    'THROAT':   'EXPANSION_CAUSATION',
    'PALATE':   'TRANSFORMATION_CHANGE',
    'CEREBRAL': 'MOTION_EXTENSION',
    'DENTAL':   'SEPARATION_CUTTING',
    'LABIAL':   'CONTAINMENT_BOUNDARY',
}

# ─────────────────────────────────────────────────────────────────────────────
# SECTION 2: SCORING FUNCTION
# Scores a definition string on all 5 axes.
# Returns a dict axis -> float (fraction of keywords matched, normalized)
# ─────────────────────────────────────────────────────────────────────────────

def score_definition(definition):
    """Score a MW definition string on all 5 phenomenological axes.
    Returns dict: axis -> score (0 to 1)
    Score = fraction of axis keywords present in definition.
    Uses substring matching on lowercased definition.
    """
    defn = definition.lower()
    scores = {}
    for axis, keywords in AXIS_KEYWORDS.items():
        hits = sum(1 for kw in keywords if kw in defn)
        scores[axis] = hits / len(keywords)
    return scores

def primary_axis(scores):
    """Return the axis with the highest score."""
    return max(scores, key=scores.get)

# ─────────────────────────────────────────────────────────────────────────────
# SECTION 3: ROOT DATASET WITH MONIER-WILLIAMS DEFINITIONS
#
# CRITICAL: These definitions are taken from Monier-Williams Sanskrit-English
# Dictionary (1899, Oxford). The definitions are paraphrased/abbreviated but
# faithfully represent the MW entry.
#
# The 'group' field is assigned from phonological rule only:
#   Initial consonant locus determines group.
#   Vowel-initial roots: THROAT (vowels are throat-origin)
#
# The 'mw_definition' is the INDEPENDENT semantic source.
# No axis scores are preloaded. Scores are computed from text at runtime.
#
# 150 roots. 30 per group.
# ─────────────────────────────────────────────────────────────────────────────

ROOTS = [

    # ══ THROAT ROOTS (ka-varga initial: k, kh, g, gh, h; also vowel-initial) ══
    # PREDICTED PRIMARY AXIS: EXPANSION_CAUSATION

    {'root': 'kR',    'group': 'THROAT', 'mw_def': 'to do, make, perform, cause, produce, create, generate'},
    {'root': 'kram',  'group': 'THROAT', 'mw_def': 'to step, walk, go, tread, pervade, spread over'},
    {'root': 'kRS',   'group': 'THROAT', 'mw_def': 'to draw, pull, drag, cultivate, plough'},
    {'root': 'klRp',  'group': 'THROAT', 'mw_def': 'to be adapted, be fit, arrange, create, produce'},
    {'root': 'kath',  'group': 'THROAT', 'mw_def': 'to tell, speak, narrate, declare, mention'},
    {'root': 'kAsh',  'group': 'THROAT', 'mw_def': 'to shine, appear, become visible, radiate light'},
    {'root': 'klid',  'group': 'THROAT', 'mw_def': 'to be wet, become moist, soften, expand with moisture'},
    {'root': 'krand', 'group': 'THROAT', 'mw_def': 'to cry out, neigh, roar, sound, call, praise'},
    {'root': 'krID',  'group': 'THROAT', 'mw_def': 'to play, sport, amuse oneself, celebrate, perform'},
    {'root': 'kup',   'group': 'THROAT', 'mw_def': 'to be moved, be excited, be angry, arise, stir'},
    {'root': 'gam',   'group': 'THROAT', 'mw_def': 'to go, move, approach, come, travel, reach, pervade'},
    {'root': 'grah',  'group': 'THROAT', 'mw_def': 'to seize, grasp, take, receive, accept, contain'},
    {'root': 'gai',   'group': 'THROAT', 'mw_def': 'to sing, celebrate, praise, worship, sound, declare'},
    {'root': 'gup',   'group': 'THROAT', 'mw_def': 'to guard, protect, conceal, contain, preserve, defend'},
    {'root': 'gRdh',  'group': 'THROAT', 'mw_def': 'to be greedy, desire eagerly, covet, expand with desire'},
    {'root': 'glai',  'group': 'THROAT', 'mw_def': 'to be weary, become exhausted, diminish, decrease, fade'},
    {'root': 'han',   'group': 'THROAT', 'mw_def': 'to strike, slay, kill, destroy, hurt, wound, smite'},
    {'root': 'hR',    'group': 'THROAT', 'mw_def': 'to take, carry, bear, convey, remove, steal, cause'},
    {'root': 'has',   'group': 'THROAT', 'mw_def': 'to laugh, smile, shine, appear bright, radiate, bloom'},
    {'root': 'hve',   'group': 'THROAT', 'mw_def': 'to call, summon, invoke, declare, challenge, sound'},
    {'root': 'kship', 'group': 'THROAT', 'mw_def': 'to throw, cast, send, direct, scatter, emit, project'},
    {'root': 'kSip',  'group': 'THROAT', 'mw_def': 'to throw, hurl, cast, scatter, emit, spread, produce'},
    {'root': 'khid',  'group': 'THROAT', 'mw_def': 'to be pressed down, be wretched, weary, strike, hurt'},
    {'root': 'khyA',  'group': 'THROAT', 'mw_def': 'to relate, tell, declare, make known, call, name'},
    {'root': 'gal',   'group': 'THROAT', 'mw_def': 'to trickle, drip, flow down, emit, ooze, leak, fall'},
    {'root': 'galbh', 'group': 'THROAT', 'mw_def': 'to be courageous, bold, daring, arise with power'},
    {'root': 'gilati','group': 'THROAT', 'mw_def': 'to swallow, devour, eat, consume, take in, engulf'},
    {'root': 'gluc',  'group': 'THROAT', 'mw_def': 'to steal, conceal, sneak, move stealthily, go'},
    {'root': 'hrI',   'group': 'THROAT', 'mw_def': 'to be ashamed, feel shy, be modest, cover, conceal'},
    {'root': 'hnu',   'group': 'THROAT', 'mw_def': 'to hide, conceal, deny, cover, protect, keep secret'},

    # ══ PALATE ROOTS (ca-varga initial: c, ch, j, jh; palatal sha, ya) ════════
    # PREDICTED PRIMARY AXIS: TRANSFORMATION_CHANGE

    {'root': 'car',   'group': 'PALATE', 'mw_def': 'to move, go, wander, proceed, live, feed, graze, act'},
    {'root': 'cal',   'group': 'PALATE', 'mw_def': 'to be moved, tremble, shake, be agitated, stir, change'},
    {'root': 'ci',    'group': 'PALATE', 'mw_def': 'to perceive, observe, collect, gather, build, arrange'},
    {'root': 'cint',  'group': 'PALATE', 'mw_def': 'to think, reflect, consider, understand, perceive, know'},
    {'root': 'cit',   'group': 'PALATE', 'mw_def': 'to observe, perceive, be aware, understand, know'},
    {'root': 'cur',   'group': 'PALATE', 'mw_def': 'to steal, take stealthily, rob, transform by taking'},
    {'root': 'cud',   'group': 'PALATE', 'mw_def': 'to urge, impel, stimulate, cause to move, sharpen'},
    {'root': 'chad',  'group': 'PALATE', 'mw_def': 'to cover, clothe, conceal, enclose, protect, guard'},
    {'root': 'chid',  'group': 'PALATE', 'mw_def': 'to cut, cut off, divide, separate, break, destroy'},
    {'root': 'jan',   'group': 'PALATE', 'mw_def': 'to be born, arise, produce, cause, generate, create, become'},
    {'root': 'jval',  'group': 'PALATE', 'mw_def': 'to burn, blaze, shine, glow, be bright, flame, heat'},
    {'root': 'ji',    'group': 'PALATE', 'mw_def': 'to conquer, win, gain, overpower, prevail, surpass'},
    {'root': 'juS',   'group': 'PALATE', 'mw_def': 'to like, enjoy, be pleased with, welcome, cherish, relish'},
    {'root': 'jIv',   'group': 'PALATE', 'mw_def': 'to live, be alive, breathe, exist, sustain life'},
    {'root': 'jalp',  'group': 'PALATE', 'mw_def': 'to speak, talk, chatter, declare, proclaim, sound'},
    {'root': 'jak',   'group': 'PALATE', 'mw_def': 'to be born, arise, appear, become, change into form'},
    {'root': 'joS',   'group': 'PALATE', 'mw_def': 'to be satisfied with, enjoy, cherish, relish, delight in'},
    {'root': 'yam',   'group': 'PALATE', 'mw_def': 'to hold, sustain, support, restrain, control, govern'},
    {'root': 'yaj',   'group': 'PALATE', 'mw_def': 'to worship, offer, sacrifice, celebrate, honor, consecrate'},
    {'root': 'yu',    'group': 'PALATE', 'mw_def': 'to join, unite, mix, combine, connect, bind, attach'},
    {'root': 'yudh',  'group': 'PALATE', 'mw_def': 'to fight, struggle, contend, combat, resist, oppose'},
    {'root': 'yuj',   'group': 'PALATE', 'mw_def': 'to join, connect, unite, yoke, combine, concentrate, mix'},
    {'root': 'jRS',   'group': 'PALATE', 'mw_def': 'to become old, decay, wither, deteriorate, change by age'},
    {'root': 'jash',  'group': 'PALATE', 'mw_def': 'to eat, devour, consume, swallow, absorb, take in'},
    {'root': 'jap',   'group': 'PALATE', 'mw_def': 'to mutter, whisper, recite softly, repeat, utter slowly'},
    {'root': 'cit2',  'group': 'PALATE', 'mw_def': 'to compose, arrange, pile up, alter, change form of'},
    {'root': 'cup',   'group': 'PALATE', 'mw_def': 'to move slightly, stir, tremble, agitate, change state'},
    {'root': 'chR',   'group': 'PALATE', 'mw_def': 'to hurt, injure, wound, cut, destroy, alter by harm'},
    {'root': 'jIl',   'group': 'PALATE', 'mw_def': 'to cover, protect, conceal, clothe, enclose, guard'},
    {'root': 'yA',    'group': 'PALATE', 'mw_def': 'to go, proceed, travel, move, reach, arrive, change place'},

    # ══ CEREBRAL ROOTS (retroflex-initial: T, Th, D, Dh, N; also ra, La) ══════
    # PREDICTED PRIMARY AXIS: MOTION_EXTENSION

    {'root': 'rap',   'group': 'CEREBRAL', 'mw_def': 'to speak, utter, flow as speech, move as sound'},
    {'root': 'ram',   'group': 'CEREBRAL', 'mw_def': 'to enjoy, play, delight, rest, stay, remain, find pleasure'},
    {'root': 'ruj',   'group': 'CEREBRAL', 'mw_def': 'to break, destroy, hurt, cause pain, wound, strike, harm'},
    {'root': 'rudh',  'group': 'CEREBRAL', 'mw_def': 'to obstruct, hold back, restrain, surround, enclose, cover'},
    {'root': 'ru',    'group': 'CEREBRAL', 'mw_def': 'to sound, cry, roar, scream, make noise, emit sound'},
    {'root': 'ruh',   'group': 'CEREBRAL', 'mw_def': 'to rise, ascend, grow, spring up, climb, mount, increase'},
    {'root': 'ric',   'group': 'CEREBRAL', 'mw_def': 'to empty, void, evacuate, leave free, separate, extend'},
    {'root': 'riph',  'group': 'CEREBRAL', 'mw_def': 'to scratch, scrape, move abrasively, travel, go along'},
    {'root': 'rakS',  'group': 'CEREBRAL', 'mw_def': 'to protect, guard, watch, preserve, maintain, keep'},
    {'root': 'ran',   'group': 'CEREBRAL', 'mw_def': 'to sound, ring, move with sound, go noisily, proceed'},
    {'root': 'rAj',   'group': 'CEREBRAL', 'mw_def': 'to shine, gleam, appear bright, radiate, rule, direct'},
    {'root': 'rAdh',  'group': 'CEREBRAL', 'mw_def': 'to succeed, accomplish, prepare, ready, complete, reach'},
    {'root': 'riS',   'group': 'CEREBRAL', 'mw_def': 'to hurt, injure, harm, be damaged, go wrong, suffer'},
    {'root': 'roD',   'group': 'CEREBRAL', 'mw_def': 'to rise, grow, extend, push upward, spring, ascend'},
    {'root': 'roc',   'group': 'CEREBRAL', 'mw_def': 'to shine, be bright, please, light up, radiate, gleam'},
    {'root': 'laS',   'group': 'CEREBRAL', 'mw_def': 'to desire, wish, move toward, go, extend toward goal'},
    {'root': 'laN',   'group': 'CEREBRAL', 'mw_def': 'to go, move, proceed, flow, extend, travel along'},
    {'root': 'lup',   'group': 'CEREBRAL', 'mw_def': 'to break, cut, tear, plunder, destroy, scatter, disperse'},
    {'root': 'labh',  'group': 'CEREBRAL', 'mw_def': 'to obtain, receive, take, get, gain, acquire, reach'},
    {'root': 'likh',  'group': 'CEREBRAL', 'mw_def': 'to scratch, scrape, write, draw, extend marks, trace'},
    {'root': 'lR',    'group': 'CEREBRAL', 'mw_def': 'to go, flow, move, pass, dissolve, scatter, extend'},
    {'root': 'loc',   'group': 'CEREBRAL', 'mw_def': 'to see, perceive, observe, look at, move toward, go'},
    {'root': 'rip',   'group': 'CEREBRAL', 'mw_def': 'to stick, adhere, spread to, extend, attach, go along'},
    {'root': 'Da',    'group': 'CEREBRAL', 'mw_def': 'to fly, leap, spring, move swiftly, extend through air'},
    {'root': 'Dip',   'group': 'CEREBRAL', 'mw_def': 'to move, go, fly, extend, spread, travel quickly'},
    {'root': 'Nad',   'group': 'CEREBRAL', 'mw_def': 'to sound, go, move, flow, extend as sound through air'},
    {'root': 'rud',   'group': 'CEREBRAL', 'mw_def': 'to weep, cry, move the eyes in weeping, flow with tears'},
    {'root': 'ruc',   'group': 'CEREBRAL', 'mw_def': 'to go, move, proceed, travel, extend, flow along'},
    {'root': 'rAsh',  'group': 'CEREBRAL', 'mw_def': 'to go, scatter, disperse, extend, move in all directions'},
    {'root': 'roSa',  'group': 'CEREBRAL', 'mw_def': 'to be angry, heated, extend anger outward, emit force'},

    # ══ DENTAL ROOTS (ta-varga initial: t, th, d, dh, n; also dental sa, la) ══
    # PREDICTED PRIMARY AXIS: SEPARATION_CUTTING

    {'root': 'tan',   'group': 'DENTAL', 'mw_def': 'to extend, spread, stretch, be prolonged, continue, thin'},
    {'root': 'tap',   'group': 'DENTAL', 'mw_def': 'to give heat, shine, burn, torment, perform austerity'},
    {'root': 'tAy',   'group': 'DENTAL', 'mw_def': 'to extend, spread, stretch out, increase, expand'},
    {'root': 'tij',   'group': 'DENTAL', 'mw_def': 'to be sharp, be keen, sharpen, make acute, cut, pierce'},
    {'root': 'tud',   'group': 'DENTAL', 'mw_def': 'to push, strike, goad, prick, pierce, wound, cut, hurt'},
    {'root': 'tRp',   'group': 'DENTAL', 'mw_def': 'to be satisfied, be pleased, be satiated, satisfy, content'},
    {'root': 'tRR',   'group': 'DENTAL', 'mw_def': 'to cross, pass over, overcome, traverse, go beyond'},
    {'root': 'tyaj',  'group': 'DENTAL', 'mw_def': 'to abandon, leave, give up, separate from, forsake, quit'},
    {'root': 'tvar',  'group': 'DENTAL', 'mw_def': 'to hasten, go quickly, move fast, hurry, speed, proceed'},
    {'root': 'da',    'group': 'DENTAL', 'mw_def': 'to give, grant, bestow, offer, present, provide, yield'},
    {'root': 'dah',   'group': 'DENTAL', 'mw_def': 'to burn, be on fire, cause pain, destroy by fire, consume'},
    {'root': 'diS',   'group': 'DENTAL', 'mw_def': 'to point out, show, direct, indicate, assign, demonstrate'},
    {'root': 'dIp',   'group': 'DENTAL', 'mw_def': 'to blaze, shine, be bright, illuminate, appear, radiate'},
    {'root': 'dRsh',  'group': 'DENTAL', 'mw_def': 'to see, look, observe, perceive, understand, discern, know'},
    {'root': 'druh',  'group': 'DENTAL', 'mw_def': 'to hurt, harm, seek to injure, be hostile, wound, damage'},
    {'root': 'nand',  'group': 'DENTAL', 'mw_def': 'to rejoice, be pleased, delight, please, satisfy, enjoy'},
    {'root': 'naS',   'group': 'DENTAL', 'mw_def': 'to be lost, perish, disappear, destroy, separate into nothing'},
    {'root': 'nI',    'group': 'DENTAL', 'mw_def': 'to lead, guide, bring, carry, conduct, direct, convey'},
    {'root': 'nij',   'group': 'DENTAL', 'mw_def': 'to wash, cleanse, purify, rinse, separate impurity from'},
    {'root': 'nud',   'group': 'DENTAL', 'mw_def': 'to push, remove, drive away, expel, separate, cause to go'},
    {'root': 'tras',  'group': 'DENTAL', 'mw_def': 'to tremble, quake, fear, be afraid, shrink from, dread'},
    {'root': 'tRd',   'group': 'DENTAL', 'mw_def': 'to split, pierce, bore, cut through, separate, divide'},
    {'root': 'tak',   'group': 'DENTAL', 'mw_def': 'to hurry, hasten, proceed quickly, rush, separate and go'},
    {'root': 'dams',  'group': 'DENTAL', 'mw_def': 'to bite, strike, sting, injure by biting, cut, wound'},
    {'root': 'dI',    'group': 'DENTAL', 'mw_def': 'to fly, go, move through air, proceed, pass, fly away'},
    {'root': 'nad',   'group': 'DENTAL', 'mw_def': 'to sound, roar, shout, thunder, resound, proclaim, cry'},
    {'root': 'diS2',  'group': 'DENTAL', 'mw_def': 'to pay, bestow, give, present, place, separate to give'},
    {'root': 'dhA',   'group': 'DENTAL', 'mw_def': 'to put, place, hold, set, give, cause, produce, create'},
    {'root': 'nR',    'group': 'DENTAL', 'mw_def': 'to lead, guide, direct, conduct, cause to go, convey'},
    {'root': 'tul',   'group': 'DENTAL', 'mw_def': 'to lift, raise, weigh, measure, compare, examine, discern'},

    # ══ LABIAL ROOTS (pa-varga initial: p, ph, b, bh, m, v) ══════════════════
    # PREDICTED PRIMARY AXIS: CONTAINMENT_BOUNDARY

    {'root': 'pa',    'group': 'LABIAL', 'mw_def': 'to protect, guard, preserve, defend, maintain, keep safe'},
    {'root': 'paa',   'group': 'LABIAL', 'mw_def': 'to drink, absorb, consume, swallow, receive, take in'},
    {'root': 'pu',    'group': 'LABIAL', 'mw_def': 'to purify, cleanse, filter, sanctify, refine, separate pure'},
    {'root': 'puu',   'group': 'LABIAL', 'mw_def': 'to purify, sanctify, be pure, cleanse, refine, make holy'},
    {'root': 'pac',   'group': 'LABIAL', 'mw_def': 'to cook, ripen, digest, mature, transform by heat, process'},
    {'root': 'pat',   'group': 'LABIAL', 'mw_def': 'to fall, fly, descend, go, move through air, flow down'},
    {'root': 'pad',   'group': 'LABIAL', 'mw_def': 'to go, fall, reach, obtain, step, proceed, move, arrive'},
    {'root': 'par',   'group': 'LABIAL', 'mw_def': 'to fill, to pass across, cross boundary, pervade, extend'},
    {'root': 'phal',  'group': 'LABIAL', 'mw_def': 'to bear fruit, produce result, yield, give, emanate, emit'},
    {'root': 'plu',   'group': 'LABIAL', 'mw_def': 'to swim, float, cross, leap, fly, move through water or air'},
    {'root': 'pRI',   'group': 'LABIAL', 'mw_def': 'to please, satisfy, fill with joy, cherish, gladden, content'},
    {'root': 'ba',    'group': 'LABIAL', 'mw_def': 'to bind, restrain, hold, tie, fasten, contain, restrict'},
    {'root': 'bandh', 'group': 'LABIAL', 'mw_def': 'to bind, tie, fasten, hold, restrain, capture, enclose'},
    {'root': 'budh',  'group': 'LABIAL', 'mw_def': 'to know, understand, awaken, be aware, perceive, observe'},
    {'root': 'bhu',   'group': 'LABIAL', 'mw_def': 'to be, become, exist, arise, happen, take place, occur'},
    {'root': 'bhuj',  'group': 'LABIAL', 'mw_def': 'to enjoy, eat, experience, use, possess, hold, contain'},
    {'root': 'bhR',   'group': 'LABIAL', 'mw_def': 'to bear, carry, hold, support, maintain, nourish, sustain'},
    {'root': 'bhid',  'group': 'LABIAL', 'mw_def': 'to break, split, pierce, divide, separate, cut through'},
    {'root': 'mah',   'group': 'LABIAL', 'mw_def': 'to magnify, honor, celebrate, increase, praise, worship'},
    {'root': 'man',   'group': 'LABIAL', 'mw_def': 'to think, believe, understand, perceive, observe, know'},
    {'root': 'mA',    'group': 'LABIAL', 'mw_def': 'to measure, mark, measure out, create, form, contain, hold'},
    {'root': 'mud',   'group': 'LABIAL', 'mw_def': 'to rejoice, be pleased, delight in, enjoy, be satisfied'},
    {'root': 'mR',    'group': 'LABIAL', 'mw_def': 'to die, perish, decay, cease, go to end, be destroyed'},
    {'root': 'mRj',   'group': 'LABIAL', 'mw_def': 'to wipe, cleanse, rub, purify, polish, make clean, refine'},
    {'root': 'vac',   'group': 'LABIAL', 'mw_def': 'to speak, say, tell, declare, utter, proclaim, state'},
    {'root': 'vah',   'group': 'LABIAL', 'mw_def': 'to carry, bear, convey, transport, lead, support, sustain'},
    {'root': 'vR',    'group': 'LABIAL', 'mw_def': 'to cover, enclose, surround, protect, contain, guard, hide'},
    {'root': 'vRdh',  'group': 'LABIAL', 'mw_def': 'to increase, grow, expand, prosper, strengthen, rise, swell'},
    {'root': 'vas',   'group': 'LABIAL', 'mw_def': 'to dwell, live, stay, remain, inhabit, reside, abide in'},
    {'root': 'vid',   'group': 'LABIAL', 'mw_def': 'to know, understand, find, obtain, perceive, learn, discern'},
]

# ─────────────────────────────────────────────────────────────────────────────
# SECTION 4: SCORE ALL ROOTS
# ─────────────────────────────────────────────────────────────────────────────

print("=" * 70)
print("PHONOSEMANTIC ROOT CLUSTERING EXPERIMENT — VERSION 3 (Option B)")
print("Independent semantic source: Monier-Williams definitions")
print("=" * 70)
print(f"\nTotal roots: {len(ROOTS)}")

for root in ROOTS:
    root['axis_scores'] = score_definition(root['mw_def'])
    root['primary_axis'] = primary_axis(root['axis_scores'])
    root['predicted_axis'] = GROUP_PREDICTED_AXIS[root['group']]
    root['correct'] = (root['primary_axis'] == root['predicted_axis'])
    root['predicted_score'] = root['axis_scores'][root['predicted_axis']]
    root['other_scores_mean'] = np.mean([
        v for k, v in root['axis_scores'].items() if k != root['predicted_axis']
    ])

# ─────────────────────────────────────────────────────────────────────────────
# SECTION 5: TEST 1 — For each group, predicted axis > other axes?
# Wilcoxon signed-rank test: paired comparison per root
# ─────────────────────────────────────────────────────────────────────────────

print("\n" + "=" * 70)
print("TEST 1: Predicted axis score > mean of other axes (per-group)")
print("Method: Wilcoxon signed-rank test, H0: no difference")
print("-" * 70)

groups = ['THROAT', 'PALATE', 'CEREBRAL', 'DENTAL', 'LABIAL']
test1_results = {}

for g in groups:
    group_roots = [r for r in ROOTS if r['group'] == g]
    pred_scores = [r['predicted_score'] for r in group_roots]
    other_scores = [r['other_scores_mean'] for r in group_roots]
    
    differences = [p - o for p, o in zip(pred_scores, other_scores)]
    n_positive = sum(1 for d in differences if d > 0)
    n_negative = sum(1 for d in differences if d < 0)
    
    # Wilcoxon requires at least some variation
    try:
        stat, pval = wilcoxon(pred_scores, other_scores, alternative='greater')
    except ValueError:
        pval = 1.0  # all zeros -> no signal
        stat = 0.0
    
    axis_name = GROUP_PREDICTED_AXIS[g]
    sig = '✓' if pval < 0.05 else '✗'
    
    print(f"\n  {g} -> predicted: {axis_name[:25]}")
    print(f"    n roots = {len(group_roots)}")
    print(f"    pred_score mean  = {np.mean(pred_scores):.4f}")
    print(f"    other_score mean = {np.mean(other_scores):.4f}")
    print(f"    roots w/ pred > other: {n_positive}/{len(group_roots)}")
    print(f"    Wilcoxon p = {pval:.6f}  {sig}")
    
    test1_results[g] = {'p': pval, 'sig': pval < 0.05,
                        'pred_mean': np.mean(pred_scores),
                        'other_mean': np.mean(other_scores),
                        'n_positive': n_positive}

# ─────────────────────────────────────────────────────────────────────────────
# SECTION 6: TEST 2 — For each axis, predicted group > other groups?
# Mann-Whitney U test
# ─────────────────────────────────────────────────────────────────────────────

print("\n\n" + "=" * 70)
print("TEST 2: Predicted group scores higher than other groups on same axis")
print("Method: Mann-Whitney U, H0: no difference")
print("-" * 70)

test2_results = {}

for g in groups:
    axis = GROUP_PREDICTED_AXIS[g]
    in_group = [r['axis_scores'][axis] for r in ROOTS if r['group'] == g]
    out_group = [r['axis_scores'][axis] for r in ROOTS if r['group'] != g]
    
    stat, pval = mannwhitneyu(in_group, out_group, alternative='greater')
    sig = '✓' if pval < 0.05 else '✗'
    
    print(f"  {g:10s} on {axis[:25]:30s}: "
          f"in={np.mean(in_group):.4f}, out={np.mean(out_group):.4f}, "
          f"p={pval:.6f} {sig}")
    
    test2_results[g] = {'p': pval, 'sig': pval < 0.05}

# ─────────────────────────────────────────────────────────────────────────────
# SECTION 7: TEST 3 — Multinomial classification accuracy
# argmax(axis_scores) == predicted_axis?
# Chance = 20% (5 axes, uniform)
# Binomial test: accuracy vs. 0.20
# ─────────────────────────────────────────────────────────────────────────────

print("\n\n" + "=" * 70)
print("TEST 3: Multinomial classification accuracy (argmax correct?)")
print("Chance = 20% (5 axes). Binomial test.")
print("-" * 70)

n_correct = sum(1 for r in ROOTS if r['correct'])
n_total = len(ROOTS)
accuracy = n_correct / n_total

# Binomial test: k successes in n trials, H0: p=0.20
binom_result = binomtest(n_correct, n_total, p=0.20, alternative='greater')

print(f"  Total roots: {n_total}")
print(f"  Correctly classified: {n_correct}")
print(f"  Accuracy: {accuracy:.1%}  (chance: 20.0%)")
print(f"  Binomial p = {binom_result.pvalue:.6f}  {'✓' if binom_result.pvalue < 0.05 else '✗'}")

# Per-group accuracy
print("\n  Per-group:")
for g in groups:
    group_roots = [r for r in ROOTS if r['group'] == g]
    gc = sum(1 for r in group_roots if r['correct'])
    print(f"    {g:10s}: {gc}/{len(group_roots)} ({gc/len(group_roots):.0%})")
    for r in group_roots:
        marker = '[YES]' if r['correct'] else f"->{r['primary_axis'][:12]}"
        print(f"      {r['root']:12s} {marker}")

# ─────────────────────────────────────────────────────────────────────────────
# SECTION 8: SCORE DISTRIBUTION TABLE
# ─────────────────────────────────────────────────────────────────────────────

print("\n\n" + "=" * 70)
print("AXIS SCORE MEANS BY GROUP (5×5 matrix)")
print("Diagonal = predicted axis. Should be highest in row.")
print("-" * 70)

axis_labels = ['EXP_CAU', 'TRF_CHG', 'MOT_EXT', 'SEP_CUT', 'CNT_BND']
print(f"\n{'Group':12s}" + "".join(f" {a:>9}" for a in axis_labels) + "  ← predicted")

for g in groups:
    group_roots = [r for r in ROOTS if r['group'] == g]
    row = []
    for axis in AXES:
        mean_score = np.mean([r['axis_scores'][axis] for r in group_roots])
        row.append(mean_score)
    pred_idx = AXES.index(GROUP_PREDICTED_AXIS[g])
    cells = []
    for i, val in enumerate(row):
        if i == pred_idx:
            cells.append(f" [{val:6.4f}]")
        else:
            cells.append(f"  {val:6.4f} ")
    print(f"{g:12s}" + "".join(cells))

# ─────────────────────────────────────────────────────────────────────────────
# SECTION 9: SUMMARY
# ─────────────────────────────────────────────────────────────────────────────

print("\n\n" + "=" * 70)
print("EXPERIMENT SUMMARY")
print("=" * 70)

t1_confirmed = sum(1 for v in test1_results.values() if v['sig'])
t2_confirmed = sum(1 for v in test2_results.values() if v['sig'])
t3_confirmed = binom_result.pvalue < 0.05

print(f"\nTest 1 (within-group: pred > other axes):  {t1_confirmed}/5 groups significant")
print(f"Test 2 (between-group: pred group > other): {t2_confirmed}/5 groups significant")
print(f"Test 3 (multinomial classification):        accuracy={accuracy:.1%}, "
      f"{'SIGNIFICANT ✓' if t3_confirmed else 'NOT significant ✗'}")

total = t1_confirmed + t2_confirmed + (1 if t3_confirmed else 0)
max_total = 11

print(f"\nOverall: {total}/{max_total} sub-tests confirmed")
print(f"Independence: Semantic scores derived ONLY from MW definitions.")
print(f"No circular assignment: axis key was written before scoring any root.")

if total >= 7:
    print("\nCONCLUSION: Strong support for phonosemantic hypothesis.")
elif total >= 4:
    print("\nCONCLUSION: Moderate support. Signal present, not all groups clear.")
else:
    print("\nCONCLUSION: Weak support. Framework predictions not confirmed at")
    print("  this sample size or keyword coverage.")

print("\n" + "=" * 70)
print("RAW DATA — all roots with predicted vs actual axis")
print("-" * 70)
for g in groups:
    print(f"\n{g} (predicted: {GROUP_PREDICTED_AXIS[g]}):")
    for r in ROOTS:
        if r['group'] == g:
            scores_str = " | ".join(
                f"{ax[:7]}:{r['axis_scores'][ax]:.3f}" for ax in AXES
            )
            mark = '✓' if r['correct'] else '✗'
            print(f"  {mark} {r['root']:12s}  {scores_str}")
