"""Linear Probe Experiment — Phonosemantic vs Phoneme-Identity vs Random"""
import numpy as np
from scipy.stats import wilcoxon
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.decomposition import PCA
import warnings
warnings.filterwarnings('ignore')
np.random.seed(42)

LOCUS = {
    'k':[1,0,0,0,0,0],'kh':[1,0,0,0,0,0],'g':[1,0,0,0,0,0],'gh':[1,0,0,0,0,0],
    'h':[1,0,0,0,0,0],'a':[1,0,0,0,0,0],'A':[1,0,0,0,0,0],'i':[1,0,0,0,0,0],
    'I':[1,0,0,0,0,0],'u':[1,0,0,0,0,0],'U':[1,0,0,0,0,0],'e':[1,0,0,0,0,0],
    'o':[1,0,0,0,0,0],'c':[0,1,0,0,0,0],'ch':[0,1,0,0,0,0],'j':[0,1,0,0,0,0],
    'jh':[0,1,0,0,0,0],'y':[0,1,0,0,0,0],'sh':[0,1,0,0,0,0],
    'T':[0,0,1,0,0,0],'Th':[0,0,1,0,0,0],'D':[0,0,1,0,0,0],'Dh':[0,0,1,0,0,0],
    'r':[0,0,1,0,0,0],'L':[0,0,1,0,0,0],'S':[0,0,1,0,0,0],
    't':[0,0,0,1,0,0],'th':[0,0,0,1,0,0],'d':[0,0,0,1,0,0],'dh':[0,0,0,1,0,0],
    's':[0,0,0,1,0,0],'l':[0,0,0,1,0,0],
    'p':[0,0,0,0,1,0],'ph':[0,0,0,0,1,0],'b':[0,0,0,0,1,0],'bh':[0,0,0,0,1,0],
    'v':[0,0,0,0,1,0],'n':[0,0,0,0,0,1],'N':[0,0,0,0,0,1],'m':[0,0,0,0,0,1],
}
MANNER = {'k':0.0,'kh':0.0,'g':0.0,'gh':0.0,'c':0.0,'ch':0.0,'j':0.0,'jh':0.0,
    'T':0.0,'Th':0.0,'D':0.0,'Dh':0.0,'t':0.0,'th':0.0,'d':0.0,'dh':0.0,
    'p':0.0,'ph':0.0,'b':0.0,'bh':0.0,'h':0.3,'sh':0.3,'S':0.3,'s':0.3,
    'n':0.4,'N':0.4,'m':0.4,'y':0.7,'r':0.7,'l':0.7,'L':0.7,'v':0.7,
    'a':1.0,'A':1.0,'i':1.0,'I':1.0,'u':1.0,'U':1.0,'e':1.0,'o':1.0}
PHONATION = {'k':[0,.2],'c':[0,.2],'T':[0,.2],'t':[0,.2],'p':[0,.2],
    'sh':[0,.2],'S':[0,.2],'s':[0,.2],'kh':[0,.8],'ch':[0,.8],'Th':[0,.8],
    'th':[0,.8],'ph':[0,.8],'g':[1,.2],'j':[1,.2],'D':[1,.2],'d':[1,.2],
    'b':[1,.2],'gh':[1,.8],'jh':[1,.8],'Dh':[1,.8],'dh':[1,.8],'bh':[1,.8],
    'n':[1,.3],'N':[1,.3],'m':[1,.3],'y':[1,.3],'r':[1,.5],'l':[1,.3],
    'L':[1,.3],'v':[1,.3],'h':[1,.6],'a':[1,.2],'A':[1,.2],'i':[1,.2],
    'I':[1,.2],'u':[1,.2],'U':[1,.2],'e':[1,.2],'o':[1,.2]}
RESONANCE = {'v':0.0,'sh':0.0,'S':0.0,'s':0.0,'b':.25,'bh':.25,'m':.25,
    'y':.25,'r':.25,'l':.25,'L':.25,'ph':.5,'D':.5,'Dh':.5,'N':.5,'t':.5,
    'th':.5,'d':.5,'dh':.5,'n':.5,'p':.5,'k':.75,'kh':.75,'g':.75,'gh':.75,
    'c':.75,'ch':.75,'j':.75,'jh':.75,'T':.75,'Th':.75,
    'a':1.0,'A':1.0,'i':1.0,'I':1.0,'u':1.0,'U':1.0,'e':1.0,'o':1.0,'h':1.0}

def phi(p):
    p = p.lower()
    return np.array(LOCUS.get(p,[1/6]*6) + [MANNER.get(p,.5)] +
                    PHONATION.get(p,[1,.3]) + [RESONANCE.get(p,.5)], dtype=float)

def get_phonemes(root_str):
    s = root_str.lower(); phonemes = []; i = 0
    while i < len(s) and len(phonemes) < 4:
        two = s[i:i+2] if i+1 < len(s) else ''
        if two in LOCUS: phonemes.append(two); i += 2
        elif s[i] in LOCUS: phonemes.append(s[i]); i += 1
        else: i += 1
    return phonemes if phonemes else ['a']

ROOTS = [
    ('kR','THROAT'),('kram','THROAT'),('kRS','THROAT'),('klRp','THROAT'),
    ('kath','THROAT'),('kAsh','THROAT'),('klid','THROAT'),('krand','THROAT'),
    ('krID','THROAT'),('kup','THROAT'),('gam','THROAT'),('grah','THROAT'),
    ('gai','THROAT'),('gup','THROAT'),('gRdh','THROAT'),('glai','THROAT'),
    ('han','THROAT'),('hR','THROAT'),('has','THROAT'),('hve','THROAT'),
    ('kship','THROAT'),('kSip','THROAT'),('khid','THROAT'),('khyA','THROAT'),
    ('gal','THROAT'),('galbh','THROAT'),('gilati','THROAT'),('gluc','THROAT'),
    ('hrI','THROAT'),('hnu','THROAT'),
    ('car','PALATE'),('cal','PALATE'),('ci','PALATE'),('cint','PALATE'),
    ('cit','PALATE'),('cur','PALATE'),('cud','PALATE'),('chad','PALATE'),
    ('chid','PALATE'),('jan','PALATE'),('jval','PALATE'),('ji','PALATE'),
    ('juS','PALATE'),('jIv','PALATE'),('jalp','PALATE'),('jak','PALATE'),
    ('joS','PALATE'),('yam','PALATE'),('yaj','PALATE'),('yu','PALATE'),
    ('yudh','PALATE'),('yuj','PALATE'),('jRS','PALATE'),('jash','PALATE'),
    ('jap','PALATE'),('cit2','PALATE'),('cup','PALATE'),('chR','PALATE'),
    ('jIl','PALATE'),('yA','PALATE'),
    ('rap','CEREBRAL'),('ram','CEREBRAL'),('ruj','CEREBRAL'),('rudh','CEREBRAL'),
    ('ru','CEREBRAL'),('ruh','CEREBRAL'),('ric','CEREBRAL'),('riph','CEREBRAL'),
    ('rakS','CEREBRAL'),('ran','CEREBRAL'),('rAj','CEREBRAL'),('rAdh','CEREBRAL'),
    ('riS','CEREBRAL'),('roD','CEREBRAL'),('roc','CEREBRAL'),('laS','CEREBRAL'),
    ('laN','CEREBRAL'),('lup','CEREBRAL'),('labh','CEREBRAL'),('likh','CEREBRAL'),
    ('lR','CEREBRAL'),('loc','CEREBRAL'),('rip','CEREBRAL'),('Da','CEREBRAL'),
    ('Dip','CEREBRAL'),('Nad','CEREBRAL'),('rud','CEREBRAL'),('ruc','CEREBRAL'),
    ('rAsh','CEREBRAL'),('roSa','CEREBRAL'),
    ('tan','DENTAL'),('tap','DENTAL'),('tAy','DENTAL'),('tij','DENTAL'),
    ('tud','DENTAL'),('tRp','DENTAL'),('tRR','DENTAL'),('tyaj','DENTAL'),
    ('tvar','DENTAL'),('da','DENTAL'),('dah','DENTAL'),('diS','DENTAL'),
    ('dIp','DENTAL'),('dRsh','DENTAL'),('druh','DENTAL'),('nand','DENTAL'),
    ('naS','DENTAL'),('nI','DENTAL'),('nij','DENTAL'),('nud','DENTAL'),
    ('tras','DENTAL'),('tRd','DENTAL'),('tak','DENTAL'),('dams','DENTAL'),
    ('dI','DENTAL'),('nad','DENTAL'),('diS2','DENTAL'),('dhA','DENTAL'),
    ('nR','DENTAL'),('tul','DENTAL'),
    ('pa','LABIAL'),('paa','LABIAL'),('pu','LABIAL'),('puu','LABIAL'),
    ('pac','LABIAL'),('pat','LABIAL'),('pad','LABIAL'),('par','LABIAL'),
    ('phal','LABIAL'),('plu','LABIAL'),('pRI','LABIAL'),('ba','LABIAL'),
    ('bandh','LABIAL'),('budh','LABIAL'),('bhu','LABIAL'),('bhuj','LABIAL'),
    ('bhR','LABIAL'),('bhid','LABIAL'),('mah','LABIAL'),('man','LABIAL'),
    ('mA','LABIAL'),('mud','LABIAL'),('mR','LABIAL'),('mRj','LABIAL'),
    ('vac','LABIAL'),('vah','LABIAL'),('vR','LABIAL'),('vRdh','LABIAL'),
    ('vas','LABIAL'),('vid','LABIAL'),
]

GROUP_LABELS = {'THROAT':0,'PALATE':1,'CEREBRAL':2,'DENTAL':3,'LABIAL':4}
y = np.array([GROUP_LABELS[g] for _,g in ROOTS])

# Build representations
all_p = sorted(LOCUS.keys())
X_ps, X_oh_raw = [], []
for root_str, _ in ROOTS:
    phonemes = get_phonemes(root_str)
    X_ps.append(np.mean([phi(p) for p in phonemes], axis=0))
    oh = np.zeros(len(all_p))
    for p in phonemes:
        if p in all_p: oh[all_p.index(p)] = 1
    X_oh_raw.append(oh)

X_ps = np.array(X_ps)
X_oh = PCA(n_components=10, random_state=42).fit_transform(np.array(X_oh_raw))
X_rand = np.random.randn(150, 10)

# Run
print("=" * 68)
print("LINEAR PROBE — Phonosemantic Grounding Framework")
print("=" * 68)
print("5-way classification | Logistic Regression | 5-fold CV | chance=20%\n")

skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
N_PERM = 1000
conditions = [('RANDOM (floor)', X_rand),
              ('ONEHOT-PCA (phoneme identity)', X_oh),
              ('PHONOSEMANTIC (geometry)', X_ps)]
results = {}; folds_all = {}

for name, X in conditions:
    pipe = make_pipeline(StandardScaler(),
                         LogisticRegression(max_iter=2000, random_state=42, C=1.0))
    folds = cross_val_score(pipe, X, y, cv=skf, scoring='accuracy')
    results[name] = {'mean':folds.mean(), 'std':folds.std(), 'folds':folds}
    folds_all[name] = folds
    print(f"  {name:<44}  {folds.mean()*100:.1f}% +/- {folds.std()*100:.1f}%")

print("\nPermutation test (n=1000):")
for name, X in conditions:
    pipe = make_pipeline(StandardScaler(),
                         LogisticRegression(max_iter=2000, random_state=42, C=1.0))
    obs = results[name]['mean']
    null = [cross_val_score(pipe, X, np.random.permutation(y), cv=skf,
                            scoring='accuracy').mean() for _ in range(N_PERM)]
    null = np.array(null)
    p = (null >= obs).mean()
    sig = '***' if p<.001 else ('**' if p<.01 else ('*' if p<.05 else 'ns'))
    results[name]['p_perm'] = p
    print(f"  {name[:44]:<46}  p={p:.4f} {sig}")

print("\nPairwise PS vs ONEHOT-PCA (Wilcoxon):")
try:
    _, p_pair = wilcoxon(folds_all['PHONOSEMANTIC (geometry)'],
                         folds_all['ONEHOT-PCA (phoneme identity)'],
                         alternative='greater')
    print(f"  PS > ONEHOT-PCA: p = {p_pair:.4f}  {'*' if p_pair < 0.05 else 'ns'}")
except Exception as e:
    print(f"  {e}")

print("\nSUMMARY")
print(f"  {'Condition':<46} {'Acc':>6}  {'Above chance':>13}  {'p-perm':>7}")
print("-" * 75)
for name, X in conditions:
    r = results[name]
    print(f"  {name:<46} {r['mean']*100:>5.1f}%  +{(r['mean']-.20)*100:>11.1f}pp  {r['p_perm']:>7.4f}")

ps = results['PHONOSEMANTIC (geometry)']['mean']*100
oh = results['ONEHOT-PCA (phoneme identity)']['mean']*100
print(f"\nGeometry over phoneme identity: {ps-oh:+.1f} pp")
print()
