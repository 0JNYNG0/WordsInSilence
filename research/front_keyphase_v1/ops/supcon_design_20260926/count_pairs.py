"""Read-only count for the supervised-contrastive DESIGN. No training, no model, no user data.

Unit of independence = (recording, word). A recording is the source clip id (segment suffix
_aNN_NNN removed). Several segments of one word in one recording count ONCE. The 224 whole-span
context rows are the SAME recordings as their base rows and never form an independent pair.
A positive pair = same word, different signer (hence different recording).
Also measured on the FROZEN 660 x (16 base + 16 context) schedule the control used: how many rows
per step would actually have an in-batch positive under that exact schedule.
"""
import json, pathlib, re
from collections import Counter, defaultdict
import numpy as np

H = pathlib.Path(__file__).resolve().parent
L = H.parents[1]
meta = json.loads((L / 'ops/context_training_pair_20260921/feature_metadata.json').read_text(encoding='utf-8'))
sched = np.load(L / 'ops/partial_encoder_pair_20260922/train_prefixes.npz', allow_pickle=False)
base, ctx = meta['base'], meta['pair_context']
assert len(base) == 2122 and len(ctx) == 224
assert sched['base_y'].tolist() == [sched['classes'].tolist().index(r['label']) for r in base]
rec = lambda i: re.sub(r'_a\d\d_\d\d\d$', '', i)
# the 18 NIKL demonstration rows carry no signer id: one separate, UNVERIFIED signer group, and a
# different extraction source (MediaPipe, native-like) from the AIHub OpenPose rows
sig = lambda r: r.get('signer_id', 'NIKL_unverified')
for r in base + ctx:
    r['_sig'] = sig(r)

# ---- dataset level -------------------------------------------------------------------------
units = defaultdict(set)               # word -> {(recording, signer)}
rows_per_word = Counter()
for r in base:
    units[r['label']].add((rec(r['id']), r['_sig']))
    rows_per_word[r['label']] += 1
classes = sched['classes'].tolist()
per_class = []
for w in classes:
    u = units.get(w, set())
    by_signer = Counter(s for _, s in u)
    n = len(u)
    same_signer_pairs = sum(c * (c - 1) // 2 for c in by_signer.values())
    cross = n * (n - 1) // 2 - same_signer_pairs
    nikl = by_signer.get('NIKL_unverified', 0)
    per_class.append({'word': w, 'rows': rows_per_word[w], 'independent_units': n,
                      'signers': len(by_signer), 'cross_signer_pairs': cross,
                      'cross_pairs_involving_nikl': nikl * (n - nikl)})
signers_hist = Counter(c['signers'] for c in per_class)
dup_segments = sum(rows_per_word.values()) - sum(c['independent_units'] for c in per_class)

# ---- frozen schedule level ------------------------------------------------------------------
bi, ci = sched['base_indices'], sched['context_indices']
per_step_anchor, per_step_pairs, steps_with_any = [], [], 0
for s in range(bi.shape[0]):
    batch = [(base[i]['label'], rec(base[i]['id']), base[i]['_sig']) for i in bi[s]] + \
            [(ctx[i]['label'], rec(ctx[i]['original_id']), ctx[i]['_sig']) for i in ci[s]]
    anchors, pairs = 0, 0
    for a in range(len(batch)):
        pos = [b for b in range(len(batch)) if b != a and batch[b][0] == batch[a][0]
               and batch[b][2] != batch[a][2] and batch[b][1] != batch[a][1]]
        anchors += bool(pos); pairs += len(pos)
    per_step_anchor.append(anchors); per_step_pairs.append(pairs // 2)
    steps_with_any += anchors > 0
pa = np.array(per_step_anchor)
out = {
    'unit_rule': 'independent unit = (recording, word); positive = same word, different signer; context rows share their base recording',
    'training_rows': {'base': 2122, 'context_same_recordings': 224},
    'signer_ids_in_training': sorted({str(r['_sig']) for r in base}),
    'cross_pairs_involving_nikl_total': sum(c['cross_pairs_involving_nikl'] for c in per_class),
    'repeated_segments_collapsed': dup_segments,
    'classes': 93,
    'classes_by_signer_count': dict(sorted(signers_hist.items())),
    'classes_with_zero_cross_signer_pairs': [c['word'] for c in per_class if c['cross_signer_pairs'] == 0],
    'total_cross_signer_pairs': sum(c['cross_signer_pairs'] for c in per_class),
    'independent_units_total': sum(c['independent_units'] for c in per_class),
    'frozen_schedule': {'steps': int(bi.shape[0]), 'batch': 32,
                        'rows_with_in_batch_positive_per_step': {'mean': round(float(pa.mean()), 2),
                                                                 'median': float(np.median(pa)), 'max': int(pa.max()),
                                                                 'zero_steps': int((pa == 0).sum())},
                        'positive_pairs_per_step_mean': round(float(np.mean(per_step_pairs)), 2),
                        'steps_with_any_positive': steps_with_any},
    'per_class': per_class,
}
(H / 'PAIR_COUNT.json').write_text(json.dumps(out, ensure_ascii=False, indent=1), encoding='utf-8')
print(json.dumps({k: v for k, v in out.items() if k != 'per_class'}, ensure_ascii=False, indent=1))


# ---- correction 2026-09-26: confirmed signers only ------------------------------------------
# Positive pairs use CONFIRMED AIHub signer ids only. The 18 NIKL rows (signer unverified) stay in
# the word classification loss but are removed from the contrastive term entirely - never an
# anchor, never a positive, never a negative - so a same-word NIKL row can never be pushed away.
conf = lambda r: r['_sig'] != 'NIKL_unverified'
units_c = defaultdict(set)
for r in base:
    if conf(r):
        units_c[r['label']].add((rec(r['id']), r['_sig']))
cross_c, classes_c = 0, 0
for w in classes:
    u = units_c.get(w, set())
    bs = Counter(s for _, s in u)
    n = len(u)
    x = n * (n - 1) // 2 - sum(c * (c - 1) // 2 for c in bs.values())
    cross_c += x
    classes_c += x > 0
pa2, pp2, nik, zero2 = [], [], [], 0
for s in range(bi.shape[0]):
    batch = [(base[i]['label'], rec(base[i]['id']), base[i]['_sig']) for i in bi[s]] + \
            [(ctx[i]['label'], rec(ctx[i]['original_id']), ctx[i]['_sig']) for i in ci[s]]
    nik.append(sum(b[2] == 'NIKL_unverified' for b in batch))
    batch = [b for b in batch if b[2] != 'NIKL_unverified']
    a_, p_ = 0, 0
    for a in range(len(batch)):
        pos = [b for b in range(len(batch)) if b != a and batch[b][0] == batch[a][0]
               and batch[b][2] != batch[a][2] and batch[b][1] != batch[a][1]]
        a_ += bool(pos); p_ += len(pos)
    pa2.append(a_); pp2.append(p_ // 2); zero2 += a_ == 0
pa2 = np.array(pa2)
corr = {
    'rule': 'positives: same word, different CONFIRMED AIHub signer, different recording. NIKL rows: word loss only, excluded from the contrastive term (not anchor/positive/negative). Same-signer same-word and same-recording pairs excluded from numerator and denominator.',
    'confirmed_cross_signer_pairs': cross_c,
    'unverified_source_pairs_not_used': out['cross_pairs_involving_nikl_total'],
    'classes_with_confirmed_pairs': classes_c,
    'classes_without_confirmed_pairs': [c['word'] for c in per_class if c['cross_signer_pairs'] - c['cross_pairs_involving_nikl'] == 0],
    'frozen_schedule_confirmed_only': {'rows_with_in_batch_positive_per_step_mean': round(float(pa2.mean()), 2),
                                       'median': float(np.median(pa2)), 'zero_steps': zero2,
                                       'positive_pairs_per_step_mean': round(float(np.mean(pp2)), 2),
                                       'nikl_rows_per_step_mean_word_loss_only': round(float(np.mean(nik)), 3),
                                       'steps_with_any_nikl_row': int(sum(x > 0 for x in nik))},
}
out['correction_20260926_confirmed_only'] = corr
(H / 'PAIR_COUNT.json').write_text(json.dumps(out, ensure_ascii=False, indent=1), encoding='utf-8')
print(json.dumps(corr, ensure_ascii=False, indent=1))
