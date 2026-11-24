"""Bounded research runner: native source-data pair (control vs +9 native clips).

Metadata-only preflight by default; --execute runs the paired comparison using
existing project helpers only. No network, shell, deletion or registration.
Exploratory native source-domain comparison only: video PTS/action completeness
and new-webcam-person performance are NOT verified by this runner.
"""
import argparse, hashlib, json, math, os, sys

SCHEMA = "wis_native_data_pair_plan_v1"
SCOPE = "native_source_data_addition_only"
N_CLASSES = 93
EXP = {"source_train": 2095, "source_validation": 696, "nikl": 18}
TRAIN_SIGNERS = ["REAL%02d" % i for i in range(1, 10)]
DEV_SIGNERS = ["REAL11", "REAL12", "REAL13"]
INT_KEYS = ("seed", "steps", "batch_size", "cpu_threads")
FLOAT_KEYS = ("encoder_lr", "head_lr", "weight_decay")


class UniformSchedule(object):
    """Deterministic replay of a precomputed uniform stream (pure stdlib)."""

    def __init__(self, values):
        self._v = [float(x) for x in values]
        for x in self._v:
            if not (0.0 <= x < 1.0) or x != x:
                raise ValueError("uniform out of [0,1): %r" % (x,))
        self._i = 0

    def integers(self, high):
        high = int(high)
        if high <= 0:
            raise ValueError("high must be positive, got %d" % high)
        if self._i >= len(self._v):
            raise RuntimeError("uniform schedule exhausted at draw %d" % self._i)
        u = self._v[self._i]
        self._i += 1
        return min(int(math.floor(u * high)), high - 1)

    @property
    def consumed(self):
        return self._i

    def digest(self):
        blob = ",".join("%.17g" % x for x in self._v).encode("utf-8")
        return hashlib.sha256(blob).hexdigest()


def _read_json(path):
    with open(path, "r", encoding="utf-8") as fh:
        return json.load(fh)


def _write_json(path, obj):
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(obj, fh, ensure_ascii=False, indent=2, sort_keys=True)


def _sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _under(root, path):
    """True iff `path` is `root` itself or lives strictly below it.

    Containment is decided on fully canonical paths (abspath -> realpath ->
    normcase) so that symlink/junction escapes, case-only differences and
    sibling-prefix collisions ("C:/data" vs "C:/data_extra") cannot fool it.
    Cross-drive comparisons raise ValueError in commonpath -> not contained.
    """
    try:
        r = os.path.normcase(os.path.realpath(os.path.abspath(root)))
        p = os.path.normcase(os.path.realpath(os.path.abspath(path)))
    except (OSError, ValueError, TypeError):
        return False
    if not r or not p:
        return False
    try:
        return os.path.commonpath([r, p]) == r
    except (ValueError, OSError):
        return False


def _resolve(root, rel):
    """Join `rel` onto `root` and return its canonical path, or raise.

    Absolute `rel` values are accepted only if they already fall inside the
    canonical root. Anything that escapes (../, symlink, junction, drive
    switch) raises ValueError instead of returning a usable path.
    """
    if not isinstance(root, str) or not isinstance(rel, str) or not rel:
        raise ValueError('path resolution requires non-empty string root and path')
    try:
        base = os.path.normcase(os.path.realpath(os.path.abspath(root)))
        joined = rel if os.path.isabs(rel) else os.path.join(root, rel)
        cand = os.path.realpath(os.path.abspath(joined))
    except (OSError, ValueError, TypeError) as exc:
        raise ValueError('unresolvable path %r under %r: %s' % (rel, root, exc))
    if os.path.normcase(cand) == base:
        raise ValueError('path %r resolves to the root itself' % (rel,))
    if not _under(root, cand):
        raise ValueError('path %r escapes root %r' % (rel, root))
    return cand


def _pos_int(v):
    return isinstance(v, int) and not isinstance(v, bool) and v > 0


def _pos_float(v):
    return isinstance(v, (int, float)) and not isinstance(v, bool) and math.isfinite(float(v)) and float(v) > 0


import os
import re

_SIGNER_RE = re.compile(r"REAL")


def _strict_signer_token(text, allowed, tag, errors):
    """Extract exactly one well-formed, permitted signer token from ``text``."""
    if not isinstance(text, str):
        errors.append("%s: expected string, got %s" % (tag, type(text).__name__))
        return None
    allowed_set = set(allowed or ())
    found = []
    bad = False
    for m in _SIGNER_RE.finditer(text):
        start, end = m.start(), m.end()
        if start > 0 and (text[start - 1].isascii() and text[start - 1].isalnum()):
            errors.append("%s: signer fragment at %d has letter/digit before 'REAL' in %r"
                          % (tag, start, text))
            bad = True
            continue
        j = end
        while j < len(text) and text[j].isascii() and text[j].isdigit():
            j += 1
        if j == end:
            errors.append("%s: 'REAL' at %d is not followed by digits in %r"
                          % (tag, start, text))
            bad = True
            continue
        if j < len(text) and text[j].isascii() and text[j].isalpha():
            errors.append("%s: signer token at %d has trailing letters in %r"
                          % (tag, start, text))
            bad = True
            continue
        found.append(text[start:j])
    if not found:
        if not bad:
            errors.append("%s: no signer token found in %r" % (tag, text))
        return None
    if bad:
        return None
    if len(found) > 1:
        if len(set(found)) == 1:
            errors.append("%s: signer token %s repeated %d times in %r"
                          % (tag, found[0], len(found), text))
        else:
            errors.append("%s: multiple signer tokens %s in %r"
                          % (tag, ",".join(found), text))
        return None
    token = found[0]
    if token not in allowed_set:
        errors.append("%s: signer token %s is not permitted in %r" % (tag, token, text))
        return None
    return token


def _source_signer(row, allowed, tag, errors):
    """Require identical permitted signer tokens in both the id and the npz name."""
    if not isinstance(row, dict):
        errors.append("%s: expected metadata dict, got %s" % (tag, type(row).__name__))
        return None
    rid = row.get("id")
    npz = row.get("npz")
    if rid is None or npz is None:
        errors.append("%s: metadata row missing 'id' or 'npz'" % tag)
        return None
    if not isinstance(npz, str):
        errors.append("%s: 'npz' must be a string, got %s" % (tag, type(npz).__name__))
        return None
    a = _strict_signer_token(rid, allowed, "%s.id" % tag, errors)
    b = _strict_signer_token(os.path.basename(npz), allowed, "%s.npz" % tag, errors)
    if a is None or b is None:
        return None
    if a != b:
        errors.append("%s: signer mismatch id=%s npz=%s (%r vs %r)" % (tag, a, b, rid, npz))
        return None
    return a

def _role_matches(row, kind):
    expected = {
        "source_train": "training",
        "development": "selection_and_regression_only",
        "nikl": "training_original_and_every_augmentation",
    }
    if not isinstance(row, dict):
        return False
    if kind not in expected:
        return False
    return row.get("role") == expected[kind]


def preflight(plan, out):
    """Read-only validation of the native data-pair plan.

    Performs no mutation, no file writes, no array/checkpoint/model loading and
    no hashing. Returns (report, prepared_metadata); prepared is None whenever
    the plan is structurally invalid or the metadata fails integrity checks.
    Semantic-review blocks gate training only -- they never suppress metadata
    inspection, so prepared is still returned with a gated report.
    """
    import re
    HEX = re.compile(r'\A[0-9a-fA-F]{64}\Z')
    TOK = re.compile(r'REAL\d+')
    S = []  # structural
    D = []  # data integrity
    M = []  # semantic (pending only)

    if isinstance(plan, str):
        try:
            plan = _read_json(plan)
        except (OSError, ValueError, TypeError) as e:
            return _report(['plan JSON unreadable: %s' % e], None), None
    if not isinstance(plan, dict):
        return _report(['plan must be a JSON object'], None), None

    if plan.get('schema') != SCHEMA:
        S.append('plan.schema must be exactly %r' % SCHEMA)
    if plan.get('scope') != SCOPE:
        S.append('plan.scope must be exactly %r' % SCOPE)
    if plan.get('augmentation') is not False:
        S.append('plan.augmentation must be exactly false (no augmentation here)')

    roots = {}
    for k in ('web_root', 'pilot_root'):
        v = plan.get(k)
        if not isinstance(v, str) or not v or not os.path.isabs(v):
            S.append('plan.%s must be an absolute path string' % k)
        elif not os.path.isdir(v):
            S.append('plan.%s does not exist as a directory' % k)
        else:
            roots[k] = os.path.realpath(os.path.abspath(v))
    web = roots.get('web_root')
    pilot = roots.get('pilot_root')

    files = {}
    for k in ('base_inputs', 'candidate_report', 'checkpoint'):
        v = plan.get(k)
        if not isinstance(v, str) or not v or not os.path.isabs(v):
            S.append('plan.%s must be an absolute path string' % k)
            continue
        if web is None:
            continue
        if not _under(web, v):
            S.append('plan.%s must live inside web_root' % k)
        elif not os.path.isfile(v):
            S.append('plan.%s does not exist as a file' % k)
        else:
            files[k] = os.path.realpath(os.path.abspath(v))

    ck = plan.get('checkpoint_sha256')
    if not isinstance(ck, str) or not HEX.match(ck):
        S.append('plan.checkpoint_sha256 must be a 64-character hex digest')

    settings = plan.get('settings')
    if not isinstance(settings, dict):
        S.append('plan.settings must be an object')
        settings = {}
    else:
        for k in INT_KEYS:
            if not _pos_int(settings.get(k)):
                S.append('settings.%s must be a positive integer (bool rejected)' % k)
        for k in FLOAT_KEYS:
            if not _pos_float(settings.get(k)):
                S.append('settings.%s must be a finite positive number' % k)

    lmap = plan.get('label_mapping')
    if not isinstance(lmap, dict) or not lmap:
        S.append('plan.label_mapping must be a non-empty object')
        lmap = {}
    else:
        for a, b in lmap.items():
            if not isinstance(a, str) or not a or not isinstance(b, str) or not b:
                S.append('label_mapping entries must be non-empty string->string')
                break

    if not isinstance(out, str) or not out or not os.path.isabs(out):
        S.append('out must be an absolute path string')
    elif web is not None:
        ok = False
        for sub in ('runs', 'evidence'):
            base = os.path.join(web, sub)
            if os.path.isdir(base) and _under(base, out) and \
               os.path.normcase(os.path.realpath(os.path.abspath(out))) != \
               os.path.normcase(os.path.realpath(os.path.abspath(base))):
                ok = True
        if not ok:
            S.append('out must be a new path strictly below web_root/runs or web_root/evidence')
        if os.path.lexists(out):
            S.append('out already exists (broken symlinks count) -- refusing to reuse')

    if S:
        return _report(S, None), None

    # ---- semantic review (never blocks metadata inspection) ----
    ev_path = None
    rev = plan.get('semantic_review')
    if not isinstance(rev, dict):
        M.append('semantic_review missing: approval is pending (not created here)')
    else:
        if rev.get('status') != 'approved':
            M.append("semantic_review.status must be 'approved' (currently %r)" % rev.get('status'))
        r = rev.get('reviewer')
        if not isinstance(r, str) or not r.strip():
            M.append('semantic_review.reviewer must be a non-empty name')
        e = rev.get('evidence')
        if not isinstance(e, str) or not e:
            M.append('semantic_review.evidence must be a path string')
        else:
            try:
                p = _resolve(web, e)
                if not os.path.isfile(p):
                    M.append('semantic_review.evidence is not an existing local file')
                else:
                    ev_path = p
            except ValueError as exc:
                M.append('semantic_review.evidence rejected: %s' % exc)

    # ---- metadata reads ----
    try:
        base = _read_json(files['base_inputs'])
        cand = _read_json(files['candidate_report'])
    except (OSError, ValueError, TypeError) as e:
        return _report(['metadata JSON unreadable: %s' % e] + M, None), None
    if not isinstance(base, dict) or not isinstance(cand, dict):
        return _report(['base_inputs and candidate_report must be JSON objects'] + M, None), None

    cols = {}
    for k, n in (('source_train', EXP['source_train']),
                 ('source_validation', EXP['source_validation']),
                 ('nikl', EXP['nikl'])):
        v = base.get(k)
        if not isinstance(v, list):
            D.append('base_inputs.%s must be a list' % k)
            cols[k] = []
        else:
            if len(v) != n:
                D.append('base_inputs.%s has %d rows, expected %d' % (k, len(v), n))
            cols[k] = v

    ids, paths, shas = set(), set(), set()

    def common(row, tag, root, kind):
        if not isinstance(row, dict):
            D.append('%s: row is not an object' % tag)
            return None
        rid = row.get('id')
        lab = row.get('label')
        ci = row.get('class_index')
        sha = row.get('npz_sha256')
        rel = row.get('npz')
        bad = False
        if not isinstance(rid, str) or not rid.strip():
            D.append('%s: id must be a non-empty string' % tag); bad = True
        if not isinstance(lab, str) or not lab.strip():
            D.append('%s: label must be a non-empty string' % tag); bad = True
        if isinstance(ci, bool) or not isinstance(ci, int) or not (0 <= ci < N_CLASSES):
            D.append('%s: class_index must be an int in 0..%d' % (tag, N_CLASSES - 1)); bad = True
        if not isinstance(sha, str) or not HEX.match(sha):
            D.append('%s: npz_sha256 must be a 64-character hex digest' % tag); bad = True
        if not isinstance(rel, str) or not rel:
            D.append('%s: npz must be a path string' % tag); bad = True
            p = None
        else:
            try:
                p = _resolve(root, rel)
                if not os.path.isfile(p):
                    D.append('%s: npz file not found: %s' % (tag, rel)); bad = True
            except ValueError as exc:
                D.append('%s: npz rejected: %s' % (tag, exc)); bad = True
                p = None
        if bad:
            return None
        key = os.path.normcase(p)
        low = sha.lower()
        if rid in ids:
            D.append('%s: duplicate id %r' % (tag, rid)); return None
        if key in paths:
            D.append('%s: duplicate npz path %s' % (tag, rel)); return None
        if low in shas:
            D.append('%s: duplicate npz_sha256 %s' % (tag, low)); return None
        ids.add(rid); paths.add(key); shas.add(low)
        return {'sample_id': rid, 'label': lab, 'class_index': ci,
                'npz': p, 'npz_sha256': low, 'kind': kind}

    def token(text, allowed, tag):
        toks = set(TOK.findall(text or ''))
        if len(toks) != 1:
            D.append('%s: expected exactly one REALnn filename token, found %s'
                     % (tag, sorted(toks) or 'none'))
            return None
        t = toks.pop()
        if t not in allowed:
            D.append('%s: signer token %s not permitted here' % (tag, t))
            return None
        return t

    train_items, nikl_items, dev_items = [], [], []
    for i, row in enumerate(cols['source_train']):
        tag = 'source_train[%d]' % i
        it = common(row, tag, pilot, 'source_train')
        if it is None:
            continue
        if not _role_matches(row, 'source_train'):
            D.append("%s: role must be 'training'" % tag); continue
        if _source_signer(row, set(TRAIN_SIGNERS), tag, D) is None:
            continue
        train_items.append(it)
    for i, row in enumerate(cols['nikl']):
        tag = 'nikl[%d]' % i
        it = common(row, tag, web, 'nikl')
        if it is None:
            continue
        if not _role_matches(row, 'nikl'):
            D.append('%s: unexpected historical role value' % tag); continue
        w = row.get('word_id')
        if not isinstance(w, str) or not w.strip():
            D.append('%s: word_id must be a non-empty string' % tag); continue
        nikl_items.append(it)

    idx2lab, lab2idx = {}, {}
    for it in train_items + nikl_items:
        ci, lab = it['class_index'], it['label']
        if idx2lab.setdefault(ci, lab) != lab:
            D.append('class_index %d maps to both %r and %r' % (ci, idx2lab[ci], lab))
        if lab2idx.setdefault(lab, ci) != ci:
            D.append('label %r maps to both index %d and %d' % (lab, lab2idx[lab], ci))
    if set(idx2lab) != set(range(N_CLASSES)):
        D.append('inferred index->label map covers %d/%d classes' % (len(idx2lab), N_CLASSES))

    for i, row in enumerate(cols['source_validation']):
        tag = 'source_validation[%d]' % i
        it = common(row, tag, pilot, 'development')
        if it is None:
            continue
        if not _role_matches(row, 'development'):
            D.append("%s: role must be 'validation'" % tag); continue
        if _source_signer(row, set(DEV_SIGNERS), tag, D) is None:
            continue
        exp = idx2lab.get(it['class_index'])
        if exp is not None and exp != it['label']:
            D.append('%s: label %r disagrees with inferred index map (%r)' % (tag, it['label'], exp))
            continue
        dev_items.append(it)

    # ---- candidate report ----
    if cand.get('status') != 'PASS_NATIVE_CANDIDATE_INPUT_ONLY':
        D.append('candidate_report.status is not PASS_NATIVE_CANDIDATE_INPUT_ONLY')
    if cand.get('pins_unchanged') is not True:
        D.append('candidate_report.pins_unchanged must be exactly true')
    if cand.get('prepared_clips') != 12:
        D.append('candidate_report.prepared_clips must be 12')
    if cand.get('prepared_frames') != 417:
        D.append('candidate_report.prepared_frames must be 417')
    crows = cand.get('rows')
    if not isinstance(crows, list) or len(crows) != 12:
        D.append('candidate_report.rows must be a list of exactly 12 objects')
        crows = crows if isinstance(crows, list) else []

    cand_train, cand_dev, cand_dev_labels = [], [], []
    frames_total, seen_signers = 0, set()
    runs_root = os.path.join(web, 'runs') if web else None
    for i, row in enumerate(crows):
        tag = 'candidate.rows[%d]' % i
        if not isinstance(row, dict):
            D.append('%s: row is not an object' % tag); continue
        cid, sid, role = row.get('clip_id'), row.get('signer_id'), row.get('split_role')
        raw, sha, npz = row.get('raw_label'), row.get('npz_sha256'), row.get('npz')
        fr = row.get('frames')
        if not isinstance(cid, str) or not cid.strip():
            D.append('%s: clip_id must be a non-empty string' % tag); continue
        if isinstance(sid, bool) or not isinstance(sid, int):
            D.append('%s: signer_id must be an integer' % tag); continue
        if role == 'train_candidate':
            allowed = set(TRAIN_SIGNERS)
        elif role == 'development_only':
            allowed = set(DEV_SIGNERS)
        else:
            D.append("%s: split_role must be 'train_candidate' or 'development_only'" % tag); continue
        t = _strict_signer_token(cid, allowed, tag, D)
        if t is None:
            continue
        if t != 'REAL%02d' % sid:
            D.append('%s: clip_id token %s disagrees with signer_id %r' % (tag, t, sid)); continue
        if t in seen_signers:
            D.append('%s: signer %s appears more than once' % (tag, t)); continue
        seen_signers.add(t)
        if row.get('numeric_contract_status') != 'PASS':
            D.append('%s: numeric_contract_status must be PASS' % tag); continue
        if row.get('training_approved') is not False:
            D.append('%s: training_approved is historical and must stay false' % tag); continue
        if isinstance(fr, bool) or not isinstance(fr, int) or fr <= 0:
            D.append('%s: frames must be a positive integer' % tag); continue
        if not isinstance(sha, str) or not HEX.match(sha):
            D.append('%s: npz_sha256 must be a 64-character hex digest' % tag); continue
        if not isinstance(npz, str) or not os.path.isabs(npz) or runs_root is None \
           or not _under(runs_root, npz) or not os.path.isfile(npz):
            D.append('%s: npz must be an existing file inside web_root/runs' % tag); continue
        if not isinstance(raw, str) or not raw.strip():
            D.append('%s: raw_label must be a non-empty string' % tag); continue
        mapped = lmap.get(raw)
        if not isinstance(mapped, str) or mapped not in lab2idx:
            D.append('%s: raw_label %r has no valid label_mapping target' % (tag, raw)); continue
        p = os.path.realpath(os.path.abspath(npz))
        key, low = os.path.normcase(p), sha.lower()
        if cid in ids or key in paths or low in shas:
            D.append('%s: duplicate clip_id/path/sha against earlier rows' % tag); continue
        ids.add(cid); paths.add(key); shas.add(low)
        frames_total += fr
        if role == 'train_candidate':
            cand_train.append(row)
        else:
            cand_dev.append(row)
            cand_dev_labels.append(mapped)

    if frames_total != 417:
        D.append('candidate frames sum to %d, expected 417' % frames_total)
    if len(cand_train) != 9:
        D.append('expected 9 accepted train candidates, got %d' % len(cand_train))
    if len(cand_dev) != 3:
        D.append('expected 3 accepted development candidates, got %d' % len(cand_dev))

    nikl_labels = sorted({it['label'] for it in nikl_items})
    if len(nikl_labels) != EXP['nikl']:
        D.append('expected %d distinct NIKL labels, got %d' % (EXP['nikl'], len(nikl_labels)))

    control = len(train_items) + len(nikl_items)
    treatment = control + len(cand_train)
    source_eval, additional_eval = len(dev_items), len(cand_dev)
    eval_n = source_eval + additional_eval
    tset = set(nikl_labels)
    tcounts = {lab: 0 for lab in nikl_labels}
    target_rows = 0
    for lab in [it['label'] for it in dev_items] + cand_dev_labels:
        if lab in tset:
            tcounts[lab] += 1
            target_rows += 1
    gt_classes = sum(1 for v in tcounts.values() if v > 0)

    counts = {'control': control, 'treatment': treatment, 'eval': eval_n,
              'source_eval': source_eval, 'additional_eval': additional_eval,
              'target_eval': target_rows, 'target_gt_classes': gt_classes,
              'nikl_labels': len(nikl_labels)}
    for name, got, want in (('control', control, 2113), ('treatment', treatment, 2122),
                            ('eval', eval_n, 699), ('target_eval', target_rows, 33),
                            ('target_gt_classes', gt_classes, 6)):
        if got != want:
            D.append('count %s is %d, expected %d' % (name, got, want))

    if D:
        rep = _report(D + M, None)
        rep['status'] = 'invalid'
        rep['observed_counts'] = counts
        return rep, None

    prepared = {
        'settings': dict(settings), 'label_mapping': dict(lmap),
        'index_to_label': {int(k): v for k, v in idx2lab.items()},
        'index_order': list(range(N_CLASSES)),
        'train_items': train_items, 'nikl_items': nikl_items, 'dev_items': dev_items,
        'candidate_train': cand_train, 'candidate_dev': cand_dev,
        'nikl_labels': nikl_labels, 'counts': counts,
        'target_counts': tcounts, 'evidence': ev_path,
    }
    return _report(M, prepared), prepared


def _report(blocks, prepared):
    """Build the immutable read-only report dict (no files are written)."""
    b = [str(x) for x in (blocks or [])]
    rep = {
        'schema': SCHEMA, 'scope': SCOPE, 'augmentation': False,
        'status': 'ready' if not b else 'gated',
        'blocks': b, 'trained_models': 0, 'arrays_loaded': 0,
        'note': ('native source-domain exploratory only; PTS/action completeness '
                 'and new-webcam-person performance unverified'),
    }
    if isinstance(prepared, dict):
        counts = prepared.get('counts') or {}
        tc = prepared.get('target_counts') or {}
        labels = prepared.get('nikl_labels') or sorted(tc)
        per = {lab: int(tc.get(lab, 0)) for lab in labels}
        supported = sum(1 for v in per.values() if v > 0)
        rep['counts'] = dict(counts)
        rep['target18_available_gt'] = {
            'per_label_gt_counts': per,
            'target_labels_declared': len(labels),
            'supported_class_count': supported,
            'total_target_rows': sum(per.values()),
            'unrepresented_labels': sorted(k for k, v in per.items() if v == 0),
            'coverage_note': ('labels with 0 ground-truth rows are unevaluated, not successes; '
                              'measured coverage is %d GT class(es), not %d-class coverage'
                              % (supported, len(labels))),
        }
    return rep


def _load_items(u, build_channels, np, prepared, mapping):
    def finite(x, who, frames=None):
        a = np.asarray(x, dtype=np.float32)
        if a.ndim != 2 or a.shape[1] != 280 or not np.isfinite(a).all():
            raise RuntimeError("%s: x must be finite Nx280, got %s" % (who, (a.shape,)))
        if frames is not None and a.shape[0] != frames:
            raise RuntimeError("%s: frames %d != reported %d" % (who, a.shape[0], frames))
        return a

    out = {"train": [], "nikl": [], "dev": [], "cand_train": [], "cand_dev": []}
    for it in prepared["train_items"]:
        x = u._channels_from_source(it["npz"], it["npz_sha256"])["x"]
        out["train"].append(dict(it, x=finite(x, it["sample_id"]), origin="source"))
    for it in prepared["nikl_items"]:
        nat, present, ts = u._native_from_nikl(it["npz"], it["npz_sha256"])
        x = build_channels(nat, present, ts)["x"]
        out["nikl"].append(dict(it, x=finite(x, it["sample_id"]), origin="nikl_original"))
    for it in prepared["dev_items"]:
        x = u._channels_from_source(it["npz"], it["npz_sha256"])["x"]
        out["dev"].append(dict(it, x=finite(x, it["sample_id"]), origin="source_dev"))
    idx = {v: k for k, v in prepared["index_to_label"].items()}
    for key, rows in (("cand_train", prepared["candidate_train"]), ("cand_dev", prepared["candidate_dev"])):
        for r in rows:
            if _sha256(r["npz"]) != r["npz_sha256"]:
                raise RuntimeError("candidate %s: sha256 mismatch" % r["clip_id"])
            with np.load(r["npz"], allow_pickle=False) as f:
                x = np.array(f["x"], dtype=np.float32, copy=True)
            lab = mapping[r["raw_label"]]
            out[key].append({"sample_id": r["clip_id"], "label": lab, "class_index": idx[lab],
                             "x": finite(x, r["clip_id"], int(r["frames"])), "origin": "native_candidate"})
    return out


def _summarize(ev, classes):
    gt = dict(ev.get("ground_truth_counts") or {})
    recall, conf = {}, {}
    for c in classes:
        n = int(gt.get(c, 0))
        recall[c] = None if n == 0 else sum(1 for r in ev["rows"] if r["true"] == c and r["pred"] == c) / float(n)
    for r in ev["rows"]:
        k = "%s>%s" % (r["true"], r["pred"])
        conf[k] = conf.get(k, 0) + 1
    return {"n": ev["n"], "correct": ev["correct"], "accuracy": ev["accuracy"], "macro_f1": ev["macro_f1"],
            "macro_f1_denominator": ev["macro_f1_denominator"], "ground_truth_counts": gt,
            "per_class_recall": recall, "sparse_confusion": conf, "rows": ev["rows"]}


def _target_eval(u, model, eval_all, classes, nikl_labels):
    """Evaluate only on clips whose TRUE label is one of the target labels.

    Clip filtering must happen before evaluate(): evaluate's restrict_to only
    masks logits, so unfiltered off-target ground truth would inflate the
    denominator. Model classes are passed through untouched.
    """
    targets = set(nikl_labels or ())
    filtered = [row for row in (eval_all or ())
                if isinstance(row, dict) and row.get('label') in targets]
    return u.evaluate(model, filtered, classes, restrict_to=nikl_labels)


def run(plan, prepared, out):
    web, pilot = plan["web_root"], plan["pilot_root"]
    for p in (web, os.path.join(web, "src"), os.path.join(pilot, "src")):
        if p not in sys.path:
            sys.path.insert(0, p)
    import numpy as np, torch
    import tools_prep_union93_candidate as u
    from pilot.data import build_channels

    st = prepared["settings"]
    torch.set_num_threads(int(st["cpu_threads"]))
    if _sha256(plan["checkpoint"]) != plan["checkpoint_sha256"]:
        raise RuntimeError("checkpoint sha256 mismatch")
    payload = torch.load(plan["checkpoint"], weights_only=True, map_location="cpu")
    probe, classes = u.rebuild_from_payload(payload)
    del probe
    if list(classes) != [prepared["index_to_label"][i] for i in range(N_CLASSES)]:
        raise RuntimeError("checkpoint classes do not match metadata index->label mapping")

    data = _load_items(u, build_channels, np, prepared, prepared["label_mapping"])
    control = data["train"] + data["nikl"]
    treatment = control + data["cand_train"]
    eval_all = data["dev"] + data["cand_dev"]
    if (len(control), len(treatment), len(eval_all)) != (2113, 2122, 699):
        raise RuntimeError("arm sizes %d/%d/%d unexpected" % (len(control), len(treatment), len(eval_all)))

    steps, bs = int(st["steps"]), int(st["batch_size"])
    uniforms = np.random.default_rng(int(st["seed"])).random(2 * steps * bs).tolist()
    setting = {"optimizer_steps": steps, "batch": {"total": bs, "source": bs, "nikl": 0},
               "lr": {"encoder": float(st["encoder_lr"]), "head": float(st["head_lr"])},
               "weight_decay": float(st["weight_decay"]), "seed": int(st["seed"]),
               "batch_size": bs, "cpu_threads": int(st["cpu_threads"]), "augmentation": False, "scope": SCOPE}

    def blocked(*a, **k):
        raise AssertionError("augment_nikl reached: augmentation is disabled for this run")

    results, original_aug = {}, u.augment_nikl
    u.augment_nikl = blocked
    try:
        for name, items in (("control", control), ("treatment", treatment)):
            u.set_seeds(int(st["seed"]))
            model, _ = u.rebuild_from_payload(payload)
            sched = UniformSchedule(uniforms)
            train_out = u.run_training(model, sorted(items, key=lambda d: d["sample_id"]), [],
                                       classes, setting, sched, log=print)
            evs = {"full": u.evaluate(model, eval_all, classes, restrict_to=None),
                   "source696": u.evaluate(model, data["dev"], classes, restrict_to=None),
                   "additional3": u.evaluate(model, data["cand_dev"], classes, restrict_to=None),
                   "target_nikl18": _target_eval(u, model, eval_all, classes, prepared["nikl_labels"])}
            for ev in evs.values():
                if not math.isfinite(float(ev["accuracy"])) or not math.isfinite(float(ev["macro_f1"])):
                    raise RuntimeError("non-finite evaluation metric in arm %s" % name)
            results[name] = {"train": train_out, "uniforms_consumed": sched.consumed,
                             "evaluations": {k: _summarize(v, classes) for k, v in evs.items()}}
            ck = dict(payload)
            ck.update({"state_dict": model.state_dict(), "classes": list(classes), "step": steps,
                       "fixed_setting": setting, "settings": setting, "augmentation": False, "scope": SCOPE,
                       "parent_checkpoint_sha256": plan["checkpoint_sha256"], "arm": name})
            torch.save(ck, os.path.join(out, "checkpoint_%s_final.pt" % name))
    finally:
        u.augment_nikl = original_aug

    meta = {"schema": SCHEMA, "scope": SCOPE, "augmentation": False, "settings": setting,
            "counts": prepared["counts"], "schedule_digest": UniformSchedule(uniforms).digest(),
            "uniforms": len(uniforms), "parent_checkpoint_sha256": plan["checkpoint_sha256"],
            "semantic_review": plan["semantic_review"], "label_mapping": prepared["label_mapping"],
            "target18_available_gt": {"original_classes": 5, "original_rows": 30, "added_classes": 1,
                                      "added_rows": 3, "note": "not 18-class coverage"},
            "promotion": "none; exploratory native source-domain comparison only"}
    _write_json(os.path.join(out, "metadata.json"), meta)
    _write_json(os.path.join(out, "results.json"), {"metadata": meta, "arms": results})
    return 0


def main(argv=None):
    ap = argparse.ArgumentParser(description="native source data-pair runner (preflight by default)")
    ap.add_argument("--plan", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--execute", action="store_true")
    args = ap.parse_args(argv)

    plan = _read_json(args.plan)
    out = os.path.abspath(args.out)
    web = plan.get("web_root", "")
    allowed = [os.path.join(web, "runs"), os.path.join(web, "evidence")] if os.path.isabs(str(web)) else []
    if not any(_under(a, out) and os.path.abspath(out) != os.path.abspath(a) for a in allowed):
        sys.stderr.write("refusing: --out must be strictly below web_root/runs or web_root/evidence\n")
        return 2
    if os.path.exists(out):
        sys.stderr.write("refusing: --out already exists\n")
        return 2

    report, prepared = preflight(plan, out)
    os.makedirs(out)
    _write_json(os.path.join(out, "preflight.json"), report)
    if report["status"] != "ready":
        return 2
    if not args.execute:
        return 0
    try:
        return run(plan, prepared, out)
    except BaseException as exc:
        _write_json(os.path.join(out, "error.json"),
                    {"error": type(exc).__name__, "message": str(exc), "cleanup": "none"})
        sys.stderr.write("execution failed: %s: %s\n" % (type(exc).__name__, exc))
        return 1


if __name__ == "__main__":
    sys.exit(main())
