"""Local, evaluation-only utilities; no model or output file opened at import."""
from pathlib import Path
import datetime
import hashlib
import importlib.util
import json
import sys

import numpy as np

sys.dont_write_bytecode = True
U = Path(__file__).resolve().parent
BP = U.parent
H = BP.parent
G = H.parent
O = G.parent
W = G / 'candidate_description_supervision'
R = W / 'source_rank_learning'
F = [2, 3, 4, 5, 7]
AX = {2: 0, 3: 1, 4: 2, 5: 4, 7: 3}
SEEDS = [42, 43, 44]
NAMES = {1: 'original_direct_motion', 2: 'old_non_target_outside_bands',
         3: 'own_terminal_body_hold', 4: 'other_family_terminal_body_hold'}


def read(path):
    return json.loads(Path(path).read_text('utf-8-sig'))


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def stamp():
    return datetime.datetime.now().astimezone().isoformat()


def write(path, value):
    path = Path(path).resolve()
    assert path.is_relative_to(U), path
    with path.open('x', encoding='utf-8') as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write('\n')


def check(pins):
    for path, digest in pins.items():
        assert sha(path) == digest, path


def stop():
    for parent in [U, BP, H, G]:
        for name in ['STOP', 'STOP.txt', 'STOP_REQUESTED', 'STOP_REQUESTED.json']:
            assert not (parent / name).exists()


def load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def summarize(old, new, y, take):
    oc, nc = (old >= 0) == (y > 0), (new >= 0) == (y > 0)
    return dict(count=int(take.sum()), old_correct=int((take & oc).sum()),
                new_correct=int((take & nc).sum()), old_failures=int((take & ~oc).sum()),
                new_failures=int((take & ~nc).sum()), regressions=int((take & oc & ~nc).sum()),
                recoveries=int((take & ~oc & nc).sum()))


def weighted_summary(old, new, y, weights):
    def bce(z):
        z = z.astype(np.float64)
        return float(((np.maximum(z, 0) - z * y + np.log1p(np.exp(-np.abs(z)))) * weights).sum())
    return dict(old_accuracy=float((((old >= 0) == (y > 0)) * weights).sum()),
                new_accuracy=float((((new >= 0) == (y > 0)) * weights).sum()),
                old_BCE=bce(old), new_BCE=bce(new), total_weight=float(weights.sum()))
