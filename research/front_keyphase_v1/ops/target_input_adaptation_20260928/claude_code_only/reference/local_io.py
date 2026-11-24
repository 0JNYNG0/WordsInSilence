"""Local data routing and evidence IO for the WORD data-addition comparison.

The learner itself is the already-approved Claude module reused verbatim from
ops/partial_encoder_pair_20260922/generated_v1.py. Nothing here changes model logic.
"""
from pathlib import Path
import hashlib, json, sys
import numpy as np
import torch

H = Path(__file__).resolve().parent
L = H.parents[1]                                  # research/front_keyphase_v1
PRIOR = L / 'ops/partial_encoder_pair_20260922'   # approved learner + verified caches
R = L / 'runs/word_data_addition_pair_v1'

sys.path[:0] = [str(H), str(PRIOR), str(L), str(L / 'src'),
                str(L.parent / 'sign_model_pilot_v1/src')]

import tools_prep_union93_candidate as u  # noqa: E402

read = lambda p: json.loads(Path(p).read_text(encoding='utf-8-sig'))
sha = lambda p: hashlib.sha256(Path(p).read_bytes()).hexdigest()


def plain(v):
    if isinstance(v, np.ndarray):
        return v.tolist()
    if isinstance(v, np.generic):
        return v.item()
    if isinstance(v, dict):
        return {k: plain(x) for k, x in v.items()}
    if isinstance(v, (tuple, list)):
        return [plain(x) for x in v]
    return v


def write(path, value):
    with Path(path).open('x', encoding='utf-8') as f:
        json.dump(plain(value), f, ensure_ascii=False, indent=2, allow_nan=False)
        f.write('\n')


def policy():
    p = read(H / 'local_policy.json')
    for path, d in p['pins'].items():
        assert sha(path) == d, path
    return p


def parent_model(p):
    ck = torch.load(p['parent']['path'], map_location='cpu', weights_only=True)
    m, classes = u.rebuild_from_payload(ck)
    m.eval()
    torch.set_num_threads(int(p['settings']['cpu_threads']))
    return ck, m, classes


def pack(xs):
    offsets = np.cumsum([0] + [len(x) for x in xs], dtype=np.int64)
    return np.concatenate(xs, axis=0), offsets


def unpack(values, offsets):
    return [values[a:b].copy() for a, b in zip(offsets[:-1], offsets[1:])]
