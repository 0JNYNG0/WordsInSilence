"""Process-local compatibility and paths; upstream code and base packages stay intact."""
from pathlib import Path
import os
import sys
import inspect
import numpy as np

UNIT = Path(__file__).resolve().parent
SOURCE = UNIT / 'external/hamer-3a01849f4148352e9260b69bf28b65d1671a4905'


def configure():
    os.environ['PYOPENGL_PLATFORM'] = 'win32'
    os.environ['TORCH_HOME'] = str(UNIT / 'cache/torch')
    if not hasattr(inspect, 'getargspec'):
        inspect.getargspec = inspect.getfullargspec
    for name, value in {'bool': bool, 'int': int, 'float': float, 'complex': complex,
                        'object': object, 'str': str, 'unicode': str}.items():
        if name not in np.__dict__:
            setattr(np, name, value)
    sys.path.insert(0, str(SOURCE))


def load_model():
    configure()
    import torch
    from hamer.configs import get_config
    from hamer.models import HAMER
    from yacs.config import CfgNode
    import json, time
    torch.set_num_threads(4)
    assets = UNIT / 'private_assets/hamer'
    checkpoint = next(assets.rglob('hamer.ckpt'))
    cfgpath = next(assets.rglob('model_config.yaml'))
    cfg = get_config(str(cfgpath), update_cachedir=False)
    cfg.defrost()
    cfg.MODEL.BBOX_SHAPE = [192, 256]
    if 'PRETRAINED_WEIGHTS' in cfg.MODEL.BACKBONE:
        cfg.MODEL.BACKBONE.pop('PRETRAINED_WEIGHTS')
    cfg.MANO.MODEL_PATH = str(UNIT / 'private_assets/mano')
    cfg.MANO.MEAN_PARAMS = str(next(assets.rglob('mano_mean_params.npz')))
    cfg.freeze()
    start = time.perf_counter()
    # Official, locally pinned checkpoint: explicit full loader for older Lightning metadata.
    stored = torch.load(checkpoint, map_location='cpu', weights_only=False, mmap=True)
    model = HAMER(cfg, init_renderer=False)
    state = stored['state_dict']
    incompatible = model.load_state_dict(state, strict=False)
    learned = {n for n, _ in model.named_parameters()}
    missing_learned = sorted(learned.intersection(incompatible.missing_keys))
    if missing_learned or incompatible.unexpected_keys:
        raise RuntimeError(f'Unexpected checkpoint mismatch: {incompatible}')
    model.eval()
    model.requires_grad_(False)
    rec = {'torch': torch.__version__, 'device': 'cpu', 'threads': 4,
           'load_seconds': time.perf_counter()-start,
           'missing_keys': list(incompatible.missing_keys),
           'unexpected_keys': list(incompatible.unexpected_keys),
           'missing_learned_parameters': missing_learned,
           'parameters': sum(p.numel() for p in model.parameters()),
           'training': False}
    (UNIT/'MODEL_LOAD.json').write_text(json.dumps(rec, indent=2), encoding='utf-8')
    del stored, state
    return model, cfg
