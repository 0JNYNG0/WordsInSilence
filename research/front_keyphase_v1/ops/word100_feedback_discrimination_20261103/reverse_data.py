"""Four existing public reversed-video caches; no model or media import."""
from pathlib import Path
import importlib.util
import io
import json
import hashlib
import numpy as np

AN=Path(__file__).resolve().parent; O=AN.parent
AJ=O/'word100_local_event_controls_20261103'; AH=O/'word100_local_event_learning_20261103'
G=O/'word100_learned_feedback_integration_20261102'; AA=O/'word100_partial_order_live_20261102'
M=O/'word100_learned_motion_order_20261102'; R=O/'word100_learned_rgb_motion_20261102'
V=O/'word100_learned_body_relation_20261102'
WORDS=('WORD0739','WORD1133','WORD1567','WORD2523')
MODEL_SHA='27aced2983a25de0a980d511e912c6a6bfaa3c9383ecf75fecf06bf316b5d9c6'

def read(p): return json.loads(Path(p).read_text(encoding='utf-8-sig'))
def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda:f.read(1048576),b''):h.update(b)
    return h.hexdigest()
def require(ok,msg):
    if not ok:raise ValueError(msg)
def write_new(p,value):
    with Path(p).open('x',encoding='utf-8') as f:json.dump(value,f,ensure_ascii=False,indent=2,allow_nan=False)
def verify_pins(pins):
    for p,h in pins.items():require(sha(p)==h,'Frozen file changed: '+p)

def aj_helper():
    spec=importlib.util.spec_from_file_location('_an_cached_aj_arithmetic',AJ/'aj_data.py')
    mod=importlib.util.module_from_spec(spec);spec.loader.exec_module(mod)
    return mod

def prepare_existing(path,geometry_stats,rgb_stats,geometry_valid):
    """Alias saved V relation_probability as probability in memory; no new estimate."""
    with np.load(path,allow_pickle=False) as a:
        relation=a['relation_probability'].copy();ids=a['frame_id'].copy();ts=a['timestamp_us'].copy()
    stream=io.BytesIO()
    np.savez(stream,probability=relation,frame_id=ids,timestamp_us=ts)
    stream.seek(0)
    return aj_helper().prepare_arrays(path,stream,geometry_stats,rgb_stats,geometry_valid)

def load_inputs(path=None):
    manifest=read(AN/'reverse_inputs.json' if path is None else path)
    require(manifest['status']=='FROZEN_FOUR_EXISTING_ACTUAL_REVERSE_INPUTS','Input preparation incomplete')
    verify_pins(manifest['pins'])
    require(len(manifest['rows'])==4 and {r['word_id'] for r in manifest['rows']}==set(WORDS),'Four public words only')
    require(manifest['model']['sha256']==MODEL_SHA,'Frozen AH model only')
    geo=read(M/'NORMALIZATION.json');rgb=read(R/'RGB_NORMALIZATION.json');result={}
    for row in manifest['rows']:
        require(row['person']==12 and row['role']=='previously_exposed_development_controls','Only registered public ID12')
        require(row['case_id']==row['word_id']+'_reverse_main','Exact reverse identity')
        p=prepare_existing(row['analysis']['path'],geo,rgb,manifest['geometry_valid'])
        require(p['ids'].tolist()==row['native_frame_ids'] and p['ts'].tolist()==row['timestamp_us'],'Native axis changed')
        require(p['image_available'].tolist()==row['image_available'] and p['evidence_available'].tolist()==row['evidence_available'],'Availability changed')
        require(p['motion_anchor'].tolist()==row['motion_anchor'] and p['observed_fragments']==row['observed_fragments'],'Anchor/fragments changed')
        require(hashlib.sha256(p['x'].tobytes()).hexdigest()==row['normalized_915_sha256'],'915 values changed')
        p.update(case_id=row['case_id'],word_id=row['word_id'],kind='reverse_main',scopes=row['scopes'],
                 source_ledger=row['source_ledger'],cache_binding=row['analysis'],video=row['video'],
                 saved_aa_result=row['saved_aa_result'],prototype_path=np.asarray(row['prototype_path'],dtype=np.int64))
        result[row['case_id']]=p
    require(sum(len(p['scopes']) for p in result.values())==5,'Exactly five fixed physical queries')
    return result
