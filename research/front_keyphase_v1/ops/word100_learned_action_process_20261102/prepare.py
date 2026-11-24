import traceback,shutil
from process_model import *

def main():
    assert not (B/'PREPARED.json').exists(); torch.set_num_threads(2)
    assert shutil.disk_usage(B).free>1024**3
    rows=add_rgb(old.inputs(True));assert len(rows)==216 and {r['person'] for r in rows}==set(range(1,10))
    pins={};manifest=[]
    (B/'prepared').mkdir(exist_ok=True)
    for r in rows:
        pins[r['input']]=r['input_sha256'];pins.update(dict(r['extra_pins']))
        dest=B/'prepared'/f"{r['clip_id']}_{r['kind']}.npz"; assert not dest.exists()
        np.savez_compressed(dest,**{k:r[k] for k in ['x','y','z','m','current','available','rgb','physical']})
        meta={k:v for k,v in r.items() if k not in ['x','y','z','m','current','available','rgb','physical']};meta.update(cache=str(dest),cache_sha256=sha(dest));manifest.append(meta)
    assert all(sha(p)==h for p,h in pins.items())
    exclusive(B/'PREPARED.json',dict(at=clock(),rows=manifest,pins=pins,known_role=sum(int((r['y']>=0).sum()) for r in rows),known_physical=sum(int((r['physical']>=0).sum()) for r in rows),from_to_pairs=sum(len(r['pairs']) for r in rows),new_sensor_extraction=0,new_backbone_inference=0,new_fit_observer_input_pass=1))
    exclusive(B/'SUPERVISION_AUDIT.json',dict(
      direct='A visual sparse roles; S/T visual sparse physical configurations/rest; mapping through actual public video transforms.',
      partial='Source clauses have partial phrase support. Physical from-to endpoint contrasts are not linguistic error labels. No exact contact or full-sign correctness.',
      unknown='Unlabelled frames stay -1. No filled dense process truth, unknown=transit, or semantic promotion of context-only cards.',
      split='FIT recorded01-09 only. No H14/15,ID16,private or mentor input.',
      roles=sum(int((r['y']>=0).sum()) for r in rows),physical=sum(int((r['physical']>=0).sum()) for r in rows),pairs=sum(len(r['pairs']) for r in rows),
      sources={str(p):sha(p) for p in [A/'ROLE_TARGETS.json',S/'PHYSICAL_STATE_TARGETS.json',T/'BACKGROUND_TARGETS.json',S/'LEARNED_STATE_GRAPH.json',P/'GUIDE_SUPPORT.json']}))
    print('prepared',len(rows),'role',sum((r['y']>=0).sum() for r in rows),'physical',sum((r['physical']>=0).sum() for r in rows),'pairs',sum(len(r['pairs']) for r in rows),flush=True)

if __name__=='__main__':
    try:main()
    except BaseException:
        if not (B/'PREP_FIRST_FAILURE.txt').exists():(B/'PREP_FIRST_FAILURE.txt').write_text(traceback.format_exc(),encoding='utf-8')
        raise
