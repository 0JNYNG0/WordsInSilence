"""Frozen all-TRAIN replay, comparing one-arm evidence to previous two-arm rule."""
import sys,importlib.util,json,hashlib,shutil,time
from pathlib import Path
import numpy as np
U=Path(__file__).resolve().parent;O=U.parent;ROOT=O.parents[2]
T=O/'word100_track_revalidation_20261018';P=O/'word100_visible_arm_duplicate_20261018';V=O/'word100_assignment_provenance_20261018';A=O/'word100_label_only_audit_20261018'
spec=importlib.util.spec_from_file_location('_frozen_recheck_driver',T/'run.py');core=importlib.util.module_from_spec(spec);spec.loader.exec_module(core)
read,write,sha,load,pinscheck=core.read,core.write,core.sha,core.load,core.pinscheck
def prepare():
    assert read(U/'CONTRACTS.json')['status']=='PASS' and not (U/'CONTEXT.json').exists()
    c=read(T/'CONTEXT.json');pins=dict(c['pins']);pins.update(read(T/'OUTPUT_PINS.json'))
    for p in [T/'candidate.py',T/'run.py',T/'FINAL_AUDIT.json',T/'ANNOTATED_COMPARISON.json',T/'CHANGES.json',T/'SUMMARY.json',T/'CONTEXT.json']+list(U.glob('*.py'))+[U/'PROTOCOL.json',U/'CONTRACTS.json']:pins[str(p)]=sha(p)
    pinscheck(pins);(U/'results').mkdir();(U/'before').mkdir()
    for p in [ROOT/'WordsInSilence_PROJECT_CONTEXT.md',ROOT/'NEW_CHAT_HANDOFF.md',O/'STATE.json']:shutil.copy2(p,U/'before'/p.name)
    write(U/'CONTEXT.json',{'rows':c['rows'],'pins':pins,'protocol':read(U/'PROTOCOL.json'),'comparison':str(T)})
    print('FROZEN',len(pins),'pins',flush=True)
def replay():
    core.U=U;core.main()
    c=read(U/'CONTEXT.json');changes=[];old4=[];counts={'frames':0,'raw_candidates':0,'unchanged':0,'withheld':0,'reassigned':0,'filled':0};differing={}
    for r in c['rows']:
        for v in ('masked','sensor'):
            old=load(T/'results'/f"{r['clip_id']}_{v}.npz");new=load(U/'results'/f"{r['clip_id']}_{v}.npz");z=load(r['views'][v]['raw'])
            counts['frames']+=len(z['raw_count']);counts['raw_candidates']+=int(z['raw_count'].sum())
            for f,i in np.argwhere(old['sides']!=new['sides']):
                assert i<int(z['raw_count'][f]);a=int(old['sides'][f,i]);b=int(new['sides'][f,i]);kind='withheld' if b<0 else 'filled' if a<0 else 'reassigned';counts[kind]+=1
                ch={'clip_id':r['clip_id'],'word':r['label'],'view':v,'index':int(f),'frame_id':int(z['frame_id'][f]),'raw_index':int(i),'old_side':None if a<0 else a,'new_side':None if b<0 else b,'kind':kind,'action':int(new['action'][f,i])};changes.append(ch);differing[(r['clip_id'],v,int(f),int(i))]=ch
    for x in read(T/'CHANGES.json'):
        path=U/'results'/f"{x['clip_id']}_{x['view']}.npz";z=load(path);want=-1 if x['new_side'] is None else x['new_side'];old4.append({**{k:x[k] for k in ('clip_id','view','index','raw_index')},'preserved':int(z['sides'][x['index'],x['raw_index']])==want})
    counts['unchanged']=counts['raw_candidates']-sum(counts[k] for k in ('withheld','reassigned','filled'))
    write(U/'DELTA_PREVIOUS.json',changes);write(U/'PREVIOUS4.json',old4)
    s=read(U/'SUMMARY.json');s['compared_to_previous']=counts;s['previous4_preserved']=sum(x['preserved'] for x in old4);s['new_changed_clips']=len({x['clip_id'] for x in changes});write(U/'SUMMARY.json',s)
    print('DELTA previous candidate:',json.dumps(counts),'clips',s['new_changed_clips'],flush=True)
if __name__=='__main__':prepare() if sys.argv[1]=='prepare' else replay()
