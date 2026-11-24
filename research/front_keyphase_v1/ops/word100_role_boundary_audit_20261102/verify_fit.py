from pathlib import Path
import json,hashlib,numpy as np
U=Path(__file__).resolve().parent
def read(p):return json.loads(Path(p).read_text(encoding='utf-8-sig'))
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def main():
    assert not (U/'VERIFIED.json').exists();plan=read(U/'FROZEN.json');d=read(U/'FIT_DIAGNOSIS.json');groups={}
    for r in d['rows']:
        assert sha(r['prediction'])==r['sha256'];a=np.load(r['prediction']);t=np.load(r['teacher'])
        for k in ['role_label','source_frame_id','image_available']:assert np.array_equal(a[k],t[k])
        pr=a['content_probability'];act=a['activity_probability'];role=a['role_label'];av=a['image_available'];rest=a['current_probability'][:,9];known=role>=0;major=role==1
        conflict=(rest>=.8)&(act[:,1]>=.8)&av;eligible=~(((1-rest)*act[:,0]+rest>=.8)&av&~conflict)
        assert np.array_equal(eligible,a['eligible']) and np.isfinite(pr).all() and np.allclose(act.sum(1),1,atol=1e-6)
        correct=(pr.argmax(1)==a['action_label'])&(a['action_label']>=0)&(a['action_label']<8)
        z=dict(role_known=int(known.sum()),role_correct=int(((act.argmax(1)==major)&known).sum()),major_known=int(major.sum()),major_excluded=int((major&~eligible).sum()),
            correct_main_excluded=int((correct&major&~eligible).sum()),old_main_class_errors=int(((t['probability'].argmax(1)!=1)&major).sum()))
        assert all(r[k]==v for k,v in z.items());assert [p['frame'] for p in r['major_exclusion_points']]==np.flatnonzero(major&~eligible).tolist()
        g=groups.setdefault(r['kind'],{k:0 for k in z})
        for k,v in z.items():g[k]+=v
    assert groups==d['groups'] and len(d['rows'])==72
    for p,h in plan['pins'].items():assert sha(p)==h
    out=dict(status='FIT_DIAGNOSTIC_VERIFIED',groups=groups,model_inference=0,whole_goal_complete=False,scope='Saved-array arithmetic only; not new physical or linguistic ground truth.')
    with (U/'VERIFIED.json').open('x',encoding='utf8') as f:json.dump(out,f,ensure_ascii=False,indent=2)
    print(json.dumps(out,ensure_ascii=False))
if __name__=='__main__':main()
