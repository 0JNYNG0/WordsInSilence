import json,hashlib,time,sys
from pathlib import Path
import numpy as np
from guarded_candidate import guard,PARAMS
U=Path(__file__).resolve().parent;G=U/'guarded';G.mkdir(exist_ok=True);F=U.parent/'word100_forearm_contour_20261018'
def read(p):return json.loads(Path(p).read_text('utf8'))
def write(p,v):Path(p).write_text(json.dumps(v,ensure_ascii=False,indent=2)+'\n',encoding='utf8')
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def contracts():
    q=np.zeros((33,2));q[15]=[30,40];q[16]=[70,40];m=np.ones(33,bool);tests=[]
    def case(name,expect,xy=q,usable=m,w=100,h=100,before=[0,1],after=[-1,1]):
        a=guard(before,after,xy,usable,w,h);assert a['sides']==expect,name;tests.append(name)
    case('two_visible_pose_wrists_protect',[0,1]);case('no_candidate_change',[0,1],after=[0,1])
    m1=m.copy();m1[15]=False;case('one_usable_does_not_confirm_two',[-1,1],usable=m1)
    case('no_usable_does_not_confirm_two',[-1,1],usable=np.zeros(33,bool))
    q1=q.copy();q1[15,1]=120;case('outside_is_not_observed',[-1,1],xy=q1)
    q2=q.copy();q2[15,0]=np.nan;case('nonfinite_is_not_observed',[-1,1],xy=q2)
    case('missing_pose_preserves',[0,1],xy=None);case('missing_mask_preserves',[0,1],usable=None)
    case('bad_size_preserves',[0,1],w=0)
    case('reverse_raw_order',[1,0],before=[1,0],after=[1,-1])
    case('coordinate_scale',[0,1],xy=q*2,w=200,h=200)
    mir=q.copy();mir[:,0]=100-mir[:,0];case('reflection',[0,1],xy=mir)
    write(G/'CONTRACTS.json',{'status':'PASS','count':len(tests),'tests':tests})
def freeze():
    assert not (G/'CONTEXT.json').exists();contracts();ctx=read(U/'CONTEXT.json');pins=dict(ctx['pins']);pins.update(read(U/'OUTPUT_PINS.json'))
    for p in [U/'guarded_candidate.py',U/'guarded_run.py',G/'CONTRACTS.json',U/'CHANGES.json',U/'RESULTS.json',U/'SUMMARY.json',U/'HISTORY_DIAGNOSTIC.json',U/'FIRST_VERIFY_ERROR.json']:pins[str(p)]=sha(p)
    for p,h in pins.items():assert sha(p)==h,p
    write(G/'CONTEXT.json',{'parameters':PARAMS,'pins':pins,
     'revision_reason':'V1 also suppresses a candidate in a real two-hand overlap scene (WORD0920_REAL05masked56). Current Pose wrists have high confidence, but V1 ignored this protective body evidence. Preserve first V1 results. V2 adds only a general two-visible-body-wrist veto; no geometry coefficients adjusted.',
     'authorization':'Same user authorized common duplicate/wrist improvement with normal/crossing preservation; local validation only, no learning/C/user data changes.',
     'execution':'Reuse frozen full899V1 geometrical results. V2 only vetoes suppression, cannot create an additional change. Evaluate every V1 changed frame from its own actual detector input and verify every final side array against baseline in separate code.',
     'acceptance':'Known normal441 and prior13F corrections unchanged. The315 geometric-crossing list contains1previously annotated error (airport) and1uncertain frame; it was never315independently verified normals. Preserve all314non-known-error crossing flags; any changed known-error must improve/withhold with source evidence. Review every new changed frame and every veto. No result-specific word/frame logic.',
     'limits':'V2 informed by publicTRAIN development failure, not independent heldout performance. Pose is supporting detector evidence, not ground truth. Suppression only removes provisional anatomical hint, not hand/points. No wrist coordinates repaired.'})
    print('GUARDED FROZEN',len(pins),flush=True)
def run():
    c=read(G/'CONTEXT.json');assert c['parameters']==PARAMS and not (G/'RESULT.json').exists()
    for p,h in c['pins'].items():assert sha(p)==h,p
    rows={r['clip_id']:r for r in read(U/'CONTEXT.json')['rows']};events=[];patches=[]
    for x in read(U/'CHANGES.json'):
        r=rows[x['clip_id']];t=x['index']
        with np.load(r['views'][x['view']]['raw']) as n:
            a=guard(x['before'],x['after'],n['pose_xy_px'][t],n['pose_usable'][t],int(n['width'][t]),int(n['height'][t]))
        e={**x,'v1_after':x['after'],'after':a['sides'],'guard':a};events.append(e)
        if e['before']!=e['after']:patches.append(e)
    # Research output is the preserved baseline manifest + explicit sparse patches.
    # No production or learned input files are written.
    write(G/'EVENTS.json',events);write(G/'SIDE_PATCHES.json',patches)
    write(G/'BASELINE_MANIFEST.json',{'U_source':str(U.parent/'word100_uncertain_hand_preservation_20261018/results'),'F_selected_overrides':str(F/'RESULTS.json'),'V1_baseline_arrays':str(U/'results'),'applicable_key':['clip_id','view','index'],'raw_points_and_masks':'unchanged read-only originals'})
    by={(x['clip_id'],x['view'],x['index']):x for x in patches}
    ann=read(U/'ANNOTATED.json')
    for a in ann:
        k=(a['clip_id'],a['view'],a['index']);a['after']=by[k]['after'][a['raw_index']] if k in by else a['before']
    cross=read(U/'CROSSING.json');knownerr={(a['clip_id'],a['view'],a['index']) for a in ann if a['verdict']=='error_identity'}
    for a in cross:
        k=(a['clip_id'],a['view'],a['index']);a['after']=by[k]['after'] if k in by else a['before'];a['previous_known_error']=k in knownerr
    write(G/'ANNOTATED.json',ann);write(G/'CROSSING.json',cross)
    rem=[a for a in ann if a['remaining9']];normal=[a for a in ann if a['verdict']=='normal_identity'];other=[a for a in cross if not a['previous_known_error']]
    res={'status':'EXECUTED_PENDING_VERIFICATION','V1_proposals':len(events),'vetoed_frames':sum(e['guard']['veto'] for e in events),'changed_frames':len(patches),'changed_hands':sum(sum(a!=b for a,b in zip(e['before'],e['after'])) for e in patches),'normal441_preserved':sum(a['before']==a['after']==a['actual_side'] for a in normal),'geometric_crossing_flags':315,'geometric_flags_unchanged':sum(a['before']==a['after'] for a in cross),'non_known_error_flags':len(other),'non_known_error_flags_unchanged':sum(a['before']==a['after'] for a in other),'remaining9_correct':sum(a['after']==a['actual_side'] for a in rem),'remaining9_withheld':sum(a['after']<0 for a in rem),'remaining9_wrong_hint':sum(a['after']>=0 and a['after']!=a['actual_side'] for a in rem),'training':0,'production_changed':False}
    write(G/'RESULT.json',res)
    for p,h in c['pins'].items():assert sha(p)==h,p
    print(json.dumps(res),flush=True)
if __name__=='__main__':freeze() if sys.argv[1]=='freeze' else run()
