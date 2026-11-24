"""Frozen RGB/pose transition guard evaluation. No training or semantic grading."""
from pathlib import Path
import json,hashlib,sys,copy
from collections import Counter
U=Path(__file__).resolve().parent;P=U.parent/'word100_temporal_shape_context_20261027'
sys.path.insert(0,str(P))
from compare import coherent
F=('index_only','open_four')
def read(p):return json.loads(Path(p).read_text(encoding='utf-8-sig'))
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def dump(p,x):Path(p).write_text(json.dumps(x,ensure_ascii=False,indent=2),encoding='utf-8')

def decide_edge(edge,cal):
    if not edge['rgb']['valid'] or not edge['pose']['valid']:
        return 'insufficient_evidence'
    changed=[]
    for key in ('rgb','pose'):
        if edge[key]['score']>cal[key+'_threshold']:changed.append(key)
    return ('change_or_viewpoint:'+','.join(changed)) if changed else 'no_change_signal'

def apply_guard(base,states,target,edges,cal):
    result=copy.deepcopy(base)
    result['checked_edges']=[]
    if base['kind']!='temporal_estimate':return result
    a=max(j for j in base['support_indices'] if j<target)
    b=min(j for j in base['support_indices'] if j>target)
    statuses=[{'edge_index':j,'status':decide_edge(edges[j],cal)} for j in range(a,b)]
    result['checked_edges']=statuses
    if any(x['status']!='no_change_signal' for x in statuses):
        has_change=any(x['status'].startswith('change_or_viewpoint') for x in statuses)
        result.update(value=None,kind='unknown',reason='change_or_viewpoint' if has_change else 'insufficient_transition_evidence')
    else:result['reason']='bidirectional_agreement_with_no_observed_change_signal'
    return result

def summarize(rows):
    out={}
    for mode in ('baseline','guard'):
        known=[r for r in rows if r[mode]['value'] is not None]
        matches=sum(r[mode]['value']==r['hidden_draft'] for r in known)
        out[mode]={'total':len(rows),'estimated':len(known),'draft_match':matches,'draft_conflict':len(known)-matches,'withheld':len(rows)-len(known)}
    out['guard_effect']={
      'withheld_previously_matching':sum(r['baseline']['value']==r['hidden_draft'] and r['guard']['value'] is None for r in rows),
      'withheld_previously_conflicting':sum(r['baseline']['value'] is not None and r['baseline']['value']!=r['hidden_draft'] and r['guard']['value'] is None for r in rows),
      'new_wrong_state':sum(r['guard']['value'] is not None and r['guard']['value']!=r['baseline']['value'] for r in rows)}
    return out

def run():
    dest=U/'COMPARISON.json';assert not dest.exists(),'Do not overwrite saved outcomes'
    frozen=read(U/'FROZEN.json');cal=read(U/'metrics/CALIBRATION.json')
    assert cal.get('rgb_threshold') is not None and cal.get('pose_threshold') is not None,'Calibration unavailable; no threshold search'
    oldcal=read(P/'COMPARISON.json')
    assert all(sha(p)==h for p,h in oldcal['pins'].items())
    newidx=read(U/'visual_inputs/INDEX.json');oldidx=read(P/'visual_inputs/INDEX.json')
    labels=read(U/'review/IMAGE_LABELS.json');oldlabels=read(P/'review/IMAGE_LABELS.json')
    lut={(r['case_id'],r['image_key']):r for r in labels['rows']+oldlabels['rows']}
    assert len(labels['rows'])==234 and len(lut)==468
    raw=read(U/'metrics/RAW_EDGES.json');edges={}
    for r in raw['rows']:
        key=(r['case_id'],r['edge_index']);assert key not in edges;edges[key]=r
    assert len(edges)==36*12
    seqs=[];masks=[];real=[];events=[];preserve=Counter()
    for case in newidx['rows']+oldidx['rows']:
        cid=case['case_id'];fresh=cid.startswith('N')
        frames=sorted(case['frames'],key=lambda x:x['actual_pts_us'])
        t=[x['actual_pts_us'] for x in frames]
        states={f:[lut[(cid,x.get('image_key',x['neutral_image_id']))][f] for x in frames] for f in F}
        ed=[edges[(cid,i)] for i in range(12)]
        for i,e in enumerate(ed):assert e['from_frame_id']==frames[i]['frame_id'] and e['to_frame_id']==frames[i+1]['frame_id']
        seq={'case_id':cid,'fresh_dense':fresh,'label':case['label'],'side':case['side'],
             'person_id':case['recorded_person_id'],'frames':[],
             'edges':[{'edge_index':i,'from_pts_us':t[i],'to_pts_us':t[i+1],'status':decide_edge(e,cal),'rgb':e['rgb'],'pose':e['pose']} for i,e in enumerate(ed)]}
        for i,fr in enumerate(frames):
            base=coherent(states,t,i,'C')[1]
            guard={f:apply_guard(base[f],states[f],i,ed,cal) for f in F}
            seq['frames'].append({'frame_id':fr['frame_id'],'pts_us':t[i],'image_key':fr.get('image_key',fr['neutral_image_id']),
               'direct':{f:states[f][i] for f in F},'baseline':base,'guard':guard})
            for f in F:
                if states[f][i] is not None:
                    assert guard[f]['value']==base[f]['value']==states[f][i];preserve['new' if fresh else 'old']+=1
                else:
                    real.append({'case_id':cid,'fresh_dense':fresh,'person_id':case['recorded_person_id'],'feature':f,'index':i,'baseline':base[f],'guard':guard[f]})
                if 0<i<len(frames)-1 and states[f][i] is not None:
                    masked=copy.deepcopy(states);masked[f][i]=None
                    b=coherent(masked,t,i,'C')[1][f]
                    g=apply_guard(b,masked[f],i,ed,cal)
                    assert g['value'] is None or g['value']==b['value']
                    masks.append({'case_id':cid,'fresh_dense':fresh,'person_id':case['recorded_person_id'],
                       'feature':f,'index':i,'frame_id':fr['frame_id'],'hidden_draft':states[f][i],'baseline':b,'guard':g})
        # Opposite observed endpoints bound a change. Do not invent the exact midpoint shape.
        for f in F:
            known=[i for i,v in enumerate(states[f]) if v is not None]
            for a,b in zip(known,known[1:]):
                if b==a+1 and states[f][a]==states[f][b]:continue
                events.append({'case_id':cid,'fresh_dense':fresh,'feature':f,'from_index':a,'to_index':b,
                  'from_pts_us':t[a],'to_pts_us':t[b],'from_value':states[f][a],'to_value':states[f][b],
                  'kind':'observed_state_change_interval' if states[f][a]!=states[f][b] else 'same_endpoints_unresolved_middle',
                  'gap_frames':b-a-1,'semantic_importance':'not_determined','intermediate_exact_shape':None})
        seqs.append(seq)
    freshm=[r for r in masks if r['fresh_dense']]
    real_summary={}
    for name,rs in [('new',[r for r in real if r['fresh_dense']]),('diagnostic_old',[r for r in real if not r['fresh_dense']])]:
        real_summary[name]={'original_unknown':len(rs),**{m:sum(r[m]['value'] is not None for r in rs) for m in ('baseline','guard')}}
    pinpaths=[U/'FROZEN.json',U/'visual_inputs/INDEX.json',U/'review/IMAGE_LABELS.json',U/'metrics/RAW_EDGES.json',U/'metrics/CALIBRATION.json',Path(__file__),P/'compare.py',P/'temporal_support.py',P/'COMPARISON.json']
    d={'status':'EVALUATED_NO_LEARNED_MODEL_OR_EDUCATIONAL_CORRECTNESS',
       'pins':{str(p):sha(p) for p in pinpaths},'calibration':cal,
       'fresh_summary':summarize(freshm),
       'fresh_different_recorded_ID_summary':summarize([r for r in freshm if r['person_id']!=1]),
       'fresh_by_person':{str(p):summarize([r for r in freshm if r['person_id']==p]) for p in (1,2,3)},
       'old_diagnostic_summary':summarize([r for r in masks if not r['fresh_dense']]),
       'real_unknown_summary':real_summary,'known_direct_preserved':dict(preserve),
       'fresh_edge_status_counts':dict(Counter(e['status'] for s in seqs if s['fresh_dense'] for e in s['edges'])),
       'sequences':seqs,'masked_trials':masks,'real_unknown_trials':real,'events':events,
       'limitations':['New dense windows of previously available TRAIN videos, not new unobserved recognition people.',
         'Two old stable windows calibrate limits; no broad-normal variation proof.',
         'Optical/2D angle changes can be rotation, occlusion or measurement noise, not only finger articulation.',
         'Same-AI image drafts, no independent human truth.',
         'Holding an uncertain state is not recovering the hidden shape or improving TCN accuracy.'],
       'training':False,'production_change':False}
    dump(dest,d)
    print(json.dumps({k:d[k] for k in ['fresh_summary','fresh_different_recorded_ID_summary','old_diagnostic_summary','real_unknown_summary','known_direct_preserved','fresh_edge_status_counts']},ensure_ascii=False,indent=2))
if __name__=='__main__':run()
