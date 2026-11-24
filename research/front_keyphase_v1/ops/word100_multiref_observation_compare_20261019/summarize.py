from pathlib import Path
from collections import Counter,defaultdict
import json,statistics,hashlib
import numpy as np
U=Path(__file__).resolve().parent
read=lambda p:json.loads(Path(p).read_text('utf-8-sig'))
def write(p,d):Path(p).write_text(json.dumps(d,ensure_ascii=False,indent=2,allow_nan=False)+'\n','utf-8')
def summary(a):
    a=[float(v) for v in a if v is not None]
    return {'n':len(a),'median':float(np.median(a)) if a else None,'p90':float(np.quantile(a,.9)) if a else None,'mean':float(np.mean(a)) if a else None,'max':max(a) if a else None}
def main():
    assert read(U/'INDEPENDENT.json')['status']=='PASS'
    f=read(U/'FROZEN.json');lookup={x['clip_id']:x for x in f['inputs']};viewdata={};rowmaps={};querymeta={};word_meta={};quality={}
    banks={};allrows=[]
    for view in f['views']:
        axis=defaultdict(lambda:Counter());vals=defaultdict(list);byclip=defaultdict(list);byword=defaultdict(Counter);rows={};knownclips=[];streaks=[];reused=0;validtrain=Counter();vbank={}
        for path in sorted((U/'words'/view).glob('*.json')):
            d=read(path);code=d['word_code'];word_meta[code]={'label':d['label'],'role_known':d['role_known'],'sides':d['alignment_sides']}
            vbank[code]={'label':d['label'],'role_known':d['role_known'],'alignment_sides':d['alignment_sides'],'references':d['references'],'baseline':d['baseline'],'combined_semantic_verdict':None}
            reused+=d['reused_metric_pairs'];streaks.extend(p['diagnostics']['max_repeated_alignment_steps'] for p in d['pairs'] if 'max_repeated_alignment_steps' in p['diagnostics'])
            for q in d['queries']:
                if q['role']=='train_candidate':
                    for m in q['metrics']:
                        for s in m['sides']:
                            if s['side'] in d['alignment_sides']:validtrain[m['key']+'_'+s['side']]+=s['value'] is not None
                    continue
                cid=q['clip_id'];info=read(U/'inputs'/view/(cid+'.json'));querymeta[(view,cid)]=info
                if d['role_known']:
                    knownclips.append(all(m['value'] is not None and d['baseline'][m['key']]['n']>=5 and d['baseline'][m['key']]['median']>1e-6 for m in q['metrics']))
                for m in q['metrics']:
                    for s in m['sides']:
                        if s['side'] not in d['alignment_sides']:continue
                        key=m['key']+'_'+s['side'];base=d['baseline'][m['key']]['sides'][s['side']]
                        measurable=s['value'] is not None;scale_ready=base['n']>=5 and base['median'] is not None and base['median']>1e-6
                        ready=measurable and scale_ready;outside=ready and s['value']>base['max']
                        axis[key].update(total=1,measurable=int(measurable),relative_ready=int(ready),outside_train_max=int(outside),unmeasurable=int(not measurable))
                        if measurable:vals[key+'_raw'].append(s['value'])
                        if ready:vals[key+'_ratio'].append(s['value']/base['median'])
                        byword[code].update(total=1,measurable=int(measurable),ready=int(ready),outside=int(outside))
                        quality_group='bend' if m['key']=='handshape' else 'handpos';qm=info['sides'][s['side']][quality_group]
                        row={'view':view,'word_code':code,'label':d['label'],'clip_id':cid,'signer_id':q['signer_id'],'axis':m['key'],'side':s['side'],'raw':s['value'],'refs':s['valid_references'],'measurable':measurable,'baseline_n':base['n'],'baseline_median':base['median'],'baseline_max':base.get('max'),'relative_ready':ready,'outside_train_max':bool(outside),'ratio':s['value']/base['median'] if ready else None,'source_time_fraction':qm['source_time_fraction'],'internal_missing_seconds':qm['internal_missing_seconds'],'role_known':d['role_known']}
                        rows[(cid,m['key'],s['side'])]=row;allrows.append(row)
        vtot={k:sum(c[k] for c in axis.values()) for k in ['total','measurable','relative_ready','outside_train_max','unmeasurable']}
        viewdata[view]={'totals':vtot,'by_axis_side':dict(axis),'by_word':dict(byword),'values':{k:summary(v) for k,v in vals.items()},'known12_complete_DEV':sum(knownclips),'known12_total_DEV':len(knownclips),'max_path_streak':summary(streaks),'reused_metric_pairs':reused,'train_measurable':dict(validtrain)}
        rowmaps[view]=rows;banks[view]=vbank
        for subset in ['train_candidate','development_only']:
            qq=[]
            for r in f['inputs']:
                if r['role']!=subset:continue
                inf=read(U/'inputs'/view/(r['clip_id']+'.json'))
                for side in ('L','R'):
                    qq.append({'side':side,**inf['sides'][side]['bend'],'full_complete_fraction':inf['sides'][side]['full_clip_complete_hand_time_fraction']})
            quality[view+'_'+subset]={side:{k:summary(x[k] for x in qq if x['side']==side) for k in ['source_time_fraction','internal_missing_seconds','full_complete_fraction']} for side in ('L','R')}
        write(U/('BANK_'+view+'.json'),{'schema':'wis_offline_multiref_100_comparison_v1','production_ready':False,'training_only':True,'views_may_not_be_mixed':True,'words':vbank,'frozen_sha256':hashlib.sha256((U/'FROZEN.json').read_bytes()).hexdigest()})
    deltas={};transitions=[]
    for v in ['masked','sensor']:
        ds=defaultdict(Counter);raw=defaultdict(list);t=[]
        for k,b in rowmaps['legacy'].items():
            x=rowmaps[v][k];key=b['axis']+'_'+b['side']
            for label,col in [('measurement','measurable'),('relative','relative_ready')]:
                action='retained' if b[col] and x[col] else 'gained' if x[col] else 'lost' if b[col] else 'neither'
                ds[key][label+'_'+action]+=1
                if label=='measurement' and action in ['gained','lost']:
                    tr={'view':v,'type':action,'legacy':b,'candidate':x};t.append(tr);transitions.append(tr)
            if b['measurable'] and x['measurable']:raw[key].append(x['raw']-b['raw'])
        deltas[v]={'by_axis_side':dict(ds),'common_measured_raw_delta':{k:summary(x) for k,x in raw.items()},'measurement_gains':sum(c['measurement_gained'] for c in ds.values()),'measurement_losses':sum(c['measurement_lost'] for c in ds.values()),'relative_gains':sum(c['relative_gained'] for c in ds.values()),'relative_losses':sum(c['relative_lost'] for c in ds.values())}
    write(U/'DEV_ROWS.json',allrows);write(U/'TRANSITIONS.json',transitions)
    same={}
    for v in f['views']:
        same[v]={str(s):dict(Counter(measurable=sum(int(x['measurable']) for x in rowmaps[v].values() if x['signer_id']==s),total=sum(x['signer_id']==s for x in rowmaps[v].values()))) for s in [11,12,13]}
    out={'status':'COMPLETE_SAVED_COMPARISON','views':viewdata,'deltas':deltas,'by_signer':same,'source_quality':quality,'scope':{'classes':100,'train':899,'dev':300,'unique_videos':1199,'view_pairs':29643,'train_direction_pairs_per_view':7184,'dev_pairs_per_view':2697,'expected_per_side_metric_DEV_units':1773,'metrics_are_not_independent_attempts':True},'limits':['Development used previously, not fresh holdout','Role known only12, semantic combined verdict absent for88','Lower distance can arise from rest removal or reduced evidence; not a recognition gain','Same fixed C/TF; no model inference/training or private capture access','DTW masked/sensor sources form different measurement contexts; no semantic error truth here']}
    write(U/'SUMMARY.json',out)
    print(json.dumps({'views':{k:{'totals':v['totals'],'known12_complete_DEV':v['known12_complete_DEV']} for k,v in viewdata.items()},'deltas':deltas,'by_signer':same},ensure_ascii=False))
if __name__=='__main__':main()
