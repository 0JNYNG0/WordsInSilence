"""Bind existing public cached axes to structural RGB support; no model/media calls."""
from pathlib import Path
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
import numpy as np
from process_rgb_support import build_rgb_support, span_support

U=Path(__file__).resolve().parent
O=U.parent
AB=O/'word100_action_completion_supervision_20261102'
read=lambda p:json.loads(Path(p).read_text(encoding='utf-8-sig'))
def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda:f.read(1024*1024),b''):h.update(b)
    return h.hexdigest()


def main():
    dest=U/'RGB_SUPPORT_MANIFEST.json'
    assert not dest.exists(), 'Preserve completed support manifest'
    pins={};checks=Counter()
    def pin(p,expected=None):
        path=str(Path(p).resolve());h=sha(path)
        if expected is not None:assert h==expected,path
        if path in pins:assert pins[path]==h,path
        pins[path]=h;checks['hash']+=1;return h
    for p in [Path(__file__),U/'process_rgb_support.py',U/'SCOPE.json',
        AB/'process_rgb_features.py',AB/'process_head.py',
        O/'word100_learned_guide_feedback_20261102/live_adapter.py',
        O/'word100_seated_web_framing_20261102/adapter.py',
        O/'word100_signrep_feasibility_20261029/common.py',
        O/'word100_learned_body_relation_20261102/demo_adapter.py',
        AB/'PROCESS_INPUT_MANIFEST.json',AB/'DEV_PROCESS_INPUT_MANIFEST.json',
        AB/'PROCESS_AUTO_NATIVE_MANIFEST.json',AB/'PROCESS_AUTO_EVALUATION.json',
        AB/'PROCESS_EVALUATION_SAVED_VERIFIED.json',AB/'PROCESS_AUTO_EVALUATION_VERIFIED.json']:
        pin(p)
    verified=read(AB/'PROCESS_AUTO_EVALUATION_VERIFIED.json')
    assert verified['result_sha256']==pins[str((AB/'PROCESS_AUTO_EVALUATION.json').resolve())]
    auto=read(AB/'PROCESS_AUTO_NATIVE_MANIFEST.json')
    assert verified['native_manifest_sha256']==pins[str((AB/'PROCESS_AUTO_NATIVE_MANIFEST.json').resolve())]
    saved=read(AB/'PROCESS_AUTO_EVALUATION.json')
    edited=read(AB/'PROCESS_INPUT_MANIFEST.json')['rows']+read(AB/'DEV_PROCESS_INPUT_MANIFEST.json')['rows']
    assert len(edited)==296 and len(auto['unique_inputs'])==48
    inputs={};alias={};full_by_clip={}
    def add(meta,full=False):
        role=meta['role'];person=meta['person']
        assert person in (range(1,10) if role=='fit' else (11,12,13))
        key=pin(meta['input'],meta['input_sha256'])
        with np.load(meta['input'],allow_pickle=False) as z:
            ids=z['frame_id'].copy();ts=z['timestamp_us'].copy();av=z['image_available'].copy()
            rgb=z['rgb'];finite=np.isfinite(rgb).all(1)
            assert rgb.shape==(len(ids),768) and finite.all()
        assert len(ids)==meta['frames'] and len(av)==len(ids)
        if full:
            assert ids.tolist()==meta['frame_ids'] and ts.tolist()==meta['timestamp_us']
            sf=ids.tolist();st=ts.tolist();rgb_meta=None
            assert av.sum()==meta['image_available_count'] and len(ids)>=16
        else:
            assert ids.tolist()==list(range(len(ids))) and ts.tolist()==meta['view_timestamps_us']
            assert av.tolist()==meta['image_available']
            sf=meta['source_frame_indices'];st=meta['source_timestamps_us']
            context_path=meta['rgb_context'];pin(context_path,meta['output_pins'][context_path])
            rgb_meta=read(context_path)
            for p,h in rgb_meta['code_pins'].items():pin(p,h)
            assert rgb_meta['real_frame_id']==ids.tolist() and rgb_meta['real_timestamp_us']==ts.tolist()
            assert rgb_meta['source_sha256']==meta['video_sha256'] and rgb_meta['supported']
        s=build_rgb_support(ts.tolist(),av.tolist(),ids.tolist(),sf,st,finite.tolist())
        if rgb_meta:
            assert [w['slot_rows'] for w in s['windows']]==rgb_meta['model_window_indices']
            assert [w['real_slot_mask'] for w in s['windows']]==rgb_meta['source_mask']
            assert s['padding_slots']==rgb_meta['padding_count']
            if len(ids)>=16:assert [x/2 for x in s['center_twice_us']]==rgb_meta['window_centers_us']
            checks['stored_rgb_layout_match']+=1
        assert key not in inputs
        inputs[key]=dict(input=str(Path(meta['input']).resolve()),input_sha256=key,
            clip_id=meta['clip_id'],person=person,role=role,
            source_sha256=meta['source_sha256'],source=meta['source'],
            actual_supplied_video=meta.get('video',meta['source']),
            actual_supplied_video_sha256=meta.get('video_sha256',meta['source_sha256']),
            route='original_full_native_cache' if full else 'actual_materialized_view_cache',
            rgb_context=meta.get('rgb_context'),support=s)
        checks['input_axis_and_finite']+=1
        if full:full_by_clip[meta['clip_id']]=key
        else:
            for a in meta['aliases']:
                assert a not in alias;alias[a]=key
        return key
    for m in edited:add(m)
    for m in auto['unique_inputs']:add(m,True)
    contexts=[];candidates=[]
    native_views={}
    def context(kind,identity,key,indices):
        s=inputs[key]['support']
        obj=dict(context_id=kind+':'+identity['query_id'] if kind!='materialized_view' else kind+':'+identity['view_id'],
            route=kind,input_sha256=key,clip_id=identity['clip_id'],query_id=identity['query_id'],
            role=identity['role'],person=identity['person'],physical_query=identity['physical_query'],
            kind=identity.get('kind','native'),span=span_support(s,indices))
        if 'view_id' in identity:obj['view_id']=identity['view_id']
        contexts.append(obj)
        return obj
    def candidates_for(row,ctx,source_ids):
        full_indices=ctx['span']['nominal_cache_rows'];key=ctx['input_sha256']
        for ci,c in enumerate(row['qualification']['interval_candidates']):
            local=c['output_indices']
            assert local==list(range(c['start'],c['end']+1))
            ii=[full_indices[i] for i in local]
            obj=dict(candidate_id=ctx['context_id']+':candidate:'+str(ci),context_id=ctx['context_id'],
                input_sha256=key,query_id=row['query_id'],role=row['role'],person=row['person'],
                physical_query=row['physical_query'],candidate_index=ci,prior_nominal_local_indices=local,
                span=span_support(inputs[key]['support'],ii),
                selection='Existing frozen candidate coordinates only; no new proposal or score selection')
            assert obj['span']['nominal_source_frames']==[source_ids[i] for i in local]
            candidates.append(obj);checks['candidate_axis']+=1
    for r in saved['view_rows']:
        key=alias[r['view_id']];item=inputs[key]
        assert item['clip_id']==r['clip_id'] and item['person']==r['person'] and item['role']==r['role']
        ctx=context('materialized_view',r,key,list(range(item['support']['rows'])))
        candidates_for(r,ctx,item['support']['source_frame_ids'])
        if r['kind']=='native':native_views[r['query_id']]=(r,ctx)
    native_meta={m['query_id']:m for m in auto['rows']}
    for r in saved['automatic_rows']:
        m=native_meta[r['query_id']];key=full_by_clip[r['clip_id']]
        assert key==m['input_sha256'] and r['proposal']==m['proposal']
        ii=m['proposal']['output_indices'];s=inputs[key]['support']
        assert [s['frame_ids'][i] for i in ii]==r['source_frame_ids']
        assert [s['timestamp_us'][i] for i in ii]==r['source_timestamps_us']
        ctx=context('automatic_native',r,key,ii);candidates_for(r,ctx,r['source_frame_ids'])
        # The same reviewed native extent on the full-cache input, for a future
        # aligned training route. This creates no prediction or target.
        nr,nctx=native_views[r['query_id']]
        mapper={f:i for i,f in enumerate(s['source_frame_ids'])}
        bounded=[mapper[f] for f in nctx['span']['nominal_source_frames']]
        assert [s['source_timestamp_us'][i] for i in bounded]==nctx['span']['nominal_source_timestamps_us']
        context('reviewed_native_bound_on_full_cache',nr,key,bounded)
        checks['native_pair_axis']+=1
    assert len(inputs)==344 and len(alias)==297 and len(contexts)==417
    summary=dict(unique_inputs=len(inputs),materialized_inputs=296,full_native_inputs=48,
        context_routes=dict(Counter(c['route'] for c in contexts)),
        candidate_count=len(candidates),candidate_roles=dict(Counter(c['role'] for c in candidates)),
        short_inputs=sum(v['support']['rows']<16 for v in inputs.values()),
        inputs_with_blackout=sum(not all(v['support']['image_available']) for v in inputs.values()),
        real_cache_rows=sum(v['support']['rows'] for v in inputs.values()),
        candidates_with_outside_nominal_context=sum(bool(c['span']['outside_nominal_consumed_rows']) for c in candidates),
        candidate_context_contains_synthetic_pixels=sum(bool(c['span']['synthetic_context_cache_rows']) for c in candidates))
    for p,h in pins.items():assert sha(p)==h,p
    obj=dict(status='STRUCTURAL_SUPPORT_BOUND_NO_NEW_MODEL_EXECUTION',at=datetime.now(timezone.utc).isoformat(),
        summary=summary,inputs=inputs,contexts=contexts,candidates=candidates,pins=pins,checks=dict(checks),
        definitions=dict(nominal='Physical target extent to be directly reviewed; not expanded because RGB sees context.',
            encoded='Union of actual supplied frame slots in encoder windows with positive np.interp coefficients.',
            consumed='Union only for rows whose RGB is not removed by image-availability/finiteness preprocessing.',
            observed='Consumed-context rows whose original image is actually available; blackout coordinates remain synthetic, never observed original frames.',
            padding='Duplicate final slot is internal encoder context, not an extra source frame or repeated physical action.',
            uncertainty='Structural dependency support is not measured neural influence, direct observation truth or exact event timing.'),
        design=dict(outside_context_allowed=True,nominal_positive_copied_from_context=False,
            same_context_incomplete_nominal_negative_controls_required=True,
            no_source_seam_oracle_as_model_input=True,local_exact_truth_assigned=False),
        new_model_inference=0,new_training=0,new_extraction=0,new_scores=0,media_decoded=0,old_files_modified=False,
        whole_goal_complete=False)
    dest.write_text(json.dumps(obj,ensure_ascii=False,separators=(',',':'))+'\n',encoding='utf8')
    print(json.dumps(dict(summary=summary,checks=dict(checks),path=str(dest),bytes=dest.stat().st_size,sha256=sha(dest)),ensure_ascii=False))

if __name__=='__main__':main()
