"""Display-only projection of already frozen public results. No model imports."""
from pathlib import Path
from collections import Counter
import hashlib
import json

U = Path(__file__).resolve().parent
O = U.parent
AN = O / 'word100_feedback_discrimination_20261103'
AJ = O / 'word100_local_event_controls_20261103'
LABELS = {'WORD0739':'순종','WORD1133':'잘하다','WORD1567':'조상','WORD2523':'이집트'}
KINDS = {'normal':'공개 원본','order':'순서 변경 시험','cut':'일부 삭제 시험','gap':'가림 시험'}
# Display names for existing shared physical families, not new word-specific rules.
FAMILIES = {
 0:('얼굴 앞의 편 오른손','right'), 1:('가슴 앞의 닫힌 오른손','right'),
 2:('어깨 옆으로 세운 오른팔','right'), 3:('왼팔을 따라 움직이는 오른손','both'),
 4:('왼손목 가까이 온 오른손','both'), 5:('위로 움직이는 오른손','right'),
 6:('양손이 가까이 있는 모습','both'), 7:('양손을 벌린 모습','both')}

def read(p): return json.loads(Path(p).read_text(encoding='utf-8-sig'))
def sha(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def write(p, obj):
    Path(p).write_text(json.dumps(obj,ensure_ascii=False,separators=(',',':'),allow_nan=False),encoding='utf-8')
def compact_range(frames, times):
    ids=sorted(set(int(f) for f in frames))
    if not ids: return None
    assert all(0<=f<len(times) for f in ids)
    return {'frames':ids,'start':times[ids[0]],'end':times[ids[-1]],'focus':times[ids[len(ids)//2]]}
def side_for(text):
    if ('left' in text or '왼손' in text) and ('right' in text or '오른손' in text): return 'both'
    if 'left' in text or '왼손' in text: return 'left'
    return 'right'
def chunks(ids):
    out=[]
    for f in ids:
        if out and f==out[-1][-1]+1: out[-1].append(f)
        else: out.append([f])
    return out

def main():
    assert not (U/'BUILD.json').exists(), 'Preserve finished display snapshot'
    pins={}
    def pin(p,expected=None):
        p=Path(p).resolve(); h=sha(p)
        if expected is not None: assert h==expected, str(p)
        pins[str(p)]=h
        return read(p)
    index=pin(AN/'COMPOSITION_RESULT.json')
    controls=pin(AJ/'INPUT_FROZEN.json')
    reversals=pin(AN/'reverse_inputs.json')
    media_by_id={r['case_id']:r for r in controls['rows']+reversals['rows']}
    results={}
    for r in index['rows']:
        d=pin(r['result']['path'],r['result']['sha256'])
        m=media_by_id[r['case_id']]
        assert m['person']==12 and 'development' in m['role']
        assert d['source_sha256']==m['video']['sha256']
        p=Path(m['video']['path']).resolve()
        assert sha(p)==m['video']['sha256'];pins[str(p)]=m['video']['sha256']
        ident=d['combined_evidence_feedback']['identity']
        ts=[t/1e6 for t in ident['timestamp_us']]
        assert ident['native_frame_ids']==list(range(len(ts)))
        assert all(a<b for a,b in zip(ts,ts[1:])) and len(ts)==d['frames']
        ov=d.get('overlay')
        if ov:
            assert len(ov['times'])==len(ov['points'])==len(ts)
            assert all(abs(a-b)<=1e-6 for a,b in zip(ts,ov['times']))
            assert all(len(p)==75 for p in ov['points'])
        assert d['whole_sign_correctness'] is None
        assert d['combined_evidence_feedback']['correction_enabled'] is False
        results[r['case_id']]={'raw':d,'meta':m,'index':r,'times':ts}

    (U/'data').mkdir(exist_ok=True)
    catalog=[];media={};totals=Counter()
    for cid,v in results.items():
        d,m,t=v['raw'],v['meta'],v['times'];word=d['word_id'];kind=v['index']['kind']
        refid=word+'_reencoded_normal'; ref=results[refid]; rd,rt=ref['raw'],ref['times']
        cards=[]
        def segment(frames,times,label,side):
            span=compact_range(frames,times)
            return dict(span=span,label=label,side=side)
        def add(card):
            card['id']=f'{cid}:{len(cards)}';card['educational_error']=None
            card['correction_certified']=False;cards.append(card);totals[card['state']]+=1
        rpairs={(p['first']['id'],p['second']['id']):p for p in rd['partial_order_feedback']['items']}
        for j,p in enumerate(d['partial_order_feedback']['items']):
            key=(p['first']['id'],p['second']['id']);rp=rpairs.get(key)
            qsegs=[];rsegs=[]
            for keyname in ['first','second']:
                family=p[keyname]['id'];label,side=FAMILIES[family]
                qsegs.append(segment(p[keyname+'_evidence']['input_frames'],t,label,side))
                if rp:rsegs.append(segment(rp[keyname+'_evidence']['input_frames'],rt,label,side))
            qsegs.sort(key=lambda z:z['span']['start'] if z['span'] else float('inf'))
            rsegs.sort(key=lambda z:z['span']['start'] if z['span'] else float('inf'))
            status=p['observation_status']
            state='difference' if status=='OBSERVED_B_BEFORE_A' else 'observed' if status=='OBSERVED_A_BEFORE_B' else 'uncertain'
            title={'difference':'두 동작의 순서가 참고와 달라 보여요',
                   'observed':'이 두 동작의 앞뒤 순서가 확인됐어요','uncertain':'이 두 동작의 순서는 아직 확인하기 어려워요'}[state]
            labels=[FAMILIES[k][0] for k in key]
            hint=('참고처럼 ‘'+labels[0]+' → '+labels[1]+'’ 순서인지 다시 확인해 보세요.' if state=='difference' else
                  '이 두 동작의 순서는 참고와 같게 관측됐어요. 전체 동작의 합격 판정은 아니에요.' if state=='observed' else
                  '앞뒤 근거가 겹치거나 부족해요. 표시한 장면을 다시 보고, 지금은 동작을 고치라고 단정하지 않아요.')
            add(dict(type='order',state=state,title=title,hint=hint,query_segments=qsegs,reference_segments=rsegs,
                     source_pointer=f'/partial_order_feedback/items/{j}',saved_status=status,
                     detail=p['message'],reference_order=labels,
                     query_order=[s['label'] for s in qsegs if s['span']],evidence_exact_boundaries=False))

        b=d['combined_evidence_feedback'];rb=rd['combined_evidence_feedback']
        for j,ev in enumerate(b['motion_events']):
            motion=ev['motion'];scope=motion['fixed_scope'];target=ev['target_index']
            other=next((e for e in rb['motion_events'] if e['target_index']==target),None)
            high=motion.get('high_confirmation') is True
            # A high score with missing frames remains partial, never a pass.
            gap=bool(motion.get('unavailable_native_frames'))
            state='observed' if high and not gap else 'uncertain'
            name=scope['name'];side=side_for(name)
            add(dict(type='movement',state=state,title='움직임을 확인한 구간이에요' if state=='observed' else '이 구간의 움직임은 확인을 보류해요',
                     hint='강조한 손·팔의 이동을 참고와 나란히 보세요. 반복 횟수나 전체 완료를 확인한 것은 아니에요.' if state=='observed' else '관측이 부족하거나 움직임 근거가 충분하지 않아요. 없는 동작이나 오답으로 단정하지 않아요.',
                     query_segments=[segment(scope['frame_ids'],t,name,side)],
                     reference_segments=[segment(other['motion']['fixed_scope']['frame_ids'],rt,name,side)] if other else [],
                     source_pointer=f'/combined_evidence_feedback/motion_events/{j}',saved_status=motion['status'],
                     detail=ev['message_ko'],guide=scope.get('source_phrases',[]),evidence_exact_boundaries=False))

        for j,c in enumerate(b['source_comparisons']):
            rc=next((c0 for c0 in rb['source_comparisons'] if c0['component_id']==c['component_id'] and c0['clause_id']==c['clause_id']),None)
            observations=sorted(set(p['observation'] for p in c['native_points']))
            for observation in observations:
                ps=[p for p in c['native_points'] if p['observation']==observation]
                state='difference' if observation=='MODEL_COUNTER_EVIDENCE' else 'observed' if observation=='MODEL_COMPATIBLE' else 'uncertain'
                side=side_for(c['component_id'])
                title='설명과 다른 모양의 근거가 있어요' if state=='difference' else '설명과 관련된 특징이 관측됐어요' if state=='observed' else '이 부분의 손모양은 확인을 보류해요'
                hint=('‘'+c['source_text']+'’에 해당하는 모습인지 확대해서 확인해 보세요. 이 시간에 그 설명이 적용되는지는 미확정이에요.' if state=='difference' else
                      '이 특징이 보인다는 뜻이에요. 이 표현 전체를 올바르게 했다는 뜻은 아니에요.' if state=='observed' else
                      '보이지 않거나 추정에 의존한 부분이에요. 손모양을 고치라고 확정하지 않아요.')
                add(dict(type='feature',state=state,title=title,hint=hint,
                         query_segments=[segment([p['frame'] for p in ps],t,c['source_text'],side)],
                         reference_segments=[segment([p['frame'] for p in rc['native_points']],rt,c['source_text'],side)] if rc else [],
                         source_pointer=f'/combined_evidence_feedback/source_comparisons/{j}',saved_status=observation,
                         detail='설명과 시점의 연결은 기존 DTW가 제안한 범위입니다. 수어 오답이나 교정 정답으로 검증된 결과가 아닙니다.',
                         guide=[c['source_text']],source_url=c['source_url'],evidence_exact_boundaries=False))
        for frames in chunks(b['global_image_unavailable_frames']):
            ledger=m['source_ledger']; rledger=ref['meta']['source_ledger']
            same_source=m['original_source']['sha256']==ref['meta']['original_source']['sha256']
            ref_lookup={src:i for i,src in enumerate(rledger['source_frame_indices'])}
            original_frames=[ref_lookup[ledger['source_frame_indices'][f]] for f in frames] if same_source else []
            add(dict(type='gap',state='uncertain',title='이 시간은 영상 관측이 비어 있어요',hint='이 장면은 검게 가린 공개 시험 영상이에요. 보이지 않은 손·팔을 그려 채우거나 오답으로 판단하지 않아요.',
                     query_segments=[segment(frames,t,'영상 관측이 없는 구간','both')],reference_segments=[segment(original_frames,rt,'가리기 전의 같은 원본 프레임','both')] if original_frames else [],
                     display_mapping='saved_edit_ledger_same_original_frames_diagnostic_only',
                     source_pointer='/combined_evidence_feedback/global_image_unavailable_frames',saved_status='CURRENT_IMAGE_UNAVAILABLE',
                     detail='인위적으로 가린 시험 조건입니다. 실제 사용자에게 발생한 오류가 아닙니다.',evidence_exact_boundaries=False))
        # Keep all cards; display order never selects/changes a score or prediction.
        cards.sort(key=lambda c:({'difference':0,'uncertain':1,'observed':2}[c['state']],0 if c['type']=='gap' else 1))
        def video(c,entry):
            data=entry['raw'];meta=entry['meta'];times=entry['times']
            media[c]=meta['video']
            return dict(id=c,url='/media/'+c,times=times,overlay=data.get('overlay'),
                        unavailable=data['combined_evidence_feedback']['global_image_unavailable_frames'],
                        duration=times[-1]+(times[-1]-times[-2]),
                        overlay_notice=data['saved_order_case_provenance'].get('overlay_notice'))
        payload=dict(id=cid,word_id=word,label=LABELS[word],kind=kind,kind_label=KINDS[kind],
                     reference=video(refid,ref),query=video(cid,v),cards=cards,
                     whole_sign_correctness=None,correction_certified=False,
                     scope='Saved public registered ID12 original/edited controls. Not private user footage; no new inference, training, scoring or DTW.',
                     comparison='Display the saved evidence spans side by side; parallel playback is not a new pairwise DTW alignment.',
                     saved_result_sha256=sha(v['index']['result']['path']))
        write(U/'data'/f'{cid}.json',payload)
        catalog.append(dict(id=cid,word_id=word,label=LABELS[word],kind=kind,kind_label=KINDS[kind],states=dict(Counter(c['state'] for c in cards)),cards=len(cards)))
    write(U/'data'/'catalog.json',dict(cases=catalog,default='WORD0739_reverse_main',words=LABELS,private_data=False,read_only=True))
    write(U/'MEDIA.json',media)
    assert len(catalog)==16 and len(media)==16
    assert all(sha(p)==h for p,h in pins.items())
    write(U/'BUILD.json',dict(status='SAVED_PUBLIC_DISPLAY_READY',cases=len(catalog),cards_by_state=dict(totals),
          input_pins=pins,output_pins={str(p):sha(p) for p in sorted((U/'data').glob('*.json'))},
          media_manifest_sha256=sha(U/'MEDIA.json'),new_inference=0,new_training=0,new_DTW=0,new_scores=0,
          certified_correction_cards=0,private_access=0,goal_remains_paused=True))
    print(json.dumps(dict(cases=len(catalog),cards=dict(totals),new_inference=0,new_training=0),ensure_ascii=False))

if __name__=='__main__': main()
