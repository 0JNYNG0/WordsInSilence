"""Fit-only numeric support audit. No imports from inference/training modules."""
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path
import numpy as np

UNIT=Path(__file__).resolve().parents[1]
OLD=UNIT.parent/'word100_core_stage_evidence_20261031'
EXPECTED=['P033','P034','P035','P045','P046','P047','P063','P064','P065','P069','P070','P071']
def read(p):return json.loads(Path(p).read_text(encoding='utf-8-sig'))
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
PINS={}
def pin(p):PINS[str(p)]=sha(p)
for p in [UNIT/'AUTHORIZATION.json',OLD/'PHASE_FIT.json',OLD/'PHASE_DATA.json',OLD/'SOURCE_PHASE_CONTRACTS.json',Path(__file__)]:pin(p)
auth=read(UNIT/'AUTHORIZATION.json')
rows=[r for r in auth['rows'] if r['role']=='fit']
assert [r['alias'] for r in rows]==EXPECTED
assert Counter(r['person'] for r in rows)=={1:4,2:4,3:4}
phase={c['alias']:c for c in read(OLD/'PHASE_DATA.json')['clips'] if c['alias'] in EXPECTED}
draft={c['alias']:c for c in read(OLD/'PHASE_FIT.json')['clips']}
contracts=read(OLD/'SOURCE_PHASE_CONTRACTS.json')
finger=[(1,2,3),(2,3,4),(0,5,6),(5,6,7),(6,7,8),(0,9,10),(9,10,11),(10,11,12),(0,13,14),(13,14,15),(14,15,16),(0,17,18),(17,18,19),(18,19,20)]

def independent(xy,valid,ts,fids):
    """Independent straight NumPy reconstruction of 69 saved feature definitions."""
    n=len(xy); out=[]; masks=[]
    scale=np.sqrt(np.square(xy[:,11]-xy[:,12]).sum(1))
    center=(xy[:,11]+xy[:,12])*.5
    body=valid[:,11]&valid[:,12]&np.isfinite(scale)&(scale>1e-8)
    norm=np.full_like(xy,np.nan,dtype=np.float64)
    norm[body]=(xy[body]-center[body,None])/scale[body,None,None]
    nv=valid&body[:,None]
    def add(v,m):
        m=m&np.isfinite(v);out.append(np.where(m,v,np.nan));masks.append(m)
    def angle(ids):
        a,b,c=ids;v=xy[:,a]-xy[:,b];w=xy[:,c]-xy[:,b]
        den=np.sqrt(np.square(v).sum(1))*np.sqrt(np.square(w).sum(1))
        ok=valid[:,list(ids)].all(1)&np.isfinite(den)&(den>1e-12)
        q=np.zeros(n);np.divide((v*w).sum(1),den,out=q,where=ok)
        add(np.arccos(np.clip(q,-1,1))/np.pi,ok)
    for off,sh,el,pw in [(33,11,13,15),(54,12,14,16)]:
        for j in [off,pw,el,off+8]:
            for d in range(2):add(norm[:,j,d],nv[:,j])
        ids=[off+j for j in [0,5,9,13,17]];p=norm[:,ids].mean(1)
        for d in range(2):add(p[:,d],nv[:,ids].all(1))
        angle((sh,el,pw))
        for tri in finger:angle(tuple(off+j for j in tri))
        axis=xy[:,off+9]-xy[:,off];mag=np.sqrt(np.square(axis).sum(1))
        ok=valid[:,[off,off+9]].all(1)&np.isfinite(mag)&(mag>1e-8)
        for d in range(2):
            q=np.full(n,np.nan);np.divide(axis[:,d],mag,out=q,where=ok);add(q,ok)
        add(np.sqrt(np.square(norm[:,off+5]-norm[:,off+17]).sum(1)),nv[:,[off+5,off+17]].all(1))
    delta=norm[:,54]-norm[:,33];ok=nv[:,[33,54]].all(1)
    for d in range(2):add(delta[:,d],ok)
    add(np.sqrt(np.square(delta).sum(1)),ok)
    for off,pw in [(33,15),(54,16)]:add(np.sqrt(np.square(norm[:,off]-norm[:,pw]).sum(1)),nv[:,[off,pw]].all(1))
    dt=np.diff(ts)/1e6;con=(np.diff(fids)==1)&(dt>0)&(dt<=.15)
    for off in [33,54]:
        ok=np.r_[False,nv[1:,off]&nv[:-1,off]&con]
        v=np.full((n,2),np.nan);v[1:]=(norm[1:,off]-norm[:-1,off])/dt[:,None]
        for d in range(2):add(v[:,d],ok)
        speed=np.sqrt(np.square(v).sum(1));mov=ok&np.isfinite(speed)&(speed>1e-8)
        for d in range(2):
            q=np.full(n,np.nan);np.divide(v[:,d],speed,out=q,where=mov);add(q,mov)
    return np.stack(out,1),np.stack(masks,1),norm,nv

GROUPS={
 'L_shape_14':[ *range(11,25)],'R_shape_14':[ *range(39,53)],
 'L_hand_wrist_position':[0,1],'R_hand_wrist_position':[28,29],
 'L_pose_wrist_position':[2,3],'R_pose_wrist_position':[30,31],
 'L_elbow_bend':[10],'R_elbow_bend':[38],
 'L_palm_axis_2d':[25,26],'R_palm_axis_2d':[53,54],
 'two_hand_wrist_relation':[56,57,58],
 'L_hand_pose_disagreement':[59],'R_hand_pose_disagreement':[60]}

def group_masks(x,m):
    out={k:m[:,v].all(1) for k,v in GROUPS.items()}
    for side,b in [('L',0),('R',28)]:
        v=x[:,[b+2,b+3]]-x[:,[b+4,b+5]]
        out[side+'_pose_forearm_direction']=m[:,[b+2,b+3,b+4,b+5]].all(1)&(np.linalg.norm(v,axis=1)>1e-8)
    for active,b,other in [('R',28,0),('L',0,28)]:
        out[active+'_hand_on_opposite_pose_forearm']=m[:,[b,b+1,other+2,other+3,other+4,other+5]].all(1)&out[('L' if active=='R' else 'R')+'_pose_forearm_direction']
    return out

def support(ids,x,m,g,rawg,observed,usable,inside,finite):
    ids=np.asarray(ids,dtype=int);n=len(ids);res={}
    for k,v in g.items():
        valid=v[ids];raw=rawg[k][ids]
        z={'frames':n,'available_frames':int(valid.sum()),'available_fraction':float(valid.mean()) if n else None,
           'missing_before_fixed_roi_frames':int((~raw).sum()),
           'lost_by_fixed_roi_frames':int((raw&~valid).sum())}
        if k in GROUPS:
            sub=m[np.ix_(ids,GROUPS[k])]
            z.update(available_cells=int(sub.sum()),expected_cells=int(sub.size),columns=GROUPS[k])
        if 'hand_on_opposite' in k:
            adjacency=np.diff(ids)==1
            steps=valid[1:]&valid[:-1]&adjacency
            z.update(consecutive_within_span_pairs=int(adjacency.sum()),available_consecutive_pairs=int(steps.sum()),
                     motion_or_contact_verified=False)
        res[k]=z
    cause={}
    for s,off in [('L',33),('R',54)]:
        js=np.arange(off,off+21)
        o=observed[np.ix_(ids,js)];u=usable[np.ix_(ids,js)];f=finite[np.ix_(ids,js)];ri=inside[np.ix_(ids,js)]
        cause[s]={'expected_hand_point_cells':int(n*21),'not_observed_cells':int((~o).sum()),
          'observed_but_unusable_or_nonfinite_cells':int((o&(~u|~f)).sum()),
          'raw_valid_but_outside_fixed_roi_cells':int((o&u&f&~ri).sum()),
          'all_21_raw_valid_frames':int((o&u&f).all(1).sum()),
          'all_21_roi_valid_frames':int((o&u&f&ri).all(1).sum()),
          'no_hand_points_observed_frames':int((~o.any(1)).sum())}
    return {'groups':res,'point_missing_causes':cause}

clips=[];core_segments=[];feature_values=defaultdict(list);allmask=[];coremask=[];checks=[];cache={};phase_checks=0
for r in rows:
    a=r['alias'];pin(r['input']);pin(r['observation']);pin(r['record'])
    assert PINS[r['input']]==r['input_sha256']
    assert PINS[r['record']]==r['record_sha256']
    with np.load(r['input'],allow_pickle=False) as z:
        x=z['x'].astype(float);m=z['mask'];ts=z['timestamp_us'];fids=z['frame_id']
    with np.load(r['observation'],allow_pickle=False) as z:
        xy=z['xy_px'].astype(float);o=z['observed'];u=z['usable'];ots=z['timestamp_us'];of=z['frame_id']
    assert np.array_equal(fids,of) and np.array_equal(ts,ots)
    assert np.array_equal(fids,np.arange(len(x))) and x.shape==(len(x),69) and m.shape==x.shape
    assert np.isfinite(x).all() and np.all(x[~m]==0)
    finite=np.isfinite(xy).all(2);raw=o&u&finite
    x0,y0,x1,y1=r['box'];inside=(xy[:,:,0]>=x0)&(xy[:,:,0]<x1)&(xy[:,:,1]>=y0)&(xy[:,:,1]<y1)
    xx,mm,nn,nv=independent(xy,raw&inside,ts,fids)
    rawx,rawm,_,_=independent(xy,raw,ts,fids)
    assert np.array_equal(m,mm)
    err=float(np.max(np.abs(x[m]-xx[m])))
    assert np.allclose(x[m],xx[m],atol=1e-5,rtol=2e-7), (a,err)
    checks.append({'alias':a,'mask_exact':True,'feature_max_abs_difference_float32':err,'input_sha_match':True,'frame_pts_match':True})
    g=group_masks(x,m);rawg=group_masks(rawx,rawm)
    segs=[s for s in phase[a]['segments'] if s['role']=='core_candidate']
    assert [(s['segment_id'],s['core']) for s in segs]==[(s['segment_id'],s['core']) for s in draft[a]['segments'] if s['role']=='core_candidate']
    ids=sorted({f for s in segs for f in range(s['core'][0],s['core'][1]+1)})
    ev=sorted({f for s in segs for f in s['evidence_frames']})
    sup=lambda ii:support(ii,x,m,g,rawg,o,u,inside,finite)
    c={'alias':a,'person':r['person'],'word_id':r['word_id'],'label':r['label'],'frames':len(x),
      'core_candidate_native_frames':len(ids),'direct_rgb_evidence_frames':ev,
      'core_native_span_support':sup(ids),'direct_rgb_anchor_support':sup(ev),
      'whole_clip_support':sup(range(len(x))),'fixed_roi_source_pixels':r['box']}
    clips.append(c)
    for s in segs:
        ix=list(range(s['core'][0],s['core'][1]+1));sev=s['evidence_frames']
        for side,b in [('L',0),('R',28)]:
            previous=s['coordinate_support'][side]
            computed={'position_frames':int(m[np.ix_(ix,[b,b+1])].all(1).sum()),
              'palm_frames':int(m[np.ix_(ix,[b+8,b+9])].all(1).sum()),
              'arm_frames':int(m[np.ix_(ix,[b+2,b+3,b+4,b+5])].all(1).sum()),
              'angle_cells':int(m[np.ix_(ix,range(b+11,b+25))].sum()),'angle_expected':len(ix)*14,'frames':len(ix)}
            assert previous==computed,(a,s['segment_id'],side,previous,computed)
            phase_checks+=len(computed)
        core_segments.append({'alias':a,'person':r['person'],'word_id':r['word_id'],'segment_id':s['segment_id'],
          'core':s['core'],'core_pts_us':[int(ts[ix[0]]),int(ts[ix[-1]])],
          'description_ko':s['description_ko'],'native_span_support':sup(ix),'direct_rgb_anchor_support':sup(sev),
          'direct_rgb_evidence_frames':sev,'source_links_anchor_scope_only':s.get('source_links',[]),
          'dense_label_truth':False})
    for j,name in enumerate(r['feature_names']):
        feature_values[name].extend(x[m[:,j],j].tolist())
    allmask.append(m);coremask.append(m[ids])
    cache[a]=(r,x,m,g,rawg,o,u,inside,finite,ts)

def combine(part):
    result={}
    for k in clips[0][part]['groups']:
        z=[c[part]['groups'][k] for c in clips]
        n=sum(v['frames'] for v in z);known=sum(v['available_frames'] for v in z)
        result[k]={'available_frames':known,'expected_frames':n,'available_fraction':known/n,
          'missing_before_fixed_roi_frames':sum(v['missing_before_fixed_roi_frames'] for v in z),
          'lost_by_fixed_roi_frames':sum(v['lost_by_fixed_roi_frames'] for v in z)}
        if 'available_cells' in z[0]:result[k].update(available_cells=sum(v['available_cells'] for v in z),expected_cells=sum(v['expected_cells'] for v in z))
    return result

source_support=[]
for e in contracts['expressions']:
    for q in e['clauses']:
        ev=[v for v in q.get('evidence',[]) if v['alias'] in EXPECTED]
        per=[]
        for v in ev:
            a=v['alias'];r,x,m,g,rawg,o,u,inside,finite,ts=cache[a]
            ids=sorted({z['frame_id'] for z in v['anchors']})
            per.append({'alias':a,'person':r['person'],'anchor_frames':ids,
              'support':support(ids,x,m,g,rawg,o,u,inside,finite)})
        persons=sorted({v['person'] for v in per})
        source_support.append({'word_id':e['word_id'],'clause_id':q['id'],'source_span':q['source_span'],
          'fit_anchor_people':persons,'fit_anchor_person_count':len(persons),'fit_evidence':per,
          'multi_person_source_grounded_label_learning_established':False,
          'reason':'Only one fit person has source-aligned RGB anchors' if len(persons)==1 else 'No fit source-aligned anchors' if not persons else 'Availability is not a validated learning target'})

domains=[]
for name,vals in feature_values.items():
    if name.endswith('_angle'):unit='projected_2d_internal_angle_divided_by_180_degrees';domain='[0,1]; not 3D flexion';meaning='2D triplet interior angle'
    elif 'palm_axis_' in name or 'motion_direction_' in name:unit='unit_vector_component';domain='[-1,1]';meaning='image-plane unit direction; palm axis is not a palm normal'
    elif 'velocity_' in name:unit='shoulder_width_per_second';domain='signed, no fixed bound';meaning='difference of per-frame shoulder-normalized detector wrist position over consecutive native PTS'
    else:unit='shoulder_width';domain='nonnegative, no fixed upper bound' if 'distance' in name or name.endswith('_width') else 'signed, no fixed bound';meaning='per-frame shoulder-center/width normalized detector geometry'
    domains.append({'name':name,'unit':unit,'mathematical_domain':domain,'meaning':meaning,
      'fit_available_cells':len(vals),'fit_empirical_min':min(vals) if vals else None,'fit_empirical_max':max(vals) if vals else None,
      'empirical_range_is_not_normal_tolerance':True})

person_table=[]
expression_summary=[]
for wid in sorted({r['word_id'] for r in rows}):
    cs=[c for c in clips if c['word_id']==wid]
    expression_summary.append({'word_id':wid,'label':cs[0]['label'],'fit_people':[c['person'] for c in cs],
      'core_candidate_frames':sum(c['core_candidate_native_frames'] for c in cs),
      'groups':{k:{'available_frames':sum(c['core_native_span_support']['groups'][k]['available_frames'] for c in cs),
                   'missing_before_fixed_roi_frames':sum(c['core_native_span_support']['groups'][k]['missing_before_fixed_roi_frames'] for c in cs),
                   'lost_by_fixed_roi_frames':sum(c['core_native_span_support']['groups'][k]['lost_by_fixed_roi_frames'] for c in cs)}
                for k in cs[0]['core_native_span_support']['groups']}})
    person_table.append({'word_id':wid,'fit_aliases':[c['alias'] for c in cs],
       'groups':{k:{'people_with_any_numeric_support':[c['person'] for c in cs if c['core_native_span_support']['groups'][k]['available_frames']>0],
                     'per_person':[{ 'person':c['person'], **c['core_native_span_support']['groups'][k]} for c in cs]}
                 for k in cs[0]['core_native_span_support']['groups']},
       'LOPO_scope':'three fit people give only three held-fit folds, each reference pool has two people; an empirical fit range is not population normality or accepted linguistic variation'})

out={'status':'PASS_NUMERIC_SUPPORT_AUDIT_WITH_SEMANTIC_LIMITS','reviewer':'Codex agent A; separate calculation from root implementation, not independent human linguistic validation',
 'scope':{'aliases':EXPECTED,'people':[1,2,3],'fit_clips':12,'frames':sum(c['frames'] for c in clips),
          'core_candidate_segments':len(core_segments),'core_candidate_native_frames':sum(c['core_candidate_native_frames'] for c in clips),
          'direct_rgb_anchor_frames':sum(len(c['direct_rgb_evidence_frames']) for c in clips),
          'new_images_viewed_this_audit':0,'new_inference':0,'new_training':0,'dev_numeric_arrays_read':0,
          'prior_fit_rgb_annotations_reused_readonly':True},
 'essential_interpretation':[
   '69 values are saved detector estimates with explicit masks. A zero value at mask=false is storage filling, never a fist, rest, correct pose or evidence.',
   'observed/usable means detector availability/eligibility, not actual anatomical truth. Pose wrist and hand detector wrist are distinct estimates; no fallback substitution.',
   'Finger and elbow angles are 2D projected interior angles divided by 180, not 3D anatomical flexion. MCP construction uses wrist/MCP/next finger joint.',
   'Palm axis is wrist to middle MCP in the image plane, not palm-facing direction or surface normal. Hidden finger detail/contact cannot be recovered from a mask.',
   'Position/distance use each frame shoulder center and shoulder width. Translation/positive-scale normalization is not viewpoint invariance; apparent velocity also includes changing normalizer.',
   'Forearm direction is derived from elbow to pose wrist, with both estimates valid; opposite-arm location/progress is a 2D spatial proxy, not contact, brushing correctness, depth or intention.',
   'Counts over native core spans describe numeric support only. The phase draft has sampled RGB anchors and uncertain boundaries; it is not dense semantic GT.',
   'No side or irrelevant body part is required globally: the resting left hand of a right-hand action can be outside ROI without making that action incorrect.',
   'SOURCE_PHASE_CONTRACTS has fit anchors only for person1, or none for the alternative clause. It does not establish multi-person source-grounded label learning.',
   'Fit3 can support empirical separate-reference geometry comparison and fit LOPO diagnosis. It cannot establish a normal population tolerance or linguistic acceptability range.'
 ],
 'feature_domains':domains,'aggregate_core_native_span_support':combine('core_native_span_support'),
 'aggregate_direct_rgb_anchor_support':combine('direct_rgb_anchor_support'),
 'clips':clips,'core_segments':core_segments,'fit_person_support_by_expression':person_table,
 'per_expression_core_summary':expression_summary,'prior_phase_data_support_integer_checks':phase_checks,
 'source_clause_fit_anchor_support':source_support,'reconstruction_checks':checks,
 'mask_filling':{'masked_cells':int((~np.concatenate(allmask)).sum()),'all_masked_values_are_exact_zero':True,'zero_implies_missing':False},
 'checks_scope':'all 69 numeric values/masks independently reconstructed for fit12; hashes/PTS/frame IDs verified; no model call, label refit, tolerance fit or DEV outcome used',
 'input_pins':PINS,'all_input_pins_unchanged_at_end':all(sha(p)==h for p,h in PINS.items())}
assert out['all_input_pins_unchanged_at_end']
dest=UNIT/'review'/'FIT_FEATURE_SUPPORT.json'
dest.write_text(json.dumps(out,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
agg=out['aggregate_core_native_span_support'];lines=[
 '공개 fit 12개 손모양·몸 관계 입력 지원 감사',
 f"전체 {out['scope']['frames']}프레임 중 핵심 후보 {len(core_segments)}구간, {out['scope']['core_candidate_native_frames']}프레임을 확인했다. 이는 저장 수치의 존재 여부이며 모든 프레임의 수어 정답 검증이 아니다.",
 '기존 원본 관측과 x69를 읽기 전용으로 재계산했다. 69개 mask가 모두 같고 수치는 float32 오차 범위에서 일치했다. 새 학습·모델 추론·개발4 숫자 읽기는 하지 않았다.', '',
 '핵심 후보의 수치 지원(좌/우는 해부학적 좌우):']
for k in ['L_shape_14','R_shape_14','L_hand_wrist_position','R_hand_wrist_position','L_elbow_bend','R_elbow_bend','L_pose_forearm_direction','R_pose_forearm_direction','two_hand_wrist_relation','R_hand_on_opposite_pose_forearm']:
 z=agg[k];lines.append(f"- {k}: {z['available_frames']}/{z['expected_frames']}프레임. 원본 지원 없음 {z['missing_before_fixed_roi_frames']}, ROI로 제외 {z['lost_by_fixed_roi_frames']}.")
lines += ['', '표현별로 필요한 근거를 나눠 보면:']
for w in expression_summary:
    z=w['groups'];n=w['core_candidate_frames']
    lines.append(f"- {w['label']}: 오른손 각도 {z['R_shape_14']['available_frames']}/{n}, 왼손 각도 {z['L_shape_14']['available_frames']}/{n}, 양손 위치관계 {z['two_hand_wrist_relation']['available_frames']}/{n}, 오른손-왼전완 관계 {z['R_hand_on_opposite_pose_forearm']['available_frames']}/{n}.")
lines.append('순종 왼손 각도 0/139는 아래 둔 왼손의 ROI 제외다. 오른손 동작이 틀렸다는 뜻이 아니며 양손 지원을 모든 표현의 필수 조건으로 삼을 수 없다.')
lines += ['', '해석:',
 '손가락 각도는 화면에 투영된 세 점의 내각/180이다. 실제 3D 굽힘이나 손모양 정답이 아니다. 손바닥 축도 손바닥이 바라보는 방향이 아니다.',
 '손검출 손목과 몸검출 손목은 별도 추정치다. 몸검출 손목이 있다고 손검출 누락을 채우지 않는다.',
 'mask=0의 저장값0은 주먹·정지·정상 자세를 뜻하지 않는다. 정상적으로 내려둔 손이 ROI 밖인 경우도 오류로 부르지 않는다.',
 '반대팔 관계는 팔꿈치→몸검출 손목 축에 대한 화면상의 위치/이동을 비교할 수 있다는 뜻이다. 접촉·스치기 성공·깊이 관계까지 확인된 것은 아니다.',
 '사전 설명과 직접 연결한 fit 앵커는 표현별 ID1 한 사람에게만 있다(대체 표현은 0명). 따라서 다인 source 라벨 학습 성립 주장은 불가하다.',
 '세 사람 각각의 수치와 시각 단계 초안은 분리해서 참고할 수 있다. fit LOPO는 3개 접힘과 매번 2명 참고뿐이므로 경험적 범위이며 정상/허용 범위의 일반 정답이 아니다.',
 '순종 앞선 얼굴 펼침 구간은 사전 미포함이어도 보존한다. 핵심 후보 전체를 조밀한 정답이나 필수 언어 순서로 변환하지 않는다.']
(UNIT/'review'/'FIT_FEATURE_SUPPORT_KO.txt').write_text('\n'.join(lines)+'\n',encoding='utf-8')
print(json.dumps({'status':out['status'],'scope':out['scope'],'aggregate':agg,'output':str(dest)},ensure_ascii=False))
