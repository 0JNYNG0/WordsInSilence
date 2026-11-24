"""Arithmetic-only held-record comparison of the single fixed physical task."""
from physical_branch import *

def main():
    cfg=read(U/'PHYSICAL_FROZEN.json');gate=read(U/'PHYSICAL_ADOPTION_CONTRACT.json');complete=read(U/'PHYSICAL_TRAINING_COMPLETE.json')
    for p,h in cfg['pins'].items():assert sha(p)==h,p
    pre=read(U/'PHYSICAL_PRETRAIN_VERIFICATION.json')
    for p,h in pre.get('additional_pins',{}).items():assert sha(p)==h,p
    rows=read(U/'PHYSICAL_INDEX.json')['rows'];z=np.load(U/'PHYSICAL_INPUTS.npz');y=z['y'];eligible=z['eligible'];known=y>=0
    group=lambda key,value:np.array([r[key]==value for r in rows])
    cross=np.array([r['word_id']=='WORD1567'and r['kind'].startswith('main_')for r in rows]);short=np.array([r['source_key']in gate['short_change']['source_hand_keys']for r in rows])
    assert int(short.sum())==36
    metric_rows=[];cell_rows=[];run_summaries=[];failed=[];seed_status=[]
    for seed in SEEDS:
        logit=np.full(len(rows),np.nan);rule=np.zeros(len(rows),bool);seen=np.zeros(len(rows),int)
        for person in PEOPLE:
            r=next(r for r in complete['runs']if r['person']==person and r['seed']==seed);assert sha(r['prediction'])==r['prediction_sha256']and sha(r['model'])==r['model_sha256']
            d=np.load(r['prediction']);take=z['person']==person;assert np.array_equal(d['held'],take)and np.array_equal(d['eligible'],eligible);logit[take]=d['logits'][take];rule[take]=d['rule_prediction'][take];seen[take]+=1
            w=np.load(U/'physical_folds'/str(person)/'WEIGHTS.npz')['weights'];tr=w>0
            run_summaries.append(dict(person=person,seed=seed,train_known_eligible=int(tr.sum()),train_mlp_correct=int(((d['logits']>=0)==y)[tr].sum()),train_rule_correct=int((d['rule_prediction']==y)[tr].sum()),seconds=r['training_seconds']))
        assert np.isfinite(logit).all()and(seen==1).all()
        pred=np.where(eligible,(logit>=0).astype(np.int8),-1);base=np.where(eligible,rule.astype(np.int8),-1)
        seedfailed=[]
        for view in ['full','upper','actual_crop']:
            for reverse in [False,True]:
                condition=group('view',view)&group('reverse',reverse)
                strata={'all':np.ones(len(rows),bool),'M':y==1,'H':y==0,'R':group('side','R'),'L':group('side','L')}
                for side in ['R','L']:
                    for val,name in [(1,'M'),(0,'H')]:strata[f'{side}_{name}']=group('side',side)&(y==val)
                for side in [None,'R','L']:strata['cross_cycle'+('_'+side if side else'')]=cross&(group('side',side)if side else True)
                strata['short_change']=short
                for name,mask in strata.items():
                    take=condition&known&mask;n=int(take.sum());a=int((pred[take]==y[take]).sum());b=int((base[take]==y[take]).sum())
                    m=dict(seed=seed,view=view,reverse=reverse,stratum=name,n=n,observed=int(eligible[take].sum()),unresolved=int((~eligible[take]).sum()),MLP=a,rule=b,net=a-b,lost=int(((base==y)&(pred!=y)&take).sum()),gained=int(((base!=y)&(pred==y)&take).sum()))
                    metric_rows.append(m)
                    if n and a<b:failed.append(m);seedfailed.append(m)
                # Source-record summaries are descriptive, not substituted for pooled gates.
                for person in PEOPLE:
                    take=condition&known&(z['person']==person)
                    metric_rows.append(dict(seed=seed,view=view,reverse=reverse,stratum=f'record_{person}',n=int(take.sum()),observed=int(eligible[take].sum()),unresolved=int((~eligible[take]).sum()),MLP=int((pred[take]==y[take]).sum()),rule=int((base[take]==y[take]).sum()),descriptive_only=True))
        for i,r in enumerate(rows):
            cell_rows.append(dict(**r,seed=seed,truth=int(y[i]),eligible=bool(eligible[i]),MLP=int(pred[i]),rule=int(base[i]),logit=float(logit[i]),short_change=bool(short[i]),cross_cycle=bool(cross[i])))
        primary=next(m for m in metric_rows if m['seed']==seed and m['view']=='full'and m['reverse']==False and m['stratum']=='all')
        seed_status.append(dict(seed=seed,all_preservation_gates_pass=not seedfailed,failed_strata=len(seedfailed),primary=primary))
    passgate=not failed;strictgain=any(s['primary']['MLP']>s['primary']['rule']for s in seed_status)
    decision='CANDIDATE_FOR_FURTHER_LINKING_ONLY'if passgate and strictgain else'NOT_ADOPTED'
    put(U/'PHYSICAL_COMPARISON.json',dict(task=cfg['task'],status='CALCULATED_AWAIT_SEPARATE_VERIFICATION',decision=decision,all_preservation_gates_pass=passgate,at_least_one_primary_strict_gain=strictgain,seeds=seed_status,metrics=metric_rows,failures=failed,training=run_summaries,counts=dict(source_windows=115,source_hand_cells=230,known_source_cells=228,unknown_truth_source_cells=2,conditions_per_seed=6,total_saved_held_cells=len(cell_rows)),limits='Window-level physical translation only. No old-task percentage difference, no G2 pass, no automatic interval or completed action claim.'))
    put(U/'PHYSICAL_HELD_PREDICTIONS.json',dict(rows=cell_rows,labels={1:'M',0:'H',-1:'U'}))
    lines=['몸 기준 손목 이동/유지 보조 판정부 — 고정 비교 결과(별도검산 전)',f'판정: {decision}','',
      '기존 “해당 설명 행동이 아님”과 실제 “손목이 멈춤” 정답을 분리했다. 기존301개 정답과 모델은 보존했다.',
      '새 직접 RGB 정답:115창×양손=230항목, M142/H86/U2.6기록ID4표현이며 같은영상의 변형을 새영상으로 세지 않는다.',
      '학습:353개 가중치 공통 양손 판정부1조건×6제외기록×3초기값=18회. 비교는 같은5학습기록에서 문턱을 정한 단순 손목위치변화 규칙이다.']
    for s in seed_status:
        p=s['primary'];lines.append(f"초기값{s['seed']} 원본 정순: 규칙{p['rule']}/228 → 새판정부{p['MLP']}/228, 관측U{p['unresolved']}; 무회귀 조건 {'통과'if s['all_preservation_gates_pass']else'미통과'}.")
    lines+=['','이 점수는 기존 C6 설명소속 점수와 다른 과제라 개선율로 빼서 보고할 수 없다.','손모양·회전·방향 정오·시작/끝·설명소속·완료를 이 보조 출력 하나가 판단한 것이 아니다. 새 웹/운영 모델로 교체하지 않았다.','관측U를 정답 분모에서 빼지 않았다. 실제crop은 손목이 화면 밖으로 나간 사례를 U로 유지한다.','기존100단어/C·T3-F/사용자 촬영은 변경·학습하지 않았다. 동일 후보의 추가 튜닝이나 반복은 하지 않는다.']
    (U/'RESULT_KO.txt').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    put(U/'GRAPH_PATCH.json',dict(branch='motion_recovery_physical_wrist',G1='DIRECT_SOURCE_CONTRACT_REPAIRED',G2='REWORK',G3='PENDING_ROOT_INTEGRATION',comparison=decision,separate_verification='PENDING',old_models='UNCHANGED',no_deployment=True,no_G2_pass_claim=True,artifacts=['PHYSICAL_SOURCE_LABELS.json','PHYSICAL_ADOPTION_CONTRACT.json','PHYSICAL_COMPARISON.json','PHYSICAL_HELD_PREDICTIONS.json','RESULT_KO.txt']))
    print(json.dumps(dict(decision=decision,seeds=seed_status),ensure_ascii=False))

if __name__=='__main__':main()
