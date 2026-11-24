"""Read-only retrospective analysis of saved predictions. No torch/model forward."""
from pathlib import Path
from collections import Counter
import base64, hashlib, io, json
from datetime import datetime, timezone, timedelta
import numpy as np
from PIL import Image
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties

U = Path(__file__).resolve().parent
S = U.parent / 'word100_cpu_model_selection_20261028'
SEEDS = [28101, 28102, 28103]
NAMES = {'v3': 'MobileNetV3-Small', 'repvit': 'RepViT-M1.0', 'v4hybrid': 'MobileNetV4 Hybrid-Medium'}
def read(p): return json.loads(p.read_text(encoding='utf-8-sig'))
def sha(p): return hashlib.sha256(p.read_bytes()).hexdigest()
def save(name, value): (U/name).write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding='utf-8')

def main():
    (U/'figures').mkdir(exist_ok=True)
    manifest = read(S/'TRAINING_MANIFEST.json')
    classes = manifest['classes']; k = len(classes)
    rows = {r['id']: r for r in manifest['rows']}
    assert len(rows) == 2977 and k == 45
    assert {r['role'] for r in rows.values()} == {'fit', 'validation'}
    source_pins = {str(S/'TRAINING_MANIFEST.json'): sha(S/'TRAINING_MANIFEST.json')}
    runs = {}; arrays = {}; checks = []
    for model in NAMES:
        runs[model] = []
        for seed in SEEDS:
            p = S/f'models/{model}_{seed}_dev.npz'; q = S/f'models/{model}_{seed}.json'
            source_pins[str(p)] = sha(p); source_pins[str(q)] = sha(q)
            with np.load(p, allow_pickle=False) as z:
                ids, y, signer, logits = [z[n].copy() for n in ['ids','targets','signer','logits']]
            assert logits.shape == (488,45) and np.isfinite(logits).all()
            assert all(rows[x]['role']=='validation' and rows[x]['target']==int(t) for x,t in zip(ids,y))
            if arrays:
                first=next(iter(arrays.values()))
                assert np.array_equal(first['ids'],ids) and np.array_equal(first['y'],y)
            pred=logits.argmax(1)
            cm=np.zeros((k,k),dtype=np.int64)
            np.add.at(cm,(y,pred),1)
            record=read(q)
            assert cm.tolist()==record['dev']['confusion']
            assert int(np.trace(cm))==record['dev']['correct'] and int(cm.sum())==488
            arrays[(model,seed)] = dict(ids=ids,y=y,pred=pred,signer=signer)
            runs[model].append(dict(seed=seed,cm=cm.tolist(),correct=int(np.trace(cm)),accuracy=float(np.trace(cm)/488)))
            checks.append(f'{model}_{seed}: stored matrix equals saved logits argmax')
    y=arrays[('v4hybrid',SEEDS[0])]['y']; ids=arrays[('v4hybrid',SEEDS[0])]['ids']
    support=np.bincount(y,minlength=k)
    fit=Counter(r['target'] for r in rows.values() if r['role']=='fit')
    v4pred=np.stack([arrays[('v4hybrid',s)]['pred'] for s in SEEDS])
    v3pred=np.stack([arrays[('v3',s)]['pred'] for s in SEEDS])
    total=np.sum([r['cm'] for r in runs['v4hybrid']],axis=0)
    pairs=[]
    for i in range(k):
        for j in range(i+1,k):
            n=int(total[i,j]+total[j,i])
            if n: pairs.append(dict(a=i,b=j,labels=[classes[i],classes[j]],count=n,ab=int(total[i,j]),ba=int(total[j,i])))
    pairs.sort(key=lambda x:(-x['count'],x['a'],x['b']))
    error=int(total.sum()-np.trace(total))
    stats=[]
    for i,label in enumerate(classes):
        row=total[i]; guessed=int(total[:,i].sum()); hit=int(total[i,i])
        stats.append(dict(label=label,fit_count=fit[i],dev_count=int(support[i]),correct_votes=hit,
                          recall=None if not support[i] else hit/int(row.sum()),precision=None if not guessed else hit/guessed,
                          predicted_votes=guessed))
    paired=dict(v4_fixes=int(((v3pred!=y)&(v4pred==y)).sum()),v4_regressions=int(((v3pred==y)&(v4pred!=y)).sum()))
    assert paired['v4_fixes']-paired['v4_regressions']==int(np.trace(total)-sum(r['correct'] for r in runs['v3']))
    summary=dict(at=datetime.now(timezone(timedelta(hours=9))).isoformat(),status='COMPLETE_SAVED_PREDICTION_DIAGNOSIS',
                 images=488,prediction_events=1464,independent_images=488,fit_images=2489,fit_people=4,dev_people=3,
                 class_count=45,dev_supported_classes=int((support>0).sum()),
                 absent_labels=[classes[i] for i in range(k) if not support[i]],
                 v4_correct_votes=int(np.trace(total)),v4_wrong_votes=error,v4_accuracy=float(np.trace(total)/total.sum()),
                 wrong_all_three=int(np.all(v4pred!=y,axis=0).sum()),right_all_three=int(np.all(v4pred==y,axis=0).sum()),
                 mixed=int((np.any(v4pred==y,axis=0)&np.any(v4pred!=y,axis=0)).sum()),
                 top4_error_share=sum(p['count'] for p in pairs[:4])/error,nonzero_confused_pairs=len(pairs),
                 paired_same_seed=paired,fit_mode_baseline=dict(label=classes[fit.most_common(1)[0][0]],correct=int(support[fit.most_common(1)[0][0]])),
                 per_class=stats,top_pairs=pairs,models={m:dict(accuracy_mean=float(np.mean([r['accuracy'] for r in rr])),correct_votes=sum(r['correct'] for r in rr)) for m,rr in runs.items()},
                 training=False,new_inference=False,operating_changed=False,user_mentor_used=False,old_final382_used=False,
                 limitations=['development-only retrospective diagnosis','three predictions per same photo, not 1464 independent images',
                              'handshape classification, not KSL word/educational correctness','no causal visual diagnosis or new ground truth'])
    save('SUMMARY.json',summary)
    save('MATRICES.json',dict(classes=classes,runs=runs,fit_counts=[fit[i] for i in range(k)],dev_counts=support.tolist()))

    # Scientific plots: original source labels preserved, zero-support rows masked.
    plt.rcParams['font.family']='Malgun Gothic'; plt.rcParams['axes.unicode_minus']=False
    active=np.flatnonzero(support)
    normalized=np.full((k,k),np.nan)
    normalized[active]=total[active]/total[active].sum(axis=1,keepdims=True)*100
    cmap=plt.get_cmap('Blues').copy(); cmap.set_bad('#e5e7eb')
    fig,ax=plt.subplots(figsize=(14,13.5))
    im=ax.imshow(np.ma.masked_invalid(normalized),cmap=cmap,vmin=0,vmax=100,interpolation='nearest')
    ax.set_xticks(range(k),classes,rotation=90,fontsize=8)
    ax.set_yticks(range(k),[f'{label}  (n={support[i]})' for i,label in enumerate(classes)],fontsize=8)
    ax.set_xlabel('모델이 고른 손 모양 · 원자료 라벨',fontsize=12)
    ax.set_ylabel('공개 자료의 정답 라벨 · n은 고유 사진 수',fontsize=12)
    ax.set_title('V4 손 모양 오차 행렬 · 같은 개발 사진 488장 × 학습 3회\n행별 비율: 대각선은 정답, 나머지는 혼동 / 회색 행은 평가 자료 없음',fontsize=15,pad=22)
    for i,j in zip(*np.where(np.isfinite(normalized)&(normalized>=10))):
        ax.text(j,i,f'{normalized[i,j]:.0f}',ha='center',va='center',fontsize=6,color='white' if normalized[i,j]>55 else '#14243c')
    fig.colorbar(im,ax=ax,shrink=.7,label='해당 정답 라벨 안에서의 예측 비율 (%)')
    fig.tight_layout(); fig.savefig(U/'figures/v4_confusion_full.png',dpi=160); fig.savefig(U/'figures/v4_confusion_full.svg'); plt.close(fig)

    fig,(ax,bx)=plt.subplots(1,2,figsize=(14,6),gridspec_kw={'width_ratios':[1.2,1]})
    top=pairs[:8]; labs=[' ↔ '.join(p['labels']) for p in top]; vals=[p['count'] for p in top]
    ax.barh(labs[::-1],vals[::-1],color='#3679b5')
    for yi,v in enumerate(vals[::-1]): ax.text(v+.4,yi,str(v),va='center')
    ax.set_xlim(0,max(vals)*1.16); ax.set_xlabel('오답 예측 건수 · 3회 합계 (동일 사진 중복 포함)')
    ax.set_title('어떤 손 모양끼리 헷갈렸나?',loc='left',fontweight='bold',pad=18)
    ax.spines[['top','right']].set_visible(False)
    labels=['3회 모두 정답','실행에 따라 달라짐','3회 모두 오답']
    numbers=[summary['right_all_three'],summary['mixed'],summary['wrong_all_three']]
    bx.barh(labels[::-1],numbers[::-1],color=['#b55741','#d3a745','#338c77'])
    for yi,v in enumerate(numbers[::-1]): bx.text(v+2,yi,f'{v}장',va='center')
    bx.set_xlim(0,max(numbers)*1.23); bx.set_xlabel('서로 다른 개발 사진 수 · 합계 488장')
    bx.set_title('같은 사진을 세 모델이 어떻게 봤나?',loc='left',fontweight='bold',pad=18)
    bx.spines[['top','right']].set_visible(False)
    fig.suptitle(f'V4 개발 정확도 평균 {summary["v4_accuracy"]*100:.2f}% · 많이 헷갈린 네 쌍은 전체 오답의 {summary["top4_error_share"]*100:.1f}%',fontsize=17,y=1.01)
    fig.tight_layout(); fig.savefig(U/'figures/v4_overview.png',dpi=160,bbox_inches='tight'); plt.close(fig)

    # Small, deterministic public examples. No user/final-test media read.
    images_read={}
    def image_data(r):
        p=Path(r['image']); assert r['role'] in ('fit','validation')
        assert p.resolve().is_relative_to((U.parent/'word100_feature_learning_pipeline_20261027/data/deephand_images').resolve())
        assert sha(p)==r['image_sha256']; images_read[r['id']]=dict(role=r['role'],path=str(p),sha256=r['image_sha256'])
        with Image.open(p) as im:
            im=im.convert('RGB'); im.thumbnail((180,180)); b=io.BytesIO(); im.save(b,format='PNG')
        return 'data:image/png;base64,'+base64.b64encode(b.getvalue()).decode()
    examples=[]
    for pair in pairs[:6]:
        a,b=pair['a'],pair['b']; chosen=[]
        candidates=[idx for idx,t in enumerate(y) if (t==a and np.any(v4pred[:,idx]==b)) or (t==b and np.any(v4pred[:,idx]==a))]
        for idx in sorted(candidates,key=lambda ii:str(ids[ii]))[:3]:
            r=rows[str(ids[idx])]
            chosen.append(dict(id=r['id'],label=r['label'],signer=r['signer'],predictions=[classes[int(x)] for x in v4pred[:,idx]],image=image_data(r),source_width=r['width'],source_height=r['height']))
        anchors=[]
        for cl in [a,b]:
            r=min((r for r in rows.values() if r['role']=='fit' and r['target']==cl),key=lambda r:r['id'])
            anchors.append(dict(label=r['label'],signer=r['signer'],image=image_data(r),id=r['id']))
        examples.append(dict(**pair,examples=chosen,training_examples=anchors))
    save('EXAMPLE_SELECTION.json',dict(rule='top6 unordered confusion pairs; first3 eligible dev IDs ascending; first fit ID per label; no visual cherry-picking',examples=[{**p,'examples':[{k:v for k,v in e.items() if k!='image'} for e in p['examples']],'training_examples':[{k:v for k,v in e.items() if k!='image'} for e in p['training_examples']]} for p in examples],images_read=images_read))
    source_pins.update({v['path']:v['sha256'] for v in images_read.values()})
    payload=dict(classes=classes,runs=runs,support=support.tolist(),fit_counts=[fit[i] for i in range(k)],summary=summary,examples=examples,names=NAMES)
    html=(U/'template.html').read_text(encoding='utf-8').replace('__PAYLOAD__',json.dumps(payload,ensure_ascii=False).replace('</','<\\/'))
    (U/'RESULT_KO.html').write_text(html,encoding='utf-8')
    save('SOURCE_PINS.json',source_pins)
    for p,h in source_pins.items(): assert sha(Path(p))==h
    for p,h in read(S/'PROTECTED.json').items(): assert sha(Path(p))==h
    save('BUILD_VERIFY.json',dict(status='PASS',saved_matrix_checks=checks,classes=45,unique_dev_images=488,
                                 source_pins=len(source_pins),operating_pins=len(read(S/'PROTECTED.json')),
                                 train_infer=False,user_media=False,verification='builder arithmetic only; separate audit follows'))
    print(json.dumps({k:v for k,v in summary.items() if k in ['v4_accuracy','right_all_three','wrong_all_three','mixed','paired_same_seed','absent_labels']},ensure_ascii=False))
if __name__=='__main__': main()
