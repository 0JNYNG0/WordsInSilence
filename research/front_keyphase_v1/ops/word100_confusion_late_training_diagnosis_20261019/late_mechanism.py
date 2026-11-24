from common import *
import torch
from collections import Counter
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

def main():
 a=read(U/'SAVED_ANALYSIS.json');runs={(r['arm'],r['seed']):r for r in a['runs']};seeds=[20261013,20261014,20261015];arms=['C','Q','R10','R01'];cor=[];states=[];pins={}
 for s in seeds:
  c=runs['C',s];y=np.array([x['sensor'] for x in c['late_series']],float)
  for arm in arms:
   r=runs[arm,s];z=np.array([x['sensor'] for x in r['late_series']],float)
   if arm!='C':cor.append({'seed':s,'arm':arm,'late_top1_r_with_C':float(np.corrcoef(y,z)[0,1]),'late_adjacent_delta_r_with_C':float(np.corrcoef(np.diff(y),np.diff(z))[0,1])})
   rp=resultpath(arm,s);lastp=Path(str(rp).replace('_RESULT.json','_last.pt'));pins[str(lastp)]=sha(lastp)
   last=torch.load(lastp,map_location='cpu',weights_only=True);state=last['model'];best=last['best'].get('state_dict',last['best'].get('model'));buf=[k for k in state if k.endswith(('running_mean','running_var'))];par=[k for k in state if not k.endswith(('running_mean','running_var','num_batches_tracked'))]
   def change(keys):
    delta=sum(float(((state[k].double()-best[k].double())**2).sum()) for k in keys);base=sum(float((best[k].double()**2).sum()) for k in keys)
    return {'relative_l2':float(np.sqrt(delta/base)),'absolute_l2':float(np.sqrt(delta))}
   states.append({'arm':arm,'seed':s,'selected_epoch':r['selected_epoch'],'final_epoch':last['epoch'],'parameter_change':change(par),'running_buffer_change':change(buf),'optimizer_lrs':[g['lr'] for g in last['optimizer']['param_groups']],'note':'Different kinds of tensors; relative changes do not quantify accuracy contribution.'})
 wrong_by_signer=Counter();class_focus=[]
 for cid,q in a['clips'].items():wrong_by_signer[str(q['signer'])]+=q['wrong']
 for cid in a['focus']:
  q=a['clips'][cid];same={str(x['signer']):x['wrong'] for x in a['clips'].values() if x['label']==q['label']};class_focus.append({'clip':cid,'label':q['label'],'wrong_by_signer_over_12':same})
 phases={str(s):{arm:runs[arm,s]['phase_demeaned_top1'] for arm in arms} for s in seeds}
 out={'at':now(),'correlations':cor,'state_changes':states,'wrong_by_signer':dict(wrong_by_signer),'focus_by_signer':class_focus,'phase_associations':phases,
 'ceil_batch_rule':{'original':[32]*28+[3],'proposed_generic_balanced_partition':[31]*29,'both_sources':899,'both_steps':29,'candidate_executed':False},
 'interpretation':'Shared same-seed late fluctuations support a common training-process hypothesis; correlation and saved state differences do not isolate LR, gradients, BatchNorm, dropout or schedule causality. All 12 runs improved late mean accuracy/loss versus epochs41-60, so monotonic late deterioration is not supported.',
 'pins':pins,'code_sha256':sha(__file__),'new_inference':0,'training':0,'production_changed':False}
 write(U/'LATE_MECHANISM.json',out)
 plt.rcParams['font.family']='Malgun Gothic';plt.rcParams['axes.unicode_minus']=False
 fig,axs=plt.subplots(1,3,figsize=(16,4.7),sharey=True);colors={'C':'#164e63','Q':'#c46d26','R10':'#798993','R01':'#9d519c'}
 for ax,s in zip(axs,seeds):
  for arm in arms:
   rr=runs[arm,s];ax.plot([x['epoch'] for x in rr['late_series']],[x['sensor']/3 for x in rr['late_series']],label=arm,color=colors[arm],linewidth=2 if arm=='C' else 1.2,marker='o',markersize=2)
  ax.set(title=f'초기값 {s}',xlabel='학습 회차');ax.grid(alpha=.2);ax.set_ylim(90,100);ax.set_xticks([61,65,70,75,80])
 axs[0].set_ylabel('공개 개발 검증 정답률 (%)');axs[-1].legend(ncol=2,loc='lower right');fig.suptitle('학습 후반의 흔들림은 기존 C에도 나타남 — 저장 기록 61~80회');fig.tight_layout();fig.savefig(U/'late_curves.png',dpi=150);plt.close(fig)
 fig,ax=plt.subplots(figsize=(10,4));names=[a['clips'][cid]['label'] for cid in reversed(a['focus'])];vals=[a['clips'][cid]['wrong'] for cid in reversed(a['focus'])];ax.barh(names,vals,color='#276d83');ax.set_xlim(0,12);ax.set_xlabel('같은 공개 영상에서 틀린 횟수 / 저장된 12개 모델');ax.set_title('반복 오답 상위 5개 영상 — 새 사용자 12명 시험이 아님')
 for i,x in enumerate(vals):ax.text(x+.1,i,f'{x}/12',va='center')
 fig.tight_layout();fig.savefig(U/'repeated_errors.png',dpi=150);plt.close(fig)
 print(json.dumps({'correlations':cor,'wrong_by_signer':dict(wrong_by_signer),'focus':class_focus,'all_late_improve':all(r['late']['sensor_mean']>r['preceding']['sensor_mean'] and r['late']['ce_mean']<r['preceding']['ce_mean'] for r in a['runs'])},ensure_ascii=False))
if __name__=='__main__':main()
