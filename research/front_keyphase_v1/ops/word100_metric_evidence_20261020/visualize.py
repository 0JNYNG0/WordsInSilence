"""Evidence timelines for the six previously inspected public videos."""
from pathlib import Path
import json,numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
U=Path(__file__).resolve().parent;OLD=U.parent/'word100_multiref_observation_compare_20261019'
read=lambda p:json.loads(Path(p).read_text('utf-8-sig'))
def load(p):
    with np.load(p,allow_pickle=False) as z:return {k:z[k] for k in z.files}
def main():
    assert read(U/'VERIFICATION.json')['status']=='PASS';items=read(OLD/'VISUAL_INDEX.json');result=[];(U/'visual').mkdir(exist_ok=True)
    plt.rcParams.update({'font.family':'Malgun Gothic','axes.unicode_minus':False,'font.size':10})
    frozen=read(U/'FROZEN.json');lookup={r['clip_id']:r for r in frozen['inputs']}
    for item in items:
        cid=item['clip_id'];row=lookup[cid];new=load(U/'features/sensor'/(cid+'.npz'));meta=read(OLD/'inputs/sensor'/(cid+'.json'));old=load(meta['sample_path']);obs=load(row['inputs']['sensor'])
        ts=new['full_timestamp_us']/1e6;samplet=new['timestamp_us']/1e6
        fig,axes=plt.subplots(4,1,figsize=(12,8),gridspec_kw={'height_ratios':[3,1,3,1]},sharex=True)
        # feature present columns: bend_L/bend_R are the last two groups.
        for si,(side,name) in enumerate([('L','왼손'),('R','오른손')]):
            ax=axes[si*2];ix=-2 if side=='L' else -1
            ax.step(samplet,old['present'][:,ix].astype(int)*15,where='mid',label='기존: 손 전체 조건을 통과한 각도 수',color='#94a3b8',linewidth=2)
            ax.step(ts,new['full_'+side+'_angle_count'],where='mid',label='보완: 실제 계산 가능한 각도 수',color='#1669a2',linewidth=1.5)
            missing=~new['full_'+side+'_any_hand_points'];inside=new['full_'+side+'_inside_source_but_view_missing']
            ax.scatter(ts[inside],np.ones(inside.sum())*16,marker='x',color='#d14e3b',label='전체 원본 안 손 근거 있음 / 이 관측엔 손점 없음')
            for lo,hi in row['intervals']:ax.axvspan(lo,hi,color='#d6e9f1',alpha=.25)
            ax.set_ylim(-1,18);ax.set_yticks([0,3,6,9,12,15]);ax.set_ylabel(name+'\n각도 개수 / 15');ax.grid(alpha=.18)
            if si==0:ax.legend(loc='upper left',fontsize=8)
            ax2=axes[si*2+1];st=new['full_'+side+'_offline_full_view_state']
            edges=np.r_[ts[0]-(ts[1]-ts[0])/2,(ts[:-1]+ts[1:])/2,ts[-1]+(ts[-1]-ts[-2])/2]
            ax2.pcolormesh(edges,[0,1],st[None,:],cmap=ListedColormap(['#cbd5e1','#e6b94a','#65aa87','#a56836']),vmin=0,vmax=3,shading='flat')
            ax2.set_yticks([]);ax2.set_ylabel('전체 원본\n위치 근거',fontsize=9)
        axes[-1].set_xlabel('영상 시간(초) · 연한 배경=공개 주석 동작 구간')
        fig.suptitle(f'{item["label"]} · ID{cid.split("REAL")[1][:2]} | 웹캠 경계 모사 관측의 근거 분리',fontsize=15,y=.98)
        fig.text(.1,.015,'위치색: 회색=불명/부분 · 노랑=손 전체가 하단 밖 · 초록=손 전체가 범위 안 · 갈색=관측된 낮은 정지 구간\n전체 원본의 위치 근거는 오프라인 진단용입니다. 실제 잘린 웹캠에서 화면 밖 손을 안다는 뜻이 아닙니다.',fontsize=9)
        fig.subplots_adjust(left=.09,right=.98,top=.92,bottom=.12,hspace=.19)
        p=U/'visual'/(cid+'.png');fig.savefig(p,dpi=130);plt.close(fig)
        frames=[]
        for frame in item['frames']:
            i=int(np.flatnonzero(new['full_frame_id']==frame['frame_id'])[0]);d={'frame_id':frame['frame_id'],'time':float(ts[i]),'hands':{}}
            for side in ('L','R'):
                d['hands'][side]={'angles':int(new['full_'+side+'_angle_count'][i]),'position':bool(new['full_'+side+'_position_observed'][i]),'full_source_state':int(new['full_'+side+'_offline_full_view_state'][i]),'source_inside_view_missing':bool(new['full_'+side+'_inside_source_but_view_missing'][i])}
            frames.append(d)
        result.append({'clip_id':cid,'label':item['label'],'timeline':str(p),'previous_public_rgb_sheet':item['sheet'],'frames':frames})
    (U/'VISUAL.json').write_text(json.dumps(result,ensure_ascii=False,indent=2),'utf-8');print('6 timelines saved')
if __name__=='__main__':main()
