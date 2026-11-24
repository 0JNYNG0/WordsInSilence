"""Evidence-linked public observation preview, never a whole-word pass.

Description requirements are structured source data, not word-name branches.
An observed count is not proof of physical contact or adequate handshape.
"""
from prepare import U,read,write

def repeated_relation(requirement,events,frame_count,observed_frames):
 n=len(events);required=requirement['count']
 if not observed_frames:
  return dict(status='unknown',message='양손의 움직임을 확인하기 어려워요.',action='두 손이 보이도록 다시 보여주세요.',source_spans=[])
 spans=[[int(e['start']),int(e['end'])]for e in events]
 assert all(0<=a<=b<frame_count for a,b in spans)
 if n<required:
  return dict(status='partial_observation',message=f'{requirement["toward_ko"]} {n}번 가까워진 움직임은 확인했지만, 나머지는 확실히 판단하지 못했어요.',
   action=f'두 손이 잘 보이게, 예시처럼 {required}번 이어서 보여주세요.',source_spans=spans,
   caution='관측된 횟수이며, 보이지 않은 움직임의 생략을 확정하지 않습니다.')
 return dict(status='observed_requirement_only',message=f'{requirement["toward_ko"]} {n}번 가까워진 움직임을 확인했어요.',
  action='손모양과 닿는 방향은 별도로 확인해야 해요.',source_spans=spans)

def main():
 V=U/'source_corrected';j=read(V/'CONFIRMATION_RESULTS.json');rows=[]
 req=dict(id='repeated_hand_approach',source_phrase='왼 손등에 두 번 댄다',toward_ko='왼손등 쪽으로',count=2,
  limitation='2D near is not physical touch; palm/back orientation is not fully qualified.')
 import numpy as np
 for r in j['rows']:
  p=np.load(V/'confirmation'/f'{r["clip_id"]}_{r["kind"]}.npz');feedback=repeated_relation(req,r['approaches'],len(p['states']),r['observed_frames'])
  rows.append(dict(clip_id=r['clip_id'],kind=r['kind'],requirement=req,feedback=feedback,whole_word_correct=None,quest_advance=False,qualified_for_live=False))
 assert all(not r['quest_advance']for r in rows)
 write(U/'BASIC_FEEDBACK_PUBLIC.json',dict(rows=rows,scope='Uses actual source-trained relation predictions and native evidence spans. Text templates render observations; no natural-language model or automatic full-guide understanding.',web_connected=False))
 print('PUBLIC EVIDENCE-LINKED PREVIEW',len(rows))
if __name__=='__main__':main()
