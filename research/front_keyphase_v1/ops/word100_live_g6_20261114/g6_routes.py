"""Read-only public connection test examples, no arbitrary file access."""
from pathlib import Path
import json,copy
U=Path(__file__).resolve().parent
CASES={
 'normal':('live_inputs/NIA_SL_WORD1133_REAL17_F.json','잘하다 · 정상 공개 동작'),
 'reverse':('controls/NIA_SL_WORD1133_REAL18_F__continuous_full_reverse.json','잘하다 · 역순 편집 시험'),
 'conflict':('controls/NIA_SL_WORD1133_REAL17_F__continuous_full_reverse.json','잘하다 · 판단 충돌'),
 'partial':('controls/NIA_SL_WORD0739_REAL17_F__prefix_before_goal.json','순종 · 중간 종료 시험'),
 'missing':('controls/NIA_SL_WORD0739_REAL17_F__all_hands_missing.json','순종 · 손 관측 누락 시험')}
def catalog():return [dict(label=label,url='/g6-public/'+name+'.json') for name,(_,label) in CASES.items()]
def handle_get(handler,route):
 if not route.startswith('/g6-public/'):return False
 name=route.removeprefix('/g6-public/').removesuffix('.json')
 if name not in CASES or not route.endswith('.json'):handler.send_error(404);return True
 data=json.loads((U/CASES[name][0]).read_text('utf-8-sig'))['result']
 if name=='normal':data['video']='/feedback-g5/media/NIA_SL_WORD1133_REAL17_F.mp4'
 data['public_example_name']=CASES[name][1];data['training_use']=False
 handler.reply(data);return True
