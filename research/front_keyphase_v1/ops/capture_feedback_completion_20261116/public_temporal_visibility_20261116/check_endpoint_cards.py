from check_integrated import *
def main():
 net=FeedbackConsumer();assert net.temporal_admitted;rows=[]
 try:
  r=next(r for r in web.read(web.H/'public_views/INPUTS.json')['rows']if r['id']=='NIA_SL_WORD1290_REAL11_L');z=dict(web.np.load(r['raw']))
  for kind in ['full','first_only']:
   raw={k:v.copy()for k,v in z.items()}
   if kind=='first_only':
    for k in ['xy_px','score','timestamp_us','frame_id']:raw[k]=raw[k][:60]
   q=net.raw(raw,'WORD1290');c=next(c for c in q['cards']if c['id']=='repeat_order');ev=len(q['lesson_acceptance']['events']);passed=ev==2 and q['overall_pass']and c['status']=='supported'if kind=='full'else ev==1 and not q['overall_pass']and c['status']=='unknown';rows.append(dict(kind=kind,events=ev,overall_pass=q['overall_pass'],card=c,passed=passed,seconds=q['compute_seconds']))
  web.write(J/'web/ENDPOINT_CARD_CHECK.json',dict(rows=rows,passed=all(r['passed']for r in rows),private_data=False));print(rows,flush=True);assert all(r['passed']for r in rows)
 finally:web.close(net)
if __name__=='__main__':main()
