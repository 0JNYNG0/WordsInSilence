from check_integrated import *
def main():
 net=FeedbackConsumer();assert net.arrival_admitted
 try:
  r=next(r for r in web.read(web.H/'public_views/INPUTS.json')['rows']if r['id']=='NIA_SL_WORD1290_REAL11_L');q=net.raw(dict(web.np.load(r['raw'])),'WORD1290');card=next(c for c in q['cards']if c['id']=='repeat_order');passed=len(q['lesson_acceptance']['events'])==1 and not q['overall_pass']and card['status']=='unknown';web.write(J/'web/PARTIAL_CARD_CHECK.json',dict(passed=passed,events=1,overall_pass=q['overall_pass'],card=card,coaching=q['coaching'],module=q.get('arrival_module'),compute_seconds=q['compute_seconds'],private_data=False));print('PARTIAL_CARD',passed,card,flush=True);assert passed
 finally:web.close(net)
if __name__=='__main__':main()
