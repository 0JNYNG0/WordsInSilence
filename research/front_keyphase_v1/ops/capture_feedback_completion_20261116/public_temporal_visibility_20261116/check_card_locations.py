from check_integrated import *
def main():
 m=load('temporal_cards_fixture',E/'check_cards.py');m.FeedbackConsumer=FeedbackConsumer;m.E=J;m.main()
 net=FeedbackConsumer();assert net.temporal_admitted
 try:
  r=next(r for r in web.read(web.H/'public_views/INPUTS.json')['rows']if r['id']=='NIA_SL_WORD1290_REAL13_L');q=net.raw(dict(web.np.load(r['raw'])),'WORD1290');c=next(c for c in q['lesson_acceptance']['criteria']if c['id']=='left_shape');card=next(c for c in q['cards']if c['id']=='left_shape');passed=card['status']=='unknown'and card['display_status']=='uncertain'and card['query_spans']==c['uncertain_spans']and not card['difference_spans'];web.write(J/'web/UNCERTAIN_LOCATION_CHECK.json',dict(passed=passed,criterion=c,card=card,private_data=False));assert passed
 finally:web.close(net)
if __name__=='__main__':main()
