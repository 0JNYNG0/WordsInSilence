from observed_gap import *
def main():
 torch.set_num_threads(2);out=V/'web';assert not(out/'DISPLAY_PATTERN.json').exists()
 r=next(r for r in read(B/'INPUTS.json')['rows']if r['person']==1);net=SparseVisible();q=net.analyze(dict(np.load(r['raw'])))
 write(out/'DISPLAY_PATTERN.json',dict(clip_id=r['clip_id'],events=q['events'],purpose='Display crop for an observed partial movement; never a missing-action completion label.'))
 print(q['events'])
if __name__=='__main__':main()
