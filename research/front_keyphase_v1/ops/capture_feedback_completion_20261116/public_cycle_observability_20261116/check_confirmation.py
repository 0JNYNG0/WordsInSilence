from verify_scope import *
def main():
 torch.set_num_threads(2);out=M/'confirmed';out.mkdir(exist_ok=True)
 write(out/'SOURCE_REVIEW.json',dict(reviewer='Codex native public RGB sheet, inspected before output',source='confirmation/*08_[LR].jpg',observations='Frames48 and58 show separated active upright hand;53 and68 show approach to supporting palm. Supporting fingers are visible sideways in R, foreshortened in L. Two approach cycles visible; no claim of physical contact or individually hidden finger state.',scope='Previously unselected camera views of FIT person8, not a new-person accuracy estimate.',training=False))
 old,new=TemporalVisibilityLesson(),CycleLesson();rows=[]
 for r in read(M/'confirmation/INPUTS.json')['rows']:
  raw=dict(np.load(r['raw']));a,b=old.analyze(raw),new.analyze(raw);row=dict(id=r['id'],before={c['id']:c['status']for c in a['criteria']},after={c['id']:c['status']for c in b['criteria']},events_before=len(a['events']),events_after=len(b['events']),**verify(a,b));rows.append(row);write(out/(r['id']+'.json'),dict(before=a,after=b));print(row,flush=True)
 write(out/'RESULT.json',dict(rows=rows,passed=all(r['passed']for r in rows),private_data=False))
if __name__=='__main__':main()
