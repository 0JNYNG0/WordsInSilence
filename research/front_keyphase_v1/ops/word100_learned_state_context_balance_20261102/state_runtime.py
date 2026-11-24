"""Attach validated native-time evidence, preserve independent observers/DTW."""
import copy,time
from state_model import *
from sequence_full import observe

class StateObserver:
    def __init__(self):
        assert read(T/'SAVED_VERIFICATION.json')['status']=='PASS_TO_RUNTIME'
        self.model=StateNet(True);p=T/'runs/anchored_context_42/model.pt';assert sha(p)==read(p.parent/'COMPLETE.json')['sha256']
        self.model.load_state_dict(torch.load(p,map_location='cpu',weights_only=True));self.model.eval();self.model.requires_grad_(False)
        self.graph=read(S/'LEARNED_STATE_GRAPH.json')['graphs'];self.groups=read(S/'PHYSICAL_STATE_TARGETS.json')['groups'];self.guides=read(old.P/'GUIDE_SUPPORT.json')['guides']
    @torch.inference_mode()
    def attach(self,result,a):
        out=copy.deepcopy(result);word=result['word_id'];n=len(a['latent']);start=time.perf_counter()
        x,l,w,_=batch([dict(x=a['latent'],y=np.full(n,-1,np.int64),word_id=word)])
        logits,base,allowed=self.model(x,l,w,True);pr=logits.softmax(-1)[0].numpy()
        ev=observe(pr,np.asarray(a.get('image_available',np.ones(n,bool))),self.graph[word],a['timestamp_us']/1e6)
        for s in ev['states']:
            g=self.groups[s['group']];s['description']=g['description_ko'];guides=[self.guides[k] for k in g['old_stage_aliases']]
            s['guide_ids']=[z['canonical_stage_id'] for z in guides];s['source_phrases']=sorted({c['text'] for z in guides for c in z.get('source_cards',[])})
        ev['head_seconds']=time.perf_counter()-start;ev['context_used_frames']=np.flatnonzero(allowed[0].numpy()).tolist()
        ev['current_latent_contains_prior_context']=True;ev['goal_conditioned_phase_head']=True;ev['independent_handshape_relation_motion_preserved']=True
        ev['limitations']=['Nine broad physical states in four expressions, plus sparse rest supervision.','Not a full linguistic correctness or mandatory-repeat verdict.','Confident current errors can remain; confidence is not independently calibrated human truth.','Source descriptions remain linked evidence; no unrestricted language understanding claim.']
        out['learned_state_sequence']=ev;out['whole_sign_correctness']=None
        assert out['stages']==result['stages'] and out['path']==result['path'] and out['motion_observations']==result['motion_observations']
        return out

def main():
    assert not (T/'RUNTIME_RESULTS.json').exists();observer=StateObserver();torch.set_num_threads(2)
    rows=[r for r in read(old.P/'INPUT_MANIFEST.json')['rows'] if r['person']==12];assert len(rows)==4
    exclusive(T/'RUNTIME_INTENT.json',dict(public_sources=[dict(clip_id=r['clip_id'],source=r['source'],sha256=r['source_sha256']) for r in rows],existing_demo_not_replaced=True))
    sys.path.insert(0,str(old.G));import learner_adapter
    learner_adapter.U=T;learner_adapter.warmup();done=[]
    for r in rows:
        assert sha(r['source'])==r['source_sha256'];d=T/'public_runtime'/r['clip_id'];start=time.perf_counter()
        base=learner_adapter.analyze(r['source'],d,r['word_id']);out=observer.attach(base,dict(np.load(d/'analysis.npz')))
        out['latency']['total_analysis_seconds']=time.perf_counter()-start;exclusive(d/'SEQUENCE_FEEDBACK.json',out)
        assert out['runtime_individual_reference_video_comparisons']==0 and out['learned_prototype_comparisons']==1
        done.append(dict(clip_id=r['clip_id'],path=str(d/'SEQUENCE_FEEDBACK.json'),seconds=out['latency']['total_analysis_seconds'],status=out['learned_state_sequence']['status']))
        print(done[-1],flush=True)
    exclusive(T/'RUNTIME_RESULTS.json',dict(rows=done,full_runtime_median_seconds=float(np.median([r['seconds'] for r in done])),public_only=True))

if __name__=='__main__':main()
