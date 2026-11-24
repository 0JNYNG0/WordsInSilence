from common import *
def main():
 assert not (U/'FROZEN.json').exists()
 a=read(U/'SOURCE_AUDIT.json');assert a['status']=='PASS'
 old=read(O/'word100_training_20261013/FROZEN_EXPERIMENT.json')
 paths=[O/'word100_body_boundary_20261017/training_C_20261013_best.pt',O/'word100_frozen_c_temporal_20261019/T3F_20261013_best.pt',
        O/'word100_body_boundary_20261017/training_C_20261013_RESULT.json',O/'word100_frozen_c_temporal_20261019/T3F_20261013_RESULT.json']
 for s in [42,43,44]:paths.append(O/f'word100_learned_guide_feedback_20261102/runs/projection_finetune_{s}/model.pt')
 write(U/'FROZEN.json',dict(status='FROZEN_BEFORE_RTM_CHANNELS_AND_TRAINING',date_kst='2026-11-08',
  source_manifest_sha256=sha(U/'SOURCES.json'),source_audit_sha256=sha(U/'SOURCE_AUDIT.json'),pins={str(p):sha(p) for p in paths},
  classes=old['classes'],labels=old['labels'],
  extraction=dict(model='RTMW-l 384x288+YOLOX-m',runtime='same pinned OpenVINO GPU Engine',all_original_frames=True,interpolation=False,
    score='raw SimCC, not visibility probability',available='finite and inside actual image; never evidence that hidden fingers were seen'),
  input=dict(schema='rtmw133_shared_2d_full_timeline_v1',tcn_channels=280,
    views=['full','upper_mask'],upper_mask='Same saved public rest-wrist boundary as historical masked condition, applied AFTER full-image inference; no trim of timeline',
    runtime='full actual camera image; no mandatory rest-wrist crop',
    missing='Coordinates and local features zero only when geometrically unavailable; mask retained; unknown supervised targets loss-masked',
    training_view='alternate full and upper_mask by deterministic clip parity and epoch, no target core trim'),
  tcn=dict(conditions=['fresh','finetune_C'],seeds=[20261013,20261014,20261015],epochs=80,batch_size=32,
    model=old['model'],lr={'fresh':.001,'finetune_C':.0003},weight_decay=.001,gradient_clip=1.,
    checkpoint='final epoch; no development checkpoint/seed/rule selection',target_word_input=False,
    comparison='same 300 old DEV; report full and upper-mask separately; old C/T3-F masked+sensor cached metrics are not a same-detector accuracy truth'),
  feedback=dict(words=['WORD0739','WORD1133','WORD1567','WORD2523'],conditions=['pose_fresh','pose_transfer'],seeds=[42,43,44],epochs=120,batch_size=6,
    lr=.001,weight_decay=.01,gradient_clip=2.,input='280 shared pose channels; no RGB model',
    model='projection280to96; Conv1d96to64 k3; Conv1d64to64 k3 dilation2; bidirectionalGRU32; 7 attributes; description-linked stages and learned train-only prototype',
    transfer='Copy shape-compatible temporal, GRU, attribute, guide, stage, prototype weights from existing same-seed guide model; new RTMW input projection trained. No claim of SignRep backbone fine-tuning.',
    losses={'reviewed_centers':1.,'stage':.35,'fit_only_dtw_prototype':.1},stage_weights={'canonical_visual':1.,'dtw_weak':.2},
    target_word_input='only masks stage candidates and chooses learned prototype; never changes observed attribute probabilities',
    checkpoint='final epoch; report all seeds; no threshold tuning',primary='7 known physical attributes; macro balanced accuracy where both classes; show per-target support',
    stage_test='prior saved visual anchors; no scoring against weak transferred labels as truth',
    controls='time speed variants, 1-2 frame missingness, longer missingness, reversed/cut main stage; engineering controls not real learner error gold'),
  timing=dict(cpu_threads=4,training='CPU PyTorch; Intel OpenVINO GPU for extraction, not CUDA training',
    reported_separately=['cold initialization','full source extraction','per-frame streaming compute','input conversion','TCN forward','feedback forward','DTW','post-capture remaining work'],
    speed_samples='first 3 listed feedback development clips, fixed before scores; warm repeated pure heads; no fresh full-video detection for speed'),
  adoption='Research candidates only. No production default replacement. Require no recognition/attribute/stage/unknown regression and real stream integration separately.',
  forbidden=['private or mentor data','fresh199 evaluation reuse','new downloads','external upload','camera','Git','original model writes'],
  historical_exposure='All DEV previously used; not fresh independent evaluation. ID16 quarantined elsewhere remains excluded.'))
 event('frozen',source_frames=a['source_frames_header'])
if __name__=='__main__':main()
