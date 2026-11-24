from common import *

def main():
    units=[('hand_articulation_learning',49809),('splice_rank_learning',9191)]
    jobs=[]
    for name,session in units:
        p=HERE/name
        if (p/'TRAINING_COMPLETE.json').exists():continue
        jobs.append(dict(unit=name,exec_session=session,progress=read(p/'TRAINING_PROGRESS.json') if (p/'TRAINING_PROGRESS.json').exists() else None))
    e=read(GRAPH/'CURRENT_EXECUTION.json');e.update(at=stamp(),jobs=jobs,training_running=bool(jobs),process_running=bool(jobs),session_id=jobs[0]['exec_session'] if jobs else None,other_session_id=jobs[1]['exec_session'] if len(jobs)>1 else None,current_command='A articulation and J splice-rank fixed training; completed runs immutable');write(GRAPH/'CURRENT_EXECUTION.json',e)
    s=read(OPS/'STATE.json');s['updated_at']=stamp();s['last_updated']=stamp();s['word100_web_completion_graph_20261111']['training_running']=bool(jobs);write(OPS/'STATE.json',s)
    print([(j['unit'],j['exec_session'])for j in jobs])

if __name__=='__main__':main()
