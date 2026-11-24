'use strict';
(() => {
  const $ = id => document.getElementById(id);
  const activeStates = new Set(['opening','countdown','live','replay']);
  const mediaStates = new Set(['opening','countdown','live','replay']);
  const names = {preparing:'분석기 준비 중',ready:'연습 준비 완료',opening:'카메라 연결 중',countdown:'곧 시작해요',live:'실시간 관측 중',replay:'공개 영상 확인 중',done:'이번 연습 종료',error:'확인 필요'};
  let token = '', meta = null, latest = null, busy = false, streamKey = '', pollTimer = 0, failures = 0, reconnectTimer = 0;
  let requestSerial=0,lastRenderedSerial=0,commandEpoch=0;
  const initialMessage = '참고 동작을 보고, 첫 번째 동작부터 따라 해 보세요.';
  const messagePresenter=new window.CoachMessagePresenter();
  const diagnosticPresenter=new window.DiagnosticPresenter();

  function setText(id, text) { const el=$(id); const value=String(text ?? ''); if(el.textContent !== value) el.textContent=value; }
  function error(message) { setText('error',message); $('error').hidden=!message; }
  async function request(path, body) {
    const headers={'X-Session-Token':token};
    const options={headers,cache:'no-store'};
    if(body!==undefined){options.method='POST';headers['Content-Type']='application/json';options.body=JSON.stringify(body);}
    const response=await fetch(path,options);
    const data=await response.json();
    if(!response.ok)throw new Error(data.error || '요청을 처리하지 못했어요.');
    return data;
  }

  function controls() {
    const state=latest?.state || (meta?.ready?'ready':'preparing');
    const active=activeStates.has(state);
    const canStart=!!token && !busy && !active && state!=='preparing';
    $('cameraStart').disabled=!canStart;
    $('publicStart').disabled=!canStart;
    $('reset').disabled=busy || !token || state==='preparing' || state==='opening';
    $('stop').disabled=busy || !active;
    $('cameraStart').textContent=active && latest?.source==='camera'?'카메라 사용 중':'카메라로 연습';
  }

  function renderChecks(checks) {
    const host=$('checks');
    const normalized=Array.isArray(checks)?checks:[];
    const signature=JSON.stringify(normalized.map(c=>[c.id,c.label,c.state]));
    if(host.dataset.signature===signature)return;
    host.dataset.signature=signature;host.replaceChildren();
    if(!normalized.length){const span=document.createElement('span');span.className='check';span.dataset.state='pending';span.textContent='관측을 시작하면 확인 항목이 나타나요';host.append(span);return;}
    const icons={ok:'✓',wrong:'!',unknown:'?',pending:'·'};
    const descriptions={ok:'항목 충족',wrong:'교정 필요',unknown:'관측 부족',pending:'확인 전'};
    for(const item of normalized){
      const state=Object.hasOwn(icons,item.state)?item.state:'unknown';
      const chip=document.createElement('span');chip.className='check';chip.dataset.state=state;chip.title=descriptions[state];
      const symbol=document.createElement('span');symbol.className='check-symbol';symbol.setAttribute('aria-hidden','true');symbol.textContent=icons[state];
      const label=document.createElement('span');label.textContent=item.label || item.id || descriptions[state];
      chip.append(symbol,label);chip.setAttribute('aria-label',`${label.textContent}: ${descriptions[state]}`);host.append(chip);
    }
  }

  function renderStages(observation, active) {
    const completed=Array.isArray(observation?.completed)?observation.completed:[];
    setText('progressCount',`${completed.slice(0,2).filter(v=>v===true).length} / 2 동작 확인`);
    for(let i=0;i<2;i++){
      const done=completed[i]===true;
      const current=!done && active && observation?.stage_index===i;
      const card=$(`step${i}`);card.dataset.state=done?'complete':current?'current':'pending';
      card.querySelector('.step-number').textContent=done?'✓':`0${i+1}`;
      setText(`stepStatus${i}`,done?'핵심 동작 확인':current?'지금 연습 중':'아직 확인 전');
      if(current)card.setAttribute('aria-current','step');else card.removeAttribute('aria-current');
    }
    $('stepConnector').style.width=completed[0]===true?'100%':'0%';
  }

  function renderActionModel(model,observation={}) {
    const states={pending:'확인 전',observed:'흐름 관측',uncertain:'아직 불확실',unavailable:'확인 못함'};
    const state=Object.hasOwn(states,model?.status)?model.status:'pending';
    const fallback={pending:'관측을 시작하면 움직임의 흐름을 함께 살펴봐요.',observed:'움직임의 흐름을 관측했어요.',uncertain:'움직임을 조금 더 살펴보고 있어요.',unavailable:'행동 모델의 관측을 받지 못했어요.'};
    $('actionObservation').dataset.state=state;
    setText('actionStatus',states[state]);
    const completed=window.completedActionObservation(model,observation);
    setText('actionMessage',completed?'팔 쓸기와 양손 위아래 움직임을 확인했어요.':model?.message || fallback[state]);
    const details=[];
    if(model?.role)details.push(model.role);
    if(model?.model_id)details.push(`모델: ${model.model_id}`);
    if(Number.isFinite(model?.observation_lag_ms))details.push(`행동 관측 시점 지연 약 ${Math.round(model.observation_lag_ms)} ms`);
    details.push('단어 인식 TCN은 이 화면에서 사용하지 않으며, 단계 확인이 수어 전체의 정답을 뜻하지 않습니다.');
    setText('actionModelDetail',details.join(' · '));
  }

  function renderDiagnostics(status,now) {
    const observation=status.observation || {};
    const presented=diagnosticPresenter.update({state:status.state,generation:status.generation,source:status.source,seq:status.seq,phase:observation.phase,diagnostic:observation.diagnostic},now);
    $('diagnosticHint').hidden=presented.hidden;
    $('diagnosticHint').dataset.tone=presented.tone;
    setText('diagnosticCategory',presented.label || '');
    setText('diagnosticHeld',presented.held?'진행 기록 잠시 유지':'');
    const summary=window.diagnosticSummary(status.diagnostics);
    setText('diagnosticTotals',summary.current);
    setText('diagnosticLastReason',summary.lastReason);
    setText('diagnosticPrevious',summary.previous);
    const host=$('diagnosticCounts'),signature=JSON.stringify(summary.rows);
    if(host.dataset.signature===signature)return;
    host.dataset.signature=signature;host.replaceChildren();
    for(const row of summary.rows){
      const item=document.createElement('div');item.className='diagnostic-count';
      const label=document.createElement('span');label.textContent=row.label;
      const value=document.createElement('span');value.textContent=`${row.count}회`;
      item.append(label,value);host.append(item);
    }
  }

  function connectStream(status) {
    if(!mediaStates.has(status.state))return;
    const key=String(status.generation ?? 'current');
    if(streamKey===key)return;
    streamKey=key;
    $('preview').hidden=true;$('cameraEmpty').hidden=false;
    $('preview').src=`/preview.mjpeg?token=${encodeURIComponent(token)}&generation=${encodeURIComponent(key)}`;
  }

  function clearPreview() {
    clearTimeout(reconnectTimer);streamKey='';
    $('preview').removeAttribute('src');$('preview').hidden=true;$('cameraEmpty').hidden=false;
    $('liveIndicator').hidden=true;
  }

  function acceptRender(status,serial,epoch) {
    if(epoch!==commandEpoch || serial<lastRenderedSerial)return;
    if(Number.isFinite(status.generation) && Number.isFinite(latest?.generation) && status.generation<latest.generation)return;
    lastRenderedSerial=serial;render(status);
  }

  function render(status) {
    latest=status;failures=0;
    const state=status.state || 'ready';const active=activeStates.has(state);const observation=status.observation || {};
    const isPublic=status.source==='public_replay';
    $('connection').dataset.state=state;setText('connectionText',names[state] || '관측 중');
    setText('liveTitle',isPublic?'공개 영상의 관측':'내 모습');
    setText('sourceBadge',isPublic?'공개 영상 재생':active?'카메라 연결':'카메라 꺼짐');
    $('sourceBadge').dataset.active=String(active);$('publicNotice').hidden=!isPublic;
    const overlay=state==='preparing'||state==='opening'||state==='countdown';
    $('videoOverlay').hidden=!overlay;$('spinner').hidden=state==='countdown';$('overlayNumber').hidden=state!=='countdown';
    setText('overlayNumber',Math.max(1,Number(status.countdown)||3));
    setText('overlayText',state==='preparing'?'분석기를 준비하고 있어요.':state==='opening'?'카메라를 연결하고 있어요.':'손을 내리고 편하게 준비해 주세요.');
    $('liveIndicator').hidden=!['live','replay'].includes(state);setText('liveIndicatorText',isPublic?'공개 영상 순차 입력':'실시간 관측');
    if(state==='ready')clearPreview();else connectStream(status);
    renderStages(observation,active || state==='done');renderChecks(observation.checks);renderActionModel(observation.action_model,observation);
    const now=performance.now();renderDiagnostics(status,now);
    const presented=messagePresenter.update({state,generation:status.generation,source:status.source,seq:status.seq,phase:observation.phase,message:observation.message || (state==='done'?'이번 관측이 끝났어요. 확인하지 못한 단계는 다시 연습해 주세요.':initialMessage),initialMessage},now);
    const phase=({waiting:'시작 자세 확인',ready:'연습 준비',opening:'카메라 연결 중',countdown:'곧 시작해요',error:'확인 필요',ended:'관측 종료',preparing:'시작 자세 준비',moving:'움직임 확인',sweeping:'첫 번째 동작 확인',repeat:'다음 움직임 확인',stage_complete:'핵심 동작 확인',correction:'동작 교정',complete:'확인한 동작 완료',unknown:'관측 확인 중',action_uncertain:'움직임 흐름 확인 중'}[presented.phase]) || '움직임 확인';
    setText('coachPhase',phase);
    setText('coachMessage',presented.message);
    document.querySelector('.coach').dataset.tone=window.coachTone(observation,state);
    setText('frameStatus',isPublic?'같은 공개 영상을 순서대로 관측합니다.':active?'손·팔을 보면서 천천히 따라 해 주세요.':'상반신과 양손이 보이도록 앉아 주세요.');
    const metrics=status.metrics || {};
    const latency=metrics.displayed_observation_age_ms ?? observation.observation_lag_ms;
    setText('liveMetric',status.source==='camera' && active && Number.isFinite(latency)?`관측 시점 지연 약 ${Math.round(latency)} ms`:'');
    error(status.error || '');
    controls();
  }

  async function loadMeta() {
    const response=await fetch('/api/meta',{cache:'no-store'});
    if(!response.ok)throw new Error('연습 정보를 불러오지 못했어요.');
    meta=await response.json();token=meta.token || '';
    if(!token)throw new Error('연습 연결을 준비하지 못했어요. 새로고침해 주세요.');
    const ref=meta.reference || '/media/reference.mp4';
    if($('reference').getAttribute('src')!==ref)$('reference').src=ref;
    for(let i=0;i<2;i++){
      const stage=meta.stages?.[i];if(!stage)continue;
      if(stage.label)setText(`stepLabel${i}`,stage.label);
      if(stage.guide)setText(`stepGuide${i}`,stage.guide);
    }
    if(meta.scope)setText('scope',meta.scope);
    if(meta.scope_detail)setText('scopeDetail',meta.scope_detail);
    controls();
  }

  async function poll() {
    clearTimeout(pollTimer);
    if(busy){pollTimer=setTimeout(poll,200);return;}
    const serial=++requestSerial,epoch=commandEpoch;
    try{
      if(!token)await loadMeta();
      const status=await request('/api/status');
      if(epoch!==commandEpoch)return;
      acceptRender(status,serial,epoch);
      if(status.state==='preparing' && meta?.ready===false && !busy){
        // State is authoritative; refresh lesson metadata after preparation finishes.
      }else if(meta?.ready===false){await loadMeta();}
    }catch(exc){
      if(epoch!==commandEpoch || serial<lastRenderedSerial)return;
      failures++;setText('connectionText',failures<3?'연결 확인 중':'서버 연결 확인 필요');
      messagePresenter.reset();
      diagnosticPresenter.reset();
      $('connection').dataset.state='error';
      $('liveIndicator').hidden=true;setText('liveMetric','');setText('frameStatus','새 관측을 받지 못하고 있어요.');
      setText('coachPhase','관측 연결 확인 중');
      setText('coachMessage','연결이 잠시 끊겼어요. 다시 연결된 뒤 움직임을 확인할게요.');
      $('diagnosticHint').hidden=false;$('diagnosticHint').dataset.tone='neutral';
      setText('diagnosticCategory','관측 연결 확인 중');setText('diagnosticHeld','');
      document.querySelector('.coach').dataset.tone='neutral';
      renderChecks((latest?.observation?.checks || []).map(c=>({...c,state:'unknown'})));
      renderActionModel({status:'unavailable',message:'연결이 돌아오면 움직임의 흐름을 다시 확인할게요.'});
      if(failures>=3)error('연습 서버와 연결되지 않아요. 서버가 실행 중인지 확인하고 새로고침해 주세요.');
    }finally{pollTimer=setTimeout(poll,failures?1000:200);}
  }

  async function command(path,body={}) {
    if(busy)return;busy=true;const epoch=++commandEpoch;controls();error('');
    try{
      const result=await request(path,body);
      if(path==='/api/start'){
        streamKey='';$('preview').hidden=true;$('cameraEmpty').hidden=false;
        setText('connectionText',body.mode==='camera'?'카메라 연결 중':'공개 영상 준비 중');
      }
      if(path==='/api/stop'){clearPreview();renderChecks([]);}
      const serial=++requestSerial;
      if(result.state)acceptRender(result,serial,epoch);
      else acceptRender(await request('/api/status'),serial,epoch);
    }catch(exc){error(exc.message || '요청을 처리하지 못했어요.');}
    finally{busy=false;controls();}
  }

  $('cameraStart').addEventListener('click',()=>command('/api/start',{mode:'camera'}));
  $('publicStart').addEventListener('click',()=>command('/api/start',{mode:'public_replay'}));
  $('reset').addEventListener('click',()=>command('/api/reset'));
  $('stop').addEventListener('click',()=>command('/api/stop'));
  $('preview').addEventListener('load',()=>{$('preview').hidden=false;$('cameraEmpty').hidden=true;});
  $('preview').addEventListener('error',()=>{
    if(!mediaStates.has(latest?.state))return;
    clearTimeout(reconnectTimer);reconnectTimer=setTimeout(()=>{
      if(mediaStates.has(latest?.state)){
        streamKey='';connectStream(latest);
      }
    },900);
  });
  $('reference').addEventListener('error',()=>{$('referenceError').hidden=false;});
  $('reference').addEventListener('loadeddata',()=>{$('referenceError').hidden=true;});
  $('referenceRestart').addEventListener('click',()=>{$('reference').currentTime=0;$('reference').play().catch(()=>{});});
  for(const button of document.querySelectorAll('.stage-watch'))button.addEventListener('click',()=>{
    const i=Number(button.dataset.stage), stage=meta?.stages?.[i];
    const start=stage?.start_seconds ?? stage?.start ?? 0;
    if(Number.isFinite(start))$('reference').currentTime=start;
    $('reference').play().catch(()=>{});setText('referenceHint',`${i+1}번째 참고 동작을 살펴보세요`);
  });
  document.addEventListener('visibilitychange',()=>{if(!document.hidden)poll();});
  window.addEventListener('pagehide',()=>{
    // Only release an already-opened camera. Page load never starts a device.
    if(!token || latest?.source!=='camera' || !activeStates.has(latest.state))return;
    fetch('/api/stop',{method:'POST',headers:{'X-Session-Token':token,'Content-Type':'application/json'},body:'{}',keepalive:true}).catch(()=>{});
  });
  poll();
})();
