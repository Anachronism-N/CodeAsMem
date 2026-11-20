/* Local, buildless UI. No external assets, analytics, or network dependencies. */
'use strict';
const $ = s => document.querySelector(s);
const $$ = s => [...document.querySelectorAll(s)];
const esc = s => String(s ?? '').replace(/[&<>"']/g, c => ({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));
const fmt = n => Number(n).toLocaleString('en-US');
const colors = ['#58dfbc','#78aaff','#f2c079','#d58ff2','#f38d92','#7ee0f1'];
const reviewLabels = {pending:'待视觉复核',candidate:'人工标记 · 候选',repair:'需要修复'};
const gameShort = {cp2077:'CP2077',gta5:'GTA V',rdr2:'RDR2'};
const state = {cases:[], filter:'all', search:'', current:null, detail:null, tracks:null, timing:null,
  frame:0, playing:false, channel:'control', bufferKind:'depth', bufferScope:'full', bufferFrame:0,
  token:0, bufferTimer:null, lastBufferUpdate:0, bufferRequest:0};

async function api(path, options) {
  const r = await fetch(path, options);
  if (!r.ok) {
    let msg = `请求失败 (${r.status})`;
    try { const v = await r.json(); msg = typeof v.detail === 'string' ? v.detail : msg; } catch (_) {}
    throw new Error(msg);
  }
  return r.json();
}
let toastTimer;
function toast(message) {
  $('#toast').textContent = message;
  $('#toast').classList.add('show');
  clearTimeout(toastTimer);
  toastTimer = setTimeout(() => $('#toast').classList.remove('show'), 5000);
}
function base() { return `/api/cases/${encodeURIComponent(state.current.game)}/${encodeURIComponent(state.current.case_id)}`; }
function mediaUrl(channel) { return `/media/${encodeURIComponent(state.current.game)}/${encodeURIComponent(state.current.case_id)}/${channel}`; }
function videos() { return [$('#video-rgb'),$('#video-white'),$('#video-proxy')]; }

function setPage(page, updateHash=true) {
  if (!['browse','method','training','quality'].includes(page)) page='browse';
  $$('.nav').forEach(el => el.classList.toggle('active',el.dataset.page===page));
  $$('.page').forEach(el => el.classList.toggle('active',el.id===`page-${page}`));
  if (updateHash) history.replaceState(null,'',`#${page}`);
  if (page !== 'browse') pause();
  if (page === 'method') loadCode();
  if (page === 'browse') requestAnimationFrame(drawTrajectory);
}

function renderList() {
  const rows = state.cases.filter(c => (state.filter==='all'||state.filter===c.game) &&
    `${c.game_name} ${c.case_id} ${c.site}`.toLowerCase().includes(state.search.toLowerCase()));
  $('#case-count').textContent = rows.length;
  $('#case-list').innerHTML = rows.length ? rows.map(c => {
    const review = c.review?.status || 'pending';
    return `<button class="case-item ${state.current?.case_id===c.case_id&&state.current?.game===c.game?'active':''}" data-game="${esc(c.game)}" data-id="${esc(c.case_id)}"><span class="item-meta"><span>${gameShort[c.game]}</span><span>${esc(c.site)}</span></span><span class="item-name mono">${esc(c.case_id)}</span><span class="item-bottom"><span>${c.frames} 帧 · ${c.duration_s.toFixed(1)} s</span><span class="review-dot ${review}">● ${review==='candidate'?'候选':review==='repair'?'需修复':'待检查'}</span></span></button>`;
  }).join('') : '<p class="empty">没有匹配的案例。</p>';
}

async function selectCase(game, id) {
  const row = state.cases.find(c => c.game===game && c.case_id===id);
  if (!row) return;
  const token = ++state.token;
  pause();
  ++state.bufferRequest;
  state.current=row; state.frame=0; state.timing=null; state.tracks=null;
  $('#play').disabled=true; $('#export-recipe').disabled=true;
  $('#case-game').textContent=row.game_name;
  $('#case-title').textContent=row.case_id;
  $('#case-subtitle').textContent=`${row.site} · ${row.frames} 帧 · ${row.duration_s.toFixed(3)} s · 主视角`;
  $('#sync-status').textContent='正在读取源视频时间与轨迹…';
  $('#frame-slider').max=row.frames-1; $('#frame-slider').value=0;
  $('#frame-label').textContent=`F 000 / ${row.frames-1}`;
  $('#time-label').textContent='0.000 s';
  $('#segment-bar').innerHTML='';
  $('#buffer-image').removeAttribute('src');
  $('#case-info').innerHTML='';
  $('#entity-legend').innerHTML='';
  drawTrajectory(); renderList();
  videos().forEach((v,i) => {
    delete v._pendingTime;
    v.removeAttribute('poster');
    v.src=mediaUrl(i===0?'rgb':i===1?'white':state.channel);
    v.load();
  });
  try {
    const [detail,timing,tracks] = await Promise.all([
      api(base()),api(base()+'/timing'),api(base()+'/tracks')]);
    if (token !== state.token) return;
    state.detail=detail; state.timing=timing; state.tracks=tracks;
    $('#play').disabled=false; $('#export-recipe').disabled=false;
    renderDetail(); renderSegments(); populateRecipe();
    $('#sync-status').textContent='源 PTS ↔ 代理帧编号';
    seekFrame(0);
  } catch (e) {
    if (token !== state.token) return;
    $('#sync-status').textContent='加载失败';
    $('#case-warning').textContent=e.message;
    toast(e.message);
  }
}

function renderDetail() {
  const d=state.detail, t=state.tracks;
  const warnings={cp2077:'当前包缺少共享静态场景和原生深度；白模以目标骨骼与近邻实体边界为主。',gta5:'静态代理来自粗略碰撞面与边界；control 动态层忽略静态遮挡，需结合 RGB 复核。',rdr2:'源交付 training_eligible=false；静态几何仍为粗略布局，进入训练前需处理源质量原因。'};
  $('#case-warning').textContent=warnings[d.game];
  const r=d.review?.status||'pending';
  $('#case-review-badge').textContent=reviewLabels[r];
  $('#case-review-badge').className=`badge ${r==='candidate'?'green':'amber'}`;
  $('#review-status').value=r; $('#review-note').value=d.review?.note||'';
  $('#review-result').textContent=d.review?.updated_at?'已有本地记录':'';
  const kind=t.tracking_kind.includes('proxy')?'骨骼代理点':'源骨骼点';
  const info=[['结构检查',d.structure_passed?'通过；视觉复核另行记录':d.structurally_validated?'存在问题':'尚未运行结构校验'],
    ['静态三角形',fmt(d.triangles)],['统一关节轨迹',`${t.entities.length} 个主体 × 20 个可选关节`],
    ['点轨迹张量',`${t.point_shape.join(' × ')} · ${kind}`],
    ['仅轨迹实体',`${d.track_only} 个；不虚构完整状态`],
    ['相机来源',d.episode.views.main.camera_source],
    ['源训练资格',d.source_training_eligibility===false?'false / 待解决源质量原因':d.source_training_eligibility===true?'true / 仍需对齐复核':'未提供 / 待复核']];
  $('#case-info').innerHTML=info.map(([k,v])=>`<dt>${esc(k)}</dt><dd>${esc(v)}</dd>`).join('');
  const path=`/download/${encodeURIComponent(d.game)}/${encodeURIComponent(d.case_id)}/`;
  $('#download-episode').href=path+'episode.json';
  $('#case-downloads').innerHTML=['qa.json','program.json','joints.npz','points.npz','timeline.npz'].map(n=>`<a href="${path+n}">${n} ↓</a>`).join('');
  $('#entity-count').textContent=`${t.entities.length} 个主体`;
  $('#entity-legend').innerHTML=t.entities.map((e,i)=>`<span><i style="background:${colors[i%colors.length]}"></i>${esc(e.source_id)} · ID ${e.instance_id}${e.is_focus?' · focus':''}</span>`).join('');
}

function renderSegments() {
  const n=state.current.frames;
  $('#segment-bar').innerHTML=state.tracks.segments.map(s=>`<span class="${s.state}" style="flex:${s.end_frame-s.start_frame+1}" title="${esc(s.state)} · F${s.start_frame}–${s.end_frame}"></span>`).join('');
  $('#segment-source').textContent=state.tracks.segment_provenance==='source_capture_frustum_audit'?'采集端视锥记录 · 非遮挡标签':'关节视锥估计 · 非遮挡标签';
}

function frameAtTime(time) {
  const pts=state.timing.source_pts_s;
  let lo=0,hi=pts.length;
  while(lo<hi) {const m=(lo+hi)>>1;if(pts[m]<=time+0.00005)lo=m+1;else hi=m;}
  return Math.min(pts.length-1,Math.max(0,lo-1));
}
function seekVideo(video, time) {
  if(video.readyState<1) { video._pendingTime=time; return; }
  const duration=video.duration;
  // Some captures have a final frame lasting less than 1 ms. A 2 ms end
  // margin would incorrectly clamp that frame into its predecessor.
  const t=Math.max(0,Number.isFinite(duration)?Math.min(time,Math.max(0,duration-0.000001)):time);
  if(Math.abs(video.currentTime-t)>.0002) video.currentTime=t;
}
function sourceTime(frame) {
  const p=state.timing.source_pts_s;
  const duration=$('#video-rgb').duration;
  const gap=frame+1<p.length?p[frame+1]-p[frame]:(Number.isFinite(duration)?Math.max(0,duration-p[frame]):1/state.timing.proxy_fps);
  const desired=p[frame]+Math.max(.000001,gap*.15);
  return Number.isFinite(duration)?Math.min(desired,duration-.000001):desired;
}
function seekFrame(frame) {
  if(!state.timing)return;
  frame=Math.min(state.current.frames-1,Math.max(0,Math.round(frame)));
  const proxy=(frame+.15)/state.timing.proxy_fps;
  seekVideo($('#video-rgb'),sourceTime(frame));
  seekVideo($('#video-white'),proxy); seekVideo($('#video-proxy'),proxy);
  updateFrame(frame,true);
}
function updateFrame(frame,force=false) {
  if(!state.timing)return;
  if(frame===state.frame&&!force)return;
  state.frame=frame;
  $('#frame-slider').value=frame;
  $('#frame-label').textContent=`F ${String(frame).padStart(3,'0')} / ${state.current.frames-1}`;
  $('#time-label').textContent=`${(state.timing.time_us[frame]/1e6).toFixed(3)} s`;
  drawTrajectory();
  const now=performance.now();
  if(!state.playing || now-state.lastBufferUpdate>800) {
    clearTimeout(state.bufferTimer);
    state.bufferTimer=setTimeout(updateBuffer,state.playing?40:130);
    state.lastBufferUpdate=now;
  }
}
function pause() {
  state.playing=false;
  videos().forEach(v=>v.pause());
  $('#play').textContent='▶ 播放';
}
async function togglePlay() {
  if(state.playing){pause();seekFrame(state.frame);return;}
  if(!state.timing)return;
  if(state.frame>=state.current.frames-1)seekFrame(0);
  state.playing=true;$('#play').textContent='Ⅱ 暂停';
  try {await Promise.all(videos().map(v=>v.play()));}
  catch(e){pause();toast('视频播放未开始，请再次点击播放。');}
}
function videoTick(now, metadata) {
  const rgb=$('#video-rgb');
  if(state.playing&&state.timing){
    const frame=frameAtTime(metadata?.mediaTime??rgb.currentTime);
    updateFrame(frame);
    const desired=(frame+.15)/state.timing.proxy_fps;
    for(const v of videos().slice(1))if(!v.seeking&&Math.abs(v.currentTime-desired)>1.4/state.timing.proxy_fps)seekVideo(v,desired);
  }
  if(rgb.requestVideoFrameCallback)rgb.requestVideoFrameCallback(videoTick);
}

function drawTrajectory() {
  const canvas=$('#trajectory'),rect=canvas.getBoundingClientRect();
  if(!rect.width)return;
  const ratio=window.devicePixelRatio||1;
  canvas.width=Math.round(rect.width*ratio);canvas.height=Math.round(rect.height*ratio);
  const ctx=canvas.getContext('2d');ctx.scale(ratio,ratio);
  const w=rect.width,h=rect.height;
  ctx.fillStyle='#101a27';ctx.fillRect(0,0,w,h);
  const entities=state.tracks?.entities;
  if(!entities){ctx.fillStyle='#8396ac';ctx.font='12px Segoe UI';ctx.fillText('读取轨迹后显示',20,36);return;}
  const coords=[];
  entities.forEach(e=>e.roots.forEach((p,i)=>{if(e.root_valid[i]&&p[0]!==null)coords.push(p);}));
  if(!coords.length)return;
  let xmin=Math.min(...coords.map(p=>p[0])),xmax=Math.max(...coords.map(p=>p[0]));
  let ymin=Math.min(...coords.map(p=>p[1])),ymax=Math.max(...coords.map(p=>p[1]));
  const padding=36,scale=Math.min((w-padding*2)/Math.max(xmax-xmin,1),(h-padding*2)/Math.max(ymax-ymin,1));
  const cx=(xmin+xmax)/2,cy=(ymin+ymax)/2;
  const xy=p=>[w/2+(p[0]-cx)*scale,h/2-(p[1]-cy)*scale];
  const gridStep=Math.pow(10,Math.floor(Math.log10(Math.max(xmax-xmin,ymax-ymin,1)/4)));
  const spacing=gridStep*scale<30?gridStep*5:gridStep;
  ctx.lineWidth=1;ctx.strokeStyle='#223044';ctx.fillStyle='#6e8299';ctx.font='9px Consolas';
  for(let x=Math.floor((cx-w/2/scale)/spacing)*spacing;x<cx+w/2/scale;x+=spacing){const sx=xy([x,cy])[0];ctx.beginPath();ctx.moveTo(sx,10);ctx.lineTo(sx,h-22);ctx.stroke();ctx.fillText(x.toFixed(1),sx+3,h-9);}
  for(let y=Math.floor((cy-h/2/scale)/spacing)*spacing;y<cy+h/2/scale;y+=spacing){const sy=xy([cx,y])[1];ctx.beginPath();ctx.moveTo(28,sy);ctx.lineTo(w-10,sy);ctx.stroke();if(sy>16&&sy<h-24)ctx.fillText(y.toFixed(1),4,sy-3);}
  entities.forEach((e,k)=>{
    ctx.strokeStyle=colors[k%colors.length]+'66';ctx.lineWidth=1.5;ctx.beginPath();let started=false;
    e.roots.forEach((p,i)=>{if(!e.root_valid[i]||p[0]===null){started=false;return;}const q=xy(p);if(started)ctx.lineTo(...q);else ctx.moveTo(...q);started=true;});ctx.stroke();
    ctx.strokeStyle=colors[k%colors.length];ctx.lineWidth=2;ctx.beginPath();started=false;
    e.roots.slice(0,state.frame+1).forEach((p,i)=>{if(!e.root_valid[i]||p[0]===null){started=false;return;}const q=xy(p);if(started)ctx.lineTo(...q);else ctx.moveTo(...q);started=true;});ctx.stroke();
    const p=e.roots[state.frame];
    if(e.root_valid[state.frame]&&p&&p[0]!==null){const q=xy(p);ctx.fillStyle=colors[k%colors.length]+'30';ctx.beginPath();ctx.arc(...q,10,0,Math.PI*2);ctx.fill();ctx.fillStyle=colors[k%colors.length];ctx.beginPath();ctx.arc(...q,4,0,Math.PI*2);ctx.fill();ctx.fillStyle='#edf4ff';ctx.font='10px Consolas';const label=`ID ${e.instance_id}`;const textWidth=ctx.measureText(label).width;ctx.fillText(label,q[0]+textWidth+12>w?q[0]-textWidth-9:q[0]+9,q[1]-8);}
  });
  ctx.fillStyle='#8396ac';ctx.font='9px Consolas';ctx.fillText('X →',w-35,h-9);ctx.fillText('Y ↑',6,14);
}

function updateBuffer() {
  if(!state.current||!state.timing)return;
  const id=++state.bufferRequest,frame=state.frame,scope=state.bufferScope;
  const url=`${base()}/buffer/${frame}?kind=${state.bufferKind}&scope=${scope}`;
  const img=new Image();
  img.onload=()=>{if(id!==state.bufferRequest)return;$('#buffer-image').src=url;state.bufferFrame=frame;state.displayedBufferScope=scope;$('#buffer-frame').textContent=`F ${frame}`;$('#pixel-info').textContent=state.bufferKind==='depth'?'相机 Z 深度（米）；点击像素查看。':'同一案例内稳定的实例 ID；点击像素查看。';};
  img.onerror=()=>{if(id===state.bufferRequest)$('#pixel-info').textContent='缓冲读取失败，请检查本地服务。';};
  img.src=url;
}

function populateRecipe() {
  const n=state.current.frames;
  $('#recipe-case').textContent=`${state.current.game_name} / ${state.current.case_id}`;
  let cutoff=Math.min(n-2,Math.max(0,Math.floor(n*.25)));
  const oos=state.tracks.segments.find(s=>s.state==='oos'&&s.start_frame>0);
  if(oos)cutoff=oos.start_frame-1;
  $('#recipe-start').value=0;$('#recipe-cutoff').value=cutoff;$('#recipe-end').value=n-1;
  ['recipe-start','recipe-cutoff','recipe-end'].forEach(id=>{$('#'+id).max=n-1;});
  $('#recipe-result').textContent='窗口仍需质量复核与数据划分；导出不会开始训练。';
}
async function exportRecipe() {
  if(!state.current)return;
  try {
    const body={task:$('#recipe-task').value,start:Number($('#recipe-start').value),cutoff:Number($('#recipe-cutoff').value),end:Number($('#recipe-end').value),condition:$('#recipe-channel').value};
    const value=await api(base()+'/recipe',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify(body)});
    const blob=new Blob([JSON.stringify(value,null,2)],{type:'application/json'}),url=URL.createObjectURL(blob);
    const a=document.createElement('a');a.href=url;a.download=`${state.current.game}_${state.current.case_id}_${body.task}_window.json`;a.click();setTimeout(()=>URL.revokeObjectURL(url),1000);
    $('#recipe-result').textContent='窗口 JSON 已导出。使用 prepare_window.py 生成分离的历史输入与未来监督。';
  }catch(e){$('#recipe-result').textContent=e.message;toast(e.message);}
}
async function loadCode() {
  try{const value=await api('/api/code/'+$('#code-select').value);$('#code-path').textContent=`${value.file} : ${value.line} · ${value.symbol}`;$('#actual-code').textContent=value.text;}
  catch(e){toast(e.message);}
}
async function saveReview() {
  const token=state.token;
  $('#save-review').disabled=true;
  try{
    const value=await api(base()+'/review',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({status:$('#review-status').value,note:$('#review-note').value})});
    if(token!==state.token)return;
    state.current.review=value;state.detail.review=value;renderDetail();renderList();$('#review-result').textContent='已保存；源数据未修改';toast('复核记录已保存到本地。');
  }catch(e){toast(e.message);}finally{$('#save-review').disabled=false;}
}

async function init() {
  $$('.nav').forEach(b=>b.addEventListener('click',()=>setPage(b.dataset.page)));
  window.addEventListener('hashchange',()=>setPage(location.hash.slice(1),false));
  $('.brand').addEventListener('click',()=>setPage('browse'));
  $$('.game-filters button').forEach(b=>b.addEventListener('click',()=>{state.filter=b.dataset.game;$$('.game-filters button').forEach(x=>x.classList.toggle('active',x===b));renderList();}));
  $('#search').addEventListener('input',e=>{state.search=e.target.value;renderList();});
  $('#case-list').addEventListener('click',e=>{const b=e.target.closest('[data-id]');if(b)selectCase(b.dataset.game,b.dataset.id);});
  $('#play').addEventListener('click',togglePlay);
  $('#prev').addEventListener('click',()=>{pause();seekFrame(state.frame-1);});
  $('#next').addEventListener('click',()=>{pause();seekFrame(state.frame+1);});
  $('#frame-slider').addEventListener('input',e=>{pause();seekFrame(Number(e.target.value));});
  videos().forEach(v=>{
    v.addEventListener('loadedmetadata',()=>{if(v._pendingTime!==undefined){seekVideo(v,v._pendingTime);delete v._pendingTime;}});
    v.addEventListener('error',()=>{if(v.currentSrc)toast('视频读取失败，请确认源数据目录可访问。');});
  });
  const rgb=$('#video-rgb');
  rgb.addEventListener('ended',()=>{pause();if(state.timing)seekFrame(state.current.frames-1);});
  if(rgb.requestVideoFrameCallback)rgb.requestVideoFrameCallback(videoTick);
  else rgb.addEventListener('timeupdate',()=>videoTick(performance.now()));
  $('#channel').addEventListener('change',e=>{
    pause();state.channel=e.target.value;$('#video-proxy').src=mediaUrl(state.channel);$('#video-proxy').load();
    $('#channel-note').textContent={control:'动态层覆盖静态代理',actors:'独立动态层 / 无静态遮挡',semantic:'几何语义 / 正常深度测试'}[state.channel];
    if(state.timing)seekVideo($('#video-proxy'),(state.frame+.15)/state.timing.proxy_fps);
  });
  $$('.buffer-tabs button').forEach(b=>b.addEventListener('click',()=>{state.bufferKind=b.dataset.buffer;$$('.buffer-tabs button').forEach(x=>x.classList.toggle('active',x===b));updateBuffer();}));
  $('#buffer-scope').addEventListener('change',e=>{state.bufferScope=e.target.value;updateBuffer();});
  $('#buffer-image').addEventListener('click',async e=>{
    const img=e.target;if(!img.naturalWidth)return;
    const r=img.getBoundingClientRect(),scale=Math.min(r.width/img.naturalWidth,r.height/img.naturalHeight);
    const dx=(r.width-img.naturalWidth*scale)/2,dy=(r.height-img.naturalHeight*scale)/2;
    const x=Math.floor((e.clientX-r.left-dx)/scale),y=Math.floor((e.clientY-r.top-dy)/scale);
    if(x<0||y<0||x>=img.naturalWidth||y>=img.naturalHeight)return;
    try{const p=await api(`${base()}/sample/${state.bufferFrame}?x=${x}&y=${y}&scope=${state.displayedBufferScope}`);$('#pixel-info').textContent=`F${state.bufferFrame} · (${x}, ${y}) · ID ${p.instance_id} · Z ${p.depth_camera_z_m===null?'∞ / 无覆盖':p.depth_camera_z_m.toFixed(3)+' m'}`;}catch(e){toast(e.message);}
  });
  $('#code-select').addEventListener('change',loadCode);
  $('#save-review').addEventListener('click',saveReview);
  $('#export-recipe').addEventListener('click',exportRecipe);
  document.addEventListener('keydown',e=>{
    if(['INPUT','TEXTAREA','SELECT'].includes(e.target.tagName)||!$('#page-browse').classList.contains('active'))return;
    if(e.key==='ArrowRight'||e.key==='ArrowLeft'){e.preventDefault();pause();seekFrame(state.frame+(e.key==='ArrowRight'?1:-1));}
    if(e.code==='Space'){e.preventDefault();togglePlay();}
  });
  new ResizeObserver(drawTrajectory).observe($('#trajectory'));
  setPage(location.hash.slice(1)||'browse',false);
  try{
    const [summary,cases]=await Promise.all([api('/api/summary'),api('/api/cases')]);
    state.cases=cases;
    const pointDisplay=summary.point_records>=1e6?(summary.point_records/1e6).toFixed(2)+'M':fmt(summary.point_records);
    $('#stats').innerHTML=`<div class="stat"><small>完整案例</small><strong>${fmt(summary.episodes)}</strong><span>${Object.values(summary.games).filter(g=>g.episodes>0).length} 个游戏</span></div><div class="stat"><small>主视角时间步</small><strong>${fmt(summary.frames)}</strong><span>按原生时间对齐</span></div><div class="stat"><small>点轨迹记录</small><strong>${pointDisplay}</strong><span>有效性掩码随附</span></div><div class="stat accent"><small>结构检查通过</small><strong>${summary.structural_passed} / ${summary.episodes}</strong><span>${summary.structural_unvalidated?summary.structural_unvalidated+' 未校验':summary.structural_failures+' 失败'}</span></div><div class="stat amber"><small>视觉验收待完成</small><strong>${summary.episodes-summary.visual_approved}</strong><span>需逐案例复核</span></div>`;
    $('#local-paths').innerHTML=Object.entries(summary.paths).map(([k,v])=>`<dt>${{data:'统一数据',code:'处理代码',web:'网页代码'}[k]}</dt><dd>${esc(v)}</dd>`).join('');
    renderList();
    $('#quality-game-table').innerHTML=Object.entries(summary.games).map(([g,v])=>`<tr><td>${esc(v.name)}</td><td>${fmt(v.episodes)}</td><td>${fmt(v.frames)}</td><td>${{cp2077:'当前交付缺共享静态场景与原生深度',gta5:'粗略碰撞面 / 边界；源目录可能含空目录',rdr2:'粗略场景；源训练资格与质量原因需逐案例检查'}[g]}</td></tr>`).join('');
    $('#quality-title').textContent=`${summary.episodes} 条已渲染案例，视觉质量逐条确认。`;
    $('#quality-count-note').textContent=`当前 ${summary.proxy_videos} 条主视角代理视频；${summary.indexed_views} 个相机引用中 ${summary.rendered_views} 个 main 视角已渲染。当前网页只列出完整主视角代理案例。`;
    const initial=cases.find(c=>c.game==='cp2077')||cases[0];
    if(!initial){$('#case-title').textContent='尚未发现完整主视角代理';$('#case-game').textContent='请先接入本地数据';$('#case-warning').textContent='设置 CODEASMEM_DATA_ROOT 或用 web/start.ps1 -DataRoot 指定包含 game/case/episode.json 的 unified_worlds 根目录。先转换并渲染，再重启服务。';$('#play').disabled=true;$('#export-recipe').disabled=true;return;}
    await selectCase(initial.game,initial.case_id);
  }catch(e){toast(e.message);$('#case-warning').textContent=e.message;}
}
init();
