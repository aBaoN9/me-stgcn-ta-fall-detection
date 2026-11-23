(() => {
  'use strict';
  const data = JSON.parse(document.getElementById('payload').textContent);
  const element = (name) => document.getElementById(name);
  const video = element('video');
  const rows = data.rows || [];
  const eventButtons = [];
  const duration = data.info ? data.info.duration : 1;
  const connections = [[0,1],[1,2],[2,3],[3,7],[0,4],[4,5],[5,6],[6,8],[9,10],
    [11,12],[11,13],[13,15],[15,17],[15,19],[15,21],[17,19],[12,14],[14,16],
    [16,18],[16,20],[16,22],[18,20],[11,23],[12,24],[23,24],[23,25],[24,26],
    [25,27],[26,28],[27,29],[28,30],[29,31],[30,32],[27,31],[28,32]];
  const svgElement = (tag, attributes) => {
    const node = document.createElementNS('http://www.w3.org/2000/svg', tag);
    Object.entries(attributes).forEach(([key,value]) => node.setAttribute(key, String(value)));
    return node;
  };
  const clock = (seconds) => `${String(Math.floor(seconds / 60)).padStart(2,'0')}:${String(Math.floor(seconds % 60)).padStart(2,'0')}`;
  const number = (value, digits=2) => value === null || value === undefined ? '—' : value.toFixed(digits);
  const lastIndex = (time) => {
    let lower = 0, upper = rows.length - 1, answer = -1;
    while (lower <= upper) {
      const middle = Math.floor((lower + upper) / 2);
      if (rows[middle].time <= time) { answer = middle; lower = middle + 1; }
      else { upper = middle - 1; }
    }
    return answer;
  };
  const charts = {};
  function makeChart(name, maximum, threshold) {
    const svg = element(name);
    [0, .5, 1].forEach((ratio) => {
      const vertical = 100 - ratio * 85;
      svg.append(svgElement('line', {x1:28,y1:vertical,x2:392,y2:vertical,stroke:'var(--line)','stroke-width':1}));
      const label = svgElement('text',{x:22,y:vertical+3,'text-anchor':'end',fill:'var(--muted)','font-size':9});
      label.textContent = maximum === 1 ? ratio.toFixed(1) : String(ratio * maximum);
      svg.append(label);
    });
    if (threshold !== null) {
      const vertical = 100 - threshold / maximum * 85;
      svg.append(svgElement('line',{x1:28,y1:vertical,x2:392,y2:vertical,stroke:'#b38b5e','stroke-dasharray':'4 4'}));
    }
    const path = svgElement('path',{fill:'none',stroke:maximum===1?'var(--accent)':'var(--secondary)','stroke-width':2,'stroke-linejoin':'round'});
    const point = svgElement('circle',{r:3,fill:'var(--accent)',visibility:'hidden'});
    const cursor = svgElement('line',{x1:28,y1:10,x2:28,y2:103,stroke:'var(--ink)','stroke-width':1,opacity:.5});
    const left = svgElement('text',{x:28,y:120,fill:'var(--muted)','font-size':9});
    left.textContent='0s';
    const right = svgElement('text',{x:392,y:120,fill:'var(--muted)','font-size':9,'text-anchor':'end'});
    right.textContent=data.info?`${duration.toFixed(1)}s`:'Chờ video';
    svg.append(path, point, cursor, left, right);
    charts[name] = {path,point,cursor,maximum};
  }
  function drawChart(name, points, time) {
    const chart = charts[name];
    let path = '', connected = false, lastPoint = null;
    points.forEach((point) => {
      if (point.value === null) {connected=false;lastPoint=null;return;}
      const horizontal = 28 + point.time / duration * 364;
      const vertical = 100 - Math.max(0,Math.min(chart.maximum,point.value)) / chart.maximum * 85;
      path += `${connected?'L':'M'}${horizontal.toFixed(2)},${vertical.toFixed(2)} `;
      connected=true;lastPoint={horizontal,vertical};
    });
    chart.path.setAttribute('d',path);
    if(lastPoint) {
      chart.point.setAttribute('cx',lastPoint.horizontal);chart.point.setAttribute('cy',lastPoint.vertical);
      chart.point.setAttribute('visibility','visible');
    } else chart.point.setAttribute('visibility','hidden');
    const horizontal = 28 + Math.min(time/duration,1)*364;
    chart.cursor.setAttribute('x1',horizontal);chart.cursor.setAttribute('x2',horizontal);
  }
  function placeSkeleton() {
    if(!video.videoWidth) return;
    const stage = element('stage');
    const scale = Math.min(stage.clientWidth/video.videoWidth,stage.clientHeight/video.videoHeight);
    const width = video.videoWidth*scale, height = video.videoHeight*scale;
    Object.assign(element('skeleton').style,{width:`${width}px`,height:`${height}px`,left:`${(stage.clientWidth-width)/2}px`,top:`${(stage.clientHeight-height)/2}px`});
    element('skeleton').setAttribute('viewBox','0 0 1000 1000');
  }
  function drawSkeleton(frame) {
    const svg = element('skeleton');
    svg.replaceChildren();
    const pose = data.poses && data.poses[frame];
    svg.style.display = pose && element('overlay').checked ? 'block':'none';
    if(!pose || !element('overlay').checked) return;
    connections.forEach(([start,end]) => {
      if(pose[start][2]<.35 || pose[end][2]<.35) return;
      svg.append(svgElement('line',{x1:pose[start][0]*1000,y1:pose[start][1]*1000,x2:pose[end][0]*1000,y2:pose[end][1]*1000,stroke:'#acf0b1','stroke-width':4,'stroke-linecap':'round'}));
    });
    pose.forEach((joint) => {
      if(joint[2]>=.35) svg.append(svgElement('circle',{cx:joint[0]*1000,cy:joint[1]*1000,r:4.5,fill:'#fcf4a3',stroke:'#163d39','stroke-width':1.5}));
    });
  }
  function update() {
    const time = Math.min(video.currentTime || 0,duration);
    const index = lastIndex(time);
    const row = rows[index];
    const frame = data.info ? Math.min(data.info.frames-1,Math.floor(time*data.info.fps)) : 0;
    const hasPose = data.poses && data.poses[frame];
    const score = row && hasPose ? row.score : null;
    let state = 'Chưa phân tích', note = 'Chọn video để bắt đầu phân tích.';
    if(row) {
      if(row.warmup) { state='Đang tích lũy';note=`Cần đủ ${data.window_seconds}s và ít nhất 32 frame quá khứ.`; }
      else if(!hasPose || row.score===null) {state='Không đủ pose';note='Không kết luận ADL khi thiếu dữ liệu.';}
      else { state=score>=.5?'Score cao / Fall':'Score thấp / ADL';note=`Cửa sổ kết thúc tại ${row.time.toFixed(2)}s · ngưỡng cố định 0.50.`; }
    }
    element('state').textContent=state;element('stateNote').textContent=note;
    element('score').replaceChildren(document.createTextNode(number(score)),Object.assign(document.createElement('small'),{textContent:'/ 1.00'}));
    element('scoreFill').style.width=`${score===null?0:score*100}%`;
    document.querySelector('.analytics').classList.toggle('fall',score!==null && score>=.5);
    element('quality').textContent=row?`${(row.quality*100).toFixed(0)}%`:'—';
    element('latency').textContent=row && row.model_ms!==null?`${number(row.model_ms,1)} ms`:'—';
    element('tick').textContent=`t = ${time.toFixed(2)}s`;
    element('clock').textContent=`${clock(time)} / ${data.info?clock(duration):'00:00'}`;
    element('seek').value=time;
    element('eventCursor').style.left=`${Math.min(time/duration,1)*100}%`;
    drawChart('scoreChart',rows.slice(0,index+1).map((entry)=>({time:entry.time,value:entry.score})),time);
    const angles=[];
    if(data.info) {
      const stride=Math.max(1,Math.round(data.info.fps/10));
      for(let frameIndex=0;frameIndex<=frame;frameIndex+=stride) angles.push({time:frameIndex/data.info.fps,value:data.angles[frameIndex]});
    }
    drawChart('angleChart',angles,time);
    drawSkeleton(frame);
    eventButtons.forEach(({button,event}) => button.setAttribute('aria-current',
      String(time >= event.start && time < event.end)));
  }
  let animation = null, previousTick = -1;
  function animate() {
    const tick=Math.floor(video.currentTime*30);
    if(tick!==previousTick){update();previousTick=tick;}
    if(!video.paused && !video.ended) animation=requestAnimationFrame(animate);
  }
  makeChart('scoreChart',1,.5);makeChart('angleChart',180,null);
  element('modelName').textContent=data.model_name;
  if(data.video) {
    const bytes=Uint8Array.from(atob(data.video),(character)=>character.charCodeAt(0));
    const source=URL.createObjectURL(new Blob([bytes],{type:'video/mp4'}));
    video.src=source;data.video=null;
    window.addEventListener('pagehide',()=>{URL.revokeObjectURL(source);cancelAnimationFrame(animation);},{once:true});
    video.hidden=false;element('empty').hidden=true;element('videoTag').hidden=false;
    ['play','seek','fullscreen'].forEach((name)=>element(name).disabled=false);
    element('seek').max=duration;element('mode').textContent='PHÂN TÍCH TRƯỚC';
    element('eventCount').textContent=`${data.events.length} vùng / toàn video`;
    element('listCount').textContent=`(${data.events.length})`;
    element('eventList').replaceChildren();
    if (!data.events.length) {
      const empty = document.createElement('p');
      empty.className='event-empty';
      empty.textContent=data.valid_windows ? 'Không có vùng score cao. Điều này không bảo đảm video không có ngã.' :
        'Không đủ dữ liệu để kết luận; video quá ngắn hoặc thiếu pose.';
      element('eventList').append(empty);
    }
    element('videoNote').textContent=`${data.info.width} × ${data.info.height} · ${data.info.fps.toFixed(1)} FPS · ${data.valid_windows} cửa sổ hợp lệ · xử lý ${data.processing_seconds.toFixed(1)}s · preview bỏ âm thanh`;
    data.events.forEach((event,index)=>{
      const marker=document.createElement('span');marker.className='event';
      marker.style.left=`${event.start/duration*100}%`;marker.style.width=`${(event.end-event.start)/duration*100}%`;
      marker.title=`${event.start.toFixed(2)}–${event.end.toFixed(2)}s, peak ${event.peak.toFixed(3)}`;
      element('events').append(marker);
      const button=document.createElement('button');
      button.className='event-item';button.type='button';
      button.setAttribute('aria-label',`Xem vùng ${index+1}, từ ${event.start.toFixed(2)} đến ${event.end.toFixed(2)} giây`);
      if (event.thumbnail) {
        const image=document.createElement('img');
        image.src=`data:image/jpeg;base64,${event.thumbnail}`;
        image.alt=`Ảnh đầu vùng ${index+1}`;image.loading='lazy';
        button.append(image);
      }
      const label=document.createElement('span'), title=document.createElement('strong'), detail=document.createElement('small');
      title.textContent=`Vùng ${index+1} · ${event.start.toFixed(2)}–${event.end.toFixed(2)}s`;
      detail.textContent=`Score cao nhất ${event.peak.toFixed(3)} · Bấm để xem lại`;
      label.append(title,detail);button.append(label);
      button.addEventListener('click',()=>{
        video.pause();video.currentTime=Math.min(event.end,event.start+.001);update();
      });
      element('eventList').append(button);eventButtons.push({button,event});
    });
  }
  element('play').addEventListener('click',async()=>{
    if(video.paused) {try{await video.play();}catch{element('stateNote').textContent='Không phát được preview. Thử tải video khác.';}}
    else video.pause();
  });
  video.addEventListener('play',()=>{element('play').textContent='Tạm dừng';cancelAnimationFrame(animation);animate();});
  video.addEventListener('pause',()=>{element('play').textContent='Phát';cancelAnimationFrame(animation);update();});
  video.addEventListener('ended',()=>{element('play').textContent='Phát lại';update();});
  video.addEventListener('loadedmetadata',()=>{placeSkeleton();update();});
  video.addEventListener('timeupdate',update);video.addEventListener('seeked',update);
  video.addEventListener('error',()=>{element('stateNote').textContent='Lỗi phát preview video.';});
  element('seek').addEventListener('input',(event)=>{video.currentTime=Number(event.target.value);update();});
  element('speed').addEventListener('change',(event)=>{video.playbackRate=Number(event.target.value);});
  element('overlay').addEventListener('change',update);
  element('fullscreen').addEventListener('click',async()=>{
    try{if(document.fullscreenElement) await document.exitFullscreen();else await element('stage').requestFullscreen();}
    catch{element('stateNote').textContent='Trình duyệt chưa cho phép toàn màn hình trong iframe.';}
  });
  new ResizeObserver(placeSkeleton).observe(element('stage'));
  document.addEventListener('fullscreenchange',placeSkeleton);
  update();
})();
