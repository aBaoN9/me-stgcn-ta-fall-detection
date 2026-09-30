const byId = (id) => document.getElementById(id);
const video = byId('video');
const canvas = byId('skeleton');
const context = canvas.getContext('2d');
let selectedFile = null;
let previewUrl = null;
let jobId = null;
let configuration = null;
let result = null;
let busy = false;

function errorMessage(message) {
  byId('error').textContent = message;
  byId('error').hidden = !message;
}

function setBusy(value) {
  busy = value;
  byId('analyze').disabled = value || !selectedFile || !configuration;
  byId('reset').disabled = value || (!selectedFile && !jobId);
  byId('file-input').disabled = value;
  byId('example').disabled = value || !configuration?.example_available;
  byId('analysis-mode').disabled = value;
  byId('window-seconds').disabled = value || byId('analysis-mode').value !== 'timeline';
}

async function request(url, options = {}) {
  const response = await fetch(url, options);
  if (!response.ok) {
    const detail = await response.json().catch(() => ({}));
    throw new Error(typeof detail.detail === 'string' ? detail.detail : `Yêu cầu thất bại (${response.status}).`);
  }
  return response;
}

function frameStrip(mask) {
  const strip = byId('frame-strip');
  strip.replaceChildren();
  Array.from({ length: 32 }, (_, index) => {
    const element = document.createElement('span');
    element.className = mask ? (mask[index] ? 'detected' : 'missing') : '';
    element.title = mask ? `Mẫu ${index + 1}: ${mask[index] ? 'phát hiện trực tiếp' : 'nội suy'}` : 'Chưa có dữ liệu';
    strip.append(element);
  });
  strip.setAttribute('aria-label', mask ? `${mask.filter(Boolean).length} trên 32 frame phát hiện trực tiếp` : 'Chưa có dữ liệu frame');
}

function resetResult() {
  result = null;
  renderEvidence(null);
  byId('verdict').className = 'verdict neutral';
  byId('verdict-icon').textContent = '—';
  byId('verdict-title').textContent = 'Sẵn sàng phân tích';
  byId('verdict-description').textContent = 'Bấm Phân tích video để xử lý toàn bộ clip.';
  byId('result-status').textContent = 'CHƯA CHẠY';
  byId('score-value').textContent = '— / 1';
  byId('score-fill').style.width = '0%';
  for (const id of ['detection', 'sampled', 'duration', 'elapsed']) byId(id).textContent = '—';
  byId('warnings').hidden = true;
  byId('downloads').hidden = true;
  byId('overlay-toggle').disabled = true;
  byId('progress-area').hidden = true;
  frameStrip();
  draw();
}

async function discardJob() {
  if (jobId) {
    await request(`/api/jobs/${jobId}`, { method: 'DELETE', headers: { 'X-PoseLab': 'local' } });
    jobId = null;
    history.replaceState(null, '', location.pathname);
  }
}

async function selectFile(file) {
  if (busy || !file) return;
  errorMessage('');
  if (!/\.(mp4|webm|mov|avi)$/i.test(file.name)) return errorMessage('Hãy chọn video MP4, WebM, MOV hoặc AVI.');
  if (file.size > 200 * 1024 * 1024 || file.size === 0) return errorMessage('File phải có dữ liệu và không vượt quá 200 MB.');
  try { await discardJob(); } catch (error) { return errorMessage(error.message); }
  if (previewUrl) URL.revokeObjectURL(previewUrl);
  selectedFile = file;
  previewUrl = URL.createObjectURL(file);
  video.src = previewUrl;
  byId('dropzone').hidden = true;
  byId('preview').hidden = false;
  byId('codec-warning').hidden = true;
  byId('filename').textContent = file.name;
  byId('filesize').textContent = `${(file.size / 1024 / 1024).toFixed(1)} MB · Chưa phân tích`;
  resetResult();
  setBusy(false);
}

function renderResult(data) {
  result = data;
  history.replaceState(null, '', `?job=${jobId}`);
  const verdicts = {
    fall: ['!', 'Nghi ngờ té ngã', 'Điểm Fall vượt ngưỡng. Hãy xem lại video; model vẫn có thể báo nhầm.'],
    adl: ['✓', 'Hoạt động thường ngày', 'Điểm Fall dưới ngưỡng. Không đồng nghĩa đã loại trừ được mọi trường hợp té.'],
    insufficient: ['?', 'Chưa đủ dữ liệu pose', 'Không đưa ra nhãn ADL/Fall. Thử clip rõ người hơn hoặc ít che khuất hơn.'],
  };
  const [symbol, title, description] = verdicts[data.label];
  byId('verdict').className = `verdict ${data.label}`;
  byId('verdict-icon').textContent = symbol;
  byId('verdict-title').textContent = title;
  byId('verdict-description').textContent = description;
  byId('result-status').textContent = 'HOÀN TẤT';
  byId('score-value').textContent = data.score === null ? 'Không kết luận' : `${data.score.toFixed(3)} / 1`;
  byId('score-fill').style.width = `${(data.score || 0) * 100}%`;
  byId('score-fill').style.background = data.label === 'fall' ? '#c7824a' : '#489f85';
  byId('detection').textContent = `${(data.detection_rate * 100).toFixed(1)}%`;
  byId('sampled').textContent = `${data.sampled_detected} / 32`;
  byId('duration').textContent = `${data.duration.toFixed(1)} s`;
  byId('elapsed').textContent = `${data.elapsed_seconds.toFixed(1)} s`;
  byId('filesize').textContent = `${data.width} × ${data.height} · ${data.fps.toFixed(2)} FPS · ${data.frame_count} frame`;
  byId('warnings').replaceChildren();
  for (const warning of data.warnings) {
    const paragraph = document.createElement('p');
    paragraph.textContent = warning;
    byId('warnings').append(paragraph);
  }
  byId('warnings').hidden = !data.warnings.length;
  byId('overlay-toggle').disabled = false;
  byId('downloads').hidden = false;
  for (const format of ['json', 'csv']) byId(`download-${format}`).href = `/api/jobs/${jobId}/download/${format}`;
  renderEvidence(data, jobId);
  frameStrip(data.sampled_mask);
  draw();
}

function draw() {
  syncEvidence(video.currentTime || 0);
  const rectangle = canvas.getBoundingClientRect();
  const ratio = window.devicePixelRatio || 1;
  if (canvas.width !== Math.round(rectangle.width * ratio) || canvas.height !== Math.round(rectangle.height * ratio)) {
    canvas.width = Math.round(rectangle.width * ratio);
    canvas.height = Math.round(rectangle.height * ratio);
  }
  context.setTransform(ratio, 0, 0, ratio, 0, 0);
  context.clearRect(0, 0, rectangle.width, rectangle.height);
  if (!result || !byId('overlay-toggle').checked || !configuration || !video.videoWidth) return;
  const index = Math.min(result.overlay.length - 1, Math.max(0, Math.floor(video.currentTime * result.fps)));
  const landmarks = result.overlay[index];
  if (!landmarks) return;
  const scale = Math.min(rectangle.width / video.videoWidth, rectangle.height / video.videoHeight);
  const width = video.videoWidth * scale;
  const height = video.videoHeight * scale;
  const left = (rectangle.width - width) / 2;
  const top = (rectangle.height - height) / 2;
  const point = (landmark) => [left + landmark[0] * width, top + landmark[1] * height];
  context.strokeStyle = '#5ff5c1';
  context.lineWidth = 2;
  context.shadowColor = '#092d22';
  context.shadowBlur = 3;
  for (const [source, target] of configuration.edges) {
    if (landmarks[source][2] < 0.3 || landmarks[target][2] < 0.3) continue;
    context.beginPath(); context.moveTo(...point(landmarks[source])); context.lineTo(...point(landmarks[target])); context.stroke();
  }
  context.fillStyle = '#effff8';
  for (const landmark of landmarks) {
    if (landmark[2] < 0.3) continue;
    context.beginPath(); context.arc(...point(landmark), 2.3, 0, Math.PI * 2); context.fill();
  }
  context.shadowBlur = 0;
}

byId('analyze').addEventListener('click', async () => {
  if (!selectedFile || busy) return;
  setBusy(true); errorMessage(''); resetResult();
  byId('progress-area').hidden = false;
  byId('progress').value = 0;
  byId('progress-percent').textContent = '0%';
  byId('progress-message').textContent = 'Đang gửi video tới backend trên máy…';
  byId('result-status').textContent = 'ĐANG XỬ LÝ';
  byId('verdict-title').textContent = 'Đang đọc chuyển động';
  byId('verdict-description').textContent = 'MediaPipe xử lý từng frame, sau đó model phân loại toàn clip.';
  try {
    await discardJob();
    const form = new FormData(); form.append('file', selectedFile);
    form.append('mode', byId('analysis-mode').value);
    form.append('window_seconds', byId('window-seconds').value);
    jobId = (await (await request('/api/jobs', { method: 'POST', headers: { 'X-PoseLab': 'local' }, body: form })).json()).id;
    while (true) {
      const status = await (await request(`/api/jobs/${jobId}`)).json();
      byId('progress').value = status.progress;
      byId('progress-percent').textContent = `${status.progress}%`;
      byId('progress-message').textContent = status.message;
      showWindowProgress(status.latest_window);
      if (status.status === 'error') throw new Error(status.message);
      if (status.status === 'done') {
        renderResult(await (await request(`/api/jobs/${jobId}/result`)).json());
        break;
      }
      await new Promise((resolve) => setTimeout(resolve, 650));
    }
  } catch (error) {
    errorMessage(error.message);
    byId('result-status').textContent = 'CÓ LỖI';
    byId('verdict-title').textContent = 'Chưa có kết quả';
    byId('verdict-description').textContent = 'Xem thông báo bên trái rồi thử lại.';
    byId('progress-area').hidden = true;
  } finally { setBusy(false); }
});

byId('reset').addEventListener('click', async () => {
  if (busy) return;
  try { await discardJob(); } catch (error) { return errorMessage(error.message); }
  video.pause(); video.removeAttribute('src'); video.load();
  if (previewUrl) URL.revokeObjectURL(previewUrl);
  previewUrl = null; selectedFile = null;
  byId('file-input').value = '';
  byId('preview').hidden = true; byId('dropzone').hidden = false;
  errorMessage(''); resetResult(); setBusy(false);
});
byId('file-input').addEventListener('change', (event) => selectFile(event.target.files[0]));
byId('example').addEventListener('click', async () => {
  setBusy(true); errorMessage('');
  try {
    const blob = await (await request('/api/example')).blob();
    setBusy(false);
    await selectFile(new File([blob], 'S1_ADL_04.mp4', { type: 'video/mp4' }));
  } catch (error) { errorMessage(error.message); setBusy(false); }
});
for (const name of ['dragenter', 'dragover']) byId('dropzone').addEventListener(name, (event) => { event.preventDefault(); if (!busy) byId('dropzone').classList.add('dragging'); });
for (const name of ['dragleave', 'drop']) byId('dropzone').addEventListener(name, (event) => { event.preventDefault(); byId('dropzone').classList.remove('dragging'); });
byId('dropzone').addEventListener('drop', (event) => selectFile(event.dataTransfer.files[0]));
byId('overlay-toggle').addEventListener('change', draw);
video.addEventListener('error', () => { if (selectedFile || jobId) byId('codec-warning').hidden = false; });
video.addEventListener('loadedmetadata', draw);
video.addEventListener('timeupdate', draw);
video.addEventListener('seeked', draw);
window.addEventListener('resize', draw);
function animate() { if (!video.paused) draw(); requestAnimationFrame(animate); }
requestAnimationFrame(animate);
frameStrip();
request('/api/config').then((response) => response.json()).then((data) => {
  configuration = data;
  byId('threshold-label').textContent = `Ngưỡng ${data.threshold.toFixed(3)}`;
  byId('threshold-marker').style.left = `${data.threshold * 100}%`;
  setBusy(false);
  const previous = new URLSearchParams(location.search).get('job');
  if (previous && /^[a-f0-9]{32}$/.test(previous)) restoreSession(previous);
}).catch((error) => errorMessage(`Chưa kết nối được backend: ${error.message}`));

async function restoreSession(identifier) {
  try {
    const report = await (await request(`/api/jobs/${identifier}/result`)).json();
    jobId = identifier;
    byId('analysis-mode').value = report.analysis_mode || 'clip';
    if (report.timeline) byId('window-seconds').value = String(report.timeline.window_seconds);
    byId('window-note').hidden = !report.timeline;
    video.src = `/api/jobs/${identifier}/video`;
    byId('filename').textContent = report.filename;
    byId('dropzone').hidden = true;
    byId('preview').hidden = false;
    renderResult(report);
    setBusy(false);
  } catch (error) {
    errorMessage(`Không khôi phục được phiên: ${error.message}`);
    history.replaceState(null, '', location.pathname);
  }
}
