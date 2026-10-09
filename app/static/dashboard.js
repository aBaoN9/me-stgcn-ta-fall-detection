'use strict';

const element = (identifier) => document.getElementById(identifier);
const video = element('video');
const canvas = element('skeleton');
const context = canvas.getContext('2d');
const textNumber = (value, digits = 2) => value == null ? '—' : Number(value).toLocaleString('vi-VN', { minimumFractionDigits: digits, maximumFractionDigits: digits });
const secondsText = (value) => value == null ? 'Toàn video' : textNumber(value) + ' s';
const percentText = (value) => value == null ? '—' : textNumber(value * 100, 1) + '%';
const escapeHtml = (value) => String(value ?? '').replace(/[&<>"']/g, (character) => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;' }[character]));
const statusNames = { pending: 'Cần xem lại', acknowledged: 'Đã xem', false_alarm: 'Đánh dấu báo nhầm' };
const stateNames = { warming: 'Đang tích lũy dữ liệu', insufficient: 'Không đủ pose', suspected: 'Nghi ngờ té ngã', alert: 'Cảnh báo té ngã', recovering: 'Đang theo dõi sau cảnh báo', below: 'Chưa ghi nhận tín hiệu ngã', clip_fall: 'Nghi ngờ té ngã', clip_adl: 'Hoạt động thường ngày', waiting: 'Chờ dữ liệu', processing: 'Đang phân tích' };
const labelNames = { fall: 'Nghi ngờ ngã', adl: 'Hoạt động thường ngày', insufficient: 'Không đủ pose' };
const headers = { 'X-PoseLab': 'local' };
let configuration = null;
let history = { sessions: [], events: [], counts: {} };
let selectedFile = null;
let previewUrl = null;
let currentJob = null;
let currentSession = null;
let currentReport = null;
let busy = false;
let focusedEvent = null;
let deleteSession = null;
let lastEvidenceKey = '';
let lastPlaybackRecord = null;
let lastPlaybackState = '';

async function request(url, options = {}) {
  const response = await fetch(url, options);
  if (!response.ok) {
    let message = 'Yêu cầu không thành công (' + response.status + ').';
    try { message = (await response.json()).detail || message; } catch {}
    throw new Error(typeof message === 'string' ? message : JSON.stringify(message));
  }
  return response;
}

function showError(message) {
  element('error').textContent = message || '';
  element('error').hidden = !message;
}

function setConnection(online) {
  element('connection').classList.toggle('offline', !online);
  element('connection').innerHTML = '<span class="status-dot"></span>' + (online ? 'Đã kết nối' : 'Mất kết nối');
}

function setBusy(value) {
  busy = value;
  element('analyze').disabled = value || !selectedFile || !configuration;
  for (const identifier of ['choose-video', 'change-video', 'file-input', 'analysis-mode', 'window-seconds']) element(identifier).disabled = value;
  element('example').disabled = value || !configuration?.example_available;
  element('analyze').textContent = value ? 'Đang xử lý…' : 'Phân tích';
  element('show-analysis').disabled = value || !currentReport;
}

function navigate() {
  if (location.hash === '#main') return;
  if (!['history', 'model', 'settings'].includes(location.hash.slice(1))) {
    location.replace('/analysis.html' + location.search);
    return;
  }
  const view = location.hash.slice(1);
  document.body.dataset.view = view;
  const viewLabels = { overview: 'Tổng quan', history: 'Lịch sử cảnh báo', model: 'Mô hình', settings: 'Cài đặt' };
  element('workspace-title').textContent = viewLabels[view];
  const titles = { overview: ['', 'Tổng quan', 'Video và cảnh báo'], history: ['', 'Lịch sử cảnh báo', 'Xem lại sự kiện và quản lý video đã lưu.'], model: ['', 'Thông tin mô hình', 'Cấu hình triển khai và kết quả đánh giá.'], settings: ['', 'Cài đặt', 'Lưu trữ, quyền riêng tư và thông báo.'] };
  for (const section of document.querySelectorAll('.view')) section.hidden = section.id !== 'view-' + view;
  if (view !== 'overview') video.pause();
  for (const link of document.querySelectorAll('[data-view]')) {
    link.classList.toggle('active', link.dataset.view === view);
    if (link.dataset.view === view) link.setAttribute('aria-current', 'page'); else link.removeAttribute('aria-current');
  }
  ['page-eyebrow', 'page-title', 'page-description'].forEach((identifier, index) => element(identifier).textContent = titles[view][index]);
  if (location.hash !== '#main') window.scrollTo(0, 0);
  if (view === 'history') refreshHistory().catch((error) => showError(error.message));
}

function setSignal(state, score, quality, timestamp, scope) {
  element('signal-heading').textContent = stateNames[state] || state;
  document.querySelector('.signal-icon use').setAttribute('href', ['below', 'clip_adl'].includes(state) ? '#i-check' : ['waiting', 'processing', 'warming'].includes(state) ? '#i-shield' : '#i-alert');
  element('signal-heading').closest('.signal-card').classList.toggle('alert', ['alert', 'recovering', 'clip_fall'].includes(state));
  element('signal-heading').closest('.signal-card').classList.toggle('warning', ['insufficient', 'suspected'].includes(state));
  const descriptions = { waiting: 'Chọn video và nhấn Phân tích để bắt đầu.', warming: 'Chưa đủ dữ liệu tại vị trí phát này.', processing: 'Đang xử lý video tải lên, không phải camera trực tiếp.', insufficient: 'Không đủ khung xương để kết luận. Cần xem lại video.', suspected: 'Một cửa sổ vượt ngưỡng; chưa tạo cảnh báo.', alert: 'Hai cửa sổ liên tiếp vượt ngưỡng. Cần kiểm tra video.', recovering: 'Đã cảnh báo; đang chờ hai cửa sổ dưới ngưỡng.', below: 'Cửa sổ dưới ngưỡng; không đồng nghĩa đã an toàn.', clip_fall: 'Điểm toàn video vượt ngưỡng. Cần kiểm tra video.', clip_adl: 'Phân loại toàn video dưới ngưỡng; không phải kết luận an toàn cho từng thời điểm.' };
  element('signal-description').textContent = descriptions[state] || '';
  window.PoseUI.animateScore(element('current-score'), element('score-fill'), score, (value) => textNumber(value, 3));
  element('score-scope').textContent = scope;

  element('current-quality').textContent = percentText(quality);
  element('current-time').textContent = timestamp == null ? '—' : secondsText(timestamp);
}

function resetAnalysis() {
  currentReport = null; currentSession = null; lastEvidenceKey = ''; lastPlaybackRecord = null; lastPlaybackState = '';
  element('analysis-results').hidden = true;
  element('show-analysis').disabled = true;
  if (element('analysis-dialog').open) element('analysis-dialog').close();
  element('progress-area').hidden = true;
  element('player-hud').hidden = true;
  element('pose-legend').hidden = true;
  context.clearRect(0, 0, canvas.width, canvas.height);
  setSignal('waiting', null, null, null, 'chưa có phạm vi');
  renderRecentEvents();
}

async function discardJob() {
  if (currentJob) {
    const identifier = currentJob;
    currentJob = null;
    try { await request('/api/jobs/' + identifier, { method: 'DELETE', headers }); } catch (error) { if (!error.message.includes('Phiên không còn tồn tại')) throw error; }
  }
}

async function selectFile(file) {
  if (!file || busy) return;
  showError('');
  if (!/\.(mp4|mov|avi|webm)$/i.test(file.name)) return showError('Chọn MP4, MOV, AVI hoặc WebM.');
  if (file.size > 200 * 1024 * 1024 || file.size === 0) return showError('Video phải có dữ liệu và không vượt quá 200 MB.');
  try { await discardJob(); } catch (error) { return showError(error.message); }
  video.pause();
  if (previewUrl) URL.revokeObjectURL(previewUrl);
  resetAnalysis();
  selectedFile = file;
  element('codec-warning').hidden = true;
  previewUrl = URL.createObjectURL(file);
  video.src = previewUrl;
  video.hidden = false; canvas.hidden = false;
  element('upload-empty').hidden = true;
  element('filename').textContent = file.name;
  element('video-meta').textContent = textNumber(file.size / 1024 / 1024, 1) + ' MB';
  element('video-state').textContent = 'Sẵn sàng phân tích';
  element('video-state').className = 'pill neutral';
  historyUrl(null);
  setBusy(false);
}

function historyUrl(identifier) {
  const url = new URL(location.href);
  url.searchParams.delete('job');
  if (identifier) url.searchParams.set('session', identifier); else url.searchParams.delete('session');
  window.history.replaceState(null, '', url);
}

async function refreshHistory() {
  history = await (await request('/api/dashboard')).json();
  setConnection(true);
  element('stat-sessions').textContent = history.counts.sessions;
  element('stat-pending').textContent = history.counts.pending;
  element('stat-false').textContent = history.counts.false_alarm;
  element('stat-insufficient').textContent = history.counts.insufficient;
  renderRecentEvents();
  renderHistory();
}

function eventPill(event) {
  return '<span class="pill ' + (event.status === 'pending' ? 'alert' : 'neutral') + '">' + escapeHtml(statusNames[event.status]) + '</span>';
}

function eventsForCurrent() {
  if (currentReport && !currentSession) return [];
  return currentSession ? history.events.filter((event) => event.session_id === currentSession) : history.events;
}

function renderRecentEvents() {
  const events = eventsForCurrent();
  element('alert-count').textContent = events.length;
  element('recent-events').innerHTML = events.length ? events.slice(0, 4).map((event) => '<button class="event-item" data-event="' + escapeHtml(event.id) + '"><span class="event-marker"><svg class="icon"><use href="#i-alert"/></svg></span><span><b>' + escapeHtml(event.filename) + '</b><small>' + secondsText(event.video_seconds) + ' · Điểm ' + textNumber(event.score, 3) + '</small>' + eventPill(event) + '</span></button>').join('') : '<p class="empty-message">' + (currentReport ? 'Chưa có cảnh báo trong video này.' : 'Chưa có cảnh báo.') + '</p>';
}

function renderHistory() {
  const filter = element('event-filter').value;
  const events = history.events.filter((event) => filter === 'all' || event.status === filter);
  element('history-events').innerHTML = events.length ? events.map((event) => '<tr><td>' + escapeHtml(event.filename) + '<small>' + escapeHtml(new Date(event.created_at).toLocaleString('vi-VN')) + '</small></td><td>' + secondsText(event.video_seconds) + '</td><td>' + textNumber(event.score, 3) + '</td><td>' + (event.kind === 'clip_flag' ? 'Phân loại toàn video' : 'Hai cửa sổ vượt ngưỡng') + '</td><td>' + eventPill(event) + '</td><td><button class="button secondary" data-event="' + escapeHtml(event.id) + '">Xem chi tiết</button></td></tr>').join('') : '<tr><td colspan="6" class="empty-message">Không có sự kiện phù hợp bộ lọc.</td></tr>';
  element('history-sessions').innerHTML = history.sessions.length ? history.sessions.map((session) => '<tr><td>' + escapeHtml(session.filename) + '<small>' + escapeHtml(new Date(session.created_at).toLocaleString('vi-VN')) + ' · ' + secondsText(session.duration) + '</small></td><td>' + escapeHtml(labelNames[session.label]) + '<small>Điểm ' + textNumber(session.score, 3) + '</small></td><td>' + percentText(session.detection_rate) + '</td><td>' + session.event_count + '</td><td><div class="table-actions"><button class="button secondary" data-session="' + escapeHtml(session.id) + '">Mở lại</button><button class="text-button" data-delete="' + escapeHtml(session.id) + '">Xóa phiên</button></div></td></tr>').join('') : '<tr><td colspan="5" class="empty-message">Chưa lưu phiên phân tích nào.</td></tr>';
}

async function openSession(identifier, timestamp = 0, showOverview = true) {
  if (busy) return showError('Hãy chờ phân tích hiện tại hoàn tất trước khi mở phiên khác.');
  showError('');
  try {
    const report = await (await request('/api/history/' + identifier + '/result')).json();
    await discardJob();
    video.pause();
    if (previewUrl) URL.revokeObjectURL(previewUrl);
    previewUrl = null; selectedFile = null;
    resetAnalysis();
    currentSession = identifier;
    video.src = '/api/history/' + identifier + '/video';
    video.hidden = false; canvas.hidden = false;
    element('upload-empty').hidden = true;
    element('filename').textContent = report.filename;
    element('analysis-mode').value = report.analysis_mode;
    if (report.timeline) element('window-seconds').value = report.timeline.window_seconds;
    updateMode();
    if (timestamp > 0) video.addEventListener('loadedmetadata', () => { video.currentTime = Math.min(timestamp, video.duration); }, { once: true });
    renderReport(report);
    historyUrl(identifier);
    if (showOverview) location.hash = 'overview';
    setBusy(false);
  } catch (error) { showError(error.message); }
}

function currentWindow() {
  let found = null;
  for (const record of currentReport?.timeline?.records || []) {
    if (record.end_seconds <= video.currentTime + 0.001) found = record; else break;
  }
  return found;
}

function drawSkeleton() {
  const rectangle = video.getBoundingClientRect();
  const pixelRatio = window.devicePixelRatio || 1;
  const widthPixels = Math.round(rectangle.width * pixelRatio);
  const heightPixels = Math.round(rectangle.height * pixelRatio);
  if (canvas.width !== widthPixels || canvas.height !== heightPixels) { canvas.width = widthPixels; canvas.height = heightPixels; }
  context.setTransform(pixelRatio, 0, 0, pixelRatio, 0, 0);
  context.clearRect(0, 0, rectangle.width, rectangle.height);
  if (!currentReport || !element('overlay-toggle').checked || !video.videoWidth) return;
  const index = Math.min(currentReport.overlay.length - 1, Math.floor(video.currentTime * currentReport.fps));
  const direct = currentReport.overlay[index];
  const landmarks = direct || currentReport.overlay_filled?.[index];
  if (!landmarks) return;
  const interpolatedFrame = !direct;
  const fit = Math.min(rectangle.width / currentReport.width, rectangle.height / currentReport.height);
  const width = currentReport.width * fit;
  const height = currentReport.height * fit;
  const left = (rectangle.width - width) / 2;
  const top = (rectangle.height - height) / 2;
  const point = (landmark) => [left + landmark[0] * width, top + landmark[1] * height];
  context.shadowColor = '#0d2a23'; context.shadowBlur = 3;
  for (const [source, target] of configuration.edges) {
    const uncertain = landmarks[source][2] < 0.3 || landmarks[target][2] < 0.3;
    context.strokeStyle = interpolatedFrame ? '#ff9e55' : (uncertain ? '#ffd45f' : '#83f7ce');
    context.lineWidth = interpolatedFrame || uncertain ? 1.6 : 2;
    context.globalAlpha = interpolatedFrame ? 0.72 : (uncertain ? 0.76 : 1);
    context.setLineDash(interpolatedFrame ? [2, 4] : (uncertain ? [5, 4] : []));
    context.beginPath(); context.moveTo(...point(landmarks[source])); context.lineTo(...point(landmarks[target])); context.stroke();
  }
  for (const landmark of landmarks) {
    const uncertain = landmark[2] < 0.3;
    context.fillStyle = interpolatedFrame ? '#ffb070' : (uncertain ? '#ffe08a' : '#effff8');
    context.globalAlpha = interpolatedFrame ? 0.78 : (uncertain ? 0.82 : 1);
    context.beginPath(); context.arc(...point(landmark), 2.3, 0, Math.PI * 2); context.fill();
  }
  context.globalAlpha = 1; context.setLineDash([]); context.shadowBlur = 0;
}

function playbackUpdate() {
  if (currentReport && !busy) {
    const record = currentWindow();
    const timeline = currentReport.analysis_mode === 'timeline';
    const state = timeline ? (record?.state || 'warming') : (currentReport.label === 'insufficient' ? 'insufficient' : 'clip_' + currentReport.label);
    if (record !== lastPlaybackRecord || state !== lastPlaybackState) {
      setSignal(state, timeline ? record?.score : currentReport.score, timeline ? record?.detection_rate : currentReport.detection_rate, timeline ? record?.end_seconds : null, timeline ? 'cửa sổ tại vị trí phát' : 'phân loại toàn video');
      lastPlaybackRecord = record; lastPlaybackState = state;
    }
    element('player-hud').hidden = false;
    element('hud-state').textContent = timeline ? stateNames[state] : 'Kết quả thuộc toàn video';
    element('hud-score').textContent = timeline ? (record ? secondsText(record.end_seconds) + ' · Fall ' + textNumber(record.score, 3) : 'Chưa có cửa sổ tại mốc này') : 'Phân loại toàn video';
    element('player-hud').classList.toggle('alert', timeline && ['alert', 'recovering'].includes(state));
    element('player-hud').classList.toggle('warning', timeline && ['insufficient', 'suspected'].includes(state));
    element('seek-time').textContent = secondsText(video.currentTime);
    element('timeline-seek').value = video.currentTime;
    const cursor = element('score-chart').querySelector('.cursor-line');
    if (cursor) { const position = 42 + Math.min(currentReport.duration, video.currentTime) / currentReport.duration * 732; cursor.setAttribute('x1', position); cursor.setAttribute('x2', position); }
    renderEvidence(record);
  }
  drawSkeleton();
}

function plot(target, times, values, options = {}) {
  const width = options.mini ? 360 : 800;
  const height = options.mini ? 155 : 210;
  const left = 42, right = 26, top = 18, bottom = 28;
  const valid = values.filter((value) => value != null && Number.isFinite(value));
  if (!valid.length) { element(target).innerHTML = '<div class="chart-empty">' + escapeHtml(options.empty || 'Chưa có số đo hợp lệ trong phạm vi này.') + '</div>'; return; }
  let minimum = options.minimum ?? Math.min(...valid);
  let maximum = options.maximum ?? Math.max(...valid);
  if (maximum === minimum) { maximum += 1; minimum -= minimum > 0 ? 1 : 0; }
  const firstTime = options.firstTime ?? times[0];
  const lastTime = options.lastTime ?? times[times.length - 1];
  const positionX = (time) => left + (time - firstTime) / Math.max(0.001, lastTime - firstTime) * (width - left - right);
  const positionY = (value) => top + (maximum - value) / (maximum - minimum) * (height - top - bottom);
  let content = '';
  for (let index = 0; index <= 2; index++) {
    const value = minimum + (maximum - minimum) * index / 2;
    content += '<line class="grid-line" x1="' + left + '" x2="' + (width - right) + '" y1="' + positionY(value) + '" y2="' + positionY(value) + '"/><text x="' + (left - 7) + '" y="' + (positionY(value) + 4) + '" text-anchor="end">' + textNumber(value, options.decimals ?? 1) + '</text>';
  }
  let path = '', started = false;
  values.forEach((value, index) => { if (value == null || !Number.isFinite(value)) { started = false; return; } path += (started ? ' L' : ' M') + positionX(times[index]).toFixed(2) + ' ' + positionY(value).toFixed(2); started = true; });
  content += '<path class="plot-line" d="' + path + '"/>';
  values.forEach((value, index) => { if (value != null && Number.isFinite(value)) content += '<circle cx="' + positionX(times[index]) + '" cy="' + positionY(value) + '" r="' + (options.mask?.[index] === false ? 3 : 1.7) + '" class="' + (options.mask?.[index] === false ? 'filled-point' : 'direct-point') + '" fill="#087f78"/>'; });
  if (options.threshold != null) content += '<line class="threshold-line" x1="' + left + '" x2="' + (width - right) + '" y1="' + positionY(options.threshold) + '" y2="' + positionY(options.threshold) + '"/>';
  for (const event of options.events || []) {
    const index = times.findIndex((time) => Math.abs(time - event.alert_at) < 0.001);
    if (index >= 0 && values[index] != null) content += '<circle class="alert-point" cx="' + positionX(times[index]) + '" cy="' + positionY(values[index]) + '" r="5"/>';
  }
  if (!options.mini) content += '<line class="cursor-line" x1="' + positionX(video.currentTime) + '" x2="' + positionX(video.currentTime) + '" y1="' + top + '" y2="' + (height - bottom) + '"/>';
  content += '<text x="' + left + '" y="' + (height - 5) + '">' + secondsText(firstTime) + '</text><text x="' + (width - right) + '" y="' + (height - 5) + '" text-anchor="end">' + secondsText(lastTime) + '</text>';
  element(target).innerHTML = '<svg viewBox="0 0 ' + width + ' ' + height + '" role="img" aria-label="' + escapeHtml(options.title || 'Diễn biến số đo theo thời gian') + '">' + content + '</svg>';
}

function renderEvidence(record) {
  const clip = element('evidence-scope').value === 'clip' || currentReport.analysis_mode === 'clip';
  const key = clip ? 'clip' : (record?.end_seconds ?? 'warming');
  if (lastEvidenceKey === key) return;
  lastEvidenceKey = key;
  const evidence = clip ? currentReport.evidence : record?.evidence;
  element('evidence-status').textContent = clip ? 'Toàn video' : record ? 'Cửa sổ ' + secondsText(record.start_seconds) + ' → ' + secondsText(record.end_seconds) : 'Chưa có cửa sổ tại vị trí phát';
  element('motion-speed').textContent = textNumber(evidence?.peak_joint_speed, 2);
  element('motion-angle').textContent = evidence ? textNumber(evidence.peak_abs_torso_angle, 1) + '°' : '—';
  element('motion-hip').textContent = evidence ? percentText(evidence.final_hip_dy) : '—';
  for (const [target, field, title] of [['speed-chart', 'mean_joint_speed', 'Vận tốc khớp trung bình, đơn vị thân mỗi giây'], ['angle-chart', 'torso_angle_degrees', 'Góc thân trong ảnh, độ'], ['hip-chart', 'hip_dy', 'Dịch chuyển hông theo chiều cao ảnh, phần trăm']]) {
    plot(target, evidence?.times || [], (evidence?.[field] || []).map((value) => field === 'hip_dy' ? value * 100 : value), { mini: true, mask: evidence?.direct_mask, title });
  }
}

function renderReport(report) {
  currentReport = report;
  lastPlaybackState = ''; lastPlaybackRecord = null; lastEvidenceKey = '';
  element('analysis-results').hidden = false; element('show-analysis').disabled = false;
  element('pose-legend').hidden = false;
  element('video-state').textContent = 'Đã phân tích'; element('video-state').className = 'pill good';
  element('video-meta').textContent = secondsText(report.duration) + ' · ' + textNumber(report.fps, 1) + ' FPS · ' + report.frame_count + ' frame';

  element('clip-summary').textContent = labelNames[report.label] + ' · điểm ' + textNumber(report.score, 3) + ' · pose trực tiếp ' + percentText(report.detection_rate);
  element('download-report').href = currentSession ? '/api/history/' + currentSession + '/download' : '/api/jobs/' + currentJob + '/download/json';
  element('timeline-seek').max = report.duration;
  const records = report.timeline?.records || [];
  element('timeline-count').textContent = records.length + ' cửa sổ · ' + (report.timeline?.events.length || 0) + ' cảnh báo';
  plot('score-chart', records.map((record) => record.end_seconds), records.map((record) => record.score), { minimum: 0, maximum: 1, firstTime: 0, lastTime: report.duration, threshold: report.threshold, events: report.timeline?.events || [], title: 'Điểm Fall từng cửa sổ, từ 0 đến 1; ngưỡng ' + textNumber(report.threshold, 3), empty: report.analysis_mode === 'clip' ? 'Chế độ toàn video không tạo điểm Fall theo thời gian.' : 'Chưa có cửa sổ đủ điều kiện; không được suy ra video an toàn.' });
  element('window-rows').innerHTML = records.length ? records.map((record) => '<tr><td>' + secondsText(record.start_seconds) + ' → ' + secondsText(record.end_seconds) + '</td><td>' + textNumber(record.score, 3) + '</td><td>' + escapeHtml(stateNames[record.state]) + '</td><td>' + percentText(record.detection_rate) + ' · ' + record.sampled_detected + '/32</td></tr>').join('') : '<tr><td colspan="4">Không có quyết định cửa sổ.</td></tr>';
  element('evidence-scope').value = report.analysis_mode === 'clip' ? 'clip' : 'cursor';
  element('evidence-scope').disabled = report.analysis_mode === 'clip';
  element('result-warnings').innerHTML = report.warnings.map((warning) => '<p>' + escapeHtml(warning) + '</p>').join('');
  element('progress-area').hidden = true;
  renderRecentEvents();
  playbackUpdate();
}

async function analyze() {
  if (!selectedFile || busy || !configuration) return;
  setBusy(true); showError(''); resetAnalysis();
  element('progress-area').hidden = false;
  element('progress').value = 0; element('progress-percent').textContent = '0%';
  element('progress-message').textContent = 'Đang gửi video tới backend cục bộ…';
  element('video-state').textContent = 'Đang xử lý'; element('video-state').className = 'pill warning';
  setSignal('processing', null, null, null, 'cửa sổ vừa xử lý');
  try {
    await discardJob();
    const data = new FormData(); data.append('file', selectedFile);
    data.append('mode', element('analysis-mode').value); data.append('window_seconds', element('window-seconds').value);
    currentJob = (await (await request('/api/jobs', { method: 'POST', headers, body: data })).json()).id;
    while (true) {
      const status = await (await request('/api/jobs/' + currentJob)).json();
      element('progress').value = status.progress; element('progress-percent').textContent = status.progress + '%';
      element('progress-message').textContent = status.message;
      if (status.latest_window) setSignal(status.latest_window.state, status.latest_window.score, null, status.latest_window.end_seconds, 'cửa sổ vừa xử lý · ' + status.latest_window.sampled_detected + '/32 mẫu pose');
      if (status.status === 'error') throw new Error(status.message);
      if (status.status === 'done') {
        const report = await (await request('/api/jobs/' + currentJob + '/result')).json();
        currentSession = report.history_saved ? currentJob : null;
        if (report.browser_preview) {
          video.src = '/api/jobs/' + currentJob + '/video';
          element('codec-warning').hidden = true;
        }
        renderReport(report);
        historyUrl(currentSession);
        await refreshHistory();
        break;
      }
      await new Promise((resolve) => setTimeout(resolve, 700));
    }
  } catch (error) {
    showError(error.message); element('progress-area').hidden = true;
    element('video-state').textContent = 'Phân tích chưa hoàn tất';
    setSignal('waiting', null, null, null, 'chưa có kết quả');
  } finally { setBusy(false); playbackUpdate(); }
}

function openEvent(identifier) {
  focusedEvent = history.events.find((event) => event.id === identifier);
  if (!focusedEvent) return;
  const event = focusedEvent;
  element('event-detail').innerHTML = '<p class="detail-title">' + escapeHtml(event.filename) + '</p>' + eventPill(event) + '<div class="detail-grid"><div><span>Loại quyết định</span><strong>' + (event.kind === 'clip_flag' ? 'Phân loại toàn video' : 'Cảnh báo cửa sổ trượt') + '</strong></div><div><span>Mốc trong video</span><strong>' + secondsText(event.video_seconds) + '</strong></div><div><span>Điểm / ngưỡng</span><strong>' + textNumber(event.score, 3) + ' / ' + textNumber(event.threshold, 3) + '</strong></div><div><span>Pose trực tiếp</span><strong>' + percentText(event.detection_rate) + '</strong></div><div><span>Thời điểm lưu trên máy</span><strong>' + escapeHtml(new Date(event.created_at).toLocaleString('vi-VN')) + '</strong></div><div><span>Thông báo ra ngoài</span><strong>Chưa gửi · chỉ lưu cục bộ</strong></div></div><p class="small-note">' + (event.kind === 'clip_flag' ? 'Kết quả toàn clip không xác định thời điểm bắt đầu ngã.' : 'Mốc cảnh báo là lúc đủ điều kiện quyết định, không phải mốc ngã thật do người dùng xác nhận.') + '</p><p class="hash-line">Mã kiểm tra phiên bản: <code>' + escapeHtml(event.model_sha256) + '</code></p>';
  element('event-note').value = event.note;
  element('dialog-error').hidden = true;
  element('event-dialog').showModal();
}

async function review(status) {
  if (!focusedEvent) return;
  try {
    await request('/api/events/' + focusedEvent.id + '/review', { method: 'POST', headers: { ...headers, 'Content-Type': 'application/json' }, body: JSON.stringify({ status, note: element('event-note').value }) });
    await refreshHistory();
    element('event-dialog').close();
  } catch (error) { element('dialog-error').textContent = error.message; element('dialog-error').hidden = false; }
}

function updateMode() {
  const timeline = element('analysis-mode').value === 'timeline';
  element('window-label').hidden = !timeline;
  element('policy-note').textContent = timeline ? 'Hai cửa sổ liên tiếp vượt ngưỡng → cảnh báo. Thiếu pose → chưa thể kết luận.' : 'Kết quả toàn video; không xác định thời điểm ngã.';
}

function initializeModel() {
  const threshold = configuration.threshold;
  element('threshold-label').textContent = 'Ngưỡng ' + textNumber(threshold, 3);
  element('threshold-marker').style.left = threshold * 100 + '%';
  element('model-threshold').textContent = textNumber(threshold, 3);
  const metrics = configuration.evaluation?.pooled_metrics;
  element('model-accuracy').textContent = percentText(metrics?.accuracy);
  element('model-sensitivity').textContent = percentText(metrics?.sensitivity);
  element('model-specificity').textContent = percentText(metrics?.specificity);
  element('model-errors').textContent = metrics ? metrics.fn + ' / ' + metrics.fp : '—';
  element('model-hash').textContent = configuration.model_sha256;
  setBusy(false);
}

element('choose-video').addEventListener('click', () => element('file-input').click());
element('change-video').addEventListener('click', () => element('file-input').click());
element('file-input').addEventListener('change', async (event) => { await selectFile(event.target.files[0]); event.target.value = ''; });
element('analyze').addEventListener('click', analyze);
element('analysis-mode').addEventListener('change', updateMode);
element('overlay-toggle').addEventListener('change', drawSkeleton);
element('evidence-scope').addEventListener('change', () => { lastEvidenceKey = ''; playbackUpdate(); });
element('timeline-seek').addEventListener('input', (event) => { video.currentTime = Number(event.target.value); playbackUpdate(); });
element('event-filter').addEventListener('change', renderHistory);
element('refresh-history').addEventListener('click', () => refreshHistory().catch((error) => showError(error.message)));
element('close-dialog').addEventListener('click', () => element('event-dialog').close());
element('review-ack').addEventListener('click', () => review('acknowledged'));
element('review-false').addEventListener('click', () => review('false_alarm'));
element('review-pending').addEventListener('click', () => review('pending'));
element('replay-event').addEventListener('click', () => { element('event-dialog').close(); openSession(focusedEvent.session_id, focusedEvent.video_seconds || 0); });
element('cancel-delete').addEventListener('click', () => element('delete-dialog').close());
element('confirm-delete').addEventListener('click', async () => {
  if (!deleteSession) return;
  try {
    await request('/api/history/' + deleteSession.id, { method: 'DELETE', headers });
    if (deleteSession.id === currentSession) {
      video.pause(); video.removeAttribute('src'); video.load();
      video.hidden = true; canvas.hidden = true; element('upload-empty').hidden = false;
      selectedFile = null; resetAnalysis(); historyUrl(null); setBusy(false);
      element('filename').textContent = 'Chưa chọn video'; element('video-meta').textContent = 'Phiên đã xóa khỏi bộ nhớ ứng dụng.';
      element('video-state').textContent = 'Chưa có video'; element('video-state').className = 'pill neutral';
    }
    await refreshHistory(); element('delete-dialog').close();
  } catch (error) { element('delete-error').textContent = error.message; element('delete-error').hidden = false; }
});
document.addEventListener('click', (event) => {
  const eventButton = event.target.closest('[data-event]');
  const sessionButton = event.target.closest('[data-session]');
  const deleteButton = event.target.closest('[data-delete]');
  if (eventButton) openEvent(eventButton.dataset.event);
  if (sessionButton) openSession(sessionButton.dataset.session);
  if (deleteButton) {
    if (busy) return showError('Hãy chờ phân tích hoàn tất trước khi xóa phiên.');
    deleteSession = history.sessions.find((session) => session.id === deleteButton.dataset.delete);
    element('delete-description').textContent = deleteSession.filename;
    element('delete-error').hidden = true;
    element('delete-dialog').showModal();
  }
});
element('example').addEventListener('click', async () => {
  if (busy) return;
  setBusy(true);
  try { const blob = await (await request('/api/example')).blob(); setBusy(false); await selectFile(new File([blob], 'S1_ADL_04.mp4', { type: 'video/mp4' })); } catch (error) { showError(error.message); setBusy(false); }
});
element('show-analysis').addEventListener('click', () => {
  if (!currentReport || busy) return;
  video.pause();
  element('analysis-dialog').showModal();
  playbackUpdate();
});
element('close-analysis').addEventListener('click', () => element('analysis-dialog').close());
element('analysis-dialog').addEventListener('close', () => { drawSkeleton(); element('show-analysis').focus(); });
element('player-stage').addEventListener('dragover', (event) => { event.preventDefault(); });
element('player-stage').addEventListener('drop', (event) => { event.preventDefault(); selectFile(event.dataTransfer.files[0]); });
element('score-chart').addEventListener('click', (event) => {
  const plotSvg = element('score-chart').querySelector('svg');
  if (!plotSvg || !currentReport) return;
  const bounds = plotSvg.getBoundingClientRect();
  const position = (event.clientX - bounds.left) / bounds.width * 800;
  video.currentTime = Math.max(0, Math.min(currentReport.duration, (position - 42) / 732 * currentReport.duration));
  playbackUpdate();
});
video.addEventListener('error', () => { if (video.getAttribute('src') && [3, 4].includes(video.error?.code)) element('codec-warning').hidden = false; });
video.addEventListener('loadedmetadata', () => { element('codec-warning').hidden = true; playbackUpdate(); });
video.addEventListener('canplay', () => { element('codec-warning').hidden = true; playbackUpdate(); });
for (const event of ['timeupdate', 'seeked', 'pause', 'play']) video.addEventListener(event, playbackUpdate);
window.addEventListener('resize', drawSkeleton);
window.addEventListener('hashchange', navigate);
function animationFrame() { if (!video.paused) playbackUpdate(); window.requestAnimationFrame(animationFrame); }
window.requestAnimationFrame(animationFrame);
function updateClock() { element('clock').textContent = new Date().toLocaleTimeString('vi-VN'); }
updateClock(); setInterval(updateClock, 1000);
navigate(); updateMode();
(async () => {
  try {
    configuration = await (await request('/api/config')).json();
    initializeModel();
    await refreshHistory();
    const parameters = new URL(location.href).searchParams;
    if (parameters.get('session')) await openSession(parameters.get('session'), 0, false);
    else if (parameters.get('job')) {
      const report = await (await request('/api/jobs/' + parameters.get('job') + '/result')).json();
      currentJob = parameters.get('job'); currentSession = report.history_saved ? currentJob : null;
      video.src = '/api/jobs/' + currentJob + '/video'; video.hidden = false; canvas.hidden = false;
      element('upload-empty').hidden = true; element('filename').textContent = report.filename;
      renderReport(report);
    }
  } catch (error) { setConnection(false); showError(error.message); }
})();
setInterval(() => refreshHistory().catch(() => setConnection(false)), 15000);
