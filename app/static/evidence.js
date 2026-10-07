const evidenceUi = (id) => document.getElementById(id);
const stateNames = { below: 'Dưới ngưỡng', suspected: 'Đang xác nhận', alert: 'Cảnh báo té ngã', recovering: 'Đang theo dõi', insufficient: 'Không đủ pose' };
let evidenceReport = null;
let evidenceJob = null;
let evidenceSelection = '';
let evidenceDownloads = [];

function exportEvidence(onset) {
  for (const url of evidenceDownloads) URL.revokeObjectURL(url);
  evidenceDownloads = [];
  const report = Object.fromEntries(Object.entries(evidenceReport).filter(([key]) => key !== 'overlay'));
  if (onset !== null) {
    const events = report.timeline?.events || [];
    const next = events.find((event) => event.alert_at >= onset);
    report.manual_reference = { onset_seconds: onset, first_new_alert_delay_seconds: next ? next.alert_at - onset : null, alerts_before_onset: events.filter((event) => event.alert_at < onset).length, note: 'User annotation for one event; video-time delay, not wall-clock latency or validated sensitivity.' };
  }
  const csvValue = (value) => {
    let text = value == null ? '' : typeof value === 'object' ? JSON.stringify(value) : String(value);
    if (typeof value === 'string' && /^[=+@-]/.test(text.trimStart())) text = "'" + text;
    return '"' + text.replaceAll('"', '""') + '"';
  };
  const csvRows = (rows) => '\ufeff' + rows.map((row) => row.map(csvValue).join(',')).join('\r\n');
  const fields = ['start_seconds', 'end_seconds', 'score', 'threshold', 'state', 'label', 'detection_rate', 'sampled_detected', 'quality_reason'];
  const timelineRows = [fields.concat(['peak_joint_speed', 'peak_abs_torso_angle_degrees', 'final_hip_dy', 'manual_onset_seconds'])];
  for (const record of report.timeline?.records || []) timelineRows.push(fields.map((key) => record[key]).concat([record.evidence?.peak_joint_speed, record.evidence?.peak_abs_torso_angle, record.evidence?.final_hip_dy, onset]));
  const payloads = [['json', JSON.stringify(report, null, 2), 'application/json'], ['csv', csvRows([['field', 'value'], ...Object.entries(report)]), 'text/csv'], ['timeline.csv', csvRows(timelineRows), 'text/csv']];
  for (const [format, payload, type] of payloads) {
    const url = URL.createObjectURL(new Blob([payload], { type }));
    evidenceDownloads.push(url);
    const link = evidenceUi(format === 'timeline.csv' ? 'download-timeline' : 'download-' + format);
    link.href = url; link.download = 'poselab-' + evidenceJob.slice(0, 8) + '.' + format;
  }
}

function svgElement(name, attributes = {}, text = null) {
  const element = document.createElementNS('http://www.w3.org/2000/svg', name);
  for (const [key, value] of Object.entries(attributes)) element.setAttribute(key, value);
  if (text !== null) element.textContent = text;
  return element;
}

function plotSeries(id, times, values, settings = {}) {
  const container = evidenceUi(id);
  container.replaceChildren();
  if (!values.length || !values.some(Number.isFinite)) {
    const message = document.createElement('p');
    message.textContent = settings.empty || 'Chưa có dữ liệu trong phạm vi này.';
    container.append(message);
    return;
  }
  const responsive = document.body.classList.contains('analysis-unified');
  const compact = responsive && !settings.wide;
  const width = responsive ? Math.max(180, Math.round(container.clientWidth)) : settings.wide ? 960 : 420;
  const height = compact ? 54 : responsive ? 104 : 210;
  const bounds = responsive ? { left: compact ? 36 : 42, right: width - 12, top: compact ? 5 : 12, bottom: height - 22 } : { left: 54, right: width - 18, top: 22, bottom: 168 };
  const finiteValues = values.filter(Number.isFinite);
  let low = settings.min ?? Math.min(0, ...finiteValues);
  let high = settings.max ?? Math.max(0, ...finiteValues);
  if (high - low < 1e-6) high = low + 1;
  const timeStart = settings.start ?? times[0];
  const timeEnd = Math.max(timeStart + 0.01, settings.end ?? times[times.length - 1]);
  const toX = (time) => bounds.left + (time - timeStart) / (timeEnd - timeStart) * (bounds.right - bounds.left);
  const toY = (value) => bounds.bottom - (value - low) / (high - low) * (bounds.bottom - bounds.top);
  const svg = svgElement('svg', { viewBox: `0 0 ${width} ${height}`, role: 'img', 'aria-label': settings.title || 'Biểu đồ đặc trưng' });
  svg.append(svgElement('title', {}, settings.title || 'Biểu đồ đặc trưng'));
  for (const fraction of compact ? [0, 1] : [0, 0.5, 1]) {
    const value = low + (high - low) * fraction;
    const vertical = toY(value);
    svg.append(svgElement('line', { x1: bounds.left, x2: bounds.right, y1: vertical, y2: vertical, stroke: '#e0e9e3' }));
    svg.append(svgElement('text', { x: bounds.left - 7, y: vertical + 4, 'text-anchor': 'end', fill: '#496355', 'font-size': responsive ? 10 : 13 }, value.toFixed(settings.digits ?? 1)));
    const time = timeStart + (timeEnd - timeStart) * fraction;
    svg.append(svgElement('text', { x: toX(time), y: responsive ? height - 5 : 192, 'text-anchor': fraction === 0 ? 'start' : fraction === 1 ? 'end' : 'middle', fill: '#496355', 'font-size': responsive ? 10 : 13 }, `${time.toFixed(1)} s`));
  }
  if (settings.threshold !== undefined) svg.append(svgElement('line', { x1: bounds.left, x2: bounds.right, y1: toY(settings.threshold), y2: toY(settings.threshold), stroke: '#a66028', 'stroke-dasharray': '6 5', 'stroke-width': 1.5 }));
  let path = '';
  let contiguous = false;
  values.forEach((value, index) => {
    if (!Number.isFinite(value)) { contiguous = false; return; }
    path += `${contiguous ? 'L' : 'M'}${toX(times[index]).toFixed(2)},${toY(value).toFixed(2)} `;
    contiguous = true;
  });
  svg.append(svgElement('path', { d: path, fill: 'none', stroke: settings.color || '#137967', 'stroke-width': 2.5 }));
  values.forEach((value, index) => {
    if (!Number.isFinite(value)) return;
    const dot = svgElement('circle', { cx: toX(times[index]), cy: toY(value), r: compact ? 1.7 : 3,
      fill: settings.mask?.[index] === false ? '#b87332' : (settings.color || '#137967') });
    dot.append(svgElement('title', {}, `${times[index].toFixed(2)} s: ${value.toFixed(3)}${settings.mask?.[index] === false ? ' · nội suy/bù' : ''}`));
    svg.append(dot);
  });
  if (settings.wide) svg.append(svgElement('line', { id: 'chart-cursor', 'data-plot-left': bounds.left, 'data-plot-right': bounds.right, x1: bounds.left, x2: bounds.left, y1: bounds.top, y2: bounds.bottom, stroke: '#223d32', 'stroke-width': 1.5 }));
  container.append(svg);
}

function drawTimelineChart() {
  if (!evidenceReport?.timeline) return;
  const records = evidenceReport.timeline.records;
  plotSeries('score-chart', records.map((record) => record.end_seconds), records.map((record) => record.score),
    { wide: true, min: 0, max: 1, start: 0, end: evidenceReport.duration, threshold: evidenceReport.threshold, title: 'Điểm Fall theo mốc quyết định; khoảng trống là không đủ pose', empty: 'Chưa có cửa sổ đủ dữ liệu' });
}

function resizeEvidence() {
  if (!evidenceReport) return;
  drawTimelineChart();
  evidenceSelection = '';
  syncEvidence(evidenceUi('video').currentTime || 0);
}

function seekEvidence(seconds) {
  const player = evidenceUi('video');
  player.currentTime = seconds;
  syncEvidence(seconds);
}

function renderEvidence(data, identifier = null) {
  evidenceReport = data;
  evidenceJob = identifier;
  evidenceSelection = '';
  evidenceUi('pane-timeline').dataset.ready = String(Boolean(data?.timeline));
  evidenceUi('pane-motion').dataset.ready = String(Boolean(data));
  evidenceUi('playback-status').hidden = !data?.timeline;
  evidenceUi('fall-onset').value = '';
  evidenceUi('download-timeline').hidden = !data?.timeline;
  if (!data) {
    for (const url of evidenceDownloads) URL.revokeObjectURL(url);
    evidenceDownloads = [];
    evidenceUi('timeline-section').hidden = true;
    evidenceUi('timeline-empty').hidden = false;
    evidenceUi('timeline-empty').textContent = 'Chưa có dữ liệu theo thời gian';
    evidenceUi('evidence-scope').disabled = true;
    evidenceUi('evidence-heading').textContent = 'Chưa có dữ liệu';
    evidenceUi('alert-events').replaceChildren();
    evidenceUi('window-rows').replaceChildren();
    evidenceUi('score-chart').replaceChildren();
    evidenceUi('timeline-seek').value = 0;
    evidenceUi('seek-time').textContent = '0,00 s';
    evidenceUi('delay-result').textContent = '';
    for (const identifier of ['motion-speed', 'motion-angle', 'motion-hip']) evidenceUi(identifier).textContent = '—';
    for (const identifier of ['speed-chart', 'angle-chart', 'hip-chart']) plotSeries(identifier, [], [], { empty: 'Chưa có dữ liệu' });
    return;
  }
  const timeline = data.timeline;
  evidenceUi('timeline-section').hidden = !timeline;
  evidenceUi('timeline-empty').hidden = Boolean(timeline);
  evidenceUi('timeline-empty').textContent = 'Chế độ toàn video không tạo điểm theo thời gian';
  evidenceUi('evidence-scope').value = timeline ? 'cursor' : 'clip';
  evidenceUi('evidence-scope').disabled = !timeline;
  evidenceUi('fall-onset').max = Math.max(0, data.duration - 0.001);
  if (timeline) {
    const records = timeline.records;
    const valid = records.filter((record) => record.score !== null).length;
    evidenceUi('timeline-summary').textContent = `Cửa sổ ${timeline.window_seconds} s · bước ${timeline.step_seconds} s · ${valid}/${records.length} cửa sổ đủ pose · ${timeline.events.length} đợt cảnh báo thử nghiệm. Tích lũy ít nhất ${timeline.window_seconds} s; model và ngưỡng chưa được kiểm định cho cửa sổ này.`;
    drawTimelineChart();
    evidenceUi('timeline-seek').max = Math.max(0, (data.frame_count - 1) / data.fps);
    const events = evidenceUi('alert-events');
    events.replaceChildren();
    for (const [index, event] of timeline.events.entries()) {
      const button = document.createElement('button');
      button.type = 'button';
      button.textContent = `Cảnh báo ${index + 1} · ${event.alert_at.toFixed(2)} s → xem lại`;
      button.addEventListener('click', () => seekEvidence(event.alert_at));
      events.append(button);
    }
    if (!timeline.events.length) events.textContent = 'Không có sự kiện';
    const rows = evidenceUi('window-rows');
    rows.replaceChildren();
    for (const record of records) {
      const row = document.createElement('tr');
      const timeCell = document.createElement('td');
      const button = document.createElement('button');
      button.type = 'button';
      button.textContent = `${record.end_seconds.toFixed(2)} s`;
      button.addEventListener('click', () => seekEvidence(record.end_seconds));
      timeCell.append(button); row.append(timeCell);
      for (const text of [`${record.start_seconds.toFixed(2)}–${record.end_seconds.toFixed(2)} s`, record.score === null ? '—' : record.score.toFixed(3), stateNames[record.state], `${(record.detection_rate * 100).toFixed(1)}% · ${record.sampled_detected}/32 mẫu`]) {
        const cell = document.createElement('td'); cell.textContent = text; row.append(cell);
      }
      rows.append(row);
    }
  }
  updateManualReference();
  syncEvidence(evidenceUi('video').currentTime || 0);
}

function updateManualReference() {
  if (!evidenceReport) return;
  const input = evidenceUi('fall-onset');
  const onset = input.value === '' ? null : Number(input.value);
  const valid = onset !== null && Number.isFinite(onset) && onset >= 0 && onset < evidenceReport.duration;
  exportEvidence(valid ? onset : null);
  if (onset === null) {
    evidenceUi('delay-result').textContent = '—';
  } else if (!valid) {
    evidenceUi('delay-result').textContent = 'Nhập mốc nằm trong thời lượng video. Mốc không hợp lệ không được đưa vào báo cáo.';
  } else {
    const events = evidenceReport.timeline?.events || [];
    const next = events.find((event) => event.alert_at >= onset);
    const before = events.filter((event) => event.alert_at < onset).length;
    evidenceUi('delay-result').textContent = `${next ? `Độ trễ theo video: ${(next.alert_at - onset).toFixed(2)} s` : 'Không có cảnh báo mới sau mốc'} · Cảnh báo trước mốc: ${before}`;
  }
}

function syncEvidence(seconds) {
  if (!evidenceReport) return;
  const timeline = evidenceReport.timeline;
  const records = timeline?.records || [];
  let index = -1;
  for (let cursor = 0; cursor < records.length; cursor += 1) {
    if (records[cursor].end_seconds <= seconds + 1e-6) index = cursor;
    else break;
  }
  const current = index < 0 ? null : records[index];
  if (timeline) {
    evidenceUi('timeline-seek').value = seconds;
    evidenceUi('seek-time').textContent = `${seconds.toFixed(2)} s`;
    evidenceUi('playback-time').textContent = `${seconds.toFixed(2)} s`;
    evidenceUi('playback-status').dataset.state = current?.state || 'warmup';
    evidenceUi('playback-state').textContent = current ? stateNames[current.state] : `Đang tích lũy ${timeline.window_seconds} s · chưa kết luận`;
    evidenceUi('playback-score').textContent = current ? `Mốc ${current.end_seconds.toFixed(2)} s · Fall ${current.score === null ? '—' : current.score.toFixed(3)}` : 'Chưa có điểm Fall';
    const cursor = evidenceUi('chart-cursor');
    if (cursor) {
      const left = Number(cursor.dataset.plotLeft);
      const right = Number(cursor.dataset.plotRight);
      const coordinate = left + Math.min(1, seconds / evidenceReport.duration) * (right - left);
      cursor.setAttribute('x1', coordinate); cursor.setAttribute('x2', coordinate);
    }
  }
  const wholeClip = evidenceUi('evidence-scope').value === 'clip';
  const selection = wholeClip ? 'clip' : `window-${index}`;
  if (selection === evidenceSelection) return;
  evidenceSelection = selection;
  const selected = wholeClip ? evidenceReport : current;
  const evidence = selected?.evidence;
  evidenceUi('evidence-heading').textContent = wholeClip ? 'Toàn video' : current ? `Cửa sổ ${current.start_seconds.toFixed(2)}–${current.end_seconds.toFixed(2)} s` : 'Chưa có dữ liệu';
  let reason;
  if (!selected) reason = 'Đang tích lũy dữ liệu. Không lấy kết quả ở tương lai để điền vào đây.';
  else if (selected.score === null) reason = selected.quality_reason === 'current_pose_missing' ? 'Mốc quyết định đang mất pose. Tạm ngừng kết luận dù các frame trước có thể đủ.' : 'Không đủ pose hoặc scale thân hợp lệ: model không đưa ra điểm Fall.';
  else {
    reason = `Điểm ${selected.score.toFixed(3)} ${selected.score >= selected.threshold ? '≥' : '<'} ngưỡng ${selected.threshold.toFixed(3)} → nhãn ${selected.label === 'fall' ? 'nghi ngờ té ngã' : 'ADL'} cho ${wholeClip ? 'toàn clip' : 'cửa sổ này'}.`;
    if (!wholeClip) reason += ` ${selected.high_streak} lần liên tiếp trên ngưỡng, ${selected.low_streak} lần dưới ngưỡng; trạng thái: ${stateNames[selected.state]}.`;
    reason += ' Đây là quy tắc chuyển điểm thành kết quả, không giải thích vì sao mạng tạo ra điểm đó.';
  }
  evidenceUi('decision-reason').textContent = reason;
  evidenceUi('evidence-quality').textContent = selected ? `${(selected.detection_rate * 100).toFixed(1)}% frame có pose · ${selected.sampled_detected}/32 mẫu trực tiếp · ${32 - selected.sampled_detected}/32 mẫu nội suy/bù. Phát hiện được không đảm bảo mọi khớp chính xác.` : 'Chưa có số đo cho phạm vi này.';
  evidenceUi('motion-speed').textContent = evidence ? evidence.peak_joint_speed.toFixed(2) : '—';
  evidenceUi('motion-angle').textContent = evidence ? `${evidence.peak_abs_torso_angle.toFixed(1)}°` : '—';
  evidenceUi('motion-hip').textContent = evidence ? `${(evidence.final_hip_dy * 100).toFixed(1)}%` : '—';
  const times = evidence?.times || [];
  plotSeries('speed-chart', times, evidence?.mean_joint_speed || [], { min: 0, mask: evidence?.direct_mask, title: 'Trung bình độ lớn vận tốc tương đối của 33 khớp' });
  plotSeries('angle-chart', times, evidence?.torso_angle_degrees || [], { min: -180, max: 180, digits: 0, mask: evidence?.direct_mask, title: 'Góc thân, độ, có dấu theo hệ tọa độ ảnh', color: '#536ba1' });
  plotSeries('hip-chart', times, evidence?.hip_dy.map((value) => value * 100) || [], { mask: evidence?.direct_mask, title: 'Dịch chuyển hông Y, phần trăm chiều cao ảnh', color: '#75598c' });
}

function showWindowProgress(record) {
  if (!record) return;
  evidenceUi('progress-message').textContent += ` · mốc ${record.end_seconds.toFixed(2)} s: ${stateNames[record.state]}${record.score === null ? '' : ` (${record.score.toFixed(3)})`}`;
}

evidenceUi('timeline-seek').addEventListener('input', (event) => seekEvidence(Number(event.target.value)));
evidenceUi('evidence-scope').addEventListener('change', () => { evidenceSelection = ''; syncEvidence(evidenceUi('video').currentTime); });
evidenceUi('fall-onset').addEventListener('input', updateManualReference);
evidenceUi('analysis-mode').addEventListener('change', () => {
  const timeline = evidenceUi('analysis-mode').value === 'timeline';
  evidenceUi('window-seconds').disabled = !timeline;
  evidenceUi('window-note').hidden = !timeline;
});
