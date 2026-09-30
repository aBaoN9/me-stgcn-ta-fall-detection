const batchUi = (id) => document.getElementById(id);
const outcomeNames = { TP: 'TP · có cảnh báo từ mốc ngã', FP: 'FP · báo động nhầm', FN: 'FN · không có cảnh báo mới từ mốc', TN: 'TN · không cảnh báo trên ADL', incomplete: 'Thiếu pose · không chấm', too_short: 'Chưa đủ cửa sổ', needs_onset: 'Thiếu mốc ngã', no_post_onset_window: 'Không có cửa sổ sau mốc', annotation_error: 'Mốc nhãn không hợp lệ', error: 'Lỗi xử lý', cancelled: 'Đã dừng', pending: 'Chưa xử lý' };
const statusNames = { uploading: 'ĐANG NHẬN FILE', running: 'ĐANG XỬ LÝ', done: 'HOÀN TẤT', cancelled: 'ĐÃ DỪNG', interrupted: 'BỊ GIÁN ĐOẠN', error: 'CÓ LỖI', pending: 'ĐANG CHỜ' };
let drafts = [];
let locked = false;
let batchId = null;
let polling = false;
let config = null;
let previewIndex = null;
let previewUrl = null;

function batchError(text) { batchUi('batch-error').textContent = text; batchUi('batch-error').hidden = !text; }
async function batchRequest(url, options = {}) {
  const response = await fetch(url, options);
  if (!response.ok) {
    const payload = await response.json().catch(() => ({}));
    throw new Error(typeof payload.detail === 'string' ? payload.detail : `Yêu cầu lỗi (${response.status}).`);
  }
  return response.json();
}
function lockEditor(value) {
  locked = value;
  for (const element of batchUi('batch-editor').querySelectorAll('input, select, button')) element.disabled = value;
  batchUi('batch-run').disabled = value || !config || !drafts.length;
  if (!value) drafts.forEach((draft) => { draft.onsetInput.disabled = draft.label !== 'fall'; });
}
function renderQueue() {
  batchUi('batch-queue').replaceChildren();
  batchUi('queue-wrap').hidden = !drafts.length;
  batchUi('selection-count').textContent = `${drafts.length} video · ${(drafts.reduce((sum, draft) => sum + draft.file.size, 0) / 1024 / 1024).toFixed(1)} MB`;
  drafts.forEach((draft, index) => {
    const row = document.createElement('tr');
    const name = document.createElement('td');
    name.textContent = `${index + 1}. ${draft.file.name}`;
    const labelCell = document.createElement('td');
    const label = document.createElement('select');
    label.setAttribute('aria-label', `Nhãn thật video ${index + 1}`);
    for (const [value, text] of [['', 'Chọn nhãn thật…'], ['adl', 'ADL · không ngã'], ['fall', 'FALL · có ngã']]) {
      const option = document.createElement('option'); option.value = value; option.textContent = text; label.append(option);
    }
    label.value = draft.label;
    const onsetCell = document.createElement('td');
    const onset = document.createElement('input');
    onset.type = 'number'; onset.min = '0'; onset.step = '0.01'; onset.placeholder = 'Chưa xác nhận';
    onset.setAttribute('aria-label', `Mốc ngã video ${index + 1} (giây)`);
    onset.disabled = draft.label !== 'fall'; onset.value = draft.onset ?? '';
    draft.onsetInput = onset;
    onset.addEventListener('input', () => { batchError(''); draft.onset = onset.value === '' ? null : Number(onset.value); });
    label.addEventListener('change', () => {
      batchError('');
      draft.label = label.value;
      onset.disabled = label.value !== 'fall';
      if (label.value !== 'fall') { draft.onset = null; onset.value = ''; }
    });
    labelCell.append(label); onsetCell.append(onset);
    const previewCell = document.createElement('td');
    const preview = document.createElement('button'); preview.type = 'button'; preview.textContent = 'Xem / ghi mốc';
    preview.addEventListener('click', () => openAnnotation(index)); previewCell.append(preview);
    row.append(name, labelCell, onsetCell, previewCell); batchUi('batch-queue').append(row);
  });
  lockEditor(false);
}
function openAnnotation(index) {
  if (locked) return;
  previewIndex = index;
  if (previewUrl) URL.revokeObjectURL(previewUrl);
  previewUrl = URL.createObjectURL(drafts[index].file);
  batchUi('annotation-title').textContent = drafts[index].file.name;
  batchUi('annotation-video').src = previewUrl;
  batchUi('annotation-codec').hidden = true;
  batchUi('use-onset').disabled = true;
  batchUi('annotation-dialog').showModal();
}
batchUi('annotation-close').addEventListener('click', () => batchUi('annotation-dialog').close());
batchUi('annotation-dialog').addEventListener('close', () => {
  batchUi('annotation-video').pause(); batchUi('annotation-video').removeAttribute('src'); batchUi('annotation-video').load();
  if (previewUrl) URL.revokeObjectURL(previewUrl);
  previewUrl = null; previewIndex = null;
});
batchUi('annotation-video').addEventListener('loadedmetadata', () => {
  batchUi('use-onset').disabled = previewIndex === null || drafts[previewIndex].label !== 'fall';
});
batchUi('annotation-video').addEventListener('timeupdate', () => { batchUi('annotation-time').textContent = `${batchUi('annotation-video').currentTime.toFixed(2)} s`; });
batchUi('annotation-video').addEventListener('error', () => { if (previewIndex !== null) batchUi('annotation-codec').hidden = false; });
batchUi('use-onset').addEventListener('click', () => {
  if (previewIndex === null || locked || drafts[previewIndex].label !== 'fall') return;
  const player = batchUi('annotation-video');
  if (!Number.isFinite(player.duration) || player.currentTime >= player.duration) return;
  const value = Math.floor(player.currentTime * 100) / 100;
  drafts[previewIndex].onset = value;
  drafts[previewIndex].onsetInput.value = value;
  batchUi('annotation-dialog').close();
});
batchUi('batch-files').addEventListener('change', (event) => {
  if (locked) return;
  batchError('');
  const files = Array.from(event.target.files);
  if (!files.length) return;
  if (files.length > 20 || files.some((file) => !file.size || file.size > 200 * 1024 * 1024 || !/\.(mp4|webm|mov|avi)$/i.test(file.name)) || files.reduce((sum, file) => sum + file.size, 0) > 600 * 1024 * 1024) {
    event.target.value = ''; drafts = []; renderQueue(); return batchError('Chọn 1–20 video hợp lệ, tối đa 200 MB/file và 600 MB tổng; không nhận file rỗng.');
  }
  drafts = files.map((file) => ({ file, label: '', onset: null })); renderQueue();
});

function percent(value) { return value === null ? '—' : `${(value * 100).toFixed(1)}%`; }
function seconds(value) { return value === null || value === undefined ? '—' : `${value.toFixed(2)} s`; }
function addCell(row, main, detail = '') {
  const cell = document.createElement('td'); const strong = document.createElement('strong'); strong.textContent = main; cell.append(strong);
  if (detail) { const small = document.createElement('small'); small.textContent = detail; cell.append(small); }
  row.append(cell); return cell;
}
function renderBatch(report) {
  batchUi('batch-results').hidden = false;
  const summary = report.summary;
  const running = report.status === 'running' || report.status === 'uploading';
  batchUi('batch-status').textContent = statusNames[report.status] || report.status;
  batchUi('run-description').textContent = `${report.config.model} · cửa sổ ${report.config.window_seconds} s · bước ${report.config.step_seconds} s · ${report.config.confirm_windows} lần xác nhận · ngưỡng ${report.config.threshold.toFixed(3)} · ${report.purpose === 'held_out' ? 'Tập giữ lại (do người dùng khai báo, chưa xác minh độc lập)' : 'Tập phát triển / phân tích lỗi'}${report.note ? ` · ${report.note}` : ''}`;
  const current = report.rows.find((row) => row.status === 'running');
  batchUi('batch-progress-text').textContent = `${summary.finished}/${summary.total} video kết thúc${current ? ` · ${current.filename}: ${current.message || 'Đang xử lý'}` : ''}`;
  batchUi('batch-progress').value = 100 * (summary.finished + (current?.progress || 0) / 100) / summary.total;
  batchUi('batch-cancel').hidden = !running;
  batchUi('batch-cancel').disabled = report.cancel_requested;
  batchUi('batch-cancel').textContent = report.cancel_requested ? 'Đã yêu cầu dừng sau video hiện tại' : 'Dừng sau video hiện tại';
  for (const format of ['json', 'csv']) {
    batchUi(`batch-${format}`).href = `/api/batches/${report.id}/download/${format}`;
    batchUi(`batch-${format}`).hidden = false;
  }
  batchUi('metric-coverage').textContent = `${summary.eligible}/${summary.total}`;
  batchUi('excluded-count').textContent = `${summary.excluded_or_pending} video chưa chấm / bị loại`;
  for (const key of ['TP', 'FP', 'FN', 'TN']) batchUi(`metric-${key}`).textContent = summary.confusion[key];
  batchUi('subset-metrics').textContent = `${running ? 'Tạm tính trên phần đã chạy' : 'Chỉ trên video đủ điều kiện'}: sensitivity ${percent(summary.sensitivity)} · specificity ${percent(summary.specificity)} · precision ${percent(summary.precision)} · accuracy ${percent(summary.accuracy)}. Độ trễ TB riêng các TP: ${seconds(summary.mean_detected_delay_seconds)}.`;
  batchUi('coverage-warning').textContent = `Độ bao phủ chấm điểm: ${percent(summary.coverage)}. Quan sát ${summary.observed_adl_alert_clips} clip ADL có cảnh báo (kể cả clip bị loại do thiếu pose). Không được lấy các tỷ lệ của phần đủ điều kiện làm độ chính xác toàn bộ. “—” là chưa có mẫu số, không phải 0%.`;
  batchUi('batch-result-rows').replaceChildren();
  for (const item of report.rows) {
    const row = document.createElement('tr');
    addCell(row, `${item.index + 1}. ${item.filename}`, `Nhãn thật: ${item.label === 'fall' ? 'FALL' : 'ADL'} · mốc ${seconds(item.onset_seconds)}`);
    addCell(row, statusNames[item.status] || item.status, item.status === 'running' ? `${item.progress}%` : '');
    addCell(row, item.has_alert === undefined ? 'Chưa có kết quả' : item.has_alert ? `Có · ${item.alert_count} đợt` : 'Không quan sát cảnh báo', item.has_alert ? `Đầu tiên: ${seconds(item.first_alert_seconds)}` : 'Không đồng nghĩa an toàn');
    addCell(row, `Độ trễ: ${seconds(item.delay_seconds)}`, `${item.matched_alert_seconds != null ? `Ghép mốc: ${seconds(item.matched_alert_seconds)}. ` : ''}${item.early_alerts ? `${item.early_alerts} đợt trước mốc. ` : ''}${item.onset_during_warmup ? 'Ngã trong giai đoạn tích lũy đầu.' : ''}`);
    addCell(row, item.pose_rate === undefined ? '—' : `Pose: ${percent(item.pose_rate)}`, item.window_count === undefined ? '' : `${item.valid_windows}/${item.window_count} cửa sổ đủ pose`);
    const cell = document.createElement('td'); const tag = document.createElement('span');
    tag.className = `outcome-tag ${['FP', 'FN'].includes(item.outcome) ? 'bad' : ['TP', 'TN', 'pending'].includes(item.outcome) ? '' : 'excluded'}`;
    tag.textContent = outcomeNames[item.outcome] || item.outcome;
    const reason = document.createElement('small'); reason.textContent = item.reason || '';
    cell.append(tag, reason); row.append(cell); batchUi('batch-result-rows').append(row);
  }
}
async function loadHistory() {
  try {
    const reports = await batchRequest('/api/batches');
    batchUi('batch-history').replaceChildren();
    if (!reports.length) batchUi('batch-history').textContent = 'Chưa có lượt đánh giá đã lưu.';
    for (const report of reports) {
      const link = document.createElement('a'); link.href = `/batch.html?batch=${report.id}`;
      link.textContent = `${new Date(report.created_at).toLocaleString('vi-VN')} · ${report.summary.total} video · ${statusNames[report.status] || report.status}`;
      const detail = document.createElement('small'); detail.textContent = `${report.note || 'Không có ghi chú'} · ${report.summary.eligible}/${report.summary.total} đủ điều kiện · ${report.purpose === 'development' ? 'Phân tích lỗi' : 'Tập giữ lại do người dùng khai báo'}`;
      link.append(detail); batchUi('batch-history').append(link);
    }
  } catch (error) { batchUi('batch-history').textContent = `Chưa đọc được lịch sử: ${error.message}`; }
}
async function pollBatch() {
  if (polling || !batchId) return;
  polling = true; batchUi('batch-retry').hidden = true;
  try {
    while (true) {
      const report = await batchRequest(`/api/batches/${batchId}`); renderBatch(report);
      if (!['running', 'uploading'].includes(report.status)) break;
      await new Promise((resolve) => setTimeout(resolve, 900));
    }
    await loadHistory();
  } catch (error) { batchError(`Mất kết nối tiến độ: ${error.message}. Lượt đã gửi có thể vẫn đang chạy; không cần gửi lại video.`); batchUi('batch-retry').hidden = false; }
  finally { polling = false; }
}
batchUi('batch-run').addEventListener('click', async () => {
  if (locked || !drafts.length) return;
  batchError('');
  if (drafts.some((draft) => !draft.label || (draft.onset !== null && (!Number.isFinite(draft.onset) || draft.onset < 0)))) return batchError('Chọn nhãn thật cho mọi video và kiểm tra mốc ngã không âm. Có thể để mốc trống nếu chưa xác nhận.');
  lockEditor(true); batchUi('batch-results').hidden = false;
  batchUi('batch-progress-text').textContent = 'Đang gửi các video tới backend trên máy…';
  const form = new FormData();
  for (const draft of drafts) form.append('files', draft.file);
  form.append('annotations', JSON.stringify(drafts.map((draft) => ({ filename: draft.file.name, label: draft.label, onset_seconds: draft.onset }))));
  form.append('window_seconds', batchUi('batch-window').value); form.append('purpose', batchUi('batch-purpose').value); form.append('note', batchUi('batch-note').value);
  try {
    const created = await batchRequest('/api/batches', { method: 'POST', headers: { 'X-PoseLab': 'local' }, body: form });
    batchId = created.id; history.replaceState(null, '', `?batch=${batchId}`); await pollBatch();
  } catch (error) { batchError(error.message); lockEditor(false); batchUi('batch-results').hidden = true; }
});
batchUi('batch-cancel').addEventListener('click', async () => {
  try { await batchRequest(`/api/batches/${batchId}/cancel`, { method: 'POST', headers: { 'X-PoseLab': 'local' } }); batchUi('batch-cancel').disabled = true; batchUi('batch-cancel').textContent = 'Đã yêu cầu dừng sau video hiện tại'; }
  catch (error) { batchError(error.message); }
});
batchUi('batch-retry').addEventListener('click', () => { batchError(''); pollBatch(); });
(async () => {
  try {
    config = await batchRequest('/api/config');
    batchUi('fixed-config').textContent = `${config.model} · ngưỡng ${config.threshold.toFixed(3)} không chỉnh ở đây · bước 0,5 s · xác nhận 2 cửa sổ liên tiếp. Tham số chưa được kiểm định cho triển khai thực tế.`;
    lockEditor(false);
    const previous = new URLSearchParams(location.search).get('batch');
    if (previous && /^[a-f0-9]{32}$/.test(previous)) { batchId = previous; batchUi('batch-editor').hidden = true; await pollBatch(); }
    await loadHistory();
  } catch (error) { batchError(`Chưa kết nối backend: ${error.message}`); }
})();
