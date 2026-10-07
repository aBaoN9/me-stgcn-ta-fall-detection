'use strict';
(() => {
  const icons = {
    shield: '<path d="m12 3 8 3v6c0 5-8 9-8 9s-8-4-8-9V6l8-3Z"/><path d="m8 12 3 3 5-6"/>',
    grid: '<rect x="3" y="3" width="7" height="7" rx="2"/><rect x="14" y="3" width="7" height="7" rx="2"/><rect x="3" y="14" width="7" height="7" rx="2"/><rect x="14" y="14" width="7" height="7" rx="2"/>',
    video: '<rect x="3" y="5" width="13" height="14" rx="3"/><path d="m16 9 5-3v12l-5-3"/>',
    history: '<path d="M3 11a9 9 0 1 1 3 8M3 4v7h7M12 7v5l3 2"/>',
    model: '<path d="m12 3 9 5-9 5-9-5 9-5ZM3 12l9 5 9-5M3 16l9 5 9-5"/>',
    settings: '<path d="M4 7h16M4 17h16"/><circle cx="9" cy="7" r="3"/><circle cx="15" cy="17" r="3"/>',
    batch: '<rect x="3" y="3" width="14" height="14" rx="3"/><path d="M7 21h11a3 3 0 0 0 3-3V7M7 8h6M7 12h4"/>'
  };
  const icon = (name) => '<svg class="icon" viewBox="0 0 24 24" aria-hidden="true">' + icons[name] + '</svg>';
  const dashboard = location.pathname === '/' || location.pathname === '/index.html';
  const routes = [
    ['analysis', 'Phân tích video', 'video', '/analysis.html'],
    ['history', 'Lịch sử cảnh báo', 'history', dashboard ? '#history' : '/#history'],
    ['batch', 'Xử lý nhiều video', 'batch', '/batch.html'],
    ['model', 'Mô hình', 'model', dashboard ? '#model' : '/#model'],
    ['settings', 'Cài đặt', 'settings', dashboard ? '#settings' : '/#settings']
  ];
  const active = document.body.dataset.page || 'overview';
  const parameters = new URLSearchParams(location.search);
  const session = parameters.get('session') || parameters.get('job');
  if (session && /^[a-f0-9]{32}$/.test(session)) routes[0][3] += '?' + (parameters.has('session') ? 'session=' : 'job=') + session;
  const navigation = document.querySelector('[data-navigation]');
  if (navigation) navigation.innerHTML = '<a href="/" class="brand">' + '<span class="brand-logo">' + icon('shield') + '</span><span>PoseLab<small>FALL DETECTION</small></span></a><nav aria-label="Điều hướng chính">' + routes.map(([key, label, symbol, href]) => '<a href="' + href + '" class="nav-item' + (key === active ? ' active' : '') + '"' + (dashboard && ['overview', 'history', 'model', 'settings'].includes(key) ? ' data-view="' + key + '"' : '') + (key === active ? ' aria-current="page"' : '') + '>' + icon(symbol) + '<span>' + label + '</span></a>').join('') + '</nav><div class="sidebar-bottom">' + icon('shield') + '<span>Xử lý cục bộ</span></div>';
  if (navigation) navigation.addEventListener('click', (event) => {
    const link = event.target.closest('a.nav-item');
    if (!link) return;
    const target = new URL(link.href, location.origin);
    if (target.pathname !== '/' || !target.hash) return;
    const current = new URLSearchParams(location.search);
    const identifier = current.get('session') || current.get('job');
    target.search = '';
    if (identifier && /^[a-f0-9]{32}$/.test(identifier)) target.searchParams.set(current.has('session') ? 'session' : 'job', identifier);
    link.href = target.href;
  });
  const analysisLink = navigation?.querySelector('a[href^="/analysis.html"]');
  if (analysisLink) analysisLink.addEventListener('click', () => {
    const current = new URLSearchParams(location.search);
    const identifier = current.get('session') || current.get('job');
    analysisLink.href = '/analysis.html' + (identifier && /^[a-f0-9]{32}$/.test(identifier) ? '?' + (current.has('session') ? 'session=' : 'job=') + identifier : '');
  });
  const titles = {analysis: 'Phân tích video', batch: 'Xử lý nhiều video'};
  const topbar = document.querySelector('[data-topbar]');
  if (topbar) topbar.innerHTML = '<div class="header-context"><span class="header-product">HỆ THỐNG CẢNH BÁO TÉ NGÃ</span><strong id="workspace-title">' + (titles[active] || 'PoseLab') + '</strong></div><div class="topbar-right"><span class="model-chip">ME-STGCN-TA</span><span id="connection" class="connection"><span class="status-dot"></span>Đang kết nối</span><time id="clock"></time></div>';
  const clock = document.getElementById('clock');
  if (clock) { const update = () => { clock.textContent = new Date().toLocaleTimeString('vi-VN'); }; update(); setInterval(update, 1000); }
  if (!dashboard) fetch('/api/config').then((response) => { if (!response.ok) throw new Error(); const status = document.getElementById('connection'); status.innerHTML = '<span class="status-dot"></span>Đã kết nối'; }).catch(() => { const status = document.getElementById('connection'); status.classList.add('offline'); status.innerHTML = '<span class="status-dot"></span>Mất kết nối'; });
  const motion = matchMedia('(prefers-reduced-motion: reduce)');
  const scores = new WeakMap();
  function animateScore(number, fill, value, formatter) {
    if (!number || !fill) return;
    let state = scores.get(number);
    if (!state) { state = { value: 0, target: null, frame: null }; scores.set(number, state); }

    const target = Number.isFinite(value) ? Math.max(0, Math.min(1, value)) : null;
    if (state.target === target && (state.frame !== null || target !== null)) return;
    if (state.frame !== null) cancelAnimationFrame(state.frame);
    state.target = target;
    number.dataset.score = target === null ? '' : String(target);
    number.setAttribute('aria-label', target === null ? 'Chưa có điểm Fall' : 'Điểm Fall ' + formatter(target));
    number.title = target === null ? '' : formatter(target);
    fill.style.width = '100%';
    if (target === null) { state.value = 0; state.frame = null; number.textContent = '—'; fill.style.transform = 'scaleX(0)'; return; }
    const startValue = state.value;
    const started = performance.now();
    const duration = motion.matches ? 0 : 480;
    const draw = (now) => {
      const progress = duration ? Math.min(1, (now - started) / duration) : 1;
      state.value = startValue + (target - startValue) * (1 - Math.pow(1 - progress, 3));
      number.textContent = formatter(state.value);
      fill.style.transform = 'scaleX(' + state.value + ')';
      state.frame = progress < 1 ? requestAnimationFrame(draw) : null;
    };
    state.frame = requestAnimationFrame(draw);
  }
  for (const inspector of document.querySelectorAll('[data-inspector]')) {
    const buttons = [...inspector.querySelectorAll('[role=tab]')];
    function select(button) {
      for (const candidate of buttons) { const selected = candidate === button; candidate.setAttribute('aria-selected', String(selected)); candidate.tabIndex = selected ? 0 : -1; }
      for (const pane of inspector.querySelectorAll('[data-pane]')) pane.hidden = pane.dataset.pane !== button.dataset.tab;
    }
    buttons.forEach((button, index) => {
      button.addEventListener('click', () => select(button));
      button.addEventListener('keydown', (event) => { if (!['ArrowLeft', 'ArrowRight', 'Home', 'End'].includes(event.key)) return; event.preventDefault(); const nextIndex = event.key === 'Home' ? 0 : event.key === 'End' ? buttons.length - 1 : (index + (event.key === 'ArrowRight' ? 1 : -1) + buttons.length) % buttons.length; select(buttons[nextIndex]); buttons[nextIndex].focus(); });
    });
    if (buttons.length) select(buttons[0]);
  }
  window.PoseUI = { animateScore };
})();
