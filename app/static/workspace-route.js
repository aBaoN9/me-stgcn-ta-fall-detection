'use strict';
(() => {
  if (!['/', '/index.html'].includes(location.pathname)) return;
  if (['#history', '#model', '#settings'].includes(location.hash)) return;
  location.replace('/analysis.html' + location.search + (location.hash === '#main' ? '#main' : ''));
})();
