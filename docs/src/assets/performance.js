/**
 * OxiBLAS Interactive Performance Dashboard
 * Zero-dependency client-side data explorer for Documenter.jl
 */

(function () {
  'use strict';

  function initDashboard() {
    const rootEl = document.getElementById('oxiblas-dashboard-app');
    if (!rootEl) return;

    // Show initial loading state
    rootEl.innerHTML = `
      <div class="ob-loading">
        <div class="ob-loading-spinner"></div>
        <div style="font-weight: 600; font-size: 1.05rem;">Loading OxiBLAS Interactive Benchmark Explorer...</div>
        <div style="font-size: 0.85rem; color: var(--ob-text-muted); margin-top: 0.4rem;">
          Fetching verified snapshot data (2,700+ benchmark cases across 1 and 17 threads)...
        </div>
      </div>
    `;

    // Candidate paths for results.json across various Documenter URL schemes
    const docBase = (typeof window !== 'undefined' && window.documenterBaseURL) ? window.documenterBaseURL : '..';
    const customSrc = rootEl.getAttribute('data-results-src');
    const candidates = [
      docBase + '/assets/benchmarks/results.json',
      customSrc,
      '../assets/benchmarks/results.json',
      '../../assets/benchmarks/results.json',
      'assets/benchmarks/results.json',
      '/assets/benchmarks/results.json'
    ].filter(Boolean);

    tryFetchCandidates(candidates, 0, (err, data) => {
      if (err || !data) {
        rootEl.innerHTML = `
          <div class="ob-error">
            <h3 style="color: var(--ob-danger); margin: 0 0 0.5rem 0;">Unable to Load Benchmark Dataset</h3>
            <p style="color: var(--ob-text-muted); font-size: 0.9rem;">
              Could not retrieve <code>results.json</code> from the documentation assets directory.
            </p>
            <button class="ob-reset-btn" onclick="location.reload()" style="margin-top: 0.5rem;">Retry Loading</button>
          </div>
        `;
        return;
      }

      mountDashboard(rootEl, data);
    });
  }

  function tryFetchCandidates(urls, index, cb) {
    if (index >= urls.length) {
      cb(new Error('All candidates failed'));
      return;
    }
    const rawUrl = urls[index];
    const url = rawUrl + (rawUrl.includes('?') ? '&' : '?') + 't=' + Date.now();
    fetch(url, { cache: 'no-store' })
      .then(r => {
        if (!r.ok) throw new Error('HTTP ' + r.status);
        return r.json();
      })
      .then(data => cb(null, data))
      .catch(() => tryFetchCandidates(urls, index + 1, cb));
  }

  function escapeHtml(str) {
    if (!str) return '';
    return String(str)
      .replace(/&/g, '&amp;')
      .replace(/</g, '&lt;')
      .replace(/>/g, '&gt;')
      .replace(/"/g, '&quot;')
      .replace(/'/g, '&#039;');
  }

  function formatMs(ms) {
    if (ms === null || isNaN(ms) || ms === undefined) return '—';
    if (ms < 0.001) return (ms * 1e6).toFixed(1) + ' ns';
    if (ms < 1.0) return (ms * 1e3).toFixed(1) + ' µs';
    if (ms >= 1000) return (ms / 1000).toFixed(2) + ' s';
    return ms.toFixed(ms < 10 ? 3 : 2) + ' ms';
  }

  function formatOrder(n) {
    if (n === null || isNaN(n)) return '—';
    return Number(n).toLocaleString();
  }

  function mountDashboard(container, data) {
    const snapshot = data.snapshot || {};
    const threadsAvailable = snapshot.threads || [1];

    // Dashboard state
    const state = {
      activeThreads: threadsAvailable.includes(17) ? 17 : threadsAvailable[0],
      search: '',
      family: '',
      op: '',
      input: '',
      eltype: '',
      size: '',
      comparison: 'all', // 'all', 'with_ref', 'faster', 'no_ref'
      sortKey: 'ratio',
      sortDir: 'asc',
      page: 1,
      pageSize: 50
    };

    // Render skeleton
    container.innerHTML = `
      <div class="ob-header">
        <div class="ob-header-top">
          <div class="ob-title-group">
            <h2>
              <span>${escapeHtml(snapshot.title || 'Benchmark Dashboard')}</span>
              <span class="ob-badge-rev">rev: ${escapeHtml(snapshot.revision || 'main')}</span>
            </h2>
          </div>
          <div class="ob-thread-switcher" id="ob-thread-controls"></div>
        </div>
        <div class="ob-env-badges">
          <span class="ob-env-badge"><strong>CPU:</strong> ${escapeHtml(snapshot.cpu || 'Apple Silicon')}</span>
          <span class="ob-env-badge"><strong>Julia:</strong> v${escapeHtml(snapshot.julia_version || '1.13')}</span>
          <span class="ob-env-badge"><strong>Reference:</strong> ${escapeHtml(snapshot.reference_backend || 'OpenBLAS')}</span>
          <span class="ob-env-badge"><strong>Updated:</strong> ${escapeHtml(snapshot.generated_at ? snapshot.generated_at.replace('T', ' ') : 'Snapshot')}</span>
        </div>
      </div>

      <div class="ob-kpi-grid" id="ob-kpi-container"></div>

      <div class="ob-charts-grid">
        <div class="ob-chart-card">
          <div class="ob-chart-title">Ratio Distribution (Speedup Buckets)</div>
          <div class="ob-chart-desc">OxiBLAS Time / Reference Time (below 1.0 = OxiBLAS is faster)</div>
          <div class="ob-svg-container" id="ob-chart-dist"></div>
        </div>
        <div class="ob-chart-card">
          <div class="ob-chart-title">Scaling Across Matrix Sizes</div>
          <div class="ob-chart-desc">Median performance ratio by dimension (Small, Medium, Large)</div>
          <div class="ob-svg-container" id="ob-chart-sizes"></div>
        </div>
        <div class="ob-chart-card">
          <div class="ob-chart-title">Top Speedups in Active Filter</div>
          <div class="ob-chart-desc">Largest relative speedups achieved against Reference BLAS/SuiteSparse</div>
          <div class="ob-svg-container" id="ob-chart-top"></div>
        </div>
      </div>

      <div class="ob-filters">
        <div class="ob-filters-row">
          <div class="ob-search-box">
            <label class="ob-filter-label" for="ob-filter-search">Search Operations & Inputs</label>
            <input type="text" id="ob-filter-search" class="ob-input" placeholder="Filter by name (e.g. csr, mul!, f64, qr)..." />
          </div>
          <div class="ob-filter-group">
            <label class="ob-filter-label" for="ob-filter-family">Family</label>
            <select id="ob-filter-family" class="ob-select"><option value="">All Families</option></select>
          </div>
          <div class="ob-filter-group">
            <label class="ob-filter-label" for="ob-filter-op">Operation</label>
            <select id="ob-filter-op" class="ob-select"><option value="">All Operations</option></select>
          </div>
          <div class="ob-filter-group">
            <label class="ob-filter-label" for="ob-filter-input">Input Kind</label>
            <select id="ob-filter-input" class="ob-select"><option value="">All Inputs</option></select>
          </div>
          <div class="ob-filter-group">
            <label class="ob-filter-label" for="ob-filter-eltype">Numeric Type</label>
            <select id="ob-filter-eltype" class="ob-select"><option value="">All Types</option></select>
          </div>
          <div class="ob-filter-group">
            <label class="ob-filter-label" for="ob-filter-size">Size</label>
            <select id="ob-filter-size" class="ob-select"><option value="">All Sizes</option></select>
          </div>
          <div class="ob-filter-group" style="flex: 0 1 170px;">
            <label class="ob-filter-label" for="ob-filter-comp">Comparison</label>
            <select id="ob-filter-comp" class="ob-select">
              <option value="all">All Cases</option>
              <option value="with_ref">With Reference</option>
              <option value="faster">OxiBLAS Faster (&lt;1.0)</option>
              <option value="no_ref">Missing Reference</option>
            </select>
          </div>
          <button class="ob-reset-btn" id="ob-reset-filters">Reset Filters</button>
        </div>
        <div class="ob-filter-status-bar">
          <div id="ob-filter-counter">Showing — cases</div>
          <div id="ob-filter-geo">Geomean: —</div>
        </div>
      </div>

      <div class="ob-table-card">
        <div class="ob-table-wrapper">
          <table class="ob-table" id="ob-data-table">
            <thead>
              <tr>
                <th data-sort="name">Case Name</th>
                <th data-sort="family">Family</th>
                <th data-sort="form">Form</th>
                <th data-sort="eltype">Type</th>
                <th data-sort="order">Order (N)</th>
                <th data-sort="oxiblas_ms">OxiBLAS Time</th>
                <th data-sort="reference_ms">Ref Time</th>
                <th data-sort="ratio">Ratio (Oxi / Ref)</th>
                <th data-sort="oxiblas_allocs">Allocs (Oxi / Ref)</th>
                <th data-sort="status">Status</th>
              </tr>
            </thead>
            <tbody id="ob-table-body"></tbody>
          </table>
        </div>
        <div class="ob-pagination-bar">
          <div>
            Show
            <select id="ob-page-size" class="ob-select" style="width: auto; display: inline-block; padding: 0.2rem 0.5rem; margin: 0 0.3rem;">
              <option value="25">25</option>
              <option value="50" selected>50</option>
              <option value="100">100</option>
              <option value="200">200</option>
            </select>
            cases per page
          </div>
          <div id="ob-page-info">Page 1 of 1</div>
          <div class="ob-page-nav">
            <button class="ob-page-btn" id="ob-btn-first">« First</button>
            <button class="ob-page-btn" id="ob-btn-prev">‹ Prev</button>
            <button class="ob-page-btn" id="ob-btn-next">Next ›</button>
            <button class="ob-page-btn" id="ob-btn-last">Last »</button>
          </div>
        </div>
      </div>
    `;

    // Populate dropdown options
    const filters = data.filters || {};
    populateSelect('ob-filter-family', filters.families || []);
    populateSelect('ob-filter-op', filters.operations || []);
    populateSelect('ob-filter-input', filters.inputs || []);
    populateSelect('ob-filter-eltype', filters.eltypes || []);
    populateSelect('ob-filter-size', filters.sizes || []);

    // Thread Switcher buttons
    const threadControls = document.getElementById('ob-thread-controls');
    threadControls.innerHTML = '';
    threadsAvailable.forEach(t => {
      const btn = document.createElement('button');
      btn.className = 'ob-thread-btn' + (t === state.activeThreads ? ' active' : '');
      btn.textContent = t + (t === 1 ? ' Thread' : ' Threads');
      btn.addEventListener('click', () => {
        if (state.activeThreads === t) return;
        state.activeThreads = t;
        state.page = 1;
        threadControls.querySelectorAll('.ob-thread-btn').forEach(b => b.classList.remove('active'));
        btn.classList.add('active');
        render();
      });
      threadControls.appendChild(btn);
    });

    // Wire Filter Events
    const searchInput = document.getElementById('ob-filter-search');
    let searchDebounce = null;
    searchInput.addEventListener('input', e => {
      clearTimeout(searchDebounce);
      searchDebounce = setTimeout(() => {
        state.search = e.target.value.trim().toLowerCase();
        state.page = 1;
        render();
      }, 150);
    });

    ['family', 'op', 'input', 'eltype', 'size', 'comp'].forEach(key => {
      const el = document.getElementById('ob-filter-' + key);
      el.addEventListener('change', e => {
        if (key === 'comp') state.comparison = e.target.value;
        else state[key] = e.target.value;
        state.page = 1;
        render();
      });
    });

    document.getElementById('ob-reset-filters').addEventListener('click', () => {
      state.search = '';
      state.family = '';
      state.op = '';
      state.input = '';
      state.eltype = '';
      state.size = '';
      state.comparison = 'all';
      state.page = 1;
      searchInput.value = '';
      ['family', 'op', 'input', 'eltype', 'size'].forEach(k => {
        document.getElementById('ob-filter-' + k).value = '';
      });
      document.getElementById('ob-filter-comp').value = 'all';
      render();
    });

    // Wire Table Header Sorting
    document.querySelectorAll('#ob-data-table th[data-sort]').forEach(th => {
      th.addEventListener('click', () => {
        const key = th.getAttribute('data-sort');
        if (state.sortKey === key) {
          state.sortDir = state.sortDir === 'asc' ? 'desc' : 'asc';
        } else {
          state.sortKey = key;
          state.sortDir = key === 'ratio' ? 'asc' : (key === 'name' ? 'asc' : 'desc');
        }
        render();
      });
    });

    // Wire Pagination
    document.getElementById('ob-page-size').addEventListener('change', e => {
      state.pageSize = parseInt(e.target.value, 10);
      state.page = 1;
      render();
    });

    document.getElementById('ob-btn-first').addEventListener('click', () => { state.page = 1; render(); });
    document.getElementById('ob-btn-prev').addEventListener('click', () => { if (state.page > 1) { state.page--; render(); } });
    document.getElementById('ob-btn-next').addEventListener('click', () => { state.page++; render(); });
    document.getElementById('ob-btn-last').addEventListener('click', () => { state.page = state.maxPage || 1; render(); });

    // Initial render
    render();

    function render() {
      const allCases = (data.cases && data.cases[String(state.activeThreads)]) || [];
      const filtered = filterCases(allCases, state);
      sortCases(filtered, state.sortKey, state.sortDir);

      // Render KPIs
      renderKPIs(allCases, filtered, state.activeThreads, data.summaries);

      // Render Charts
      renderDistChart(filtered);
      renderSizeChart(filtered);
      renderTopChart(filtered);

      // Render Table & Pagination
      renderTable(filtered, state);
    }
  }

  function populateSelect(id, items) {
    const el = document.getElementById(id);
    items.forEach(it => {
      const opt = document.createElement('option');
      opt.value = it;
      opt.textContent = it;
      el.appendChild(opt);
    });
  }

  function filterCases(cases, state) {
    const s = state.search;
    const fam = state.family;
    const op = state.op;
    const inp = state.input;
    const eltype = state.eltype;
    const sz = state.size;
    const comp = state.comparison;

    return cases.filter(c => {
      if (fam && c.family !== fam) return false;
      if (op && c.op !== op) return false;
      if (inp && c.input !== inp) return false;
      if (eltype && c.eltype !== eltype) return false;
      if (sz && c.size !== sz) return false;

      const hasRef = c.status === 'ok' && c.ratio !== null && !isNaN(c.ratio) && c.ratio > 0;
      if (comp === 'with_ref' && !hasRef) return false;
      if (comp === 'faster' && (!hasRef || c.ratio >= 1.0)) return false;
      if (comp === 'no_ref' && hasRef) return false;

      if (s) {
        const full = (c.name + ' ' + c.op + ' ' + c.input + ' ' + c.eltype + ' ' + (c.note || '')).toLowerCase();
        if (!full.includes(s)) return false;
      }
      return true;
    });
  }

  function sortCases(cases, key, dir) {
    cases.sort((a, b) => {
      let valA = a[key];
      let valB = b[key];

      // Handle nulls / NaNs
      const aNull = valA === null || valA === undefined || (typeof valA === 'number' && isNaN(valA));
      const bNull = valB === null || valB === undefined || (typeof valB === 'number' && isNaN(valB));
      if (aNull && bNull) return 0;
      if (aNull) return 1;
      if (bNull) return -1;

      if (typeof valA === 'string') {
        valA = valA.toLowerCase();
        valB = valB.toLowerCase();
      }

      if (valA < valB) return dir === 'asc' ? -1 : 1;
      if (valA > valB) return dir === 'asc' ? 1 : -1;
      return 0;
    });
  }

  function renderKPIs(allCases, filtered, threads, summaries) {
    const kpiContainer = document.getElementById('ob-kpi-container');
    const compFiltered = filtered.filter(c => c.status === 'ok' && c.ratio !== null && !isNaN(c.ratio) && c.ratio > 0);
    const ratios = compFiltered.map(c => c.ratio);

    let geomean = NaN;
    let median = NaN;
    let fasterCount = 0;

    if (ratios.length > 0) {
      const sumLog = ratios.reduce((acc, r) => acc + Math.log(r), 0);
      geomean = Math.exp(sumLog / ratios.length);
      const sorted = [...ratios].sort((a, b) => a - b);
      const mid = Math.floor(sorted.length / 2);
      median = sorted.length % 2 !== 0 ? sorted[mid] : (sorted[mid - 1] + sorted[mid]) / 2;
      fasterCount = ratios.filter(r => r < 1.0).length;
    }

    const fasterPct = ratios.length > 0 ? ((fasterCount / ratios.length) * 100).toFixed(1) : '0';

    kpiContainer.innerHTML = `
      <div class="ob-kpi-card">
        <div class="ob-kpi-label">Active Filtered Cases</div>
        <div class="ob-kpi-value">${filtered.length.toLocaleString()}</div>
        <div class="ob-kpi-sub">out of ${allCases.length.toLocaleString()} total</div>
      </div>
      <div class="ob-kpi-card">
        <div class="ob-kpi-label">Comparable vs Ref</div>
        <div class="ob-kpi-value">${compFiltered.length.toLocaleString()}</div>
        <div class="ob-kpi-sub">${((compFiltered.length / (filtered.length || 1)) * 100).toFixed(0)}% of active cases</div>
      </div>
      <div class="ob-kpi-card highlight">
        <div class="ob-kpi-label">Geometric Mean Ratio</div>
        <div class="ob-kpi-value" style="color: ${geomean < 1.0 ? 'var(--ob-success)' : 'var(--ob-text)'};">
          ${isNaN(geomean) ? '—' : geomean.toFixed(2) + '×'}
        </div>
        <div class="ob-kpi-sub">${geomean < 1.0 ? ((1 / geomean).toFixed(1) + '× faster than ref') : 'across compared cases'}</div>
      </div>
      <div class="ob-kpi-card">
        <div class="ob-kpi-label">Median Ratio</div>
        <div class="ob-kpi-value" style="color: ${median < 1.0 ? 'var(--ob-success)' : 'var(--ob-text)'};">
          ${isNaN(median) ? '—' : median.toFixed(2) + '×'}
        </div>
        <div class="ob-kpi-sub">${median < 1.0 ? ((1 / median).toFixed(1) + '× faster (50th percentile)') : '50th percentile'}</div>
      </div>
      <div class="ob-kpi-card">
        <div class="ob-kpi-label">OxiBLAS Faster</div>
        <div class="ob-kpi-value" style="color: var(--ob-success);">
          ${fasterCount.toLocaleString()}
        </div>
        <div class="ob-kpi-sub">${fasterPct}% of compared cases</div>
      </div>
    `;

    // Filter status bar text
    document.getElementById('ob-filter-counter').textContent =
      `Showing ${filtered.length.toLocaleString()} of ${allCases.length.toLocaleString()} cases (${compFiltered.length.toLocaleString()} with reference)`;
    document.getElementById('ob-filter-geo').textContent =
      isNaN(geomean) ? 'No valid reference ratios in filter' : `Filter Geomean: ${geomean.toFixed(2)}× (Median: ${median.toFixed(2)}×)`;
  }

  // Native SVG Chart 1: Ratio Distribution
  function renderDistChart(cases) {
    const container = document.getElementById('ob-chart-dist');
    const comp = cases.filter(c => c.status === 'ok' && c.ratio !== null && !isNaN(c.ratio) && c.ratio > 0);
    if (comp.length === 0) {
      container.innerHTML = '<div style="padding: 2rem; text-align: center; color: var(--ob-text-muted); font-size: 0.85rem;">No comparable cases in current filter</div>';
      return;
    }

    const buckets = [
      { label: '<0.5×', desc: '>2× faster', color: '#15803d', count: 0 },
      { label: '0.5–0.8×', desc: '1.25–2× faster', color: '#22c55e', count: 0 },
      { label: '0.8–1.0×', desc: 'faster', color: '#4ade80', count: 0 },
      { label: '1.0–1.25×', desc: 'similar', color: '#94a3b8', count: 0 },
      { label: '1.25–2.0×', desc: 'ref faster', color: '#f59e0b', count: 0 },
      { label: '>2.0×', desc: 'ref >2× faster', color: '#ef4444', count: 0 }
    ];

    comp.forEach(c => {
      const r = c.ratio;
      if (r < 0.5) buckets[0].count++;
      else if (r < 0.8) buckets[1].count++;
      else if (r < 1.0) buckets[2].count++;
      else if (r <= 1.25) buckets[3].count++;
      else if (r <= 2.0) buckets[4].count++;
      else buckets[5].count++;
    });

    const maxCount = Math.max(...buckets.map(b => b.count), 1);
    const width = 460;
    const height = 170;
    const barW = 54;
    const gap = 18;
    const startX = 20;
    const baseY = 135;
    const chartH = 100;

    let svg = `<svg viewBox="0 0 ${width} ${height}" style="width: 100%; height: auto; display: block;" xmlns="http://www.w3.org/2000/svg">`;

    // Baseline
    svg += `<line x1="${startX - 5}" y1="${baseY}" x2="${startX + buckets.length * (barW + gap)}" y2="${baseY}" stroke="var(--ob-border)" stroke-width="1.5" />`;

    buckets.forEach((b, i) => {
      const x = startX + i * (barW + gap);
      const h = (b.count / maxCount) * chartH;
      const y = baseY - h;
      const pct = ((b.count / comp.length) * 100).toFixed(0);

      // Bar
      svg += `
        <rect x="${x}" y="${y}" width="${barW}" height="${Math.max(h, 2)}" rx="4" fill="${b.color}">
          <title>${b.label} (${b.desc}): ${b.count} cases (${pct}%)</title>
        </rect>
        <text x="${x + barW / 2}" y="${y - 6}" font-size="11" font-weight="700" fill="var(--ob-text)" text-anchor="middle">${b.count}</text>
        <text x="${x + barW / 2}" y="${baseY + 16}" font-size="11" font-weight="600" fill="var(--ob-text)" text-anchor="middle">${b.label}</text>
        <text x="${x + barW / 2}" y="${baseY + 28}" font-size="9" fill="var(--ob-text-muted)" text-anchor="middle">${pct}%</text>
      `;
    });

    svg += '</svg>';
    container.innerHTML = svg;
  }

  // Native SVG Chart 2: Scaling Across Sizes (s, m, l)
  function renderSizeChart(cases) {
    const container = document.getElementById('ob-chart-sizes');
    const sizes = ['s', 'm', 'l'];
    const sizeLabels = { s: 'Small (s)', m: 'Medium (m)', l: 'Large (l)' };

    const dataPoints = sizes.map(sz => {
      const sub = cases.filter(c => c.size === sz && c.status === 'ok' && c.ratio !== null && !isNaN(c.ratio) && c.ratio > 0);
      if (sub.length === 0) return { size: sz, label: sizeLabels[sz], median: NaN, count: 0 };
      const sorted = sub.map(c => c.ratio).sort((a, b) => a - b);
      const mid = Math.floor(sorted.length / 2);
      const med = sorted.length % 2 !== 0 ? sorted[mid] : (sorted[mid - 1] + sorted[mid]) / 2;
      return { size: sz, label: sizeLabels[sz], median: med, count: sub.length };
    });

    const validPoints = dataPoints.filter(d => !isNaN(d.median));
    if (validPoints.length === 0) {
      container.innerHTML = '<div style="padding: 2rem; text-align: center; color: var(--ob-text-muted); font-size: 0.85rem;">No comparable cases in current filter</div>';
      return;
    }

    const width = 460;
    const height = 170;
    const barH = 26;
    const startY = 22;
    const gapY = 44;
    const leftX = 100;
    const maxBarW = 240;

    let svg = `<svg viewBox="0 0 ${width} ${height}" style="width: 100%; height: auto; display: block;" xmlns="http://www.w3.org/2000/svg">`;

    dataPoints.forEach((d, i) => {
      const y = startY + i * gapY;
      const isFaster = !isNaN(d.median) && d.median < 1.0;
      const barColor = isNaN(d.median) ? 'var(--ob-border)' : (isFaster ? '#16a34a' : '#f59e0b');
      // Visual scaling: ratio 1.0 maps to 120px width
      const normW = isNaN(d.median) ? 0 : Math.min(Math.max(d.median * 120, 10), maxBarW);

      svg += `
        <text x="${leftX - 10}" y="${y + 17}" font-size="11" font-weight="600" fill="var(--ob-text)" text-anchor="end">${d.label}</text>
        <rect x="${leftX}" y="${y}" width="${maxBarW}" height="${barH}" rx="5" fill="var(--ob-bg-subtle)" stroke="var(--ob-border)" />
        ${!isNaN(d.median) ? `<rect x="${leftX}" y="${y}" width="${normW}" height="${barH}" rx="5" fill="${barColor}"><title>Median ratio: ${d.median.toFixed(2)}×</title></rect>` : ''}
        <line x1="${leftX + 120}" y1="${y - 4}" x2="${leftX + 120}" y2="${y + barH + 4}" stroke="var(--ob-text-muted)" stroke-dasharray="2,2" stroke-width="1" />
        <text x="${leftX + normW + 8}" y="${y + 17}" font-size="11" font-weight="700" fill="var(--ob-text)">
          ${isNaN(d.median) ? '—' : `${d.median.toFixed(2)}× ${isFaster ? `(${(1 / d.median).toFixed(1)}× faster)` : ''} (${d.count} cases)`}
        </text>
      `;
    });

    // Reference mark legend
    svg += `
      <text x="${leftX + 120}" y="${height - 8}" font-size="9" fill="var(--ob-text-muted)" text-anchor="middle">1.0× Reference Parity</text>
    `;

    svg += '</svg>';
    container.innerHTML = svg;
  }

  // Native SVG Chart 3: Top Speedups
  function renderTopChart(cases) {
    const container = document.getElementById('ob-chart-top');
    const comp = cases.filter(c => c.status === 'ok' && c.ratio !== null && !isNaN(c.ratio) && c.ratio > 0);
    comp.sort((a, b) => a.ratio - b.ratio);
    const top = comp.slice(0, 5);

    if (top.length === 0) {
      container.innerHTML = '<div style="padding: 2rem; text-align: center; color: var(--ob-text-muted); font-size: 0.85rem;">No comparable cases in current filter</div>';
      return;
    }

    const width = 460;
    const height = 170;
    const barH = 18;
    const startY = 12;
    const gapY = 30;
    const leftX = 170;
    const maxBarW = 180;
    const minRatio = top[0].ratio;
    const maxSpeedup = Math.max(1 / minRatio, 2);

    let svg = `<svg viewBox="0 0 ${width} ${height}" style="width: 100%; height: auto; display: block;" xmlns="http://www.w3.org/2000/svg">`;

    top.forEach((c, i) => {
      const y = startY + i * gapY;
      const speedup = 1 / c.ratio;
      const w = Math.min((speedup / maxSpeedup) * maxBarW, maxBarW);
      const shortName = c.name.length > 24 ? c.name.slice(0, 22) + '…' : c.name;

      svg += `
        <text x="${leftX - 8}" y="${y + 13}" font-size="10.5" font-family="var(--ob-font-mono)" font-weight="600" fill="var(--ob-primary)" text-anchor="end">
          <title>${escapeHtml(c.name)}</title>${escapeHtml(shortName)}
        </text>
        <rect x="${leftX}" y="${y}" width="${Math.max(w, 8)}" height="${barH}" rx="4" fill="#16a34a">
          <title>${escapeHtml(c.name)}: ${speedup.toFixed(1)}× faster (${c.ratio.toFixed(3)} ratio)</title>
        </rect>
        <text x="${leftX + w + 6}" y="${y + 13}" font-size="10" font-weight="700" fill="var(--ob-text)">
          ${speedup.toFixed(1)}× (${c.ratio.toFixed(2)}×)
        </text>
      `;
    });

    svg += '</svg>';
    container.innerHTML = svg;
  }

  // Render Table & Pagination
  function renderTable(cases, state) {
    const tbody = document.getElementById('ob-table-body');
    const total = cases.length;
    const pageSize = state.pageSize;
    const totalPages = Math.max(Math.ceil(total / pageSize), 1);
    state.maxPage = totalPages;
    if (state.page > totalPages) state.page = totalPages;

    const startIdx = (state.page - 1) * pageSize;
    const pageRows = cases.slice(startIdx, startIdx + pageSize);

    // Update table header sort indicators
    document.querySelectorAll('#ob-data-table th[data-sort]').forEach(th => {
      th.classList.remove('sort-asc', 'sort-desc');
      if (th.getAttribute('data-sort') === state.sortKey) {
        th.classList.add(state.sortDir === 'asc' ? 'sort-asc' : 'sort-desc');
      }
    });

    if (pageRows.length === 0) {
      tbody.innerHTML = `
        <tr>
          <td colspan="10" style="text-align: center; padding: 2.5rem; color: var(--ob-text-muted);">
            No benchmark cases match the current filter criteria.
          </td>
        </tr>
      `;
    } else {
      let rowsHtml = '';
      for (let i = 0; i < pageRows.length; i++) {
        const r = pageRows[i];
        const hasRef = r.status === 'ok' && r.ratio !== null && !isNaN(r.ratio) && r.ratio > 0;
        let ratioBadge = '<span class="ob-badge ob-badge-neutral">—</span>';

        if (hasRef) {
          const isFaster = r.ratio < 0.98;
          const isSimilar = r.ratio >= 0.98 && r.ratio <= 1.05;
          const cls = isFaster ? 'ob-badge-green' : (isSimilar ? 'ob-badge-neutral' : 'ob-badge-amber');
          const speedupText = isFaster ? ` (${(1 / r.ratio).toFixed(1)}× faster)` : '';
          ratioBadge = `<span class="ob-badge ${cls}">${r.ratio.toFixed(2)}×${speedupText}</span>`;
        }

        let statusBadge = '<span class="ob-badge ob-badge-green">ok</span>';
        if (r.status === 'unsupported') statusBadge = '<span class="ob-badge ob-badge-neutral">unsupported</span>';
        else if (r.status === 'mismatch') statusBadge = '<span class="ob-badge ob-badge-red">mismatch</span>';
        else if (r.status === 'error') statusBadge = '<span class="ob-badge ob-badge-red">error</span>';
        else if (r.status === 'unchecked') statusBadge = '<span class="ob-badge ob-badge-amber">unchecked</span>';

        const allocsText = (r.oxiblas_allocs >= 0 ? r.oxiblas_allocs : '—') + ' / ' + (r.reference_allocs >= 0 ? r.reference_allocs : '—');

        rowsHtml += `
          <tr>
            <td class="ob-case-cell" title="${escapeHtml(r.name)}">${escapeHtml(r.name)}</td>
            <td><span class="ob-badge ob-badge-neutral">${escapeHtml(r.family)}</span></td>
            <td><span class="ob-badge ob-badge-neutral">${escapeHtml(r.form)}</span></td>
            <td><code>${escapeHtml(r.eltype)}</code></td>
            <td>${formatOrder(r.order)}</td>
            <td style="font-weight: 600;">${formatMs(r.oxiblas_ms)}</td>
            <td style="color: var(--ob-text-muted);">${formatMs(r.reference_ms)}</td>
            <td>${ratioBadge}</td>
            <td style="font-family: var(--ob-font-mono); font-size: 0.75rem;">${allocsText}</td>
            <td title="${escapeHtml(r.note || '')}">${statusBadge}</td>
          </tr>
        `;
      }
      tbody.innerHTML = rowsHtml;
    }

    // Pagination info & buttons
    const fromNum = total === 0 ? 0 : startIdx + 1;
    const toNum = Math.min(startIdx + pageSize, total);
    document.getElementById('ob-page-info').textContent =
      `Page ${state.page} of ${totalPages} (cases ${fromNum.toLocaleString()}–${toNum.toLocaleString()} of ${total.toLocaleString()})`;

    document.getElementById('ob-btn-first').disabled = state.page <= 1;
    document.getElementById('ob-btn-prev').disabled = state.page <= 1;
    document.getElementById('ob-btn-next').disabled = state.page >= totalPages;
    document.getElementById('ob-btn-last').disabled = state.page >= totalPages;
  }

  // Auto-init on load
  if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', initDashboard);
  } else {
    initDashboard();
  }
})();
