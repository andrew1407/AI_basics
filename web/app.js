// AI Basics Labs showcase: lab panels, SVG and canvas charts, and copy actions.
// Every lab runs its Python in a Pyodide Web Worker (pyworker.js -> worker.js -> bridge.py -> showcase.py),
// so the page never waits on it. The labs' matplotlib figures come back as PNGs or as data drawn here.

const REPO = 'https://github.com/Andrew1407/AI_basics';
const BLOB = `${REPO}/blob/main`;
const TREE = `${REPO}/tree/main`;
const $ = (s, root = document) => root.querySelector(s);
const $$ = (s, root = document) => [...root.querySelectorAll(s)];
const esc = (s) => String(s).replace(/[&<>"]/g, (c) => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;' }[c]));
const fmt = (v, d = 4) => {
  if (typeof v !== 'number' || !Number.isFinite(v)) return v === null || v === undefined ? '–' : String(v);
  const a = Math.abs(v);
  if (a !== 0 && (a >= 1e7 || a < 1e-3)) return v.toExponential(2);
  return String(+v.toFixed(d));
};
const int = (v) => Number(v).toLocaleString('en-US');
const range = (n, f = (i) => i) => Array.from({ length: n }, (_, i) => f(i));
const stat = (value, label, cls = '') => `<div class="stat ${cls}"><b>${value}</b><span>${label}</span></div>`;
const css = (name) => getComputedStyle(document.documentElement).getPropertyValue(name).trim();
const debounce = (f, ms) => { let t; return (...a) => { clearTimeout(t); t = setTimeout(() => f(...a), ms); }; };

// ----------------------------------------------------------------- runtime --
let pyReady = false;
const pyWaiters = [];
window.addEventListener('py-ready', () => {
  pyReady = true;
  $$('.py-status').forEach((el) => el.remove());
  pyWaiters.splice(0).forEach((w) => w.ok());
});
window.addEventListener('py-failed', ({ detail }) => {
  $$('.py-status').forEach((el) => {
    el.classList.add('slow');
    $('b', el).textContent = 'The Python runtime failed to load.';
    $('.note-text', el).textContent = `${detail || ''} Reload the page to try again.`;
  });
  pyWaiters.splice(0).forEach((w) => w.fail(new Error(`The Python runtime failed to load: ${detail}`)));
});
const whenPy = () => (pyReady ? Promise.resolve() : new Promise((ok, fail) => pyWaiters.push({ ok, fail })));

/** Runs showcase.RUNNERS[name](**params) in the worker; status(text) gets download and progress notes. */
async function runPy(name, params = {}, status = () => {}) {
  await whenPy();
  const need = await window.py('need', name);
  if (need) {
    status(`Loading ${need}…`);
    const err = await window.py('prepare', name);
    if (err) throw new Error(err);
  }
  status('');
  const res = JSON.parse(await window.py('run', name, JSON.stringify(params), (p) => status(p.text || '')));
  if (!res.ok) throw new Error(res.error);
  return res.data;
}

// ------------------------------------------------------------------ charts --
const charts = new Map();
let chartSeq = 0;
const PALETTE = ['--c1', '--c2', '--c3', '--c4', '--c5', '--c6', '--c7'];

function niceStep(span, count) {
  const raw = span / Math.max(1, count);
  const p = 10 ** Math.floor(Math.log10(raw || 1));
  const m = raw / p;
  return (m >= 5 ? 10 : m >= 2 ? 5 : m >= 1 ? 2 : 1) * p;
}
function ticks(min, max, count) {
  if (min === max) { min -= 1; max += 1; }
  const step = niceStep(max - min, count);
  const lo = Math.floor(min / step) * step, hi = Math.ceil(max / step) * step;
  const out = [];
  for (let v = lo; v <= hi + step / 2; v += step) out.push(+v.toPrecision(12));
  return { lo, hi, ticks: out };
}

/** spec: { title, x, y, series: [{ name, xs, ys, color, dash, scatter, width }], height, tightX, logY } */
function chart(spec) {
  const id = `c${++chartSeq}`;
  spec.series.forEach((s, i) => { s.color ||= PALETTE[i % PALETTE.length]; });
  charts.set(id, spec);
  const keys = spec.series.length > 1
    ? `<span class="keys">${spec.series.map((s) => `<span><i style="background:var(${s.color})${s.dash ? ';opacity:.6' : ''}"></i>${esc(s.name)}</span>`).join('')}</span>` : '';
  return `<figure class="chart" data-chart="${id}" data-copyable data-title="${esc(spec.title)}"><figcaption><span>${esc(spec.title)}</span>${keys}${actions('png', 'csv')}</figcaption><div class="plot"></div></figure>`;
}

function drawCharts(root = document) {
  $$('figure.chart', root).forEach((fig) => {
    const spec = charts.get(fig.dataset.chart);
    const plot = $('.plot', fig);
    if (!spec || !plot.offsetParent) return;
    drawChart(plot, spec);
  });
  $$('[data-canvas]', root).forEach((box) => { if (box.offsetParent) canvases.get(box.dataset.canvas)?.(); });
}
let resizeTimer;
window.addEventListener('resize', () => { clearTimeout(resizeTimer); resizeTimer = setTimeout(() => drawCharts(), 120); });
window.addEventListener('themechange', () => drawCharts());

function drawChart(plot, spec) {
  const W = Math.max(260, plot.clientWidth), H = spec.height || 230;
  const M = { l: 58, r: 14, t: 8, b: 38 };
  const all = spec.series.flatMap((s) => s.xs.map((x, i) => [x, s.ys[i]])).filter(([x, y]) => Number.isFinite(x) && Number.isFinite(y));
  if (!all.length) { plot.innerHTML = '<p class="hint">No data.</p>'; return; }
  let xmin = Infinity, xmax = -Infinity, ymin = Infinity, ymax = -Infinity;
  for (const [x, y] of all) { if (x < xmin) xmin = x; if (x > xmax) xmax = x; if (y < ymin) ymin = y; if (y > ymax) ymax = y; }
  if (spec.yMin !== undefined) ymin = Math.min(ymin, spec.yMin);
  if (spec.yMax !== undefined) ymax = Math.max(ymax, spec.yMax);
  const xr = ticks(xmin, xmax, Math.max(3, Math.floor((W - M.l) / 90)));
  const yr = ticks(ymin, ymax, 5);
  const x0 = spec.tightX === false ? xr.lo : xmin, x1 = spec.tightX === false ? xr.hi : xmax;
  const X = (v) => M.l + ((v - x0) / (x1 - x0 || 1)) * (W - M.l - M.r);
  const Y = (v) => H - M.b - ((v - yr.lo) / (yr.hi - yr.lo || 1)) * (H - M.t - M.b);
  const xt = xr.ticks.filter((v) => v >= x0 - 1e-9 && v <= x1 + 1e-9);
  const grid = yr.ticks.map((v) => `<line x1="${M.l}" x2="${W - M.r}" y1="${Y(v)}" y2="${Y(v)}" stroke="var(--grid)"/><text x="${M.l - 8}" y="${Y(v)}" text-anchor="end" dominant-baseline="middle">${fmt(v, 6)}</text>`).join('')
    + xt.map((v) => `<line x1="${X(v)}" x2="${X(v)}" y1="${H - M.b}" y2="${H - M.b + 4}" stroke="var(--muted)"/><text x="${X(v)}" y="${H - M.b + 16}" text-anchor="middle">${fmt(v, 6)}</text>`).join('');
  const axes = `<line x1="${M.l}" x2="${W - M.r}" y1="${H - M.b}" y2="${H - M.b}" stroke="var(--line)"/>`
    + (yr.lo < 0 && yr.hi > 0 ? `<line x1="${M.l}" x2="${W - M.r}" y1="${Y(0)}" y2="${Y(0)}" stroke="var(--muted)" stroke-opacity=".45"/>` : '')
    + `<text x="${(M.l + W - M.r) / 2}" y="${H - 4}" text-anchor="middle">${esc(spec.x)}</text>`
    + `<text transform="translate(12 ${(M.t + H - M.b) / 2}) rotate(-90)" text-anchor="middle">${esc(spec.y)}</text>`;
  const lines = spec.series.map((s) => {
    const ok = (i) => Number.isFinite(s.xs[i]) && Number.isFinite(s.ys[i]);
    if (s.scatter) return s.xs.map((x, i) => (ok(i) ? `<circle cx="${X(x).toFixed(1)}" cy="${Y(s.ys[i]).toFixed(1)}" r="${s.r || 2.8}" fill="var(${s.color})" fill-opacity="${s.opacity || .75}"/>` : '')).join('');
    const pts = s.xs.map((x, i) => (ok(i) ? `${X(x).toFixed(1)},${Y(s.ys[i]).toFixed(1)}` : '')).filter(Boolean).join(' ');
    const dots = spec.dots || s.dots ? s.xs.map((x, i) => (ok(i) ? `<circle cx="${X(x)}" cy="${Y(s.ys[i])}" r="2.6" fill="var(${s.color})"/>` : '')).join('') : '';
    return `<polyline points="${pts}" fill="none" stroke="var(${s.color})" stroke-width="${s.width || 1.8}" stroke-linejoin="round" ${s.dash ? 'stroke-dasharray="5 4"' : ''}/>${dots}`;
  }).join('');
  plot.innerHTML = `<svg viewBox="0 0 ${W} ${H}" height="${H}" role="img" aria-label="${esc(spec.title)}">${grid}${axes}${lines}<line class="cross" y1="${M.t}" y2="${H - M.b}" stroke="var(--muted)" stroke-dasharray="3 3" visibility="hidden"/><g class="marks"></g></svg><div class="tip" hidden></div>`;

  // hover: nearest sample (by x) of each series
  const svg = $('svg', plot), tip = $('.tip', plot), cross = $('.cross', svg), marks = $('.marks', svg);
  const sorted = spec.series.map((s) => {
    const idx = range(s.xs.length).filter((i) => Number.isFinite(s.xs[i]) && Number.isFinite(s.ys[i]));
    if (s.scatter) idx.sort((a, b) => s.xs[a] - s.xs[b]);
    return idx;
  });
  const nearest = (s, idx, v) => {
    if (!idx.length) return -1;
    let lo = 0, hi = idx.length - 1;
    while (hi - lo > 1) { const m = (lo + hi) >> 1; if (s.xs[idx[m]] < v) lo = m; else hi = m; }
    return Math.abs(s.xs[idx[lo]] - v) <= Math.abs(s.xs[idx[hi]] - v) ? idx[lo] : idx[hi];
  };
  const move = (e) => {
    const rect = svg.getBoundingClientRect();
    const px = ((e.clientX - rect.left) / rect.width) * W;
    if (px < M.l - 6 || px > W - M.r + 6) return hide();
    const v = x0 + ((px - M.l) / (W - M.l - M.r)) * (x1 - x0);
    const hits = spec.series.map((s, k) => { const i = nearest(s, sorted[k], v); return i < 0 ? null : { s, x: s.xs[i], y: s.ys[i] }; }).filter(Boolean);
    if (!hits.length) return hide();
    const hx = X(hits[0].x);
    cross.setAttribute('x1', hx); cross.setAttribute('x2', hx); cross.setAttribute('visibility', 'visible');
    marks.innerHTML = hits.map((h) => `<circle cx="${X(h.x)}" cy="${Y(h.y)}" r="4" fill="var(--panel)" stroke="var(${h.s.color})" stroke-width="2"/>`).join('');
    tip.innerHTML = `<div class="tx">${esc(spec.x)} = ${fmt(hits[0].x, 6)}</div>` + hits.map((h) => `<div><i style="background:var(${h.s.color})"></i>${esc(h.s.name || spec.y)}: ${fmt(h.y, 6)}${h.s.scatter && h.x !== hits[0].x ? ` <span class="tx">at ${fmt(h.x, 4)}</span>` : ''}</div>`).join('');
    tip.hidden = false;
    const left = (hx / W) * rect.width;
    const tw = tip.offsetWidth;
    tip.style.left = `${left + 12 + tw > rect.width ? Math.max(0, left - tw - 12) : left + 12}px`;
    tip.style.top = `${M.t}px`;
  };
  const hide = () => { tip.hidden = true; cross.setAttribute('visibility', 'hidden'); marks.innerHTML = ''; };
  svg.addEventListener('pointermove', move);
  svg.addEventListener('pointerdown', move);
  svg.addEventListener('pointerleave', hide);
}

// canvas drawings: draw(canvas) is called on first show, resize and theme change
const canvases = new Map();
let canvasSeq = 0;
function canvasBox(title, draw, { height = 340, note = '', csv = null } = {}) {
  const id = `k${++canvasSeq}`;
  canvases.set(id, () => {
    const box = $(`[data-canvas="${id}"]`);
    if (!box) return;
    const cv = $('canvas', box), w = Math.max(260, cv.parentElement.clientWidth), dpr = window.devicePixelRatio || 1;
    const h = typeof height === 'function' ? height(w) : height;
    cv.width = Math.round(w * dpr); cv.height = Math.round(h * dpr);
    cv.style.height = `${h}px`;
    const ctx = cv.getContext('2d');
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    draw(ctx, w, h);
  });
  if (csv) csvSources.set(id, csv);
  return `<figure class="chart" data-canvas="${id}" data-copyable data-title="${esc(title)}"><figcaption><span>${esc(title)}</span>${note ? `<span class="keys">${note}</span>` : ''}${actions(...(csv ? ['png', 'csv'] : ['png']))}</figcaption><div class="canvas-plot"><canvas></canvas><div class="tip" hidden></div></div></figure>`;
}
const csvSources = new Map();

/** Axes on a canvas: returns X, Y mappers for the plot area. */
function canvasAxes(ctx, W, H, xlo, xhi, ylo, yhi, xlabel, ylabel, { square = false } = {}) {
  const M = { l: 58, r: 14, t: 10, b: 38 };
  let pw = W - M.l - M.r, ph = H - M.t - M.b;
  if (square) {
    const k = Math.min(pw / (xhi - xlo), ph / (yhi - ylo));
    pw = k * (xhi - xlo); ph = k * (yhi - ylo);
  }
  const X = (v) => M.l + ((v - xlo) / (xhi - xlo || 1)) * pw;
  const Y = (v) => M.t + ph - ((v - ylo) / (yhi - ylo || 1)) * ph;
  ctx.font = '11px system-ui, sans-serif';
  ctx.fillStyle = css('--muted'); ctx.strokeStyle = css('--grid'); ctx.lineWidth = 1;
  const xt = ticks(xlo, xhi, Math.max(3, Math.floor(pw / 90))).ticks.filter((v) => v >= xlo - 1e-9 && v <= xhi + 1e-9);
  const yt = ticks(ylo, yhi, 5).ticks.filter((v) => v >= ylo - 1e-9 && v <= yhi + 1e-9);
  ctx.textAlign = 'right'; ctx.textBaseline = 'middle';
  for (const v of yt) { ctx.beginPath(); ctx.moveTo(M.l, Y(v)); ctx.lineTo(M.l + pw, Y(v)); ctx.stroke(); ctx.fillText(fmt(v, 6), M.l - 8, Y(v)); }
  ctx.textAlign = 'center'; ctx.textBaseline = 'top';
  for (const v of xt) { ctx.beginPath(); ctx.moveTo(X(v), M.t); ctx.lineTo(X(v), M.t + ph); ctx.stroke(); ctx.fillText(fmt(v, 6), X(v), M.t + ph + 6); }
  ctx.fillText(xlabel, M.l + pw / 2, H - 14);
  ctx.save(); ctx.translate(12, M.t + ph / 2); ctx.rotate(-Math.PI / 2); ctx.textBaseline = 'middle'; ctx.fillText(ylabel, 0, 0); ctx.restore();
  return { X, Y, M, pw, ph };
}

/** A canvas tooltip: find(x, y) returns html or null. */
function canvasTip(box, find) {
  const cv = $('canvas', box), tip = $('.tip', box);
  cv.onpointermove = (e) => {
    const rect = cv.getBoundingClientRect(), x = e.clientX - rect.left, y = e.clientY - rect.top;
    const html = find(x, y);
    if (!html) { tip.hidden = true; return; }
    tip.innerHTML = html; tip.hidden = false;
    const tw = tip.offsetWidth;
    tip.style.left = `${x + 14 + tw > rect.width ? Math.max(0, x - tw - 14) : x + 14}px`;
    tip.style.top = `${Math.max(0, y - 10)}px`;
  };
  cv.onpointerleave = () => { tip.hidden = true; };
}

// ------------------------------------------------------------------ copying --
const ACTION_LABEL = { png: 'Copy image', csv: 'Copy CSV', table: 'Copy table', results: 'Copy results', code: 'Copy code', text: 'Copy text' };
const ACTION_TIP = { png: 'Copy as a PNG image (downloads it if the clipboard is unavailable)', csv: 'Copy the plotted data as CSV', table: 'Copy the table as tab-separated text, pastes into spreadsheets', results: 'Copy the inputs and results as text', code: 'Copy the source code', text: 'Copy the printed output' };
const actions = (...kinds) => `<span class="copy-actions">${kinds.map((k) => `<button type="button" class="mini" data-act="${k}" title="${ACTION_TIP[k]}">${ACTION_LABEL[k]}</button>`).join('')}</span>`;

// a canvas with the caption on top, filled with the panel colour
function framed(draw, w, h, title) {
  const scale = 2, head = title ? 30 : 0, pad = 14;
  const c = document.createElement('canvas');
  c.width = (w + pad * 2) * scale; c.height = (h + head + pad * 2) * scale;
  const ctx = c.getContext('2d');
  ctx.scale(scale, scale);
  ctx.fillStyle = css('--panel');
  ctx.fillRect(0, 0, w + pad * 2, h + head + pad * 2);
  if (title) {
    ctx.fillStyle = css('--ink');
    ctx.font = `600 14px ${getComputedStyle(document.body).fontFamily}`;
    ctx.fillText(title, pad, pad + 14);
  }
  return { c, ctx, pad, head, legend: (items) => {
    ctx.font = `12px ${getComputedStyle(document.body).fontFamily}`;
    let x = w + pad;
    for (const { name, color } of [...items].reverse()) {
      x -= ctx.measureText(name).width;
      ctx.fillStyle = css('--muted'); ctx.fillText(name, x, pad + 13);
      x -= 19; ctx.fillStyle = css(color); ctx.fillRect(x, pad + 7, 14, 3); x -= 14;
    }
  } };
}
const toBlob = (c) => new Promise((ok, fail) => c.toBlob((b) => (b ? ok(b) : fail(new Error('PNG encoding failed'))), 'image/png'));

function svgPng(svg, title, legend = []) {
  const box = svg.viewBox.baseVal;
  const w = (box && box.width) || +svg.getAttribute('width') || svg.clientWidth;
  const h = (box && box.height) || +svg.getAttribute('height') || svg.clientHeight;
  const clone = svg.cloneNode(true);
  const src = [svg, ...svg.querySelectorAll('*')], dst = [clone, ...clone.querySelectorAll('*')];
  src.forEach((el, i) => {
    const cs = getComputedStyle(el), c = dst[i];
    for (const a of ['fill', 'stroke', 'stroke-opacity', 'fill-opacity', 'stroke-width', 'opacity', 'visibility']) c.setAttribute(a, cs.getPropertyValue(a));
    if (el.tagName === 'text') for (const a of ['font-family', 'font-size', 'font-weight']) c.setAttribute(a, cs.getPropertyValue(a));
    c.removeAttribute('style'); c.removeAttribute('class');
  });
  clone.querySelectorAll('title, .marks').forEach((t) => t.remove());
  clone.setAttribute('xmlns', 'http://www.w3.org/2000/svg');
  clone.setAttribute('width', w); clone.setAttribute('height', h);
  if (!clone.getAttribute('viewBox')) clone.setAttribute('viewBox', `0 0 ${w} ${h}`);
  const url = URL.createObjectURL(new Blob([new XMLSerializer().serializeToString(clone)], { type: 'image/svg+xml' }));
  return new Promise((resolve, reject) => {
    const img = new Image();
    img.onload = () => {
      const f = framed(null, w, h, title);
      f.legend(legend);
      f.ctx.drawImage(img, f.pad, f.pad + f.head, w, h);
      URL.revokeObjectURL(url);
      resolve(toBlob(f.c));
    };
    img.onerror = () => { URL.revokeObjectURL(url); reject(new Error('SVG render failed')); };
    img.src = url;
  });
}
function canvasPng(cv, title) {
  const w = cv.clientWidth, h = cv.clientHeight;
  const f = framed(null, w, h, title);
  f.ctx.imageSmoothingEnabled = !cv.classList.contains('pixels');
  f.ctx.drawImage(cv, f.pad, f.pad + f.head, w, h);
  return toBlob(f.c);
}
function imgPng(img, title) {
  const w = img.naturalWidth, h = img.naturalHeight;
  const f = framed(null, w, h, title);
  f.ctx.drawImage(img, f.pad, f.pad + f.head, w, h);
  return toBlob(f.c);
}

const csvCell = (v) => (/[",\n]/.test(String(v)) ? `"${String(v).replace(/"/g, '""')}"` : String(v));
const toCsv = (rows) => rows.map((r) => r.map(csvCell).join(',')).join('\n');
function chartCsv(spec) {
  const s0 = spec.series[0];
  const shared = spec.series.every((s) => s.xs.length === s0.xs.length && s.xs.every((x, i) => x === s0.xs[i]));
  if (shared) return toCsv([[spec.x, ...spec.series.map((s) => s.name || spec.y)], ...s0.xs.map((x, i) => [x, ...spec.series.map((s) => s.ys[i])])]);
  return toCsv([['series', spec.x, spec.y], ...spec.series.flatMap((s) => s.xs.map((x, i) => [s.name, x, s.ys[i]]))]);
}
const tableTsv = (table) => [...table.rows].filter((r) => !r.hidden).map((r) => [...r.cells].map((c) => c.innerText.trim().replace(/\s+/g, ' ')).join('\t')).join('\n');

function resultsText(sec) {
  const lab = LABS.find((l) => `panel-${l.id}` === sec.id);
  const lines = [`${labTitle(lab)}: ${lab.name}`];
  if (lab.fields?.length) {
    const p = params(lab);
    lines.push(`Parameters: ${lab.fields.filter((f) => f.type !== 'textarea').map((f) => `${f.label} = ${p[f.key] ?? ''}`).join(', ')}`);
    lab.fields.filter((f) => f.type === 'textarea').forEach((f) => lines.push(`${f.label}: ${p[f.key]}`));
  }
  const out = $('.out', sec);
  out.querySelectorAll('.big, .note, .verdict').forEach((el) => lines.push(el.innerText.trim().replace(/\s+/g, ' ')));
  out.querySelectorAll('.stats').forEach((st) => {
    const group = st.dataset.group;
    if (group) lines.push(`${group}:`);
    st.querySelectorAll('.stat').forEach((x) => lines.push(`${group ? '  ' : ''}${$('span', x).innerText.trim()}: ${$('b', x).innerText.trim()}`));
  });
  out.querySelectorAll('pre.console.printed').forEach((p) => lines.push('Printed output:', p.innerText.trimEnd()));
  return lines.join('\n');
}

async function copyAction(btn) {
  const kind = btn.dataset.act, box = btn.closest('[data-copyable]');
  const done = (text) => { btn.style.minWidth = btn.style.minWidth || btn.getBoundingClientRect().width + 'px'; btn.textContent = text; setTimeout(() => (btn.textContent = ACTION_LABEL[kind]), 1400); };
  const name = (box?.dataset.title || 'chart').toLowerCase().replace(/[^a-z0-9]+/g, '-').replace(/^-|-$/g, '');
  try {
    if (kind === 'png') {
      const title = box.dataset.title;
      let make;
      const img = box.querySelector('img.fig');
      const cv = box.querySelector('canvas');
      if (img) make = () => imgPng(img, title);
      else if (cv) make = () => canvasPng(cv, title);
      else {
        const svg = box.querySelector('.plot svg, svg');
        const spec = charts.get(box.dataset.chart);
        const legend = spec && spec.series.length > 1 ? spec.series.map((x) => ({ name: x.name, color: x.color })) : [];
        make = () => svgPng(svg, title, legend);
      }
      try {
        // pass the promise straight to ClipboardItem so Safari keeps the user gesture
        await navigator.clipboard.write([new ClipboardItem({ 'image/png': make() })]);
        return done('Copied');
      } catch {
        const a = document.createElement('a');
        a.href = URL.createObjectURL(await make());
        a.download = `${name}.png`;
        a.click();
        setTimeout(() => URL.revokeObjectURL(a.href), 1000);
        return done('Downloaded');
      }
    }
    const text = kind === 'csv' ? (box.dataset.canvas ? csvSources.get(box.dataset.canvas)() : chartCsv(charts.get(box.dataset.chart)))
      : kind === 'table' ? tableTsv(box.querySelector('table'))
      : kind === 'code' || kind === 'text' ? box.querySelector('pre').innerText
      : resultsText(btn.closest('section.panel'));
    await navigator.clipboard.writeText(text);
    done('Copied');
  } catch (e) { console.error(e); done('Failed'); }
}
document.addEventListener('click', (e) => { const b = e.target.closest('button[data-act]'); if (b) copyAction(b); });

// ------------------------------------------------------------ shared views --
const table = (head, rows, { title = '', cls = '', max = '' } = {}) => `<div data-copyable data-title="${esc(title)}">${title ? `<div class="copy-head"><h2>${title}</h2>${actions('table')}</div>` : `<div class="out-actions">${actions('table')}</div>`}
  <div class="table-wrap" ${max ? `style="max-height:${max}"` : ''}><table class="${cls}"><thead><tr>${head.map((h) => `<th${/^[#\d]|num:/.test(h) ? ' class="num"' : ''}>${esc(String(h).replace(/^num:/, ''))}</th>`).join('')}</tr></thead><tbody>${rows.join('')}</tbody></table></div></div>`;
const printed = (text, title = 'Printed output') => (text && text.trim() ? `<div data-copyable data-title="${esc(title)}"><div class="copy-head" style="margin-top:18px"><h2>${title}</h2>${actions('text')}</div><pre class="console printed">${esc(text.trimEnd())}</pre></div>` : '');
function gallery(figs, captions = []) {
  return `<div class="gallery">${figs.map((f, i) => {
    const cap = captions[i] || f.title || `Figure ${i + 1}`;
    return `<figure data-copyable data-title="${esc(cap)}"><figcaption><span><span class="fig-n">${i + 1}.</span> ${esc(cap)}</span>${actions('png')}</figcaption><img class="fig" alt="${esc(cap)}" src="data:image/png;base64,${f.png}"></figure>`;
  }).join('')}</div>`;
}
const sectionHead = (title, extra = '') => `<div class="gallery-head"><h2>${title}</h2>${extra}</div>`;

// bipolar/0-1 bitmaps on small canvases
function paintBits(cv, bits, side, { on = '#1b1f24', off = '#ffffff' } = {}) {
  cv.width = side; cv.height = side;
  const ctx = cv.getContext('2d'), img = ctx.createImageData(side, side);
  const a = hex(on), b = hex(off);
  for (let i = 0; i < side * side; i++) {
    const c = bits[i] === '1' || bits[i] === 1 ? a : b;
    img.data.set([...c, 255], i * 4);
  }
  ctx.putImageData(img, 0, 0);
}
const hex = (h) => [1, 3, 5].map((i) => parseInt(h.slice(i, i + 2), 16));

// ------------------------------------------------------------------- labs --
const fuzzyNote = '<a href="https://pypi.org/project/scikit-fuzzy/">scikit-fuzzy</a> 0.5.0 from PyPI, with scipy, networkx and matplotlib from Pyodide';

const LABS = [
  {
    id: 'lab1', tab: '1', name: 'Fuzzy control system', files: ['1-lab/1-lab.py', '1-lab/simulator.py', '1-lab/terms.py', '1-lab/tables.py', '1-lab/control_system.py'],
    intro: 'Approximates f(x, y) = x·sin(x + y) with Mamdani fuzzy controllers built by <code>skfuzzy.control</code>: the inputs x and y get <i>n</i> membership functions, the output f gets <i>m</i>, and a rule per input pair maps it to the output term closest to f at that grid point. Each of the triangular, trapezoidal and Gaussian variants is simulated along the diagonal x = y, with the full rule table and with its diagonal only, and its relative error of the mean is printed.',
    runtime: `Runs <code>Simulator.simulate_variants()</code> from the lab's own modules with ${fuzzyNote}. Its matplotlib figures are rendered as PNGs.`,
    fields: [
      { key: 'inputs', label: 'Input terms (IN_AMOUNT)', value: 6, min: 2, max: 12 },
      { key: 'outputs', label: 'Output terms (OUT_AMOUNT)', value: 9, min: 2, max: 20 },
      { key: 'step', label: 'Universe step', value: 0.04, min: 0.002, max: 0.1, step: 'any' },
    ],
    hint: 'The lab samples the universe with a step of 0.001 (VAL_RANGE = (0, 1, 10e-4)), 1200 points simulated six times. That takes minutes in the browser, so the default step here is 0.04 (30 points).',
    run: (p, s) => runPy('lab1', p, s),
    render: lab1View,
  },
  {
    id: 'lab2', tab: '2', name: 'Membership functions', files: ['2-lab/2-lab.py'],
    intro: 'Plots eleven scikit-fuzzy membership functions (triangular, trapezoidal, Gaussian, two-sided Gaussian, generalised bell, sigmoids, Z, Π and S) with their fuzzy NOT, then the fuzzy OR and AND of two of them.',
    runtime: `Runs <code>2-lab.py</code> as written with ${fuzzyNote}; each figure it saves to <code>out/</code> is shown here.`,
    fields: [
      { key: 'step', label: 'Range step', value: 0.01, min: 0.001, max: 0.1, step: 'any' },
      { key: 'center', label: 'CENTER', value: 0.5, min: 0, max: 1, step: 'any' },
      { key: 'sigma', label: 'SIGMA', value: 0.1, min: 0.01, max: 1, step: 'any' },
      { key: 'fn1', label: 'OR / AND of', value: 'gauss2mf', options: ['trimf', 'trapmf', 'gaussmf', 'gauss2mf', 'gbellmf', 'sigmf', 'dsigmf', 'psigmf', 'zmf', 'pimf', 'smf'] },
      { key: 'fn2', label: 'and', value: 'zmf', options: ['trimf', 'trapmf', 'gaussmf', 'gauss2mf', 'gbellmf', 'sigmf', 'dsigmf', 'psigmf', 'zmf', 'pimf', 'smf'] },
    ],
    run: (p, s) => runPy('lab2', p, s),
    render: lab2View,
  },
  {
    id: 'lab3', tab: '3', name: 'Fuzzy c-means', files: ['3-lab/3-lab.py', '3-lab/dataset.txt'],
    intro: 'Clusters the 5000 points of <code>dataset.txt</code> (scaled by 10⁻⁵) into fuzzy clusters with <code>skfuzzy.cmeans</code> (fuzzifier m = 1.5) and plots the objective function J<sub>m</sub> per iteration.',
    runtime: `Runs <code>3-lab.py</code> with ${fuzzyNote}. <code>plot_coords</code> draws the points with one <code>plt.plot</code> call each, which takes many seconds in the browser, so the page records the <code>cmeans</code> result and draws the points in one canvas, coloured by their strongest cluster.`,
    fields: [
      { key: 'clusters', label: 'CLUSTERS_AMOUNT', value: 15, min: 2, max: 40 },
      { key: 'error', label: 'error', value: 0.005, min: 1e-6, max: 1, step: 'any' },
      { key: 'maxiter', label: 'maxiter', value: 1000, min: 1, max: 5000 },
    ],
    run: (p, s) => runPy('lab3', p, s),
    render: lab3View,
  },
  {
    id: 'lab4', tab: '4', name: 'Backpropagation networks', files: ['4-lab/4-lab.py', '4-lab/backprops.py'], port: 'Theano/Lasagne → NumPy',
    intro: 'Trains six networks to approximate z = x·sin(x + y) on the diagonal x = y ∈ [−1, 1): feed-forward nets with 10 and 20 tanh units and cascade-forward nets with [20] and [10, 10] units (Theano + Lasagne, SGD on the mean squared error), and Elman recurrent nets with [15] and [5, 5, 5] units (neurolab, gradient descent).',
    runtime: 'The Elman networks run <code>elman_recurrent_backprop</code> from <code>backprops.py</code> with neurolab 0.3.5 (a pure-Python wheel the workflow builds from its source release, with a NumPy 2 shim). Theano and Lasagne don\'t run in the browser, so the feed-forward and cascade-forward networks are <a href="https://github.com/Andrew1407/AI_basics/blob/main/web/ports/backprops_np.py">ported to NumPy</a> with the same layers, Glorot initialisation, epochs and learning rate.',
    fields: [
      { key: 'epochs', label: 'EPOCHS (feed/cascade)', value: 1000, min: 1, max: 20000 },
      { key: 'learning_rate', label: 'LEARNING_RATE', value: 0.1, min: 0.0001, max: 5, step: 'any' },
      { key: 'elman_epochs', label: 'Elman epochs', value: 200, min: 1, max: 5000 },
      { key: 'goal', label: 'GOAL (Elman)', value: 0.001, min: 0, max: 1, step: 'any' },
    ],
    hint: 'The lab trains the Elman networks for EPOCHS = 1000 too. neurolab\'s training loop is slow in the browser (about 15 s for both), so they get 200 epochs by default here.',
    run: (p, s) => runPy('lab4', p, s),
    render: lab4View,
  },
  {
    id: 'lab5', tab: '5', name: 'Hebbian learning', files: ['5-lab/5-lab.py', '5-lab/hebb.py'],
    intro: 'Trains one Hebb neuron per letter (a, n, i, r) on a 100×100 bipolar image of it (white = −1, ink = +1), then identifies the valid and the distorted image of each letter by the neurons\' output sums, highest first.',
    runtime: 'Runs <code>5-lab.py</code> as written (Pillow from Pyodide reads the BMPs). Draw on the grid below to identify your own letter with the trained neurons.',
    fields: [],
    run: (p, s) => runPy('lab5', p, s),
    render: lab5View,
    after: lab5Bind,
  },
  {
    id: 'lab6', tab: '6', name: 'ANFIS', files: ['6-lab/6-lab.py', '6-lab/anfis.py'], port: 'TensorFlow → NumPy',
    intro: 'Trains an adaptive neuro-fuzzy inference system (Gaussian memberships → rule products → normalisation → linear consequents → sum) on the first 800 rows of the concrete compressive strength dataset, validating on the next 200, with Adam and the Huber loss in batches of 10.',
    runtime: 'Runs <code>6-lab.py</code> as written, reading <code>Concrete_Data.xls</code> with pandas and xlrd. TensorFlow isn\'t available in the browser, so <code>import anfis</code> gets a <a href="https://github.com/Andrew1407/AI_basics/blob/main/web/ports/anfis_np.py">NumPy port of anfis.py</a>: the same layers and initialisers, Keras\' Adam and Huber loss, the gradients written out by hand.',
    fields: [
      { key: 'epochs', label: 'n_epochs', value: 30, min: 1, max: 400 },
      { key: 'n_memb', label: 'n_memb', value: 2, min: 2, max: 3 },
      { key: 'loss', label: 'loss', value: 'huber_loss', options: ['huber_loss', 'mse'] },
      { key: 'optimizer', label: 'optimizer', value: 'adam', options: ['adam', 'sgd'] },
    ],
    hint: 'The lab trains for 200 epochs, about 15 s in the browser, so the default here is 30. With n_memb = 3 there are 3⁸ = 6561 rules, which is much slower.',
    run: (p, s) => runPy('lab6', p, s),
    render: lab6View,
  },
  {
    id: 'lab7', tab: '7', name: 'Evolutionary optimisation', files: ['7-lab/7-lab.py'],
    intro: 'Minimises f₁(x) = sin|x|·cos(3x/2) on [0, 3) and maximises f₂(x, y) = x·sin(x + y) on [2, 6) × [1, 5) with the <code>evol</code> library: survive the best half, breed parents by averaging their genes, mutate by ±σ/2.',
    runtime: 'Runs <code>7-lab.py</code> with <a href="https://pypi.org/project/evol/">evol</a> 0.5.3 from PyPI (its optional <code>multiprocess</code> pool is stubbed, a browser has no processes). The page draws the two plots.',
    fields: [
      { key: 'fraction', label: 'survive fraction', value: 0.5, min: 0.05, max: 0.95, step: 'any' },
      { key: 'sigma', label: 'mutate sigma', value: 0.1, min: 0, max: 5, step: 'any' },
      { key: 'n', label: 'n (repeat and evolve)', value: 10, min: 1, max: 50 },
      { key: 'seed', label: 'random.seed', value: 26, min: 0, max: 1e9 },
    ],
    run: (p, s) => runPy('lab7', p, s),
    render: lab7View,
  },
  {
    id: 'lab8', tab: '8', name: 'Genetic packing', files: ['8-lab/8-lab.py'],
    intro: 'Places up to N random rectangles in an A × B area with a genetic algorithm. A chromosome holds the rectangle centres; its fitness is the overlap area plus LEARNING_RATE times the connection length weighted by a random symmetric matrix C. Each generation keeps the best chromosomes, crosses them over at a random point and mutates 20% of the offspring, and the search stops when a placement has no overlap.',
    runtime: 'Runs <code>8-lab.py</code> as written, recording the best placement of every generation so the page can replay the search.',
    fields: [
      { key: 'a', label: 'A', value: 30, min: 4, max: 200 },
      { key: 'b', label: 'B', value: 40, min: 4, max: 200 },
      { key: 'n', label: 'N', value: 15, min: 3, max: 40 },
      { key: 'iterations', label: 'MAX_ITERATIONS_BOUND', value: 100, min: 1, max: 2000 },
      { key: 'learning_rate', label: 'LEARNING_RATE', value: 0.001, min: 0, max: 1, step: 'any' },
      { key: 'seed', label: 'NumPy seed', value: '', type: 'text', placeholder: 'random' },
    ],
    hint: 'The lab allows up to 1000 generations, which can take 25 s in the browser; the default bound here is 100.',
    run: (p, s) => runPy('lab8', { ...p, seed: String(p.seed).trim() === '' ? null : Number(p.seed) }, s),
    render: lab8View,
    after: lab8Bind,
  },
  {
    id: 'lab9', tab: '9', name: 'MNIST digits', files: ['9-lab/9-lab.py'], port: 'Keras → NumPy inference',
    intro: 'A dense network (Flatten → 128 relu → 128 relu → 10 softmax) trained with Adam for 50 epochs on the MNIST digits, then asked to read the lab\'s own handwritten BMP examples.',
    runtime: 'The network was trained once by <code>9-lab.py</code> itself (<a href="https://github.com/Andrew1407/AI_basics/blob/main/web/models/export_models.py">models/export_models.py</a>), and its weights run here in NumPy with the lab\'s preprocessing: invert, then <code>keras.utils.normalize</code> along axis 1, as for the training images.',
    fields: [],
    run: (p, s) => runPy('lab9', p, s),
    render: lab9View,
    after: lab9Bind,
  },
  {
    id: 'lab10', tab: '10', name: 'AlexNet', files: ['10-lab/10-lab.py'], port: 'architecture only',
    intro: 'An AlexNet-style convolutional network for CIFAR-10: images standardised per image and resized from 32×32 to 227×227, five convolutions with batch normalisation and three max-poolings, then two 4096-unit dense layers with dropout. Trained with SGD (lr 0.001) for 50 epochs on 8000 images, validated on 2000.',
    runtime: 'Training it needs a GPU for hours (58 million parameters, about a billion multiply-adds per image), so it doesn\'t run in the browser. The page runs the lab\'s own <code>Sequential([...])</code> statement against <a href="https://github.com/Andrew1407/AI_basics/blob/main/web/ports/keras_shapes.py">shape-inferring Keras layers</a> and lists the architecture; the totals match Keras\' <code>model.count_params()</code>.',
    fields: [
      { key: 'side', label: 'input_shape side', value: 227, min: 67, max: 512 },
      { key: 'classes', label: 'classes', value: 10, min: 2, max: 1000 },
    ],
    run: (p, s) => runPy('lab10', p, s),
    render: (d) => archView(d, 'AlexNet'),
    after: archBind,
  },
  {
    id: 'lab11', tab: '11', name: 'GoogLeNet', files: ['11-lab/11-lab.py'], port: 'architecture only',
    intro: 'A small GoogLeNet (MiniGoogLeNet) for 32×32 CIFAR-10 images: a convolution module, two inception modules, a downsample module, five more inceptions, another downsample, two inceptions, 7×7 average pooling and a softmax. Trained with SGD (momentum 0.9) and a linear learning-rate decay for 50 epochs on 8000 augmented images.',
    runtime: 'Training takes hours without a GPU, so the page runs the lab\'s <code>GoogleNet()</code>, <code>inception_module()</code>, <code>downsample_module()</code> and <code>conv_module()</code> against <a href="https://github.com/Andrew1407/AI_basics/blob/main/web/ports/keras_shapes.py">shape-inferring Keras layers</a> and shows the architecture and the <code>poly_decay</code> schedule; the totals match Keras\' <code>model.count_params()</code>.',
    fields: [
      { key: 'width', label: 'width', value: 32, min: 28, max: 256 },
      { key: 'height', label: 'height', value: 32, min: 28, max: 256 },
      { key: 'depth', label: 'depth', value: 3, min: 1, max: 4 },
      { key: 'classes', label: 'classes', value: 10, min: 2, max: 1000 },
    ],
    hint: 'The final 7×7 average pooling needs a 7×7 map, which 32×32 inputs give after the two downsamples; much smaller inputs fail as they would in Keras.',
    run: (p, s) => runPy('lab11', p, s),
    render: (d) => archView(d, 'GoogLeNet'),
    after: archBind,
  },
  {
    id: 'lab12', tab: '12', name: 'Bidirectional LSTM', files: ['12-lab/12-lab.py'], port: 'Keras → NumPy inference',
    intro: 'A sentiment classifier for IMDB reviews: TextVectorization (1000 words) → Embedding(64, masked) → Bidirectional LSTM(64) → Dense(64, relu) → Dense(1) logit, trained with Adam (1e-4) for 10 epochs on 8000 reviews of <code>imdb_reviews</code>.',
    runtime: 'The model was trained once by <code>12-lab.py</code> itself (<a href="https://github.com/Andrew1407/AI_basics/blob/main/web/models/export_models.py">models/export_models.py</a>); its vocabulary and weights run here with a NumPy LSTM, checked against Keras on the same weights.',
    fields: [{ key: 'text', label: 'Review', value: 'One of the best films I have seen this year. The story is moving, the acting is brilliant and the music is wonderful.', type: 'textarea' }],
    examples: 'sentiment',
    run: (p, s) => runPy('lab12', p, s),
    render: lab12View,
  },
  {
    id: 'exam', tab: 'Exam', name: 'ELSTM sentiment app', files: ['exam-task/app/main.py', 'exam-task/app/sentiment_analyzer/analyzer.py', 'exam-task/app/sentiment_analyzer/elstm.py', 'exam-task/app/gui/window_client.py'], port: 'TensorFlow → NumPy',
    intro: 'The exam\'s desktop app: a Tkinter window that classifies a text as positive or negative with a custom extended LSTM cell (ELSTMCell, an LSTM whose forget gate is scaled by an "emotion" matrix updated at every step). The model was trained on all 25 000 <code>imdb_reviews</code> training reviews with frozen GloVe embeddings, Conv1D and max-pooling in front of the recurrent layer.',
    runtime: 'TensorFlow doesn\'t run in the browser, so <code>SentimentAnalyzer.predict</code> is <a href="https://github.com/Andrew1407/AI_basics/blob/main/web/ports/sentiment_np.py">ported to NumPy</a>: the same Tokenizer, <code>pad_sequences</code> and ELSTMCell equations, with the variables of the SavedModel in <code>exam-task/model/model1</code>. It matches TensorFlow running the SavedModel to six decimals.',
    fields: [{ key: 'text', label: 'Enter some text:', value: '', type: 'textarea' }],
    examples: 'exam',
    run: (p, s) => runPy('exam', { text: p.text, reset: !!p.reset }, s),
    render: examView,
    runLabel: 'Predict',
  },
];
const labTitle = (lab) => (lab.tab === 'Exam' ? 'Exam task' : `Lab ${lab.tab}`);

// ------------------------------------------------------------ lab views --
function lab1View(d) {
  const variants = ['trim', 'trap', 'gauss'];
  const captions = variants.flatMap((v) => [`${v}: x terms (x.view())`, `${v}: f terms (z.view())`, `${v}: f(x, x) and the fuzzy output`, `${v} (diagonal): f(x, x) and the fuzzy output`]);
  const errCards = d.errors.map((e) => stat(`${fmt(e.err, 3)}%`, `${e.title} error`, e.err < 1 ? 'good' : '')).join('');
  const series = [{ name: 'f(x, x)', xs: d.x, ys: d.f, color: '--ink', width: 2.4 }, ...d.runs.map((r, i) => ({ name: r.title, xs: d.x, ys: r.z, color: PALETTE[i], dash: r.title.includes('diagonal') }))];
  const rows = Object.keys(d.table);
  const cols = Object.keys(d.table[rows[0]]);
  const ruleRows = rows.map((y) => `<tr><th>${y}</th>${cols.map((x) => `<td${d.diagonal[y][x] ? ' style="font-weight:700"' : ' class="muted"'}>${d.table[y][x]}</td>`).join('')}</tr>`);
  return `<div class="stats">${errCards}</div>
    <div class="charts">${chart({ title: 'f(x, x) and the six fuzzy controllers along x = y', x: 'x = y', y: 'f', series, height: 280 })}</div>
    <div class="two-col" style="margin-top:16px">${table(['', ...cols], ruleRows, { title: 'Rule table <small>· output term per (my, mx); bold: kept by the diagonal variant</small>', cls: 'grid-matrix' })}
    ${printed(d.stdout)}</div>
    ${sectionHead('The lab\'s figures <small>· matplotlib, rendered in the worker</small>')}${gallery(d.figures, captions)}`;
}

function lab2View(d) {
  const names = d.curves.map((c) => c.name);
  const items = d.curves.map((c) => ({ name: c.name, xs: d.x, ys: c.y, color: PALETTE[names.indexOf(c.name) % PALETTE.length] }));
  const half = Math.ceil(items.length / 2);
  return `<div class="charts two">${chart({ title: 'Membership functions (1 of 2)', x: 'x', y: 'μ(x)', series: items.slice(0, half), height: 260 })}${chart({ title: 'Membership functions (2 of 2)', x: 'x', y: 'μ(x)', series: items.slice(half), height: 260 })}</div>
    <div style="margin-top:16px">${table(['Function', 'Arguments'], d.curves.map((c) => `<tr><td><code>fuzz.${c.name}</code></td><td><code>${esc(c.args)}</code></td></tr>`), { title: 'FUNCTIONS' })}</div>
    ${sectionHead('Figures saved to out/ <small>· matplotlib, rendered in the worker</small>')}${gallery(d.figures)}`;
}

function lab3View(d) {
  const k = d.centers.length;
  const color = (i) => `hsl(${(i * 360) / k} 62% 52%)`;
  const xlo = Math.min(...d.x), xhi = Math.max(...d.x), ylo = Math.min(...d.y), yhi = Math.max(...d.y);
  const scatter = canvasBox('Clusters', (ctx, W, H) => {
    const { X, Y } = canvasAxes(ctx, W, H, xlo, xhi, ylo, yhi, 'x', 'y');
    for (let i = 0; i < d.x.length; i++) { ctx.fillStyle = color(d.labels[i]); ctx.fillRect(X(d.x[i]) - 1.5, Y(d.y[i]) - 1.5, 3, 3); }
    d.centers.forEach(([cx, cy], i) => {
      ctx.beginPath(); ctx.arc(X(cx), Y(cy), 7, 0, 2 * Math.PI);
      ctx.fillStyle = css('--panel'); ctx.fill(); ctx.lineWidth = 3; ctx.strokeStyle = color(i); ctx.stroke();
      ctx.beginPath(); ctx.arc(X(cx), Y(cy), 2.5, 0, 2 * Math.PI); ctx.fillStyle = css('--ink'); ctx.fill();
    });
    lab3View.map = { X, Y };
  }, { height: (w) => Math.min(520, Math.max(300, w * 0.62)), note: `${int(d.x.length)} points · ${k} centres`, csv: () => toCsv([['x', 'y', 'cluster'], ...d.x.map((x, i) => [x, d.y[i], d.labels[i] + 1])]) });
  lab3View.data = d;
  const rows = d.centers.map(([x, y], i) => `<tr><td><span class="sw" style="background:${color(i)}"></span>${i + 1}</td><td class="num">${fmt(x, 4)}</td><td class="num">${fmt(y, 4)}</td><td class="num">${d.sizes[i]}</td></tr>`);
  return `<div class="stats">${stat(d.iterations, 'iterations')}${stat(fmt(d.history[d.history.length - 1], 5), 'final Jm')}${stat(fmt(d.fpc, 4), 'fuzzy partition coefficient')}${stat(`${fmt(d.seconds, 2)} s`, 'cmeans time')}</div>
    <div class="charts">${scatter}</div>
    <div class="two-col" style="margin-top:16px">${chart({ title: 'Objective function change', x: 'iterations', y: 'fn values', series: [{ name: 'Jm', xs: range(d.history.length), ys: d.history, color: '--c1' }], dots: d.history.length < 40 })}
    ${table(['Cluster', 'num:x', 'num:y', 'num:points'], rows, { title: 'Centres', max: '300px' })}</div>`;
}

function lab4View(d) {
  const cards = d.nets.map((n) => stat(fmt(n.errors[n.errors.length - 1], 5), `${n.title} · ${n.engine} · ${fmt(n.seconds, 2)} s`)).join('');
  const charts2 = d.nets.map((n) => chart({ title: `${n.key}: z and the prediction`, x: 'x = y', y: 'z', series: [{ name: 'z = x·sin(2x)', xs: d.x, ys: d.target, color: '--ink' }, { name: n.title, xs: d.x, ys: n.z, color: '--c2' }] })
    + chart({ title: `${n.key} errors`, x: 'epoch', y: n.engine === 'neurolab' ? 'error (SSE)' : 'loss (MSE)', series: [{ name: 'error', xs: range(n.errors.length, (i) => i + 1), ys: n.errors, color: '--c5' }] })).join('');
  return `<div class="stats" data-group="Final error">${cards}</div><div class="charts two">${charts2}</div>`;
}

function lab5View(d) {
  const keys = Object.keys(d.images);
  const thumbs = keys.map((k) => {
    const res = d.results[k], top = Object.keys(res)[0], ok = top === k[0];
    return `<div class="thumb ${ok ? 'ok' : 'bad'}"><button type="button" data-load="${k}" title="Edit this image on the grid"><canvas class="pixels" data-bits="${k}"></canvas></button>${esc(k)}.bmp<br>identified: <b>${top}</b></div>`;
  }).join('');
  const letters = Object.keys(d.weights);
  const rows = keys.map((k) => `<tr><td>${esc(k)}</td>${letters.map((l) => `<td class="num"${Object.keys(d.results[k])[0] === l ? ' style="font-weight:700"' : ''}>${d.results[k][l]}</td>`).join('')}</tr>`);
  return `<div class="stats">${letters.map((l) => stat(d.iterations[l], `train('${l}') iterations`)).join('')}${stat(`${d.side}×${d.side}`, 'image size')}</div>
    ${sectionHead('Images <small>· green: identified as its own letter</small>')}<div class="thumbs">${thumbs}</div>
    <div class="two-col" style="margin-top:16px">${table(['Image', ...letters], rows, { title: 'Output sums <small>· Σ x·w + w₀ per letter neuron</small>' })}${printed(d.stdout)}</div>
    ${sectionHead('Identify your own letter')}
    <div class="board" data-copyable data-title="Hebb grid"><div><canvas class="pixels" id="hebb-grid" width="100" height="100" style="width:300px;height:300px;image-rendering:pixelated;cursor:crosshair"></canvas>
      <div class="tools"><div class="seg" role="radiogroup"><label><input type="radio" name="hebb-mode" value="draw" checked>Draw</label><label><input type="radio" name="hebb-mode" value="erase">Erase</label></div>
      <label class="field" style="flex-direction:row;align-items:center;gap:6px">Brush <select id="hebb-brush"><option>3</option><option selected>6</option><option>10</option></select></label>
      <button type="button" class="mini" id="hebb-clear">Clear</button>${actions('png')}</div></div>
      <div class="side"><h2>Neuron outputs</h2><div class="bars" id="hebb-bars"></div><p class="hint">Click an image above to load it onto the grid, or draw on it (100×100 pixels, as the BMPs). Each change is identified by <code>hebb.identify()</code> with the trained neurons.</p></div></div>
    ${sectionHead('What each neuron learned <small>· weights w₁…w₁₀₀₀₀ after training (ink = +1)</small>')}<div class="thumbs">${letters.map((l) => `<div class="thumb"><canvas class="pixels" data-weights="${l}"></canvas>neuron <b>${l}</b>, w₀ = ${d.bias[l]}</div>`).join('')}</div>`;
}

function lab5Bind(out, d) {
  $$('canvas[data-bits]', out).forEach((cv) => paintBits(cv, d.images[cv.dataset.bits], d.side));
  $$('canvas[data-weights]', out).forEach((cv) => paintBits(cv, d.weights[cv.dataset.weights].map((v) => (v > 0 ? 1 : 0)), d.side, { on: '#9c4421', off: '#ffe7da' }));
  const cv = $('#hebb-grid', out), ctx = cv.getContext('2d'), side = d.side;
  let bits = d.images.a_invalid.split('').map(Number);
  const paint = () => paintBits(cv, bits, side);
  const bars = $('#hebb-bars', out);
  const identify = debounce(async () => {
    const res = await runPy('lab5_identify', { pixels: bits.join('') });
    const vals = Object.values(res), max = Math.max(...vals.map(Math.abs), 1), top = Object.keys(res)[0];
    bars.innerHTML = Object.entries(res).map(([l, v]) => `<div class="bar${l === top ? ' top' : ''}"><span>${l}</span><i style="width:${(Math.max(0, v) / max) * 100}%"></i><span>${v}</span></div>`).join('');
  }, 120);
  const at = (e) => { const r = cv.getBoundingClientRect(); return [Math.floor(((e.clientX - r.left) / r.width) * side), Math.floor(((e.clientY - r.top) / r.height) * side)]; };
  let down = false, last = null;
  const stamp = ([x, y]) => {
    const b = +$('#hebb-brush', out).value, v = $('input[name="hebb-mode"]:checked', out).value === 'draw' ? 1 : 0;
    for (let j = -b; j <= b; j++) for (let i = -b; i <= b; i++) {
      const xx = x + i, yy = y + j;
      if (xx >= 0 && yy >= 0 && xx < side && yy < side && i * i + j * j <= b * b) bits[yy * side + xx] = v;
    }
  };
  const line = (a, b) => { const n = Math.max(Math.abs(b[0] - a[0]), Math.abs(b[1] - a[1]), 1); for (let t = 0; t <= n; t++) stamp([Math.round(a[0] + ((b[0] - a[0]) * t) / n), Math.round(a[1] + ((b[1] - a[1]) * t) / n)]); };
  cv.onpointerdown = (e) => { down = true; cv.setPointerCapture(e.pointerId); last = at(e); stamp(last); paint(); };
  cv.onpointermove = (e) => { if (!down) return; const p = at(e); line(last, p); last = p; paint(); };
  cv.onpointerup = cv.onpointercancel = () => { if (down) { down = false; identify(); } };
  $('#hebb-clear', out).onclick = () => { bits = bits.map(() => 0); paint(); identify(); };
  $$('[data-load]', out).forEach((b) => b.onclick = () => { bits = d.images[b.dataset.load].split('').map(Number); paint(); identify(); cv.scrollIntoView({ block: 'nearest', behavior: 'smooth' }); });
  paint(); identify();
}

function lab6View(d) {
  const ep = range(d.loss.length, (i) => i + 1);
  const lo = Math.min(...d.actual, ...d.predicted), hi = Math.max(...d.actual, ...d.predicted);
  const mfCharts = d.inputs.map((name, n) => chart({ title: `Input ${n + 1}: ${name}`, x: 'sigmoid(x)', y: 'μ', height: 170, series: [
    ...d.mfs[n].map((ys, m) => ({ name: `μ${m + 1}`, xs: d.xn, ys, color: PALETTE[m] })),
    ...d.init[n].map((ys, m) => ({ name: `μ${m + 1} initial`, xs: d.xn, ys, color: PALETTE[m], dash: true, width: 1.2 })),
  ] })).join('');
  const rows = d.inputs.map((name, n) => `<tr><td>${n + 1}. ${esc(name)}</td>${d.mus[n].map((m, k) => `<td class="num">${fmt(m, 3)}</td><td class="num">${fmt(d.sigmas[n][k], 3)}</td>`).join('')}</tr>`);
  return `<div class="stats">${stat(fmt(d.loss[d.loss.length - 1], 3), 'final loss')}${stat(fmt(d.val_loss[d.val_loss.length - 1], 3), 'final val_loss')}${stat(`${fmt(d.mae, 2)} MPa`, 'validation MAE')}${stat(int(d.rules), 'rules (n_memb ^ 8)')}${stat(`${d.train} / ${d.validation}`, 'train / validation rows')}${stat(`${fmt(d.seconds, 1)} s`, 'run time')}</div>
    <div class="charts two">${chart({ title: 'Loss curves (history.history)', x: 'epoch', y: 'loss', series: [{ name: 'loss', xs: ep, ys: d.loss, color: '--c1' }, { name: 'val_loss', xs: ep, ys: d.val_loss, color: '--c2' }], dots: ep.length <= 30 })}
    ${chart({ title: 'Validation: predicted vs actual strength', x: 'actual, MPa', y: 'predicted, MPa', series: [{ name: 'y = x', xs: [lo, hi], ys: [lo, hi], color: '--muted', dash: true, width: 1.2 }, { name: 'rows 801–1000', xs: d.actual, ys: d.predicted, color: '--c3', scatter: true }] })}</div>
    ${sectionHead('Membership functions <small>· fis.plotmfs(show_initial_weights=True): trained, and initial dashed</small>')}<div class="charts two">${mfCharts}</div>
    <div style="margin-top:16px">${table(['Input', ...d.mus[0].flatMap((_, k) => [`num:μ${k + 1} mean`, `num:μ${k + 1} σ`])], rows, { title: 'Trained premise parameters' })}</div>`;
}

function lab7View(d) {
  const [x2lo, x2hi, ylo, yhi] = [d.x2[0], d.x2[d.x2.length - 1], d.y2[0], d.y2[d.y2.length - 1]];
  const flat = d.grid.flat(), zlo = Math.min(...flat), zhi = Math.max(...flat);
  const heat = (z) => { const t = (z - zlo) / (zhi - zlo || 1); return `hsl(${250 - 200 * t} 65% ${32 + 30 * t}%)`; };
  const surface = canvasBox('f₂(x, y) = x·sin(x + y) and the evolved population', (ctx, W, H) => {
    const dx = (x2hi - x2lo) / (d.x2.length - 1), dy = (yhi - ylo) / (d.y2.length - 1);
    // the population can leave the plotted grid (mutation has no bounds), so the axes cover both
    const px = [...d.pop2.map((p) => p[0]), d.max.x], py = [...d.pop2.map((p) => p[1]), d.max.y];
    const { X, Y } = canvasAxes(ctx, W, H, Math.min(x2lo - dx / 2, ...px), Math.max(x2hi + dx / 2, ...px), Math.min(ylo - dy / 2, ...py), Math.max(yhi + dy / 2, ...py), 'x', 'y', { square: true });
    d.grid.forEach((row, j) => row.forEach((z, i) => { ctx.fillStyle = heat(z); ctx.fillRect(X(d.x2[i] - dx / 2), Y(d.y2[j] + dy / 2), X(d.x2[i] + dx / 2) - X(d.x2[i] - dx / 2) + .5, Y(d.y2[j] - dy / 2) - Y(d.y2[j] + dy / 2) + .5); }));
    d.pop2.forEach(([x, y]) => { ctx.beginPath(); ctx.arc(X(x), Y(y), 3, 0, 2 * Math.PI); ctx.fillStyle = '#ffffff'; ctx.fill(); ctx.lineWidth = 1; ctx.strokeStyle = '#1b1f24'; ctx.stroke(); });
    ctx.beginPath(); ctx.arc(X(d.max.x), Y(d.max.y), 8, 0, 2 * Math.PI); ctx.lineWidth = 3; ctx.strokeStyle = css('--c5'); ctx.stroke();
    lab7View.map = { X, Y, dx, dy };
  }, { height: (w) => Math.min(460, Math.max(300, w * 0.7)), note: `white: population after evolve · ring: the maximum · colour: z from ${fmt(zlo, 2)} to ${fmt(zhi, 2)}`, csv: () => toCsv([['x', 'y', 'z'], ...d.grid.flatMap((row, j) => row.map((z, i) => [d.x2[i], d.y2[j], z]))]) });
  lab7View.data = { d, heat };
  return `<div class="stats">${stat(`x = ${fmt(d.min.x, 4)}`, 'f₁ minimum at')}${stat(fmt(d.min.y, 5), 'f₁(x)')}${stat(`(${fmt(d.max.x, 3)}, ${fmt(d.max.y, 3)})`, 'f₂ maximum at (x, y)')}${stat(fmt(d.max.z, 5), 'f₂(x, y)')}</div>
    <div class="charts two">${chart({ title: 'f₁(x) = sin|x|·cos(3x/2) and the evolved population', x: 'x', y: 'f₁(x)', height: 300, series: [{ name: 'f₁(x)', xs: d.x1, ys: d.f1, color: '--c1' }, { name: 'population', xs: d.pop1.map((p) => p[0]), ys: d.pop1.map((p) => p[1]), color: '--c2', scatter: true, r: 3.5 }, { name: 'minimum', xs: [d.min.x], ys: [d.min.y], color: '--c5', scatter: true, r: 6, opacity: 1 }] })}
    ${surface}</div>${printed(d.stdout)}`;
}

function lab8View(d) {
  lab8View.data = d;
  const last = d.frames[d.frames.length - 1];
  const rows = d.shapes.map(([w, h], i) => `<tr><td class="num">${i + 1}</td><td class="num">${fmt(w, 3)}</td><td class="num">${fmt(h, 3)}</td><td class="num">${fmt(last.centers[i][0], 3)}</td><td class="num">${fmt(last.centers[i][1], 3)}</td></tr>`);
  const solved = d.overlap === 0;
  return `<div class="stats">${stat(fmt(d.fitness, 6), 'fitness')}${stat(fmt(d.overlap, 6), 'overlap share s', solved ? 'good' : 'bad')}${stat(fmt(d.length, 4), 'connection length share l')}${stat(d.iterations[d.iterations.length - 1][0], 'generations')}${stat(d.shapes.length, 'rectangles')}${stat(`${fmt(d.seconds, 2)} s`, 'run time')}</div>
    <p class="note">${solved ? 'A placement without overlaps was found.' : 'The generation bound was reached; the best placement still overlaps.'}</p>
    <div class="two-col" style="margin-top:12px"><div class="panel" style="margin:0;padding:12px" data-copyable data-title="Best placement"><div class="copy-head"><h2 id="pack-label">Best placement</h2>${actions('png')}</div>
      <div class="player"><button type="button" class="mini" id="pack-play">Play</button><input type="range" id="pack-frame" min="0" max="${d.frames.length - 1}" value="${d.frames.length - 1}" aria-label="Generation"><span id="pack-gen"></span></div>
      <div class="plot" id="pack"></div><div class="legend"><span><span class="sw" style="background:color-mix(in srgb, var(--c1) 35%, transparent);border:1px solid var(--c1)"></span>rectangle</span><span><span class="sw" style="background:var(--err)"></span>overlap</span></div></div>
    <div>${chart({ title: 'Best fitness per generation', x: 'generation', y: 'fitness', series: [{ name: 'fitness', xs: d.iterations.map((p) => p[0]), ys: d.iterations.map((p) => p[1]), color: '--c3' }] })}
    <div style="margin-top:12px">${table(['#', 'num:width a', 'num:height b', 'num:x', 'num:y'], rows, { title: 'Rectangles and final centres', max: '260px' })}</div></div></div>
    ${printed(d.stdout)}`;
}

function drawPacking(el, d, frame) {
  const W = Math.max(240, Math.min(el.clientWidth, 520)), k = W / d.A, H = d.B * k;
  const rects = d.shapes.map(([w, h], i) => { const [x, y] = frame.centers[i]; return [x - w / 2, y - h / 2, w, h]; });
  const overlaps = [];
  for (let i = 0; i < rects.length; i++) for (let j = i + 1; j < rects.length; j++) {
    const [a, b] = [rects[i], rects[j]];
    const x0 = Math.max(a[0], b[0]), y0 = Math.max(a[1], b[1]), x1 = Math.min(a[0] + a[2], b[0] + b[2]), y1 = Math.min(a[1] + a[3], b[1] + b[3]);
    if (x1 > x0 && y1 > y0) overlaps.push([x0, y0, x1 - x0, y1 - y0]);
  }
  const Y = (y, h) => H - (y + h) * k;
  el.innerHTML = `<svg viewBox="-1 -1 ${W + 2} ${H + 2}" width="${W + 2}" height="${H + 2}" style="max-width:100%;height:auto" role="img" aria-label="Rectangle placement">
    <rect x="0" y="0" width="${W}" height="${H}" fill="var(--bg)" stroke="var(--muted)"/>
    ${rects.map(([x, y, w, h], i) => `<rect x="${x * k}" y="${Y(y, h)}" width="${w * k}" height="${h * k}" fill="var(--c1)" fill-opacity=".28" stroke="var(--c1)" stroke-width="1.2"><title>#${i + 1}: ${fmt(w, 2)} × ${fmt(h, 2)} at (${fmt(frame.centers[i][0], 2)}, ${fmt(frame.centers[i][1], 2)})</title></rect><text x="${(x + w / 2) * k}" y="${Y(y, h) + (h * k) / 2}" text-anchor="middle" dominant-baseline="middle" style="fill:var(--ink);font-size:10px">${i + 1}</text>`).join('')}
    ${overlaps.map(([x, y, w, h]) => `<rect x="${x * k}" y="${Y(y, h)}" width="${w * k}" height="${h * k}" fill="var(--err)" fill-opacity=".75"/>`).join('')}</svg>`;
}

function lab8Bind(out, d) {
  const el = $('#pack', out), slider = $('#pack-frame', out), gen = $('#pack-gen', out), play = $('#pack-play', out);
  const show = () => { const f = d.frames[+slider.value]; drawPacking(el, d, f); gen.textContent = `generation ${f.gen} · fitness ${fmt(f.fitness, 6)}`; };
  slider.oninput = show;
  let timer = null;
  play.onclick = () => {
    if (timer) { clearInterval(timer); timer = null; play.textContent = 'Play'; return; }
    if (+slider.value >= d.frames.length - 1) slider.value = 0;
    play.textContent = 'Pause';
    timer = setInterval(() => {
      if (!el.isConnected) return clearInterval(timer);
      slider.value = +slider.value + 1; show();
      if (+slider.value >= d.frames.length - 1) { clearInterval(timer); timer = null; play.textContent = 'Play'; }
    }, Math.max(30, 3000 / d.frames.length));
  };
  show();
  new ResizeObserver(debounce(show, 80)).observe(el);
}

function lab9View(d) {
  const ok = d.examples.filter((e) => e.predicted === e.label).length;
  const thumbs = d.examples.map((e, i) => `<div class="thumb ${e.predicted === e.label ? 'ok' : 'bad'}"><button type="button" data-ex="${i}" title="Load onto the drawing board"><canvas class="pixels" data-digit="${i}"></canvas></button>${esc(e.name)}<br>predicted <b>${e.predicted}</b> <span class="muted">(${fmt(e.probs[e.predicted] * 100, 1)}%)</span></div>`).join('');
  const ep = range(d.history_loss.length, (i) => i + 1);
  return `<div class="stats">${stat(`${fmt(d.accuracy * 100, 2)}%`, 'test accuracy (10 000 images)')}${stat(fmt(d.loss, 4), 'test loss')}${stat('50', 'epochs')}${stat(`${ok} / ${d.examples.length}`, 'BMP examples read right', ok > d.examples.length / 2 ? 'good' : 'bad')}</div>
    ${sectionHead('Draw a digit')}
    <div class="board" data-copyable data-title="Digit drawing"><div><canvas id="digit-pad" width="280" height="280" style="width:280px;height:280px;cursor:crosshair"></canvas>
      <div class="tools"><button type="button" class="mini" id="digit-clear">Clear</button><label class="field" style="flex-direction:row;align-items:center;gap:6px">Brush <select id="digit-brush"><option>14</option><option selected>20</option><option>28</option></select></label>${actions('png')}</div></div>
      <div class="side"><h2>recognizer.predict</h2><div class="bars" id="digit-bars"></div><p class="hint">Draw dark on white, like the BMP examples. The drawing is shrunk to 28×28 (shown in the corner), inverted and normalised the way <code>9-lab.py</code> prepares its images.</p><canvas id="digit-small" class="pixels" width="28" height="28" style="width:84px;height:84px;image-rendering:pixelated;border:1px solid var(--line);border-radius:6px"></canvas></div></div>
    ${sectionHead(`The lab's examples <small>· examples/*.bmp, ${ok} of ${d.examples.length} read right; click one to load it</small>`)}<div class="thumbs">${thumbs}</div>
    <div class="charts two" style="margin-top:16px">${chart({ title: 'Training accuracy per epoch', x: 'epoch', y: 'accuracy', series: [{ name: 'accuracy', xs: ep, ys: d.history_accuracy, color: '--c3' }] })}${chart({ title: 'Training loss per epoch', x: 'epoch', y: 'loss', series: [{ name: 'loss', xs: ep, ys: d.history_loss, color: '--c5' }] })}</div>`;
}

function lab9Bind(out, d) {
  const grey = (cv, px) => { cv.width = 28; cv.height = 28; const ctx = cv.getContext('2d'), img = ctx.createImageData(28, 28); px.forEach((v, i) => img.data.set([v, v, v, 255], i * 4)); ctx.putImageData(img, 0, 0); };
  $$('canvas[data-digit]', out).forEach((cv) => grey(cv, d.examples[+cv.dataset.digit].pixels));
  const pad = $('#digit-pad', out), ctx = pad.getContext('2d'), small = $('#digit-small', out), bars = $('#digit-bars', out);
  const clear = () => { ctx.fillStyle = '#ffffff'; ctx.fillRect(0, 0, 280, 280); };
  const shrink = () => {
    const tmp = document.createElement('canvas'); tmp.width = 28; tmp.height = 28;
    const t = tmp.getContext('2d'); t.imageSmoothingEnabled = true; t.imageSmoothingQuality = 'high';
    t.drawImage(pad, 0, 0, 28, 28);
    const data = t.getImageData(0, 0, 28, 28).data;
    const px = range(784, (i) => Math.round((data[i * 4] + data[i * 4 + 1] + data[i * 4 + 2]) / 3));
    grey(small, px);
    return px;
  };
  const predict = debounce(async () => {
    const probs = await runPy('lab9_predict', { pixels: shrink() });
    const top = probs.indexOf(Math.max(...probs));
    bars.innerHTML = probs.map((p, i) => `<div class="bar${i === top ? ' top' : ''}"><span>${i}</span><i style="width:${p * 100}%"></i><span>${fmt(p * 100, 1)}%</span></div>`).join('');
  }, 80);
  let down = false, last = null;
  const at = (e) => { const r = pad.getBoundingClientRect(); return [((e.clientX - r.left) / r.width) * 280, ((e.clientY - r.top) / r.height) * 280]; };
  const stroke = (a, b) => { ctx.strokeStyle = '#000000'; ctx.lineCap = 'round'; ctx.lineJoin = 'round'; ctx.lineWidth = +$('#digit-brush', out).value; ctx.beginPath(); ctx.moveTo(...a); ctx.lineTo(...b); ctx.stroke(); };
  pad.onpointerdown = (e) => { down = true; pad.setPointerCapture(e.pointerId); last = at(e); stroke(last, last); };
  pad.onpointermove = (e) => { if (!down) return; const p = at(e); stroke(last, p); last = p; };
  pad.onpointerup = pad.onpointercancel = () => { if (down) { down = false; predict(); } };
  $('#digit-clear', out).onclick = () => { clear(); predict(); };
  const load = (i) => {
    const tmp = document.createElement('canvas'); grey(tmp, d.examples[i].pixels);
    ctx.imageSmoothingEnabled = false; ctx.drawImage(tmp, 0, 0, 280, 280); predict();
  };
  $$('[data-ex]', out).forEach((b) => b.onclick = () => { load(+b.dataset.ex); pad.scrollIntoView({ block: 'nearest', behavior: 'smooth' }); });
  clear(); load(6);
}

function archView(d, name) {
  archView.data = d;
  const fmtShape = (s) => `(${['None', ...s].join(', ')})`;
  const groups = [];
  d.layers.forEach((l, i) => {
    const g = l.group?.id || `layer-${i}`;
    if (!groups.length || groups[groups.length - 1].id !== g) groups.push({ id: g, group: l.group, layers: [] });
    groups[groups.length - 1].layers.push(l);
  });
  const grouped = groups.some((g) => g.group);
  const rows = groups.flatMap((g) => {
    const rowsOf = g.layers.map((l) => `<tr data-g="${g.id}"${g.group ? ' hidden' : ''}><td>${g.group ? '&nbsp;&nbsp;&nbsp;' : ''}${esc(l.name)}</td><td>${l.kind}</td><td>${esc(l.config)}</td><td>${fmtShape(l.output)}</td><td class="num">${int(l.params)}</td><td>${esc(l.inbound.join(', '))}</td></tr>`);
    if (!g.group) return rowsOf;
    const out = g.layers[g.layers.length - 1].output, params = g.layers.reduce((s, l) => s + l.params, 0);
    return [`<tr class="group" data-toggle="${g.id}"><td><span class="caret">▸</span>${g.group.fn}() #${g.group.n}</td><td>${g.layers.length} layers</td><td></td><td>${fmtShape(out)}</td><td class="num">${int(params)}</td><td></td></tr>`, ...rowsOf];
  });
  const weighted = d.layers.filter((l) => l.params > 0);
  const maxP = Math.max(...weighted.map((l) => l.params));
  const W = 640, RH = 16, labelW = 170, H = weighted.length * RH + 24;
  const bars = `<svg viewBox="0 0 ${W} ${H}" width="${W}" height="${H}" role="img" aria-label="Parameters per layer" style="display:block">${weighted.map((l, i) => {
    const w = Math.max(1, (Math.log10(l.params + 1) / Math.log10(maxP + 1)) * (W - labelW - 90));
    const c = l.kind === 'Dense' ? '--c2' : l.kind === 'Conv2D' ? '--c1' : '--c3';
    return `<text x="${labelW - 6}" y="${i * RH + 12}" text-anchor="end">${esc(l.name)}</text><rect x="${labelW}" y="${i * RH + 3}" width="${w}" height="${RH - 5}" rx="3" fill="var(${c})"><title>${esc(l.name)}: ${int(l.params)} parameters</title></rect><text x="${labelW + w + 6}" y="${i * RH + 12}">${int(l.params)}</text>`;
  }).join('')}<text x="${labelW}" y="${H - 4}">log scale · blue Conv2D, orange Dense, green BatchNormalization</text></svg>`;
  const flow = grouped ? googleFlow(groups) : '';
  return `<div class="stats">${stat(int(d.total), 'total parameters')}${stat(int(d.trainable), 'trainable')}${stat(int(d.non_trainable), 'non-trainable (BN moving stats)')}${stat(d.layers.length, 'layers')}${stat(`${fmt(d.macs / 1e9, 3)} G`, 'multiply-adds per image')}${stat(`${fmt((d.total * 4) / 2 ** 20, 1)} MB`, 'float32 weights')}</div>
    ${flow}
    <div style="margin-top:16px">${table(['Layer', 'Type', 'Configuration', 'Output shape', 'num:Params', 'Input from'], rows, { title: `Layers <small>· as model.summary()${grouped ? '; click a module to expand it' : ''}</small>`, max: '560px' })}</div>
    <div class="panel" style="margin:16px 0 0;padding:14px;overflow:auto" data-copyable data-title="${name} parameters per layer"><div class="copy-head"><h2>Parameters per layer</h2>${actions('png')}</div><div class="plot" style="max-height:440px;overflow:auto">${bars}</div></div>
    ${d.lr ? `<div class="charts" style="margin-top:16px">${chart({ title: 'poly_decay learning rate per epoch', x: 'epoch', y: 'learning rate', series: [{ name: 'lr', xs: range(d.lr.length), ys: d.lr, color: '--c4' }] })}</div>` : ''}`;
}

function googleFlow(groups) {
  // one block per module, in order, with its output shape; inception blocks show their four branches
  const blocks = groups.map((g) => {
    const last = g.layers[g.layers.length - 1];
    const fn = g.group?.fn || last.kind;
    let detail = '';
    if (fn === 'inception_module') {
      const convs = g.layers.filter((l) => l.kind === 'Conv2D').map((l) => `${l.config.split(' × ')[0]}×${l.config.split(' × ')[1].split(',')[0]}`);
      detail = convs.join(' · ');
    } else if (fn === 'downsample_module') detail = g.layers.filter((l) => l.kind === 'Conv2D').map((l) => `${l.config.split(',')[0]} /2 + max-pool`).join('');
    else if (g.group) detail = g.layers[0].config.split(',').slice(0, 1).join('');
    else detail = last.config;
    const label = g.group ? `${fn.replace('_module', '')} ${fn === 'conv_module' ? '' : `#${g.group.n}`}` : last.kind;
    return { label, detail, shape: last.output.join('×'), params: g.layers.reduce((s, l) => s + l.params, 0), kind: fn };
  });
  const cls = (k) => (k === 'inception_module' ? '--c1' : k === 'downsample_module' ? '--c2' : k === 'conv_module' ? '--c3' : '--muted');
  return `<div class="panel" style="margin:0;padding:14px" data-copyable data-title="GoogLeNet modules"><div class="copy-head"><h2>Modules in order <small>· output shape and parameters</small></h2></div>
    <div style="display:flex;flex-wrap:wrap;gap:8px;align-items:stretch">${blocks.map((b) => `<div style="border:1px solid var(${cls(b.kind)});border-left-width:4px;border-radius:8px;padding:6px 9px;font-size:12px;min-width:118px;background:var(--bg)"><b style="font-size:13px">${esc(b.label)}</b><div class="muted">${esc(b.detail)}</div><div style="font-family:var(--mono)">${esc(b.shape)}</div><div class="muted">${int(b.params)} params</div></div>`).join('<span class="muted" style="align-self:center">→</span>')}</div></div>`;
}

function archBind(out) {
  $$('tr[data-toggle]', out).forEach((tr) => tr.onclick = () => {
    const open = tr.dataset.open !== '1';
    tr.dataset.open = open ? '1' : '';
    $('.caret', tr).textContent = open ? '▾' : '▸';
    $$(`tr[data-g="${tr.dataset.toggle}"]`, out).forEach((r) => { r.hidden = !open; });
  });
}

const meter = (p) => `<div class="meter" title="0 = negative, 1 = positive"><b style="left:${Math.min(100, Math.max(0, p * 100))}%"></b></div>`;
const tokenView = (words, ids, oov) => `<div class="tokens">${words.map((w, i) => `<span class="${ids[i] === oov ? 'oov' : ''}">${esc(w)}<small>${ids[i]}</small></span>`).join('')}</div>`;

function lab12View(d) {
  if (d.prob === null) return '<div class="err">No words to classify.</div>';
  const pos = d.logit >= 0;
  const ep = range(d.history.accuracy.length, (i) => i + 1);
  return `<div class="verdict"><div class="big ${pos ? 'ok' : 'bad'}">${pos ? 'Positive' : 'Negative'} <span class="muted">· sigmoid(logit) = ${fmt(d.prob, 4)}</span></div>${meter(d.prob)}</div>
    <div class="stats">${stat(fmt(d.logit, 4), 'logit (Dense(1))')}${stat(d.tokens, 'tokens')}${stat(d.unknown, '[UNK] tokens', d.unknown ? 'bad' : '')}${stat(int(d.vocab), 'vocabulary')}${stat(`${fmt(d.ms, 1)} ms`, 'inference time')}</div>
    <h2>TextVectorization output <small>· token and its index; red: out of vocabulary</small></h2>${tokenView(d.words, d.ids, 1)}
    <div class="charts two" style="margin-top:16px">${chart({ title: 'Model accuracy (history)', x: 'epoch', y: 'accuracy', series: [{ name: 'train', xs: ep, ys: d.history.accuracy, color: '--c1' }, { name: 'test', xs: ep, ys: d.history.val_accuracy, color: '--c2' }], dots: true })}
    ${chart({ title: 'Model loss (history)', x: 'epoch', y: 'loss', series: [{ name: 'train', xs: ep, ys: d.history.loss, color: '--c1' }, { name: 'test', xs: ep, ys: d.history.val_loss, color: '--c2' }], dots: true })}</div>`;
}

function examView(d) {
  if (d.value === undefined) return `<div class="big muted">value:<br>sentiment:</div><p class="hint">The ELSTM accumulators are back to the SavedModel's values.</p>`;
  const pos = d.sentiment === 'positive';
  return `<div class="verdict"><div class="big">value: ${fmt(d.value, 3)}<br>sentiment: <span class="${pos ? 'ok' : 'bad'}">${d.sentiment}</span></div>${meter(d.value)}</div>
    <div class="stats">${stat(d.tokens, 'tokens')}${stat(d.tokens - d.known, '&lt;OOV&gt; tokens', d.tokens - d.known ? 'bad' : '')}${stat(d.steps, 'ELSTM time steps (1000 / 4)')}${stat(d.calls, 'predictions since reset')}${stat(`${fmt(d.ms, 0)} ms`, 'inference time')}</div>
    ${d.truncated ? '<p class="note">The text has more than 1000 tokens; like <code>pad_sequences</code>, the model sees only the last 1000.</p>' : ''}
    <p class="hint">Like the app, each prediction runs a batch of 32 (the text and 31 empty strings), and the cell's h_sp, exp_sum and e_t accumulators keep the values the previous predictions left. <button type="button" class="mini" id="elstm-reset">Reset to the SavedModel state</button></p>
    <h2 style="margin-top:16px">Tokenizer output <small>· the text as str(bytes), as in training; token and its index, red: &lt;OOV&gt;</small></h2>${tokenView(d.words, d.ids, 1)}`;
}

// ---------------------------------------------------------------- sources --
const sources = new Map();
async function loadSource(path) {
  if (!sources.has(path)) sources.set(path, fetch(`../${path}`).then((r) => (r.ok ? r.text() : Promise.reject(new Error(`HTTP ${r.status}`)))));
  return sources.get(path);
}
function sourceBox(lab) {
  return `<details class="src"><summary>Source code: ${lab.files.map((f) => `<code>${esc(f)}</code>`).join(', ')}</summary>
    ${lab.files.length > 1 ? `<div class="tabs">${lab.files.map((f, i) => `<button type="button" class="tab" data-file="${esc(f)}" aria-selected="${i === 0}">${esc(f.split('/').pop())}</button>`).join('')}</div>` : '<div style="height:8px"></div>'}
    <div data-copyable data-title="${esc(lab.files[0])}"><div class="copy-head"><a class="src-link" href="${BLOB}/${lab.files[0]}" target="_blank" rel="noopener">${esc(lab.files[0])} on GitHub</a>${actions('code')}</div><pre class="console"><code>Loading…</code></pre></div></details>`;
}
function bindSource(sec, lab) {
  const det = $('details.src', sec), pre = $('pre code', det), link = $('.src-link', det), box = $('[data-copyable]', det);
  const show = async (file) => {
    $$('.tab', det).forEach((t) => t.setAttribute('aria-selected', t.dataset.file === file));
    link.href = `${BLOB}/${file}`; link.textContent = `${file} on GitHub`; box.dataset.title = file;
    pre.textContent = 'Loading…';
    try {
      const text = await loadSource(file);
      pre.textContent = file.endsWith('dataset.txt') ? `${text.split('\n').slice(0, 40).join('\n')}\n… (${text.split('\n').length} lines)` : text;
    } catch (e) { pre.textContent = `Could not load ${file}: ${e.message}`; }
  };
  det.addEventListener('toggle', () => { if (det.open && pre.textContent === 'Loading…') show(lab.files[0]); });
  $$('.tab', det).forEach((t) => t.onclick = () => show(t.dataset.file));
}

// --------------------------------------------------------------- framework --
const nav = $('#nav'), main = $('#labs');
const state = new Map(); // lab id -> { ran, running }

function fieldHtml(lab, f) {
  const id = `${lab.id}-${f.key}`;
  if (f.options) return `<label class="field" for="${id}">${esc(f.label)}<select id="${id}" name="${f.key}">${f.options.map((o) => `<option${o === f.value ? ' selected' : ''}>${o}</option>`).join('')}</select></label>`;
  if (f.type === 'textarea') return `<label class="field" for="${id}" style="flex:1 1 100%">${esc(f.label)}<textarea class="text" id="${id}" name="${f.key}" spellcheck="true">${esc(f.value)}</textarea></label>`;
  const type = f.type || 'number';
  const attrs = type === 'number' ? ` min="${f.min}" max="${f.max}" step="${f.step || 1}" required` : ` spellcheck="false"${f.placeholder ? ` placeholder="${f.placeholder}"` : ''}`;
  return `<label class="field${f.wide ? ' wide' : ''}" for="${id}">${esc(f.label)}<input id="${id}" name="${f.key}" type="${type}" value="${esc(f.value)}"${attrs}></label>`;
}

// review examples: the exam app's two (from main.py's docstring) and a few short ones
const SHORT = [['Positive', 'One of the best films I have seen this year. The story is moving, the acting is brilliant and the music is wonderful.'],
  ['Negative', 'A boring, badly written mess. The acting is awful and the plot makes no sense. A complete waste of time.'],
  ['Mixed', 'The first half is great fun, but the ending is weak and far too long.']];
async function examples(lab, sec) {
  const box = $('.examples', sec);
  let list = SHORT;
  try {
    const src = await loadSource('exam-task/app/main.py');
    const neg = src.split('[Negative]:\n')[1].split('\n\n[Positive]')[0].replaceAll("\\'", "'");
    const pos = src.split('[Positive]:\n')[1].split('\n"""')[0].replaceAll("\\'", "'");
    list = [['main.py negative', neg], ['main.py positive', pos], ...SHORT];
  } catch { /* the short examples still work */ }
  box.innerHTML = `<span>Try</span>${list.map(([name], i) => `<button type="button" class="chip" data-i="${i}">${esc(name)}</button>`).join('')}`;
  $$('.chip', box).forEach((c) => c.onclick = () => { $('textarea', sec).value = list[+c.dataset.i][1]; run(lab); });
  return list;
}

function build() {
  for (const lab of LABS) {
    const b = document.createElement('button');
    b.type = 'button'; b.role = 'tab'; b.dataset.lab = lab.id;
    b.setAttribute('aria-controls', `panel-${lab.id}`);
    b.innerHTML = `<span class="lang py">Py</span>${labTitle(lab).replace('Exam task', 'Exam')}<span class="long muted" style="font-weight:400">· ${esc(lab.name)}</span>`;
    b.onclick = () => select(lab.id, true);
    nav.appendChild(b);

    const sec = document.createElement('section');
    sec.className = 'panel'; sec.id = `panel-${lab.id}`; sec.role = 'tabpanel'; sec.hidden = true;
    const main0 = lab.files[0];
    const extra = lab.id === 'exam' ? '<button class="mini" type="button" id="exam-clear" style="padding:8px 16px;font-size:14px">Clear</button>' : '';
    sec.innerHTML = `<div class="lab-head"><span class="lang py">Python</span>${lab.port ? `<span class="lang port">${esc(lab.port)}</span>` : ''}<h3>${labTitle(lab)}: ${esc(lab.name)}</h3><a href="${lab.files.length > 1 ? `${TREE}/${main0.split('/').slice(0, -1).join('/')}` : `${BLOB}/${main0}`}" target="_blank" rel="noopener">View source</a></div>
      <p class="intro">${lab.intro}</p>
      <div class="notes"><p class="note port">${lab.runtime}</p></div>
      <form class="params" novalidate>${lab.fields.map((f) => fieldHtml(lab, f)).join('')}<button class="run" type="submit">${lab.runLabel || 'Run'}</button>${extra}</form>
      ${lab.examples ? '<div class="examples"></div>' : ''}
      ${lab.hint ? `<p class="hint">${lab.hint}</p>` : ''}
      <div class="out" style="margin-top:18px"></div>
      ${sourceBox(lab)}`;
    main.appendChild(sec);
    state.set(lab.id, { ran: false, running: false });
    const form = $('form', sec);
    form.addEventListener('submit', (e) => { e.preventDefault(); run(lab); });
    if (lab.examples) examples(lab, sec).then((list) => { if (lab.id === 'exam' && !$('textarea', sec).value) $('textarea', sec).value = list[0][1]; });
    if (lab.id === 'exam') {
      $('#exam-clear', sec).onclick = () => { $('textarea', sec).value = ''; $('.out', sec).innerHTML = '<div class="big muted">value:<br>sentiment:</div>'; };
      sec.addEventListener('click', (e) => { if (e.target.id === 'elstm-reset') run(lab, { reset: true }); });
    }
    bindSource(sec, lab);
    if (!pyReady) $('.out', sec).before(document.getElementById('py-loading').content.cloneNode(true));
  }
}

function params(lab) {
  const form = $(`#panel-${lab.id} form`);
  const fd = new FormData(form), p = {};
  for (const f of lab.fields) {
    const v = fd.get(f.key);
    p[f.key] = f.type === 'text' || f.type === 'textarea' || f.options ? v : Number(v);
  }
  return p;
}

function invalid(lab, p) {
  for (const f of lab.fields) {
    if (f.type === 'text' || f.type === 'textarea' || f.options) continue;
    const v = p[f.key];
    if (!Number.isFinite(v) || v < f.min || v > f.max || (f.step !== 'any' && !Number.isInteger(v))) return `${f.label} must be ${f.step === 'any' ? 'a number' : 'an integer'} from ${f.min} to ${f.max}.`;
  }
  if (lab.id === 'lab8' && String(p.seed).trim() !== '' && !/^\d{1,9}$/.test(String(p.seed).trim())) return 'The seed must be empty or a whole number.';
  if ((lab.id === 'lab12' || lab.id === 'exam') && !String(p.text).trim()) return lab.id === 'exam' ? null : 'Enter some text.';
  return null;
}

async function run(lab, extra = {}) {
  const st = state.get(lab.id);
  if (st.running) return;
  const sec = $(`#panel-${lab.id}`), btn = $('.run', sec), out = $('.out', sec);
  const p = { ...params(lab), ...extra };
  const bad = invalid(lab, p);
  if (bad) { out.innerHTML = `<div class="err">${esc(bad)}</div>`; return; }
  if (lab.id === 'exam' && !p.text.trim() && !p.reset) return;
  st.running = true; st.ran = true;
  // Python runs in the worker, so the page stays responsive; show the elapsed time and progress
  const t0 = Date.now();
  let note = '';
  const label = () => (!pyReady ? 'Waiting for Python…' : `${note && note.startsWith('Loading') ? note : `Running… ${Math.floor((Date.now() - t0) / 1000)} s${note ? ` · ${note}` : ''}`}`);
  btn.disabled = true; btn.textContent = label();
  const timer = setInterval(() => { btn.textContent = label(); }, 250);
  out.style.opacity = '.45';
  try {
    const data = await lab.run(p, (text) => { note = text; btn.textContent = label(); });
    out.innerHTML = `<div class="out-actions">${actions('results')}</div>` + lab.render(data, p);
    drawCharts(out);
    lab.after?.(out, data);
    if (lab.id === 'lab3') bindLab3(out);
    if (lab.id === 'lab7') bindLab7(out);
  } catch (e) {
    console.error(e);
    out.innerHTML = `<div class="err">${esc(e.message || e)}</div>`;
  } finally {
    clearInterval(timer);
    st.running = false; btn.disabled = false; btn.textContent = lab.runLabel || 'Run'; out.style.opacity = '';
  }
}

function bindLab3(out) {
  const box = $('[data-canvas]', out), d = lab3View.data;
  canvasTip(box, (x, y) => {
    const { X, Y } = lab3View.map;
    let best = -1, bd = 64;
    d.centers.forEach(([cx, cy], i) => { const dd = (X(cx) - x) ** 2 + (Y(cy) - y) ** 2; if (dd < bd) { bd = dd; best = i; } });
    if (best >= 0) return `<div><b>Centre ${best + 1}</b></div><div class="tx">(${fmt(d.centers[best][0], 4)}, ${fmt(d.centers[best][1], 4)})</div><div>${d.sizes[best]} points</div>`;
    let bi = -1; bd = 36;
    for (let i = 0; i < d.x.length; i++) { const dd = (X(d.x[i]) - x) ** 2 + (Y(d.y[i]) - y) ** 2; if (dd < bd) { bd = dd; bi = i; } }
    return bi < 0 ? null : `<div class="tx">(${fmt(d.x[bi], 4)}, ${fmt(d.y[bi], 4)})</div><div>cluster ${d.labels[bi] + 1}</div>`;
  });
}
function bindLab7(out) {
  const box = $('[data-canvas]', out), { d } = lab7View.data;
  canvasTip(box, (x, y) => {
    const { X, Y } = lab7View.map;
    const i = d.x2.findIndex((v) => Math.abs(X(v) - x) <= (X(d.x2[1]) - X(d.x2[0])) / 2);
    const j = d.y2.findIndex((v) => Math.abs(Y(v) - y) <= (Y(d.y2[0]) - Y(d.y2[1])) / 2);
    if (i < 0 || j < 0) return null;
    return `<div class="tx">x = ${fmt(d.x2[i], 3)}, y = ${fmt(d.y2[j], 3)}</div><div>z = ${fmt(d.grid[j][i], 4)}</div>`;
  });
}

function select(id, push) {
  const lab = LABS.find((l) => l.id === id) || LABS[0];
  nav.querySelectorAll('button').forEach((b) => b.setAttribute('aria-selected', b.dataset.lab === lab.id));
  main.querySelectorAll('section.panel[role="tabpanel"]').forEach((s) => s.hidden = s.id !== `panel-${lab.id}`);
  if (push) try { history.replaceState(null, '', `#${lab.id}`); } catch {}
  document.title = `${labTitle(lab)} · AI Basics Labs`;
  if (!state.get(lab.id).ran) {
    if (lab.id === 'exam') {
      // wait for the main.py example to land in the text box
      const ta = $(`#panel-${lab.id} textarea`);
      const go = () => (ta.value ? run(lab) : setTimeout(go, 100));
      go();
    } else run(lab);
  } else drawCharts($(`#panel-${lab.id}`));
}

build();
// elapsed counter for the Python loading notices
const started = Date.now();
const tick = setInterval(() => {
  if (pyReady) return clearInterval(tick);
  const sec = Math.round((Date.now() - started) / 1000);
  $$('.py-status').forEach((el) => {
    if (el.classList.contains('slow') && sec < 45) return;
    $('.elapsed', el).textContent = sec ? `${sec} s` : '';
    if (sec >= 45) { el.classList.add('slow'); $('.note-text', el).textContent = 'This is taking longer than usual. Check your connection, or reload the page.'; }
  });
}, 1000);
select(location.hash.slice(1), false);
window.addEventListener('hashchange', () => select(location.hash.slice(1), false));
