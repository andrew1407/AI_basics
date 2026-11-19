// Starts worker.js (Pyodide in a module Web Worker) so Python never blocks the page, then exposes
// window.py(type, name, params) -> Promise<string> and fires 'py-ready' (or 'py-failed').
// Progress reports from a run go to the onProgress callback passed with it.
// Calls are plain JSON postMessage round trips, so no COOP/COEP headers or SharedArrayBuffer are needed.
const worker = new Worker(new URL('./worker.js', import.meta.url), { type: 'module' });
const calls = new Map();
let seq = 0;

window.py = (type, name, params, onProgress) => new Promise((resolve, reject) => {
  const id = ++seq;
  calls.set(id, { resolve, reject, onProgress });
  worker.postMessage({ id, type, name, params });
});

worker.onmessage = ({ data }) => {
  if (data.type === 'ready') return window.dispatchEvent(new CustomEvent('py-ready'));
  if (data.type === 'failed') {
    console.error(data.error);
    return window.dispatchEvent(new CustomEvent('py-failed', { detail: data.error }));
  }
  const call = calls.get(data.id);
  if (!call) return;
  if ('progress' in data) return call.onProgress?.(data.progress);
  calls.delete(data.id);
  'error' in data ? call.reject(new Error(data.error)) : call.resolve(data.result);
};
worker.onerror = (e) => {
  console.error(e.message || e);
  window.dispatchEvent(new CustomEvent('py-failed', { detail: e.message }));
};
