// Web Worker that runs the Python (Pyodide) off the page's main thread, so long runs never freeze the UI.
// It loads the packages and files listed in pyodide.toml (PyScript's [files] format), then bridge.py, and answers
// JSON messages: in {id, type: 'run' | 'need' | 'prepare', name, params},
// out {type: 'ready'} | {type: 'failed', error} | {id, progress} | {id, result} | {id, error}.
import { loadPyodide } from 'https://cdn.jsdelivr.net/pyodide/v314.0.3/full/pyodide.mjs';  // the Pyodide PyScript 2026.7.3 uses

const HOME = '/home/pyodide';
let current = 0;
// Python calls js.pyProgress(json) during a run; the page shows it next to the elapsed time
self.pyProgress = (json) => postMessage({ id: current, progress: JSON.parse(json) });
const fetchOk = async (url) => {
  const res = await fetch(url);
  if (!res.ok) throw new Error(`Could not load ${url} (HTTP ${res.status})`);
  return res;
};
const boot = (async () => {
  const py = await loadPyodide();
  py.FS.writeFile(`${HOME}/pyodide.toml`, await (await fetchOk('./pyodide.toml')).text());
  const cfg = JSON.parse(py.runPython(`import json, tomllib; json.dumps(tomllib.load(open('pyodide.toml', 'rb')))`));
  const { packages = [], files = {} } = cfg;
  // [files] syntax: keys like "{FROM}" define placeholders used by the other entries.
  const vars = Object.entries(files).filter(([k]) => /^\{\w+\}$/.test(k));
  const sub = (s) => vars.reduce((acc, [k, v]) => acc.replaceAll(k, v), s);
  const entries = Object.entries(files).filter(([k]) => !/^\{\w+\}$/.test(k)).map(([src, dest]) => [sub(src), sub(dest)]);
  await Promise.all([py.loadPackage(packages), ...entries.map(async ([src, dest]) => {
    const path = `${HOME}/${dest.replace(/^\.\//, '')}`;
    py.FS.mkdirTree(path.slice(0, path.lastIndexOf('/')));
    py.FS.writeFile(path, new Uint8Array(await (await fetchOk(src)).arrayBuffer()));
  })]);
  py.runPython(`import sys\nif '${HOME}' not in sys.path: sys.path.insert(0, '${HOME}')`);
  await py.runPythonAsync(await (await fetchOk('./bridge.py')).text());
  return py.globals;
})();

boot.then(() => postMessage({ type: 'ready' }), (e) => postMessage({ type: 'failed', error: String(e?.message || e) }));

// one message at a time, so progress reports belong to the run in flight
let queue = Promise.resolve();
onmessage = ({ data: { id, type, name, params } }) => {
  queue = queue.then(async () => {
    try {
      const g = await boot;
      current = id;
      const result = type === 'need' ? g.get('need')(name) : type === 'prepare' ? await g.get('prepare')(name) : g.get('run')(name, params);
      postMessage({ id, result });
    } catch (e) {
      postMessage({ id, error: String(e?.message || e) });
    }
  });
};
