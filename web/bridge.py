# Worker entry (worker.js runs this in Pyodide): need(name), prepare(name) and run(name, params) for the page, JSON in and out.
# Packages and data files are fetched the first time a lab needs them, so the page starts with numpy alone.
import json
import os
from pyodide.http import pyfetch
import pyodide_js

HOME = os.getcwd()
FUZZY = ['scipy', 'networkx', 'matplotlib', 'pypi:scikit-fuzzy==0.5.0']
PACKAGES = dict(lab1=FUZZY, lab2=FUZZY, lab3=FUZZY, lab4=['matplotlib', 'wheel:neurolab-0.3.5-py3-none-any.whl'], lab5=['pillow'],
                lab6=['pandas', 'xlrd'], lab7=['matplotlib', 'pypi-nodeps:evol==0.5.3'], lab8=['matplotlib'], lab9=['pillow'])
PACKAGES['lab5_identify'], PACKAGES['lab9_predict'] = PACKAGES['lab5'], PACKAGES['lab9']
LETTERS = [f'{c}_{k}.bmp' for c in 'anir' for k in ('valid', 'invalid')]
DIGITS = ['0_1', '0_2', '1_1', '1_2', '2_1', '2_2', '3', '4_1', '4_2', '4_3', '5_1', '5_2', '6_1', '6_2', '7_1', '7_2', '8_1', '8_2', '9_1', '9_2']
# paths relative to web/; ../<lab>/... lands in labs/<lab>/... next to the lab's sources
DATA = dict(lab3=['../3-lab/dataset.txt'], lab5=[f'../5-lab/letters/{f}' for f in LETTERS], lab6=['../6-lab/Concrete_Data.xls'],
            lab9=[f'../9-lab/examples/{f}.bmp' for f in DIGITS] + ['models/mnist.npz'], lab12=['models/lab12.npz'],
            exam=['models/elstm.npz', '../exam-task/model/word_index.json'])
ready = set()


def local(path):
  return os.path.join(HOME, 'labs', path[3:]) if path.startswith('../') else os.path.join(HOME, path)


async def install(pkg):
  if pkg in ready: return
  if ':' in pkg:
    kind, spec = pkg.split(':', 1)
    await pyodide_js.loadPackage('micropip')
    import micropip
    if kind == 'wheel':  # built by build_wheels.sh
      import js
      await micropip.install(str(js.URL.new(f'wheels/{spec}', js.location.href).href))
    else:
      await micropip.install(spec, deps=kind != 'pypi-nodeps')
  else:
    await pyodide_js.loadPackage(pkg)
  ready.add(pkg)


def missing(name):
  return [p for p in PACKAGES.get(name, []) if p not in ready] + [p for p in DATA.get(name, []) if not os.path.exists(local(p))]


async def prepare(name):
  try:
    for pkg in PACKAGES.get(str(name), []): await install(pkg)
    for path in DATA.get(str(name), []):
      dest = local(path)
      if os.path.exists(dest): continue
      res = await pyfetch(path)
      if not res.ok: raise RuntimeError(f'Could not load {path} (HTTP {res.status})')
      os.makedirs(os.path.dirname(dest), exist_ok=True)
      with open(dest, 'wb') as f: f.write(await res.bytes())
    return ''
  except Exception as e:
    return f'{type(e).__name__}: {e}'


def need(name):
  """What a lab still has to download, for the page's status line."""
  names = []
  for p in missing(str(name)):
    p = p.split(':', 1)[-1].split('==')[0].split('-0.')[0]
    names.append(os.path.basename(p))
  return ', '.join(dict.fromkeys(names))


def plain(o):
  return o.item() if hasattr(o, 'item') else str(o)


import showcase


def run(name, params):
  try:
    return json.dumps(dict(ok=True, data=showcase.RUNNERS[str(name)](**json.loads(str(params)))), default=plain)
  except Exception as e:  # surface failures in the UI instead of hanging it
    return json.dumps(dict(ok=False, error=f'{type(e).__name__}: {e}'))
