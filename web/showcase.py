"""Browser-side runners for the GitHub Pages showcase (run by bridge.py in a Pyodide Web Worker).

The lab scripts are written to run top to bottom: they compute, print and open matplotlib windows.
run_script() executes a script's statements in order, the way `python N-lab.py` would, with some
constants or keyword arguments replaced by the page's inputs and, where the page draws its own charts,
the plotting statements left out. Each runner returns JSON-ready data for app.js.

Where a lab needs a library the browser can't run (Theano/Lasagne, TensorFlow), the module in ports/
stands in for it; see each port's docstring.
"""
import ast
import base64
import contextlib
import io
import json
import math
import os
import re
import sys
import time
import types
import warnings

import numpy as np

warnings.filterwarnings('ignore')
os.environ['MPLBACKEND'] = 'Agg'  # set before anything imports matplotlib.pyplot (there's no display in a worker)
HOME = os.getcwd()
LABS = os.path.join(HOME, 'labs')
sys.path.insert(0, os.path.join(HOME, 'ports'))

try:
  import js  # the worker's global scope: pyProgress() reports progress for the current run
except ImportError:  # running natively
  js = None


def progress(done, text=''):
  if js is not None and hasattr(js, 'pyProgress'): js.pyProgress(json.dumps(dict(done=done, text=text)))


def r(v, d=6):
  v = float(v)
  return round(v, d) if math.isfinite(v) else None


def rl(xs, d=6):
  return [r(v, d) for v in np.asarray(xs, float).ravel()]


@contextlib.contextmanager
def in_lab(folder):
  """cwd and sys.path set to the lab's folder, as when the script is run from there."""
  cwd = os.getcwd()
  path = os.path.join(LABS, folder)
  os.chdir(path)
  sys.path.insert(0, path)
  try:
    yield path
  finally:
    os.chdir(cwd)
    sys.path.remove(path)


class _Kwargs(ast.NodeTransformer):
  def __init__(self, values):
    self.values = values

  def visit_Call(self, node):
    self.generic_visit(node)
    for kw in node.keywords:
      if kw.arg in self.values: kw.value = ast.parse(repr(self.values[kw.arg]), mode='eval').body
    return node


def run_script(path, consts=None, kwargs=None, skip=None, before=None, stop=None):
  """Runs the script's top-level statements in order and returns (namespace, printed output).

  consts: NAME -> value, applied right after the script assigns NAME (later statements see the new value)
  kwargs: keyword -> value, replaced in every call that passes that keyword
  skip(src): True for statements to leave out (e.g. plotting); before(src, ns): hook run before each statement
  stop(src): True for the statement to stop before
  """
  consts, kwargs = consts or {}, kwargs or {}
  tree = ast.parse(open(path).read(), path)
  if kwargs: tree = ast.fix_missing_locations(_Kwargs(kwargs).visit(tree))
  ns = dict(__name__='__main__', __file__=path)
  out = io.StringIO()
  with contextlib.redirect_stdout(out):
    for node in tree.body:
      src = ast.unparse(node)
      if stop and stop(src): break
      if skip and skip(src): continue
      if before: before(src, ns)
      exec(compile(ast.Module([node], type_ignores=[]), path, 'exec'), ns)
      if isinstance(node, ast.Assign):
        for t in node.targets:
          if isinstance(t, ast.Name) and t.id in consts: ns[t.id] = consts[t.id]
  return ns, out.getvalue()


def load_defs(path, names=()):
  """Only a script's imports, functions, classes and the named assignments (no top-level runs)."""
  tree = ast.parse(open(path).read(), path)
  keep = []
  for node in tree.body:
    if isinstance(node, (ast.Import, ast.ImportFrom, ast.FunctionDef, ast.ClassDef)): keep.append(node)
    elif isinstance(node, ast.Assign) and any(isinstance(t, ast.Name) and t.id in names for t in node.targets): keep.append(node)
  return keep


def plotting(src):
  """Statements that only draw with matplotlib (the page draws these figures itself)."""
  return re.search(r'\b(plt|ax|fig)\.', src) is not None or src.startswith('fig, ax')


# ------------------------------------------------------------- matplotlib --
MUTED = '#8b95a3'


def mpl():
  import matplotlib
  matplotlib.use('Agg')
  import matplotlib.pyplot as plt
  plt.close('all')
  # transparent figures with grey text read on both page themes
  plt.rcParams.update({'figure.facecolor': 'none', 'axes.facecolor': 'none', 'savefig.facecolor': 'none', 'savefig.transparent': True,
                       'text.color': MUTED, 'axes.labelcolor': MUTED, 'axes.edgecolor': MUTED, 'xtick.color': MUTED, 'ytick.color': MUTED,
                       'axes.titlecolor': MUTED, 'legend.frameon': False, 'font.size': 10, 'figure.max_open_warning': 0})
  return plt


def png(fig, dpi=100):
  buf = io.BytesIO()
  fig.savefig(buf, format='png', dpi=dpi, bbox_inches='tight')
  return base64.b64encode(buf.getvalue()).decode()


def figure_title(fig):
  for ax in fig.axes:
    if ax.get_title(): return ax.get_title()
  return fig._suptitle.get_text() if getattr(fig, '_suptitle', None) else ''


def open_figures(plt):
  out = []
  for num in plt.get_fignums():
    fig = plt.figure(num)
    if not any(ax.has_data() for ax in fig.axes): continue
    out.append(dict(title=figure_title(fig), png=png(fig)))
  plt.close('all')
  return out


# ------------------------------------------------------------------ lab 1 --
def lab1(inputs, outputs, step):
  plt = mpl()
  with in_lab('1-lab'):
    import control_system, tables
    ns, _ = run_script('1-lab.py', consts=dict(IN_AMOUNT=inputs, OUT_AMOUNT=outputs, VAL_RANGE=(0, 1, step)),
                       stop=lambda s: s.startswith('simulator.simulate_variants'))
    runs = []
    original = control_system.simulate2d

    def recorded(system, input_range):
      progress(len(runs) / 6, f'Simulating system {len(runs) + 1} of 6')
      res = original(system=system, input_range=input_range)
      runs.append(res)
      return res

    control_system.simulate2d = recorded
    out = io.StringIO()
    try:
      with contextlib.redirect_stdout(out):
        ns['simulator'].simulate_variants()
    finally:
      control_system.simulate2d = original
    sim = ns['simulator']
    xs = sim._Simulator__plot_range
    fn = ns['VARIANT_FN']
    names = tables.get_fns_table(amounts=(inputs, outputs), val_range=(0, 1, step), out_fn=fn)
    diag = {y: dict(row) for y, row in names.items()}
    tables.left_diagonal_values(diag)
  lines = out.getvalue().strip().splitlines()
  errs = [dict(title=l.split(' err = ')[0], err=r(l.split(' err = ')[1].rstrip('%'))) for l in lines if ' err = ' in l]
  return dict(x=rl(xs, 4), f=rl([fn(x, x) for x in xs]), runs=[dict(title=e['title'], z=rl(z)) for e, z in zip(errs, runs)],
              errors=errs, stdout=out.getvalue(), figures=open_figures(plt), table=names, diagonal=diag)


# ------------------------------------------------------------------ lab 2 --
def lab2(step, center, sigma, fn1, fn2):
  plt = mpl()
  saved = []
  original = plt.savefig

  def savefig(fname, **kw):
    saved.append(dict(title=os.path.splitext(os.path.basename(str(fname)))[0], png=png(plt.gcf(), kw.get('dpi', 100))))

  plt.savefig = savefig
  try:
    with in_lab('2-lab'):
      ns, printed = run_script('2-lab.py', consts=dict(RANGE=(0, 1, step), CENTER=center, SIGMA=sigma),
                               skip=lambda s: s.startswith('operators_interpretation('))
      fuzz = ns['fuzz']
      ns['operators_interpretation'](getattr(fuzz, fn1), getattr(fuzz, fn2))
  finally:
    plt.savefig = original
  plt.close('all')
  x = ns['np_range']
  curves = [dict(name=f.__name__, args=repr(a), y=rl(f(x, *a))) for f, a in ns['FUNCTIONS'].items()]
  return dict(x=rl(x, 4), curves=curves, figures=saved)


# ------------------------------------------------------------------ lab 3 --
def lab3(clusters, error, maxiter):
  mpl()
  import skfuzzy
  rec = {}
  original = skfuzzy._cluster.cmeans

  def cmeans(*args, **kw):
    res = original(*args, **kw)
    rec['res'] = res
    return res

  # plot_coords draws its 5000 points with one plt.plot call each; the page draws them in one go instead
  plt_stub = types.SimpleNamespace(**{k: (lambda *a, **kw: None) for k in ('title', 'plot', 'figure', 'xlabel', 'ylabel', 'show')})
  skfuzzy._cluster.cmeans = cmeans
  try:
    with in_lab('3-lab'):
      t0 = time.perf_counter()
      ns, printed = run_script('3-lab.py', consts=dict(CLUSTERS_AMOUNT=clusters), kwargs=dict(error=error, maxiter=maxiter),
                               before=lambda src, ns: ns.update(plt=plt_stub) if src.startswith('plot_coords(') else None)
      elapsed = time.perf_counter() - t0
      data = np.vstack(ns['read_dataset'](ns['FILENAME']))
  finally:
    skfuzzy._cluster.cmeans = original
  cntr, u, u0, d, jm, p, fpc = rec['res']
  labels = np.argmax(u, axis=0)
  return dict(x=rl(data[0], 4), y=rl(data[1], 4), labels=labels.tolist(), centers=[rl(c, 4) for c in cntr], history=rl(jm),
              iterations=int(p), fpc=r(fpc), seconds=r(elapsed, 3), sizes=np.bincount(labels, minlength=len(cntr)).tolist())


# ------------------------------------------------------------------ lab 4 --
def lab4(epochs, learning_rate, elman_epochs, goal):
  import backprops_np
  mpl()
  with in_lab('4-lab'):
    defs = load_defs('backprops.py')
    # backprops.py imports neurolab, theano and lasagne; the Elman function runs as written with neurolab,
    # the Theano/Lasagne networks come from ports/backprops_np.py
    keep = [n for n in defs if not (isinstance(n, ast.Import) and n.names[0].name in ('theano', 'theano.tensor', 'lasagne'))]
    keep = [n for n in keep if not (isinstance(n, ast.FunctionDef) and n.name in ('cascade_forward_backprop', 'feed_forward_backprop', '__handle_train'))]
    probs = types.ModuleType('backprops')
    exec(compile(ast.Module(keep, type_ignores=[]), 'backprops.py', 'exec'), probs.__dict__)
    probs.feed_forward_backprop = backprops_np.feed_forward_backprop
    probs.cascade_forward_backprop = backprops_np.cascade_forward_backprop
    sys.modules['backprops'] = probs
    ns, _ = run_script('4-lab.py', consts=dict(EPOCHS=epochs, LEARNING_RATE=learning_rate, GOAL=goal),
                       stop=lambda s: s.startswith('z_predicted10'))
  XY, Z, P = ns['XY'], ns['Z'], ns['probs']
  nets = [('ffb_10', 'Feed-forward [10]', lambda: P.feed_forward_backprop(XY, Z, hidden_num=[10], epochs=epochs, learning_rate=learning_rate)),
          ('ffb_20', 'Feed-forward [20]', lambda: P.feed_forward_backprop(XY, Z, hidden_num=[20], epochs=epochs, learning_rate=learning_rate)),
          ('cfb_20', 'Cascade-forward [20]', lambda: P.cascade_forward_backprop(XY, Z, hidden_num=[20], epochs=epochs, learning_rate=learning_rate)),
          ('cfb_10_10', 'Cascade-forward [10, 10]', lambda: P.cascade_forward_backprop(XY, Z, hidden_num=[10, 10], epochs=epochs, learning_rate=learning_rate)),
          ('el_15', 'Elman [15]', lambda: P.elman_recurrent_backprop(XY, Z, hidden_num=[15], epochs=elman_epochs, goal=goal)),
          ('el_5_5_5', 'Elman [5, 5, 5]', lambda: P.elman_recurrent_backprop(XY, Z, hidden_num=[5, 5, 5], epochs=elman_epochs, goal=goal))]
  out = []
  for i, (key, title, train) in enumerate(nets):
    progress(i / len(nets), f'Training {title}')
    t0 = time.perf_counter()
    z, e = train()
    out.append(dict(key=key, title=title, z=rl(z), errors=rl(e), seconds=r(time.perf_counter() - t0, 3), engine='neurolab' if key.startswith('el') else 'NumPy port'))
  return dict(x=rl(ns['VAL_RANGE'], 4), target=rl(Z), nets=out)


# ------------------------------------------------------------------ lab 5 --
LAB5 = {}


def lab5():
  with in_lab('5-lab'):
    ns, printed = run_script('5-lab.py')
    hebb = ns['hebb']
    iterations = {}
    # 5-lab.py ignores train()'s return value; train fresh copies to report the iterations
    for k, c in ns['TRAINED'].items():
      copy = hebb.new_entity_container(image=ns['IMGS'][f'{k}_valid'])
      iterations[k] = hebb.train(copy)
  LAB5.update(ns=ns)
  side = int(round(math.sqrt(len(next(iter(ns['IMGS'].values()))))))
  results = {k: {l: int(v) for l, v in hebb.identify(image=img, trained=ns['TRAINED']).items()} for k, img in ns['IMGS'].items()}
  bits = lambda a: ''.join('1' if v > 0 else '0' for v in a)
  return dict(side=side, images={k: bits(v) for k, v in ns['IMGS'].items()}, results=results, stdout=printed, iterations=iterations,
              weights={k: np.asarray(c.weights[1:], int).tolist() for k, c in ns['TRAINED'].items()},
              bias={k: int(c.weights[0]) for k, c in ns['TRAINED'].items()})


def lab5_identify(pixels):
  ns = LAB5['ns']
  image = np.array([1 if ch == '1' else -1 for ch in pixels])
  return {l: int(v) for l, v in ns['hebb'].identify(image=image, trained=ns['TRAINED']).items()}


# ------------------------------------------------------------------ lab 6 --
def lab6(epochs, n_memb, loss, optimizer):
  import anfis_np
  sys.modules['anfis'] = anfis_np
  progress(0, 'Reading Concrete_Data.xls')
  with in_lab('6-lab'):
    fit_kw = {}

    def before(src, ns):
      if src.startswith('history = fis.fit('):
        fit_kw['on_epoch'] = lambda e: progress((e + 1) / epochs, f'Epoch {e + 1} of {epochs}')
        ns['fis'].fit = (lambda f: lambda *a, **kw: f(*a, **kw, **fit_kw))(ns['fis'].fit)

    t0 = time.perf_counter()
    ns, printed = run_script('6-lab.py', kwargs=dict(n_epochs=epochs, n_memb=n_memb, loss=loss, optimizer=optimizer),
                             before=before, stop=lambda s: s.startswith('fis.plotmfs('))
    elapsed = time.perf_counter() - t0
  fis, X, Y, n_train = ns['fis'], ns['X'], ns['Y'], ns['AMOUNT_TRAIN']
  xn, curves, init = fis.membership_curves()
  pred = fis(X[n_train:]).ravel()
  cols = ['Cement component', 'Blast Furnace Slag', 'Fly Ash', 'Water', 'Superplasticizer', 'Coarse Aggregate', 'Fine Aggregate', 'Age']
  return dict(loss=rl(ns['history'].history['loss']), val_loss=rl(ns['history'].history['val_loss']), xn=rl(xn, 4),
              mfs=[[rl(curves[m, n], 4) for m in range(fis.m)] for n in range(fis.n)],
              init=[[rl(init[m, n], 4) for m in range(fis.m)] for n in range(fis.n)], inputs=cols,
              mus=[rl(fis.mus[:, n], 3) for n in range(fis.n)], sigmas=[rl(fis.sigmas[:, n], 3) for n in range(fis.n)],
              actual=rl(Y[n_train:].ravel(), 3), predicted=rl(pred, 3), rules=fis.m ** fis.n, seconds=r(elapsed, 2),
              train=int(n_train), validation=int(len(X) - n_train), mae=r(np.mean(np.abs(pred - Y[n_train:].ravel())), 3))


# ------------------------------------------------------------------ lab 7 --
def lab7(fraction, sigma, n, seed):
  if 'multiprocess' not in sys.modules:
    # evol imports multiprocess.pool.Pool for its optional parallel evaluation; a browser has no processes
    mp = types.ModuleType('multiprocess')
    mp.pool = types.ModuleType('multiprocess.pool')
    mp.pool.Pool = None
    sys.modules['multiprocess'], sys.modules['multiprocess.pool'] = mp, mp.pool
  mpl()
  import random
  with in_lab('7-lab'):
    random.seed(seed)  # in place of the script's random.seed(26), so the page can change the seed
    ns, printed = run_script('7-lab.py', kwargs=dict(fraction=fraction, sigma=sigma, n=n),
                             skip=lambda s: plotting(s) or s.startswith(('x_mesh', 'z = ', 'random.seed(')))
  pop1 = sorted(((float(i.chromosome), float(i.fitness)) for i in ns['evolved_1']), key=lambda p: p[1])
  pop2 = sorted(((float(i.chromosome[0]), float(i.chromosome[1]), float(i.fitness)) for i in ns['evolved_2']), key=lambda p: -p[2])
  fn1, fn2 = ns['FN_1'], ns['FN_2']
  X1, X2, Y = ns['X_1'], ns['X_2'], ns['Y']
  grid = [[r(fn2(x, y), 4) for x in X2] for y in Y]
  return dict(stdout=printed, x1=rl(X1, 3), f1=rl([fn1(x) for x in X1]), x2=rl(X2, 3), y2=rl(Y, 3), grid=grid,
              pop1=[[r(a), r(b)] for a, b in pop1], pop2=[[r(a), r(b), r(c)] for a, b, c in pop2],
              min=dict(x=r(ns['minimized_x']), y=r(fn1(ns['minimized_x']))),
              max=dict(x=r(ns['maximized_x']), y=r(ns['maximized_y']), z=r(fn2(ns['maximized_x'], ns['maximized_y']))))


# ------------------------------------------------------------------ lab 8 --
def lab8(a, b, n, iterations, learning_rate, seed):
  mpl()
  if seed is not None: np.random.seed(int(seed))
  frames = []

  def before(src, ns):
    if src.startswith('result, iterations = ga()'):
      crossover, best = ns['crossover'], ns['fits_best']
      total = ns['MAX_ITERATIONS_BOUND']

      def recorded():
        chrom, fit = best()
        frames.append(dict(gen=len(frames), fitness=r(fit), centers=[[r(x, 3), r(y, 3)] for x, y in chrom]))
        if len(frames) % 10 == 0: progress(len(frames) / max(1, total), f'Generation {len(frames)}')
        return crossover()

      ns['crossover'] = recorded

  with in_lab('8-lab'):
    t0 = time.perf_counter()
    ns, printed = run_script('8-lab.py', consts=dict(A=a, B=b, N=n, MAX_ITERATIONS_BOUND=iterations, LEARNING_RATE=learning_rate),
                             skip=lambda s: plotting(s), before=before)
    elapsed = time.perf_counter() - t0
  centers, fitness = ns['result']
  l, s = ns['calc_fitness_params'](centers)
  frames.append(dict(gen=ns['iterations'][-1][0], fitness=r(fitness), centers=[[r(x, 3), r(y, 3)] for x, y in centers]))
  step = max(1, len(frames) // 300)
  frames = frames[:-1:step] + frames[-1:]
  return dict(A=ns['A'], B=ns['B'], shapes=[[r(w, 3), r(h, 3)] for w, h in ns['SHAPES']], frames=frames, stdout=printed,
              iterations=[[int(i), r(f)] for i, f in ns['iterations']], fitness=r(fitness), length=r(l), overlap=r(s),
              seconds=r(elapsed, 2), C=np.asarray(ns['C']).tolist())


# ------------------------------------------------------------------ lab 9 --
LAB9 = {}


def _mnist():
  if 'w' not in LAB9:
    d = np.load(os.path.join(HOME, 'models', 'mnist.npz'))
    LAB9.update(w=[d[f'w{i}'] for i in range(6)], meta=dict(loss=r(d['loss'], 4), accuracy=r(d['accuracy'], 4),
                history_loss=rl(d['history_loss'], 4), history_accuracy=rl(d['history_accuracy'], 4)))
  return LAB9


def normalize(x, axis=-1, order=2):
  """keras.utils.normalize"""
  l2 = np.atleast_1d(np.linalg.norm(x, order, axis))
  l2[l2 == 0] = 1
  return x / np.expand_dims(l2, axis)


def mnist_predict(img_arr):
  """recognizer.predict: Flatten -> Dense(128, relu) -> Dense(128, relu) -> Dense(10, softmax)"""
  w = _mnist()['w']
  h = img_arr.reshape(len(img_arr), -1)
  h = np.maximum(h @ w[0] + w[1], 0)
  h = np.maximum(h @ w[2] + w[3], 0)
  z = h @ w[4] + w[5]
  e = np.exp(z - z.max(1, keepdims=True))
  return e / e.sum(1, keepdims=True)


def lab9():
  from PIL import Image
  rows = []
  folder = os.path.join(LABS, '9-lab', 'examples')
  for name in sorted(os.listdir(folder)):
    img = np.array(Image.open(os.path.join(folder, name)).convert('RGB'))[:, :, 2]   # cv.imread(...)[:,:,0] is the blue channel
    img_arr = np.invert(np.array([img]))
    p = mnist_predict(normalize(img_arr.astype(float), axis=1))[0]
    rows.append(dict(name=name, label=int(name[0]), pixels=img.ravel().tolist(), probs=rl(p, 4), predicted=int(np.argmax(p))))
  return dict(examples=rows, **_mnist()['meta'])


def lab9_predict(pixels):
  """pixels: 784 grey levels of a black-on-white drawing, like the BMP examples"""
  img = np.array(pixels, np.uint8).reshape(28, 28)
  img_arr = np.invert(np.array([img]))
  return rl(mnist_predict(normalize(img_arr.astype(float), axis=1))[0], 4)


# --------------------------------------------------------------- labs 10-11 --
def _summary(rec):
  rows = rec.layers
  total = sum(l['params'] for l in rows)
  trainable = sum(l['trainable'] for l in rows)
  return dict(layers=rows, total=total, trainable=trainable, non_trainable=total - trainable, macs=sum(l['macs'] for l in rows))


def lab10(side, classes):
  import keras_shapes
  rec = keras_shapes.reset()
  with in_lab('10-lab'):
    tree = ast.parse(open('10-lab.py').read())
  # the lab's `model = keras.models.Sequential([...])` statement, run against the shape-inferring layers
  node = next(n for n in tree.body if isinstance(n, ast.Assign) and ast.unparse(n.targets[0]) == 'model')
  node = ast.fix_missing_locations(_Kwargs(dict(input_shape=(side, side, 3))).visit(node))
  src = ast.unparse(node)
  if classes != 10: src = src.replace('keras.layers.Dense(10,', f'keras.layers.Dense({classes},')
  exec(src, dict(keras=keras_shapes.module()))
  return dict(**_summary(rec), source=ast.unparse(node))


def lab11(width, height, depth, classes):
  import keras_shapes
  rec = keras_shapes.reset(groups=('inception_module', 'downsample_module', 'conv_module'))
  with in_lab('11-lab'):
    keep = [n for n in load_defs('11-lab.py', names=('NUM_EPOCHS', 'INIT_LR')) if not isinstance(n, (ast.Import, ast.ImportFrom))]
  ns = {k: getattr(keras_shapes, k) for k in ('BatchNormalization', 'Conv2D', 'AveragePooling2D', 'MaxPooling2D', 'Activation', 'Dropout',
                                               'Dense', 'Flatten', 'Input', 'concatenate')}
  ns['Model'] = keras_shapes.Model
  exec(compile(ast.Module(keep, type_ignores=[]), '11-lab.py', 'exec'), ns)
  ns['GoogleNet'](width=width, height=height, depth=depth, classes=classes)
  lr = [r(ns['poly_decay'](e), 6) for e in range(ns['NUM_EPOCHS'])]
  return dict(**_summary(rec), lr=lr)


# ---------------------------------------------------------- lab 12 + exam --
SENT = {}


def lab12(text):
  import sentiment_np
  if 'bilstm' not in SENT:
    SENT['bilstm'] = sentiment_np.BiLSTMModel(np.load(os.path.join(HOME, 'models', 'lab12.npz')))
  m = SENT['bilstm']
  t0 = time.perf_counter()
  logit, words, ids = m.predict(text)
  hist = {k[8:]: rl(m.w[k], 4) for k in m.w if k.startswith('history_')}
  return dict(logit=r(logit) if logit is not None else None, prob=r(1 / (1 + math.exp(-logit))) if logit is not None else None,
              words=words[:400], ids=ids[:400], tokens=len(ids), unknown=sum(1 for i in ids if i == 1), vocab=len(m.vocab),
              history=hist, ms=r((time.perf_counter() - t0) * 1000, 1))


def exam(text, reset=False):
  import sentiment_np
  if 'elstm' not in SENT:
    with open(os.path.join(LABS, 'exam-task', 'model', 'word_index.json')) as f:
      SENT['elstm'] = sentiment_np.SentimentAnalyzer(np.load(os.path.join(HOME, 'models', 'elstm.npz')), json.load(f))
  a = SENT['elstm']
  if reset: a.model.reset()
  if not text: return dict(calls=a.model.calls)
  t0 = time.perf_counter()
  value, sentiment, info = a.predict(text)
  return dict(value=r(value), sentiment=sentiment, calls=a.model.calls, ms=r((time.perf_counter() - t0) * 1000, 1), **info)


RUNNERS = dict(lab1=lab1, lab2=lab2, lab3=lab3, lab4=lab4, lab5=lab5, lab5_identify=lab5_identify, lab6=lab6, lab7=lab7, lab8=lab8,
               lab9=lab9, lab9_predict=lab9_predict, lab10=lab10, lab11=lab11, lab12=lab12, exam=exam)
