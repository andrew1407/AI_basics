"""NumPy port of 6-lab/anfis.py for the browser, where TensorFlow isn't available.

The same five-layer ANFIS (FuzzyLayer -> RuleLayer -> NormLayer -> DefuzzLayer -> SummationLayer) with the
same initialisers, trained like keras.Model.fit with Adam (Keras defaults) and the Huber loss: shuffled
mini-batches of a fixed batch size, the gradients written out by hand. Only 'gaussian' memberships are
ported, the ones 6-lab.py uses. The class names and call signatures match anfis.py, so 6-lab.py runs as is.
"""
import numpy as np


class fis_parameters():
  def __init__(self, n_input=3, n_memb=3, batch_size=16, n_epochs=25, memb_func='gaussian', optimizer='sgd', loss='mse'):
    self.n_input = n_input
    self.n_memb = n_memb
    self.batch_size = batch_size
    self.n_epochs = n_epochs
    self.memb_func = memb_func
    self.optimizer = optimizer
    self.loss = loss


class History:
  def __init__(self):
    self.history = dict(loss=[], val_loss=[])


def huber(err, delta=1.0):
  a = np.abs(err)
  return np.where(a <= delta, 0.5 * err ** 2, delta * a - 0.5 * delta ** 2), np.clip(err, -delta, delta)


def mse(err):
  return err ** 2, 2 * err


class Adam:
  """tf.keras.optimizers.Adam with its defaults."""
  def __init__(self, lr=0.001, beta_1=0.9, beta_2=0.999, eps=1e-7):
    self.lr, self.b1, self.b2, self.eps, self.t, self.m, self.v = lr, beta_1, beta_2, eps, 0, {}, {}

  def step(self, params, grads):
    self.t += 1
    lr_t = self.lr * np.sqrt(1 - self.b2 ** self.t) / (1 - self.b1 ** self.t)
    for k, g in grads.items():
      m = self.m[k] = self.b1 * self.m.get(k, 0) + (1 - self.b1) * g
      v = self.v[k] = self.b2 * self.v.get(k, 0) + (1 - self.b2) * g * g
      params[k] -= lr_t * m / (np.sqrt(v) + self.eps)


class SGD:
  def __init__(self, lr=0.01):
    self.lr = lr

  def step(self, params, grads):
    for k, g in grads.items(): params[k] -= self.lr * g


class _Model:
  """Stands in for the keras.Model that ANFIS builds: compile() picks the optimizer and loss."""
  def __init__(self, owner):
    self.owner = owner

  def compile(self, optimizer='adam', loss='mse'):
    self.optimizer = Adam() if optimizer == 'adam' else SGD()
    self.loss = huber if loss in ('huber_loss', 'huber') else mse

  def predict(self, X):
    return self.owner.forward(np.asarray(X, float))[0]


class ANFIS:
  def __init__(self, n_input, n_memb, batch_size=16, memb_func='gaussian', name='MyAnfis', seed=None):
    if memb_func != 'gaussian': raise ValueError("this port implements the 'gaussian' membership functions only")
    self.n, self.m, self.batch_size, self.memb_func, self.name = n_input, n_memb, batch_size, memb_func, name
    rng = np.random.default_rng(seed)
    # FuzzyLayer: mu equally spaced in [-1.5, 1.5] per membership, sigma ~ U(0.7, 1.3)
    mu = np.tile(np.linspace(-1.5, 1.5, n_memb).reshape(-1, 1), (1, n_input))
    sigma = np.random.default_rng(1).uniform(0.7, 1.3, (n_memb, n_input))
    rules = n_memb ** n_input
    # DefuzzLayer: consequence bias and weights ~ U(-2, 2)
    self.params = dict(mu=mu, sigma=sigma, bias=rng.uniform(-2, 2, (1, rules)), weights=rng.uniform(-2, 2, (n_input, rules)))
    self.model = _Model(self)
    self.update_weights()

  def __call__(self, X):
    return self.model.predict(X)

  def update_weights(self):
    p = self.params
    self.mus, self.sigmas, self.bias, self.weights = p['mu'], p['sigma'], p['bias'], p['weights']

  def _rules(self):
    """One-hot (rules, m*n) map: rule k takes membership i_d of input d, input 0 varying slowest (as tf.matmul builds it)."""
    if getattr(self, '_onehot', None) is None:
      m, n = self.m, self.n
      digits = np.array(np.unravel_index(np.arange(m ** n), (m,) * n)).T   # (rules, n)
      self._onehot = np.zeros((m ** n, m * n))
      self._onehot[np.arange(m ** n)[:, None], digits * n + np.arange(n)] = 1
    return self._onehot

  def forward(self, X):
    p, B = self.params, len(X)
    xs = 1 / (1 + np.exp(-X))                                          # FuzzyLayer: sigmoid of the inputs
    diff = xs[:, None, :] - p['mu']                                    # (B, m, n)
    log_mf = -diff ** 2 / p['sigma'] ** 2                              # gaussian memberships, as logs
    rule = np.exp(log_mf.reshape(B, -1) @ self._rules().T)             # RuleLayer: product over the inputs
    s = rule.sum(1, keepdims=True)
    w = rule / s                                                       # NormLayer
    f = X @ p['weights'] + p['bias']                                   # DefuzzLayer
    out = (w * f).sum(1, keepdims=True)                                # SummationLayer
    return out, (diff, rule, s, w, f)

  def gradients(self, X, y):
    p, B = self.params, len(X)
    out, (diff, rule, s, w, f) = self.forward(X)
    loss, dloss = self.model.loss(out - y)
    g = dloss / B                                                      # d mean-loss / d out, (B, 1)
    gw = g * w
    grads = dict(weights=X.T @ gw, bias=gw.sum(0, keepdims=True))
    drule = g * (f - out) / s                                          # through the normalisation
    dlog = ((drule * rule) @ self._rules()).reshape(diff.shape)        # d loss / d log-membership
    sig2 = p['sigma'] ** 2
    grads['mu'] = (dlog * 2 * diff).sum(0) / sig2
    grads['sigma'] = (dlog * 2 * diff ** 2).sum(0) / (sig2 * p['sigma'])
    return float(loss.mean()), grads

  def evaluate(self, X, y):
    bs = self.batch_size
    losses = [float(self.model.loss(self.forward(X[i:i + bs])[0] - y[i:i + bs])[0].mean()) for i in range(0, len(X) - bs + 1, bs)]
    return float(np.mean(losses))

  def fit(self, X, y, epochs=1, batch_size=None, validation_data=None, shuffle=True, on_epoch=None, **kwargs):
    X, y = np.asarray(X, float), np.asarray(y, float).reshape(-1, 1)
    bs = batch_size or self.batch_size
    self.init_weights = [self.params['mu'].copy(), self.params['sigma'].copy()]
    history = History()
    rng = np.random.default_rng()
    for epoch in range(epochs):
      order = rng.permutation(len(X)) if shuffle else np.arange(len(X))
      losses = []
      for i in range(0, len(X) - bs + 1, bs):   # the model's batch size is fixed, so a partial batch can't run
        idx = order[i:i + bs]
        loss, grads = self.gradients(X[idx], y[idx])
        self.model.optimizer.step(self.params, grads)
        losses.append(loss)
      history.history['loss'].append(float(np.mean(losses)))
      if validation_data is not None:
        vx, vy = validation_data
        history.history['val_loss'].append(self.evaluate(np.asarray(vx, float), np.asarray(vy, float).reshape(-1, 1)))
      if on_epoch: on_epoch(epoch)
    self.update_weights()
    return history

  def membership_curves(self, points=100):
    """The data plotmfs() draws: x grid, trained and initial curves per input and membership."""
    def curves(mus, sigmas):
      return np.exp(-np.square(xn - mus[..., None]) / np.square(sigmas[..., None]))
    mus, sigmas = np.around(self.params['mu'], 2), np.around(self.params['sigma'], 2)
    xn = np.linspace(np.min(mus) - 2 * np.max(abs(sigmas)), np.max(mus) + 2 * np.max(abs(sigmas)), points)
    init = getattr(self, 'init_weights', None)
    return xn, curves(mus, sigmas), (curves(*np.around(init, 2)) if init is not None else None)

  def plotmfs(self, show_initial_weights=False):
    pass  # drawn by the page from membership_curves()
