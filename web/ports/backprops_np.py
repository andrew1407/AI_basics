"""NumPy port of the Theano/Lasagne half of 4-lab/backprops.py (Theano and Lasagne don't run in the browser).

feed_forward_backprop and cascade_forward_backprop keep the lab's architecture and training exactly:
DenseLayer(tanh) hidden layers with Lasagne's GlorotUniform weights and zero biases, a linear output unit,
the mean squared error over the whole sample and plain SGD (lasagne.updates.sgd), one full-batch step per
epoch. In the cascade network each hidden layer sees the concatenation of the input and every earlier
hidden layer (lasagne.layers.ConcatLayer), and the output sees the last hidden layer only.
"""
import numpy as np


def _glorot(rng, n_in, n_out):
  a = np.sqrt(3) * np.sqrt(2.0 / (n_in + n_out))
  return rng.uniform(-a, a, (n_in, n_out))


def _train(xy, z, hidden_num, epochs, learning_rate, cascade, seed=None):
  rng = np.random.default_rng(seed)
  x = np.asarray(xy, float)
  t = np.asarray(z, float).reshape(-1, 1)
  widths = [x.shape[1]]
  hidden = []
  for n in hidden_num:
    n_in = sum(widths) if cascade else widths[-1]
    hidden.append([_glorot(rng, n_in, n), np.zeros(n)])
    widths.append(n)
  out = [_glorot(rng, widths[-1], 1), np.zeros(1)]

  def forward():
    acts = [x]
    for W, b in hidden:
      inp = np.concatenate(acts, axis=1) if cascade else acts[-1]
      acts.append(np.tanh(inp @ W + b))
    return acts, acts[-1] @ out[0] + out[1]

  errs = []
  for _ in range(epochs):
    acts, y = forward()
    diff = y - t
    errs.append(float(np.mean(diff ** 2)))
    g = 2 * diff / diff.size                       # d mean((y - t)^2) / dy
    grads_out = (acts[-1].T @ g, g.sum(0))
    # back through the hidden layers; in the cascade net a layer's output also feeds every later layer
    d_acts = [np.zeros_like(a) for a in acts]
    d_acts[-1] += g @ out[0].T
    grads = [None] * len(hidden)
    for i in range(len(hidden) - 1, -1, -1):
      W, b = hidden[i]
      da = d_acts[i + 1] * (1 - acts[i + 1] ** 2)   # tanh'
      inp = np.concatenate(acts[:i + 1], axis=1) if cascade else acts[i]
      grads[i] = (inp.T @ da, da.sum(0))
      d_inp = da @ W.T
      if cascade:
        offset = 0
        for k in range(i + 1):
          w = acts[k].shape[1]
          d_acts[k] += d_inp[:, offset:offset + w]
          offset += w
      else:
        d_acts[i] += d_inp
    for (W, b), (gW, gb) in zip(hidden, grads):
      W -= learning_rate * gW
      b -= learning_rate * gb
    out[0] -= learning_rate * grads_out[0]
    out[1] -= learning_rate * grads_out[1]
  return forward()[1], errs


def cascade_forward_backprop(xy, z, hidden_num=[], epochs=100, learning_rate=1, seed=None):
  return _train(xy, z, hidden_num, epochs, learning_rate, cascade=True, seed=seed)


def feed_forward_backprop(xy, z, hidden_num=[], epochs=100, learning_rate=1, seed=None):
  return _train(xy, z, hidden_num, epochs, learning_rate, cascade=False, seed=seed)
