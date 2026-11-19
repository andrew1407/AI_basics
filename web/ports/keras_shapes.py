"""Shape and parameter inference for the Keras layers 10-lab and 11-lab use, without TensorFlow.

The labs' own model code (10-lab's Sequential list, 11-lab's conv/inception/downsample modules and GoogleNet)
runs against these classes. Each layer computes its output shape and parameter count the way Keras does
(valid/same padding, BatchNormalization's 2 trainable + 2 moving statistics per channel), so the page
can list the architecture exactly without building the network.
"""
import math
import sys
import types


class Tensor:
  def __init__(self, shape, layer=None):
    self.shape = tuple(shape)  # without the batch dimension
    self.layer = layer


class Recorder:
  def __init__(self, groups=()):
    self.layers = []
    self.counts = {}
    self.groups = set(groups)  # lab function names whose calls group the layers (e.g. inception_module)
    self.group_ids = {}

  def name(self, base):
    n = self.counts.get(base, 0)
    self.counts[base] = n + 1
    return base if n == 0 else f'{base}_{n}'

  def group(self):
    """The outermost grouping function on the call stack and which call of it this is."""
    found = None
    f = sys._getframe(2)
    while f is not None:
      if f.f_code.co_name in self.groups: found = f
      f = f.f_back
    if found is None: return None
    key = id(found)  # the frame is kept referenced below, so its id can't be reused by a later call
    if key not in self.group_ids:
      fname = found.f_code.co_name
      n = sum(1 for g, _, _ in self.group_ids.values() if g == fname)
      self.group_ids[key] = (fname, n + 1, found)
    fname, n, _ = self.group_ids[key]
    return dict(fn=fname, n=n, id=f'{fname}_{n}')


REC = Recorder()


def pair(v):
  return tuple(v) if isinstance(v, (tuple, list)) else (v, v)


def conv_out(size, k, s, padding):
  return math.ceil(size / s) if padding == 'same' else (size - k) // s + 1


class Layer:
  kind = 'Layer'
  base = 'layer'

  def __init__(self, *args, input_shape=None, name=None, **kw):
    self.input_shape = input_shape
    self.name_arg = name
    self.kw = kw

  def __call__(self, inputs):
    ins = inputs if isinstance(inputs, list) else [inputs]
    out, params, trainable, config, macs = self.build([t.shape for t in ins])
    row = dict(name=self.name_arg or REC.name(self.base), kind=self.kind, config=config, output=list(out),
               params=params, trainable=trainable, macs=macs, inbound=[t.layer['name'] for t in ins if t.layer],
               group=REC.group())
    REC.layers.append(row)
    return Tensor(out, row)

  def build(self, shapes):
    return shapes[0], 0, 0, '', 0


class Conv2D(Layer):
  kind, base = 'Conv2D', 'conv2d'

  def __init__(self, filters, kernel_size, strides=(1, 1), padding='valid', activation=None, use_bias=True, **kw):
    super().__init__(**kw)
    self.filters, self.k, self.s, self.padding, self.activation, self.use_bias = filters, pair(kernel_size), pair(strides), padding, activation, use_bias

  def build(self, shapes):
    h, w, c = shapes[0]
    oh, ow = conv_out(h, self.k[0], self.s[0], self.padding), conv_out(w, self.k[1], self.s[1], self.padding)
    params = self.k[0] * self.k[1] * c * self.filters + (self.filters if self.use_bias else 0)
    config = f'{self.filters} × {self.k[0]}×{self.k[1]}, stride {self.s[0]}, {self.padding}' + (f', {self.activation}' if self.activation else '')
    return (oh, ow, self.filters), params, params, config, oh * ow * self.k[0] * self.k[1] * c * self.filters


class BatchNormalization(Layer):
  kind, base = 'BatchNormalization', 'batch_normalization'

  def __init__(self, axis=-1, **kw):
    super().__init__(**kw)
    self.axis = axis

  def build(self, shapes):
    c = shapes[0][self.axis]
    return shapes[0], 4 * c, 2 * c, 'γ, β + moving mean, variance', 0


class _Pool(Layer):
  def __init__(self, pool_size=(2, 2), strides=None, padding='valid', **kw):
    super().__init__(**kw)
    self.p, self.s, self.padding = pair(pool_size), pair(strides if strides is not None else pool_size), padding

  def build(self, shapes):
    h, w, c = shapes[0]
    out = (conv_out(h, self.p[0], self.s[0], self.padding), conv_out(w, self.p[1], self.s[1], self.padding), c)
    return out, 0, 0, f'{self.p[0]}×{self.p[1]}, stride {self.s[0]}, {self.padding}', 0


class MaxPooling2D(_Pool):
  kind, base = 'MaxPooling2D', 'max_pooling2d'


class AveragePooling2D(_Pool):
  kind, base = 'AveragePooling2D', 'average_pooling2d'


MaxPool2D = MaxPooling2D


class Flatten(Layer):
  kind, base = 'Flatten', 'flatten'

  def build(self, shapes):
    return (math.prod(shapes[0]),), 0, 0, '', 0


class Dense(Layer):
  kind, base = 'Dense', 'dense'

  def __init__(self, units, activation=None, use_bias=True, **kw):
    super().__init__(**kw)
    self.units, self.activation, self.use_bias = units, activation, use_bias

  def build(self, shapes):
    n = shapes[0][-1]
    params = n * self.units + (self.units if self.use_bias else 0)
    return shapes[0][:-1] + (self.units,), params, params, f'{self.units} units' + (f', {self.activation}' if self.activation else ''), n * self.units


class Dropout(Layer):
  kind, base = 'Dropout', 'dropout'

  def __init__(self, rate, **kw):
    super().__init__(**kw)
    self.rate = rate

  def build(self, shapes):
    return shapes[0], 0, 0, f'rate {self.rate}', 0


class Activation(Layer):
  kind, base = 'Activation', 'activation'

  def __init__(self, activation, **kw):
    super().__init__(**kw)
    self.activation = activation

  def build(self, shapes):
    return shapes[0], 0, 0, str(self.activation), 0


class Concatenate(Layer):
  kind, base = 'Concatenate', 'concatenate'

  def __init__(self, axis=-1, **kw):
    super().__init__(**kw)
    self.axis = axis

  def build(self, shapes):
    out = list(shapes[0])
    out[self.axis] = sum(s[self.axis] for s in shapes)
    return tuple(out), 0, 0, f'{len(shapes)} inputs', 0


def concatenate(inputs, axis=-1, **kw):
  return Concatenate(axis=axis, **kw)(list(inputs))


class InputLayer(Layer):
  kind, base = 'InputLayer', 'input_layer'


def Input(shape, name=None, **kw):
  t = Tensor(shape)
  row = dict(name=name or REC.name('input_layer'), kind='InputLayer', config='', output=list(shape), params=0, trainable=0, macs=0, inbound=[], group=None)
  REC.layers.append(row)
  t.layer = row
  return t


class Model:
  def __init__(self, inputs=None, outputs=None, name='model'):
    self.name = name
    self.layers = list(REC.layers)


class Sequential(Model):
  def __init__(self, layers=(), name='sequential'):
    x = None
    for layer in layers:
      if x is None: x = Input(layer.input_shape)
      x = layer(x)
    super().__init__(name=name)


def reset(groups=()):
  global REC
  REC = Recorder(groups)
  return REC


def module():
  """A stand-in `keras` package (keras.layers.*, keras.models.Sequential) for the labs' code to import."""
  keras = types.ModuleType('keras')
  keras.layers = types.ModuleType('keras.layers')
  for name in ('Conv2D', 'BatchNormalization', 'MaxPool2D', 'MaxPooling2D', 'AveragePooling2D', 'Flatten', 'Dense', 'Dropout',
               'Activation', 'Input', 'Concatenate', 'concatenate'):
    setattr(keras.layers, name, globals()[name])
  keras.models = types.ModuleType('keras.models')
  keras.models.Sequential, keras.models.Model = Sequential, Model
  return keras
