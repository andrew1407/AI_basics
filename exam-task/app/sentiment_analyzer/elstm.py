import tensorflow as tf
from tensorflow.keras.initializers import GlorotUniform, Orthogonal, Zeros
from keras.layers import Layer
import keras.backend as K
from keras.utils import tf_utils


class ELSTMCell(Layer):
  def __init__(self, units, epochs, **kwargs):
    self.state_size = [units, units]
    self.units = units
    self.h_sp = tf.Variable(
        [[.0] * units for _ in range(units)], trainable=False)
    self.exp_sum = tf.Variable(
        [[.0] * units for _ in range(units)], trainable=False)
    self.e_t = tf.Variable(
        [[.0] * units for _ in range(units)], dtype=tf.float64, trainable=False)
    self.ro_min = tf.Variable(.0001, dtype=tf.float64, trainable=False)
    self.ro_max = tf.Variable(.01, dtype=tf.float64, trainable=False)
    self.cur_epoch = tf.Variable(1, dtype=tf.float64, trainable=False)
    self.epochs = epochs
    super(ELSTMCell, self).__init__(**kwargs)

  def get_config(self):
    config = super().get_config()
    config.update({
        "units": self.units,
        "epochs": self.epochs
    })
    return config

  def set_current_epoch(self, val):
    self.cur_epoch.assign(val)

  def clear_accumulators(self):
    self.h_sp.assign([[.0] * self.units for _ in range(self.units)])
    self.exp_sum.assign([[.0] * self.units for _ in range(self.units)])
    self.e_t.assign([[.0] * self.units for _ in range(self.units)])

  @tf_utils.shape_type_conversion
  def build(self, input_shape):
    input_dim = input_shape[-1]

    self.kernel = self.add_weight(shape=(input_dim, self.units * 4),
                                  name='kernel',
                                  initializer=GlorotUniform())

    self.recurrent_kernel = self.add_weight(shape=(self.units, self.units * 4),
                                            name='recurrent_kernel',
                                            initializer=Orthogonal())

    self.bias = self.add_weight(shape=(self.units * 4,),
                                name='bias',
                                initializer=Zeros())

    self.built = True

  @tf.function
  def call(self, inputs, states, training=None):
    h_tm1 = states[0]
    c_tm1 = states[1]

    self.exp_sum.assign_add(tf.math.exp(tf.transpose(h_tm1) * self.h_sp))
    inputs_i = inputs
    inputs_f = inputs
    inputs_c = inputs
    inputs_o = inputs

    k_i, k_f, k_c, k_o = tf.split(
        self.kernel, num_or_size_splits=4, axis=1)

    x_i = K.dot(inputs_i, k_i)
    x_f = K.dot(inputs_f, k_f)
    x_c = K.dot(inputs_c, k_c)
    x_o = K.dot(inputs_o, k_o)

    b_i, b_f, b_c, b_o = tf.split(self.bias, num_or_size_splits=4, axis=0)

    x_i = K.bias_add(x_i, b_i)
    x_f = K.bias_add(x_f, b_f)
    x_c = K.bias_add(x_c, b_c)
    x_o = K.bias_add(x_o, b_o)

    h_tm1_i = h_tm1
    h_tm1_f = h_tm1
    h_tm1_c = h_tm1
    h_tm1_o = h_tm1

    x = (x_i, x_f, x_c, x_o)
    h_tm1 = (h_tm1_i, h_tm1_f, h_tm1_c, h_tm1_o)

    c, o = self._compute_carry_and_output(x, h_tm1, c_tm1)
    self.h_sp.assign_add(states[0])
    h = o * tf.keras.activations.tanh(c)

    return h, [h, c]

  @tf.function
  def _compute_carry_and_output(self, x, h_tm1, c_tm1):
    x_i, x_f, x_c, x_o = x
    h_tm1_i, h_tm1_f, h_tm1_c, h_tm1_o = h_tm1

    i = tf.keras.activations.hard_sigmoid(
      x_i + K.dot(h_tm1_i, self.recurrent_kernel[:, :self.units]))

    es = self.emotion_estimator(h_tm1_i)
    self.emotion_modulator(x_o, es)
    f = tf.keras.activations.hard_sigmoid(x_f + K.dot(
      h_tm1_f, self.recurrent_kernel[:, self.units:self.units * 2]))
    f = tf.cast(self.e_t, dtype=tf.float32) * f
    o = tf.keras.activations.hard_sigmoid(
      x_o + K.dot(h_tm1_o, self.recurrent_kernel[:, self.units * 3:]))
    a = tf.keras.activations.tanh(x_c + K.dot(
      h_tm1_c, self.recurrent_kernel[:, self.units * 2:self.units * 3]))

    c = f * c_tm1 + i * a
    return c, o

  @tf.function
  def emotion_estimator(self, h_t):
    exp_new = tf.math.exp(tf.transpose(h_t) * self.h_sp)
    p = exp_new / self.exp_sum
    return tf.math.reduce_mean(p) > .3

  @tf.function
  def emotion_modulator(self, x_o, es):
    to_f64 = lambda x: tf.cast(x, dtype=tf.float64)
    ro_dif = self.ro_max - self.ro_min
    ep = (self.epochs - self.cur_epoch) / self.epochs
    ro = to_f64(ro_dif) * ep + self.ro_min
    ro_h = ro * to_f64(tf.math.abs(x_o - self.h_sp))
    if es: self.e_t.assign(self.e_t + ro_h)
    else: self.e_t.assign(self.e_t - ro_h)
    self.normalize_e_t()

  @tf.function
  def normalize_e_t(self):
    min_e = tf.reduce_min(self.e_t)
    max_e = tf.reduce_max(self.e_t)
    if max_e > min_e:
      normalized = 2. * (self.e_t - min_e) / (max_e - min_e) - 1.
      self.e_t.assign(normalized)
