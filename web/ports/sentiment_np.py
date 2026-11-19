"""NumPy forward passes for the two sentiment models (TensorFlow doesn't run in the browser).

- Exam app: the preprocessing of exam-task/app/sentiment_analyzer/analyzer.py (Keras Tokenizer with the
  training vocabulary, pad_sequences to 1000) and the SavedModel exam-task/model/model1: Embedding(2001, 300)
  -> Conv1D(32, 4, same, relu) -> MaxPooling1D(4) -> RNN(ELSTMCell(32)) -> Dense(1, sigmoid), with the cell's
  equations from elstm.py, including its non-trainable accumulators h_sp, exp_sum and e_t, which every
  prediction keeps updating as in TensorFlow. Checked against the SavedModel run by TensorFlow (same output to 1e-6).
- 12-lab: TextVectorization (lower_and_strip_punctuation, whitespace split) -> Embedding(mask_zero) ->
  Bidirectional(LSTM(64)) -> Dense(64, relu) -> Dense(1) logit, with the weights 12-lab.py trained.
"""
import re
import numpy as np

KERAS_FILTERS = '!"#$%&()*+,-./:;<=>?@[\\]^_`{|}~\t\n'


def text_to_word_sequence(text, filters=KERAS_FILTERS, lower=True, split=' '):
  """keras.preprocessing.text.text_to_word_sequence"""
  if lower: text = text.lower()
  text = text.translate(str.maketrans({c: split for c in filters}))
  return [w for w in text.split(split) if w]


class Tokenizer:
  """The parts of keras.preprocessing.text.Tokenizer that texts_to_sequences uses."""
  def __init__(self, num_words=None, oov_token=None):
    self.num_words, self.oov_token, self.word_index = num_words, oov_token, {}

  def texts_to_sequences(self, texts):
    oov = self.word_index.get(self.oov_token)
    out = []
    for text in texts:
      seq = []
      for w in text_to_word_sequence(text):
        i = self.word_index.get(w)
        if i is not None:
          if self.num_words and i >= self.num_words:
            if oov is not None: seq.append(oov)
          else:
            seq.append(i)
        elif self.oov_token is not None:
          seq.append(oov)
      out.append(seq)
    return out


def pad_sequences(seqs, maxlen):
  """keras pad_sequences with its defaults: zeros in front, longer sequences keep their last maxlen tokens."""
  out = np.zeros((len(seqs), maxlen), np.int64)
  for i, s in enumerate(seqs):
    s = s[-maxlen:]
    if s: out[i, -len(s):] = s
  return out


def hard_sigmoid(x):
  return np.clip(0.2 * x + 0.5, 0, 1).astype(np.float32)  # tf.keras.activations.hard_sigmoid (TF 2.7)


class ELSTMModel:
  STATE = ('h_sp', 'exp_sum', 'e_t')

  def __init__(self, weights, epochs=10):
    self.w = {k: weights[k] for k in weights.files} if hasattr(weights, 'files') else dict(weights)
    self.epochs = epochs  # ELSTMCell(BATCH, EPOCHS) in the notebook
    self.reset()

  def reset(self):
    self.state = {k: self.w[k].copy() for k in self.STATE}
    self.calls = 0

  def predict(self, seq):
    w, st = self.w, self.state
    x = w['embeddings'][seq]                                            # (B, 1000, 300)
    B, T, _ = x.shape
    xp = np.pad(x, ((0, 0), (1, 2), (0, 0)))                             # 'same' padding for a width-4 kernel
    conv = sum(xp[:, j:j + T] @ w['conv_kernel'][j] for j in range(4)) + w['conv_bias']
    seq_in = np.maximum(conv, 0).reshape(B, T // 4, 4, -1).max(2)       # relu, MaxPooling1D(4)
    u = w['recurrent_kernel'].shape[0]
    R = w['recurrent_kernel']
    k_i, k_f, k_c, k_o = np.split(w['kernel'], 4, 1)
    b_i, b_f, b_c, b_o = np.split(w['bias'], 4)
    ro = (w['ro_max'] - w['ro_min']) * ((self.epochs - w['cur_epoch']) / self.epochs) + w['ro_min']
    h = np.zeros((B, u), np.float32)
    c = np.zeros((B, u), np.float32)
    h_sp, exp_sum, e_t = st['h_sp'], st['exp_sum'], st['e_t']
    emotional = 0
    with np.errstate(over='ignore', invalid='ignore', divide='ignore'):
      for t in range(seq_in.shape[1]):
        inp = seq_in[:, t]
        exp_sum = exp_sum + np.exp(h.T * h_sp)
        x_i, x_f, x_c, x_o = inp @ k_i + b_i, inp @ k_f + b_f, inp @ k_c + b_c, inp @ k_o + b_o
        i = hard_sigmoid(x_i + h @ R[:, :u])
        es = np.mean(np.exp(h.T * h_sp) / exp_sum) > .3                 # emotion_estimator
        emotional += bool(es)
        ro_h = ro * np.abs(x_o - h_sp).astype(np.float64)              # emotion_modulator
        e_t = e_t + ro_h if es else e_t - ro_h
        lo, hi = e_t.min(), e_t.max()                                   # normalize_e_t
        if hi > lo: e_t = 2. * (e_t - lo) / (hi - lo) - 1.
        f = hard_sigmoid(x_f + h @ R[:, u:2 * u]) * e_t.astype(np.float32)
        o = hard_sigmoid(x_o + h @ R[:, 3 * u:])
        a = np.tanh(x_c + h @ R[:, 2 * u:3 * u])
        c = f * c + i * a
        h_sp = h_sp + h
        h = o * np.tanh(c)
    st.update(h_sp=h_sp, exp_sum=exp_sum, e_t=e_t)
    self.calls += 1
    z = h @ w['dense_kernel'] + w['dense_bias']
    return 1 / (1 + np.exp(-z[:, 0])), dict(steps=int(seq_in.shape[1]), emotional_steps=emotional)


class SentimentAnalyzer:
  """exam-task/app/sentiment_analyzer/analyzer.py with the ported model."""
  def __init__(self, weights, word_index):
    self.vocab_size, self.oov, self.max_sequence_len = 2000, '<OOV>', 1000
    self.tokenizer = Tokenizer(num_words=self.vocab_size, oov_token=self.oov)
    self.tokenizer.word_index = word_index
    self.model = ELSTMModel(weights)

  def predict(self, value):
    value_arr = [str(value.encode())] + [''] * 31
    seqns = self.tokenizer.texts_to_sequences(value_arr)
    padded = pad_sequences(seqns, maxlen=self.max_sequence_len)
    predicted, info = self.model.predict(padded)
    result = float(predicted[0])
    tokens = seqns[0]
    info.update(tokens=len(tokens), known=sum(1 for t in tokens if t != 1), truncated=len(tokens) > self.max_sequence_len,
                words=text_to_word_sequence(value_arr[0])[-self.max_sequence_len:][:400], ids=tokens[-self.max_sequence_len:][:400])
    return result, 'positive' if result >= .5 else 'negative', info


STRIP = re.compile(r'[!"#$%&()\*\+,\-\./:;<=>?@\[\\\]^_`{|}~\']')
ASCII_LOWER = str.maketrans('ABCDEFGHIJKLMNOPQRSTUVWXYZ', 'abcdefghijklmnopqrstuvwxyz')


def sigmoid(x):
  return 1 / (1 + np.exp(-x))


class BiLSTMModel:
  def __init__(self, weights):
    self.w = {k: weights[k] for k in weights.files}
    self.vocab = [str(v) for v in self.w['vocab']]
    self.index = {v: i for i, v in enumerate(self.vocab)}

  def vectorize(self, text):
    """TextVectorization(max_tokens=1000) with its defaults; tf.strings.lower only lowercases ASCII."""
    words = STRIP.sub('', text.translate(ASCII_LOWER)).split()
    return words, [self.index.get(t, 1) for t in words]

  def lstm(self, xs, prefix):
    w = self.w
    W, U, b = w[f'{prefix}_kernel'], w[f'{prefix}_recurrent'], w[f'{prefix}_bias']
    n = U.shape[0]
    h = np.zeros(n, np.float32)
    c = np.zeros(n, np.float32)
    proj = xs @ W + b
    for z0 in proj:
      z = z0 + h @ U
      i, f, g, o = sigmoid(z[:n]), sigmoid(z[n:2 * n]), np.tanh(z[2 * n:3 * n]), sigmoid(z[3 * n:])
      c = f * c + i * g
      h = o * np.tanh(c)
    return h

  def predict(self, text):
    words, ids = self.vectorize(text)
    ids = [i for i in ids if i != 0]
    if not ids: return None, words, ids
    xs = self.w['embedding'][ids]
    h = np.concatenate([self.lstm(xs, 'fw'), self.lstm(xs[::-1], 'bw')])
    d = np.maximum(h @ self.w['d1_kernel'] + self.w['d1_bias'], 0)
    return float((d @ self.w['d2_kernel'] + self.w['d2_bias'])[0]), words, ids
