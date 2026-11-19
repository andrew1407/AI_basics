"""Exports the trained weights the showcase runs in the browser (NumPy forward passes in Pyodide).

Run once with TensorFlow, tensorflow-datasets and tf_keras installed, from the repo root:
  python web/models/export_models.py [mnist] [lab12] [elstm]
- mnist: runs 9-lab/9-lab.py itself (same model, 50 epochs) and saves its weights to mnist.npz
- lab12: runs 12-lab/12-lab.py itself (same model, 10 epochs) and saves its vocabulary and weights to lab12.npz
- elstm: copies the exam SavedModel's variables to elstm.npz and rebuilds the Tokenizer vocabulary the model
  was trained with (fit on imdb_reviews 'train', as in exam-task/model/ELSTM.ipynb) into exam-task/model/word_index.json
"""
import json
import os
import sys

import numpy as np

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
OUT = os.path.join(ROOT, 'web', 'models')
os.environ.setdefault('MPLBACKEND', 'Agg')


def run_lab(folder, script):
  cwd = os.getcwd()
  os.chdir(os.path.join(ROOT, folder))
  try:
    ns = dict(__name__='__main__')
    exec(compile(open(script).read(), script, 'exec'), ns)
    return ns
  finally:
    os.chdir(cwd)


def mnist():
  ns = run_lab('9-lab', '9-lab.py')
  model = ns['recognizer']
  arrays = {f'w{i}': w.astype(np.float32) for i, w in enumerate(model.get_weights())}
  np.savez_compressed(os.path.join(OUT, 'mnist.npz'), **arrays,
                      loss=np.float32(ns['loss']), accuracy=np.float32(ns['accuracy']),
                      history_loss=np.array(ns['recognizer'].history.history['loss'], np.float32),
                      history_accuracy=np.array(ns['recognizer'].history.history['accuracy'], np.float32))


def lab12():
  ns = run_lab('12-lab', '12-lab.py')
  model, encoder, history = ns['model'], ns['encoder'], ns['history']
  emb, bi, d1, d2 = model.layers[1], model.layers[2], model.layers[3], model.layers[4]
  fw, bw = bi.forward_layer.get_weights(), bi.backward_layer.get_weights()
  np.savez_compressed(os.path.join(OUT, 'lab12.npz'),
                      vocab=np.array(encoder.get_vocabulary()), embedding=emb.get_weights()[0],
                      fw_kernel=fw[0], fw_recurrent=fw[1], fw_bias=fw[2], bw_kernel=bw[0], bw_recurrent=bw[1], bw_bias=bw[2],
                      d1_kernel=d1.get_weights()[0], d1_bias=d1.get_weights()[1], d2_kernel=d2.get_weights()[0], d2_bias=d2.get_weights()[1],
                      **{f'history_{k}': np.array(v, np.float32) for k, v in history.history.items()})


def elstm():
  import tensorflow as tf
  import tensorflow_datasets as tfds
  from tf_keras.preprocessing.text import Tokenizer
  model = tf.saved_model.load(os.path.join(ROOT, 'exam-task', 'model', 'model1'))
  names = ['embeddings', 'conv_kernel', 'conv_bias', 'kernel', 'recurrent_kernel', 'bias', 'h_sp', 'exp_sum', 'e_t',
           'ro_min', 'ro_max', 'cur_epoch', 'dense_kernel', 'dense_bias']
  np.savez_compressed(os.path.join(OUT, 'elstm.npz'), **{n: v.numpy() for n, v in zip(names, model.variables)})
  # the notebook's preprocessing: Tokenizer(num_words=2000, oov_token='<OOV>') fit on str(bytes) of each train review
  train = tfds.load('imdb_reviews', split='train', as_supervised=True)
  tokenizer = Tokenizer(num_words=2000, oov_token='<OOV>')
  tokenizer.fit_on_texts([str(s.numpy()) for s, _ in train])
  word_index = {w: i for w, i in tokenizer.word_index.items() if i < 2000}  # texts_to_sequences maps the rest to <OOV>
  with open(os.path.join(ROOT, 'exam-task', 'model', 'word_index.json'), 'w') as f:
    json.dump(word_index, f, separators=(',', ':'))


if __name__ == '__main__':
  todo = sys.argv[1:] or ['mnist', 'lab12', 'elstm']
  for name in todo: globals()[name]()
