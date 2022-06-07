from math import sin
import os
import numpy as np
import matplotlib.pyplot as plt
import backprops as probs


OUT_DIR = 'out'
EPOCHS = 1000
LEARNING_RATE = 0.1
GOAL = 0.001
VAL_RANGE = np.arange(-1, 1, .05)
VARIANT_FN = lambda x, y: x * sin(x + y)

XY = tuple((x, x) for x in VAL_RANGE)
Z = tuple(VARIANT_FN(x, x) for x in VAL_RANGE)

def plot(vals, title, next=False):
  z, e = vals
  plt.title(title)
  plt.plot(VAL_RANGE, Z)
  plt.plot(VAL_RANGE, z)
  plt.savefig(f'{OUT_DIR}/{title}.png', dpi=100)
  plt.figure()
  plt.title(title + ' errors')
  plt.plot(e)
  plt.savefig(f'{OUT_DIR}/{title}.error.png', dpi=100)
  if next: plt.figure()


z_predicted10 = probs.feed_forward_backprop(XY, Z, hidden_num=[10],
  epochs=EPOCHS, learning_rate=LEARNING_RATE)

z_predicted20 = probs.feed_forward_backprop(XY, Z, hidden_num=[20],
  epochs=EPOCHS, learning_rate=LEARNING_RATE)

z_predicted_casc20 = probs.cascade_forward_backprop(XY, Z, hidden_num=[20],
  epochs=EPOCHS, learning_rate=LEARNING_RATE)

z_predicted_casc10 = probs.cascade_forward_backprop(XY, Z, hidden_num=[10, 10],
  epochs=EPOCHS, learning_rate=LEARNING_RATE)

el1 = probs.elman_recurrent_backprop(XY, Z, hidden_num=[15],
  epochs=EPOCHS, goal=GOAL)

el2 = probs.elman_recurrent_backprop(XY, Z, hidden_num=[5, 5, 5],
  epochs=EPOCHS, goal=GOAL)

if not os.path.exists(OUT_DIR):
  os.mkdir(OUT_DIR)

plot(z_predicted10, 'ffb_10', next=True)
plot(z_predicted20, 'ffb_20', next=True)
plot(z_predicted_casc20, 'cfb_20', next=True)
plot(z_predicted_casc10, 'cfb_10_10', next=True)
plot(el1, 'el_15', next=True)
plot(el1, 'el_5_5_5')
