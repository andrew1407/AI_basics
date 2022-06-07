import os
import numpy as np
import skfuzzy as fuzz
import matplotlib.pyplot as plt


RANGE = (0, 1, 0.01)
OUT_DIR = 'out'
CENTER = RANGE[1] / 2
SIGMA = 0.1
FUNCTIONS = dict()

FUNCTIONS[fuzz.trimf] = ([0.1, 0.4, 0.7],)
FUNCTIONS[fuzz.trapmf] = ([0.1, 0.3, CENTER, 0.7],)

FUNCTIONS[fuzz.gaussmf] = (CENTER, SIGMA)
FUNCTIONS[fuzz.gauss2mf] = (CENTER, SIGMA, 0.7, SIGMA / 10)
FUNCTIONS[fuzz.gbellmf] = (SIGMA, 4, CENTER)

FUNCTIONS[fuzz.sigmf] = (CENTER, 10)
FUNCTIONS[fuzz.dsigmf] = (CENTER - 0.2, 10, CENTER + 0.2, 30)
FUNCTIONS[fuzz.psigmf] = (0.2, 40, CENTER, 20)

FUNCTIONS[fuzz.zmf] = (0.1, 0.8)
FUNCTIONS[fuzz.pimf] = (0.1, 0.4, CENTER, 0.8)
FUNCTIONS[fuzz.smf] = (0.1, 0.8)

np_range = np.arange(*RANGE)

def operators_interpretation(fn1, fn2):
  mf1 = fn1(np_range, *FUNCTIONS[fn1])
  mf2 = fn2(np_range, *FUNCTIONS[fn2])
  _, or_out = fuzz.fuzzy_or(np_range, mf1, np_range, mf2)
  _, and_out = fuzz.fuzzy_and(np_range, mf1, np_range, mf2)
  title_or = f'{fn1.__name__}_or_{fn2.__name__}'
  title_and = f'{fn1.__name__}_and_{fn2.__name__}' 
  plt.title(title_or)
  plt.plot(np_range, or_out)
  plt.savefig(f'{OUT_DIR}/{title_or}.png', dpi=100)
  plt.figure()
  plt.title(title_and)
  plt.plot(np_range, and_out)  
  plt.savefig(f'{OUT_DIR}/{title_and}.png', dpi=100)


if not os.path.exists(OUT_DIR):
  os.mkdir(OUT_DIR)

for fn in FUNCTIONS:
  args = FUNCTIONS[fn]
  out = fn(np_range, *args)
  out_not = fuzz.fuzzy_not(out)
  plt.title(fn.__name__)
  plt.plot(np_range, out, label='original')
  plt.plot(np_range, out_not, label='not')
  plt.legend()
  plt.savefig(f'{OUT_DIR}/{fn.__name__}.png', dpi=100)
  plt.figure()

operators_interpretation(fuzz.gauss2mf, fuzz.zmf)
