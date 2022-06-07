import numpy as np
import matplotlib.pyplot as plt
import skfuzzy as fuzz


FILENAME = 'dataset.txt'
CLUSTERS_AMOUNT = 15

def read_dataset(filename):
  res = x_arr, y_arr = list(), list()
  coef = 10e-6    
  with open(filename) as f:
    lines = f.read().splitlines()
    for line in lines:
      x, y = tuple(line.strip().split())
      x, y = (float(x) * coef, float(y) * coef)
      x_arr.append(x)
      y_arr.append(y)
  return res


def plot_coords(dataset, clusters):
  coords = x, y = np.vstack(dataset)
  plt.title('clusters')
  for i in range(len(x)):
    plt.plot(x[i], y[i], '.', color='blue')
  cntr, _, _, _, history, _, _ = fuzz._cluster.cmeans(
          coords, clusters, 1.5, error=0.005, maxiter=1000, init=None)
  for center_coords in cntr:
    x, y = center_coords
    plt.plot(x, y, '*', color='red', ms=20)

  plt.figure()
  plt.xlabel('iterations')
  plt.ylabel('fn values')
  plt.title('objective function change')
  plt.plot(history)
  plt.show()


plot_coords(read_dataset(FILENAME), CLUSTERS_AMOUNT)
