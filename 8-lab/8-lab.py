import copy
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import numpy as np


A = 30  
B = 40
N = 15
MAX_ITERATIONS_BOUND = 1000
LEARNING_RATE = 0.001

C = np.random.randint(0, N, size=(N, N))
C = np.tril(C) + np.tril(C, -1).T
np.fill_diagonal(C, 0)

SHAPES = list()
while sum(a * b for (a, b) in SHAPES) < A * B and len(SHAPES) <= N:
  a = np.random.uniform(1, int(A / 4))
  b = np.random.uniform(1, int(B / 4))
  SHAPES.append((a, b))
SHAPES.pop(-1)

fits_best = lambda: min(CHROMOSOMES, key=lambda chrom: chrom[1])

CHROMOSOMES = list()
while len(CHROMOSOMES) < N:
  CHROMOSOMES.append((list(), 0.0))
  z, _ = CHROMOSOMES[-1]
  for a, b in SHAPES:
    bound_a = a / 2
    bound_b = b / 2
    x = np.random.uniform(bound_a, A - bound_a)
    y = np.random.uniform(bound_b, B - bound_b)
    z.append((x, y))


L_MAX = (N ** 2) * (A ** 2 + B ** 2) ** 0.5
COMMON_AREA_COEF = N * A * B
def calc_fitness_params(z):
  i = 0
  length = 0
  area = 0
  while i < len(z) - 1:
    u = i + 1
    x1, y1 = z[i]
    a1, b1 = SHAPES[i]
    while u < len(z):
      x2, y2 = z[u]
      a2, b2 = SHAPES[u]
      distance = pow(pow(x2 - x1, 2) + pow(y2 - y1, 2), 0.5)
      length += distance * C[i][u]
      r1 = [x1 - a1 / 2, y1 - b1 / 2, x1 + a1 / 2, y1 + b1 / 2]
      r2 = [x2 - a2 / 2, y2 - b2 / 2, x2 + a2 / 2, y2 + b2 / 2]
      if r1[0] >= r2[2] or r1[2] <= r2[0] or \
        r1[3] <= r2[1] or r1[1] >= r2[3]:
        area += 0
      else:
        area += (0.5 * (a2 + a1) - abs(x2 - x1)) * (0.5 * (b2 + b1) - abs(y2 - y1))
      u += 1
    i += 1
  return length / L_MAX, area / COMMON_AREA_COEF


def crossover():
  new_generation = list()
  chroms_sorted = sorted(CHROMOSOMES, key=lambda ch: ch[1])[:int(A / 2)]
  while len(new_generation) < N:
    p1 = np.random.randint(0, len(chroms_sorted))
    p2 = np.random.randint(0, len(chroms_sorted))
    while p1 == p2:
      p2 = np.random.randint(0, len(chroms_sorted))
    a = copy.deepcopy(chroms_sorted[p1][0])
    b = copy.deepcopy(chroms_sorted[p2][0])
    division = np.random.randint(1, len(a) - 1)
    offspring_1 = a[:division] + b[division:]
    offspring_2 = b[:division] + a[division:]
    if np.random.randint(0, 100) < 20:
      gen_i = np.random.randint(0, len(offspring_1))
      gen = offspring_1[gen_i]
      mutation_x = gen[0] + np.random.uniform(-A / 10, A / 10)
      mutation_y = gen[1] + np.random.uniform(-B / 100, B / 100)
      if mutation_x > 0 and mutation_x < A:
        offspring_1[gen_i] = (mutation_x, gen[1])
      if mutation_y > 0 and mutation_y < B:
        offspring_1[gen_i] = (mutation_x, mutation_y)
    if np.random.randint(0, 100) < 20:
      gen_i = np.random.randint(0, len(offspring_1))
      gen = offspring_2[gen_i]
      mutation_x = gen[0] + np.random.uniform(-A / 100, A / 100)
      mutation_y = gen[1] + np.random.uniform(-B / 100, B / 100)
      if mutation_x > 0 and mutation_x < A:
        offspring_2[gen_i] = (mutation_x, gen[1])
      if mutation_y > 0 and mutation_y < B:
        offspring_2[gen_i] = (mutation_x, mutation_y)
    new_generation.append((offspring_1, 0.0))
    new_generation.append((offspring_2, 0.0))
  return new_generation


def ga():
  global CHROMOSOMES
  i = 0
  iterations = list()
  while i <= MAX_ITERATIONS_BOUND:
    u = 0
    while u < len(CHROMOSOMES):
      z, _ = CHROMOSOMES[u]
      l, s = calc_fitness_params(z)
      fitness = LEARNING_RATE * l + s
      CHROMOSOMES[u] = (z, fitness)
      if s == 0:
        iterations.append((i, CHROMOSOMES[u][1]))
        return CHROMOSOMES[u], iterations
      u += 1
    iterations.append((i, fits_best()[1]))
    if i == MAX_ITERATIONS_BOUND: break
    CHROMOSOMES = crossover()
    i += 1
  return fits_best(), iterations


result, iterations = ga()
centers, fitness = result
print('Fitness:', fitness)
print('Centers:', centers)
print('Iterations:', iterations[-1][0])

fig, ax = plt.subplots()
ax.set_xlim(0, A)
ax.set_ylim(0, B)
i = 0
while i < len(SHAPES):
  w, h = SHAPES[i]
  x, y = centers[i]
  xy = (x - 0.5 * w, y - 0.5 * h)
  ax.add_patch(Rectangle(xy, w, h, edgecolor='r'))
  i += 1

plt.figure()

plt.plot(tuple(i for i, _ in iterations), tuple(f for _, f in iterations))

plt.show()
