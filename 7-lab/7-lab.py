from math import sin, cos
import random

import numpy as np
import matplotlib.pyplot as plt
from evol import Population, Evolution


FN_1 = lambda x: sin(abs(x)) * cos(3 * x / 2)
FN_2 = lambda x, y: x * sin(x + y)

X_1 = np.arange(0, 3, 0.01)
X_2 = np.arange(2, 6, 0.1)
Y = np.arange(1, 5, 0.1)

plt.plot(X_1, np.vectorize(FN_1)(X_1))
plt.figure()


x_mesh, y_mesh = np.meshgrid(X_2, Y)
z = np.vectorize(FN_2)(x_mesh, y_mesh)

ax = plt.axes(projection='3d')
ax.plot_surface(x_mesh, y_mesh, z, rstride=1, cstride=1, cmap='viridis', edgecolor='none')
ax.set_xlabel('x')
ax.set_ylabel('y')
ax.set_zlabel('z')


random.seed(26)

pick_parent = lambda arr: (np.random.choice(arr), np.random.choice(arr))
combine = lambda a, b: np.mean((a, b))
mutate_fn = lambda x, sigma: x + sigma * (random.random() - 0.5)
apply_2d = lambda fn, xy: tuple(fn(x) for x in xy)

# one-dimentional
population_1 = Population(chromosomes=X_1, eval_function=FN_1, maximize=False)
ev_1 = (Evolution()
  .survive(fraction=0.5)
  .breed(parent_picker=pick_parent, combiner=combine)
  .mutate(mutate_function=mutate_fn, sigma=0.1))

evolved_1 = population_1.evolve(
  evolution=Evolution().repeat(ev_1, n=10).evaluate(),
  n=10
)

# output
minimized_x = max(evolved_1, key=lambda x: x.fitness).chromosome
print(f'Min value for fn1: x = {minimized_x} where y = {FN_1(minimized_x)}')


# two-dimensional
population_2 = Population(
  chromosomes=tuple((X_2[i], Y[i]) for i in range(len(Y))),
  eval_function=lambda xy: FN_2(*xy),
  maximize=True
)
ev_2 = (Evolution()
  .survive(fraction=0.5)
  .breed(
    parent_picker=pick_parent,
    combiner=lambda a, b: (combine(a[0], b[0]), combine(a[1], b[1]))
  )
  .mutate(mutate_function=lambda xy, sigma: tuple(mutate_fn(x, sigma) for x in xy), sigma=0.1))

evolved_2 = population_2.evolve(
  evolution=Evolution().repeat(ev_2, n=10).evaluate(),
  n=10
)

# output
maximized_x, maximized_y = max(evolved_2, key=lambda x: x.fitness).chromosome
print(f'Max values for fn2: x = {maximized_x}; y = {maximized_y} where z = {FN_2(maximized_x, maximized_y)}')

plt.show()
