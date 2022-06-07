import skfuzzy as fuzz


def trim_gen(value, val_range, amount):
  start, end, _ = val_range
  step = (end - start) / (amount - 1)
  coords = [start, start, step]
  counter = 0
  while counter < amount:
    value[f'm{value.label}{counter + 1}'] = fuzz.trimf(value.universe, coords)
    coords[0] += step if counter else 0
    coords[1] += step
    coords[2] += step
    counter += 1


def trap_gen(value, val_range, amount):
  start, end, _ = val_range
  step = (end - start) / (amount - 1)
  coords = [start, start, step * 0.75, step]
  counter = 0
  while counter < amount:
    value[f'm{value.label}{counter + 1}'] = fuzz.trapmf(value.universe, coords)
    if counter == 0:
       coords[0] += step * 0.75
    else:
      coords[0] += step
    coords[1] += step
    coords[2] += step
    coords[3] += step
    counter += 1


def gauss_gen(value, val_range, amount):
  start, end, _ = val_range
  step = (end - start) / (amount - 1)
  mean = start
  counter = 0
  sigma = 10e-2
  while counter < amount:
    value[f'm{value.label}{counter + 1}'] = fuzz.gaussmf(value.universe, mean, sigma)
    mean += step
    counter += 1


def gen_terms(variables, val_range, amount, mem_gen_fn):
  for x in variables:
    mem_gen_fn(x, val_range, amount)
