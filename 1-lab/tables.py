def left_diagonal_values(table):
  left = 1
  key_val = lambda key: f'm{key}{left}'
  while left <= len(table):
    y = key_val('y')
    x = key_val('x')
    table_y = table[y]
    val = table_y[x]
    table_y.clear()
    table_y[x] = val
    left += 1


def get_fns_table(amounts, val_range, out_fn):
  in_amout, out_amount = amounts
  res_table, borders = __get_results_table(val_range, in_amout, out_fn)
  start, end = borders
  names_table = dict()
  step = (end - start) / (out_amount - 1)
  fns_vals = list()
  for i in range(out_amount):
    fns_vals.append(step * i)
  counter_y = 1
  for y in res_table:
    y_table = res_table[y]
    key_y = 'my%d' % counter_y
    names_table[key_y] = dict()
    counter_y += 1
    y_names = names_table[key_y]
    counter_x = 1
    for x in y_table:
      val = y_table[x]
      closest = min(fns_vals, key=lambda x: abs(x - val))
      fn_name = f'mf{fns_vals.index(closest) + 1}'
      y_names['mx%d' % counter_x] = fn_name
      counter_x += 1
  return names_table


def __get_results_table(val_range, amount, out_fn):
  fn_results_table = dict()
  start, end, _ = val_range
  step = (end - start) / (amount - 1)
  iterator = start
  val_min, val_max = float('inf'), 0.0
  while iterator <= end:
    fn_results_table[iterator] = dict()
    inner_table = fn_results_table[iterator]
    inner_iterator = start
    while inner_iterator <= end:
      inner_table[inner_iterator] = out_fn(inner_iterator, iterator)
      fn_res = inner_table[inner_iterator]
      if fn_res < val_min:
        val_min = fn_res
      elif fn_res > val_max:
        val_max = fn_res
      inner_iterator += step
    iterator += step
  return fn_results_table, (val_min, val_max)


def __show_table(table):
  for y in table:
    for x in table[y]:
      print(f'{y=} {x=} => {table[y][x]}')
