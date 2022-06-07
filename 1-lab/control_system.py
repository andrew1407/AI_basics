from skfuzzy import control as ctrl
from skfuzzy.control.antecedent_consequent import Consequent


def gen_rules_system(inputs, output, names_table):
  x, y = inputs
  z = output
  rules_raw = dict()
  rules = list()
  for my in names_table:
    for mx in names_table[my]:
      z_val = names_table[my][mx]
      rule = rules_raw.get(z_val)
      condition = x[mx] & y[my]
      rules_raw[z_val] = (rule | condition) if rule else condition
  for key in rules_raw:
    condition = rules_raw[key]
    rule = ctrl.Rule(antecedent=condition, consequent=z[key], label=f'rule for {key}')
    rules.append(rule)
  return ctrl.ControlSystem(rules)


def simulate2d(system, input_range):
  sim = ctrl.ControlSystemSimulation(system)
  output = list()
  for i in input_range:
    sim.input['x'] = i
    sim.input['y'] = i
    sim.compute()
    output.append(sim.output['f'])
  return output
