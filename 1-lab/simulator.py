import numpy as np
from skfuzzy import control as ctrl
import matplotlib.pyplot as plt

import terms
import tables
import control_system


class Simulator:
  def __init__(self, inputs, outputs, val_range, fn):
    # passed variables
    self.inputs = inputs
    self.outputs = outputs
    self.val_range = val_range
    self.fn = fn
    self.mem_fn = None
    # default values
    self.define_inner_variables()


  def simulate_variants(self, variants=[terms.trim_gen, terms.trap_gen, terms.gauss_gen]):
    self.__output_xy = [ self.fn(x, x) for x in self.__plot_range ]
    xy_avg = np.average(self.__output_xy)
    for mem_fn in variants:
      self.mem_fn = mem_fn
      first = True
      for diagonal in [False, True]:
        sys = self.__get_control_system(diagonal)
        self.__output_z = control_system.simulate2d(system=sys,
          input_range=self.__plot_range)
        title = mem_fn.__name__[:-4]
        if diagonal:
          title += ' (diagonal)'
        self.__calc_err(xy_avg, title)
        self.__plot_results(first)
        first = False
    self.mem_fn = None
    plt.show()


  def define_inner_variables(self):
    self.__plot_range = np.arange(self.val_range[0], self.val_range[1] + 0.2, self.val_range[2])
    self.__x = ctrl.Antecedent(self.__plot_range, 'x')
    self.__y = ctrl.Antecedent(self.__plot_range, 'y')
    self.__z = ctrl.Consequent(self.__plot_range, 'f')
    self.__output_xy = None
    self.__output_z = None


  def __set_terms(self):
    terms.gen_terms(
      variables=(self.__x, self.__y),
      val_range=self.val_range,
      amount=self.inputs,
      mem_gen_fn=self.mem_fn)
    terms.gen_terms(
      variables=(self.__z,),
      val_range=self.val_range,
      amount=self.outputs,
      mem_gen_fn=self.mem_fn)


  def __get_control_system(self, diagonal=False):
    names_table = tables.get_fns_table(
      val_range=self.val_range,
      amounts=(self.inputs, self.outputs),
      out_fn=self.fn)
    if diagonal:
      tables.left_diagonal_values(names_table)
    self.__set_terms()
    sys = control_system.gen_rules_system(
      inputs=(self.__x, self.__y),
      output=self.__z,
      names_table=names_table)
    return sys


  def __plot_results(self, plot_var=True):
    if plot_var:
      self.__x.view()
      # self.__y.view()
      self.__z.view()
    plt.figure()
    plt.plot(self.__plot_range, self.__output_xy)
    plt.plot(self.__plot_range, self.__output_z)


  def __calc_err(self, avg_xy, title):
    avg_z = np.average(self.__output_z)
    err = abs((avg_xy + 1) - (avg_z + 1)) / ((avg_xy + 1))
    print(f'{title} err = {err * 100}%')
