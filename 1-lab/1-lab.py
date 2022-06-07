from math import sin
from simulator import Simulator


IN_AMOUNT = 6
OUT_AMOUNT = 9
VAL_RANGE = (0, 1, 10e-4)
VARIANT_FN = lambda x, y: x * sin(x + y)

simulator = Simulator(
  inputs=IN_AMOUNT,
  outputs=OUT_AMOUNT,
  val_range=VAL_RANGE,
  fn=VARIANT_FN
)

simulator.simulate_variants()
