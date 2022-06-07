import pandas as pd
import anfis
import matplotlib.pyplot as plt


dataset = pd.read_excel('Concrete_Data.xls')

AMOUNT_TOTAL = 1000
AMOUNT_TRAIN = int(AMOUNT_TOTAL * 0.8)

X = pd.DataFrame(
  data=dataset,
  columns=('Cement component', 'Blast Furnace Slag', 'Fly Ash', 'Water', 'Superplasticizer', 'Coarse Aggregate', 'Fine Aggregate', 'Age')
).to_numpy()[:AMOUNT_TOTAL]

Y = pd.DataFrame(
  data=dataset,
  columns=('Concrete compressive strength',)
).to_numpy()[:AMOUNT_TOTAL]

param = anfis.fis_parameters(
  n_input = len(X[0]),
  n_memb = 2,
  batch_size = 10,
  memb_func = 'gaussian',
  optimizer = 'adam',
  loss = 'huber_loss',
  n_epochs = 200
)

fis = anfis.ANFIS(n_input = param.n_input,
  n_memb = param.n_memb,
  batch_size = param.batch_size,
  memb_func = param.memb_func,
  name = 'myanfis'
)

fis.model.compile(optimizer=param.optimizer, loss=param.loss)

history = fis.fit(X[:AMOUNT_TRAIN], Y[:AMOUNT_TRAIN],
  epochs=param.n_epochs,
  batch_size=param.batch_size,
  validation_data = (X[AMOUNT_TRAIN:], Y[AMOUNT_TRAIN:]),
)

fis.plotmfs(show_initial_weights=True)

loss_curves = pd.DataFrame(history.history)
loss_curves.plot()

plt.show()
