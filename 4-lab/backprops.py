import neurolab as nl
import theano
import theano.tensor as T
import lasagne


def elman_recurrent_backprop(xy, z,
  hidden_num=[],
  epochs=100,
  goal=0.1
):
  net = nl.net.newelm([[-1, 1], [-1, 1]], hidden_num + [1])
  for i in range(len(hidden_num)):
    net.layers[i].initf = nl.init.InitRand([-0.1, 0.1], 'wb')
  net.init()
  net.trainf = nl.train.train_gd
  error = net.train(xy, tuple((_z,) for _z in z),
    epochs=epochs, show=None, goal=goal)
  output = net.sim(xy)
  return output, error


def cascade_forward_backprop(xy, z,
  hidden_num=[],
  epochs=100,
  learning_rate=1
):
  l_in = lasagne.layers.InputLayer(shape=(len(xy), 2))
  ls_inner = [l_in]
  for l_num in hidden_num:
    merged = lasagne.layers.ConcatLayer(incomings=tuple(ls_inner), axis=1)
    l_hidden = lasagne.layers.DenseLayer(merged, num_units=l_num,
      nonlinearity=lasagne.nonlinearities.tanh)
    ls_inner.append(l_hidden)
  l_output = lasagne.layers.DenseLayer(ls_inner[-1], num_units=1,
    nonlinearity=None)
  layers = (l_in, l_output)
  return __handle_train(xy, z, layers, epochs, learning_rate)


def feed_forward_backprop(xy, z,
  hidden_num=[],
  epochs=100,
  learning_rate=1
):
  l_in = lasagne.layers.InputLayer(shape=(len(xy), 2))
  l_next = l_in
  for l_num in hidden_num:
    l_next = lasagne.layers.DenseLayer(l_next, num_units=l_num,
      nonlinearity=lasagne.nonlinearities.tanh)
  l_output = lasagne.layers.DenseLayer(l_next, num_units=1,
    nonlinearity=None)
  layers = (l_in, l_output)
  return __handle_train(xy, z, layers, epochs, learning_rate)

def __handle_train(xy, z, layers, epochs, learning_rate):
  l_in, l_output = layers
  net_output = lasagne.layers.get_output(l_output)
  true_output = T.matrix('true_output')
  objective = lasagne.objectives.squared_error(net_output, true_output)
  loss = objective.mean()
  all_params = lasagne.layers.get_all_params(l_output)

  updates = lasagne.updates.sgd(loss, all_params, learning_rate=learning_rate)
  train = theano.function([l_in.input_var, true_output], loss, updates=updates)
  get_output = theano.function([l_in.input_var], net_output)

  errs = list()
  for _ in range(epochs):
    err = train(xy, tuple((_z,) for _z in z))
    errs.append(err)

  z_predicted = get_output(xy)
  return z_predicted, errs
