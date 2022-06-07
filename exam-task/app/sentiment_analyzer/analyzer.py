import tensorflow as tf
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing.text import Tokenizer

from .elstm import ELSTMCell


class SentimentAnalyzer():
  """
    An analyzer that provides emotional text style prediction
    using tensorflow module architecture
  """

  def __init__(self, model_path):
    """
      Creates a new instance of a model facade that predicts
      emotional style of text; takes a path to a picked trained model
    """
    self.__model_path = model_path
    self.__vocab_size = 2000
    self.__oov = '<OOV>'
    self.__max_sequence_len = 1000

  
  def setup(self):
    """
      Prepares environment for predictions;
      creates a tokenizer, loads a model by a given path
    """
    self.__tokenizer = Tokenizer(num_words=self.__vocab_size, oov_token=self.__oov)
    self.__model = tf.keras.models.load_model(self.__model_path,
      custom_objects={'ELSTMCell': ELSTMCell})

  
  def predict(self, value):
    """
      Analyses given text (value) and estimates it using the passed model (engine);
      returns a tuple of numeric result (bounds: [0, 1]) and sentimental type
      (positive or negative)
    """
    value_arr = [value] + [''] * 31
    # value tokenization
    self.__tokenizer.fit_on_texts(value_arr)
    seqns = self.__tokenizer.texts_to_sequences(value_arr)
    padded = pad_sequences(seqns, maxlen=self.__max_sequence_len)
    # value result prediction
    predicted = self.__model.predict(padded)
    result = predicted[0]     # first given is a result of value
    return (result, 'positive' if result >= .5 else 'negative')
