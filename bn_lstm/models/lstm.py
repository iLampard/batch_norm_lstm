""" Define LSTM Cell structure """

import tensorflow as tf
from tensorflow.keras import layers


class LSTM(layers.Layer):
    def __init__(self, hidden_dim, apply_bn=False):
        """ Initialize the LSTM Cell class  """
        super(LSTM, self).__init__()
        self.hidden_dim = hidden_dim
        self.apply_bn = apply_bn

    def build(self, inputs_shape):
        """ Build the network given the inputs_shape passed  """

        return

    def call(self, inputs, **kwargs):
        """ Return the hidden states for all time steps """
        return

    def one_step(self):
        """ Move along the time axis by one step  """
        return
