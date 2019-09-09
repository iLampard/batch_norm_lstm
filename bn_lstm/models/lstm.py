""" Define LSTM Cell structure """

import tensorflow as tf
from tensorflow.keras import layers


class LSTM(layers.Layer):
    def __init__(self, hidden_dim, apply_bn=False):
        """
            Initialize the LSTM layer
            Build the network given the inputs_shape passed
            Vanilla LSTM architecture -  (Hochreiter & Schmidhuber, 1997)
            Batch norm architecture - https://arxiv.org/abs/1603.09025
        """
        super(LSTM, self).__init__()
        self.hidden_dim = hidden_dim
        self.apply_bn = apply_bn
        self.layer_fc = layers.Dense(self.hidden_dim)

    def call(self, inputs, **kwargs):
        """ Return the hidden states for all time steps """
        # Get the batch size from inputs
        self.batch_size = tf.shape(inputs)[0]

        self.init_hidden_state = tf.zeros(dtype=tf.float32, shape=[self.batch_size, self.hidden_dim])
        self.init_cell_state = tf.zeros(dtype=tf.float32, shape=[self.batch_size, self.hidden_dim])

        # (batch_size, num_steps, input_dim)
        inputs = tf.expand_dims(inputs, axis=-1)

        # (num_steps, batch_size, input_dim)
        inputs_ = tf.transpose(inputs, perm=[1, 0, 2])

        # use scan to run over all time steps
        state_tuple = tf.scan(self.one_step,
                              elems=inputs_,
                              initializer=(self.init_hidden_state, self.init_cell_state))

        # (batch_size, num_steps, hidden_dim)
        all_hidden_state = tf.transpose(state_tuple[0], perm=[1, 0, 2])
        return all_hidden_state

    def one_step(self, prev_state_tuple, current_input):
        """ Move along the time axis by one step  """
        hidden_state, cell_state = prev_state_tuple

        # (batch_size, hidden_dim + input_dim)
        concat_input = tf.concat([hidden_state, current_input], axis=-1)

        # (batch_size * 4, hidden_dim + input_dim)
        concat_input_tiled = tf.tile(concat_input, [4, 1])

        forget_, input_, output_, cell_bar = tf.split(self.layer_fc(concat_input_tiled),
                                                      axis=0,
                                                      num_or_size_splits=4)

        # (batch_size, hidden_dim)
        cell_state = tf.nn.sigmoid(forget_) * cell_state + tf.nn.sigmoid(input_) * cell_bar
        hidden_state = tf.nn.sigmoid(output_) * cell_state

        return (hidden_state, cell_state)


class ClassificationModel:
    def __init__(self, hidden_dim, num_class):
        self.hidden_dim = hidden_dim
        self.num_class = num_class
        return

    def build(self):
        """ Build up the model  """

        self.input_x = tf.placeholder(tf.float32, shape=[None, None])
        self.label = tf.placeholder(tf.float32, shape=[None, self.num_class])
        self.lr = tf.placeholder(tf.float32)
        self.is_training = tf.placeholder(tf.bool)

        self.lstm_layer = LSTM(self.hidden_dim)
        self.prediction_layer = layers.Dense(self.num_class)

        self.pred, self.loss = self.forward(self.input_x)

        self.opt_op = tf.train.AdamOptimizer(self.lr).minimize(self.loss)

    def forward(self, input_x):
        """ Make predictions and compute loss  """
        # (batch_size, num_steps, hidden_dim)
        all_states = self.lstm_layer(input_x)

        # (batch_size, hidden_dim)
        last_state = all_states[:, -1, :]

        # (batch_size, num_class)
        pred = tf.nn.softmax(self.prediction_layer(last_state))

        loss = tf.reduce_mean(-tf.reduce_sum(self.label * tf.log(pred)))

        return pred, loss

    def train(self, sess, batch_data, lr):
        """ Define the train process """
        batch_x, batch_y = batch_data
        feed_dict = {self.input_x: batch_x,
                     self.label: batch_y,
                     self.lr: lr,
                     self.is_training: True}

        _, loss, prediction = sess.run([self.opt_op, self.loss, self.pred], feed_dict=feed_dict)

        return loss, prediction

    def predict(self, sess, batch_data, lr):
        """ Define the prediction process """
        batch_x, batch_y = batch_data
        feed_dict = {self.input_x: batch_x,
                     self.label: batch_y,
                     self.lr: lr,
                     self.is_training: False}

        _, loss, prediction = sess.run([self.opt_op, self.loss, self.pred], feed_dict=feed_dict)

        return loss, prediction
