""" Define LSTM Cell structure """

import tensorflow as tf
from tensorflow.keras import layers


class LSTM(layers.Layer):
    def __init__(self,
                 hidden_dim,
                 apply_bn=False,
                 is_training=False,
                 decay=0.9):
        """
            Initialize the LSTM layer
            Build the network given the inputs_shape passed
            Vanilla LSTM architecture -  (Hochreiter & Schmidhuber, 1997)
            Batch norm architecture - https://arxiv.org/abs/1603.09025
        """
        super(LSTM, self).__init__()
        self.hidden_dim = hidden_dim
        self.apply_bn = apply_bn
        self.is_training = is_training
        self.decay = decay
        self.idx_step = 0
        self.layer_fc = layers.Dense(self.hidden_dim)

    def batch_norm(self, inputs, idx_step, scope, offset=0, scale=0.1, variance_epsilon=1e-5):
        with tf.variable_scope(scope):
            input_dim = inputs.get_shape().as_list()[-1]
            # Initialize the population stats for all time steps
            self.pop_mean = tf.get_variable(name='pop_mean',
                                            shape=[self.num_steps, input_dim],
                                            initializer=tf.zeros_initializer())

            self.pop_var = tf.get_variable(name='pop_var',
                                           shape=[self.num_steps, input_dim],
                                           initializer=tf.ones_initializer())
            pop_mean = self.pop_mean[idx_step]
            pop_var = self.pop_var[idx_step]
            batch_mean, batch_var = tf.nn.moments(inputs, [0])

            def batch_statistics():
                pop_mean_new = pop_mean * self.decay + batch_mean * (1 - self.decay)
                pop_var_new = pop_var * self.decay + batch_var * (1 - self.decay)
                with tf.control_dependencies([pop_mean.assign(pop_mean_new),
                                              pop_var.assign(pop_var_new)]):
                    return tf.nn.batch_normalization(inputs,
                                                     batch_mean,
                                                     batch_var,
                                                     offset,
                                                     scale,
                                                     variance_epsilon)

            def population_statistics():
                return tf.nn.batch_normalization(inputs,
                                                 pop_mean,
                                                 pop_var,
                                                 offset,
                                                 scale,
                                                 variance_epsilon)

            return tf.cond(self.is_training, batch_statistics, population_statistics)

    def call(self, inputs, **kwargs):
        """ Return the hidden states for all time steps """
        # Get the batch size from inputs
        # self.batch_size, self.num_steps = tf.shape(inputs)[0], tf.shape(inputs)[1]
        self.batch_size = tf.shape(inputs)[0]
        self.num_steps = inputs.get_shape().as_list()[1]

        self.init_hidden_state = tf.random_normal([self.batch_size, self.hidden_dim])
        self.init_cell_state = tf.random_normal([self.batch_size, self.hidden_dim])

        # Initialize the input - (batch_size, num_steps, input_dim)
        inputs = tf.expand_dims(inputs, axis=-1)

        # (num_steps, batch_size, input_dim)
        inputs_ = tf.transpose(inputs, perm=[1, 0, 2])

        # use scan to run over all time steps
        state_tuple = tf.scan(self.one_step,
                              elems=inputs_,
                              initializer=(self.init_hidden_state,
                                           self.init_cell_state,
                                           0))

        # (batch_size, num_steps, hidden_dim)
        all_hidden_state = tf.transpose(state_tuple[0], perm=[1, 0, 2])
        return all_hidden_state

    def one_step(self, prev_state_tuple, current_input):
        """ Move along the time axis by one step  """
        hidden_state, cell_state, idx_step = prev_state_tuple

        # (batch_size, hidden_dim + input_dim)
        concat_input = tf.concat([hidden_state, current_input], axis=-1)

        if self.apply_bn:
            concat_input = self.batch_norm(concat_input, idx_step, 'lstm_input')

        # (batch_size * 4, hidden_dim + input_dim)
        concat_input_tiled = tf.tile(concat_input, [4, 1])

        forget_, input_, output_, cell_bar = tf.split(self.layer_fc(concat_input_tiled),
                                                      axis=0,
                                                      num_or_size_splits=4)

        # (batch_size, hidden_dim)
        cell_state = tf.nn.sigmoid(forget_) * cell_state + tf.nn.sigmoid(input_) * tf.nn.tanh(cell_bar)
        if self.apply_bn:
            cell_state = self.batch_norm(cell_state, idx_step, 'lstm_cell_state')
        hidden_state = tf.nn.sigmoid(output_) * tf.nn.tanh(cell_state)

        return (hidden_state, cell_state, idx_step + 1)


class ClassificationModel:
    def __init__(self, hidden_dim, input_x_dim, num_class, apply_bn):
        self.hidden_dim = hidden_dim
        self.num_class = num_class
        self.apply_bn = apply_bn
        self.input_x_dim = input_x_dim

    def build(self):
        """ Build up the model  """

        self.input_x = tf.placeholder(tf.float32, shape=[None, self.input_x_dim])
        self.label = tf.placeholder(tf.float32, shape=[None, self.num_class])
        self.lr = tf.placeholder(tf.float32)
        self.is_training = tf.placeholder(tf.bool)

        self.lstm_layer = LSTM(self.hidden_dim, self.apply_bn, self.is_training)
        self.prediction_layer = layers.Dense(self.num_class)

        self.pred, self.loss = self.forward(self.input_x)

        self.opt_op = tf.train.RMSPropOptimizer(self.lr).minimize(self.loss)

    def forward(self, input_x):
        """ Make predictions and compute loss  """
        # (batch_size, num_steps, hidden_dim)
        all_states = self.lstm_layer(input_x)

        # (batch_size, hidden_dim)
        last_state = all_states[:, -1, :]

        # (batch_size, num_class)
        pred = tf.nn.softmax(self.prediction_layer(last_state), axis=1) + 1e-32

        loss = -tf.reduce_mean(tf.reduce_sum(self.label * tf.log(pred), axis=1))

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

    def predict(self, sess, batch_data):
        """ Define the prediction process """
        batch_x, batch_y = batch_data
        feed_dict = {self.input_x: batch_x,
                     self.label: batch_y,
                     self.is_training: False}

        loss, prediction = sess.run([self.loss, self.pred], feed_dict=feed_dict)

        return loss, prediction
