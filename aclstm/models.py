import tensorflow as tf
from tensorflow.contrib import layers
from tensorflow.python.ops.rnn import _transpose_batch_time


class Model(object):
    def __init__(self, name):
        self.name = name

    @property
    def vars(self):
        return tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=self.name)

    @property
    def trainable_vars(self):
        return tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=self.name)

    @property
    def perturbable_vars(self):
        return [var for var in self.trainable_vars if 'LayerNorm' not in var.name]


class ACMultiLayerdLSTM(Model):
    """Auto-Conditioned MultiLayerd LSTM."""

    def __init__(self,
                 nb_time_steps,
                 nb_lstm_units,
                 nb_lstm_layers,
                 batch_size,
                 name='ACLSTM',
                 layer_norm=True):
        """
        Auto-Conditioned MultiLayerd LSTM

        :param nb_time_steps: int
                    the time steps in one unroll
        :param nb_lstm_units: int
                    the number of lstm units in one lstm layer
        :param nb_lstm_layers: int
                    the number of lstm layers
        :param batch_size: int
        :param name: String
        :param layer_norm: Bool
                    if use layer_norm
        """
        super(ACMultiLayerdLSTM, self).__init__(name=name)
        self.nb_lstm_units = nb_lstm_units
        self.nb_lstm_layers = nb_lstm_layers
        self.nb_time_steps = nb_time_steps
        self.batch_size = batch_size
        self.layer_norm = layer_norm

    def __call__(self, input, conditioned_lst, reuse=False):
        """
        Use this to construct tensorflow network graph.

        :param input: tf.Placeholder
                the shape of tensor should be (batch_size, time_steps, feature_size)
        :param conditioned_lst: tf.Placeholder
                the shape of tensor should be (time_steps)
        :param reuse: Bool
                if reuse variable
        :return: network out tensor
                the shape of tensor should be [batch_size, time_steps, feature_size]
        """
        with tf.variable_scope(self.name) as scope:
            if reuse:
                scope.reuse_variables()

            lstm_layer = tf.nn.rnn_cell.MultiRNNCell([tf.nn.rnn_cell.BasicLSTMCell(self.nb_lstm_units,
                                                                                   name='lstm_{}'.format(i))
                                                      for i in range(self.nb_lstm_layers)])

            batch_size = self.batch_size

            initial_state = lstm_layer.zero_state(batch_size=batch_size, dtype=tf.float32)

            # raw_rnn expects time major inputs as TensorArrays
            time_steps = self.nb_time_steps
            inputs_ta = tf.TensorArray(dtype=tf.float32, size=time_steps, clear_after_read=False, name='Inputs')
            inputs_ta = inputs_ta.unstack(_transpose_batch_time(input))  # model_input is the input placeholder
            input_dim = input.get_shape()[-1].value  # the dimensionality of the input to each time step
            output_dim = input_dim  # the dimensionality of the model's output at each time step
            conditioned_ta = tf.TensorArray(dtype=tf.bool, size=time_steps, clear_after_read=False, name='Conditioned')
            conditioned_ta = conditioned_ta.unstack(conditioned_lst)

            def loop_fn(time, cell_output, cell_state, loop_state):
                elements_finished = (time >= time_steps)
                finished = tf.reduce_all(elements_finished)

                if cell_output is None:
                    next_cell_state = initial_state
                    emit_output = tf.zeros([output_dim])
                    # create input
                    next_input = inputs_ta.read(time)
                else:
                    next_cell_state = cell_state
                    emit_output = tf.layers.dense(cell_output, output_dim, reuse=tf.AUTO_REUSE)
                    if self.layer_norm:
                        emit_output = layers.layer_norm(emit_output, center=True, scale=True)
                    emit_output = tf.nn.relu(emit_output)

                    # if conditioned_lst[time] is 0, use current_output
                    next_input = tf.cond(finished,
                                         lambda: tf.zeros([batch_size, input_dim], dtype=tf.float32),
                                         lambda: tf.cond(conditioned_ta.read(time),
                                                         lambda: inputs_ta.read(time),
                                                         lambda: emit_output))

                # loop state not used in this example
                next_loop_state = None
                return elements_finished, next_input, next_cell_state, emit_output, next_loop_state

            out_ta, _, _ = tf.nn.raw_rnn(lstm_layer, loop_fn)
            out = _transpose_batch_time(out_ta.stack())
        return out




