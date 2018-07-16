import numpy as np
import tensorflow as tf


class ACLSTM(object):
    """ Auto-Conditioned LSTM Original paper: https://arxiv.org/abs/1707.05363."""

    def __init__(self,
                 ac_multilayerd_lstm,
                 obs_shape,
                 min_obs,
                 max_obs,
                 lr,
                 time_steps,
                 nb_gtconditioned=5,
                 nb_autoconditioned=5,
                 obs_range=(0, 1)):
        """
        Auto-Conditioned LSTM.

        :param ac_multilayerd_lstm: Model
        :param obs_shape: int
                            the size of the observation,
                            e.g. number of observed joint values
        :param min_obs: np.array[float]
                            the min value of observation
                            the size of the array should
                            equal with observation_shape
        :param max_obs: np.array[float]
                            the max value of observation
                            the size of the array should
                            equal with observation_shape
        :param lr: float
                    learning rate
        :param time_steps: int
                    the time steps over which model will unroll
        """
        self.sess = tf.Session()

        # Input placeholders.
        self.obs_shape, self.min_obs, self.max_obs = obs_shape, min_obs, max_obs
        self.obs_series = tf.placeholder(tf.float32, [None, time_steps, obs_shape], 'obs_series')
        self.time_steps = time_steps
        self.condition_lst_ph = tf.placeholder(tf.bool, [time_steps], 'conditioned_lst')

        # Parameters.
        self.lr = lr
        self.obs_range = obs_range
        self.nb_gtconditioned = nb_gtconditioned
        self.nb_autoconditioned = nb_autoconditioned
        self.condition_lst = self.get_conditioned_lst(self.nb_gtconditioned, self.nb_autoconditioned, self.time_steps)

        # Models.
        self.ac_multilayerd_lstm = ac_multilayerd_lstm

        # Create network
        self.ac_multilayerd_lstm_tf = self.ac_multilayerd_lstm(self.obs_series, self.condition_lst_ph)

        # Set up optimizers
        self.setup_optimizer()

    def setup_optimizer(self):
        print('Setting up optimizer...')
        self.ac_multilayerd_lstm_loss = \
            tf.reduce_mean(tf.square(self.ac_multilayerd_lstm_tf[:, 0:self.time_steps-1] *
                                     (self.max_obs - self.min_obs) +
                                     self.min_obs -
                                     self.obs_series[:, 1:self.time_steps]))
        self.ac_multilayerd_lstm_grads = tf.gradients(self.ac_multilayerd_lstm_loss,
                                                      self.ac_multilayerd_lstm.trainable_vars)
        self.ac_multilayerd_lstm_optimizer = \
            tf.train.AdamOptimizer(self.lr).minimize(self.ac_multilayerd_lstm_loss,
                                                     var_list=self.ac_multilayerd_lstm.trainable_vars)

    def train(self, batch_data):
        ops = [self.ac_multilayerd_lstm_grads, self.ac_multilayerd_lstm_loss]

        ac_multilayerd_lstm_grads, ac_multilayerd_lstm_loss = self.sess.run(ops, feed_dict={
            self.obs_series: batch_data,
            self.condition_lst_ph: self.condition_lst
        })

        self.sess.run(self.ac_multilayerd_lstm_optimizer, feed_dict={self.obs_series: batch_data,
                                                                     self.condition_lst_ph: self.condition_lst})

        return ac_multilayerd_lstm_loss

    def initialize(self, sess):
        self.sess = sess
        self.sess.run(tf.global_variables_initializer())

    def reset(self):
        pass

    def get_conditioned_lst(self, nb_gtconditioned, nb_autoconditioned, time_steps):
        nb_gt_auto_pairs = int(time_steps / (nb_gtconditioned + nb_autoconditioned)) + 1
        gt_lst = np.ones((nb_gt_auto_pairs, nb_gtconditioned))
        auto_lst = np.zeros((nb_gt_auto_pairs, nb_autoconditioned))
        lst = np.concatenate((gt_lst, auto_lst), -1).reshape(-1)
        return np.array(lst[0:time_steps], dtype=bool)



