import tensorflow as tf


def synthesize(agent,
               output_dir,
               boost_data,
               model_path=None):
    """
    Synthesize a batch of new motion using learnt model.

    :param agent:
    :param model_path:
    :param output_dir:
    :param boost_data:
    :return:
    """
    with tf.Session() as sess:
        agent.initialize(sess)

        agent.load()

        agent.synthsize(boost_data)



