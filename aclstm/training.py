import tensorflow as tf


def train(agent,
          demonstration_interface,
          nb_epochs,
          batch_size,
          time_steps,
          sample_speed):
    with tf.Session() as sess:
        agent.initialize(sess)

        tf.summary.FileWriter('../log', sess.graph)

        for epoch in range(nb_epochs):
            batch_data = demonstration_interface.next_batch(batch_size, time_steps, sample_speed)
            loss = agent.train(batch_data)

            if epoch % 10 == 0:
                print("The {} time epoch: {}".format(epoch, loss))
