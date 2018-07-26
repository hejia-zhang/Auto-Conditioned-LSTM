import numpy as np


class DemonstrationInterface(object):
    """The interface which manages demonstrations"""

    def __init__(self, data):
        """
        Demonstration interface class.

        :param data: List[np.array]
                list of demonstrations
        """
        self.data = data
        self.obs_shape = data[0].shape[-1]

        self.max_obs = data[0][0]
        self.min_obs = data[0][0]

        # Get the max value and min value of demonstrations
        for demonstration in data:
            for obs in demonstration:
                for i in range(len(obs)):
                    self.max_obs[i] = max(self.max_obs[i], obs[i])
                    self.min_obs[i] = min(self.min_obs[i], obs[i])

        self.min_length = data[0].shape[0]
        for demonstration in data:
            self.min_length = min(self.min_length, demonstration.shape[0])

    def sample_seq_batch(self,
                         batch_size,
                         time_steps,
                         sample_speed):
        """
        Sample a batch of motion sequence.

        :param batch_size: int
                    size of the batch
        :param time_steps: int
                    length of one sequence
        :param sample_speed: int
                    speed of sampleing
        :return motion sequence batch: np.array[batch_size, time_steps]
        """
        batch = []

        for i in range(batch_size):
            demonstration_idx = np.random.randint(0, len(self.data))
            demonstration = self.data[demonstration_idx]

            # sample a trajectory with length of time_steps
            # the first and last several samples may be very noisy
            start_id = np.random.randint(10, demonstration.shape[0] - time_steps * sample_speed - 10)

            seq = []
            for j in range(time_steps):
                seq.append(demonstration[start_id + j * sample_speed])

            batch.append(seq)

        return np.array(batch)
