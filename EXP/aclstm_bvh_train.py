import os

import numpy as np

from aclstm.aclstm import ACLSTM
from aclstm.models import ACMultiLayerdLSTM
from aclstm.training import train
from data.demonstration_interface import DemonstrationInterface


def load_demonstrations(folder):
    demonstrations = []

    for file in os.listdir(folder):
        demonstration = np.load(os.path.join(folder, file))
        demonstrations.append(demonstration)

    demonstration_interface = DemonstrationInterface(demonstrations)

    return demonstration_interface


def run(demonstrations_dir,
        batch_size=32,
        time_steps=100,
        frame_rate=120,
        desired_frame_rate=30):
    demonstration_interface = load_demonstrations(demonstrations_dir)
    time_steps = min(time_steps, demonstration_interface.min_length)
    ac_multilayerd_lstm = ACMultiLayerdLSTM(time_steps,
                                            nb_lstm_units=64,
                                            nb_lstm_layers=3,
                                            batch_size=batch_size)
    agent = ACLSTM(ac_multilayerd_lstm,
                   demonstration_interface.obs_shape,
                   demonstration_interface.min_obs,
                   demonstration_interface.max_obs,
                   time_steps)

    sample_speed = frame_rate / desired_frame_rate

    train(agent,
          demonstration_interface,
          nb_epochs=200000,
          batch_size=batch_size,
          sample_speed=sample_speed,
          time_steps=time_steps)


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='Use this to start a EXP with aclstm!')
    parser.add_argument('--demonstrations_dir', action='store', required=True, type=str)

    args = parser.parse_args()

    run(args.demonstrations_dir)
