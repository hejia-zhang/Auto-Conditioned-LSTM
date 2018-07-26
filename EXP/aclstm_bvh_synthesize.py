import os

import numpy as np

from aclstm.aclstm import ACLSTM
from aclstm.models import ACMultiLayerdLSTM
from aclstm.synthesizing import synthesize
from data.demonstration_interface import DemonstrationInterface


def load_demonstrations(folder):
    demonstrations = []

    for file in os.listdir(folder):
        demonstration = np.load(os.path.join(folder, file))
        demonstrations.append(demonstration)

    demonstration_interface = DemonstrationInterface(demonstrations)

    return demonstration_interface


def run(output_dir,
        demonstrations_dir,
        boost_seq_len,
        batch_size=32,
        frame_rate=120,
        synthesize_len=300,
        desired_frame_rate=30,
        model_path=None):
    demonstration_interface = load_demonstrations(demonstrations_dir)
    boost_seq_len = min(boost_seq_len, demonstration_interface.min_length)
    sample_speed = frame_rate / desired_frame_rate
    boost_data = demonstration_interface.sample_seq_batch(batch_size,
                                                          boost_seq_len,
                                                          sample_speed)
    ac_multilayerd_lstm = ACMultiLayerdLSTM(synthesize_len,
                                            nb_lstm_units=64,
                                            nb_lstm_layers=3,
                                            batch_size=batch_size)

    agent = ACLSTM(ac_multilayerd_lstm,
                   demonstration_interface.obs_shape,
                   demonstration_interface.min_obs,
                   demonstration_interface.max_obs)

    synthesize(agent,
               output_dir,
               boost_data)


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='Use this to systhesize a motion using aclstm!')
    parser.add_argument('--model_path', action='store', required=True, type=str)
    parser.add_argument('--output_dir', action='store', required=True, type=str)
    parser.add_argument('--demonstrations_dir', action='store', required=False, type=str)
    parser.add_argument('--boost_seq_length', action='store', required=True, type=int)
    parser.add_argument('--synthesize_length', action='store', required=False, type=int)

    args = parser.parse_args()

    run(args.output_dir,
        args.demonstrations_dir,
        args.boost_seq_length)
