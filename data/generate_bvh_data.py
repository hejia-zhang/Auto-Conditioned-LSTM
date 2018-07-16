import os

import numpy as np

from data import read_bvh


def generate_bvh_from_traindata(input_dir, output_dir):
    print('Generating bvh data for ' + input_dir)

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    entries = os.listdir(input_dir)
    for entry in entries:
        entry_name, ext = os.path.splitext(entry)
        if ext == '.npy':
            print("Processing ", entry)
            demonstration_data = np.load(os.path.join(input_dir, entry))
            demonstration_data2 = []
            for i in range(int(demonstration_data.shape[0] / 8)):
                demonstration_data2 = demonstration_data2 + [demonstration_data[i * 8]]
            print(len(demonstration_data2))
            read_bvh.write_traindata_to_bvh(os.path.join(output_dir, entry_name + '.bvh'), np.array(demonstration_data2))


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='Use this to generate training data from bvh files!')
    parser.add_argument('--input_dir', action='store', required=True, type=str)
    parser.add_argument('--output_dir', action='store', required=True, type=str)

    args = parser.parse_args()

    generate_bvh_from_traindata(args.input_dir, args.output_dir)