import os

import numpy as np

from data import read_bvh


def generate_traindata_from_bvh(input_dir, output_dir):
    print("Parsing bvh files in " + input_dir)

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    for entry in os.listdir(input_dir):
        entry_name, ext = os.path.splitext(entry)

        if ext == '.bvh':
            print("Processing " + entry)
            data = read_bvh.get_training_data(os.path.join(input_dir, entry))
            np.save(os.path.join(output_dir, entry_name + ".npy"), data)


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='Use this to generate training data from bvh files!')
    parser.add_argument('--input_dir', action='store', required=True, type=str)
    parser.add_argument('--output_dir', action='store', required=True, type=str)

    args = parser.parse_args()

    generate_traindata_from_bvh(args.input_dir, args.output_dir)
