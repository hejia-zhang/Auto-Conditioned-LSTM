import os


def make_path(path):
    return os.makedirs(path, exist_ok=True)
