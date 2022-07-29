import os


def get_dir_size(dir):
    size = 0
    for root, dirs, files in os.walk(dir):
        size += sum([os.path.getsize(os.path.join(root, name)) for name in files])
    return size


def get_suit_size(size):
    return size / 1024.0 / 1024.0, 'MB'
