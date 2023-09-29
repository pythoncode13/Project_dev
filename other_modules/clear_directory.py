import os
import glob


def clear_directory(dir_path):
    files = glob.glob(f'{dir_path}*')
    for f in files:
        if os.path.exists(f):
            os.remove(f)
