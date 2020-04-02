import os
import re
import time
import shutil
import json
from glob import glob
from datetime import datetime


from utils.sheet_utils import *


# Sort-related functions #
def atoi(text):
    """Returns text as string unless it is made of digits
    Args:
        text (str): input text
    Returns:
        (int or str): return int if the text is made of digits; return string otherwise
    """

    return int(text) if text.isdigit() else text


def natural_keys(text):
    """Partitions text into letter and digit chunks for ordering purposes
    Args:
        text (str): input text
    Returns:
        (list of str and/or int): returns a list containing the divided chunks.
    Example:
        input = 'hey123how456are7you'
        output = ['hey', 123, 'how', 456, 'are', 7, 'you', '']
    """

    return [atoi(c) for c in re.split(r'(\d+)', text)]


# File management functions #
def listdir_fullpath(src):
    """Returns the full path of the files that exist in the input directory [src]
    Args:
        src (str): source directory that contains files
    Returns:
        (list of str): contains the full path of the files in the input directory
    """

    return [os.path.join(src, f) for f in os.listdir(src)]


def check_make_dest(dir_path):
    """Checks if the input directory exists, if not it creates it. Returns the full path of the
    input directory in either case.
    Args:
        dir_path (str): full path of the input directory
    Returns:
        dir_path (str): full path of the input directory
    """

    if not os.path.exists(dir_path):
        os.makedirs(dir_path)

    return dir_path


def get_all_files(src, file_types):
    """Returns all the files in [src] of type [file_types]
    Args:
        src (str): path of the source directory
        file_types (str or list of str): contains the extensions of the files to retrieve
    Returns:
        all_files (list of str): the list of the paths to files of the required type under the source directory
    """

    if isinstance(file_types, str):
        file_types = [file_types]

    all_files = []
    for ft in file_types:
        all_files.extend([y for x in os.walk(src) for y in glob(os.path.join(x[0], '*%s' % ft))])

    return all_files


def load_json(json_src):
    with open(json_src) as f:
        json_file = json.load(f)
    return json_file


def save_json_file(json_file, save_path):
    json.dump(json_file, open(save_path, 'w'))
    print('JSON file saved: ', save_path)


def merge_json_files(json_names, path_prefix=None):
    count = 0
    merged = dict()

    for json_name in json_names:
        jf = load_json(json_name)

        if path_prefix is not None:
            jf = update_path_prefix(jf, path_prefix)

        count += len(jf)
        merged = {**merged, **jf}

    print('[INFO] Master JSON file has been created')

    return merged


def update_path_prefix(json_file, prefix):
    for value in json_file.values():
        value['image_path'] = os.path.join(prefix, value['image_path'])

    return json_file