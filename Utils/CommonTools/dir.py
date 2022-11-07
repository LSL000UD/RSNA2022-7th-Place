# -*- encoding: utf-8 -*-
import os


def get_sub_dirs(root_dir):
    list_outputs = []

    list_files = os.listdir(root_dir)
    no_sub_dir = True
    for file in list_files:
        cur_path = root_dir + '/' + file
        if os.path.isdir(cur_path):
            no_sub_dir = False
            list_outputs += get_sub_dirs(cur_path)

    if no_sub_dir:
        list_outputs.append(root_dir)

    return list_outputs


def try_mkdir(path, print_log=True):
    if not os.path.exists(path) and path != '':
        os.mkdir(path)

        if print_log:
            print('==> Make dir: {:s}'.format(path))


def try_recursive_mkdir(path, top_level=1, print_log=True):
    path_split = path.split('/')
    for i in range(top_level, len(path_split)):
        cur_path = str.join('/', path_split[:i + 1])
        try_mkdir(cur_path, print_log)


def find_files_in_dir(input_dir,
                      must_include_all=None,
                      must_include_one_of=None,
                      must_exclude_all = None
                      ):
    assert must_include_all is None or isinstance(must_include_all, str) or isinstance(must_include_all, list)
    assert must_include_one_of is None or isinstance(must_include_one_of, str) or isinstance(must_include_one_of, list)
    assert must_exclude_all is None or isinstance(must_exclude_all, str) or isinstance(must_exclude_all, list)

    if isinstance(must_include_all, str):
        must_include_all = [must_include_all]
    if isinstance(must_include_one_of, str):
        must_include_one_of = [must_include_one_of]
    if isinstance(must_exclude_all, str):
        must_exclude_all = [must_exclude_all]

    output = []
    for file in os.listdir(input_dir):
        is_target = True
        if must_include_all is not None:
            for target_str in must_include_all:
                assert isinstance(target_str, str)
                if file.lower().find(target_str.lower()) == -1:
                    is_target = False
                    break
        if not is_target:
            continue

        if must_include_one_of is not None:
            is_target = False
            for target_str in must_include_one_of:
                assert isinstance(target_str, str)
                if file.lower().find(target_str.lower()) > -1:
                    is_target = True
                    break
            if not is_target:
                continue

        is_target = True
        if must_exclude_all is not None:
            for target_str in must_exclude_all:
                assert isinstance(target_str, str)
                if file.lower().find(target_str.lower()) > -1:
                    is_target = False
                    break
        if not is_target:
            continue
        output.append(f"{input_dir}/{file}")

    return output
