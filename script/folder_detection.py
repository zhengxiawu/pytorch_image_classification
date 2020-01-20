import os
import sys
from datetime import datetime
import time
import shutil

walk_dir = sys.argv[1]

print('walk_dir = ' + walk_dir)

print('walk_dir (absolute) = ' + os.path.abspath(walk_dir))


def get_folder_list(walk_dir):
    folder_list = []
    walk_dir = os.path.abspath(walk_dir)
    method_list = os.listdir(walk_dir)
    for method in method_list:
        model_list = os.listdir(os.path.join(walk_dir, method))
        if len(method_list) > 0:
            for model in model_list:
                dataset_list = os.listdir(os.path.join(walk_dir, method, model))
                for dataset in dataset_list:
                    train_name_list = os.listdir(os.path.join(walk_dir, method, model, dataset))
                    for train_name in train_name_list:
                        this_train_folder = os.path.join(walk_dir, method, model, dataset, train_name)
                        folder_list.append(this_train_folder)
    return folder_list


def folder_clean(walk_dir, remove_flag = False):
    folder_list = get_folder_list(walk_dir)
    for this_train_folder in folder_list:
        log_file = os.path.join(this_train_folder, 'logger.log')
        with open(log_file) as f:
            log_lines = f.readlines()
        time_str = log_lines[-1][0:17]
        time_str = time_str.replace(' ', '_')
        time_str = '2020/'+time_str
        FMT = '%Y/%m/%d_%H:%M:%S_%p'
        time_fmt = datetime.strptime(time_str, FMT)
        now_time = datetime.now()
        time_interval = now_time - time_fmt
        if time_interval.seconds > 60 * 60 * 2:
            if not os.path.isfile(os.path.join(this_train_folder, 'best.pth.tar')):
                print(this_train_folder)
                if remove_flag:
                    shutil.rmtree(this_train_folder)


def read_result(walk_dir):
    folder_list = get_folder_list(walk_dir)
    for this_train_folder in folder_list:
        log_file = os.path.join(this_train_folder, 'logger.log')
        with open(log_file) as f:
            log_lines = f.readlines()
        if 'Final' in log_lines[-1]:
            if 'dali' in this_train_folder.split('/')[-1]:
                print(this_train_folder)
                print(log_lines[-1])


if __name__ == '__main__':
    folder_clean(walk_dir, True)
    # read_result(walk_dir)