import os
import sys
from datetime import datetime
import time
import shutil

walk_dir = sys.argv[1]

print('walk_dir = ' + walk_dir)

print('walk_dir (absolute) = ' + os.path.abspath(walk_dir))


def folder_reader(walk_dir, remove_flag = False):
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


if __name__ == '__main__':
    folder_reader(walk_dir, True)