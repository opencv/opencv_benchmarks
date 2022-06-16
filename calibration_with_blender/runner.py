#!/usr/bin/env python

import os
import pathlib
import shutil
import subprocess
import yaml
import random

binary_path = '/home/xperience/development/opencv-fork/cmake-build-release/bin'
image_distort_path = os.path.join(binary_path, 'example_cpp_image_distort')
calibration_benchmark_path = os.path.join(binary_path, 'example_cpp_calibration_benchmark')

datasets_path = '/home/xperience/development/opencv_benchmarks/calibration_with_blender/datasets'
dataset_path = os.path.join(datasets_path, 'checkerboard', '22-06-14-0')

work_dir = os.path.join('/home/xperience/development/opencv_benchmarks/calibration_with_blender', 'work')
distorted_dir = os.path.join(work_dir, 'distorted')
result_dir = os.path.join(work_dir, 'result')

pathlib.Path(distorted_dir).mkdir(parents=True, exist_ok=True)
pathlib.Path(result_dir).mkdir(parents=True, exist_ok=True)


def clear_dir(folder):
    for filename in os.listdir(folder):
        file_path = os.path.join(folder, filename)
        try:
            if os.path.isfile(file_path) or os.path.islink(file_path):
                os.unlink(file_path)
            elif os.path.isdir(file_path):
                shutil.rmtree(file_path)
        except Exception as e:
            print('Failed to delete %s. Reason: %s' % (file_path, e))


if __name__ == '__main__':

    result = {}
    for i in range(10):
        d1 = random.uniform(-0.3, 0)
        d2 = random.uniform(-0.1, 0)

        clear_dir(distorted_dir)
        for entry in os.listdir(dataset_path):
            entry_path = os.path.join(dataset_path, entry)
            if os.path.isfile(entry_path):
                subprocess.run([image_distort_path, entry_path, 'pinhole', '1067', '1067', '0', '0', str(d1), str(d2), distorted_dir])

        image_list_path = os.path.join(work_dir, 'image_list')
        with open(image_list_path, 'w') as image_list:
            for entry in os.listdir(distorted_dir):
                entry_path = os.path.join(distorted_dir, entry)
                if os.path.isfile(entry_path):
                    image_list.write(entry_path + '\n')

        calibration_result_path = os.path.join(result_dir, 'c-{}.yaml'.format(i))
        subprocess.run([calibration_benchmark_path, '-w=13', '-h=18', '-s=1', '-op', '-o={}'.format(calibration_result_path), image_list_path])

        result['c-{}.yml'.format(i)] = {'d': [d1, d2]}
        with open(os.path.join(result_dir, 'result.yml'), 'w') as result_file:
            yaml.dump(result, result_file, default_flow_style=False)
