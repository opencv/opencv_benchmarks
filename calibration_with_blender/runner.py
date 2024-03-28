import json
import os
import pathlib
import shutil
import subprocess
import yaml
import random


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
    binary_path = '/home/xperience/development/opencv-fork/cmake-build-release/bin'
    image_distort_path = os.path.join(binary_path, 'example_cpp_image_distort')
    calibration_benchmark_path = os.path.join(binary_path, 'example_cpp_calibration_benchmark')

    datasets_path = '/home/xperience/development/datasets'
    pattern = 'checkerboard'
    dataset_path = os.path.join(datasets_path, pattern)

    work_dir = os.path.join('/home/xperience/development/opencv_benchmarks/calibration_with_blender', 'work', pattern)
    distorted_dir = os.path.join(work_dir, 'distorted')
    result_dir = os.path.join(work_dir, 'result')

    pathlib.Path(distorted_dir).mkdir(parents=True, exist_ok=True)
    pathlib.Path(result_dir).mkdir(parents=True, exist_ok=True)

    result = {}
    N = 3
    for i in range(N):
        image_number = 50
        k1 = random.uniform(-0.3, 0)
        k2 = random.uniform(-0.1, 0)
        p1 = random.uniform(-0.3, 0.3)
        p2 = random.uniform(-0.1, 0.1)
        k3 = random.uniform(-0.01, 0)

        clear_dir(distorted_dir)
        # Distort images and update info
        k = 0
        for entry in os.listdir(dataset_path):
            entry_path = os.path.join(dataset_path, entry)
            if os.path.isfile(entry_path):
                if entry != 'info.json':
                    if k >= image_number:
                        break
                    subprocess.run(
                        [image_distort_path, entry_path, 'pinhole', '1067', '1067', '0', '0', str(k1), str(k2), str(p1),
                         str(p2), str(k3),
                         distorted_dir])
                    k += 1
                else:
                    # with open(entry_path, 'r') as file:
                    #     info = json.load(file)
                    # info['camera']['d'] = [d1, d2]
                    #
                    # with open(os.path.join(work_dir, entry), 'w') as file:
                    #     json.dump(info, file, indent=4)
                    shutil.copy(entry_path, os.path.join(work_dir, entry))

        # Create image list
        image_list_path = os.path.join(work_dir, 'image_list')
        with open(image_list_path, 'w') as image_list:
            for entry in os.listdir(distorted_dir):
                entry_path = os.path.join(distorted_dir, entry)
                if os.path.isfile(entry_path):
                    image_list.write(entry_path + '\n')

        # Run calibration
        result_filename = 'c-{:04d}.yaml'.format(i)
        calibration_result_path = os.path.join(result_dir, result_filename)
        subprocess.run(
            [calibration_benchmark_path, '-w=13', '-h=18', '-s=1', '-op', '-o={}'.format(calibration_result_path),
             image_list_path])

        result[result_filename] = {'d': [k1, k2, p1, p2, k3]}
        with open(os.path.join(result_dir, 'result.yml'), 'w') as result_file:
            yaml.dump(result, result_file, default_flow_style=False)
