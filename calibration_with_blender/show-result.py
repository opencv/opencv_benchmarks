import yaml
import os
import cv2 as cv
import matplotlib.pyplot as plt
import seaborn as sns

if __name__ == '__main__':
    data_dir = '/home/xperience/development/opencv_benchmarks/calibration_with_blender/work/checkerboard'

    with open(os.path.join(data_dir, 'result', 'result.yml')) as file:
        data = yaml.safe_load(file)
        print(data)

    errors = []
    d1_errors = []
    result_dir = os.path.join(data_dir, 'result')
    for entry in os.listdir(result_dir):
        if entry != 'result.yml':
            camera_file = cv.FileStorage(os.path.join(result_dir, entry), cv.FILE_STORAGE_READ)
            error = camera_file.getNode('avg_reprojection_error').real()
            errors.append(error)

            D = camera_file.getNode('distortion_coefficients').mat()
            d1_errors.append(data[entry]['d'][0] - D[0])
    #sns.distplot(errors)
    sns.distplot(d1_errors)
    plt.show()
