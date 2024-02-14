#!/usr/bin/env python

"""qr.py
Usage example:
python qr.py -o out -p qrcodes/detection
-H, --help - show help
-o, --output - output path
-p, --path - input dataset path (default qrcodes/detection)
--per_image_statistic - print the per image statistic
-a, --accuracy_threshold - input accuracy_threshold (default 20 pixels)
-alg, --algorithm - input alg (default opencv)
--metric - input norm (default l_inf)
"""

import time
import cv2 as cv
import numpy as np
from numpy import linalg as LA
from collections import OrderedDict
import argparse
import os
import glob
import pandas as pd
import datetime
import matplotlib.pyplot as plt
import json
from enum import Enum
from iteration_utilities import deepflatten

# l_1 - https://en.wikipedia.org/wiki/Norm_(mathematics)
# l_inf - Chebyshev norm https://en.wikipedia.org/wiki/Chebyshev_distance
TypeNorm = Enum('TypeNorm', 'l1 l2 l3 l_inf intersection_over_union')


def get_max_error(accuracy_threshold, metric):
    if metric is TypeNorm.l1 or metric is TypeNorm.l2 or metric is TypeNorm.l_inf:
        return 2 * accuracy_threshold
    if metric is TypeNorm.intersection_over_union:
        return 1.
    raise TypeError("this TypeNorm isn't supported")


def get_norm(gold_corner, corner, metric):
    if metric is TypeNorm.l1:
        return LA.norm((gold_corner - corner).flatten(), ord=1)
    if metric is TypeNorm.l2 or metric is TypeNorm.intersection_over_union:
        return LA.norm((gold_corner - corner).flatten(), ord=2)
    if metric is TypeNorm.l_inf:
        return LA.norm((gold_corner - corner).flatten(), ord=np.inf)
    raise TypeError("this TypeNorm isn't supported")


def get_norm_to_rotate_qr(gold_corner, corner, accuracy_threshold, metric):
    dist = get_max_error(accuracy_threshold, metric)

    if metric is TypeNorm.intersection_over_union:
        rect = cv.boundingRect(np.concatenate((gold_corner, corner), dtype=np.float32))
        mask1 = np.zeros((rect[2], rect[3]), dtype=np.uint8)
        mask1 = cv.fillConvexPoly(mask1, gold_corner.astype(np.int32) - (rect[0], rect[1]), 255)
        mask2 = np.zeros((rect[2], rect[3]), dtype=np.uint8)
        mask2 = cv.fillConvexPoly(mask2, np.rint(corner).astype(np.int32) - (rect[0], rect[1]), 255)
        a = cv.countNonZero(cv.bitwise_and(mask2, mask1))
        b = max(cv.countNonZero(cv.bitwise_or(mask2, mask1)), 1)
        return 1. - a / b

    dist = min(dist, get_norm(gold_corner, corner, metric))
    for i in range(0, 3):
        corner = np.roll(corner, 1, 0)
        dist = min(dist, get_norm(gold_corner, corner, metric))
    corner = np.flip(corner, 0)
    dist = min(dist, get_norm(gold_corner, corner, metric))
    for i in range(0, 3):
        corner = np.roll(corner, 1, 0)
        dist = min(dist, get_norm(gold_corner, corner, metric))
    return dist


class DetectorQR:
    TypeDetector = Enum('TypeDetector', 'opencv opencv_aruco opencv_wechat')

    def __init__(self):
        self.detected_corners = np.array([])
        self.decoded_info = []
        self.detector = None

    def detect_and_decode(self, image):
        return False, self.decoded_info, self.detected_corners

    def detect_and_check(self, image, gold_corners, accuracy_threshold, metric, show_detected=False):
        ret, decoded_info, corners = self.detect_and_decode(image)

        nearest_distance_from_gold = np.full(len(gold_corners), get_max_error(accuracy_threshold, metric))
        nearest_id_from_gold = np.full(len(gold_corners), -1)
        decoded_info_gold = [""] * len(gold_corners)

        if ret is True:
            if show_detected:
                for corner in corners:
                    cv.line(image, corner[0].astype(np.int32), corner[1].astype(np.int32), (127, 127, 127), 6)
                    cv.line(image, corner[1].astype(np.int32), corner[2].astype(np.int32), (127, 127, 127), 6)
                    cv.line(image, corner[2].astype(np.int32), corner[3].astype(np.int32), (127, 127, 127), 6)
                    cv.line(image, corner[3].astype(np.int32), corner[0].astype(np.int32), (127, 127, 127), 6)
                if min(image.shape) > 1080:
                    image = cv.resize(image, None, fx=0.5, fy=0.5)
                cv.imshow("qr", image)
                cv.waitKey(0)

            nearest_gold_distance = np.full(len(corners), get_max_error(accuracy_threshold, metric))
            nearest_gold_id = np.full(len(corners), -1)

            for i, gold_corner in enumerate(gold_corners):
                for j, corner in enumerate(corners):
                    distance = get_norm_to_rotate_qr(gold_corner, corner, accuracy_threshold, metric)
                    if distance < nearest_distance_from_gold[i]:
                        nearest_distance_from_gold[i] = distance
                        nearest_id_from_gold[i] = j
                    if decoded_info[j] != "" and distance < nearest_gold_distance[j]:
                        nearest_gold_distance[j] = distance
                        nearest_gold_id[j] = i
            for i, detected_id in enumerate(nearest_gold_id):
                if detected_id != -1:
                    decoded_info_gold[detected_id] = decoded_info[i]
        return nearest_distance_from_gold, decoded_info_gold


class CvObjDetector(DetectorQR):
    def __init__(self):
        super().__init__()
        self.detector = cv.QRCodeDetector()

    def detect(self, image):
        _, decoded_info, corners, _ = self.detector.detectAndDecodeMulti(image)
        if corners is None or len(corners) == 0:
            return False, np.array([])
        self.decoded_info = decoded_info
        self.detected_corners = corners
        return True, corners

    def decode(self, image):
        if len(self.decoded_info) == 0:
            return 0, [], None
        return True, self.decoded_info, self.detected_corners

    def detect_and_decode(self, image):
        ret, decoded_info, corners, _ = self.detector.detectAndDecodeMulti(image)
        self.decoded_info = decoded_info
        self.detected_corners = corners
        return ret, decoded_info, corners


class CvArucoDetector(CvObjDetector):
    def __init__(self):
        super().__init__()
        self.detector = cv.QRCodeDetectorAruco()


class CvWechatDetector(DetectorQR):
    def __init__(self, path_to_model="./"):
        super().__init__()
        self.detector = cv.wechat_qrcode_WeChatQRCode(path_to_model + "detect.prototxt",
                                                      path_to_model + "detect.caffemodel",
                                                      path_to_model + "sr.prototxt",
                                                      path_to_model + "sr.caffemodel")

    def detect_and_decode(self, image):
        decoded_info, corners = self.detector.detectAndDecode(image)
        if corners is None or len(corners) == 0:
            return False, [], []
        corners = np.array(corners).reshape(-1, 4, 2)
        self.decoded_info = decoded_info
        self.detected_corners = corners
        return True, decoded_info, corners

    def detect(self, image):
        decoded_info, corners = self.detector.detectAndDecode(image)
        if len(decoded_info) == 0:
            return False, np.array([])
        corners = np.array(corners).reshape(-1, 4, 2)
        self.decoded_info = decoded_info
        self.detected_corners = corners
        return True, corners

    def decode(self, image):
        if len(self.decoded_info) == 0:
            return 0, [], None
        return True, self.decoded_info, self.detected_corners


def create_instance_qr(type_detector=DetectorQR.TypeDetector.opencv, path_to_model="./"):
    if type_detector is DetectorQR.TypeDetector.opencv:
        return CvObjDetector()
    if type_detector is DetectorQR.TypeDetector.opencv_aruco:
        return CvArucoDetector()
    if type_detector is DetectorQR.TypeDetector.opencv_wechat:
        return CvWechatDetector(path_to_model)
    raise TypeError("this type_detector isn't supported")


def find_images_path(dir_path):
    images = glob.glob(dir_path + '/*.jpg')
    images += glob.glob(dir_path + '/*.png')
    return images


def get_gold_corners(label_path):
    f = open(label_path, "r")
    corners = []
    for line in f.readlines():
        try:
            f_list = [float(i) for i in line.split(" ")]
            corners += f_list
        except ValueError as e:
            pass
    return np.array(corners).reshape(-1, 4, 2)


def get_time():
    return datetime.datetime.now().strftime("%Y_%m_%d-%I_%M_%S_%p")


class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return json.JSONEncoder.default(self, obj)


def set_plt():
    # Turn interactive plotting off
    plt.ioff()
    plt.rcParams["figure.figsize"] = (15, 9)
    plt.rcParams["figure.subplot.bottom"] = 0.3
    plt.rcParams["figure.subplot.left"] = 0.05
    plt.rcParams["figure.subplot.right"] = 0.99


def get_and_print_category_statistic(obj_type, category, statistics, accuracy_threshold, path):
    if obj_type == "decode":
        decoded_info = list(deepflatten(statistics, ignore=str))
        decoded = sum(1 for info in decoded_info if info != "")
        category_statistic = OrderedDict(
            [("category", category), ("decoded", decoded/len(decoded_info)), ("total decoded", decoded)])
        return category_statistic
    objs = np.array(list(deepflatten(statistics)))
    detected = objs[objs < accuracy_threshold]
    category_statistic = OrderedDict(
        [("category", category), ("detected " + obj_type, len(detected) / max(1, len(objs))),
         ("total detected " + obj_type, len(detected)), ("total " + obj_type, len(objs)),
         ("average detected error " + obj_type, np.mean(detected))])
    data_frame = pd.DataFrame(objs)
    data_frame.hist(bins=500)
    plt.title(category + ' ' + obj_type)
    plt.xlabel('error')
    plt.xticks(np.arange(0., float(accuracy_threshold) + .25, .25))
    plt.ylabel('frequency')
    # plt.show()
    plt.savefig(path + '/' + category + '_' + obj_type + '.jpg')
    plt.close()
    return category_statistic


def print_statistics(filename, distances, accuracy_threshold, output_path, per_image_statistic):
    output_dict = output_path + '/' + filename
    if not os.path.exists(output_dict):
        os.mkdir(output_dict)
    with open(output_dict + "/" + "distances" + '.json', 'w') as fp:
        json.dump(distances, fp, cls=NumpyEncoder)
    result = []
    set_plt()
    for obj_type, statistics in distances.items():
        for category, image_names, category_statistics in zip(statistics[0], statistics[1], statistics[2]):
            result.append(
                get_and_print_category_statistic(obj_type, category, category_statistics, accuracy_threshold, output_dict))
            if not per_image_statistic:
                continue
            if not os.path.exists(output_dict + '/' + category):
                os.mkdir(output_dict + '/' + category)
            for image_name, image_statistics in zip(image_names, category_statistics):
                data_frame = pd.DataFrame({"error": image_statistics})
                data_frame.plot.bar(y='error')
                plt.xlabel('id')
                plt.ylabel('error')
                plt.savefig(output_dict + '/' + category + '/' + obj_type + '_' + image_name + '.jpg')
                plt.close()
        result.append(get_and_print_category_statistic(obj_type, 'all', statistics[2], accuracy_threshold, output_dict))
    if len(result) > 0:
        data_frame = pd.DataFrame(result).groupby('category', as_index=False, sort=True).last()
        print(data_frame.to_string(index=False))
    else:
        print("no data found, use --configuration=generate_run or --configuration=generate")


def dump_log(filename, img_name, output, distances, decoded=""):
    output_dict = output + '/' + filename
    if not os.path.exists(output_dict):
        os.mkdir(output_dict)
    with open(output_dict+"/"+"log.txt", 'a', encoding="utf-8") as f:
        f.write(img_name+":\n")
        for distance, info in zip(distances, decoded):
            f.write("decoded:" + info+'\n')
            f.write("distance:" + str(distance)+'\n')
        f.close()


def main():
    # parse command line options
    parser = argparse.ArgumentParser(description="bench QR code dataset", add_help=False)
    parser.add_argument("-H", "--help", help="show help", action="store_true", dest="show_help")
    parser.add_argument("-o", "--output", help="output path", default="output", action="store", dest="output")
    parser.add_argument("-p", "--path", help="input dataset path", default="qrcodes/detection", action="store",
                        dest="dataset_path")
    parser.add_argument("--per_image_statistic", help="print the per image statistic", action="store_true")
    parser.add_argument("-m", "--model", help="path to opencv_wechat model (detect.prototxt, detect.caffemodel,"
                                              "sr.prototxt, sr.caffemodel), build opencv+contrib to get model",
                        default="./", action="store",
                        dest="model_path")
    parser.add_argument("-a", "--accuracy_threshold", help="input accuracy_threshold", default="20", action="store", dest="accuracy_threshold",
                        type=float)
    parser.add_argument("-alg", "--algorithm", help="QR detect algorithm", default="opencv", action="store",
                        dest="algorithm", choices=['opencv', 'opencv_aruco', 'opencv_wechat'], type=str)
    parser.add_argument("--metric", help="Metric for distance between QR corners", default="l2", action="store",
                        dest="metric", choices=['l1', 'l2', 'l_inf', 'intersection_over_union'], type=str)

    args = parser.parse_args()
    show_help = args.show_help
    if show_help:
        parser.print_help()
        return
    output = args.output
    dataset_path = args.dataset_path
    model_path = args.model_path
    accuracy_threshold = args.accuracy_threshold
    algorithm = args.algorithm
    metric = TypeNorm.l_inf
    if args.metric == "l1":
        metric = TypeNorm.l1
    elif args.metric == "l2":
        metric = TypeNorm.l2
    elif args.metric == "intersection_over_union":
        metric = TypeNorm.intersection_over_union

    list_dirs = glob.glob(dataset_path + "/*")

    qr = create_instance_qr(DetectorQR.TypeDetector[algorithm], model_path)
    error_by_categories = {}
    filename = "report_" + get_time()
    for directory in list_dirs:
        if directory.split('/')[-1].split('\\')[-1].split('_')[0] == "report":
            continue
        distances = {"qr": [[], []], "decode": [[], []]}
        category = directory.split('/')[-1].split('\\')[-1]
        category = '_' + category if category[0] != '_' else category
        images_path = find_images_path(directory)
        for img_path in images_path:
            label_path = img_path[:-3] + "txt"
            gold_corners = get_gold_corners(label_path)
            image = cv.imread(img_path, cv.IMREAD_IGNORE_ORIENTATION)
            img_name = img_path[:-4].replace('\\', '_').replace('/', '_')

            nearest_distance_from_gold, decoded, = qr.detect_and_check(image, gold_corners, accuracy_threshold, metric)
            dump_log(filename, img_name, output, nearest_distance_from_gold, decoded)
            distance = {"qr": nearest_distance_from_gold, "decode": decoded}
            for key, value in distance.items():
                distances[key][0] += [img_name]
                distances[key][1] += [value]

        for key, value in distances.items():
            if key not in error_by_categories:
                error_by_categories[key] = [[], [], []]
            error_by_categories[key][0].append(category)
            error_by_categories[key][1].append(value[0])
            error_by_categories[key][2].append(value[1])
    print_statistics(filename, error_by_categories, accuracy_threshold, output, args.per_image_statistic)


if __name__ == '__main__':
    start = time.time()
    main()
    print("Total time:", time.time() - start, "sec")

