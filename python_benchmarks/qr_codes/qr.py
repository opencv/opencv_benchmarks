#!/usr/bin/env python

"""qr.py
Usage example:
python qr.py -o out.yaml -p qrcodes/detection
-H, --help - show help
-o, --output - output file (default out.yaml)
-p, --path - input dataset path (default qrcodes/detection)
-a, --accuracy - input accuracy (default 20 pixels)
-alg, --algorithm - input alg (default opencv)
--metric - input norm (default l_inf)
"""

import argparse
import glob
from enum import Enum

import time
import numpy as np
from numpy import linalg as LA
import cv2 as cv


class DetectorQR:
    TypeDetector = Enum('TypeDetector', 'opencv opencv_wechat')

    def __init__(self):
        self.detected_corners = np.array([])
        self.decoded_info = []
        self.detector = None


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


class CvWechatDetector(DetectorQR):
    def __init__(self, path_to_model="./"):
        super().__init__()
        self.detector = cv.wechat_qrcode_WeChatQRCode(path_to_model + "detect.prototxt",
                                                      path_to_model + "detect.caffemodel",
                                                      path_to_model + "sr.prototxt",
                                                      path_to_model + "sr.caffemodel")

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


# l_1 - https://en.wikipedia.org/wiki/Norm_(mathematics)
# l_inf - Chebyshev norm https://en.wikipedia.org/wiki/Chebyshev_distance
TypeNorm = Enum('TypeNorm', 'l1 l2 l_inf intersection_over_union')


def get_norm(gold_corners, corners, type_dist):
    if type_dist is TypeNorm.l1:
        return LA.norm((gold_corners - corners).flatten(), 1)
    if type_dist is TypeNorm.l2 or type_dist is TypeNorm.intersection_over_union:
        return LA.norm((gold_corners - corners).flatten(), 2)
    if type_dist is TypeNorm.l_inf:
        return LA.norm((gold_corners - corners).flatten(), np.inf)
    raise TypeError("this TypeNorm isn't supported")


def get_norm_to_rotate_qr(gold_corner, corners, accuracy, type_dist=TypeNorm.l_inf):
    corners = corners.reshape(-1, 4, 2)
    dist = 1e9
    best_id = 0
    cur_id = 0
    for one_corners in corners:
        prev_dist = dist
        dist = min(dist, get_norm(gold_corner, one_corners, type_dist))
        for i in range(0, 3):
            one_corners = np.roll(one_corners, 1, 0)
            dist = min(dist, get_norm(gold_corner, one_corners, type_dist))
        one_corners = np.flip(one_corners, 0)
        dist = min(dist, get_norm(gold_corner, one_corners, type_dist))
        for i in range(0, 3):
            one_corners = np.roll(one_corners, 1, 0)
            dist = min(dist, get_norm(gold_corner, one_corners, type_dist))
        if dist < prev_dist:
            best_id = cur_id
        cur_id += 1

    if type_dist is TypeNorm.intersection_over_union:
        rect = cv.boundingRect(np.concatenate((gold_corner, corners[best_id]), dtype=np.float32))
        mask1 = np.zeros((rect[2], rect[3]), dtype=np.uint8)
        mask1 = cv.fillConvexPoly(mask1, gold_corner.astype(np.int32) - (rect[0], rect[1]), 255)
        mask2 = np.zeros((rect[2], rect[3]), dtype=np.uint8)
        mask2 = cv.fillConvexPoly(mask2, corners[best_id].astype(np.int32) - (rect[0], rect[1]), 255)
        a = cv.countNonZero(cv.bitwise_and(mask2, mask1))
        b = max(cv.countNonZero(cv.bitwise_or(mask2, mask1)), 1)
        dist = 1. - a / b
    return dist, best_id


def main():
    # parse command line options
    parser = argparse.ArgumentParser(description="bench QR code dataset", add_help=False)
    parser.add_argument("-H", "--help", help="show help", action="store_true", dest="show_help")
    parser.add_argument("-o", "--output", help="output file", default="out.yaml", action="store", dest="output")
    parser.add_argument("-p", "--path", help="input dataset path", default="qrcodes/detection", action="store",
                        dest="dataset_path")
    parser.add_argument("-m", "--model", help="path to opencv_wechat model (detect.prototxt, detect.caffemodel,"
                                              "sr.prototxt, sr.caffemodel), build opencv+contrib to get model",
                        default="./", action="store",
                        dest="model_path")
    parser.add_argument("-a", "--accuracy", help="input accuracy", default="20", action="store", dest="accuracy",
                        type=float)
    parser.add_argument("-alg", "--algorithm", help="QR detect algorithm", default="opencv", action="store",
                        dest="algorithm", choices=['opencv', 'opencv_wechat'], type=str)
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
    accuracy = args.accuracy
    algorithm = args.algorithm
    metric = TypeNorm.l_inf
    if args.metric == "l1":
        metric = TypeNorm.l1
    elif args.metric == "l2":
        metric = TypeNorm.l2
    elif args.metric == "intersection_over_union":
        metric = TypeNorm.intersection_over_union

    list_dirs = glob.glob(dataset_path + "/*")
    fs = cv.FileStorage(output, cv.FILE_STORAGE_WRITE)
    detect_dict = {}
    decode_dict = {}
    fs.write("dataset_path", dataset_path)
    gl_count = 0
    gl_detect = 0
    gl_decode = 0
    gl_dist = 0
    gl_pos_dist = 0
    qr = create_instance_qr(DetectorQR.TypeDetector[algorithm], model_path)
    for dir in list_dirs:
        imgs_path = find_images_path(dir)
        qr_count = 0
        qr_detect = 0
        qr_decode = 0
        category_dist = 0
        category_pos_dist = 0
        for img_path in imgs_path:
            label_path = img_path[:-3] + "txt"
            gold_corners = get_gold_corners(label_path)
            qr_count += gold_corners.shape[0]
            image = cv.imread(img_path, cv.IMREAD_IGNORE_ORIENTATION)

            ret, corners = qr.detect(image)
            img_name = img_path[:-4].replace('\\', '_')
            img_name = "img_" + img_name.replace('/', '_')
            fs.startWriteStruct(img_name, cv.FILE_NODE_MAP)
            fs.write("bool", int(ret))
            fs.write("gold_corners", gold_corners)
            fs.write("corners", corners)
            local_image_dist = 0.
            if ret is True:
                i = 0
                r, decoded_info, straight_qrcode = qr.decode(image)
                decoded_corners = []
                if len(decoded_info) > 0:
                    for info in decoded_info:
                        if info != "":
                            qr_decode += 1
                    for i in range(corners.shape[0]):
                        decoded_corners.append(corners[i])
                decoded_corners = np.array(decoded_corners)
                fs.write("decoded_info", decoded_info)
                for corner, decoded_info in zip(decoded_corners, decoded_info):
                    dist, best_id = get_norm_to_rotate_qr(corner, gold_corners, accuracy, metric)
                    gl_dist += dist
                    category_dist += dist
                    local_image_dist += (1.-dist) if metric is TypeNorm.intersection_over_union else dist
                    if dist <= accuracy:
                        qr_detect += 1
                    fs.write("dist_to_gold_corner_" + str(i), dist)
                    i += 1
                    if decoded_info != "":
                        gl_pos_dist += dist
                        category_pos_dist += dist
            fs.endWriteStruct()
        category = (dir.replace('\\', '_')).replace('/', '_').split('_')[-1]
        detect_dict[category] = {"nums": qr_count, "detected": qr_detect, "detected_prop": qr_detect / max(1, qr_count)}
        decode_dict[category] = {"nums": qr_count, "decoded": qr_decode, "decoded_prop": qr_decode / max(1, qr_count)}
        print(dir, qr_detect / max(1, qr_count), qr_decode / max(1, qr_count), qr_count)
        print("category_dist", category_dist / max(1, qr_detect), " category_pos_dist", category_pos_dist / max(1, qr_decode))
        gl_count += qr_count
        gl_detect += qr_detect
        gl_decode += qr_decode
    print(gl_count)
    print(gl_detect)
    print(gl_decode)
    print("gl_dist", gl_dist / max(1, gl_count), "gl_pos_dist", gl_pos_dist / max(1, gl_decode))
    print("detect", gl_detect / max(1, gl_count))
    print("decode", gl_decode / max(1, gl_count))
    detect_dict["total"] = {"nums": gl_count, "detected": gl_detect, "detected_prop": gl_detect / max(1, gl_count)}

    fs.startWriteStruct("category_detected", cv.FILE_NODE_MAP)
    for category in detect_dict:
        fs.startWriteStruct(category, cv.FILE_NODE_MAP)
        fs.write("nums", detect_dict[category]["nums"])
        fs.write("detected", detect_dict[category]["detected"])
        fs.write("detected_prop", detect_dict[category]["detected_prop"])
        fs.endWriteStruct()
    fs.endWriteStruct()

    decode_dict["total"] = {"nums": gl_count, "decoded": gl_decode, "decoded_prop": gl_decode / max(1, gl_count)}
    fs.startWriteStruct("category_decoded", cv.FILE_NODE_MAP)
    for category in decode_dict:
        fs.startWriteStruct(category, cv.FILE_NODE_MAP)
        fs.write("nums", decode_dict[category]["nums"])
        fs.write("decoded", decode_dict[category]["decoded"])
        fs.write("decoded_prop", decode_dict[category]["decoded_prop"])
        fs.endWriteStruct()
    fs.endWriteStruct()


if __name__ == '__main__':
    start = time.time()
    main()
    print(time.time() - start)
