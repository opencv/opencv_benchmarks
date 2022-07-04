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

import numpy as np
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
        ret, corners = self.detector.detectMulti(image)
        if ret is False:
            return False, np.array([])
        self.detected_corners = corners
        return ret, corners

    def decode(self, image):
        if self.detected_corners.size == 0:
            return 0, [], None
        r, decoded_info, straight_qrcode = self.detector.decodeMulti(image, self.detected_corners)
        self.decoded_info = decoded_info
        return r, decoded_info, straight_qrcode


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
TypeNorm = Enum('TypeNorm', 'l1 l_inf')


def get_norm(gold_corners, corners, type_dist):
    if type_dist is TypeNorm.l1:
        ar = np.sum(np.absolute(gold_corners - corners), 1)
        return np.amax(ar)
    if type_dist is TypeNorm.l_inf:
        return abs(np.amax(gold_corners - corners))
    raise TypeError("this TypeNorm isn't supported")


def get_norm_to_rotate_qr(gold_corner, corners, accuracy, type_dist=TypeNorm.l_inf):
    corners = corners.reshape(-1, 4, 2)
    dist = 1e9
    for one_corners in corners:
        dist = get_norm(gold_corner, one_corners, type_dist)
        if dist > accuracy:
            for i in range(0, 3):
                if dist > accuracy:
                    one_corners = np.roll(one_corners, 1, 0)
                    dist = min(dist, get_norm(gold_corner, one_corners, type_dist))
                else:
                    return dist
        if dist > accuracy:
            one_corners = np.flip(one_corners, 0)
            dist = min(dist, get_norm(gold_corner, one_corners, type_dist))
            for i in range(0, 3):
                if dist > accuracy:
                    one_corners = np.roll(one_corners, 1, 0)
                    dist = min(dist, get_norm(gold_corner, one_corners, type_dist))
                else:
                    return dist
    return dist


def read_output(path):
    fs = cv.FileStorage(path, cv.FILE_STORAGE_READ)
    root = fs.root()
    for image in root.keys():
        if image.split('_')[0] == "img":
            image_category = image.split("_")[-2]
            image_info = root.getNode(image)
            corners = image_info.getNode("corners").mat()
            decoded_info = image_info.getNode("decoded_info")
            if not decoded_info.empty():
                for i in range(decoded_info.size()):
                    # print(decoded_info.at(i).string())
                    pass


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
                        type=int)
    parser.add_argument("-alg", "--algorithm", help="QR detect algorithm", default="opencv", action="store",
                        dest="algorithm", choices=['opencv', 'opencv_wechat'], type=str)
    parser.add_argument("--metric", help="Metric for distance between QR corners", default="l_inf", action="store",
                        dest="metric", choices=['l1', 'l_inf'], type=str)

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
    metric = args.metric

    list_dirs = glob.glob(dataset_path + "/*")
    fs = cv.FileStorage(output, cv.FILE_STORAGE_WRITE)
    detect_dict = {}
    decode_dict = {}
    fs.write("dataset_path", dataset_path)
    gl_count = 0
    gl_detect = 0
    gl_decode = 0
    qr = create_instance_qr(DetectorQR.TypeDetector[algorithm], model_path)
    for dir in list_dirs:
        imgs_path = find_images_path(dir)
        qr_count = 0
        qr_detect = 0
        qr_decode = 0
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
            if ret is True:
                i = 0
                r, decoded_info, straight_qrcode = qr.decode(image)
                if len(decoded_info) > 0:
                    for info in decoded_info:
                        if info != "":
                            qr_decode += 1
                fs.write("decoded_info", decoded_info)
                for one_gold_corners in gold_corners:
                    dist = get_norm_to_rotate_qr(one_gold_corners, corners, accuracy)
                    fs.write("dist_to_gold_corner_" + str(i), dist)
                    if dist <= accuracy:
                        qr_detect += 1
                    i += 1
            fs.endWriteStruct()
        category = (dir.replace('\\', '_')).replace('/', '_').split('_')[-1]
        detect_dict[category] = {"nums": qr_count, "detected": qr_detect, "detected_prop": qr_detect / max(1, qr_count)}
        decode_dict[category] = {"nums": qr_count, "decoded": qr_decode, "decoded_prop": qr_decode / max(1, qr_count)}
        print(dir, qr_detect / max(1, qr_count), qr_decode / max(1, qr_count), qr_count)
        gl_count += qr_count
        gl_detect += qr_detect
        gl_decode += qr_decode
    print(gl_count)
    print(gl_detect)
    print(gl_detect / gl_count)
    print("decode", gl_decode / gl_count)
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
    main()
