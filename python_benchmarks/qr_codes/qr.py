#!/usr/bin/env python

"""qr.py
Usage example:
python qr.py -o out.yaml -p qrcodes/detection
-H, --help - show help
-o, --output - output file (default out.yaml)
-p, --path - input dataset path (default qrcodes/detection)
-a, --accuracy - input accuracy (default 47)
-alg - input alg (default OpenCV)
-m, --metric - input metric (default ~)
"""

import argparse
import glob
import sys

import cv2 as cv
import numpy as np


def init_qr_detector_open_cv():
    return cv.QRCodeDetector()


def init_wechat_qr_detector_open_cv(path_to_model):
    return cv.wechat_qrcode.WeChatQRCode(path_to_model+'/detect.prototxt', path_to_model+'/detect.caffemodel',
                                         path_to_model+'sr.caffemodel')


def detect_qr_opencv(image_path, detector):
    image = cv.imread(image_path, cv.IMREAD_IGNORE_ORIENTATION)
    #ret, corners = detector.detect(image)
    ret, corners = detector.detectMulti(image)
    return ret, corners


def detect_wechat_qr_opencv(image_path, detector):
    image = cv.imread(image_path, cv.IMREAD_IGNORE_ORIENTATION)
    ret, corners = detector.detectAndDecode(image)
    return ret, corners


def find_images_path(dir_path):
    images = glob.glob(dir_path+'/*.jpg')
    images += glob.glob(dir_path+'/*.png')
    return images


def get_corners(label_path):
    f = open(label_path, "r")
    corners = []
    for line in f.readlines():
        try:
            f_list = [float(i) for i in line.split(" ")]
            corners += f_list
        except:
            pass
    return np.array(corners).reshape(-1, 4, 2)


def get_distance(gold_corners, corners):
    return abs(np.amax(gold_corners - corners))


def get_distance_to_rotate_qr(gold_corner, corners, accuracy):
    corners = corners.reshape(-1, 4, 2)
    dist = 1e9
    for one_corners in corners:
        dist = get_distance(gold_corner, one_corners)
        if dist > accuracy:
            for i in range(0, 3):
                if dist > accuracy:
                    one_corners = np.roll(one_corners, 1, 0)
                    dist = min(dist, get_distance(gold_corner, one_corners))
                else:
                    return dist
        if dist > accuracy:
            one_corners = np.flip(one_corners, 0)
            dist = min(dist, get_distance(gold_corner, one_corners))
            for i in range(0, 3):
                if dist > accuracy:
                    one_corners = np.roll(one_corners, 1, 0)
                    dist = min(dist, get_distance(gold_corner, one_corners))
                else: return dist
    return dist


def read_node():
    pass


def main():
    # parse command line options
    parser = argparse.ArgumentParser(description="bench QR code dataset", add_help=False)
    parser.add_argument("-H", "--help", help="show help", action="store_true", dest="show_help")
    parser.add_argument("-o", "--output", help="output file", default="out.svg", action="store", dest="output")
    parser.add_argument("-p", "--path", help="input dataset path", default="qrcodes/detection", action="store", dest="path")
    parser.add_argument("-a", "--accuracy", help="input accuracy", default="47", action="store", dest="accuracy", type=int)

    args = parser.parse_args()
    show_help = args.show_help
    if show_help:
        parser.print_help()
        return
    output = args.output
    path = args.path
    accuracy = args.accuracy


    listDirs = glob.glob(path+"/*")
    fs = cv.FileStorage("out.yaml", cv.FILE_STORAGE_WRITE)
    fs.write("path", path)
    gl_qr_count = 0
    gl_qr_detect = 0

    detector = init_qr_detector_open_cv()
    
    for dir in listDirs:
        imgsPath = find_images_path(dir)
        qr_count = 0
        qr_detect = 0
        for imgPath in imgsPath:
            labelPath = imgPath[:-3] + "txt"
            gold_corners = get_corners(labelPath)
            if gold_corners.shape[0] == 1 or True:
                qr_count += gold_corners.shape[0]
                ret, corners = detect_qr_opencv(imgPath, detector)

                img_name = imgPath[:-4].replace('\\', '_')
                img_name = img_name.replace('/', '_')

                fs.startWriteStruct(img_name, cv.FILE_NODE_MAP)
                fs.write("bool", int(ret))
                fs.write("gold_corners", gold_corners)
                fs.write("corners", corners)
                if ret is True:
                    i = 0
                    for one_gold_corners in gold_corners:
                        dist = get_distance_to_rotate_qr(one_gold_corners, corners, accuracy)
                        fs.write("dist_to_gold_corner_"+str(i), dist)
                        if dist <= accuracy:
                            qr_detect += 1
                        i += 1
                fs.endWriteStruct()
        #print(dir, qr_count, qr_detect, qr_detect / max(1, qr_count))
        print(dir, qr_detect / max(1, qr_count))
        gl_qr_count += qr_count
        gl_qr_detect += qr_detect
    print(gl_qr_count)
    print(gl_qr_detect)
    print(gl_qr_detect/gl_qr_count)


if __name__ == '__main__':
    main()
