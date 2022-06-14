import glob
import sys
import yaml

import cv2 as cv
import numpy as np

accuracy = 10


def detectQROpenCV(imagePath):
    image = cv.imread(imagePath, cv.IMREAD_IGNORE_ORIENTATION)
    detector = cv.QRCodeDetector()
    #ret, corners = detector.detect(image)
    ret, corners = detector.detectMulti(image)
    return ret, corners


def findImagesPath(dirPath):
    images = glob.glob(dirPath+'/*.jpg')
    images += glob.glob(dirPath+'/*.png')
    return images


def getCorners(labelPath):
    f = open(labelPath, "r")
    corners = []
    for line in f.readlines():
        try:
            f_list = [float(i) for i in line.split(" ")]
            corners += f_list
        except:
            pass
    return np.array(corners).reshape(-1, 4, 2)


def getDistance(gold_corners, corners):
    return abs(np.amax(gold_corners - corners))


def getDistanceToRotateQR(gold_corner, corners):
    corners = corners.reshape(-1, 4, 2)
    dist = 1e9
    for one_corners in corners:
        dist = getDistance(gold_corner, one_corners)
        if dist > accuracy:
            for i in range(0, 3):
                if dist > accuracy:
                    one_corners = np.roll(one_corners, 1, 0)
                    dist = min(dist, getDistance(gold_corner, one_corners))
                else: return dist            
        if dist > accuracy:
            one_corners = np.flip(one_corners, 0)
            dist = min(dist, getDistance(gold_corner, one_corners))
            for i in range(0, 3):
                if dist > accuracy:
                    one_corners = np.roll(one_corners, 1, 0)
                    dist = min(dist, getDistance(gold_corner, one_corners))
                else: return dist
    return dist


def bench(path = 'qrcodes/detection/*', loc_accuracy = 47):
    accuracy = loc_accuracy
    listDirs = glob.glob(path)
    gl_qr_count = 0
    gl_qr_detect = 0
    
    for dir in listDirs:
        imgsPath = findImagesPath(dir)
        qr_count = 0
        qr_detect = 0
        for imgPath in imgsPath:
            labelPath = imgPath[:-3] + "txt"
            gold_corners = getCorners(labelPath)
            if gold_corners.shape[0] == 1 or True:
                qr_count += gold_corners.shape[0]
                ret, corners = detectQROpenCV(imgPath)
                if ret is True:
                    for one_gold_corners in gold_corners:
                        dist = getDistanceToRotateQR(one_gold_corners, corners)
                        if dist > accuracy:
                            #print(imgPath)
                            print(dist)
                            #print(one_gold_corners)
                            #print()
                            #print(corners)
                            #print()
                        else:
                            qr_detect += 1
        #print(dir, qr_count, qr_detect, qr_detect / max(1, qr_count))
        print(dir, qr_detect / max(1, qr_count))
        gl_qr_count += qr_count
        gl_qr_detect += qr_detect
    print(gl_qr_count)
    print(gl_qr_detect)
    print(gl_qr_detect/gl_qr_count)


if __name__ == '__main__':
    print(cv.__version__)
    bench()
