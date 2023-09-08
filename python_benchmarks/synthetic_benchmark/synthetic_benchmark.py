#!/usr/bin/env python

"""synthetic_benchmark.py
Usage example:
python augmentation_benchmark.py -o out.yaml -p path
-H, --help - show help
-o, --output - output file (default out.yaml)
-p, --path - input dataset path
-a, --accuracy - input accuracy (default 20 pixels)
--metric - input norm (default l_inf)
"""

import argparse
from enum import Enum
import numpy as np
from numpy import linalg as LA
import cv2 as cv

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


def get_synthetic_rt(yaw, pitch, distance):
    rvec = np.zeros((3, 1), np.float64)
    tvec = np.zeros((3, 1), np.float64)

    rotZ = np.array([[0.], [0.], [-0.5 * np.pi]])
    rotX = np.array([[np.pi], [0.], [0.]])
    rvecTmp, tvecTmp = cv.composeRT(rotZ, np.zeros((3, 1), np.float64),
                                    rotX, np.zeros((3, 1), np.float64))[:2]

    rotPitch = np.array([[-pitch], [0], [0]])
    rotYaw = np.array([[0], [yaw], [0]])
    rvec, tvec = cv.composeRT(rotPitch, np.zeros((3, 1), np.float64),
                              rotYaw, np.zeros((3, 1), np.float64))[:2]

    rvec, tvec = cv.composeRT(rvecTmp, np.zeros((3, 1), np.float64),
                              rvec, np.zeros((3, 1), np.float64))[:2]
    tvec = np.array([[0.], [0.], [distance]])
    return rvec, tvec


def get_coord(num_rows, num_cols, start_x=0, start_y=0):
    i, j = np.ogrid[:num_rows, :num_cols]
    v = np.empty((num_rows, num_cols, 2), dtype=np.float32)
    v[..., 0] = j + start_y
    v[..., 1] = i + start_x
    v.shape = (1, -1, 2)
    return v


def find_img_points(name, chessboard):
    criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 30, 0.001)
    img_points = []
    img = cv.imread(name)
    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    # Find the chess board corners
    ret, corners = cv.findChessboardCorners(gray, chessboard, criteria)

    # If found, add object points, image points (after refining them)
    if ret is True:
        corners = cv.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
        img_points.append(corners)
        board_points = corners.astype(np.int32)

        if 0:
            cv.drawContours(gray, board_points, -1, (127, 127, 127), 10)
            cv.imshow(name, gray)
            cv.waitKey(0)
    return img_points, ret is True


class TransformObject:
    def transform_image(self, image):
        return image

    def transform_points(self, points):
        return points


class RotateTransform(TransformObject):
    def __init__(self, *, angle, rel_center=(0.5, 0.5)):
        self.angle = angle
        self.rel_center = rel_center
        self.rot_mat = None

    def transform_image(self, image):
        self.rot_mat = cv.getRotationMatrix2D(
            [self.rel_center[0] * image.shape[1], self.rel_center[0] * image.shape[0]],
            self.angle, 1.0)
        warp_rotate_dst = cv.warpAffine(image, self.rot_mat, (image.shape[1], image.shape[0]))
        return warp_rotate_dst

    def transform_points(self, points):
        assert self.rot_mat is not None
        points = np.array(points)
        assert len(points.shape) == 2
        project_mat = np.copy(self.rot_mat[:, :-1])
        if points.shape[1] == 3:
            points = points[:, :-1]
        points = points.transpose()
        res_points = np.dot(project_mat, points)
        res_points[0] += self.rot_mat[0, 2]
        res_points[1] += self.rot_mat[1, 2]
        res_points = res_points.transpose()
        return res_points


class BluerTransform(TransformObject):
    def __init__(self, *, ksize=(5, 5)):
        self.ksize = ksize

    def transform_image(self, image):
        return cv.blur(image, self.ksize)

    def transform_points(self, points):
        return super().transform_points(points)


# TODO: need to use rel_center
class OverlayTransform(TransformObject):
    def __init__(self, *, rel_center=(0.5, 0.5), background_object):
        self.rel_center = rel_center
        assert background_object.image is not None
        self.background_image = np.copy(background_object.image)
        self.row_offset = 0
        self.col_offset = 0

    def transform_image(self, image):
        self.row_offset = (self.background_image.shape[0] - image.shape[0]) // 2
        self.col_offset = (self.background_image.shape[1] - image.shape[1]) // 2
        background_image = np.copy(self.background_image)
        background_image[self.col_offset:self.col_offset + image.shape[0],
                         self.row_offset:self.row_offset + image.shape[1]] = image
        image = background_image
        return image

    def transform_points(self, points):
        points = np.array(points)
        assert len(points.shape) == 2
        if points.shape[1] == 3:
            points = points[:, :-1]
        points[:, 0] += self.col_offset
        points[:, 1] += self.row_offset
        return points


class UndistortFisheyeTransform:
    def __init__(self, *, img_size):
        self.cameraMatrix = np.eye(3, 3, dtype=np.float64)
        self.cameraMatrix[0, 0] = img_size[0]
        self.cameraMatrix[1, 1] = img_size[0]
        self.cameraMatrix[0, 2] = img_size[0] / 2
        self.cameraMatrix[1, 2] = img_size[0] / 2
        self.distCoeffs = np.zeros((4, 1), np.float64)
        self.distCoeffs[0] = -0.5012997
        self.distCoeffs[1] = -0.50116057

    def transform_image(self, image):
        undistorted_img = cv.fisheye.undistortImage(image, K=self.cameraMatrix, D=self.distCoeffs, Knew=self.cameraMatrix)
        return undistorted_img

    def transform_points(self, points):
        points = np.array(points)
        assert len(points.shape) == 2
        if points.shape[1] == 3:
            points = points[:, :-1]
        points = cv.fisheye.undistortPoints(points.reshape(1, -1, 2), K=self.cameraMatrix, D=self.distCoeffs, R=None, P=self.cameraMatrix)
        return points


class SyntheticObject:
    def __init__(self):
        self.image = None

    def transform_object(self, transform_object):
        return self

    def show(self, wait_key=0):
        pass


class BackGroundObject(SyntheticObject):
    def __init__(self, *, num_rows, num_cols):
        self.image = np.zeros((num_rows, num_cols), dtype=np.uint8)

    def show(self, wait_key=0):
        cv.imshow("BackGroundObject", self.image)
        cv.waitKey(wait_key)


class SyntheticCharuco(SyntheticObject):
    def __init__(self, *, board_size, cell_img_size, square_marker_length_rate=0.5, dict_id=0):
        self.board_size = board_size
        self.board_image_size = [board_size[0] * cell_img_size, board_size[1] * cell_img_size]
        self.square_marker_length_rate = square_marker_length_rate
        self.dict_id = dict_id
        self.dict = cv.aruco.getPredefinedDictionary(dict_id)
        self.charuco_board = cv.aruco.CharucoBoard(board_size, 1., 1. * square_marker_length_rate, self.dict)
        self.image = self.charuco_board.generateImage(self.board_image_size)
        self.aruco_corners = (np.array(self.charuco_board.getObjPoints(), dtype=np.float32)
                              * cell_img_size).reshape(-1, 3)[:, :-1]
        self.aruco_ids = np.array(self.charuco_board.getIds())
        self.chessboard_corners = (np.array(self.charuco_board.getChessboardCorners(), dtype=np.float32)
                                   * cell_img_size)[:, :-1]
        # Todo: need add check with detector =(((

    def transform_object(self, transform_object):
        self.image = transform_object.transform_image(self.image)
        self.aruco_corners = np.array(transform_object.transform_points(self.aruco_corners), dtype=np.float32)
        self.chessboard_corners = np.array(transform_object.transform_points(self.chessboard_corners), dtype=np.float32)
        return self

    def show(self, wait_key=0):
        assert self.image is not None
        image = np.copy(self.image)
        aruco = np.array(self.aruco_corners.reshape(-1, 1, 4, 2), dtype=np.float32)
        aruco = [el for el in aruco]

        cv.aruco.drawDetectedMarkers(image, aruco)
        cv.aruco.drawDetectedCornersCharuco(image, self.chessboard_corners)
        cv.imshow("SyntheticCharuco", image)
        cv.waitKey(wait_key)


class CharucoChecker:
    def __init__(self, synthetic_charuco):
        self.synthetic_charuco = synthetic_charuco
        self.charuco_board = synthetic_charuco.charuco_board
        self.aruco_detector = cv.aruco.ArucoDetector(self.charuco_board.getDictionary())
        self.charuco_detector = cv.aruco.CharucoDetector(self.charuco_board)

    def detect_and_check_aruco(self, type_dist=TypeNorm.l_inf):
        marker_corners, marker_ids, _ = self.aruco_detector.detectMarkers(self.synthetic_charuco.image)
        detected = {}
        for marker_id, marker in zip(marker_ids, marker_corners):
            detected[int(marker_id)] = marker.reshape(4, 2)
        gold_corners, gold_ids = self.synthetic_charuco.aruco_corners.reshape(-1, 4, 2),\
            self.synthetic_charuco.aruco_ids
        gold = {}
        for marker_id, marker in zip(gold_ids, gold_corners):
            gold[int(marker_id)] = marker
        dist = 0.
        detected_count = 0
        total_count = len(gold_ids)
        for gold_id in gold_ids:
            gold_corner = gold_corners[int(gold_id)]
            if int(gold_id) in detected:
                corner = detected[int(gold_id)]
                dist += get_norm(gold_corner, corner, type_dist)
                detected_count += 1
        dist /= detected_count
        return detected_count / total_count, dist, total_count

    def detect_and_check_charuco(self, type_dist):
        pass


def main():
    # parse command line options
    parser = argparse.ArgumentParser(description="augmentation benchmark", add_help=False)
    parser.add_argument("-H", "--help", help="show help", action="store_true", dest="show_help")
    parser.add_argument("-o", "--output", help="output file", default="out.yaml", action="store", dest="output")
    parser.add_argument("-p", "--path", help="input dataset path", default="", action="store",
                        dest="dataset_path")
    parser.add_argument("-a", "--accuracy", help="input accuracy", default="20", action="store", dest="accuracy",
                        type=float)
    parser.add_argument("--metric", help="Metric for distance between result and gold ", default="l2", action="store",
                        dest="metric", choices=['l1', 'l2', 'l_inf', 'intersection_over_union'], type=str)

    args = parser.parse_args()
    show_help = args.show_help
    if show_help:
        parser.print_help()
        return
    output = args.output
    dataset_path = args.dataset_path
    accuracy = args.accuracy
    metric = TypeNorm.l_inf
    if args.metric == "l1":
        metric = TypeNorm.l1
    elif args.metric == "l2":
        metric = TypeNorm.l2
    elif args.metric == "intersection_over_union":
        metric = TypeNorm.intersection_over_union

    cell_img_size = 100
    board_size = (5, 5)
    background = BackGroundObject(num_rows=700, num_cols=700)
    background.show()

    charuco_object = SyntheticCharuco(board_size=board_size, cell_img_size=cell_img_size)
    charuco_object.show()

    concat_object = OverlayTransform(background_object=background)
    charuco_object.transform_object(concat_object)
    charuco_object.show()

    b1 = BluerTransform()
    b2 = RotateTransform(angle=30)
    charuco_object.transform_object(b1)
    charuco_object.transform_object(b2)
    charuco_object.show()

    b3 = UndistortFisheyeTransform(img_size=charuco_object.image.shape)
    charuco_object.transform_object(b3)
    charuco_object.show()

    charuco_checker = CharucoChecker(charuco_object)
    res = charuco_checker.detect_and_check_aruco()
    print(res)


if __name__ == '__main__':
    main()
