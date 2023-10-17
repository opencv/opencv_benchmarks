#!/usr/bin/env python

"""objdetect_benchmark.py
Usage example:
python objdetect_benchmark.py -p path
-H, --help - show help
--configuration - script launch configuration (default generate_run)
-p, --path - path to the input/output of the dataset or detect statistics
-a, --accuracy - input accuracy, this is the object detection threshold (default 10 pixels)
--metric - input norm (default l_inf)
--marker_length_rate square marker length rate for charuco (default 0.5)
--board_x - input board x size (default 6)
--board_y - input board y size (default 6)
--rel_center_x - relative x-axis location of the center of the board in the image (default 0.5)
--rel_center_y - relative y-axis location of the center of the board in the image (default 0.5)
--synthetic_object - type of synthetic object: aruco or charuco or chessboard (default charuco)
"""

import argparse
from enum import Enum
import numpy as np
from numpy import linalg as LA
import json
import itertools
import glob
import os
import cv2 as cv
import pandas as pd
import datetime
import matplotlib.pyplot as plt
from iteration_utilities import deepflatten

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


def get_max_error(accuracy, type_dist):
    if type_dist is TypeNorm.l1 or TypeNorm.l2 or TypeNorm.l3 or TypeNorm.l_inf:
        return accuracy
    if type_dist is TypeNorm.intersection_over_union:
        return 1
    raise TypeError("this TypeNorm isn't supported")


def get_synthetic_rt(yaw, pitch, distance):
    rvec = np.zeros((3, 1), np.float64)
    tvec = np.zeros((3, 1), np.float64)

    rotPitch = np.array([[-pitch], [0], [0]])
    rotYaw = np.array([[0], [yaw], [0]])

    rvec, tvec = cv.composeRT(rotPitch, np.zeros((3, 1), np.float64),
                              rotYaw, np.zeros((3, 1), np.float64))[:2]

    tvec = np.array([[0], [0], [distance]])
    return rvec, tvec


def get_coord(num_rows, num_cols, start_x=0, start_y=0):
    i, j = np.ogrid[:num_rows, :num_cols]
    v = np.empty((num_rows, num_cols, 2), dtype=np.float32)
    v[..., 0] = j + start_y
    v[..., 1] = i + start_x
    v.shape = (1, -1, 2)
    return v


class TransformObject:
    def __init__(self):
        self.name = "none"

    def transform_image(self, image):
        return image

    def transform_points(self, points):
        return points


class PerspectiveTransform(TransformObject):
    def __init__(self, *, img_size, yaw, pitch, distance=1.0):
        self.yaw = yaw
        self.pitch = pitch
        self.distance = distance
        self.name = "perspective"

        rvec, tvec = get_synthetic_rt(yaw, pitch, distance)
        camera_matrix = np.zeros((3, 3), dtype=np.float64)
        camera_matrix[0, 0] = img_size[1]
        camera_matrix[1, 1] = img_size[0]
        camera_matrix[0, 2] = img_size[1] / 2
        camera_matrix[1, 2] = img_size[0] / 2
        camera_matrix[2, 2] = 1.
        obj_points = np.array([
            [0, 0, 0],
            [1, 0, 0],
            [1, 1, 0],
            [0, 1, 0],
        ], np.float32)
        original_corners = np.array([
            [0, 0],
            [img_size[1], 0],
            [img_size[1], img_size[0]],
            [0, img_size[0]],
        ], np.float32)
        obj_points[:, :-1] -= 0.5
        obj_points[:, -1:] = 0.

        corners, _ = cv.projectPoints(obj_points, rvec, tvec, camera_matrix, np.zeros((5, 1), dtype=np.float64))
        self.transformation = cv.getPerspectiveTransform(original_corners, corners)

    def transform_image(self, image):
        border_value = 0
        aux = cv.warpPerspective(image, self.transformation, (image.shape[1], image.shape[0]), None,
                                 cv.INTER_CUBIC, cv.BORDER_CONSTANT, border_value)
        assert (image.shape == aux.shape)
        return aux

    def transform_points(self, points):
        points = np.array(points, dtype=np.float32).reshape(-1, 1, 2)
        points = cv.perspectiveTransform(points, self.transformation)
        return points.reshape(-1, 2)


class RotateTransform(TransformObject):
    def __init__(self, *, angle, rel_center=(0.5, 0.5)):
        self.angle = angle
        self.rel_center = rel_center
        self.rot_mat = None
        self.name = ""

    def transform_image(self, image):
        self.rot_mat = cv.getRotationMatrix2D(
            [self.rel_center[0] * image.shape[1], self.rel_center[1] * image.shape[0]],
            self.angle, 1.0)
        warp_rotate_dst = cv.warpAffine(image, self.rot_mat, (image.shape[1], image.shape[0]), flags=cv.INTER_CUBIC)
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


class BlurTransform(TransformObject):
    def __init__(self, *, ksize=(5, 5)):
        self.ksize = ksize
        self.name = "blur"

    def transform_image(self, image):
        return cv.blur(image, self.ksize)

    def transform_points(self, points):
        return super().transform_points(points)


class GaussNoiseTransform(TransformObject):
    def __init__(self):
        self.name = "gaussNoise"

    def transform_image(self, image):
        row, col = image.shape
        mean = 0
        sigma = 16
        gauss = np.random.normal(mean, sigma, (row, col))
        gauss = gauss.reshape(row, col)
        return np.clip(image + gauss, 0, 255).astype(np.uint8)

    def transform_points(self, points):
        return super().transform_points(points)


class PastingTransform(TransformObject):
    def __init__(self, *, rel_center=(0.5, 0.5), background_object):
        self.rel_center = rel_center
        assert background_object.image is not None
        self.background_image = np.copy(background_object.image)
        self.row_offset = 0
        self.col_offset = 0
        self.name = ""

    def transform_image(self, image):
        self.row_offset = int(self.background_image.shape[0] * self.rel_center[0] - image.shape[0] / 2)
        self.col_offset = int(self.background_image.shape[1] * self.rel_center[1] - image.shape[1] / 2)
        background_image = np.copy(self.background_image)
        background_image[self.row_offset:self.row_offset + image.shape[0],
                         self.col_offset:self.col_offset + image.shape[1]] = image
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
        self.distCoeffs[0] = -0.65012997
        self.distCoeffs[1] = -0.650116057
        self.name = "undistorted"

    def transform_image(self, image):
        undistorted_img = cv.fisheye.undistortImage(image, K=self.cameraMatrix, D=self.distCoeffs,
                                                    Knew=self.cameraMatrix)
        return undistorted_img

    def transform_points(self, points):
        points = np.array(points)
        assert len(points.shape) == 2
        if points.shape[1] == 3:
            points = points[:, :-1]
        points = cv.fisheye.undistortPoints(points.reshape(1, -1, 2), K=self.cameraMatrix, D=self.distCoeffs, R=None,
                                            P=self.cameraMatrix)[0]
        return points


class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return json.JSONEncoder.default(self, obj)


class SyntheticObject:
    def __init__(self):
        self.image = None
        self.fields = None
        self.history = []

    def transform_object(self, transform_object):
        return self

    def show(self, wait_key=0):
        pass


class BackGroundObject(SyntheticObject):
    def __init__(self, *, num_rows, num_cols, color=0):
        self.image = np.zeros((num_rows, num_cols), dtype=np.uint8) + color

    def show(self, wait_key=0):
        cv.imshow("BackGroundObject", self.image)
        cv.waitKey(wait_key)


class SyntheticAruco(SyntheticObject):
    def __get_size(self, cell_img_size):
        board_image_size = [0, 0]
        pix = cell_img_size / (1. + self.marker_separation)
        for i in range(0, 2):
            board_image_size[i] = max(0, cell_img_size * (self.board_size[i] - 2))
            board_image_size[i] += pix * (1 + self.marker_separation / 2) * min(2, self.board_size[i])
            board_image_size[i] = round(board_image_size[i])
        return board_image_size, pix

    def __init__(self, *, board_size, cell_img_size, marker_separation=0.5, dict_id=0):
        self.board_size = board_size
        self.marker_separation = marker_separation
        self.dict_id = dict_id
        self.dict = cv.aruco.getPredefinedDictionary(dict_id)
        self.grid_board = cv.aruco.GridBoard(board_size, 1., marker_separation, self.dict)
        board_image_size, pix = self.__get_size(cell_img_size)
        self.image = self.grid_board.generateImage(board_image_size)
        self.aruco_corners = np.round((np.array(self.grid_board.getObjPoints(),
                                                dtype=np.float32) * pix).reshape(-1, 3)[:, :-1])
        self.aruco_ids = np.array(self.grid_board.getIds())
        self.fields = {"board_size": None, "marker_separation": None, "dict_id": None, "aruco_corners": None,
                       "aruco_ids": None}
        self.history = []
        background = BackGroundObject(num_rows=int(self.image.shape[0] + cell_img_size),
                                      num_cols=int(self.image.shape[1] + cell_img_size), color=255)
        pasting_object = PastingTransform(background_object=background)
        self.transform_object(pasting_object)

    def transform_object(self, transform_object):
        self.image = transform_object.transform_image(self.image)
        self.aruco_corners = np.array(transform_object.transform_points(self.aruco_corners), dtype=np.float32)
        if transform_object.name != "":
            self.history.append(transform_object.name)
        return self

    def show(self, wait_key=0):
        assert self.image is not None
        image = np.copy(self.image)
        aruco = np.array(self.aruco_corners.reshape(-1, 1, 4, 2), dtype=np.float32)
        aruco = [el for el in aruco]

        cv.aruco.drawDetectedMarkers(image, aruco)
        cv.imshow("SyntheticAruco", image)
        cv.waitKey(wait_key)

    def write(self, path="test", filename="test"):
        for name in self.fields:
            self.fields[name] = getattr(self, name)
        with open(path + "/" + filename + '.json', 'w') as fp:
            json.dump(self.fields, fp, cls=NumpyEncoder)
        cv.imwrite(path + "/" + filename + ".png", self.image)

    def read(self, path="test", filename="test"):
        with open(path + "/" + filename + ".json", 'r') as fp:
            data_loaded = json.load(fp)
            for name, value in data_loaded.items():
                setattr(self, name, value)
            self.dict = cv.aruco.getPredefinedDictionary(self.dict_id)
            self.aruco_ids = np.asarray(self.aruco_ids)
            self.aruco_corners = np.asarray(self.aruco_corners)
            self.history = []
        self.image = cv.imread(path + "/" + filename + ".png", cv.IMREAD_GRAYSCALE)


def set_plt():
    # Turn interactive plotting off
    plt.ioff()
    plt.rcParams["figure.figsize"] = (15, 9)
    plt.rcParams["figure.subplot.bottom"] = 0.3
    plt.rcParams["figure.subplot.left"] = 0.05
    plt.rcParams["figure.subplot.right"] = 0.99


def get_and_print_category_statistic(obj_type, category, statistics, accuracy, path):
    objs = np.array(list(deepflatten(statistics)))
    detected = objs[objs < accuracy]
    category_statistic = {"category": category, "detected " + obj_type: len(detected)/max(1, len(objs)),
                          "total detected " + obj_type: len(detected), "total " + obj_type: len(objs),
                          "average error " + obj_type: np.mean(detected)}
    data_frame = pd.DataFrame(objs)
    data_frame.hist(bins=500)
    plt.title(category + ' ' + obj_type)
    plt.xlabel('error')
    plt.xticks(np.arange(0., float(accuracy)+.25, .25))
    plt.ylabel('frequency')
    # plt.show()
    plt.savefig(path + '/' + category + '_' + obj_type + '.jpg')
    plt.close()
    return category_statistic


def get_time():
    return datetime.datetime.now().strftime("%Y_%m_%d-%I_%M_%S_%p")


def print_statistics(distances, accuracy, output_path, per_image_statistic):
    filename = get_time()
    with open(output_path + "/" + filename + '.json', 'w') as fp:
        json.dump(distances, fp, cls=NumpyEncoder)
    output_dict = output_path + '/' + filename
    os.mkdir(output_dict)
    result = []
    set_plt()
    for obj_type, statistics in distances.items():
        for category, image_names, category_statistics in zip(statistics[0], statistics[1], statistics[2]):
            result.append(get_and_print_category_statistic(obj_type, category, category_statistics, accuracy, output_dict))
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
        result.append(get_and_print_category_statistic(obj_type, 'all', statistics[2], accuracy, output_dict))
    data_frame = pd.DataFrame(result).groupby('category', as_index=False, sort=False).last()
    print(data_frame.to_string(index=False))


def read_distances(filename):
    distances = {}
    with open(filename, 'r') as fp:
        distances = json.load(fp)
    return distances


class Checker:
    def __init__(self, accuracy, type_dist, path):
        self.accuracy = accuracy
        self.type_dist = type_dist
        self.path = path

    @staticmethod
    def get_obj_names():
        return ["empty"]


class ArucoChecker(Checker):
    def __init__(self, accuracy, type_dist, path, read_params=True):
        super().__init__(accuracy, type_dist, path)
        self.aruco_params = cv.aruco.DetectorParameters()
        ArucoChecker.update_aruco_params(self.aruco_params, path, read_params)

    @staticmethod
    def update_aruco_params(aruco_params, path, read_params):
        if read_params and os.path.isfile(path+"/aruco_params.yml"):
            fs_read = cv.FileStorage(path+"/aruco_params.yml", cv.FileStorage_READ)
            aruco_params.readDetectorParameters(fs_read.root())
            fs_read.release()
        else:
            fs_write = cv.FileStorage(path+"/aruco_params.yml", cv.FileStorage_WRITE)
            aruco_params.writeDetectorParameters(fs_write)
            fs_write.release()

    @staticmethod
    def get_obj_names():
        return ["aruco"]

    @staticmethod
    def check_aruco(synthetic_aruco, marker_corners, marker_ids, accuracy, type_dist):
        gold = {}
        gold_corners, gold_ids = synthetic_aruco.aruco_corners.reshape(-1, 4, 2), \
            synthetic_aruco.aruco_ids
        for marker_id, marker in zip(gold_ids, gold_corners):
            gold[int(marker_id)] = marker
        detected = {}
        if marker_ids is not None and len(marker_ids) > 0:
            for marker_id, marker in zip(marker_ids, marker_corners):
                detected[int(marker_id)] = marker.reshape(4, 2)
        max_error = get_max_error(accuracy, type_dist)
        distances = np.full(len(gold_ids), max_error)
        for i, gold_id in enumerate(gold_ids):
            gold_corner = gold_corners[int(gold_id)]
            if int(gold_id) in detected:
                corner = detected[int(gold_id)]
                distances[i] = min(max_error, get_norm(gold_corner, corner, type_dist))
        return distances

    def detect_and_check(self, synthetic_aruco):
        aruco_detector = cv.aruco.ArucoDetector(synthetic_aruco.grid_board.getDictionary(), self.aruco_params)
        marker_corners, marker_ids, _ = aruco_detector.detectMarkers(synthetic_aruco.image)
        ar_dist = ArucoChecker.check_aruco(synthetic_aruco, marker_corners, marker_ids, self.accuracy, self.type_dist)
        return {self.get_obj_names()[0]: ar_dist}


class SyntheticCharuco(SyntheticObject):
    def __init__(self, *, board_size, cell_img_size, square_marker_length_rate=0.5, dict_id=0):
        self.board_size = board_size
        self.square_marker_length_rate = square_marker_length_rate
        self.dict_id = dict_id
        self.dict = cv.aruco.getPredefinedDictionary(dict_id)
        self.charuco_board = cv.aruco.CharucoBoard(board_size, 1., 1. * square_marker_length_rate, self.dict)
        board_image_size = [board_size[0] * cell_img_size, board_size[1] * cell_img_size]
        self.image = self.charuco_board.generateImage(board_image_size)
        self.aruco_corners = np.round((np.array(self.charuco_board.getObjPoints(), dtype=np.float32)
                                       * cell_img_size).reshape(-1, 3)[:, :-1])
        self.aruco_ids = np.array(self.charuco_board.getIds())
        self.chessboard_corners = (np.array(self.charuco_board.getChessboardCorners(), dtype=np.float32)
                                   * cell_img_size)[:, :-1]
        self.fields = {"board_size": None, "square_marker_length_rate": None, "dict_id": None, "aruco_corners": None,
                       "aruco_ids": None, "chessboard_corners": None}
        self.history = []
        background = BackGroundObject(num_rows=int(self.image.shape[0] + cell_img_size),
                                      num_cols=int(self.image.shape[1] + cell_img_size), color=255)
        pasting_object = PastingTransform(background_object=background)
        self.transform_object(pasting_object)

    def transform_object(self, transform_object):
        self.image = transform_object.transform_image(self.image)
        self.aruco_corners = np.array(transform_object.transform_points(self.aruco_corners), dtype=np.float32)
        self.chessboard_corners = np.array(transform_object.transform_points(self.chessboard_corners), dtype=np.float32)
        if transform_object.name != "":
            self.history.append(transform_object.name)
        return self

    def show(self, wait_key=0):
        assert self.image is not None
        image = np.copy(self.image)
        aruco = np.array(self.aruco_corners.reshape(-1, 1, 4, 2), dtype=np.float32)
        aruco = [el for el in aruco]

        cv.aruco.drawDetectedMarkers(image, aruco)
        chessboard_corners = self.chessboard_corners.reshape(-1, 1, 2)
        cv.aruco.drawDetectedCornersCharuco(image, chessboard_corners)
        cv.imshow("SyntheticCharuco", image)
        cv.waitKey(wait_key)

    def write(self, path="test", filename="test"):
        for name in self.fields:
            self.fields[name] = getattr(self, name)
        with open(path + "/" + filename + '.json', 'w') as fp:
            json.dump(self.fields, fp, cls=NumpyEncoder)
        cv.imwrite(path + "/" + filename + ".png", self.image)

    def read(self, path="test", filename="test"):
        with open(path + "/" + filename + ".json", 'r') as fp:
            data_loaded = json.load(fp)
            for name, value in data_loaded.items():
                setattr(self, name, value)
            self.dict = cv.aruco.getPredefinedDictionary(self.dict_id)
            self.charuco_board = cv.aruco.CharucoBoard(self.board_size, 1., 1. * self.square_marker_length_rate,
                                                       self.dict)
            self.aruco_ids = np.asarray(self.aruco_ids)
            self.aruco_corners = np.asarray(self.aruco_corners)
            self.chessboard_corners = np.asarray(self.chessboard_corners)
            self.history = []
        self.image = cv.imread(path + "/" + filename + ".png", cv.IMREAD_GRAYSCALE)


class CharucoChecker(Checker):
    def __init__(self, accuracy, type_dist, path="", read_params=True):
        super().__init__(accuracy, type_dist, path)
        self.aruco_params = cv.aruco.DetectorParameters()
        ArucoChecker.update_aruco_params(self.aruco_params, path, read_params)

    @staticmethod
    def get_obj_names():
        return ["aruco", "chessboard"]

    def _check_charuco(self, synthetic_charuco, charuco_corners, charuco_ids):
        gold = {}
        gold_corners = synthetic_charuco.chessboard_corners.reshape(-1, 2)
        for charuco_id, charuco_corner in zip(range(len(gold_corners)), gold_corners):
            gold[charuco_id] = charuco_corner
        detected = {}
        if charuco_ids is not None and len(charuco_ids) > 0:
            for charuco_id, charuco_corner in zip(charuco_ids, charuco_corners):
                detected[int(charuco_id)] = charuco_corner
        max_error = get_max_error(self.accuracy, self.type_dist)
        total_count = len(gold_corners)
        distances = np.full(total_count, max_error)
        for i, gold_id in enumerate(range(total_count)):
            gold_corner = gold_corners[int(gold_id)]
            if int(gold_id) in detected:
                corner = detected[int(gold_id)]
                distances[i] = min(max_error, get_norm(gold_corner, corner, self.type_dist))
        return distances

    def detect_and_check(self, synthetic_charuco):
        charuco_detector = cv.aruco.CharucoDetector(synthetic_charuco.charuco_board)
        charuco_detector.setDetectorParameters(self.aruco_params)
        charuco_corners, charuco_ids, marker_corners, marker_ids = charuco_detector.detectBoard(synthetic_charuco.image)
        ar_dist = ArucoChecker.check_aruco(synthetic_charuco, marker_corners, marker_ids, self.accuracy, self.type_dist)

        ch_dist = self._check_charuco(synthetic_charuco, charuco_corners, charuco_ids)
        return {self.get_obj_names()[0]: ar_dist, self.get_obj_names()[1]: ch_dist}


class SyntheticChessboard(SyntheticObject):
    def __init__(self, *, board_size, cell_img_size):
        self.board_size = board_size
        board_image_size = [board_size[0] * cell_img_size, board_size[1] * cell_img_size]
        self.image = ((np.indices((board_size[1], board_size[0])).sum(axis=0) % 2) * 255).astype(dtype=np.uint8)
        self.image = cv.resize(self.image, board_image_size, interpolation=cv.INTER_NEAREST)
        # TODO: board_size means (x,y) for ChArUco and means (points_per_row,points_per_colum) for findChessboardCorners
        self.chessboard_corners = np.zeros(((board_size[0] - 1) * (board_size[1] - 1), 2), np.float32)
        self.chessboard_corners[:, :2] = np.mgrid[0:board_size[0] - 1, 0:board_size[1] - 1].T.reshape(-1, 2)
        self.chessboard_corners *= cell_img_size
        self.chessboard_corners += cell_img_size
        self.fields = {"board_size": None, "chessboard_corners": None}
        self.history = []
        background = BackGroundObject(num_rows=int(self.image.shape[0] + cell_img_size),
                                      num_cols=int(self.image.shape[1] + cell_img_size), color=255)
        pasting_object = PastingTransform(background_object=background)
        self.transform_object(pasting_object)

    def transform_object(self, transform_object):
        self.image = transform_object.transform_image(self.image)
        self.chessboard_corners = np.array(transform_object.transform_points(self.chessboard_corners), dtype=np.float32)
        if transform_object.name != "":
            self.history.append(transform_object.name)
        return self

    def show(self, wait_key=0):
        assert self.image is not None
        image = np.copy(self.image)
        chessboard_corners = self.chessboard_corners.reshape(-1, 1, 2)
        cv.aruco.drawDetectedCornersCharuco(image, chessboard_corners)
        cv.imshow("SyntheticChessboard", image)
        cv.waitKey(wait_key)

    def write(self, path="test", filename="test"):
        for name in self.fields:
            self.fields[name] = getattr(self, name)
        with open(path + "/" + filename + '.json', 'w') as fp:
            json.dump(self.fields, fp, cls=NumpyEncoder)
        cv.imwrite(path + "/" + filename + ".png", self.image)

    def read(self, path="test", filename="test"):
        with open(path + "/" + filename + ".json", 'r') as fp:
            data_loaded = json.load(fp)
            for name, value in data_loaded.items():
                setattr(self, name, value)
            self.chessboard_corners = np.asarray(self.chessboard_corners)
            self.history = []
        self.image = cv.imread(path + "/" + filename + ".png", cv.IMREAD_GRAYSCALE)


class ChessboardChecker(Checker):
    @staticmethod
    def get_obj_names():
        return ["chessboard"]

    def __check_chessboard(self, synthetic_chessboard, corners):
        gold = {}
        gold_corners = synthetic_chessboard.chessboard_corners.reshape(-1, 2)
        max_error = get_max_error(self.accuracy, self.type_dist)
        total_count = len(gold_corners)
        distances = np.full(total_count, max_error)
        if corners is not None:
            for i, gold_corner in enumerate(gold_corners):
                distances[i] = min(max_error, np.min([get_norm(gold_corner, corner, self.type_dist)
                                                      for corner in corners]))
        return distances

    def detect_and_check(self, synthetic_chessboard):
        criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 30, 0.001)
        gray = synthetic_chessboard.image
        # Find the chess board corners
        chessboard = (synthetic_chessboard.board_size[1] - 1, synthetic_chessboard.board_size[0] - 1)
        ret, corners = cv.findChessboardCorners(gray, chessboard, criteria)

        # If found, add object points, image points (after refining them)
        if ret is True:
            corners = cv.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
        ch_dist = self.__check_chessboard(synthetic_chessboard, corners)
        return {self.get_obj_names()[0]: ch_dist}


def generate_dataset(args, synthetic_object, background_color=0):
    output = args.dataset_path
    background = BackGroundObject(num_rows=int(synthetic_object.image.shape[0] * 2.),
                                  num_cols=int(synthetic_object.image.shape[1] * 2.), color=background_color)
    rel_center_x, rel_center_y = args.rel_center_x, args.rel_center_y
    pasting_object = PastingTransform(background_object=background, rel_center=(rel_center_y, rel_center_x))
    synthetic_object.transform_object(pasting_object)
    synthetic_object.write(output)

    empty_t = TransformObject()
    blur_t = BlurTransform()
    gauss_noise_t = GaussNoiseTransform()

    perspective_t1 = PerspectiveTransform(img_size=synthetic_object.image.shape, yaw=0., pitch=0.5)
    perspective_t2 = PerspectiveTransform(img_size=synthetic_object.image.shape, yaw=0.5, pitch=0.)
    perspective_t3 = PerspectiveTransform(img_size=synthetic_object.image.shape, yaw=0.5, pitch=0.5)
    undistort_t = UndistortFisheyeTransform(img_size=synthetic_object.image.shape)
    transforms_list = [[perspective_t1, perspective_t2, perspective_t3, empty_t],
                      [undistort_t, empty_t],
                      [blur_t, gauss_noise_t, empty_t]]
    transforms_comb = list(itertools.product(*transforms_list))

    dictionary = {}
    for transforms in transforms_comb:
        for angle in range(0, 360, 31):
            synthetic_object.read(output)
            rotate_t = RotateTransform(angle=angle, rel_center=(rel_center_x, rel_center_y))
            synthetic_object.transform_object(rotate_t)
            for transform in transforms:
                synthetic_object.transform_object(transform)
            folder = '_' + '_'.join(synthetic_object.history)
            dictionary[folder] = dictionary.get(folder, -1) + 1
            if not os.path.exists(output + "/" + folder):
                os.mkdir(output + "/" + folder)
            synthetic_object.write(output + "/" + folder, folder + '_' + str(dictionary[folder]))
        # synthetic_object.show()


def main():
    # parse command line options
    parser = argparse.ArgumentParser(description="augmentation benchmark", add_help=False)
    parser.add_argument("-H", "--help", help="show help", action="store_true", dest="show_help")
    parser.add_argument("--configuration", help="script launch configuration", default="generate_run", action="store",
                        dest="configuration", choices=['generate_run', 'generate', 'run', 'show'], type=str)
    parser.add_argument("-p", "--path", help="input/output dataset path", default=".", action="store",
                        dest="dataset_path")
    parser.add_argument("-d1", help="d1", default="", action="store", dest="d1")
    parser.add_argument("-d2", help="d2", default="", action="store", dest="d2")
    parser.add_argument("--per_image_statistic", help="print the per image statistic", action="store_true")
    parser.add_argument("-a", "--accuracy", help="input accuracy", default="10", action="store", dest="accuracy",
                        type=float)
    parser.add_argument("--marker_length_rate", help="square marker length rate for charuco", default=".5",
                        action="store", dest="marker_length_rate", type=float)
    parser.add_argument("--cell_img_size", help="the size of one board cell in the image in pixels", default="100",
                        action="store", dest="cell_img_size", type=int)
    parser.add_argument("--board_x", help="input board x size", default="6", action="store", dest="board_x", type=int)
    parser.add_argument("--board_y", help="input board y size", default="6", action="store", dest="board_y", type=int)
    parser.add_argument("--rel_center_x", help="the relative x-axis location of the center of the board in the image",
                        default=".5", action="store", dest="rel_center_x", type=float)
    parser.add_argument("--rel_center_y", help="the relative x-axis location of the center of the board in the image",
                        default=".5", action="store", dest="rel_center_y", type=float)
    parser.add_argument("--metric", help="Metric for distance between result and gold", default="l_inf", action="store",
                        dest="metric", choices=['l1', 'l2', 'l_inf', 'intersection_over_union'], type=str)
    parser.add_argument("--synthetic_object", help="type of synthetic object", default="charuco", action="store",
                        dest="synthetic_object", choices=['aruco', 'charuco', 'chessboard'], type=str)

    args = parser.parse_args()
    show_help = args.show_help
    if show_help:
        parser.print_help()
        return

    dataset_path = args.dataset_path
    if not os.path.exists(dataset_path):
        os.mkdir(dataset_path)
    accuracy = args.accuracy
    metric = TypeNorm.l_inf
    if args.metric == "l1":
        metric = TypeNorm.l1
    elif args.metric == "l2":
        metric = TypeNorm.l2
    elif args.metric == "intersection_over_union":
        metric = TypeNorm.intersection_over_union

    cell_img_size = args.cell_img_size

    if args.synthetic_object == "charuco":
        board_size = [args.board_x, args.board_y]
        synthetic_object = SyntheticCharuco(board_size=board_size, cell_img_size=cell_img_size,
                                            square_marker_length_rate=args.marker_length_rate)
        checker = CharucoChecker(accuracy, metric, dataset_path)
    elif args.synthetic_object == "aruco":
        board_size = [args.board_x, args.board_y]
        synthetic_object = SyntheticAruco(board_size=board_size, cell_img_size=cell_img_size,
                                          marker_separation=args.marker_length_rate)
        checker = ArucoChecker(accuracy, metric, dataset_path)
    elif args.synthetic_object == "chessboard":
        board_size = [args.board_x, args.board_y]
        synthetic_object = SyntheticChessboard(board_size=board_size, cell_img_size=cell_img_size)
        checker = ChessboardChecker(accuracy, metric, dataset_path)
    else:
        synthetic_object = None

    configuration = args.configuration
    if configuration == "generate" or configuration == "generate_run":
        generate_dataset(args, synthetic_object)
        if configuration == "generate":
            return
    elif configuration == "show":
        distances1 = read_distances(dataset_path+'/'+args.d1)
        distances3 = distances1
        if args.d2 != '':
            distances2 = read_distances(dataset_path + '/' + args.d2)
            for key, value in distances1.items():
                if key in distances2:
                    for i, category in enumerate(value[2]):
                        for j, image in enumerate(category):
                            for k, corner in enumerate(image):
                                distances3[key][2][i][j][k] = corner - distances2[key][2][i][j][k]
        print_statistics(distances3, accuracy, dataset_path, args.per_image_statistic)
        return

    print("distance threshold:", checker.accuracy, "\n")

    list_folders = next(os.walk(dataset_path))[1]
    error_by_categories = {}
    for folder in list_folders:
        if folder[0] != '_':
            continue
        configs = glob.glob(dataset_path + '/' + folder + '/*.json')
        distances = {}
        for name in checker.get_obj_names():
            distances[name] = [[], []]
        for config in configs:
            config_name = config.split('/')[-1].split('\\')[-1].split('.')[0]
            synthetic_object.read(dataset_path + '/' + folder, config_name)
            # synthetic_object.show()
            distance = checker.detect_and_check(synthetic_object)
            for key, value in distance.items():
                distances[key][0] += [config_name]
                distances[key][1] += [value]
        for key, value in distances.items():
            if key not in error_by_categories:
                error_by_categories[key] = [[], [], []]
            error_by_categories[key][0].append(folder)
            error_by_categories[key][1].append(value[0])
            error_by_categories[key][2].append(value[1])
    print_statistics(error_by_categories, accuracy, dataset_path, args.per_image_statistic)


if __name__ == '__main__':
    main()
