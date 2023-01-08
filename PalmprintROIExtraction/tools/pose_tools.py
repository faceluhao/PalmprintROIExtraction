import math

import cv2
import mediapipe as mp
import imgaug.augmenters as iaa
from imgaug.augmentables import KeypointsOnImage

from tools.tools import timeit

mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=1,
    min_detection_confidence=0.75,
    min_tracking_confidence=0.75)


@timeit
def get_points(image):
    points = []
    left_or_right = None
    origin_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results = hands.process(origin_image)
    if results.multi_hand_landmarks:
        points = landmark_to_point(results.multi_hand_landmarks)
        height, width = image.shape[:2]
        points = [[point[0] * width, point[1] * height] for point in points]
        left_or_right = results.multi_handedness[0].classification[0].label
    return points, left_or_right


def landmark_to_point(landmarks):
    return [
        [landmarks[0].landmark[i].x, landmarks[0].landmark[i].y]
        for i in range(len(landmarks[0].landmark))
    ]


@timeit
def rotate_with_points(image, points, rotate):
    kps = KeypointsOnImage.from_xy_array(points, shape=image.shape)
    seq = iaa.Sequential([
        iaa.Rotate(rotate=-rotate, fit_output=True),
    ])
    image_aug, kps_aug = seq(image=image, keypoints=kps)
    return image_aug, kps_aug.to_xy_array()


@timeit
def resize_by_height_with_points(image, points, height):
    kps = KeypointsOnImage.from_xy_array(points, shape=image.shape)
    seq = iaa.Sequential([
        iaa.Resize({"height": height, "width": "keep-aspect-ratio"})
    ])
    image_aug, kps_aug = seq(image=image, keypoints=kps)
    return image_aug, kps_aug.to_xy_array()


@timeit
def PadToSquare_with_points(image, points):
    kps = KeypointsOnImage.from_xy_array(points, shape=image.shape)
    seq = iaa.Sequential([
        iaa.PadToSquare(position="center")
    ])
    image_aug, kps_aug = seq(image=image, keypoints=kps)
    return image_aug, kps_aug.to_xy_array()


@timeit
def centercrop_with_points(image, points, width, height):
    kps = KeypointsOnImage.from_xy_array(points, shape=image.shape)
    seq = iaa.Sequential([
        iaa.CropToFixedSize(width=width, height=height, position="center"),
    ])
    image_aug, kps_aug = seq(image=image, keypoints=kps)
    return image_aug, kps_aug.to_xy_array()


# @timeit
# def Crop_with_points(image, points, up, right, bottom, left):
#     kps = KeypointsOnImage.from_xy_array(points, shape=image.shape)
#     seq = iaa.Sequential([
#         iaa.Crop(px=(up, image.shape[1] - right, image.shape[0] - bottom, left), keep_size=False),
#         # up, right, bottom, left
#     ])
#     image_aug, kps_aug = seq(image=image, keypoints=kps)
#     return image_aug, kps_aug.to_xy_array()

@timeit
def Crop_with_points(image, points, left, right, up, bottom):
    croped_image = image[int(left):int(right), int(up):int(bottom)]
    croped_points = [[point[0] - int(up), point[1] - int(left)] for point in points]
    return croped_image, croped_points

@timeit
def draw_points(image, points, size=37):
    kps = KeypointsOnImage.from_xy_array(points, shape=image.shape)
    return kps.draw_on_image(image, size=size)


is_open = [[5, 6], [9, 10], [13, 14], [17, 18]]
must_exist = [0, 5, 6, 9, 10, 13, 14, 17, 18]


def is_exist(points, image_shape):
    if len(points) == 0:
        return False
    return not any(
        not 0 < points[index][0] < image_shape[1]
        or not 0 < points[index][1] < image_shape[0]
        for index in must_exist
    )


def angle(v1, v2):
    angle1 = math.atan2(v1[1], v1[0])
    angle1 = int(angle1 * 180 / math.pi)
    angle2 = math.atan2(v2[1], v2[0])
    angle2 = int(angle2 * 180 / math.pi)
    if angle1 * angle2 >= 0:
        included_angle = abs(angle1 - angle2)
    else:
        included_angle = abs(angle1) + abs(angle2)
        if included_angle > 180:
            included_angle = 360 - included_angle
    return included_angle


def is_obtuse_angle(point1, point2, point3):
    x = (point2[0] - point1[0], point2[1] - point1[1])
    y = (point2[0] - point3[0], point2[1] - point3[1])
    # x = np.array(point2 - point1)
    # y = np.array(point2 - point3)
    return angle(x, y) > 90


def is_straight(points):
    return is_obtuse_angle(points[0], points[1], points[2]) and is_obtuse_angle(points[1], points[2], points[3])


def get_angle_between_thumb_forefinger(points):
    return angle((points[4][0] - points[0][0], points[4][1] - points[0][1]),
                 (points[5][0] - points[0][0], points[5][1] - points[0][1]))


def get_angle_between_index_middle(points):
    return angle((points[is_open[0][0]][0] - points[is_open[0][1]][0],
                  points[is_open[0][0]][1] - points[is_open[0][1]][1]), (
                     points[is_open[0 + 1][0]][0] - points[is_open[0 + 1][1]][0],
                     points[is_open[0 + 1][0]][1] - points[is_open[0 + 1][1]][1]))


def get_angle_between_middle_ring(points):
    return angle((points[is_open[1][0]][0] - points[is_open[1][1]][0],
                  points[is_open[1][0]][1] - points[is_open[1][1]][1]), (
                     points[is_open[1 + 1][0]][0] - points[is_open[1 + 1][1]][0],
                     points[is_open[1 + 1][0]][1] - points[is_open[1 + 1][1]][1]))


def get_angle_between_ring_little(points):
    return angle((points[is_open[2][0]][0] - points[is_open[2][1]][0],
                  points[is_open[2][0]][1] - points[is_open[2][1]][1]), (
                     points[is_open[2 + 1][0]][0] - points[is_open[2 + 1][1]][0],
                     points[is_open[2 + 1][0]][1] - points[is_open[2 + 1][1]][1]))


def thumb_is_straight(points):
    angel = get_angle_between_thumb_forefinger(points)
    return angel > 20


def index_finger_is_straight(points):
    return is_straight(points[5:9])


def middle_finger_is_straight(points):
    return is_straight(points[9:13])


def ring_finger_is_straight(points):
    return is_straight(points[13:17])


def little_finger_is_straight(points):
    return is_straight(points[17:21])


def get_fingers_angel(points):
    return (
        [
            angle(
                (
                    points[is_open[i][0]][0] - points[is_open[i][1]][0],
                    points[is_open[i][0]][1] - points[is_open[i][1]][1],
                ),
                (
                    points[is_open[i + 1][0]][0]
                    - points[is_open[i + 1][1]][0],
                    points[is_open[i + 1][0]][1]
                    - points[is_open[i + 1][1]][1],
                ),
            )
            for i in range(len(is_open) - 1)
        ]
        if thumb_is_straight(points)
        and index_finger_is_straight(points)
        and middle_finger_is_straight(points)
        and ring_finger_is_straight(points)
        and little_finger_is_straight(points)
        else [0, 0, 0]
    )


if __name__ == '__main__':
    import numpy as np

    image = cv2.imdecode(
        np.fromfile(r'E:\Dataset\SemanticSegmentation\北京交通大学数据库\1-100-002\1-100\001F\001_F_L1.jpg', np.uint8),
        cv2.IMREAD_COLOR)
    height, width = image.shape[:2]
    image = cv2.resize(image, (int(width / 5), int(height / 5)))
    points = get_points(image)
    #
    # drew_image = draw_points(image, points)
    # cv2.imshow('drew_image', drew_image)
    # cv2.waitKey(0)
    #
    # rotated_image, rotated_points = Crop_with_points(image, points, up=100, right=image.shape[1], bottom=500, left=0)
    # drew_image = draw_points(rotated_image, rotated_points)
    # cv2.imshow('drew_image2', drew_image)
    # cv2.waitKey(0)

    # drew_image = draw_points(image, points)
    # cv2.imshow('drew_image', drew_image)
    # cv2.waitKey(0)
    #
    rotated_image, rotated_points = rotate_with_points(image, points, 20)
    drew_image = draw_points(rotated_image, rotated_points)
    cv2.imshow('drew_image', drew_image)
    cv2.waitKey(0)
    #
    # resized_image, resized_points = resize_by_height_with_points(image, points, round(height/5))
    # drew_image = draw_points(resized_image, resized_points)
    # cv2.imshow('drew_image', drew_image)
    # cv2.waitKey(0)
    #
    # resized_image, resized_points = PadToSquare_with_points(image, points)
    # drew_image = draw_points(resized_image, resized_points)
    # cv2.imshow('drew_image', drew_image)
    # cv2.waitKey(0)
    #
    # resized_image, resized_points = centercrop_with_points(resized_image, resized_points, round(width/5), round(height/5))
    # drew_image = draw_points(resized_image, resized_points)
    # cv2.imshow('drew_image', drew_image)
    # cv2.waitKey(0)
