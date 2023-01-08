import numpy as np
import math

from tools.tools import get_angel


def landmark_to_point(landmarks):
    return [
        [landmarks[0].landmark[i].x, landmarks[0].landmark[i].y]
        for i in range(len(landmarks[0].landmark))
    ]


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


def is_point_on_line_left(point, line_start, line_end):
    x1 = line_start[0]
    y1 = line_start[1]
    x2 = line_end[0]

    y2 = line_end[1]
    x = point[0]
    y = point[1]
    return (y1 - y2) * x + (x2 - x1) * y + x1 * y2 - x2 * y1 > 0


# def thumb_is_straight(points):
#     temp = is_point_on_line_left(points[4], points[2], points[6])
#     temp2 = is_point_on_line_left(points[20], points[2], points[6])
#     if temp and not temp2 or not temp and temp2:
#         return True
#     else:
#         return False

def thumb_is_straight(points):
    angel = angle((points[4][0] - points[0][0], points[4][1] - points[0][1]),
                  (points[5][0] - points[0][0], points[5][1] - points[0][1]))
    print(angel)
    return angel > 25


def index_finger_is_straight(points):
    return is_straight(points[5:9])


def middle_finger_is_straight(points):
    return is_straight(points[9:13])


def ring_finger_is_straight(points):
    return is_straight(points[13:17])


def little_finger_is_straight(points):
    return is_straight(points[17:21])


gestures_dict = {
    '0': 'zero',  # [0, 0, 0, 0, 0]
    '1': 'little_finger',  # [0, 0, 0, 0, 1]
    '7': 'ok',  # [0, 0, 1, 1, 1]
    '8': 'one',  # [0, 1, 0, 0, 0]
    '12': 'two',  # [0, 1, 1, 0, 0]
    '14': 'three',  # [0, 1, 1, 1, 0]
    '15': 'four',  # [0, 1, 1, 1, 1]
    '31': 'five'  # [1, 1, 1, 1, 1]
}


def angle(v1, v2):
    dx1 = v1[0]
    dy1 = v1[1]
    dx2 = v2[0]
    dy2 = v2[1]
    angle1 = math.atan2(dy1, dx1)
    angle1 = int(angle1 * 180 / math.pi)
    angle2 = math.atan2(dy2, dx2)
    angle2 = int(angle2 * 180 / math.pi)
    if angle1 * angle2 >= 0:
        included_angle = abs(angle1 - angle2)
    else:
        included_angle = abs(angle1) + abs(angle2)
        if included_angle > 180:
            included_angle = 360 - included_angle
    return included_angle


is_open = [[5, 6], [9, 10], [13, 14], [17, 18]]
k_angle = [7, 5, 7]
must_exist = [0, 5, 6, 9, 10, 13, 14, 17, 18]


def is_exist(points):
    if len(points) == 0:
        return False
    return not any(
        not 0 < points[index][0] < 1
        or not 0 < points[index][1] < 1
        for index in must_exist
    )


def is_fully_open(points):
    if (
        not thumb_is_straight(points)
        or not index_finger_is_straight(points)
        or not middle_finger_is_straight(points)
        or not ring_finger_is_straight(points)
        or not little_finger_is_straight(points)
    ):
        return False
    for i in range(len(is_open) - 1):
        print(angle((points[is_open[i][0]][0] - points[is_open[i][1]][0],
                     points[is_open[i][0]][1] - points[is_open[i][1]][1]), (
                        points[is_open[i + 1][0]][0] - points[is_open[i + 1][1]][0],
                        points[is_open[i + 1][0]][1] - points[is_open[i + 1][1]][1])))
        if angle((points[is_open[i][0]][0] - points[is_open[i][1]][0],
                  points[is_open[i][0]][1] - points[is_open[i][1]][1]), (
                         points[is_open[i + 1][0]][0] - points[is_open[i + 1][1]][0],
                         points[is_open[i + 1][0]][1] - points[is_open[i + 1][1]][1])) < k_angle[i]:
            return False
    return True


def is_palm(points, left_or_right):
    """
    判断手是否是正面
    Args:
        points:
        left_or_right:

    Returns:

    """
    angle1 = get_angel(points[0], points[9])
    angle2 = get_angel(points[0], points[4])

    if left_or_right == 'Left':
        if angle1 < 0:
            angle1 += 360
            if angle2 < 0:
                angle2 += 360
        return angle2 < angle1 < angle2 + 90
    else:
        if angle2 < 0:
            angle2 += 360
            if angle1 < 0:
                angle1 += 360
        return angle2 - 90 < angle1 < angle2


def judge_gestures(points):
    index = str(
        thumb_is_straight(points) * (2 ** 4) + index_finger_is_straight(points) * (2 ** 3) + middle_finger_is_straight(
            points) * (2 ** 2) + ring_finger_is_straight(points) * (2 ** 1) + little_finger_is_straight(points) * (
                2 ** 0))
    return gestures_dict.get(index, 'none')
