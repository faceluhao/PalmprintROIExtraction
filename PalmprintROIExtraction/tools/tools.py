import time

import numpy
import numpy as np
import math

import cv2
from math import fabs, sin, cos, radians

import paddle
# import torch
from PIL import Image, ImageDraw, ImageFont
from tools.Logger import logger


def timeit(func):
    def _warp(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        elastic_time = time.time() - start_time
        logger.debug("'%s': %.6fs" % (func.__name__, elastic_time))
        return result

    return _warp


@timeit
def get_min_enclosing_rectangle(image, mask):    # TODO: 最小矩形框越界时有bug
    """
    根据mask得到的最小外接矩形，裁剪image和mask
    Args:
        image: 图片
        mask: 掩码

    Returns:
        最小外接矩形裁剪成的图片和掩码

    """
    contours, hierarchy = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    areas = [cv2.contourArea(contours[c]) for c in range(len(contours))]

    max_id = areas.index(max(areas))
    #
    # max_rect = cv2.minAreaRect(contours[max_id])
    # max_box = cv2.boxPoints(max_rect)
    # max_box = np.int0(max_box)
    # # image = cv2.drawContours(image, [max_box], 0, (0, 255, 0), 2)  # 绘制最小外接矩形
    #
    # # max_rect[0]为中心点坐标
    # pts1 = np.float32(max_box)
    # pts2 = np.float32([[max_rect[0][0] - max_rect[1][1] / 2, max_rect[0][1] - max_rect[1][0] / 2],
    #                    [max_rect[0][0] + max_rect[1][1] / 2, max_rect[0][1] - max_rect[1][0] / 2],
    #                    [max_rect[0][0] + max_rect[1][1] / 2, max_rect[0][1] + max_rect[1][0] / 2],
    #                    [max_rect[0][0] - max_rect[1][1] / 2, max_rect[0][1] + max_rect[1][0] / 2]])
    # M = cv2.getPerspectiveTransform(pts1, pts2)
    # dst = cv2.warpPerspective(image, M, (image.shape[1], image.shape[0]))
    # dst_mask = cv2.warpPerspective(mask, M, (image.shape[1], image.shape[0]))
    #
    # pts2[pts2 < 0] = 0
    # target_image = dst[int(pts2[1][1]):int(pts2[2][1]), int(pts2[3][0]):int(pts2[2][0]), :]
    # target_mask = dst_mask[int(pts2[1][1]):int(pts2[2][1]), int(pts2[3][0]):int(pts2[2][0])]
    x, y, w, h = cv2.boundingRect(contours[max_id])
    target_image = image[y: y + h, x: x + w]
    target_mask = mask[y: y + h, x: x + w]
    return target_image, target_mask


@timeit
def adjust_direction(image, mask):
    """
    根据mask，计算最大内切圆和最大外接圆，以最小内切圆到最大外接圆的方向为垂直向上，调整image和mask方向
    Args:
        image:
        mask:

    Returns:
        调整方向后的image和mask
    """
    dist_map = cv2.distanceTransform(mask, cv2.DIST_L2, cv2.DIST_MASK_PRECISE)
    _, radius, _, center = cv2.minMaxLoc(dist_map)

    # 内切圆
    result = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
    cv2.circle(result, tuple(center), int(20), (0, 0, 255), 2, cv2.LINE_8, 0)
    cv2.circle(result, tuple(center), int(radius), (0, 0, 255), 2, cv2.LINE_8, 0)

    # 外接圆
    contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)[-2:]
    center2, radius2 = cv2.minEnclosingCircle(np.concatenate(contours, 0))
    cv2.circle(result, (int(center2[0]), int(center2[1])), int(20), (0, 255, 0,), 2)
    cv2.circle(result, (int(center2[0]), int(center2[1])), int(radius2), (0, 255, 0,), 2)

    remoted_mask = rotate(mask, -get_angel(center2, center))
    remoted_image = rotate(image, -get_angel(center2, center))

    return remoted_image, remoted_mask


def resize_keep_aspectratio(image_src, dst_size, value=[0, 0, 0]):
    """
    resize图像，且保持原图像的长款比例
    Args:
        image_src:
        dst_size:

    Returns:

    """
    src_h, src_w = image_src.shape[:2]

    dst_h, dst_w = dst_size

    # 判断应该按哪个边做等比缩放
    h = dst_w * (float(src_h) / src_w)  # 按照ｗ做等比缩放
    w = dst_h * (float(src_w) / src_h)  # 按照h做等比缩放

    h = int(h)
    w = int(w)

    if h <= dst_h:
        image_dst = cv2.resize(image_src, (dst_w, int(h)))
    else:
        image_dst = cv2.resize(image_src, (int(w), dst_h))

    h_, w_ = image_dst.shape[:2]

    top = int((dst_h - h_) / 2)
    down = int((dst_h - h_ + 1) / 2)
    left = int((dst_w - w_) / 2)
    right = int((dst_w - w_ + 1) / 2)

    borderType = cv2.BORDER_CONSTANT
    image_dst = cv2.copyMakeBorder(image_dst, top, down, left, right, borderType, None, value)

    return image_dst


# @timeit
# def bwboundaries(mask):
#     ret, thresh = cv2.threshold(mask, 127, 255, 0)
#     contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
#     points = []
#     for point in contours[0]:
#         points.append((point[0][1], point[0][0]))
#     return points

@timeit
def bwboundaries(mask):
    """
    边界追踪，返回图像中最大的边界区域
    Args:
        mask:

    Returns:

    """
    ret, thresh = cv2.threshold(mask, 127, 255, 0)
    contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)

    # 找到最大联通区域
    target_contour_index = contours[0]
    for contour in contours:
        if contour.shape[0] > target_contour_index.shape[0]:
            target_contour_index = contour
    points = target_contour_index.reshape(-1, 2)
    points[:, [0, 1]] = points[:, [1, 0]]
    # points = []
    # for point in target_contour_index:
    #     points.append((point[0][1], point[0][0]))
    return points


@timeit
def old_bwboundaries(mask):
    """
    之前使用的边界追踪算法，已废弃
    Args:
        mask:

    Returns:

    """
    img_bin = cv2.copyMakeBorder(mask, 1, 1, 1, 1, borderType=cv2.BORDER_CONSTANT, value=0)  # 在图片四周各补零一行，即增加一行黑色像素
    # 初始化起始点
    start_x = -1
    start_y = -1
    is_start_point = False  # 判断是否为起始点的标志

    # 寻找起始点
    h, w = img_bin.shape
    for i in range(h):
        for j in range(w):
            if img_bin[i, j] != 0:
                start_x = i
                start_y = j
                is_start_point = True
                break
        if is_start_point:
            break

    # 定义链码相对应的增量坐标
    neibor = [(0, 1), (-1, 1), (-1, 0), (-1, -1), (0, -1), (1, -1), (1, 0), (1, 1)]  # 邻域点
    temp = 2  # 链码值，也是neibor的索引序号，这里是从链码的2号位进行搜索
    contours = [(start_x - 1, start_y - 1)]  # 用于存储轮廓点

    # 将当前点设为轮廓的开始点
    current_x = start_x
    current_y = start_y

    # temp=2，表示从链码的2方向进行邻域检索，通过当前点和邻域点集以及链码值确定邻域点
    neibor_x = current_x + neibor[temp][0]
    neibor_y = current_y + neibor[temp][1]

    # 因为当前点的值为起始点，而终止检索的条件又是这个，所以会产生冲突，因此先寻找第二个边界点
    is_contour_point = False
    while not is_contour_point:  # 邻域点循环，当是目标像素点时跳出
        if img_bin[neibor_x, neibor_y] != 0:
            # 将符合条件的邻域点设为当前点进行下一次的边界点搜索
            current_x = neibor_x
            current_y = neibor_y
            is_contour_point = True
            contours.append([current_x - 1, current_y - 1])
            temp = (temp - 2) % len(neibor)  # 作为下一个边界点的邻域检测起始点,顺时针旋转90度
        else:
            temp = (temp + 1) % len(neibor)  # 逆时针旋转45度进行搜索
        neibor_x = current_x + neibor[temp][0]
        neibor_y = current_y + neibor[temp][1]

    # 开始第三个及以后的边界点的查找
    while not ((current_x == start_x) and (current_y == start_y)):  # 轮廓扫描循环
        is_contour_point = False
        while not is_contour_point:  # 邻域点循环，当是目标像素点时跳出
            if img_bin[neibor_x, neibor_y] != 0:  # 邻域是白点时，即为边界
                # 将符合条件的邻域点设为当前点进行下一次的边界点搜索
                current_x = neibor_x
                current_y = neibor_y
                is_contour_point = True  # 将判断是否为边界点的标签设置为1，用于跳出循环
                contours.append([current_x - 1, current_y - 1])
                temp = (temp - 2) % len(neibor)  # 作为下一个边界点的邻域检测起始点,顺时针旋转90度
            else:
                temp = (temp + 1) % len(neibor)  # 逆时针旋转45度进行搜索
            neibor_x = current_x + neibor[temp][0]
            neibor_y = current_y + neibor[temp][1]
    return contours[:-1]


def get_angel(point1, point2):
    """
    Args:
        point1:
        point2:

    Returns:
        两点连成的直线与水平线的夹角
    """
    y1, x1 = point1
    y2, x2 = point2
    return math.atan2(y2 - y1, x2 - x1) * 180 / math.pi


def cut_image(image, center, length, width, theta):
    """
    theta最好在[-90, 90]
    """
    center = [int(center[0]), int(center[1])]
    length = int(length)
    width = int(width)
    box = cv2.boxPoints((center, [length, width], theta))
    to_box = cv2.boxPoints(([int(length/2), int(width/2)], [length, width], 0))
    p1 = np.float32(box)
    p2 = np.float32(to_box)
    M = cv2.getPerspectiveTransform(p1,p2)
    dst = cv2.warpPerspective(image, M, (length, width))
    return dst


@timeit
def tuple_list_to_array(tuple_list):
    return [
        [tuple_list[i][0], tuple_list[i][1]] for i in range(tuple_list.size[0])
    ]


def Distance(point1, point2):
    return math.sqrt((point1[0] - point2[0]) ** 2 + (point1[1] - point2[1]) ** 2)


@timeit
def extremum(distances, threshold):
    # from scipy import signal
    # maximum = []
    # # maximum = signal.find_peaks(np.array(distances), distance=threshold * 2)[0]
    # minimum = signal.find_peaks(-1 * np.array(distances), distance=threshold * 2)[0]
    maximum = []
    minimum = []

    # it = iter(range(threshold, len(distances) - threshold))
    # for index in it:
    #     # if distances[index] == max(distances[index - threshold:index + threshold]):
    #     #     maximum.append(index)
    #     #     for i in range(threshold):
    #     #         next(it)
    #     if distances[index] == min(distances[index - threshold:index + threshold]):
    #         minimum.append(index)
    #         if index + threshold >= len(distances) - threshold:
    #             break
    #         for i in range(threshold):
    #             next(it)
    i = threshold
    while i < len(distances) - threshold:
        if distances[i] == min(distances[i - threshold:i + threshold]):
            minimum.append(i)
            i += threshold
        else:
            i += 1
    # from scipy import signal
    # maximum = []
    # minimum = signal.argrelextrema(np.array(distances), np.less_equal, order=threshold * 2)[0]
    # real_minimum = []
    # for i in range(len(minimum)):
    #     if i < len(minimum) - 1 and minimum[i + 1] - minimum[i] < threshold:
    #         continue
    #     elif threshold < minimum[i] < len(distances) - threshold:
    #         real_minimum.append(minimum[i])
    # minimum = real_minimum
    return maximum, minimum


# 需要半分钟左右, 已废弃
# @timeit
# def tangent(outline1, outline2):
#     for point1 in outline1:
#         for point2 in outline2:
#             if point1[1] == point2[1]:
#                 continue
#             k = (point1[0] - point2[0]) / (point1[1] - point2[1])
#
#             count = 0  # 这个变量的作用是允许一两个点在切线上
#             flag = True
#             for temp_point1 in outline1:
#                 if point2[0] - temp_point1[0] < k * (point2[1] - temp_point1[1]):
#                     count += 1
#                     if count > 1:
#                         flag = False
#                         break
#
#             if not flag:
#                 continue
#
#             count = 0  # 这个变量的作用是允许一两个点在切线上
#             flag = True
#             for temp_point2 in outline2:
#                 if point2[0] - temp_point2[0] < k * (point2[1] - temp_point2[1]):
#                     count += 1
#                     if count > 1:
#                         flag = False
#                         break
#
#             if not flag:
#                 continue
#
#             return point1, point2
#     raise AssertionError('没有求出切线')


# 稳定0.001s
@timeit
def paddle_tangent(list1, list2):  # shape: 2 * n, 上是y, 下是x
    device = paddle.set_device('gpu:0')
    array1 = np.asarray(list1).transpose(1, 0)
    array2 = np.asarray(list2).transpose(1, 0)
    shape1 = array1.shape[1]
    shape2 = array2.shape[1]
    # t1, t2 = torch.from_numpy(array1), torch.from_numpy(array2)
    t1 = paddle.tensor.to_tensor(array1, dtype=paddle.float64, place=device)
    t2 = paddle.tensor.to_tensor(array2, dtype=paddle.float64, place=device)
    all_point = paddle.reshape(paddle.concat((t1, t2), 1), (2, 1, 1, -1))  # 2 * 1 * 1 * (n+m)
    # t1 = paddle.reshape(t1, (2, shape1, 1))
    # t2 = paddle.reshape(t2, (2, 1, shape2))
    t1 = paddle.tensor.broadcast_to(paddle.reshape(t1, (2, shape1, 1)), (2, shape1, shape2))
    t2 = paddle.tensor.broadcast_to(paddle.reshape(t2, (2, 1, shape2)), (2, shape1, shape2))
    # t1 = t1.reshape(2, shape1, 1).expand(2, shape1, shape2)
    # t2 = t2.reshape(2, 1, shape2).expand(2, shape1, shape2)

    sub = t2 - t1
    k = sub[0] / sub[1]  # n * m
    sub_all = all_point - t1.unsqueeze(3)  # 2 * n * m * (n+m)
    result = sub_all[0] - sub_all[1] * k.unsqueeze(2)  # n * m * (n+m)
    # error_point_sum = paddle.sum(result > 0, 2)  # n * m
    error_point_sum = paddle.to_tensor(result > 0, dtype='int32', place=device)
    error_point_sum = paddle.sum(error_point_sum, 2)
    min_error_point = paddle.argmin(paddle.flatten(error_point_sum, 0))
    index = min_error_point.numpy()[0]
    return list1[index // shape2], list2[index % shape2]


# @timeit  # 稳定0.001s
# def torch_tangent(list1, list2):  # shape: 2 * n, 上是y, 下是x
#     device = torch.device('cuda:0')
#     array1 = np.asarray(list1).transpose(1, 0)
#     array2 = np.asarray(list2).transpose(1, 0)
#     shape1 = array1.shape[1]
#     shape2 = array2.shape[1]
#     # t1, t2 = torch.from_numpy(array1), torch.from_numpy(array2)
#     t1 = torch.tensor(array1, dtype=torch.float, device=device)
#     t2 = torch.tensor(array2, dtype=torch.float, device=device)
#     all_point = torch.cat((t1, t2), 1).reshape(2, 1, 1, -1)  # 2 * 1 * 1 * (n+m)
#     t1 = t1.reshape(2, shape1, 1).repeat(1, 1, shape2)
#     t2 = t2.reshape(2, 1, shape2).repeat(1, shape1, 1)
#     # t1 = t1.reshape(2, shape1, 1).expand(2, shape1, shape2)
#     # t2 = t2.reshape(2, 1, shape2).expand(2, shape1, shape2)
#     sub = t2 - t1
#     k = sub[0] / sub[1]  # n * m
#     sub_all = all_point - t1.unsqueeze(3)  # 2 * n * m * (n+m)
#     result = sub_all[0] - sub_all[1] * k.unsqueeze(2)  # n * m * (n+m)
#     error_point_sum = torch.sum(result > 0, 2)  # n * m
#     min_error_point = torch.min(error_point_sum.view(-1), 0)
#     index = min_error_point[1].item()
#     return list1[index // shape2], list2[index % shape2]


def rotate(image, degree):
    """
    将image逆时针旋转degree
    Args:
        image:
        degree:

    Returns:

    """
    height, width = image.shape[:2]
    heightNew = int(width * fabs(sin(radians(degree))) + height * fabs(cos(radians(degree))))
    widthNew = int(height * fabs(sin(radians(degree))) + width * fabs(cos(radians(degree))))

    matRotation = cv2.getRotationMatrix2D((width / 2, height / 2), degree, 1)

    matRotation[0, 2] += (widthNew - width) / 2
    matRotation[1, 2] += (heightNew - height) / 2
    rotated_image = cv2.warpAffine(image, matRotation, (widthNew, heightNew), borderValue=(0, 0, 0))
    return rotated_image


def cartesian_coordinate_to_polar_coordinate(target_point, origin_point):
    distance = Distance(target_point, origin_point)
    angel = get_angel(origin_point, target_point)
    return distance, angel


def polar_coordinates_to_cartesian_coordinate(distance, angel, origin_point):
    x = origin_point[0] + cos(angel / 180 * math.pi) * distance
    y = origin_point[1] + sin(angel / 180 * math.pi) * distance
    return round(x), round(y)


def cv2ImgAddText(img, text, left, top, textColor=(0, 255, 0), textSize=100):
    if isinstance(img, numpy.ndarray):  # 判断是否OpenCV图片类型
        img = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    # 创建一个可以在给定图像上绘图的对象
    draw = ImageDraw.Draw(img)
    # 字体的格式
    fontStyle = ImageFont.truetype("font/simsun.ttc", textSize, encoding="utf-8")
    # 绘制文本
    draw.text((left, top), text, textColor, font=fontStyle)
    # 转换回OpenCV格式
    return cv2.cvtColor(numpy.asarray(img), cv2.COLOR_RGB2BGR)


def get_finger_seam(points, image, image_mask=None):
    """
    切割输入image的指缝图像
    Args:
        points:
        image:
        image_mask:

    Returns:

    """
    finger_seam_list = [[5, 6, 10, 9], [9, 10, 14, 13], [13, 14, 18, 17]]
    masked_images = []
    masked_image_masks = []
    angular_points = []

    height, width = image.shape[:2]
    channels = image.shape[2]

    for point_list in finger_seam_list:
        mask = np.zeros(image.shape, dtype=np.uint8)
        roi_corners = np.array(
            [[[points[point_list[0]][0], points[point_list[0]][1]],
              [points[point_list[1]][0], points[point_list[1]][1]],
              [points[point_list[2]][0], points[point_list[2]][1]],
              [points[point_list[3]][0], points[point_list[3]][1]]]], dtype=np.int32)

        x_min = max(0, min([points[point_list[0]][0],
                            points[point_list[1]][0],
                            points[point_list[2]][0],
                            points[point_list[3]][0]]))
        x_max = min(width - 1, max([points[point_list[0]][0],
                                    points[point_list[1]][0],
                                    points[point_list[2]][0],
                                    points[point_list[3]][0]]))

        y_min = max(0, min([points[point_list[0]][1],
                            points[point_list[1]][1],
                            points[point_list[2]][1],
                            points[point_list[3]][1]]))
        y_max = min(height - 1, max([points[point_list[0]][1],
                                     points[point_list[1]][1],
                                     points[point_list[2]][1],
                                     points[point_list[3]][1]]))

        channel_count = channels
        ignore_mask_color = (255,) * channel_count
        # 创建mask层
        cv2.fillPoly(mask, roi_corners, ignore_mask_color)
        # 为每个像素进行与操作，除mask区域外，全为0
        masked_image = cv2.bitwise_and(image, mask)
        masked_image = masked_image[round(y_min): round(y_max + 1), round(x_min): round(x_max + 1)]
        masked_images.append(masked_image)

        angular_points.append([round(y_min), round(y_max + 1), round(x_min), round(x_max + 1)])

        if image_mask is not None:
            masked_image_mask = cv2.bitwise_and(image_mask, mask)
            masked_image_mask = masked_image_mask[round(y_min): round(y_max + 1), round(x_min): round(x_max + 1)]
            masked_image_masks.append(masked_image_mask)

    return masked_images, masked_image_masks, angular_points


def get_min_point_list(mask):  # TODO: 某些情况下无法正确工作
    """
    使用二分法寻找指缝mask的最低点邻域
    Args:
        mask:

    Returns:

    """

    k = 70

    edge = cv2.Canny(mask, 50, 120)
    counts = np.sum(edge != 0, 1)

    height, width = mask.shape[:2]

    up = height
    down = 0

    target_height = (up + down) // 2

    while True:
        if up == down or up == down + 1:
            target_height = down
            break
        elif counts[target_height] >= 4:
            down = target_height
            target_height = (up + down) // 2
        else:
            if counts[target_height] <= 2 and counts[target_height - 1] > 2:
                target_height = target_height - 1
                break
            up = target_height
            target_height = (up + down) // 2

    result = np.argwhere(edge[target_height] != 0)

    x = result[len(result) // 2][0]

    temp = edge[target_height - k:target_height + 1, x - k:x + k]
    x_list = np.nonzero(temp)[0] + target_height - k
    y_list = np.nonzero(temp)[1] + x - k
    return list(zip(*[y_list, x_list]))


if __name__ == '__main__':

    def draw_points(image, points):
        import imgaug.augmenters as iaa
        from imgaug.augmentables import KeypointsOnImage
        kps = KeypointsOnImage.from_xy_array(points, shape=image.shape)
        return kps.draw_on_image(image, size=37)


    def rotate_points(points, center, theta):
        rotated_points = np.zeros_like(points)
        rotated_points[:, 0] = (points[:, 0] - center[0]) * cos(-theta) - (points[:, 1] - center[1]) * sin(-theta) + \
                               center[0]
        rotated_points[:, 1] = (points[:, 1] - center[1]) * sin(-theta) - (points[:, 1] - center[1]) * cos(-theta) + \
                               center[1]
        return rotated_points

    from HandPose import Mediapipe
    handpose = Mediapipe()

    image = cv2.imread('../test/001_F_R1.jpg')
    image = cv2.resize(image, [int(image.shape[1] / 2), int(image.shape[0] / 2)])
    points = np.zeros([21, 2])
    points1, _ = handpose.predict(image)
    for i, point in enumerate(points1):
        points[i, 0] = point[0] * image.shape[1]
        points[i, 1] = point[1] * image.shape[0]
    drew_image = draw_points(image, points)
    cv2.imshow('drew', drew_image)

    rotated_image = cv2.rotate(image, cv2.ROTATE_90_CLOCKWISE)
    rotated_points = rotate_points(points, [int(image.shape[1] / 2), int(image.shape[0] / 2)], 90 / 180 * math.pi)
    drew_image = draw_points(rotated_image, rotated_points)
    cv2.imshow('rotated', drew_image)
    cv2.waitKey(0)
    print(points)
