from math import sin, ceil

import cv2
import mediapipe as mp
import numpy as np

from tools.Logger import logger
from tools.gestures_tools import is_palm
from tools.inference import Paddle_Seg
from tools.inference_l_or_r import Predictor
from tools.pose_tools import landmark_to_point, get_fingers_angel, is_exist, PadToSquare_with_points, Crop_with_points, \
    rotate_with_points, resize_by_height_with_points, draw_points
from tools.tools import get_angel, bwboundaries, resize_keep_aspectratio, extremum, Distance, paddle_tangent, cut_image, \
    rotate


class GetPoints:
    def __init__(self,
                 static_image_mode=False,
                 max_num_hands=1,
                 min_detection_confidence=0.75,
                 min_tracking_confidence=0.75):
        self.hands = mp.solutions.hands.Hands(
            static_image_mode=static_image_mode,
            max_num_hands=max_num_hands,
            min_detection_confidence=min_detection_confidence,
            min_tracking_confidence=min_tracking_confidence)

    """
    通过手部姿态估计预测手部21个关键点的坐标
    输入:
        origin_image: 原图像
    输出:
        left_or_right: 左右手信息
        origin_points: 原始图像对应的21个关键点坐标
        percent_points: 原始图像对应的21个关键点的百分比坐标
    """

    def __call__(self, cutter):
        cutter.set_show_dict_image('origin_image', cutter.origin_image)
        origin_image = cv2.cvtColor(cutter.origin_image, cv2.COLOR_BGR2RGB)
        results = self.hands.process(origin_image)
        if not results.multi_hand_landmarks:
            logger.warning("未找到手")
            return False
        points = landmark_to_point(results.multi_hand_landmarks)
        cutter.percent_points = points
        height, width = origin_image.shape[:2]
        points = [[point[0] * width, point[1] * height] for point in points]
        left_or_right = results.multi_handedness[0].classification[0].label
        if cutter.is_test:
            cutter.show_dict['origin_image'] = draw_points(cutter.show_dict['origin_image'], points,
                                                           size=origin_image.shape[0] // 50)
            cv2.arrowedLine(cutter.show_dict['origin_image'], (ceil(points[0][0]), ceil(points[0][1])),
                            (ceil(points[9][0]), ceil(points[9][1])), (0, 0, 255), 2, 3, 0, 0.1)  # 画箭头
            cutter.info['hand_landmarks'] = results.multi_hand_landmarks[0]
        cutter.origin_points = points
        cutter.left_or_right = left_or_right
        return True


class Check:
    """
        检查是否存在手、是否手心面向摄像头、手掌是否完全张开
        输入:
            origin_image
            origin_points
        输出:
            无
    """

    def __call__(self, cutter):
        image = cutter.origin_image
        points = cutter.origin_points
        if not is_exist(points, image.shape):
            cutter.info['state'] = '未检测到手掌'
            logger.warning('未检测到手掌')
            return False
        if cutter.strict:
            if not is_palm(points, cutter.left_or_right):
                cutter.info['state'] = '请将手心对准摄像头'
                logger.warning('请将手心对准摄像头')
                return False
            fingers_angel = get_fingers_angel(points)
            logger.debug(f'fingers_angel: {str(fingers_angel)} threshold: {cutter.finger_angle_threshold}')
            if not (np.asarray(cutter.finger_angle_threshold) <= np.asarray(fingers_angel)).all():
                cutter.info['state'] = '手掌未完全张开'
                logger.warning('手掌未完全张开')
                return False
        return True


class RotateImageByPoints:
    """
        根据获得的手部关键点信息，旋转图像
        输入:
            origin_image
            origin_points
        输出:
            rotated_image
            rotated_points
    """

    def __call__(self, cutter):
        points = cutter.origin_points
        cutter.rotated_image, cutter.rotated_points = rotate_with_points(cutter.origin_image, cutter.origin_points,
                                                                         180 - get_angel(points[0], points[9]))
        cutter.rotated_pointed_image, _ = rotate_with_points(cutter.show_dict['origin_image'], cutter.origin_points,
                                                                         180 - get_angel(points[0], points[9]))
        cutter.set_show_dict_image('rotated_image', cutter.rotated_pointed_image)
        return True


class RotateHandByMask:
    """
        根据mask，计算最大内切圆和最大外接圆，以最小内切圆到最大外接圆的方向为垂直向上，调整image和mask方向
        输入:
            resized_image
            resized_mask
        输出:
            rotated_image
            rotated_points
    """

    def __call__(self, cutter):
        expanded_image = cutter.expanded_image
        image = cutter.resized_image
        mask = cutter.resized_mask
        dist_map = cv2.distanceTransform(mask, cv2.DIST_L2, cv2.DIST_MASK_PRECISE)
        _, radius, _, center = cv2.minMaxLoc(dist_map)

        # 外接圆
        contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)[-2:]
        center2, radius2 = cv2.minEnclosingCircle(np.concatenate(contours, 0))
        # 内切圆
        if cutter.is_test:
            center_radius = cutter.show_dict['resized_mask'].shape[0] // 50
            cv2.circle(cutter.show_dict['resized_mask'], tuple(center), center_radius, (0, 0, 255), 2, cv2.LINE_8, 0)
            cv2.circle(cutter.show_dict['resized_mask'], tuple(center), int(radius), (0, 0, 255), 2, cv2.LINE_8, 0)
            cv2.circle(cutter.show_dict['resized_mask'], (int(center2[0]), int(center2[1])), center_radius,
                       (0, 255, 0,), 2)
            cv2.circle(cutter.show_dict['resized_mask'], (int(center2[0]), int(center2[1])), int(radius2), (0, 255, 0,),
                       2)

        angel = -get_angel(center2, center)

        cutter.resized_mask = rotate(mask, angel)
        cutter.resized_image = rotate(image, angel)
        cutter.expanded_image = rotate(expanded_image, angel)
        return True


class RotateImageByL_or_R:
    def __init__(self, l_or_r_model='models/l_or_r'):
        self.l_or_r_infer = Predictor(model_folder_dir=l_or_r_model)

    def __call__(self, cutter):
        origin_image = cutter.origin_image
        L_or_R = self.l_or_r_infer.infer(origin_image)

        if L_or_R == 0:
            origin_image = rotate(origin_image, 90)
        else:
            origin_image = rotate(origin_image, 270)

        cutter.origin_image = origin_image
        return True


class CutHand:
    """
        在旋转过后的图像中裁减出手掌图像
        输入:
            rotated_image
            rotated_points
        输出:
            hand_image
            hand_points
    """

    def __call__(self, cutter):
        image = cutter.rotated_image
        points = cutter.rotated_points
        left = image.shape[1]
        right = 0
        up = image.shape[0]
        bottom = 0
        for point in points:
            if point[1] < left:
                left = point[1]
            if point[1] > right:
                right = point[1]
            if point[0] < up:
                up = point[0]
            if point[0] > bottom:
                bottom = point[0]
        len_width = right - left
        len_height = bottom - up
        ratio = 0.05
        target_left = max(left - ratio * len_width, 0)
        target_right = min(right + ratio * len_width, image.shape[0])
        target_up = max(up - ratio * len_height, 0)
        target_bottom = min(bottom + ratio * len_height, image.shape[1])

        image, points = Crop_with_points(image, points,
                                         int(target_left), int(target_right),
                                         int(target_up), int(target_bottom))
        cutter.hand_image, cutter.hand_points = PadToSquare_with_points(image, points)
        show_hand_image, show_points = Crop_with_points(cutter.rotated_pointed_image, cutter.hand_points,
                                         int(target_left), int(target_right),
                                         int(target_up), int(target_bottom))
        show_hand_image, _ = PadToSquare_with_points(show_hand_image, show_points)
        cutter.set_show_dict_image('hand_image', show_hand_image)
        return True


class GetFingerSeam:
    """
        在手掌图像中裁减指缝区域
        输入:
            hand_image
            hand_points
        输出:
            masked_images: 指缝图像
            angular_points: 指缝图像原点在原图像中的坐标，以便之后映射到原图上
    """

    def __init__(self, finger_size):
        self.finger_size = finger_size

    def __call__(self, cutter):
        cutter.finger_size = self.finger_size
        image = cutter.hand_image
        points = cutter.hand_points
        finger_seam_list = [[5, 6, 10, 9], [13, 14, 18, 17]]
        masked_images = []
        angular_points = []

        height, width = image.shape[:2]
        channels = image.shape[2]

        for point_list in finger_seam_list:
            mask = np.zeros(image.shape, dtype=np.uint8)
            roi_corners = np.array([np.hstack((np.array([points[point_list][:, 0]]).T,
                                               np.array([points[point_list][:, 1]]).T))], dtype=np.int32)
            x_min = max(0, min(points[point_list][:, 0]))
            x_max = min(width - 1, max(points[point_list][:, 0]))
            y_min = max(0, min(points[point_list][:, 1]))
            y_max = min(height - 1, max(points[point_list][:, 1]))

            channel_count = channels
            ignore_mask_color = (255,) * channel_count
            # 创建mask层
            cv2.fillPoly(mask, roi_corners, ignore_mask_color)
            # 为每个像素进行与操作，除mask区域外，全为0
            masked_image = cv2.bitwise_and(image, mask)
            masked_image = masked_image[round(y_min): round(y_max + 1), round(x_min): round(x_max + 1)]
            masked_images.append(masked_image)
            angular_points.append([round(y_min), round(y_max + 1), round(x_min), round(x_max + 1)])
        cutter.masked_images = masked_images
        cutter.angular_points = angular_points
        return True


class GetTangentByFingerSeam:
    """
        输入:
            masked_images
            angular_points
        输出:
            finger_seam_points1
            finger_seam_points2
    """

    def __init__(self, finger_size, finger_model):
        model_folder_dir = f'models/finger/{finger_size}/{finger_model}'
        self.finger_inferencer = Paddle_Seg(model_folder_dir=model_folder_dir,
                                            infer_img_size=finger_size,
                                            use_tensorrt=False)

    def __call__(self, cutter):
        if not self.get_finger_seam_points_by_finger_mask(cutter, True):
            logger.warning('未找到指谷点1')
            return False
        if not self.get_finger_seam_points_by_finger_mask(cutter, False):
            logger.warning('未找到指谷点2')
            return False
        if not self.get_tangent(cutter):
            logger.warning('寻找切点失败')
            return False
        return True

    def get_finger_seam_points_by_finger_mask(self, cutter, flag):
        if flag:
            finger_seam_image = cutter.masked_images[0]
            angular_points = cutter.angular_points[0]
            cutter.set_show_dict_image('finger_seam_image1', finger_seam_image)
        else:
            finger_seam_image = cutter.masked_images[1]
            angular_points = cutter.angular_points[1]
            cutter.set_show_dict_image('finger_seam_image2', finger_seam_image)

        height, width = finger_seam_image.shape[:2]

        temp_image = resize_keep_aspectratio(finger_seam_image, (cutter.finger_size, cutter.finger_size))

        mask = self.finger_inferencer.infer(temp_image)
        if flag:
            cutter.set_show_dict_image('finger_seam_mask1', mask)
        else:
            cutter.set_show_dict_image('finger_seam_mask2', mask)

        ret, thresh = cv2.threshold(mask, 127, 255, 0)
        contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)

        # 找到最大联通区域
        target_contour = contours[0]
        for contour in contours:
            if contour.shape[0] > target_contour.shape[0]:
                target_contour = contour
        point_list = target_contour.reshape(-1, 2)
        point_list[:, [0, 1]] = point_list[:, [1, 0]]

        hull = cv2.convexHull(target_contour, returnPoints=False)
        defects = cv2.convexityDefects(target_contour, hull)
        max_distance_index = 0
        max_distance = defects[0, 0, 3]
        for i in range(defects.shape[0]):
            if defects[i, 0, 3] > max_distance:
                max_distance = defects[i, 0, 3]
                max_distance_index = i

        min_index = defects[max_distance_index, 0, 2]

        if cutter.is_test:
            if flag:
                img = cutter.show_dict['finger_seam_mask1']
            else:
                img = cutter.show_dict['finger_seam_mask2']
            for i in range(defects.shape[0]):
                s, e, f, d = defects[i, 0]
                start = tuple(target_contour[s][0])
                end = tuple(target_contour[e][0])
                far = tuple(target_contour[f][0])
                cv2.line(img, (start[1], start[0]), (end[1], end[0]), [0, 255, 0], 5)
                cv2.circle(img, (far[1], far[0]), 5, [255, 0, 0], -1)
            cv2.circle(img, (point_list[min_index][1], point_list[min_index][0]),
                       10, [0, 0, 255], -1)

        point_num = 100

        if flag:
            cutter.set_show_dict_image('finger_seam_mask1-pointed', mask)
            img2 = cutter.show_dict['finger_seam_mask1-pointed']
        else:
            cutter.set_show_dict_image('finger_seam_mask2-pointed', mask)
            img2 = cutter.show_dict['finger_seam_mask2-pointed']
        radius = img2.shape[0] // 100
        for point in point_list[min_index - point_num: min_index + point_num]:
            cv2.circle(img2, (point[1], point[0]), radius, (0, 0, 255), 2)

        if height > width:
            finger_seam_points = [(round(point[0] * height / cutter.finger_size + angular_points[0]),
                                   round(point[1] * height / cutter.finger_size - (height - width) / 2 + angular_points[
                                       2]))
                                  for point in point_list[min_index - point_num: min_index + point_num]]
        else:
            finger_seam_points = [
                (round(point[0] * width / cutter.finger_size - (width - height) / 2 + angular_points[0]),
                 round(point[1] * width / cutter.finger_size + angular_points[2]))
                for point in point_list[min_index - point_num: min_index + point_num]]

        if not finger_seam_points:
            logger.error('finger_seam_points 为空')
            return False

        # if cutter.is_test:
        #     radius = cutter.show_dict['hand_image'].shape[0] // 100
        #     for point in finger_seam_points:
        #         cv2.circle(cutter.show_dict['hand_image'], (point[1], point[0]), radius, (0, 0, 255), 2)

        if flag:
            self.finger_seam_points1 = finger_seam_points
        else:
            self.finger_seam_points2 = finger_seam_points

        return True

    def get_tangent(self, cutter):
        point1_tangent, point2_tangent = paddle_tangent(self.finger_seam_points1, self.finger_seam_points2)
        if cutter.left_or_right == 'Left':
            point1_tangent, point2_tangent = point2_tangent, point1_tangent
        cutter.point1_tangent = point1_tangent
        cutter.point2_tangent = point2_tangent
        return True


class GetTangentByPalmMask:
    """
        通过手掌mask的方式获得指缝切点
        输入:
            resized_mask
        输出:
            point1_tangent
            point2_tangent
    """

    def __init__(self, use_handpose=False):
        self.use_handpose = use_handpose

    def __call__(self, cutter):
        resized_mask = cutter.resized_mask
        try_num = 5
        threshold = 50

        point_list = bwboundaries(resized_mask)  # 获得二值图像的边缘点的列表，列表形式为[y, x]

        wrist_points_line = np.argwhere(point_list[:, 0] == point_list[:, 0].max())
        index_start = wrist_points_line[0][0]
        index_end = wrist_points_line[-1][0]
        wrist_middle = point_list[wrist_points_line[len(wrist_points_line) // 2][0]]

        # 求出从手腕线起点开始，绕手一周到手腕终点，所有点距离手腕终点的距离
        distances = np.sqrt(
            np.sum(np.power(np.vstack((point_list[index_start:0:-1], point_list[-1:index_end:-1])) - wrist_middle, 2),
                   1))

        # 计算极值
        for i in range(1, try_num + 1):
            maximum, minimum = extremum(distances, threshold)  # 初始阈值为50
            real_maximum_index = [index_start - item for item in maximum]
            real_minimum_index = [index_start - item for item in minimum]
            if len(real_minimum_index) in {3, 4}:  # 只有极小值数目等于3或者4才能找到目标关键点
                break
            logger.warning(f'极小值数为{len(real_minimum_index)}, 当前阈值为{threshold}, 开始第{i}次重试')
            if len(real_minimum_index) < 3:  # 极小值数目小于3，说明阈值过大，调小阈值重试
                threshold = round(3 / 4 * threshold)
            elif len(real_minimum_index) > 4:  # 极小值数目大于4，说明阈值过小，调大阈值重试
                threshold = round(5 / 4 * threshold)

        # 找出两个关键点
        if len(real_minimum_index) == 3:
            target_point1_index = real_minimum_index[0]
            target_point2_index = real_minimum_index[-1]
            if get_angel(point_list[target_point1_index], point_list[target_point2_index]) > 45:
                logger.error(
                    f'Wrong minimum point, expected angel to be less than 45. but got {get_angel(point_list[target_point1_index], point_list[target_point2_index])}')

                return False
        elif len(real_minimum_index) == 4:  # 根据第一点与第三个点的距离与第二个点与第四个点之间的距离来判断哪两个点是目标点
            if Distance(point_list[real_minimum_index[0]], point_list[real_minimum_index[2]]) > \
                    Distance(point_list[real_minimum_index[1]], point_list[real_minimum_index[3]]):
                target_point1_index = real_minimum_index[1]
                target_point2_index = real_minimum_index[3]
            else:
                target_point1_index = real_minimum_index[0]
                target_point2_index = real_minimum_index[2]
        else:
            logger.error(f'Minimum number error, expected is 3 or 4, bug got {len(real_minimum_index)}.')

            return False

        while target_point1_index < 0:
            target_point1_index += len(point_list)
        while target_point2_index < 0:
            target_point2_index += len(point_list)

        if target_point1_index < 50:
            target_point_list1 = np.vstack(
                (point_list[target_point1_index - 50:], point_list[:target_point1_index + 50]))
        elif target_point1_index > len(point_list) - 50:
            target_point_list1 = np.vstack(
                (point_list[target_point1_index - 50:], point_list[:target_point1_index + 50 - len(point_list)]))
        else:
            target_point_list1 = point_list[target_point1_index - 50:target_point1_index + 50]

        if target_point2_index < 50:
            target_point_list2 = np.vstack(
                (point_list[target_point2_index - 50:], point_list[:target_point2_index + 50]))
        elif target_point2_index > len(point_list) - 50:
            target_point_list2 = np.vstack(
                (point_list[target_point2_index - 50:], point_list[:target_point2_index + 50 - len(point_list)]))
        else:
            target_point_list2 = point_list[target_point2_index - 50:target_point2_index + 50]

        if self.use_handpose:
            ratio = cutter.hand_image.shape[0] / cutter.resized_mask.shape[0]
        else:
            ratio = cutter.expanded_image.shape[0] / cutter.resized_mask.shape[0]

        point1_tangent, point2_tangent = paddle_tangent(target_point_list1, target_point_list2)

        cutter.point1_tangent = point1_tangent * ratio
        cutter.point2_tangent = point2_tangent * ratio
        return True


class RotateAfterTangent:
    """
        通过切点，旋转图像
        输入:
            point1_tangent
            point2_tangent
            hand_image
            hand_points
            resized_mask
            resized_points
        输出:
            rotated_origin_image
            rotated_tangent_points
            rotated_origin_points
            resized_mask
            resized_points
    """

    def __init__(self, use_handpose=False):
        self.use_handpose = use_handpose

    def __call__(self, cutter):
        point1_tangent = cutter.point1_tangent
        point2_tangent = cutter.point2_tangent
        resized_mask = cutter.resized_mask
        if self.use_handpose:
            rotated_image = cutter.hand_image
            rotated_points = cutter.hand_points
            resized_points = cutter.resized_points
        else:
            rotated_image = cutter.expanded_image

        angel = get_angel(point1_tangent, point2_tangent)

        rotated_origin_image, rotated_tangent_points = rotate_with_points(rotated_image,
                                                                          [[point1_tangent[1], point1_tangent[0]],
                                                                           [point2_tangent[1], point2_tangent[0]]],
                                                                          angel)
        cutter.set_show_dict_image('rotated_origin_image', rotated_origin_image)

        if self.use_handpose:
            rotated_origin_image, rotated_origin_points = rotate_with_points(rotated_image, rotated_points, angel)
            resized_mask, resized_points = rotate_with_points(resized_mask, resized_points, angel)
        else:
            resized_mask = rotate(resized_mask, angel)

        cutter.rotated_origin_image = rotated_origin_image
        cutter.rotated_tangent_points = rotated_tangent_points
        cutter.resized_mask = resized_mask
        if self.use_handpose:
            cutter.rotated_origin_points = rotated_origin_points
            cutter.resized_points = resized_points
        return True


class ProcessBeforePredictHand:
    """
        对手掌图像做语义分割前的准备工作
        输入:
            rotated_image
            rotated_points
        输出:
            resized_image
            resized_points
    """

    def __init__(self, hand_size, use_handpose=False):
        self.hand_size = hand_size
        self.use_handpose = use_handpose

    def __call__(self, cutter):
        cutter.hand_size = self.hand_size

        if self.use_handpose:
            cutter.resized_image, cutter.resized_points = resize_by_height_with_points(cutter.hand_image,
                                                                                       cutter.hand_points,
                                                                                       cutter.hand_size)
        else:
            height, width = cutter.origin_image.shape[:2]
            higher = height if height > width else width
            cutter.expanded_image = resize_keep_aspectratio(cutter.origin_image, (higher, higher))
            cutter.resized_image = resize_keep_aspectratio(cutter.expanded_image, (cutter.hand_size, cutter.hand_size))

            cutter.set_show_dict_image('expanded_image', cutter.expanded_image)
        return True


class PredictHand:
    """
        获得手掌mask
        输入:
            resized_image
        输出:
            resized_mask
    """

    def __init__(self, hand_size, hand_model):
        model_folder_dir = f'models/hand/{hand_size}/{hand_model}'
        self.hand_inferencer = Paddle_Seg(model_folder_dir=model_folder_dir,
                                          infer_img_size=hand_size,
                                          use_tensorrt=False)

    def __call__(self, cutter):
        cutter.resized_mask = self.hand_inferencer.infer(cutter.resized_image)
        return True


class GetHandWidth:
    """
        计算手掌宽度
        输入:
            rotated_origin_image
            rotated_tangent_points
            resized_mask
        输出:
            hand_width
    """

    def __call__(self, cutter):
        rotated_origin_image = cutter.rotated_origin_image
        rotated_tangent_points = cutter.rotated_tangent_points
        # rotated_origin_image, rotated_tangent_points = PadToSquare_with_points(cutter.rotated_origin_image,
        #                                                                        cutter.rotated_tangent_points)
        _, resized_tangent_points = resize_by_height_with_points(rotated_origin_image, rotated_tangent_points,
                                                                 cutter.resized_mask.shape[0])
        rotated_tangent_points = cutter.rotated_tangent_points
        cutter.set_show_dict_image('rotated_resized_mask', cutter.resized_mask)
        cutter.set_show_dict_image('resized_mask', cutter.resized_mask)

        rotated_tangent_point1 = (round(rotated_tangent_points[0][0]), round(rotated_tangent_points[0][1]))
        rotated_tangent_point2 = (round(rotated_tangent_points[1][0]), round(rotated_tangent_points[1][1]))

        cutter.rotated_tangent_point1 = rotated_tangent_point1
        cutter.rotated_tangent_point2 = rotated_tangent_point2

        if cutter.is_test:
            radius = cutter.show_dict['rotated_origin_image'].shape[0] // 100
            thickness = cutter.show_dict['rotated_origin_image'].shape[0] // 100
            cv2.circle(cutter.show_dict['rotated_origin_image'], rotated_tangent_point1, radius, (255, 0, 0,),
                       thickness)
            cv2.circle(cutter.show_dict['rotated_origin_image'], rotated_tangent_point2, radius, (255, 0, 0,),
                       thickness)
            cv2.line(cutter.show_dict['rotated_origin_image'], rotated_tangent_point1, rotated_tangent_point2,
                     (0, 0, 255,), thickness)

        rotated_resize_point1 = (round(resized_tangent_points[0][0]), round(resized_tangent_points[0][1]))
        rotated_resize_point2 = (round(resized_tangent_points[1][0]), round(resized_tangent_points[1][1]))

        cutter.rotated_resize_point1 = rotated_resize_point1
        cutter.rotated_resize_point2 = rotated_resize_point2

        if cutter.is_test:
            radius = cutter.show_dict['rotated_resized_mask'].shape[0] // 100
            thickness = cutter.show_dict['rotated_resized_mask'].shape[0] // 100
            cv2.circle(cutter.show_dict['rotated_resized_mask'], rotated_resize_point1, radius, (255, 0, 0,), thickness)
            cv2.circle(cutter.show_dict['rotated_resized_mask'], rotated_resize_point2, radius, (255, 0, 0,), thickness)
            cv2.line(cutter.show_dict['rotated_resized_mask'], rotated_resize_point1, rotated_resize_point2,
                     (0, 0, 255,), thickness)

        rotated_resize_point1 = cutter.rotated_resize_point1
        rotated_resize_point2 = cutter.rotated_resize_point2
        resized_mask = cutter.resized_mask

        # 向下平移一段距离，计算手掌宽度
        distance_with_point1_and_point2 = rotated_resize_point2[0] - rotated_resize_point1[0]
        middle_with_point1_and_point2 = (rotated_resize_point1[0] + rotated_resize_point2[0]) / 2
        length_list = []
        edges = cv2.Canny(resized_mask, 50, 120)
        for i in range(round(distance_with_point1_and_point2 * cutter.k[0]),
                       round(distance_with_point1_and_point2 * cutter.k[1])):
            y = rotated_resize_point1[1] + i
            if y >= edges.shape[0]:
                logger.error('超出边界')
                return False
            edge_points = np.argwhere(edges[y, :] != 0)
            left_point = edge_points[edge_points < middle_with_point1_and_point2]
            if len(left_point) == 0:
                logger.error('无法找到left_point')
                return False
            left = left_point[-1]
            right_point = edge_points[edge_points > middle_with_point1_and_point2]
            if len(right_point) == 0:
                logger.error('right_point')
                return False
            right = right_point[0]
            length_list.append(right - left)
        if not length_list:
            logger.warning("length_list为空")
            return False
        cutter.hand_width = np.median(length_list)
        return True


class CutPalmROI:
    """
        切割掌纹ROI
        输入:
            rotated_origin_image
            resized_mask
            rotated_tangent_point1
            rotated_tangent_point2
            hand_with
            k2
        输出:
            ROI['palmROI']
    """

    def __call__(self, cutter):
        rotated_origin_image = cutter.rotated_origin_image
        resized_mask = cutter.resized_mask
        rotated_tangent_point1 = cutter.rotated_tangent_point1
        rotated_tangent_point2 = cutter.rotated_tangent_point2
        hand_width = cutter.hand_width
        k2 = cutter.k2

        ratio = rotated_origin_image.shape[0] / resized_mask.shape[0]

        # 切割ROI
        ROI_center_point = (
            rotated_tangent_point1[1] + hand_width * k2[0] * ratio,
            (rotated_tangent_point1[0] + rotated_tangent_point2[0]) / 2)
        ROI_length = hand_width * k2[1] * ratio

        ROI = dict()

        ROI_box = np.array([round(ROI_center_point[0] - ROI_length / 2),
                            round(ROI_center_point[0] - ROI_length / 2) + round(ROI_length),
                            round(ROI_center_point[1] - ROI_length / 2),
                            round(ROI_center_point[1] - ROI_length / 2) + round(ROI_length)])

        ROI_box_mask = np.round(ROI_box / ratio).astype(int)

        palmROI_in_resized_mask = resized_mask[ROI_box_mask[0]: ROI_box_mask[1], ROI_box_mask[2]: ROI_box_mask[3]]
        # if cutter.is_test:
        #     cv2.line(cutter.show_dict['rotated_resized_mask'], (ROI_box_mask[2], ROI_box_mask[0]),
        #              (ROI_box_mask[2], ROI_box_mask[1]), (255, 0, 0), 3)
        #     cv2.line(cutter.show_dict['rotated_resized_mask'], (ROI_box_mask[2], ROI_box_mask[0]),
        #              (ROI_box_mask[3], ROI_box_mask[0]), (255, 0, 0), 3)
        #     cv2.line(cutter.show_dict['rotated_resized_mask'], (ROI_box_mask[2], ROI_box_mask[1]),
        #              (ROI_box_mask[3], ROI_box_mask[1]), (255, 0, 0), 3)
        #     cv2.line(cutter.show_dict['rotated_resized_mask'], (ROI_box_mask[3], ROI_box_mask[0]),
        #              (ROI_box_mask[3], ROI_box_mask[1]), (255, 0, 0), 3)

        if cutter.strict and len(np.where(palmROI_in_resized_mask == 0)[0]) > palmROI_in_resized_mask.shape[0] / 5:
            logger.error('palmROI区域超出手掌mask区域')
            return False

        ROI['palmROI'] = rotated_origin_image[ROI_box[0]: ROI_box[1], ROI_box[2]: ROI_box[3]]
        cutter.set_show_dict_image('palmROI', ROI['palmROI'])
        if cutter.is_test:
            cutter.info['state'] = '检测到ROI'
        cutter.ROI = ROI
        return True


class CutFingerROI:
    """
    切割指节纹ROI。四个手指，每个手指两个
    输入:
        rotated_origin_image
        resized_mask
        rotated_origin_points
        percent_points
    输出:
        ROI['fingerROI']
    """

    def __call__(self, cutter):
        image = cutter.rotated_origin_image
        mask = cutter.resized_mask
        points = cutter.rotated_origin_points
        percent_points = cutter.percent_points

        ratio = image.shape[0] / mask.shape[0]
        resized_points = [[point[0] / ratio, point[1] / ratio] for point in points]

        finger_angle = []
        finger_angle2 = []
        finger_angle_index = [[5, 8], [9, 12], [13, 16], [17, 20]]
        for i in finger_angle_index:
            finger_angle.append((get_angel(resized_points[i[0]], resized_points[i[1]]) + 360) % 360 - 90)
            finger_angle2.append(180 - get_angel(resized_points[i[0]], resized_points[i[1]]))

        edges = cv2.Canny(mask, 50, 120)
        distances = np.zeros(8)
        finger_indexs = [6, 7, 10, 11, 14, 15, 18, 19]
        finger_points = np.zeros([8, 2])
        for i, finger_index in enumerate(finger_indexs):
            if any(x < 0 or x > 1 for x in percent_points[finger_indexs[i]]):
                continue
            edge_points = np.argwhere(edges[int(resized_points[finger_index][1]), :] != 0)
            left_point = edge_points[edge_points < resized_points[finger_index][0]]
            if len(left_point) == 0:
                logger.error('无法找到left_point')
                return False
            left = left_point[-1]
            right_point = edge_points[edge_points > resized_points[finger_index][0]]
            if len(right_point) == 0:
                logger.error('无法找到right_point')
                return False
            right = right_point[0]
            finger_points[i, 0] = (left + right) / 2
            finger_points[i, 1] = resized_points[finger_index][1]
            # distances.append((right_point - left_point) * abs(cos((180 - finger_angle[i // 2]) / 180)))
            distances[i] = ((right - left) * abs(sin(finger_angle[i // 2] * np.pi / 180)))

        finger_points = finger_points * ratio

        fingerROIs = [None] * len(finger_points)
        for i in range(len(finger_points)):
            if any(x < 0 or x > 1 for x in percent_points[finger_indexs[i]]):
                continue
            fingerROIs[i] = cut_image(image, finger_points[i], distances[i] * ratio, distances[i] * ratio,
                                      finger_angle2[i // 2])
            if cutter.is_test:
                cutter.set_show_dict_image(f'fingerROI{i // 2}-{i % 2}', fingerROIs[i])

        cutter.ROI['fingerROI'] = fingerROIs
        return True
