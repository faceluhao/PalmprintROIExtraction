import collections

import cv2

from ROI_Cutter.Base import Base_ROI_Cutter
from ROI_Cutter.Process import GetPoints, Check, RotateImageByPoints, CutHand, RotateAfterTangent, \
    ProcessBeforePredictHand, PredictHand, \
    GetHandWidth, CutPalmROI, CutFingerROI, GetFingerSeam, GetTangentByFingerSeam

k = [0.4, 0.6]  # 测量手掌宽度的范围
k2 = [0.34, 0.66]  # 第一个参数代表ROI中心y坐标与关键点中点y坐标距离与手掌宽度的比值，第二个参数代表ROI宽度与手掌宽度的比值


class ROI_Cutter(Base_ROI_Cutter):
    def __init__(self, hand_size=512, finger_size=512, hand_model='segformer_B2',
                 finger_model='segformer_B2', strict=False, finger_angle_threshold=None,
                 static_image_mode=True,
                 min_detection_confidence=0.75,
                 min_tracking_confidence=0.75
                 ):
        if finger_angle_threshold is None:
            finger_angle_threshold = [5, 0, 5]
        elif type(finger_angle_threshold) == int:
            finger_angle_threshold = [finger_angle_threshold] * 3
        super().__init__()
        self.strict = strict
        self.finger_angle_threshold = finger_angle_threshold
        self.k = k
        self.k2 = k2

        self.processes = [
            GetPoints(static_image_mode=static_image_mode,
                 min_detection_confidence=min_detection_confidence,
                 min_tracking_confidence=min_tracking_confidence),
            Check(),
            RotateImageByPoints(),
            CutHand(),
            ProcessBeforePredictHand(hand_size=hand_size, use_handpose=True),
            PredictHand(hand_size=hand_size, hand_model=hand_model),
            GetFingerSeam(finger_size=finger_size),
            GetTangentByFingerSeam(finger_size=finger_size, finger_model=finger_model),
            RotateAfterTangent(use_handpose=True),
            GetHandWidth(),
            CutPalmROI(),
            # CutFingerROI()
        ]

    def init_dict(self):
        # 初始化
        self.show_dict = collections.OrderedDict()
        self.show_dict['origin_image'] = None
        self.show_dict['rotated_image'] = None
        self.show_dict['hand_image'] = None
        self.show_dict['finger_seam_image1'] = None
        self.show_dict['finger_seam_mask1'] = None
        self.show_dict['finger_seam_image2'] = None
        self.show_dict['finger_seam_mask2'] = None
        self.show_dict['finger_seam_mask1-pointed'] = None
        self.show_dict['finger_seam_mask2-pointed'] = None
        self.show_dict['resized_mask'] = None
        self.show_dict['rotated_resized_mask'] = None
        self.show_dict['rotated_origin_image'] = None
        self.show_dict['palmROI'] = None
        # for i in range(4):
        #     for j in range(2):
        #         self.show_dict[f'fingerROI{i}-{j}'] = None
                
    def set_show_dict_image(self, key, image):
        if self.is_test:
            temp = image.copy()
            if len(image.shape) == 2:
                temp = cv2.cvtColor(temp, cv2.COLOR_GRAY2BGR)
            self.show_dict[key] = temp

    def __call__(self, image):
        self.reset()
        self.set_show_dict_image('origin_image', image)
        self.origin_image = image
        for process in self.processes:
            if not process(self):
                return None
        return self.ROI
