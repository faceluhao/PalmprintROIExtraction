import collections

from ROI_Cutter.Base import Base_ROI_Cutter
from ROI_Cutter.Process import GetPoints, Check, RotateHandByMask, RotateImageByPoints, CutHand, RotateAfterTangent, \
    ProcessBeforePredictHand, PredictHand, \
    GetHandWidth, CutPalmROI, CutFingerROI, GetTangentByPalmMask

k = [0.4, 0.6]  # 测量手掌宽度的范围
k2 = [0.34, 0.66]  # 第一个参数代表ROI中心y坐标与关键点中点y坐标距离与手掌宽度的比值，第二个参数代表ROI宽度与手掌宽度的比值


class ROI_Cutter(Base_ROI_Cutter):
    def __init__(self, hand_size=512, hand_model='segformer_B2', use_handpose=False,
                 cut_finger=False, strict=False, finger_angle_threshold=None,
                 static_image_mode=True, min_detection_confidence=0.75, min_tracking_confidence=0.75):
        if finger_angle_threshold is None:
            finger_angle_threshold = [5, 5, 5]
        elif type(finger_angle_threshold) == int:
            finger_angle_threshold = [finger_angle_threshold] * 3
        super().__init__()
        self.use_handpose = use_handpose
        self.strict = strict
        self.finger_angle_threshold = finger_angle_threshold
        self.k = k
        self.k2 = k2

        if use_handpose:
            self.processes = [
                GetPoints(static_image_mode=static_image_mode,
                          min_detection_confidence=min_detection_confidence,
                          min_tracking_confidence=min_tracking_confidence),
                Check(),
                RotateImageByPoints(),
                CutHand(),
                ProcessBeforePredictHand(hand_size=hand_size, use_handpose=True),
                PredictHand(hand_size=hand_size, hand_model=hand_model),
                GetTangentByPalmMask(use_handpose=True),
                RotateAfterTangent(use_handpose=True),
                GetHandWidth(),
                CutPalmROI(),
            ]
        else:
            self.processes = [
                ProcessBeforePredictHand(hand_size=hand_size),
                PredictHand(hand_size=hand_size, hand_model=hand_model),
                RotateHandByMask(),
                GetTangentByPalmMask(),
                RotateAfterTangent(),
                GetHandWidth(),
                CutPalmROI(),
            ]

        if cut_finger:
            self.processes.append(CutFingerROI())

    def init_dict(self):
        # 初始化
        self.show_dict = collections.OrderedDict()
        self.show_dict['origin_image'] = None

        if self.use_handpose:
            self.show_dict['rotated_image'] = None
            self.show_dict['hand_image'] = None
        else:
            self.show_dict['expanded_image'] = None
        self.show_dict['resized_mask'] = None
        self.show_dict['rotated_resized_mask'] = None
        self.show_dict['rotated_origin_image'] = None
        self.show_dict['palmROI'] = None
        
    def set_show_dict_image(self, key, image):
        if self.is_test:
            self.show_dict[key] = image.copy()

    def __call__(self, image):
        self.reset()
        self.origin_image = image
        
        self.set_show_dict_image('origin_image', image)
        
        for process in self.processes:
            if not process(self):
                return None
        return self.ROI
