import cv2
import paddle
from paddleseg import utils
from paddleseg.core import infer
from paddleseg.cvlibs import Config
import numpy as np

from tools.tools import timeit

paddle.set_device('gpu')
cfg = Config('palmSegmentation/configs/my/ocrnet_128×128.yml')
transforms = cfg.val_dataset.transforms

model_finger_seam = cfg.model
utils.utils.load_entire_model(model_finger_seam, 'palmSegmentation/models/finger_ocr_128.pdparams')
model_finger_seam.eval()


@timeit
def predict_finger_seam(image):
    """
    对输入图像进行预测，输出指缝语义分割图
    Args:
        image:

    Returns:

    """
    with paddle.no_grad():
        ori_shape = image.shape[:2]
        im, _ = transforms(image)
        im = im[np.newaxis, ...]
        im = paddle.to_tensor(im)

        pred = infer.inference(
            model_finger_seam,
            im,
            ori_shape=ori_shape,
            transforms=transforms.transforms)
        pred = paddle.squeeze(pred)
        pred = pred.numpy().astype('uint8')

        # save pseudo color prediction
        mask = utils.visualize.get_pseudo_color_map(pred)
        mask = cv2.cvtColor(np.asarray(mask), cv2.COLOR_BGR2RGB)
        mask = cv2.cvtColor(np.asarray(mask), cv2.COLOR_RGB2GRAY) * 255
        return mask
