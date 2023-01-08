import cv2
import numpy as np
import paddle
from PIL import Image
from paddleseg import utils
from paddleseg.core import infer
from paddleseg.cvlibs import Config

from tools.tools import timeit

paddle.set_device('gpu')
cfg = Config('palmSegmentation/configs/my/ocrnet_128×128.yml')
transforms = cfg.val_dataset.transforms

model_hand = cfg.model
utils.utils.load_entire_model(model_hand, 'palmSegmentation/models/hand-ocrnet-128.pdparams')
model_hand.eval()


@timeit
def predict_hand(image):
    """
    对输入图像进行预测，输出手部语义分割图
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
            model_hand,
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


def masking(image, mask):
    """
    结合原图像与mask，分割出手部图像
    Args:
        image:
        mask:

    Returns:

    """
    mat = np.array(mask)
    mat = mat.astype(np.uint8)
    dst = Image.fromarray(mat, 'P')
    bin_colormap = [0, 0, 0] + [255, 255, 255] * 254  # 二值调色板
    dst.putpalette(bin_colormap)
    label = np.asarray(dst)
    # 将image的相素值和mask像素值相加得到结果
    masked = cv2.add(image, np.zeros(np.shape(image), dtype=np.uint8), mask=label)
    return masked
