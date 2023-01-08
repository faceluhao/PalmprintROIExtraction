import cv2
import numpy as np
from flask import Flask, request

from ROI_Cutter.new_ROICutter import ROI_Cutter as new_ROICutter
from ROI_Cutter.old_ROICutter import ROI_Cutter as old_ROICutter

app = Flask(__name__)

oldROICutter = old_ROICutter(use_handpose=False)
newROICutter = new_ROICutter()


@app.route('/new', methods=['POST'])
def new():
    image = request.files.get('file')

    image = image.read()
    image = bytearray(image)

    image = cv2.imdecode(np.array(image, dtype='uint8'), cv2.IMREAD_UNCHANGED)  # 从二进制图片数据中读取

    ROI = newROICutter(image)['palmROI']
    if ROI is None:
        return "failed"
    ROI = np.array(cv2.imencode('.jpg', ROI)[1]).tobytes()

    return ROI


@app.route('/old', methods=['POST'])
def old():
    image = request.files.get('file')

    image = image.read()
    image = bytearray(image)

    image = cv2.imdecode(np.array(image, dtype='uint8'), cv2.IMREAD_UNCHANGED)  # 从二进制图片数据中读取

    ROI = oldROICutter(image)['palmROI']
    if ROI is None:
        return "failed"

    ROI = np.array(cv2.imencode('.jpg', ROI)[1]).tobytes()
    return ROI


if __name__ == '__main__':
    app.run(host="0.0.0.0", port=61212)
