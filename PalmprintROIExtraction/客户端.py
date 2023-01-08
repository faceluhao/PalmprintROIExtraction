from pathlib import Path
import cv2
import requests
import numpy as np

url = "http://127.0.0.1:61212//new"


def getROI(image):
    files = {'file': ('image', cv2.imencode('.jpg', image)[1], 'image/jpg')}
    try:
        r = requests.post(url, files=files)
        if r.status_code != 200:
            return None
        if r.text != 'failed':
            img_np = np.frombuffer(r.content, np.uint8)  # 从byte数据读取为np.array形式
            return cv2.imdecode(img_np, cv2.COLOR_RGB2BGR)
        else:
            print(r.text)
        return None
    except:
        print('连接失败')
        return None


if __name__ == '__main__':
    image_path = Path(r'test/101_1.jpg')
    image = cv2.imread(str(image_path))
    ROI = getROI(image)
    if ROI is not None:
        cv2.imshow('ROI', ROI)
        cv2.waitKey(0)
