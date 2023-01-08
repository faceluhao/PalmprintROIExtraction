from pathlib import Path

import cv2
import numpy as np

from ROI_Cutter.Base import Reader


class PhotoReader(Reader):
    def __init__(self, root_path: Path):
        self.root_path = root_path

    def get_image(self):
        for image_path in self.root_path.glob('*.jpg'):
            image = cv2.imdecode(np.fromfile(str(image_path), np.uint8), cv2.IMREAD_COLOR)
            yield image, image_path.stem
            
    def close(self):
        ...


class VideoReader(Reader):
    def __init__(self, capture, flip=False):
        self.capture = capture
        self.flip = flip

    def get_image(self):
        while True:
            ret, frame = self.capture.read()
            if ret:
                if self.flip:
                    frame = cv2.flip(frame, 1)
                yield frame, None
            else:
                self.close()
                break

    def close(self):
        self.capture.release()
