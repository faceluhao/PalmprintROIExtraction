import abc

import cv2
import numpy as np
from math import ceil, sqrt
from tools.tools import resize_keep_aspectratio
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.figure import Figure


class BaseDisplayEngine(metaclass=abc.ABCMeta):
    def __init__(self, height=1000, width=1900):
        self.images = []
        self.height = height
        self.width = width

    def add(self, image, title):
        if len(image.shape) == 2:
            image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
        self.images.append((image, title))

    def draw(self):
        pass

    def get_canvas(self):
        pass

    def clear(self):
        self.images.clear()


class OpenCV(BaseDisplayEngine):
    def __init__(self, height=1000, width=1900):
        super().__init__(height, width)
        self.canvas = np.zeros([height, width, 3], dtype=np.uint8)
        self.canvas[:, :, :] = 255

    def draw(self):
        if len(self.images) == 0:
            return
        line_num = ceil(sqrt(len(self.images)))
        row_num = ceil(len(self.images) / line_num)

        cell_height = int(self.height / row_num)
        cell_width = int(self.width / line_num)
        fontFace, fontScale, thickness = cv2.FONT_HERSHEY_SIMPLEX, cell_height / 500, 1
        for i, (image, title) in enumerate(self.images):
            text_size, baseline = cv2.getTextSize(title, fontFace, fontScale, thickness)
            text_point = (int((i % line_num + 0.5) * cell_width - text_size[0] / 2), int((i // line_num) * cell_height + text_size[1]))
            cv2.putText(self.canvas, title, text_point, fontFace, fontScale, (0, 0, 0), thickness)
            resized_image = resize_keep_aspectratio(image, (cell_height - text_size[1] * 2, cell_width), value=[255, 255, 255])
            self.canvas[((i // line_num) * cell_height + text_size[1] * 2): (i // line_num + 1) * cell_height,
            (i % line_num) * cell_width:(i % line_num + 1) * cell_width, :] = resized_image

    def get_canvas(self):
        return self.canvas

    def clear(self):
        super(OpenCV, self).clear()
        self.canvas[:, :, :] = 255


class Matplot(BaseDisplayEngine):
    def __init__(self, height=1000, width=1900):
        super().__init__(height, width)
        self.fig = Figure(figsize=(width // 100, height // 100), dpi=100, constrained_layout=True)
        self.canvas = FigureCanvasAgg(self.fig)

    def add(self, image, title):
        if len(image.shape) == 2:
            image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = image[:, :, [2, 1, 0]]
        self.images.append((image, title))

    def draw(self):
        line_num = ceil(sqrt(len(self.images)))
        row_num = ceil(len(self.images) / line_num)

        for i, (image, title) in enumerate(self.images, start=1):
            ax = self.fig.add_subplot(row_num, line_num, i)
            ax.imshow(image)
            ax.set_title(title)
        self.canvas.draw()

    def get_canvas(self):
        rgba = np.asarray(self.canvas.buffer_rgba())
        return cv2.cvtColor(rgba, cv2.COLOR_RGBA2RGB)

    def clear(self):
        super(Matplot, self).clear()
        self.fig.clear()
