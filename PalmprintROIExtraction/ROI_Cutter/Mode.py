from math import sqrt, ceil

import cv2
import numpy as np
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.figure import Figure
from sortedcontainers import SortedList

from ROI_Cutter.Base import Mode
from tools.Logger import logger
from tools.tools import timeit
from tools.DisplayEngine import OpenCV, Matplot


class Save(Mode):
    def __init__(self, ROI_size=None, save_dir=None, save_name_by_origin_name=True, save_origin=False):
        super().__init__()
        self.ROI_size = ROI_size
        self.save_dir = save_dir
        self.save_name_by_origin_name = save_name_by_origin_name
        self.save_origin = save_origin
        if not self.save_dir.is_dir():
            self.save_dir.mkdir(parents=True)
        if not self.save_name_by_origin_name:
            self.count = 1
        if self.save_origin:
            self.origin_dir = save_dir.joinpath('origin')
            if not self.origin_dir.is_dir():
                self.origin_dir.mkdir(parents=True)

    def __call__(self, *args, **kwargs):
        image = kwargs['image']
        ROI = kwargs['ROI']
        if ROI is None:
            return True
        if self.save_name_by_origin_name:
            stem = kwargs['save_stem']
        else:
            stem = self.count
            self.count += 1
        if len(ROI) == 1 and not self.save_origin:
            for save_dir in ROI:
                if ROI[save_dir] is None:
                    continue
                if type(ROI[save_dir]) is not list:
                    cv2.imencode('.jpg', ROI[save_dir])[1].tofile(str(self.save_dir.joinpath(f'{stem}.jpg')))
                    logger.info(f"save: {self.save_dir.joinpath(f'{stem}.jpg')}")
                else:
                    for i in range(len(ROI[save_dir])):
                        if ROI[save_dir][i] is None:
                            continue
                        cv2.imencode('.jpg', ROI[save_dir][i])[1].tofile(str(self.save_dir.joinpath(f'{stem}_{i + 1}.jpg')))
                        logger.info(f"save: {self.save_dir.joinpath(save_dir, f'{stem}_{i + 1}.jpg')}")
        else:
            for save_dir in ROI:
                if not self.save_dir.joinpath(save_dir).is_dir():
                    self.save_dir.joinpath(save_dir).mkdir(parents=True)
                    logger.info(f"mkdir: {self.save_dir.joinpath(save_dir)}")
            for save_dir in ROI:
                if ROI[save_dir] is None:
                    continue
                if self.save_origin:
                    cv2.imencode('.jpg', image)[1].tofile(str(self.origin_dir.joinpath(f'{stem}.jpg')))
                    logger.info(f"save: {self.origin_dir.joinpath(f'{stem}.jpg')}")
                if type(ROI[save_dir]) is not list:
                    cv2.imencode('.jpg', ROI[save_dir])[1].tofile(str(self.save_dir.joinpath(save_dir, f'{stem}.jpg')))
                    logger.info(f"save: {self.save_dir.joinpath(save_dir, f'{stem}.jpg')}")
                else:
                    for i in range(len(ROI[save_dir])):
                        if ROI[save_dir][i] is None:
                            continue
                        cv2.imencode('.jpg', ROI[save_dir][i])[1].tofile(str(self.save_dir.joinpath(save_dir, f'{stem}_{i + 1}.jpg')))
                        logger.info(f"save: {self.save_dir.joinpath(save_dir, f'{stem}_{i + 1}.jpg')}")
        return True


class SaveTopK(Mode):
    def __init__(self, save_dir=None, k=20, is_show=False, video_save_path=None):
        self.save_dir = save_dir
        self.k = k
        self.is_show = is_show
        self.ROIs = SortedList()
        self.video_save_path = video_save_path
        if self.video_save_path is not None:
            self.writer = cv2.VideoWriter(
                str(video_save_path), cv2.VideoWriter_fourcc(*"mp4v"), 30, (1900, 1000))

        if is_show or self.video_save_path is not None:
            self.shower = OpenCV(height=1000, width=1900)

    @timeit
    def __call__(self, *args, **kwargs):
        ROI = kwargs['ROI']
        if ROI is not None:
            image = ROI['palmROI']
            score = cv2.Laplacian(image, cv2.CV_64F).var()
            if len(self.ROIs) < self.k:
                self.ROIs.add([score, image])
            elif len(self.ROIs) == self.k and self.ROIs[0][0] < score:
                del self.ROIs[0]
                self.ROIs.add([score, image])

        if self.is_show or self.video_save_path is not None:
            self.shower.clear()
            self.shower.add(kwargs['image'], 'image')
            for i, ROI in enumerate(self.ROIs, start=1):
                self.shower.add(ROI[1], str(i))

            self.shower.draw()
            canvas = self.shower.get_canvas()
            if self.video_save_path is not None:
                self.writer.write(canvas)
            if self.is_show:
                cv2.imshow("test", canvas)
                k = cv2.waitKey(1)
                if k == ord('e'):
                    return False
        return True

    def close(self):
        if self.save_dir is not None:
            for i, item in enumerate(self.ROIs.__reversed__(), start=1):
                cv2.imencode('.jpg', item[1])[1].tofile(str(self.save_dir.joinpath(f'{i}.jpg')))
                logger.info(f"save: {self.save_dir.joinpath(f'{i}.jpg')}")


class Test(Mode):
    def __init__(self, video_save_path=None, image_save_path=None, is_show=True):
        super().__init__()
        self.video_save_path = video_save_path
        self.image_save_path = image_save_path
        self.is_show = is_show
        if self.video_save_path is not None:
            self.writer = cv2.VideoWriter(
                str(video_save_path), cv2.VideoWriter_fourcc(*"mp4v"), 30, (1700, 1000))

        self.shower = OpenCV(height=1000, width=1700)

    def __call__(self, *args, **kwargs):
        show_dict = kwargs['show_dict']
        self.shower.clear()

        for key in show_dict:
            if show_dict[key] is None:
                continue
            self.shower.add(show_dict[key], key)
        self.shower.draw()
        canvas = self.shower.get_canvas()

        if self.video_save_path is not None:
            self.writer.write(canvas)
        if self.is_show:
            cv2.imshow("test", canvas)
            k = cv2.waitKey(1)
            if k == ord('e'):
                return False
        if self.image_save_path is not None:
            for key in show_dict:
                if show_dict[key] is None:
                    continue
                cv2.imencode('.jpg', show_dict[key])[1].tofile(
                    str(self.image_save_path.joinpath(f"{kwargs['save_stem']}_{key}.jpg")))
            cv2.imencode('.jpg', canvas)[1].tofile(
                str(self.image_save_path.joinpath(f"{kwargs['save_stem']}.jpg")))
            logger.info(f"""save: {self.image_save_path.joinpath(f"{kwargs['save_stem']}.jpg")}""")
        return True

    def close(self):
        if self.is_show:
            cv2.destroyAllWindows()
        if self.video_save_path is not None:
            self.writer.release()
