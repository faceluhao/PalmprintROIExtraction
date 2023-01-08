import time

from ROI_Cutter.Base import Reader, Base_ROI_Cutter
from ROI_Cutter.Mode import Test
from tools.Logger import logger


class Runner:
    def __init__(self, reader: Reader, cutter: Base_ROI_Cutter, mode: list):
        self.reader = reader
        self.cutter = cutter
        self.mode = mode
        
        for m in mode:
            if type(m) == Test:
                self.cutter.is_test = True

    def run(self):
        for image, save_stem in self.reader.get_image():
            start_time = time.time()
            ROI = self.cutter(image)
            end_time = time.time()
            logger.info(end_time - start_time)
            for mode in self.mode:
                flag = mode(image=image, ROI=ROI, save_stem=save_stem, info=self.cutter.info, show_dict=self.cutter.show_dict)
                if not flag:
                    self.close()
                    return
        self.close()

    def close(self):
        for mode in self.mode:
            mode.close()
