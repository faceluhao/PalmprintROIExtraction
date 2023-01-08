import abc


class Reader(metaclass=abc.ABCMeta):
    def get_image(self):
        pass


class Base_ROI_Cutter(metaclass=abc.ABCMeta):
    def __init__(self):
        self.info = {}
        self.show_dict = None
        self.is_test = False

    def reset(self):
        self.info = {}
        self.init_dict()

    def init_dict(self):
        pass

    def __call__(self, image):
        pass


class Mode(metaclass=abc.ABCMeta):
    def __call__(self, *args, **kwargs):
        pass

    def close(self):
        pass
