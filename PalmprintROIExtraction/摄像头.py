import cv2

from ROI_Cutter import Mode
from ROI_Cutter.Reader import VideoReader
from ROI_Cutter.Runner import Runner
from ROI_Cutter.new_ROICutter import ROI_Cutter

cutter = ROI_Cutter()
if __name__ == '__main__':
    capture = cv2.VideoCapture(0)
    reader = VideoReader(capture=capture)
    mode = [
        # Mode.SaveTopK(video_save_path=r'I:\output\new-512-512.mp4')
        Mode.Test()
    ]
    runner = Runner(reader=reader, cutter=cutter, mode=mode)
    runner.run()
