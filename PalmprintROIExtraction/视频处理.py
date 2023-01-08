from pathlib import Path

import cv2

from ROI_Cutter import Mode
from ROI_Cutter.Reader import VideoReader
from ROI_Cutter.Runner import Runner
from ROI_Cutter.new_ROICutter import ROI_Cutter

data_path = Path(r'E:\视频')
save_path = Path(r'E:\new')

cutter = ROI_Cutter()
if __name__ == '__main__':  # 81R2有错
    for video_path in data_path.rglob('*.mp4'):
        capture = cv2.VideoCapture(str(video_path))
        target_path = save_path.joinpath(video_path.stem)
        if not target_path.is_dir():
            target_path.mkdir(parents=True)
        if target_path.joinpath(video_path.name).is_file():
            continue
        reader = VideoReader(capture=capture)
        mode = [Mode.Test(video_save_path=target_path.joinpath(video_path.name), is_show=False)]
        runner = Runner(reader=reader, cutter=cutter, mode=mode)
        runner.run()
