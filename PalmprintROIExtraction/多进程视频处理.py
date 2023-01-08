from multiprocessing import Pool
from pathlib import Path
from shutil import rmtree

import cv2
import random

from ROI_Cutter import Mode
from ROI_Cutter.Reader import VideoReader
from ROI_Cutter.Runner import Runner
from ROI_Cutter.new_ROICutter import ROI_Cutter


def worker(videos_path):
    cutter = ROI_Cutter()
    for video_path in videos_path:
        cutter.finger_angle_threshold = [20, 0, 20]
        target_path = save_path.joinpath(video_path.stem)
        if not target_path.is_dir():
            target_path.mkdir(parents=True)
        else:
            continue
        while len(list(target_path.glob('*.jpg'))) != 20:
            if cutter.finger_angle_threshold == [0, 0, 0]:
                rmtree(target_path)
                break
            capture = cv2.VideoCapture(str(video_path))
            reader = VideoReader(capture=capture)
            mode = [
                Mode.SaveTopK(save_dir=target_path)
                # Mode.Save(save_dir=target_path, save_name_by_origin_name=False, save_origin=False),
                # Mode.Test(video_save_path=target_path.joinpath(video_path.name), is_show=False)
            ]
            runner = Runner(reader=reader, cutter=cutter, mode=mode)
            runner.run()
            cutter.finger_angle_threshold = [cutter.finger_angle_threshold[0] - 5, 0, cutter.finger_angle_threshold[0] - 5]


data_path = Path(r'I:\Palm_Datasets\掌纹合集\HFUT_video\HFUT_video_renamed')
save_path = Path(r'I:\new_output\HFUT_video_top20')

if __name__ == "__main__":
    process_num = 2
    po = Pool(process_num)
    videos_path = list(data_path.rglob('*.mp4'))
    random.shuffle(videos_path)
    videos_num = len(videos_path)
    for i in range(process_num):
        po.apply_async(worker, (videos_path[videos_num * i // process_num:
                                            videos_num * (i + 1) // process_num], ))

    po.close()
    po.join()
