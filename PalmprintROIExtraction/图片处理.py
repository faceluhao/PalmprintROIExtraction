from pathlib import Path

from ROI_Cutter import Mode
from ROI_Cutter.Reader import PhotoReader
from ROI_Cutter.Runner import Runner

from ROI_Cutter.new_ROICutter import ROI_Cutter

data_root_path = Path(r'C:\Users\18554\Desktop\新建文件夹')
save_root_path = Path(r'C:\Users\18554\Desktop\结果2')
image_parents = {data_path.parent for data_path in data_root_path.rglob('*.jpg')}
# image_parents = sorted(list(image_parents), key=lambda x: (x.parts[-3], int(x.parts[-2])))
photo_dirs = [(image_parent, Path(str(image_parent).replace(str(data_root_path), str(save_root_path)))) for image_parent in image_parents]

cutter = ROI_Cutter()
if __name__ == '__main__':
    for data_path, save_path in photo_dirs:
        reader = PhotoReader(root_path=data_path)
        mode = [
            # Mode.Save(save_dir=save_path)
            Mode.Test(image_save_path=save_path)
        ]
        runner = Runner(reader=reader, cutter=cutter, mode=mode)
        runner.run()
