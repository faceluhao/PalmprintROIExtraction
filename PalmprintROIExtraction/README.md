# ROICutter

## 依赖

- [PaddlePaddle](https://github.com/PaddlePaddle/Paddle)
- [Mediapipe](https://github.com/google/mediapipe)
- [imgaug](https://github.com/aleju/imgaug)

## 使用方法

本项目将掌纹ROI切割分为三个模块，Reader、Cutter、Mode

- Reader: 数据读取模块，负责读取数据，使得后续单元无需在意数据的读取方式
- Cutter: ROI提取模块，负责ROI的提取
- Mode: 对提取的ROI和涉及到的中间变量的处理方式

## 关于ROI_Cutter的使用

### ROI_Cutter

- old_ROICutter: 直接对整个手掌进行语义分割，并且由于在此之前手掌还会做旋转变换，最终还会resize到512 $\times$ 512大小，因此此种方法的精度较低，对手指之间的夹角要求略高，如果手指之间的夹角较小，就会形成粘连，进而导致无法寻找到指谷点，得到错误的ROI
- new_ROICutter: 仅对指缝区域做语义分割，因此此种方法对手指之间的夹角要求较低，但是此方法需要做三次语义分割，相较于old方法，需要花费更多时间
- Process: 对应ROI提取所需的一个操作

## Mode

对ROI_Cutter得到的结果进行处理，可以并存

- Save: 保存模式，可选参数为
  - ROI_size: 保存的ROI大小，若为None，则保存原大小
  - save_dir: 保存的路径
  - save_name_by_origin_name: 是否根据原文件名保存
  - save_origin: 是否保存原图，此参数主要用于视频中
- Test: 测试模式，可选参数为
  - video_save_path: 测试视频的保存路径，若为None，则不保存
  - image_save_path: 测试图片的保存路径，若为None，则不保存
  - is_show: 是否显示
- SaveTopK: 根据某一规则，保存规则得分最高的K张ROI
  - video_save_path: 测试视频的保存路径，若为None，则不保存
  - save_dir: 测试图片的保存路径，若为None，则不保存
  - is_show: 是否显示
  - k: k的值，默认为20
