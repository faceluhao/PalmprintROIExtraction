import cv2
import numpy as np
from paddle.inference import Config
from paddle.inference import create_predictor
from paddle.inference import PrecisionType
from tools.tools import resize_keep_aspectratio, timeit


class Predictor:
    def __init__(self, model_folder_dir, infer_img_size=224, use_gpu=True,
                 gpu_memory=5000, use_tensorrt=False, precision_mode="int8"):
        self.model_folder_dir = model_folder_dir
        self.infer_img_size = infer_img_size  # 模型预测的输入图像尺寸
        self.use_gpu = use_gpu  # 是否使用GPU，默认False
        self.gpu_memory = gpu_memory  # GPU的显存，默认500
        self.use_tensorrt = use_tensorrt  # 是否使用TensorRT，默认False
        self.precision = precision_mode  # TensorRT的precision_mode为"fp16"、"fp32"、"int8"
        self.predictor = self.predict_config()


    def predict_config(self):
        # ——————————————模型配置、预测相关函数————————————————— #
        # 根据预测部署的实际情况，设置Config
        config = Config()
        # 读取模型文件
        config.set_prog_file(f'{self.model_folder_dir}/inference.pdmodel')
        config.set_params_file(f'{self.model_folder_dir}/inference.pdiparams')
        precision_map = {
            "int8": PrecisionType.Int8,
            "fp16": PrecisionType.Half,
            "fp32": PrecisionType.Float32}
        if self.use_gpu:
            config.enable_use_gpu(self.gpu_memory, 0)
            if self.use_tensorrt:
                use_calib_mode = self.precision == "int8"
                use_static = True
                config.enable_tensorrt_engine(workspace_size=1 << 30, precision_mode=precision_map[self.precision],
                                              max_batch_size=1, min_subgraph_size=self.infer_img_size,
                                              use_static=use_static, use_calib_mode=use_calib_mode)
        print("----------------------------------------------")
        print("                 RUNNING CONFIG                 ")
        print("----------------------------------------------")
        print(f"Model input size: {[self.infer_img_size, self.infer_img_size, 3]}")
        print(f"Use GPU is: {config.use_gpu()}")
        print(f"GPU device id is: {config.gpu_device_id()}")
        print(f"Init mem size is: {config.memory_pool_init_size_mb()}")
        print(f"Use TensorRT: {self.use_tensorrt}")
        print(f"Precision mode: {precision_map[self.precision]}")
        print("----------------------------------------------")
        # 可以设置开启IR优化、开启内存优化
        config.switch_ir_optim()
        config.enable_memory_optim()
        return create_predictor(config)


    def preprocess(self, image):
        mean = [0.485, 0.456, 0.406]
        std = [0.229, 0.224, 0.225]
        size = self.infer_img_size
        scale = 1.0 / 255.0

        temp = resize_keep_aspectratio(image, (size, size))
        # Normalization
        img = (temp.astype('float32') * scale - mean) / std
        # ToCHWImage
        x = img.transpose((2, 0, 1))  # float64

        # 设置输入
        input = x.astype("float32")
        return input[None]


    @timeit
    def infer(self, image):
        # 获取输入的名称
        input_names = self.predictor.get_input_names()
        input_handle = self.predictor.get_input_handle(input_names[0])

        input = self.preprocess(image)
        input_handle.copy_from_cpu(input)

        # 运行predictor
        self.predictor.run()

        output_names = self.predictor.get_output_names()
        output_handle = self.predictor.get_output_handle(output_names[0])
        result = output_handle.copy_to_cpu()[0]
        return np.argmax(result)


if __name__ == "__main__":
    predictor = Predictor(model_dir='../models/l_or_r')
    image = cv2.imdecode(np.fromfile(str(r'I:\Palm_Datasets\掌纹合集\HFUT_iphone\HFUT_iphone_image\train\1\L\1.jpg'), np.uint8), cv2.IMREAD_COLOR)

    result = predictor.infer(image)
    print(result)
