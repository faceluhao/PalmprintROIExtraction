import cv2
import numpy as np
from paddle.inference import Config
from paddle.inference import PrecisionType
from paddle.inference import create_predictor
from tools.tools import timeit


class Paddle_Seg:
    def __init__(self, model_folder_dir, infer_img_size=128, use_gpu=True,
                 gpu_memory=5000, use_tensorrt=False, precision_mode="int8"):
        self.model_folder_dir = model_folder_dir
        self.infer_img_size = infer_img_size  # 模型预测的输入图像尺寸
        self.use_gpu = use_gpu  # 是否使用GPU，默认False
        self.gpu_memory = gpu_memory  # GPU的显存，默认500
        self.use_tensorrt = use_tensorrt  # 是否使用TensorRT，默认False
        self.precision = precision_mode  # TensorRT的precision_mode为"fp16"、"fp32"、"int8"
        self.init()

    def init(self):
        # 初始化预测模型
        self.predictor = self.predict_config()

        infer_image = np.zeros([self.infer_img_size, self.infer_img_size, 3], dtype=np.uint8)
        self.infer(infer_image)

    def predict_config(self):
        # ——————————————模型配置、预测相关函数————————————————— #
        # 根据预测部署的实际情况，设置Config
        config = Config()
        # 读取模型文件
        config.set_prog_file(f'{self.model_folder_dir}/model.pdmodel')
        config.set_params_file(f'{self.model_folder_dir}/model.pdiparams')
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
        # 可以设置开启IR优化、开启内存优化
        config.switch_ir_optim()
        config.enable_memory_optim()
        return create_predictor(config)

    def preprocess(self, img):
        mean = [0.5, 0.5, 0.5]
        std = [0.5, 0.5, 0.5]

        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        mean = np.array(mean)[np.newaxis, np.newaxis, :]
        std = np.array(std)[np.newaxis, np.newaxis, :]
        im = img.astype(np.float32, copy=False) / 255.0
        im -= mean
        im /= std
        input = im.transpose((2, 0, 1))

        return input[None]

    def predict(self, predictor, data):
        input_names = predictor.get_input_names()
        input_handle = predictor.get_input_handle(input_names[0])
        output_names = predictor.get_output_names()
        output_handle = predictor.get_output_handle(output_names[0])
        input_handle.reshape(data.shape)
        input_handle.copy_from_cpu(data)
        # 执行Predictor
        predictor.run()
        # 获取输出
        return output_handle.copy_to_cpu()

    @timeit
    def infer(self, img):
        data = self.preprocess(img)
        res = self.predict(self.predictor, data)
        mask = res[0].astype('uint8') * 255
        return mask


if __name__ == "__main__":
    ###################
    model_folder_dir = "palmSegmentation/models/hand/128/ann"
    infer_img_size = 128
    gpu_memory = 7981
    use_tensorrt = False
    ###################
    paddle_seg = Paddle_Seg(model_folder_dir=model_folder_dir,
                            infer_img_size=infer_img_size,
                            gpu_memory=gpu_memory,
                            use_tensorrt=use_tensorrt)

    img = cv2.imread("/home/user/PalmprintROIExtraction/test/001_F_R1.jpg")
    img = cv2.resize(img, (128, 128))

    import time
    start_time = time.time()
    mask = paddle_seg.infer(img)
    elastic_time = time.time() - start_time
    print("%.6fs" % (elastic_time))

    cv2.imshow('image', mask)
    cv2.waitKey(0)
